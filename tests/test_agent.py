"""Тесты RAG-инструментов и LangGraph-агента на fake-модели."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Sequence

import pytest
import httpx
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage

from book_expert.agent.graph import (
    AgentError,
    AgentRunner,
    AgentTimeoutError,
    build_chat_model,
    invoke_with_timeout,
    normalize_agent_error,
)
from book_expert.agent.audit import AuditLogger
from book_expert.agent.trace import TraceRecorder
from book_expert.agent.tools.rag import build_rag_tools
from book_expert.config import Settings
from book_expert.models import Answer, Citation, SearchHit
from book_expert.providers import ProviderUnavailableError
from book_expert.retrieval.hybrid import SearchOutcome


class ToolFakeModel(GenericFakeChatModel):
    def bind_tools(self, tools: Any, **kwargs: Any) -> ToolFakeModel:
        return self


class TransportErrorModel(ToolFakeModel):
    def _generate(self, *args: Any, **kwargs: Any) -> Any:
        raise httpx.ConnectError("connection refused")


class InternalErrorModel(ToolFakeModel):
    def _generate(self, *args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("secret internals")


class FakeSearcher:
    def __init__(self, hits: Sequence[SearchHit] = ()) -> None:
        self.hits = tuple(hits)
        self.calls: list[dict[str, Any]] = []

    def search(
        self,
        query: str,
        *,
        book_ids: list[str] | None = None,
        limit: int | None = None,
        dense: bool = True,
        lexical: bool = True,
    ) -> SearchOutcome:
        self.calls.append({"query": query, "book_ids": book_ids, "limit": limit})
        return SearchOutcome(hits=self.hits)


class FakeAnswerer:
    def __init__(self, answer: Answer) -> None:
        self.answer = answer
        self.calls: list[tuple[str, list[str] | None]] = []

    def ask(
        self,
        question: str,
        *,
        book_ids: list[str] | None = None,
        auto_route: bool = True,
        limit: int | None = None,
    ) -> Answer:
        self.calls.append((question, book_ids))
        return self.answer


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        _env_file=None,
        workspace=tmp_path,
        data_dir=tmp_path / "data",
        agent_max_steps=4,
        agent_timeout_s=5,
    )


def _hit() -> SearchHit:
    return SearchHit(
        chunk_id="c1",
        book_id="book-1",
        book_title="Book One",
        text="Relevant synthetic passage.",
        score=0.8,
        pdf_page_index=2,
        printed_page_label="12",
        section_path="Chapter 1",
    )


def _tools() -> tuple[list[Any], FakeSearcher, FakeAnswerer]:
    searcher = FakeSearcher([_hit()])
    answerer = FakeAnswerer(
        Answer(
            text="Grounded answer [1].",
            citations=(
                Citation(
                    book_id="book-1",
                    book_title="Book One",
                    pdf_page_index=2,
                    chunk_id="c1",
                    printed_page_label="12",
                ),
            ),
        )
    )
    return build_rag_tools(searcher, answerer), searcher, answerer


def test_rag_search_tool_returns_hits() -> None:
    tools, searcher, _ = _tools()
    search_tool = next(tool for tool in tools if tool.name == "rag_search")

    payload = json.loads(search_tool.invoke({"query": "retrieval", "limit": 3}))

    assert payload["query"] == "retrieval"
    assert payload["hits"][0]["chunk_id"] == "c1"
    assert payload["hits"][0]["page"] == "12"
    assert payload["hits"][0]["text"] == "Relevant synthetic passage."
    assert searcher.calls[0] == {
        "query": "retrieval",
        "book_ids": None,
        "limit": 3,
    }


def test_rag_search_tool_respects_book_filter() -> None:
    tools, searcher, _ = _tools()
    search_tool = next(tool for tool in tools if tool.name == "rag_search")

    search_tool.invoke({"query": "retrieval", "book_ids": ["book-2"]})

    assert searcher.calls[0]["book_ids"] == ["book-2"]


def test_rag_ask_tool_returns_citations() -> None:
    tools, _, answerer = _tools()
    ask_tool = next(tool for tool in tools if tool.name == "rag_ask")

    payload = json.loads(ask_tool.invoke({"question": "what?"}))

    assert payload["answer"] == "Grounded answer [1]."
    assert payload["refused"] is False
    assert payload["citations"][0]["chunk_id"] == "c1"
    assert answerer.calls == [("what?", None)]


def test_agent_returns_final_answer_and_trace(tmp_path: Path) -> None:
    model = ToolFakeModel(messages=iter([AIMessage(content="Final answer.")]))
    runner = AgentRunner(_settings(tmp_path), [], model=model)

    result = runner.run("What is retrieval?")

    assert result.text == "Final answer."
    assert result.trace_path.exists()
    lines = result.trace_path.read_text(encoding="utf-8").splitlines()
    events = [json.loads(line)["event"] for line in lines]
    assert events == ["agent_start", "agent_end"]
    assert "Final answer." not in result.trace_path.read_text(encoding="utf-8")


def test_agent_executes_rag_tool(tmp_path: Path) -> None:
    tools, searcher, _ = _tools()
    model = ToolFakeModel(
        messages=iter(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "rag_search",
                            "args": {"query": "retrieval"},
                            "id": "call_1",
                        }
                    ],
                ),
                AIMessage(content="Final answer with [1]."),
            ]
        )
    )
    runner = AgentRunner(_settings(tmp_path), tools, model=model)

    result = runner.run("Find retrieval passages")

    assert result.text == "Final answer with [1]."
    assert searcher.calls[0]["query"] == "retrieval"
    assert any(isinstance(message, HumanMessage) for message in result.messages)


def test_invoke_with_timeout_raises() -> None:
    def slow() -> dict[str, Any]:
        time.sleep(0.2)
        return {}

    with pytest.raises(AgentTimeoutError, match="exceeded"):
        invoke_with_timeout(slow, 0.05)


def test_agent_transport_error_is_provider_error(tmp_path: Path) -> None:
    trace = TraceRecorder(tmp_path / "trace.jsonl")
    runner = AgentRunner(
        _settings(tmp_path),
        [],
        model=TransportErrorModel(messages=iter([])),
        trace=trace,
    )

    with pytest.raises(ProviderUnavailableError) as info:
        runner.run("What is retrieval?")

    assert "connection refused" not in str(info.value)
    assert "connection refused" not in trace.path.read_text(encoding="utf-8")
    events = [
        json.loads(line)["event"]
        for line in trace.path.read_text(encoding="utf-8").splitlines()
    ]
    assert events == ["agent_start", "agent_error"]


def test_normalize_agent_error_unwraps_provider_error() -> None:
    provider_error = ProviderUnavailableError("provider is down")
    wrapper = RuntimeError("langchain wrapper")
    wrapper.__cause__ = provider_error

    assert normalize_agent_error(wrapper) is provider_error


def test_agent_internal_error_hides_details(tmp_path: Path) -> None:
    runner = AgentRunner(
        _settings(tmp_path),
        [],
        model=InternalErrorModel(messages=iter([])),
    )

    with pytest.raises(AgentError) as info:
        runner.run("What is retrieval?")

    assert "secret internals" not in str(info.value)
    assert "RuntimeError" in str(info.value)


def test_trace_redacts_text_by_default(tmp_path: Path) -> None:
    trace = TraceRecorder(tmp_path / "trace.jsonl")
    trace.write("event", query="secret query", answer="secret answer", count=1)

    record = json.loads(trace.path.read_text(encoding="utf-8"))
    assert record["query"] == "<12 chars>"
    assert record["answer"] == "<13 chars>"
    assert record["count"] == 1

    verbose = TraceRecorder(tmp_path / "verbose.jsonl", include_text=True)
    verbose.write("event", query="secret query")
    assert json.loads(verbose.path.read_text())["query"] == "secret query"


def test_build_chat_model_applies_ollama_think(tmp_path: Path) -> None:
    settings = Settings(
        _env_file=None,
        workspace=tmp_path,
        data_dir=tmp_path / "data",
        ollama_think=True,
    )
    model = build_chat_model(settings)
    assert getattr(model, "reasoning", None) is True


def test_rag_tools_write_audit(tmp_path: Path) -> None:
    audit = AuditLogger(tmp_path / "audit.jsonl")
    audited_tools = build_rag_tools(
        FakeSearcher([_hit()]),
        FakeAnswerer(Answer(text="answer")),
        audit=audit,
    )
    search_tool = next(tool for tool in audited_tools if tool.name == "rag_search")

    search_tool.invoke({"query": "secret query"})

    record = json.loads(audit.path.read_text(encoding="utf-8"))
    assert record["tool"] == "rag_search"
    assert record["ok"] is True
    assert record["args"]["query"] == "<12 chars>"
    assert "secret query" not in audit.path.read_text(encoding="utf-8")
