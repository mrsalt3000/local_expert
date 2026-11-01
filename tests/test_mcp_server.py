"""Тесты MCP-сервера: инструменты, ответы и ошибки."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, Sequence

import pytest
from mcp.server.mcpserver.exceptions import ToolError

from book_expert.catalog import Catalog
from book_expert.config import Settings
from book_expert.mcp_server import RAG_TOOL_NAMES, BookExpertMCP, _answer_payload
from book_expert.models import Answer, Book, Citation, SearchHit
from book_expert.providers import (
    ChatMessage,
    ChatResult,
    EmbeddingResult,
    ProviderHealth,
)
from book_expert.retrieval.hybrid import SearchOutcome


class FakeProvider:
    name = "fake"
    chat_model = "fake-chat"
    embedding_model = "fake-embed"

    def __init__(self) -> None:
        self.closed = False

    def chat(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> ChatResult:
        raise NotImplementedError

    def embed(self, texts: Sequence[str]) -> EmbeddingResult:
        raise NotImplementedError

    def health(self) -> ProviderHealth:
        return ProviderHealth(ok=True, detail="fake")

    def close(self) -> None:
        self.closed = True


class FakeSearcher:
    def __init__(self, hits: Sequence[SearchHit] = ()) -> None:
        self.hits = tuple(hits)
        self.closed = False
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
        return SearchOutcome(hits=self.hits, sources=("dense", "lexical"))

    def close(self) -> None:
        self.closed = True


class FakeAnswerer:
    def __init__(self, answer: Answer) -> None:
        self.answer = answer
        self.calls: list[tuple[str, list[str] | None, int | None]] = []
        self.filters: list[bool] = []

    def ask(
        self,
        question: str,
        *,
        book_ids: list[str] | None = None,
        auto_route: bool = True,
        limit: int | None = None,
        relevance_filter: bool = False,
    ) -> Answer:
        self.calls.append((question, book_ids, limit))
        self.filters.append(relevance_filter)
        return self.answer


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        _env_file=None,
        workspace=tmp_path,
        data_dir=tmp_path / "data",
    )


def _catalog(tmp_path: Path) -> Catalog:
    catalog = Catalog(tmp_path / "catalog.sqlite3")
    catalog.initialize()
    catalog.upsert_book(
        Book(
            id="book-1",
            title="Synthetic Book",
            author="Synthetic Author",
            path=tmp_path / "book.pdf",
            content_hash="hash",
            page_count=2,
            chunk_count=1,
            description="A synthetic book.",
        )
    )
    return catalog


def _runtime(
    tmp_path: Path,
) -> tuple[BookExpertMCP, FakeProvider, FakeSearcher, FakeAnswerer]:
    catalog = _catalog(tmp_path)
    provider = FakeProvider()
    searcher = FakeSearcher(
        [
            SearchHit(
                chunk_id="c1",
                book_id="book-1",
                book_title="Synthetic Book",
                text="Relevant fragment.",
                score=0.7,
                pdf_page_index=1,
                printed_page_label="2",
                section_path="Chapter 1",
            )
        ]
    )
    answerer = FakeAnswerer(
        Answer(
            text="Grounded answer [1].",
            citations=(
                Citation(
                    book_id="book-1",
                    book_title="Synthetic Book",
                    pdf_page_index=1,
                    chunk_id="c1",
                    printed_page_label="2",
                ),
            ),
            confidence=0.9,
            book_ids=("book-1",),
        )
    )
    runtime = BookExpertMCP(
        _settings(tmp_path),
        catalog,
        provider,
        searcher,
        answerer,
    )
    return runtime, provider, searcher, answerer


def _call(runtime: BookExpertMCP, name: str, arguments: dict[str, Any]) -> Any:
    return asyncio.run(runtime.server.call_tool(name, arguments))


def _payload(result: Any) -> Any:
    return json.loads(result.content[0].text)


def test_tools_are_registered(tmp_path: Path) -> None:
    runtime, _, _, _ = _runtime(tmp_path)
    names = {tool.name for tool in asyncio.run(runtime.server.list_tools())}
    assert set(RAG_TOOL_NAMES) == {
        "list_books",
        "get_outline",
        "search_books",
        "ask_books",
    }
    assert names == set(RAG_TOOL_NAMES)


def test_list_books(tmp_path: Path) -> None:
    runtime, _, _, _ = _runtime(tmp_path)
    result = _call(runtime, "list_books", {})
    books = _payload(result)["books"]
    assert result.is_error is False
    assert books[0]["id"] == "book-1"
    assert books[0]["title"] == "Synthetic Book"
    assert books[0]["status"] == "pending"


def test_get_outline_and_unknown_book(tmp_path: Path) -> None:
    runtime, _, _, _ = _runtime(tmp_path)
    known = _call(runtime, "get_outline", {"book_id": "book-1"})
    payload = _payload(known)
    assert payload["book_id"] == "book-1"
    assert payload["sections"] == []

    with pytest.raises(ToolError, match="book not found"):
        _call(runtime, "get_outline", {"book_id": "missing"})


def test_search_books(tmp_path: Path) -> None:
    runtime, _, searcher, _ = _runtime(tmp_path)
    result = _call(
        runtime,
        "search_books",
        {"query": "retrieval", "book_ids": ["book-1"], "limit": 3},
    )
    payload = _payload(result)
    assert result.is_error is False
    assert payload["sources"] == ["dense", "lexical"]
    assert payload["hits"][0]["chunk_id"] == "c1"
    assert payload["hits"][0]["page"] == "2"
    assert searcher.calls[0] == {
        "query": "retrieval",
        "book_ids": ["book-1"],
        "limit": 3,
    }


def test_ask_books(tmp_path: Path) -> None:
    runtime, _, _, answerer = _runtime(tmp_path)
    result = _call(runtime, "ask_books", {"question": "What?", "book_ids": ["book-1"]})
    payload = _payload(result)
    assert result.is_error is False
    assert payload["answer"] == "Grounded answer [1]."
    assert payload["citations"][0]["chunk_id"] == "c1"
    assert answerer.calls == [("What?", ["book-1"], 5)]
    assert answerer.filters == [True]

    result = _call(
        runtime,
        "ask_books",
        {"question": "What?", "book_ids": ["book-1"], "limit": 3},
    )
    assert result.is_error is False
    assert answerer.calls[-1] == ("What?", ["book-1"], 3)
    assert answerer.filters[-1] is True


def test_answer_payload_limits_book_ids_to_citations() -> None:
    answer = Answer(
        text="Answer [1].",
        citations=(
            Citation(
                book_id="book-2",
                book_title="Second",
                pdf_page_index=0,
                chunk_id="c2",
            ),
        ),
        book_ids=tuple(f"book-{index}" for index in range(1, 21)),
    )

    payload = _answer_payload(answer)

    assert payload["book_ids"] == ["book-2"]


def test_answer_payload_caps_book_ids_without_citations() -> None:
    answer = Answer(
        text="INSUFFICIENT_CONTEXT",
        refused=True,
        book_ids=tuple(f"book-{index}" for index in range(1, 21)),
    )

    payload = _answer_payload(answer)

    assert len(payload["book_ids"]) == 5


def test_close_releases_services(tmp_path: Path) -> None:
    runtime, provider, searcher, _ = _runtime(tmp_path)
    runtime.close()
    assert provider.closed is True
    assert searcher.closed is True
