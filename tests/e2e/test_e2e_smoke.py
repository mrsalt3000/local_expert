"""E2E smoke: synthetic PDF проходит ingest, поиск, ответ, MCP и агент."""

from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import pymupdf
import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from qdrant_client import QdrantClient

from book_expert.agent.graph import AgentRunner
from book_expert.agent.tools.rag import build_rag_tools
from book_expert.catalog import Catalog
from book_expert.config import Settings
from book_expert.ingestion.indexer import BookIndexer, IndexingError
from book_expert.mcp_server import BookExpertMCP
from book_expert.models import BookStatus
from book_expert.providers import (
    ChatMessage,
    ChatResult,
    EmbeddingResult,
    ProviderHealth,
    ProviderUnavailableError,
)
from book_expert.retrieval.answer import GroundedAnswerer
from book_expert.retrieval.hybrid import HybridSearcher
from book_expert.retrieval.router import BookRouter


class ToolFakeModel(GenericFakeChatModel):
    def bind_tools(self, tools: Any, **kwargs: Any) -> ToolFakeModel:
        return self


class E2EProvider:
    name = "fake"
    chat_model = "fake-chat"
    embedding_model = "fake-embed"

    def __init__(self, dim: int = 4) -> None:
        self.dim = dim
        self.chat_calls = 0

    def embed(self, texts: Sequence[str]) -> EmbeddingResult:
        vectors = []
        for text in texts:
            digest = hashlib.sha256(text.encode()).digest()
            vectors.append(tuple(digest[index] / 255.0 for index in range(self.dim)))
        return EmbeddingResult(tuple(vectors), "fake-embed", self.dim)

    def chat(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> ChatResult:
        self.chat_calls += 1
        system = messages[0].content
        if "маршрутизации" in system:
            payload = json.loads(messages[1].content)
            book_id = payload["books"][0]["id"]
            return ChatResult(
                json.dumps(
                    {
                        "book_ids": [book_id],
                        "confidence": 0.9,
                        "reason": "synthetic routing",
                    }
                ),
                "fake-chat",
            )
        return ChatResult("Grounded answer from fragments [1].", "fake-chat")

    def health(self) -> ProviderHealth:
        return ProviderHealth(ok=True, detail="fake")


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        _env_file=None,
        workspace=tmp_path,
        data_dir=tmp_path / "data",
        embedding_dim=4,
        chunk_size=200,
        chunk_overlap=20,
        top_k=4,
    )


def _pdf(path: Path) -> Path:
    document = pymupdf.open()
    first = document.new_page()
    first.insert_text(
        (72, 72),
        "Chapter 1\nRetrieval augmented generation combines dense and sparse search.",
        fontsize=11,
    )
    second = document.new_page()
    second.insert_text(
        (72, 72),
        "Evaluation measures recall and citation accuracy.",
        fontsize=11,
    )
    document.save(path)
    document.close()
    return path


def _stack(
    tmp_path: Path,
) -> tuple[
    Settings,
    Catalog,
    E2EProvider,
    QdrantClient,
    BookIndexer,
    HybridSearcher,
    GroundedAnswerer,
]:
    settings = _settings(tmp_path)
    catalog = Catalog(settings.catalog_path)
    catalog.initialize()
    provider = E2EProvider()
    client = QdrantClient(":memory:")
    indexer = BookIndexer(settings, provider, catalog, client=client)
    searcher = HybridSearcher(settings, provider, catalog, client=client)
    router = BookRouter(settings, provider, catalog)
    answerer = GroundedAnswerer(settings, provider, searcher, router)
    return settings, catalog, provider, client, indexer, searcher, answerer


def test_end_to_end_synthetic_book(tmp_path: Path) -> None:
    settings, catalog, provider, client, indexer, searcher, answerer = _stack(tmp_path)
    book = indexer.index_pdf(_pdf(tmp_path / "synthetic.pdf"), title="Synthetic")

    assert book.status is BookStatus.ready
    assert book.chunk_count >= 2
    assert catalog.stats()["chunks"] == book.chunk_count
    assert indexer.count_points(book.id) == book.chunk_count

    found = searcher.search("retrieval", book_ids=[book.id])
    assert found.hits
    assert found.sources == ("dense", "lexical", "lexical_or")
    assert all(hit.book_id == book.id for hit in found.hits)

    answer = answerer.ask("How does retrieval work?")
    assert answer.refused is False
    assert answer.citations
    assert answer.citations[0].chunk_id in {hit.chunk_id for hit in found.hits}
    assert answer.confidence == 1.0
    assert answer.route_reason == "single ready book; routing skipped"
    assert provider.chat_calls == 1

    runtime = BookExpertMCP(settings, catalog, provider, searcher, answerer)
    try:
        tools = {tool.name for tool in asyncio.run(runtime.server.list_tools())}
        assert tools == {"list_books", "get_outline", "search_books", "ask_books"}

        search_result = asyncio.run(
            runtime.server.call_tool(
                "search_books",
                {"query": "retrieval", "book_ids": [book.id]},
            )
        )
        search_payload = json.loads(search_result.content[0].text)
        assert search_payload["hits"]
        assert search_payload["hits"][0]["book_id"] == book.id

        ask_result = asyncio.run(
            runtime.server.call_tool(
                "ask_books",
                {"question": "How does retrieval work?", "book_ids": [book.id]},
            )
        )
        ask_payload = json.loads(ask_result.content[0].text)
        assert ask_payload["refused"] is False
        assert ask_payload["citations"]
    finally:
        runtime.close()

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
                AIMessage(content="Agent final answer [1]."),
            ]
        )
    )
    tools = build_rag_tools(searcher, answerer)
    runner = AgentRunner(settings, tools, model=model)
    result = runner.run("Find retrieval passages", book_ids=[book.id])
    assert result.text == "Agent final answer [1]."
    assert result.trace_path.exists()
    assert "Agent final answer" not in result.trace_path.read_text(encoding="utf-8")

    client.close()


def test_lexical_only_without_qdrant(tmp_path: Path) -> None:
    settings, catalog, provider, client, indexer, _, _ = _stack(tmp_path)
    book = indexer.index_pdf(_pdf(tmp_path / "synthetic.pdf"))
    client.close()

    empty_client = QdrantClient(":memory:")
    searcher = HybridSearcher(settings, provider, catalog, client=empty_client)
    with pytest.raises(ProviderUnavailableError):
        searcher.search("retrieval", book_ids=[book.id])
    outcome = searcher.search("retrieval", book_ids=[book.id], dense=False)

    assert outcome.sources == ("lexical",)
    assert outcome.hits
    empty_client.close()


def test_unavailable_qdrant_raises_clear_error(tmp_path: Path) -> None:
    settings, catalog, provider, client, indexer, _, _ = _stack(tmp_path)
    book = indexer.index_pdf(_pdf(tmp_path / "synthetic.pdf"))
    client.close()

    offline = QdrantClient(url="http://127.0.0.1:6399", timeout=1)
    searcher = HybridSearcher(settings, provider, catalog, client=offline)
    with pytest.raises(ProviderUnavailableError, match="unavailable|missing"):
        searcher.search("retrieval", book_ids=[book.id])
    offline.close()


def test_empty_pdf_marks_failed(tmp_path: Path) -> None:
    _, catalog, _, client, indexer, _, _ = _stack(tmp_path)
    document = pymupdf.open()
    document.new_page()
    blank = tmp_path / "blank.pdf"
    document.save(blank)
    document.close()

    with pytest.raises(IndexingError, match="no extractable text"):
        indexer.index_pdf(blank)

    stored = catalog.get_book_by_path(blank)
    assert stored is not None
    assert stored.status is BookStatus.failed
    client.close()
