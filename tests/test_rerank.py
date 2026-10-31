"""Тесты cross-encoder reranking поверх гибридного поиска."""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any, Sequence

import pytest
from qdrant_client import QdrantClient, models

from book_expert.catalog import Catalog
from book_expert.config import Settings
from book_expert.doctor import WARN, check_reranker
from book_expert.models import Book, BookStatus, Chunk
from book_expert.providers import (
    ChatResult,
    EmbeddingResult,
    ProviderHealth,
)
from book_expert.retrieval.hybrid import HybridSearcher
from book_expert.retrieval.rerank import RerankError, create_reranker


class FakeProvider:
    name = "fake"
    chat_model = "fake-chat"
    embedding_model = "fake-embed"

    def chat(
        self,
        messages: Sequence[Any],
        *,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> ChatResult:
        raise NotImplementedError

    def embed(self, texts: Sequence[str]) -> EmbeddingResult:
        return EmbeddingResult(
            vectors=tuple((1.0, 0.0) for _ in texts),
            model="fake-embed",
            dim=2,
        )

    def health(self) -> ProviderHealth:
        return ProviderHealth(ok=True, detail="fake")


class FakeReranker:
    def __init__(self, scores: dict[str, float] | None = None) -> None:
        self.scores = scores or {}
        self.closed = False
        self.calls: list[tuple[str, list[str]]] = []

    def rerank(self, query: str, documents: Sequence[str]) -> list[float]:
        self.calls.append((query, list(documents)))
        return [self.scores.get(document, 0.0) for document in documents]

    def close(self) -> None:
        self.closed = True


class FailingReranker:
    def rerank(self, query: str, documents: Sequence[str]) -> list[float]:
        raise RerankError("model is offline")

    def close(self) -> None:
        pass


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        _env_file=None,
        workspace=tmp_path,
        data_dir=tmp_path / "data",
        embedding_dim=2,
        top_k=3,
        rerank_top_n=5,
    )


def _catalog(tmp_path: Path) -> Catalog:
    catalog = Catalog(tmp_path / "catalog.sqlite3")
    catalog.initialize()
    catalog.upsert_book(
        Book(
            id="book-1",
            title="Synthetic",
            path=tmp_path / "book.pdf",
            content_hash="hash-1",
            status=BookStatus.ready,
        )
    )
    catalog.replace_chunks(
        "book-1",
        [
            Chunk(
                id="c1",
                book_id="book-1",
                text="alpha",
                pdf_page_index=0,
                ordinal=0,
            ),
            Chunk(
                id="c2",
                book_id="book-1",
                text="beta",
                pdf_page_index=1,
                ordinal=1,
            ),
            Chunk(
                id="c3",
                book_id="book-1",
                text="gamma",
                pdf_page_index=2,
                ordinal=2,
            ),
        ],
    )
    return catalog


def _point(chunk_id: str, text: str, vector: list[float]) -> Any:
    return models.PointStruct(
        id=str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id)),
        vector=vector,
        payload={
            "book_id": "book-1",
            "book_title": "Synthetic",
            "chunk_id": chunk_id,
            "text": text,
            "pdf_page_index": 0,
            "section_path": "Chapter 1",
        },
    )


def _searcher(tmp_path: Path, reranker: Any) -> tuple[HybridSearcher, QdrantClient]:
    client = QdrantClient(":memory:")
    client.create_collection(
        "books",
        vectors_config=models.VectorParams(size=2, distance=models.Distance.COSINE),
    )
    client.upsert(
        "books",
        points=[
            _point("c1", "alpha", [1.0, 0.0]),
            _point("c2", "beta", [0.9, 0.1]),
            _point("c3", "gamma", [0.8, 0.2]),
        ],
        wait=True,
    )
    instance = HybridSearcher(
        _settings(tmp_path),
        FakeProvider(),
        _catalog(tmp_path),
        client=client,
        reranker=reranker,
    )
    return instance, client


def test_rerank_reorders_candidates(tmp_path: Path) -> None:
    reranker = FakeReranker({"alpha": 0.1, "beta": 0.9, "gamma": 0.5})
    searcher, client = _searcher(tmp_path, reranker)
    try:
        outcome = searcher.search("query", dense=True, lexical=False)
    finally:
        client.close()

    assert outcome.sources == ("dense", "rerank")
    assert outcome.hits[0].chunk_id == "c2"
    assert outcome.hits[0].score == pytest.approx(0.9)
    assert outcome.note is None
    assert reranker.calls


def test_rerank_only_uses_top_n(tmp_path: Path) -> None:
    reranker = FakeReranker({"alpha": 0.1, "beta": 0.9})
    searcher, client = _searcher(tmp_path, reranker)
    try:
        searcher.search("query", dense=True, lexical=False)
    finally:
        client.close()

    documents = reranker.calls[0][1]
    assert len(documents) <= searcher.settings.rerank_top_n
    assert set(documents) == {"alpha", "beta", "gamma"}


def test_rerank_failure_falls_back_to_fused_order(tmp_path: Path) -> None:
    searcher, client = _searcher(tmp_path, FailingReranker())
    try:
        outcome = searcher.search("query", dense=True, lexical=False)
    finally:
        client.close()

    assert outcome.sources == ("dense",)
    assert outcome.note is not None and "rerank failed" in outcome.note
    assert [hit.chunk_id for hit in outcome.hits] == ["c1", "c2", "c3"]


def test_close_closes_reranker(tmp_path: Path) -> None:
    reranker = FakeReranker()
    searcher, client = _searcher(tmp_path, reranker)
    client.close()
    searcher.close()
    assert reranker.closed is True


def test_create_reranker_disabled_by_default(tmp_path: Path) -> None:
    assert create_reranker(_settings(tmp_path)) is None


def test_create_reranker_enabled(tmp_path: Path) -> None:
    settings = _settings(tmp_path).model_copy(
        update={"rerank_enabled": True, "rerank_model": "fake/model"}
    )
    reranker = create_reranker(settings)
    assert reranker is not None
    assert reranker.model_name == "fake/model"  # type: ignore[attr-defined]


def test_doctor_warns_when_reranker_model_missing(tmp_path: Path) -> None:
    settings = _settings(tmp_path).model_copy(update={"rerank_enabled": True})
    result = check_reranker(settings)
    assert result.status == WARN
    assert "не скачана" in result.detail


def test_doctor_ok_when_reranker_model_cached(tmp_path: Path) -> None:
    settings = _settings(tmp_path).model_copy(update={"rerank_enabled": True})
    folder = (
        settings.artifacts_dir
        / "models"
        / f"models--{settings.rerank_model.replace('/', '--')}"
    )
    folder.mkdir(parents=True)
    result = check_reranker(settings)
    assert result.status == "ok"
    assert "в кэше" in result.detail
