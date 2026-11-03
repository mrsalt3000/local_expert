"""Тесты гибридного поиска и RRF-фузии."""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any, Sequence

import pytest
from qdrant_client import QdrantClient, models

from book_expert.catalog import Catalog
from book_expert.config import Settings
from book_expert.models import Book, BookStatus, Chunk, SearchHit, SearchSource
from book_expert.providers import (
    ChatResult,
    EmbeddingResult,
    ProviderHealth,
    ProviderUnavailableError,
)
from book_expert.retrieval.hybrid import HybridSearcher, rrf_fuse


class FakeProvider:
    name = "fake"
    chat_model = "fake-chat"
    embedding_model = "fake-embed"

    def __init__(self, vector: tuple[float, ...] = (1.0, 0.0)) -> None:
        self.vector = vector
        self.calls = 0

    def chat(
        self,
        messages: Sequence[Any],
        *,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> ChatResult:
        raise NotImplementedError

    def embed(self, texts: Sequence[str]) -> EmbeddingResult:
        self.calls += 1
        return EmbeddingResult(
            vectors=tuple(self.vector for _ in texts),
            model="fake-embed",
            dim=len(self.vector),
        )

    def health(self) -> ProviderHealth:
        return ProviderHealth(ok=True, detail="fake")


def _settings(tmp_path: Path, **overrides: Any) -> Settings:
    return Settings(
        _env_file=None,
        workspace=tmp_path,
        data_dir=tmp_path / "data",
        embedding_dim=2,
        top_k=3,
        **overrides,
    )


def _catalog(tmp_path: Path) -> Catalog:
    catalog = Catalog(tmp_path / "catalog.sqlite3")
    catalog.initialize()
    catalog.upsert_book(
        Book(
            id="book-1",
            title="Synthetic One",
            path=tmp_path / "one.pdf",
            content_hash="hash-1",
            status=BookStatus.ready,
        )
    )
    catalog.upsert_book(
        Book(
            id="book-2",
            title="Synthetic Two",
            path=tmp_path / "two.pdf",
            content_hash="hash-2",
            status=BookStatus.ready,
        )
    )
    catalog.replace_chunks(
        "book-1",
        [
            Chunk(
                id="c1",
                book_id="book-1",
                text="neural retrieval augmented generation",
                pdf_page_index=0,
                ordinal=0,
                section_path="Chapter 1",
            ),
            Chunk(
                id="c2",
                book_id="book-1",
                text="unrelated cooking recipe",
                pdf_page_index=1,
                ordinal=1,
            ),
        ],
    )
    catalog.replace_chunks(
        "book-2",
        [
            Chunk(
                id="c3",
                book_id="book-2",
                text="retrieval evaluation metrics",
                pdf_page_index=0,
                ordinal=0,
            )
        ],
    )
    return catalog


def _point(chunk_id: str, book_id: str, vector: list[float], text: str) -> Any:
    return models.PointStruct(
        id=str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id)),
        vector=vector,
        payload={
            "book_id": book_id,
            "book_title": "Synthetic One" if book_id == "book-1" else "Synthetic Two",
            "chunk_id": chunk_id,
            "text": text,
            "pdf_page_index": 0,
            "printed_page_label": "1",
            "section_path": "Chapter 1",
        },
    )


@pytest.fixture()
def searcher(tmp_path: Path) -> HybridSearcher:
    catalog = _catalog(tmp_path)
    client = QdrantClient(":memory:")
    client.create_collection(
        "books",
        vectors_config=models.VectorParams(size=2, distance=models.Distance.COSINE),
    )
    client.upsert(
        "books",
        points=[
            _point("c1", "book-1", [1.0, 0.0], "neural retrieval augmented generation"),
            _point("c2", "book-1", [0.6, 0.4], "unrelated cooking recipe"),
            _point("c3", "book-2", [0.0, 1.0], "retrieval evaluation metrics"),
        ],
        wait=True,
    )
    instance = HybridSearcher(
        _settings(tmp_path),
        FakeProvider(),
        catalog,
        client=client,
    )
    yield instance
    client.close()


def test_dense_and_lexical_are_fused(searcher: HybridSearcher) -> None:
    outcome = searcher.search("retrieval")

    assert outcome.sources == ("dense", "lexical")
    assert outcome.hits[0].chunk_id == "c1"
    assert outcome.hits[0].source is SearchSource.fused
    assert outcome.hits[0].book_title == "Synthetic One"
    assert "c3" in [hit.chunk_id for hit in outcome.hits]


def test_dense_only(searcher: HybridSearcher) -> None:
    outcome = searcher.search("anything", lexical=False)
    assert outcome.sources == ("dense",)
    assert [hit.chunk_id for hit in outcome.hits][:2] == ["c1", "c2"]
    assert all(hit.source is SearchSource.dense for hit in outcome.hits)


def test_lexical_only(searcher: HybridSearcher) -> None:
    outcome = searcher.search("retrieval", dense=False)
    assert outcome.sources == ("lexical",)
    assert {hit.chunk_id for hit in outcome.hits} == {"c1", "c3"}
    assert all(hit.source is SearchSource.lexical for hit in outcome.hits)


def test_lexical_query_mode_or_prefix5(tmp_path: Path) -> None:
    catalog = _catalog(tmp_path)
    client = QdrantClient(":memory:")
    instance = HybridSearcher(
        _settings(tmp_path, lexical_query_mode="or_prefix5"),
        FakeProvider(),
        catalog,
        client=client,
    )
    try:
        outcome = instance.search("neural cooking", dense=False, book_ids=["book-1"])
    finally:
        client.close()

    assert {hit.chunk_id for hit in outcome.hits} == {"c1", "c2"}


def test_book_filter(searcher: HybridSearcher) -> None:
    outcome = searcher.search("retrieval", book_ids=["book-2"])
    assert [hit.chunk_id for hit in outcome.hits] == ["c3"]
    assert outcome.hits[0].book_id == "book-2"


def test_empty_book_selection(searcher: HybridSearcher) -> None:
    outcome = searcher.search("retrieval", book_ids=[])
    assert outcome.hits == ()
    assert outcome.note == "no books selected"


def test_limit(searcher: HybridSearcher) -> None:
    outcome = searcher.search("retrieval", limit=1)
    assert len(outcome.hits) == 1


def test_empty_query(searcher: HybridSearcher) -> None:
    outcome = searcher.search("   ")
    assert outcome.hits == ()
    assert outcome.note == "empty query"


def test_missing_collection_requires_lexical_only(tmp_path: Path) -> None:
    catalog = _catalog(tmp_path)
    client = QdrantClient(":memory:")
    instance = HybridSearcher(
        _settings(tmp_path),
        FakeProvider(),
        catalog,
        client=client,
    )
    with pytest.raises(ProviderUnavailableError, match="missing"):
        instance.search("retrieval")
    outcome = instance.search("retrieval", dense=False)
    assert outcome.sources == ("lexical",)
    assert {hit.chunk_id for hit in outcome.hits} == {"c1", "c3"}
    client.close()


def _hit(chunk_id: str, score: float = 0.0) -> SearchHit:
    return SearchHit(
        chunk_id=chunk_id,
        book_id="book-1",
        text="text",
        score=score,
        pdf_page_index=0,
    )


def test_rrf_fuse_sums_ranks_and_is_deterministic() -> None:
    dense = [_hit("a"), _hit("b")]
    lexical = [_hit("b"), _hit("c")]
    fused = rrf_fuse(dense, lexical, limit=3)
    assert [hit.chunk_id for hit in fused] == ["b", "a", "c"]
    assert fused[0].score == pytest.approx(1 / 62 + 1 / 61)
    assert all(hit.source is SearchSource.fused for hit in fused)

    tied = rrf_fuse([_hit("z")], [_hit("y")], limit=2)
    assert [hit.chunk_id for hit in tied] == ["y", "z"]


def test_rrf_weights_shift_ranking() -> None:
    dense = [_hit("a"), _hit("b")]
    lexical = [_hit("c"), _hit("d")]

    equal = rrf_fuse(dense, lexical, limit=4)
    assert [hit.chunk_id for hit in equal] == ["a", "c", "b", "d"]

    lexical_heavy = rrf_fuse(dense, lexical, limit=4, lexical_weight=2.0)
    assert [hit.chunk_id for hit in lexical_heavy] == ["c", "d", "a", "b"]

    dense_heavy = rrf_fuse(dense, lexical, limit=4, dense_weight=2.0)
    assert [hit.chunk_id for hit in dense_heavy] == ["a", "b", "c", "d"]


def test_rrf_weights_must_not_both_be_zero() -> None:
    with pytest.raises(ValueError, match="at least one"):
        rrf_fuse(
            [_hit("a")], [_hit("b")], limit=1, dense_weight=0.0, lexical_weight=0.0
        )
    with pytest.raises(ValueError, match="non-negative"):
        rrf_fuse([_hit("a")], [_hit("b")], limit=1, lexical_weight=-1.0)


def test_rrf_extra_list_contributes_to_score() -> None:
    dense = [_hit("a")]
    lexical = [_hit("b")]
    extra = [_hit("c"), _hit("b")]

    fused = rrf_fuse(
        dense,
        lexical,
        limit=3,
        lexical_weight=0.1,
        extra=extra,
        extra_weight=0.1,
    )

    assert [hit.chunk_id for hit in fused] == ["a", "b", "c"]
    b = next(hit for hit in fused if hit.chunk_id == "b")
    assert b.score == pytest.approx(0.1 / 61 + 0.1 / 62)

    only_extra = rrf_fuse(
        [],
        [],
        limit=1,
        dense_weight=0.0,
        lexical_weight=0.0,
        extra=[_hit("x")],
        extra_weight=1.0,
    )
    assert [hit.chunk_id for hit in only_extra] == ["x"]


def test_search_adds_or_lexical_when_weight_set(
    searcher: HybridSearcher, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str | None] = []
    original = Catalog.search_lexical

    def spy(self: Catalog, query: str, **kwargs: Any) -> list[SearchHit]:
        calls.append(kwargs.get("query_mode"))
        return original(self, query, **kwargs)

    monkeypatch.setattr(Catalog, "search_lexical", spy)
    monkeypatch.setattr(searcher.settings, "lexical_or_weight", 0.1)

    outcome = searcher.search("retrieval")

    assert "and" in calls
    assert "or_prefix5" in calls
    assert "lexical_or" in outcome.sources
    assert outcome.hits
