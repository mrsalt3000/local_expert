"""Тесты индексации: Qdrant local mode, каталог, идемпотентность."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Sequence

import pymupdf
import pytest
from qdrant_client import QdrantClient, models

from book_expert.catalog import Catalog
from book_expert.config import Settings
from book_expert.ingestion.indexer import (
    EMBED_MAX_BYTES,
    BookIndexer,
    CollectionDimensionError,
    IndexingError,
    _batched_texts,
)
from book_expert.models import BookStatus
from book_expert.providers import (
    ChatResult,
    EmbeddingResult,
    ProviderError,
    ProviderHealth,
)
from book_expert.retrieval.hybrid import HybridSearcher


class FakeProvider:
    name = "fake"
    chat_model = "fake-chat"
    embedding_model = "fake-embed"

    def __init__(self, dim: int = 4) -> None:
        self.dim = dim
        self.calls = 0
        self.texts: list[str] = []

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
        self.texts.extend(texts)
        vectors = []
        for text in texts:
            digest = hashlib.sha256(text.encode()).digest()
            vectors.append(
                tuple(round(digest[index] / 255.0, 6) for index in range(self.dim))
            )
        return EmbeddingResult(vectors=tuple(vectors), model="fake-embed", dim=self.dim)

    def health(self) -> ProviderHealth:
        return ProviderHealth(ok=True, detail="fake")


class FlakyProvider(FakeProvider):
    def __init__(self, *, fail_after: int, dim: int = 4) -> None:
        super().__init__(dim)
        self.fail_after = fail_after

    def embed(self, texts: Sequence[str]) -> EmbeddingResult:
        if self.calls >= self.fail_after:
            raise ProviderError("embedding down")
        return super().embed(texts)


def _settings(tmp_path: Path, *, dim: int = 4) -> Settings:
    return Settings(
        _env_file=None,
        workspace=tmp_path,
        data_dir=tmp_path / "data",
        embedding_dim=dim,
        chunk_size=200,
        chunk_overlap=20,
    )


def _write_pdf(path: Path, pages: list[str]) -> Path:
    document = pymupdf.open()
    for text in pages:
        page = document.new_page()
        y = 72.0
        for line in text.splitlines():
            page.insert_text((72, y), line, fontsize=11)
            y += 22
    document.save(path)
    document.close()
    return path


@pytest.fixture()
def catalog(tmp_path: Path) -> Catalog:
    instance = Catalog(tmp_path / "data" / "artifacts" / "catalog.sqlite3")
    instance.initialize()
    return instance


@pytest.fixture()
def client() -> QdrantClient:
    instance = QdrantClient(":memory:")
    yield instance
    instance.close()


def _indexer(
    tmp_path: Path,
    catalog: Catalog,
    client: QdrantClient,
    provider: FakeProvider | None = None,
    *,
    dim: int = 4,
) -> tuple[BookIndexer, FakeProvider]:
    provider = provider or FakeProvider(dim)
    return (
        BookIndexer(
            _settings(tmp_path, dim=dim),
            provider,
            catalog,
            client=client,
            batch_size=2,
        ),
        provider,
    )


def test_ingest_indexes_book(
    tmp_path: Path, catalog: Catalog, client: QdrantClient
) -> None:
    pdf = _write_pdf(
        tmp_path / "book.pdf",
        [
            "Chapter 1\nAlpha synthetic sentence about retrieval.",
            "Beta synthetic sentence about evaluation.",
        ],
    )
    indexer, provider = _indexer(tmp_path, catalog, client)

    book = indexer.index_pdf(pdf, title="Synthetic Book")

    assert book.status is BookStatus.ready
    assert book.title == "Synthetic Book"
    assert book.page_count == 2
    assert book.chunk_count == 2
    assert provider.calls == 1
    assert any("Книга: Synthetic Book" in text for text in provider.texts)
    assert any("Раздел:" in text for text in provider.texts)
    assert catalog.stats()["chunks"] == 2
    assert indexer.count_points(book.id) == 2

    records, _ = client.scroll("books", limit=10, with_payload=True)
    payloads = [record.payload for record in records]
    assert {payload["book_id"] for payload in payloads} == {book.id}
    assert {payload["pdf_page_index"] for payload in payloads} == {0, 1}
    assert all(payload["text"] for payload in payloads)


def test_second_ingest_is_idempotent(
    tmp_path: Path, catalog: Catalog, client: QdrantClient
) -> None:
    pdf = _write_pdf(tmp_path / "book.pdf", ["Stable synthetic text."])
    indexer, provider = _indexer(tmp_path, catalog, client)

    first = indexer.index_pdf(pdf)
    calls_after_first = provider.calls
    second = indexer.index_pdf(pdf)

    assert first.id == second.id
    assert provider.calls == calls_after_first
    assert indexer.count_points(first.id) == 1


def test_force_reindexes(
    tmp_path: Path, catalog: Catalog, client: QdrantClient
) -> None:
    pdf = _write_pdf(tmp_path / "book.pdf", ["Stable synthetic text."])
    indexer, provider = _indexer(tmp_path, catalog, client)

    indexer.index_pdf(pdf)
    calls_after_first = provider.calls
    indexer.index_pdf(pdf, force=True)

    assert provider.calls > calls_after_first
    assert indexer.count_points() == 1


def test_changed_file_replaces_old_points(
    tmp_path: Path, catalog: Catalog, client: QdrantClient
) -> None:
    pdf = _write_pdf(tmp_path / "book.pdf", ["First version synthetic text."])
    indexer, _ = _indexer(tmp_path, catalog, client)
    first = indexer.index_pdf(pdf)

    _write_pdf(pdf, ["Second version with different synthetic text."])
    second = indexer.index_pdf(pdf)

    assert first.id == second.id
    assert second.content_hash != first.content_hash
    records, _ = client.scroll("books", limit=10, with_payload=True)
    texts = " ".join(str(record.payload["text"]) for record in records)
    assert "Second version" in texts
    assert "First version" not in texts


def test_empty_pdf_marks_failed(
    tmp_path: Path, catalog: Catalog, client: QdrantClient
) -> None:
    pdf = _write_pdf(tmp_path / "blank.pdf", [""])
    indexer, _ = _indexer(tmp_path, catalog, client)

    with pytest.raises(IndexingError, match="no extractable text"):
        indexer.index_pdf(pdf)

    stored = catalog.get_book_by_path(pdf)
    assert stored is not None
    assert stored.status is BookStatus.failed
    assert indexer.count_points() == 0


def test_embedding_dimension_mismatch_marks_failed(
    tmp_path: Path, catalog: Catalog, client: QdrantClient
) -> None:
    pdf = _write_pdf(tmp_path / "book.pdf", ["Synthetic text."])
    indexer, _ = _indexer(tmp_path, catalog, client, provider=FakeProvider(dim=3))

    with pytest.raises(IndexingError, match="embedding dim"):
        indexer.index_pdf(pdf)

    stored = catalog.get_book_by_path(pdf)
    assert stored is not None
    assert stored.status is BookStatus.failed


def test_collection_dimension_mismatch(
    tmp_path: Path, catalog: Catalog, client: QdrantClient
) -> None:
    client.create_collection(
        "books",
        vectors_config=models.VectorParams(size=2, distance=models.Distance.COSINE),
    )
    pdf = _write_pdf(tmp_path / "book.pdf", ["Synthetic text."])
    indexer, _ = _indexer(tmp_path, catalog, client)

    with pytest.raises(CollectionDimensionError, match="has dim 2"):
        indexer.index_pdf(pdf)


def test_delete_book_removes_points(
    tmp_path: Path, catalog: Catalog, client: QdrantClient
) -> None:
    pdf = _write_pdf(tmp_path / "book.pdf", ["Synthetic text."])
    indexer, _ = _indexer(tmp_path, catalog, client)
    book = indexer.index_pdf(pdf)

    assert indexer.delete_book(book.id) is True
    assert catalog.get_book(book.id) is None
    assert indexer.count_points() == 0


def test_failed_reindex_hides_old_chunks(
    tmp_path: Path, catalog: Catalog, client: QdrantClient
) -> None:
    pdf = _write_pdf(tmp_path / "book.pdf", ["Stable oldtoken synthetic text."])
    provider = FlakyProvider(fail_after=1)
    indexer, _ = _indexer(tmp_path, catalog, client, provider=provider)
    book = indexer.index_pdf(pdf)
    assert catalog.search_lexical("oldtoken")

    with pytest.raises(ProviderError, match="embedding down"):
        indexer.index_pdf(pdf, force=True)

    stored = catalog.get_book(book.id)
    assert stored is not None
    assert stored.status is BookStatus.failed
    assert catalog.search_lexical("oldtoken") == []

    searcher = HybridSearcher(
        indexer.settings, FakeProvider(dim=4), catalog, client=client
    )
    assert searcher.search("oldtoken", book_ids=[book.id], dense=False).hits == ()
    assert searcher.search("oldtoken", book_ids=[book.id]).hits == ()


def test_batched_texts_limits_count_and_bytes() -> None:
    batches = list(_batched_texts(["a" * 10, "b" * 10, "c" * 10], 2, 25))

    assert [[len(text) for text in batch] for batch in batches] == [[10, 10], [10]]


def test_batched_texts_splits_on_byte_budget() -> None:
    batches = list(_batched_texts(["a" * 10, "b" * 10], 10, 15))

    assert [len(batch) for batch in batches] == [1, 1]


def test_batched_texts_rejects_oversized_single_chunk() -> None:
    with pytest.raises(IndexingError, match="chunk is too large"):
        list(_batched_texts(["a" * (EMBED_MAX_BYTES + 1)], 2, EMBED_MAX_BYTES))


def test_batched_texts_counts_utf8_bytes_not_characters() -> None:
    batches = list(_batched_texts(["я" * 10, "я" * 10], 10, 25))

    assert [len(batch) for batch in batches] == [1, 1]
