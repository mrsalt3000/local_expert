"""Индексация книги: PDF → chunks → embeddings → Qdrant + SQLite FTS5."""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Iterator

import httpx
from qdrant_client import QdrantClient, models
from qdrant_client.http import exceptions as qdrant_exceptions

from book_expert.catalog import Catalog, utcnow
from book_expert.config import Settings
from book_expert.ingestion.chunking import chunk_document
from book_expert.ingestion.extract import extract_document
from book_expert.ingestion.metadata import extract_description
from book_expert.models import Book, BookStatus, Chunk
from book_expert.providers import EmbeddingResult, Provider

PAYLOAD_INDEXES: tuple[tuple[str, models.PayloadSchemaType], ...] = (
    ("book_id", models.PayloadSchemaType.KEYWORD),
    ("chunk_id", models.PayloadSchemaType.KEYWORD),
    ("pdf_page_index", models.PayloadSchemaType.INTEGER),
    ("printed_page_label", models.PayloadSchemaType.KEYWORD),
    ("section_path", models.PayloadSchemaType.KEYWORD),
)
EMBED_MAX_BYTES = 8 * 1024 * 1024
UPSERT_BATCH_SIZE = 256


class IndexingError(RuntimeError):
    """Книгу не удалось проиндексировать."""


class CollectionDimensionError(IndexingError):
    """Размерность коллекции не совпадает с настройками embeddings."""


class BookIndexer:
    """Оркестратор ingestion: каталог, embeddings и Qdrant."""

    def __init__(
        self,
        settings: Settings,
        provider: Provider,
        catalog: Catalog,
        *,
        client: QdrantClient | None = None,
        batch_size: int = 32,
    ) -> None:
        self.settings = settings
        self.provider = provider
        self.catalog = catalog
        self.collection = settings.qdrant_collection
        self.batch_size = batch_size
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.client = client if client is not None else _qdrant_client(settings)
        self._owns_client = client is None

    def close(self) -> None:
        if self._owns_client:
            self.client.close()

    def ensure_collection(self) -> None:
        if self.client.collection_exists(self.collection):
            vectors = self.client.get_collection(self.collection).config.params.vectors
            if isinstance(vectors, models.VectorParams):
                size = vectors.size
            elif isinstance(vectors, dict):
                size = next(iter(vectors.values())).size
            else:
                raise CollectionDimensionError(
                    f"collection {self.collection!r} has no vector config"
                )
            if size != self.settings.embedding_dim:
                raise CollectionDimensionError(
                    f"collection {self.collection!r} has dim {size}, "
                    f"expected {self.settings.embedding_dim}"
                )
            return
        self.client.create_collection(
            collection_name=self.collection,
            vectors_config=models.VectorParams(
                size=self.settings.embedding_dim,
                distance=models.Distance.COSINE,
            ),
        )
        for field_name, schema in PAYLOAD_INDEXES:
            self.client.create_payload_index(
                collection_name=self.collection,
                field_name=field_name,
                field_schema=schema,
                wait=True,
            )

    def index_pdf(
        self,
        path: str | Path,
        *,
        title: str | None = None,
        force: bool = False,
    ) -> Book:
        source_path = Path(path).expanduser()
        document = extract_document(
            source_path,
            strip_running_lines=self.settings.strip_running_headers,
        )
        existing = self.catalog.get_book_by_path(source_path)
        book_id = existing.id if existing is not None else _book_id(document.path)
        book_title = title or document.title or (existing.title if existing else None)

        if (
            existing is not None
            and existing.content_hash == document.content_hash
            and existing.status is BookStatus.ready
            and not force
        ):
            return existing

        pending = Book(
            id=book_id,
            title=book_title or document.path.stem,
            author=document.author,
            path=document.path,
            content_hash=document.content_hash,
            page_count=document.page_count,
            status=BookStatus.indexing,
            format=document.format,
            description=extract_description(document.path, document.format),
            metadata={
                "source": document.metadata or {},
                "format": document.format,
                "empty_pages": len(document.empty_page_indices),
            },
        )
        self.catalog.upsert_book(pending)

        try:
            result = chunk_document(
                document,
                book_id,
                chunk_size=self.settings.chunk_size,
                chunk_overlap=self.settings.chunk_overlap,
            )
            if not result.chunks:
                raise IndexingError(
                    "no extractable text (OCR is not enabled; scanned pages are skipped)"
                )
            self.ensure_collection()
            embeddings = self._embed(result.chunks, pending.title)
            self._delete_points(book_id)
            points = [
                models.PointStruct(
                    id=str(uuid.uuid5(uuid.NAMESPACE_URL, chunk.id)),
                    vector=list(vector),
                    payload={
                        "book_id": book_id,
                        "book_title": pending.title,
                        "book_format": document.format,
                        "chunk_id": chunk.id,
                        "text": chunk.text,
                        "pdf_page_index": chunk.pdf_page_index,
                        "printed_page_label": chunk.printed_page_label,
                        "section_path": chunk.section_path,
                        "ordinal": chunk.ordinal,
                    },
                )
                for chunk, vector in zip(result.chunks, embeddings, strict=True)
            ]
            for start in range(0, len(points), UPSERT_BATCH_SIZE):
                self.client.upsert(
                    collection_name=self.collection,
                    points=points[start : start + UPSERT_BATCH_SIZE],
                    wait=True,
                )
            self.catalog.replace_chunks(book_id, result.chunks)
            ready = replace(
                pending,
                status=BookStatus.ready,
                chunk_count=len(result.chunks),
                updated_at=utcnow(),
            )
            self.catalog.upsert_book(ready, result.sections)
            return self.catalog.get_book(book_id) or ready
        except (
            qdrant_exceptions.ResponseHandlingException,
            qdrant_exceptions.ApiException,
            httpx.HTTPError,
        ) as exc:
            self._safe_delete_points(book_id)
            message = _qdrant_failure_message(exc)
            self.catalog.set_status(
                book_id,
                BookStatus.failed,
                description=message[:500],
            )
            raise IndexingError(message) from exc
        except Exception as exc:
            self._safe_delete_points(book_id)
            self.catalog.set_status(
                book_id,
                BookStatus.failed,
                description=str(exc)[:500],
            )
            raise

    def delete_book(self, book_id: str) -> bool:
        self._delete_points(book_id)
        return self.catalog.delete_book(book_id)

    def count_points(self, book_id: str | None = None) -> int:
        if not self.client.collection_exists(self.collection):
            return 0
        if book_id is None:
            return self.client.count(self.collection, exact=True).count
        result = self.client.count(
            self.collection,
            count_filter=_book_filter(book_id),
            exact=True,
        )
        return result.count

    def _embed(
        self,
        chunks: Sequence[Chunk],
        book_title: str,
    ) -> list[tuple[float, ...]]:
        vectors: list[tuple[float, ...]] = []
        texts = [_contextual_text(book_title, chunk) for chunk in chunks]
        for batch in _batched_texts(texts, self.batch_size, EMBED_MAX_BYTES):
            result: EmbeddingResult = self.provider.embed(batch)
            if result.dim != self.settings.embedding_dim:
                raise IndexingError(
                    f"embedding dim {result.dim} != configured "
                    f"{self.settings.embedding_dim}"
                )
            if len(result.vectors) != len(batch):
                raise IndexingError("embedding count does not match chunk count")
            vectors.extend(result.vectors)
        return vectors

    def _delete_points(self, book_id: str) -> None:
        if not self.client.collection_exists(self.collection):
            return
        self.client.delete(
            collection_name=self.collection,
            points_selector=models.FilterSelector(filter=_book_filter(book_id)),
            wait=True,
        )

    def _safe_delete_points(self, book_id: str) -> None:
        try:
            self._delete_points(book_id)
        except Exception:
            pass


def _contextual_text(book_title: str, chunk: Chunk) -> str:
    parts = [f"Книга: {book_title}"]
    if chunk.section_path:
        parts.append(f"Раздел: {chunk.section_path}")
    parts.append(chunk.text)
    return "\n".join(parts)


def _qdrant_failure_message(exc: Exception) -> str:
    """Не маскирует таймаут/HTTP-ошибку под «qdrant unavailable»."""
    detail = str(exc).replace("\n", " ")[:300]
    if isinstance(exc, qdrant_exceptions.ResponseHandlingException):
        return (
            "qdrant request failed or timed out (client timeout 30 s): "
            f"{detail}. Check Qdrant health and retry the book."
        )
    if isinstance(exc, qdrant_exceptions.ApiException):
        status = getattr(exc, "status_code", "?")
        return f"qdrant API error {status}: {detail}"
    if isinstance(exc, httpx.TimeoutException):
        return f"qdrant timeout (client timeout 30 s): {detail}"
    if isinstance(exc, httpx.ConnectError):
        return "qdrant is unavailable; start it with `docker compose up -d`"
    return f"qdrant request failed ({type(exc).__name__}): {detail}"


def _qdrant_client(settings: Settings) -> QdrantClient:
    api_key = (
        settings.qdrant_api_key.get_secret_value()
        if settings.qdrant_api_key is not None
        else None
    )
    return QdrantClient(url=settings.qdrant_url, api_key=api_key, timeout=30)


def _book_filter(book_id: str) -> models.Filter:
    return models.Filter(
        must=[
            models.FieldCondition(
                key="book_id",
                match=models.MatchValue(value=book_id),
            )
        ]
    )


def _book_id(path: Path) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, str(path.resolve())))


def _batched_texts(
    texts: Sequence[str], size: int, max_bytes: int
) -> Iterator[list[str]]:
    """Режет тексты и по числу, и по байтам: Ollama отклоняет большие payload'ы."""
    batch: list[str] = []
    total = 0
    for text in texts:
        text_bytes = len(text.encode("utf-8"))
        if text_bytes > max_bytes:
            raise IndexingError(
                f"chunk is too large for embeddings: {text_bytes} bytes "
                f"> {max_bytes}; lower chunk_size"
            )
        if batch and (len(batch) >= size or total + text_bytes > max_bytes):
            yield batch
            batch = []
            total = 0
        batch.append(text)
        total += text_bytes
    if batch:
        yield batch
