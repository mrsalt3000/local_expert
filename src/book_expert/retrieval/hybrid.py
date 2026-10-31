"""Гибридный поиск: dense Qdrant + lexical FTS5, фузия RRF."""

from __future__ import annotations

from dataclasses import dataclass, replace

import httpx
from qdrant_client import QdrantClient, models
from qdrant_client.http import exceptions as qdrant_exceptions

from book_expert.catalog import Catalog
from book_expert.config import Settings
from book_expert.models import BookStatus, SearchHit, SearchSource
from book_expert.providers import Provider, ProviderUnavailableError

RRF_K = 60


@dataclass(frozen=True, slots=True)
class SearchOutcome:
    hits: tuple[SearchHit, ...]
    sources: tuple[str, ...] = ()
    note: str | None = None


class HybridSearcher:
    """Объединяет семантический и лексический поиск через Reciprocal Rank Fusion."""

    def __init__(
        self,
        settings: Settings,
        provider: Provider,
        catalog: Catalog,
        *,
        client: QdrantClient | None = None,
    ) -> None:
        self.settings = settings
        self.provider = provider
        self.catalog = catalog
        self.collection = settings.qdrant_collection
        self.client = client if client is not None else _qdrant_client(settings)
        self._owns_client = client is None

    def close(self) -> None:
        if self._owns_client:
            self.client.close()

    def search(
        self,
        query: str,
        *,
        book_ids: list[str] | None = None,
        limit: int | None = None,
        dense: bool = True,
        lexical: bool = True,
    ) -> SearchOutcome:
        text = query.strip()
        if not text:
            return SearchOutcome(hits=(), note="empty query")
        if book_ids is not None and not book_ids:
            return SearchOutcome(hits=(), note="no books selected")

        result_limit = self.settings.top_k if limit is None else limit
        if result_limit <= 0:
            return SearchOutcome(hits=(), note="limit must be positive")
        candidate_limit = max(result_limit * 4, 20)

        dense_hits: list[SearchHit] = []
        sources: list[str] = []
        if dense:
            try:
                if not self.client.collection_exists(self.collection):
                    raise ProviderUnavailableError(
                        f"qdrant collection {self.collection!r} is missing; "
                        "start Qdrant with `docker compose up -d` or use lexical-only"
                    )
                dense_hits = self._dense_search(text, book_ids, candidate_limit)
            except ProviderUnavailableError:
                raise
            except (
                qdrant_exceptions.ResponseHandlingException,
                qdrant_exceptions.ApiException,
                httpx.HTTPError,
            ) as exc:
                raise ProviderUnavailableError(
                    "qdrant is unavailable; start it with `docker compose up -d` "
                    "or use lexical-only"
                ) from exc
            sources.append("dense")

        lexical_hits: list[SearchHit] = []
        if lexical:
            lexical_hits = self.catalog.search_lexical(
                text, book_ids=book_ids, limit=candidate_limit
            )
            sources.append("lexical")

        if dense_hits and lexical_hits:
            hits = rrf_fuse(dense_hits, lexical_hits, limit=result_limit)
        elif dense_hits:
            hits = tuple(dense_hits[:result_limit])
        else:
            hits = tuple(lexical_hits[:result_limit])

        return SearchOutcome(hits=hits, sources=tuple(sources))

    def _dense_search(
        self, query: str, book_ids: list[str] | None, limit: int
    ) -> list[SearchHit]:
        embedding = self.provider.embed([query]).vectors[0]
        points = self.client.query_points(
            collection_name=self.collection,
            query=list(embedding),
            query_filter=_book_filter(book_ids),
            limit=limit,
            with_payload=True,
        ).points
        hits = [_hit_from_point(point) for point in points]
        ready = {
            book.id
            for book in self.catalog.books_by_ids({hit.book_id for hit in hits})
            if book.status is BookStatus.ready
        }
        return [hit for hit in hits if hit.book_id in ready]


def rrf_fuse(
    dense: list[SearchHit],
    lexical: list[SearchHit],
    *,
    limit: int,
    k: int = RRF_K,
) -> tuple[SearchHit, ...]:
    scores: dict[str, float] = {}
    best: dict[str, SearchHit] = {}
    for hits in (dense, lexical):
        for rank, hit in enumerate(hits, start=1):
            scores[hit.chunk_id] = scores.get(hit.chunk_id, 0.0) + 1.0 / (k + rank)
            best.setdefault(hit.chunk_id, hit)
    ordered = sorted(scores, key=lambda chunk_id: (-scores[chunk_id], chunk_id))
    return tuple(
        replace(best[chunk_id], score=scores[chunk_id], source=SearchSource.fused)
        for chunk_id in ordered[:limit]
    )


def _hit_from_point(point: models.ScoredPoint) -> SearchHit:
    payload = point.payload or {}
    page = payload.get("pdf_page_index")
    return SearchHit(
        chunk_id=str(payload.get("chunk_id") or point.id),
        book_id=str(payload.get("book_id") or ""),
        text=str(payload.get("text") or ""),
        score=float(point.score),
        pdf_page_index=int(page) if isinstance(page, int) else -1,
        printed_page_label=_optional_str(payload.get("printed_page_label")),
        section_path=_optional_str(payload.get("section_path")),
        book_title=_optional_str(payload.get("book_title")),
        source=SearchSource.dense,
    )


def _book_filter(book_ids: list[str] | None) -> models.Filter | None:
    if book_ids is None:
        return None
    match: models.Match = (
        models.MatchValue(value=book_ids[0])
        if len(book_ids) == 1
        else models.MatchAny(any=book_ids)
    )
    return models.Filter(must=[models.FieldCondition(key="book_id", match=match)])


def _optional_str(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


def _qdrant_client(settings: Settings) -> QdrantClient:
    api_key = (
        settings.qdrant_api_key.get_secret_value()
        if settings.qdrant_api_key is not None
        else None
    )
    return QdrantClient(url=settings.qdrant_url, api_key=api_key)


__all__ = ["HybridSearcher", "SearchOutcome", "rrf_fuse"]
