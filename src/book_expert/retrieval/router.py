"""Авто-routing: выбор релевантных книг по каталогу через LLM."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from book_expert.agent.trace import TraceRecorder
from book_expert.catalog import Catalog
from book_expert.config import Settings
from book_expert.models import Book, BookStatus
from book_expert.providers import ChatMessage, Provider, ProviderError
from book_expert.retrieval.book_profiles import PROFILE_LIMIT, BookProfileStore

OUTLINE_LIMIT = 8
CANDIDATE_LIMIT = 12
CANDIDATE_HITS = 50
LEXICAL_CANDIDATES = 8
_META_STOPWORDS = {
    "and",
    "does",
    "for",
    "how",
    "the",
    "what",
    "why",
    "with",
    "your",
    "без",
    "все",
    "для",
    "как",
    "или",
    "при",
    "про",
    "что",
    "это",
}
SYSTEM_PROMPT = (
    "Ты — компонент маршрутизации локальной RAG-системы по книгам. "
    "Выбери книги, которые нужны для ответа на вопрос пользователя. "
    "Используй только id из предоставленного каталога. "
    "Если уверенности недостаточно, верни пустой book_ids и низкий confidence. "
    "Не выдумывай книги и не добавляй пояснений вне JSON. "
    'Формат ответа: {"book_ids": ["..."], "confidence": 0.0, "reason": "..."}'
)


@dataclass(frozen=True, slots=True)
class BookRoute:
    book_ids: tuple[str, ...]
    confidence: float
    reason: str
    fallback: bool = False
    error: str | None = None


class BookRouter:
    """LLM выбирает книги; при неуверенности — fallback на всю библиотеку."""

    def __init__(
        self,
        settings: Settings,
        provider: Provider,
        catalog: Catalog,
        *,
        trace: TraceRecorder | None = None,
    ) -> None:
        self.settings = settings
        self.provider = provider
        self.catalog = catalog
        self.trace = trace
        loaded = (
            BookProfileStore.load(settings.book_profiles_path)
            if provider.name != "ollama"
            else None
        )
        self.profiles = (
            loaded
            if loaded is not None and loaded.model == provider.embedding_model
            else None
        )

    def route(self, query: str, *, books: list[Book] | None = None) -> BookRoute:
        route = self._route(query, books)
        if self.trace is not None:
            self.trace.write(
                "router_decision",
                query=query,
                book_ids=list(route.book_ids),
                confidence=route.confidence,
                fallback=route.fallback,
                reason=route.reason,
            )
        return route

    def _route(self, query: str, books: list[Book] | None) -> BookRoute:
        candidates = self.catalog.list_books() if books is None else books
        candidates = [book for book in candidates if book.status is BookStatus.ready]
        if not candidates:
            return BookRoute((), 0.0, "library is empty")
        if not query.strip():
            return self._fallback(candidates, "empty query")
        if len(candidates) == 1:
            only = candidates[0]
            return BookRoute(
                (only.id,),
                1.0,
                "single ready book; routing skipped",
            )

        candidates = _candidate_books(
            query,
            candidates,
            self.catalog,
            profiles=self.profiles,
            provider=self.provider,
            use_lemmas=self.settings.lexical_use_lemmas,
        )
        messages = [
            ChatMessage(role="system", content=SYSTEM_PROMPT),
            ChatMessage(
                role="user",
                content=_routing_payload(query, candidates, self.catalog),
            ),
        ]
        try:
            response = self.provider.chat(
                messages,
                temperature=0.0,
                max_tokens=300,
            )
        except ProviderError as exc:
            return self._fallback(candidates, f"router unavailable: {exc}")

        try:
            selected, confidence, reason = _parse_route(response.text)
        except (ValueError, TypeError) as exc:
            return self._fallback(candidates, f"invalid router response: {exc}")

        known = {book.id for book in candidates}
        selected = [book_id for book_id in selected if book_id in known]
        if not selected or confidence < self.settings.router_confidence_threshold:
            route = self._fallback(candidates, reason or "low confidence")
            return BookRoute(
                book_ids=route.book_ids,
                confidence=confidence,
                reason=route.reason,
                fallback=True,
            )
        return BookRoute(
            book_ids=tuple(selected),
            confidence=confidence,
            reason=reason,
        )

    def _fallback(self, books: list[Book], reason: str) -> BookRoute:
        return BookRoute(
            book_ids=tuple(book.id for book in books),
            confidence=0.0,
            reason=reason,
            fallback=True,
        )


def _candidate_books(
    query: str,
    books: list[Book],
    catalog: Catalog,
    *,
    profiles: BookProfileStore | None = None,
    provider: Provider | None = None,
    use_lemmas: bool = True,
) -> list[Book]:
    """Сужает каталог до правдоподобных книг до вызова LLM."""
    by_id = {book.id: book for book in books}
    ordered: list[Book] = []
    seen: set[str] = set()

    if profiles is not None and provider is not None:
        try:
            vector = provider.embed([query]).vectors[0]
            profile_ids = profiles.search(vector, limit=PROFILE_LIMIT)
        except Exception:
            profile_ids = []
        for book_id in profile_ids:
            book = by_id.get(book_id)
            if book is not None and book.id not in seen:
                ordered.append(book)
                seen.add(book.id)
            if len(ordered) >= CANDIDATE_LIMIT:
                break

    try:
        hits = catalog.search_lexical(
            query,
            limit=CANDIDATE_HITS,
            query_mode="or_prefix5",
            use_lemmas=use_lemmas,
        )
    except Exception:
        hits = []
    for hit in hits:
        book = by_id.get(hit.book_id)
        if book is not None and book.id not in seen:
            ordered.append(book)
            seen.add(book.id)
            if len(ordered) >= CANDIDATE_LIMIT:
                break

    query_tokens = _meta_tokens(query)
    if query_tokens and len(ordered) < CANDIDATE_LIMIT:
        scored = sorted(
            (
                (_metadata_score(book, query_tokens), book)
                for book in books
                if book.id not in seen
            ),
            key=lambda item: (-item[0], item[1].title.casefold()),
        )
        for score, book in scored:
            if score <= 0 or len(ordered) >= CANDIDATE_LIMIT:
                break
            ordered.append(book)
            seen.add(book.id)

    if not ordered:
        ordered = books[:CANDIDATE_LIMIT]
    return ordered[:CANDIDATE_LIMIT]


def _meta_tokens(text: str) -> set[str]:
    return {
        word[:5]
        for word in re.findall(r"[а-яёa-z0-9]{3,}", text.casefold())
        if word not in _META_STOPWORDS
    }


def _metadata_score(book: Book, query_tokens: set[str]) -> int:
    title_tokens = _meta_tokens(book.title)
    text_tokens = _meta_tokens(" ".join(filter(None, (book.author, book.description))))
    return 3 * len(query_tokens & title_tokens) + len(query_tokens & text_tokens)


def _routing_payload(query: str, books: list[Book], catalog: Catalog) -> str:
    entries = []
    for book in books:
        outline = catalog.get_outline(book.id)[:OUTLINE_LIMIT]
        entries.append(
            {
                "id": book.id,
                "title": book.title,
                "author": book.author,
                "description": book.description,
                "language": book.language,
                "sections": [section.section_path for section in outline],
            }
        )
    return json.dumps(
        {"question": query, "books": entries},
        ensure_ascii=False,
    )


def _parse_route(text: str) -> tuple[list[str], float, str]:
    payload = _extract_json(text)
    raw_ids = payload.get("book_ids", [])
    if not isinstance(raw_ids, list) or not all(
        isinstance(item, str) for item in raw_ids
    ):
        raise ValueError("book_ids must be a list of strings")
    raw_confidence = payload.get("confidence", 0.0)
    try:
        confidence = float(raw_confidence)
    except (TypeError, ValueError) as exc:
        raise ValueError("confidence must be a number") from exc
    confidence = min(max(confidence, 0.0), 1.0)
    reason = payload.get("reason", "")
    return raw_ids, confidence, reason if isinstance(reason, str) else ""


def _extract_json(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:]
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start < 0 or end < start:
        raise ValueError("no JSON object in response")
    payload = json.loads(cleaned[start : end + 1])
    if not isinstance(payload, dict):
        raise ValueError("JSON payload must be an object")
    return payload


__all__ = ["BookRoute", "BookRouter"]
