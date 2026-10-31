"""Авто-routing: выбор релевантных книг по каталогу через LLM."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from book_expert.agent.trace import TraceRecorder
from book_expert.catalog import Catalog
from book_expert.config import Settings
from book_expert.models import Book, BookStatus
from book_expert.providers import ChatMessage, Provider, ProviderError

OUTLINE_LIMIT = 20
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
