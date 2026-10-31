"""RAG-инструменты агента: поиск и ответ по книгам."""

from __future__ import annotations

import json
from dataclasses import asdict
from time import perf_counter
from typing import Any, Callable, Sequence

from langchain_core.tools import BaseTool, tool

from book_expert.agent.audit import AuditLogger
from book_expert.models import SearchHit
from book_expert.retrieval.answer import GroundedAnswerer
from book_expert.retrieval.hybrid import HybridSearcher

MAX_LIMIT = 20
SNIPPET_LIMIT = 1500


def build_rag_tools(
    searcher: HybridSearcher,
    answerer: GroundedAnswerer,
    *,
    default_book_ids: Sequence[str] | None = None,
    audit: AuditLogger | None = None,
) -> list[BaseTool]:
    def resolve(book_ids: list[str] | None) -> list[str] | None:
        if book_ids:
            return list(book_ids)
        return list(default_book_ids) if default_book_ids else None

    @tool
    def rag_search(
        query: str,
        book_ids: list[str] | None = None,
        limit: int = 6,
    ) -> str:
        """Ищет фрагменты в книгах и возвращает JSON с цитатами.

        Используй для вопросов, где нужны конкретные места из книг. Передай
        book_ids, если пользователь указал книги; иначе поиск идёт по библиотеке.
        """
        result_limit = max(1, min(limit, MAX_LIMIT))

        def action() -> str:
            outcome = searcher.search(
                query, book_ids=resolve(book_ids), limit=result_limit
            )
            return json.dumps(
                {
                    "query": query,
                    "sources": list(outcome.sources),
                    "note": outcome.note,
                    "hits": [
                        _hit_payload(index, hit)
                        for index, hit in enumerate(outcome.hits, start=1)
                    ],
                },
                ensure_ascii=False,
            )

        return _audited("rag_search", {"query": query}, audit, action)

    @tool
    def rag_ask(
        question: str,
        book_ids: list[str] | None = None,
    ) -> str:
        """Отвечает на вопрос по книгам с citations.

        Возвращает JSON: answer, refused, confidence, citations. Используй, когда
        нужен готовый ответ, а не сырые фрагменты.
        """

        def action() -> str:
            answer = answerer.ask(question, book_ids=resolve(book_ids))
            return json.dumps(
                {
                    "answer": answer.text,
                    "refused": answer.refused,
                    "confidence": answer.confidence,
                    "citations": [asdict(citation) for citation in answer.citations],
                },
                ensure_ascii=False,
            )

        return _audited("rag_ask", {"question": question}, audit, action)

    return [rag_search, rag_ask]


def _audited(
    name: str,
    args: dict[str, Any],
    audit: AuditLogger | None,
    action: Callable[[], str],
) -> str:
    if audit is None:
        return action()
    started = perf_counter()
    try:
        result = action()
    except Exception as exc:
        audit.log(
            name,
            args,
            ok=False,
            duration_ms=(perf_counter() - started) * 1000,
            error=str(exc),
        )
        raise
    audit.log(
        name,
        args,
        ok=True,
        duration_ms=(perf_counter() - started) * 1000,
    )
    return result


def _hit_payload(index: int, hit: SearchHit) -> dict[str, object]:
    return {
        "source_id": index,
        "chunk_id": hit.chunk_id,
        "book_id": hit.book_id,
        "book_title": hit.book_title,
        "page": hit.printed_page_label or hit.pdf_page_index,
        "section": hit.section_path,
        "score": round(hit.score, 6),
        "text": hit.text[:SNIPPET_LIMIT],
    }


__all__ = ["build_rag_tools"]
