"""Per-stage evaluation: routing, retrieval, citations, latency.

Каждый кейс проходит три стадии отдельно, чтобы было видно, где теряется
качество: выбор книг, поиск фрагментов и генерация ответа с цитатами.

Retrieval gate симулируется двумя способами, ответ при этом считается всегда —
иначе нельзя измерить false refusals:

* ``signal`` — наблюдаемый сигнал: гибридный поиск не вернул ни одного хита;
* ``oracle`` — верхняя граница экономии: в top-k нет ни одной ожидаемой
  страницы (требует разметки, поэтому это не production-gate).

Метрики цитат считаются по страницам: chunk id в размеченных наборах
относится к конкретной сборке индекса и расходится после переиндексации,
а страница остаётся проверяемой.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Literal

from book_expert.catalog import Catalog
from book_expert.config import Settings
from book_expert.eval.faithfulness import FaithfulnessJudge
from book_expert.eval.metrics import load_cases
from book_expert.providers import get_provider
from book_expert.retrieval.answer import GroundedAnswerer
from book_expert.retrieval.hybrid import HybridSearcher
from book_expert.retrieval.metadata import apply_metadata_filter
from book_expert.retrieval.rewrite import rewrite_query
from book_expert.retrieval.router import BookRouter, _candidate_books

ScopeMode = Literal["expected", "router", "candidates"]


def run_stage_eval(
    settings: Settings,
    cases_path: str | Path,
    *,
    limit: int | None = None,
    scope: ScopeMode = "expected",
    relevance_filter: bool = False,
    faithfulness: bool = False,
) -> dict[str, Any]:
    cases = load_cases(cases_path)
    if limit:
        cases = cases[:limit]
    settings.ensure_dirs()
    catalog = Catalog(settings.catalog_path)
    catalog.initialize()
    provider = get_provider(settings)
    searcher = HybridSearcher(settings, provider, catalog)
    router = BookRouter(settings, provider, catalog)
    answerer = GroundedAnswerer(settings, provider, searcher, router)
    judge = FaithfulnessJudge(provider) if faithfulness else None
    pages = {chunk.id: chunk.pdf_page_index for chunk in catalog.iter_chunks()}
    ready_books = [
        book for book in catalog.list_books() if book.status.value == "ready"
    ]
    rows: list[dict[str, Any]] = []
    try:
        for index, case in enumerate(cases, start=1):
            expected_books = set(case.expected_book_ids)
            expected_chunks = set(case.expected_chunk_ids)
            expected_pages = {pages[c] for c in expected_chunks if c in pages}
            query = (
                rewrite_query(case.question)
                if settings.query_rewrite_enabled
                else case.question
            )

            route_start = time.perf_counter()
            route = router.route(case.question)
            route_ms = (time.perf_counter() - route_start) * 1000
            search_books = _scope_book_ids(
                scope,
                case.expected_book_ids,
                route.book_ids,
                query,
                ready_books,
                catalog,
                router,
                provider,
                settings,
            )

            retrieval_start = time.perf_counter()
            outcome = searcher.search(
                query,
                book_ids=search_books,
                limit=5,
            )
            retrieval_ms = (time.perf_counter() - retrieval_start) * 1000
            retrieved_pages = [hit.pdf_page_index for hit in outcome.hits]
            page_hits = len(expected_pages & set(retrieved_pages))
            chunk_hits = len(expected_chunks & {hit.chunk_id for hit in outcome.hits})

            answer_start = time.perf_counter()
            answer = answerer.ask(
                case.question,
                book_ids=search_books,
                limit=5,
                relevance_filter=relevance_filter,
            )
            answer_ms = (time.perf_counter() - answer_start) * 1000
            judgement = None
            if judge is not None and not answer.refused:
                judgement = judge.evaluate(
                    case.question,
                    answer.text,
                    tuple(hit.text for hit in answer.sources),
                )
            cited_pages = {citation.pdf_page_index for citation in answer.citations}
            citation_hits = len(cited_pages & expected_pages)
            row = {
                "index": index,
                "route_top1": bool(route.book_ids)
                and route.book_ids[0] in expected_books,
                "route_recall3": bool(expected_books & set(route.book_ids[:3])),
                "route_fallback": route.fallback,
                "route_ms": round(route_ms, 1),
                "page_recall@5": page_hits / len(expected_pages)
                if expected_pages
                else 0.0,
                "chunk_recall@5": chunk_hits / len(expected_chunks)
                if expected_chunks
                else 0.0,
                "page_mrr@5": _page_mrr(retrieved_pages, expected_pages, 5),
                "retrieval_ms": round(retrieval_ms, 1),
                "answer_ms": round(answer_ms, 1),
                "answer_refused": answer.refused,
                "citations": len(answer.citations),
                "citation_precision": citation_hits / len(cited_pages)
                if cited_pages
                else 0.0,
                "citation_recall": citation_hits / len(expected_pages)
                if expected_pages
                else 0.0,
                "faithfulness": judgement.faithfulness if judgement else None,
                "judged_claims": len(judgement.claims) if judgement else 0,
                "faithfulness_error": (
                    judgement.error if judgement is not None else None
                ),
                "gate_signal": not outcome.hits,
                "gate_oracle": bool(expected_pages) and page_hits == 0,
            }
            rows.append(row)
            if index % 10 == 0:
                print(f"{index}/{len(cases)}", flush=True)
    finally:
        searcher.close()
        provider.close()
    return _aggregate(
        rows,
        scope=scope,
        relevance_filter=relevance_filter,
        faithfulness=faithfulness,
    )


def _scope_book_ids(
    scope: ScopeMode,
    expected_books: tuple[str, ...],
    routed_books: tuple[str, ...],
    query: str,
    ready_books: list[Any],
    catalog: Catalog,
    router: BookRouter,
    provider: Any,
    settings: Settings,
) -> list[str] | None:
    if scope == "router":
        return list(routed_books) or None
    if scope == "candidates":
        candidates = _candidate_books(
            query,
            apply_metadata_filter(ready_books, query),
            catalog,
            profiles=router.profiles,
            provider=provider,
            use_lemmas=settings.lexical_use_lemmas,
        )
        return [book.id for book in candidates] or None
    return list(expected_books) or None


def _page_mrr(
    retrieved_pages: list[int],
    expected_pages: set[int],
    k: int,
) -> float:
    if not expected_pages:
        return 0.0
    for rank, page in enumerate(retrieved_pages[:k], start=1):
        if page in expected_pages:
            return 1.0 / rank
    return 0.0


def _aggregate(
    rows: list[dict[str, Any]],
    *,
    scope: ScopeMode,
    relevance_filter: bool,
    faithfulness: bool = False,
) -> dict[str, Any]:
    routed = [row for row in rows if not row["route_fallback"]]
    return {
        "cases": len(rows),
        "scope": scope,
        "relevance_filter": relevance_filter,
        "faithfulness_enabled": faithfulness,
        "gate": {
            "signal": _gate_summary(rows, "gate_signal"),
            "oracle": _gate_summary(rows, "gate_oracle"),
        },
        "routing": {
            "top1": _share(rows, "route_top1"),
            "recall_at_3": _share(rows, "route_recall3"),
            "fallback": sum(row["route_fallback"] for row in rows),
            "non_fallback": {
                "cases": len(routed),
                "top1": _share(routed, "route_top1"),
                "recall_at_3": _share(routed, "route_recall3"),
            },
        },
        "retrieval": {
            "page_recall_at_5": _mean(rows, "page_recall@5"),
            "page_mrr_at_5": _mean(rows, "page_mrr@5"),
            "chunk_recall_at_5": _mean(rows, "chunk_recall@5"),
        },
        "answers": {
            "refused": sum(bool(row["answer_refused"]) for row in rows),
            "citation_precision": _mean(rows, "citation_precision"),
            "citation_recall": _mean(rows, "citation_recall"),
            "faithfulness": _mean_optional(rows, "faithfulness"),
            "judged_cases": sum(1 for row in rows if row["faithfulness"] is not None),
            "faithfulness_errors": sum(1 for row in rows if row["faithfulness_error"]),
            "no_citations": sum(
                row["citations"] == 0 and not row["answer_refused"] for row in rows
            ),
            "zero_precision": sum(
                row["citation_precision"] == 0.0 and not row["answer_refused"]
                for row in rows
            ),
        },
        "latency_ms": {
            "route_p50": _percentile([row["route_ms"] for row in rows], 50),
            "route_p95": _percentile([row["route_ms"] for row in rows], 95),
            "retrieval_p50": _percentile([row["retrieval_ms"] for row in rows], 50),
            "retrieval_p95": _percentile([row["retrieval_ms"] for row in rows], 95),
            "answer_p50": _percentile([row["answer_ms"] for row in rows], 50),
            "answer_p95": _percentile([row["answer_ms"] for row in rows], 95),
        },
        "details": rows,
    }


def _gate_summary(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    fired = [row for row in rows if row[key]]
    false_refusals = [
        row
        for row in fired
        if not row["answer_refused"] and row["citation_precision"] > 0
    ]
    return {
        "fired": len(fired),
        "false_refusals": len(false_refusals),
        "answer_ms_saved": round(sum(row["answer_ms"] for row in fired), 1),
    }


def _mean(rows: list[dict[str, Any]], key: str) -> float:
    return round(sum(row[key] for row in rows) / len(rows), 4) if rows else 0.0


def _mean_optional(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [row[key] for row in rows if row.get(key) is not None]
    return round(sum(values) / len(values), 4) if values else None


def _share(rows: list[dict[str, Any]], key: str) -> float:
    return round(sum(bool(row[key]) for row in rows) / len(rows), 4) if rows else 0.0


def _percentile(values: list[float], percent: int) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, round((percent / 100) * (len(ordered) - 1)))
    return round(ordered[index], 1)


def write_report(report: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


__all__ = ["ScopeMode", "run_stage_eval", "write_report"]
