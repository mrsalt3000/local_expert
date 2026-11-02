"""Eval режима «Агент» (super): путь, который UI открывает по умолчанию.

Метрики те же, что в ``stage_eval``, но ответ строит ``SuperAdvancedAnswerer``:
план запросов, дополнительные раунды и синтез. Page recall считается по
цитатам ответа — именно их видит пользователь.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from book_expert.catalog import Catalog
from book_expert.config import Settings
from book_expert.eval.metrics import load_cases
from book_expert.eval.stage_eval import _percentile, _report_meta
from book_expert.providers import get_provider
from book_expert.retrieval.hybrid import HybridSearcher
from book_expert.retrieval.router import BookRouter
from book_expert.retrieval.super_advanced import SuperAdvancedAnswerer


def run_agent_eval(
    settings: Settings,
    cases_path: str | Path,
    *,
    limit: int | None = None,
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
    answerer = SuperAdvancedAnswerer(settings, provider, searcher, router)
    pages = {chunk.id: chunk.pdf_page_index for chunk in catalog.iter_chunks()}
    rows: list[dict[str, Any]] = []
    try:
        for index, case in enumerate(cases, start=1):
            expected_pages = {
                pages[chunk_id]
                for chunk_id in case.expected_chunk_ids
                if chunk_id in pages
            }
            started = time.perf_counter()
            answer = answerer.ask(case.question, book_ids=None, auto_route=True)
            elapsed_ms = (time.perf_counter() - started) * 1000
            cited_pages = {citation.pdf_page_index for citation in answer.citations}
            page_hits = len(cited_pages & expected_pages)
            rows.append(
                {
                    "index": index,
                    "page_recall": page_hits / len(expected_pages)
                    if expected_pages
                    else 0.0,
                    "citation_precision": page_hits / len(cited_pages)
                    if cited_pages
                    else 0.0,
                    "refused": answer.refused,
                    "citations": len(answer.citations),
                    "plan": len(answer.plan),
                    "rounds": answer.rounds,
                    "elapsed_ms": round(elapsed_ms, 1),
                }
            )
            if index % 5 == 0:
                print(f"{index}/{len(cases)}", flush=True)
    finally:
        searcher.close()
        close = getattr(provider, "close", None)
        if callable(close):
            close()
    report = _aggregate(rows)
    report["meta"] = _report_meta(
        settings,
        provider,
        cases_path,
        result_k=settings.top_k,
        scope="candidates",
        relevance_filter=False,
        faithfulness=False,
        cases=len(cases),
    )
    report["meta"]["advanced_query_limit"] = settings.advanced_query_limit
    report["mode"] = "super"
    return report


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "cases": len(rows),
        "retrieval": {
            "page_recall": _mean(rows, "page_recall"),
            "citation_precision": _mean(rows, "citation_precision"),
        },
        "answers": {
            "refused": sum(1 for row in rows if row["refused"]),
            "zero_precision": sum(
                1
                for row in rows
                if not row["refused"] and row["citation_precision"] == 0.0
            ),
            "no_citations": sum(
                1 for row in rows if not row["refused"] and row["citations"] == 0
            ),
        },
        "plan": {
            "avg_queries": round(_mean(rows, "plan"), 2),
            "avg_rounds": round(_mean(rows, "rounds"), 2),
        },
        "latency_ms": {
            "p50": _percentile([row["elapsed_ms"] for row in rows], 50),
            "p95": _percentile([row["elapsed_ms"] for row in rows], 95),
        },
        "details": rows,
    }


def _mean(rows: list[dict[str, Any]], key: str) -> float:
    return round(sum(row[key] for row in rows) / len(rows), 4) if rows else 0.0


__all__ = ["run_agent_eval"]
