"""Eval режима «Агент» (super): путь, который UI открывает по умолчанию.

Метрики те же, что в ``stage_eval``, но ответ строит ``SuperAdvancedAnswerer``:
план запросов, дополнительные раунды и синтез. Page recall считается по
цитатам ответа — именно их видит пользователь.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from book_expert.catalog import Catalog
from book_expert.config import Settings
from book_expert.eval.faithfulness import FaithfulnessJudge
from book_expert.eval.metrics import load_cases
from book_expert.eval.multigold import (
    GRADE_ADMISSIBLE,
    MultiGoldCase,
    load_annotations,
)
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
    offset: int = 0,
    faithfulness: bool = False,
    judge_provider: str | None = None,
    judge_model: str | None = None,
    strict_judge: bool = False,
    strict_judge_v2: bool = False,
    quote_judge: bool = False,
    multigold: str | Path | None = None,
) -> dict[str, Any]:
    cases = load_cases(cases_path)
    cases = cases[offset : offset + limit] if limit else cases[offset:]
    annotations: tuple[MultiGoldCase, ...] | None = None
    if multigold is not None:
        annotations = load_annotations(multigold)[offset : offset + len(cases)]
        if len(annotations) != len(cases):
            raise ValueError(
                "multigold annotations must cover all evaluated cases "
                f"({len(annotations)} < {len(cases)})"
            )
    settings.ensure_dirs()
    catalog = Catalog(settings.catalog_path)
    catalog.initialize()
    provider = get_provider(settings)
    searcher = HybridSearcher(settings, provider, catalog)
    router = BookRouter(settings, provider, catalog)
    answerer = SuperAdvancedAnswerer(settings, provider, searcher, router)
    judge_settings = (
        settings.with_provider(judge_provider)
        if faithfulness and judge_provider
        else None
    )
    if judge_settings is not None and judge_model:
        judge_settings = judge_settings.model_copy(update={"chat_model": judge_model})
    judge_client = get_provider(judge_settings) if judge_settings else None
    judge = (
        FaithfulnessJudge(
            judge_client or provider,
            strict=strict_judge,
            strict_v2=strict_judge_v2,
            quote_mode=quote_judge,
        )
        if faithfulness
        else None
    )
    pages = {
        chunk.id: (chunk.book_id, chunk.pdf_page_index)
        for chunk in catalog.iter_chunks()
    }
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
            expected_book = next(iter(case.expected_book_ids), None)
            cited_pages = {
                (citation.book_id, citation.pdf_page_index)
                for citation in answer.citations
            }
            citation_book_hits = sum(
                1 for citation in answer.citations if citation.book_id == expected_book
            )
            context = answer.context or answer.sources
            source_pages = {(hit.book_id, hit.pdf_page_index) for hit in context}
            page_hits = len(cited_pages & expected_pages)
            source_hits = len(source_pages & expected_pages)
            multigold_metrics = (
                _multigold_page_metrics(
                    cited_pages,
                    source_pages,
                    annotations[index - 1].pages(pages),
                )
                if annotations is not None
                else None
            )
            judgement = None
            if judge is not None and not answer.refused:
                context_index = {
                    hit.chunk_id: position
                    for position, hit in enumerate(context, start=1)
                }
                marker_map = {
                    citation.number: context_index.get(
                        citation.chunk_id, citation.number
                    )
                    for citation in answer.citations
                }
                judgement = judge.evaluate(
                    case.question,
                    answer.text,
                    tuple(hit.text for hit in context),
                    marker_map=marker_map,
                    citation_count=len(answer.citations),
                )
            rows.append(
                {
                    "index": index,
                    "page_recall": page_hits / len(expected_pages)
                    if expected_pages
                    else 0.0,
                    "citation_precision": page_hits / len(cited_pages)
                    if cited_pages
                    else 0.0,
                    "citation_book_precision": citation_book_hits
                    / len(answer.citations)
                    if answer.citations
                    else 0.0,
                    "source_page_recall": source_hits / len(expected_pages)
                    if expected_pages
                    else 0.0,
                    "refused": answer.refused,
                    "citations": len(answer.citations),
                    "plan": len(answer.plan),
                    "rounds": answer.rounds,
                    "extra_queries": answer.extra_queries,
                    "answer_words": len(answer.text.split()),
                    "faithfulness": judgement.faithfulness if judgement else None,
                    "citation_support_precision": (
                        judgement.citation_support_precision if judgement else None
                    ),
                    "citation_usage": (judgement.citation_usage if judgement else None),
                    "citation_support_rate": (
                        judgement.citation_support_rate if judgement else None
                    ),
                    "citations_available": (
                        judgement.citations_available if judgement else 0
                    ),
                    "citations_referenced": (
                        judgement.citations_referenced if judgement else 0
                    ),
                    "citations_supporting": (
                        judgement.citations_supporting if judgement else 0
                    ),
                    "judged_claims": len(judgement.claims) if judgement else 0,
                    "claims_truncated": bool(judgement and judgement.truncated),
                    "faithfulness_error": (
                        judgement.error if judgement is not None else None
                    ),
                    "multigold": multigold_metrics,
                    "elapsed_ms": round(elapsed_ms, 1),
                }
            )
            if index % 5 == 0:
                print(f"{index}/{len(cases)}", flush=True)
    finally:
        searcher.close()
        if judge_client is not None:
            close_judge = getattr(judge_client, "close", None)
            if callable(close_judge):
                close_judge()
        close = getattr(provider, "close", None)
        if callable(close):
            close()
    report = _aggregate(rows, cases, pages, annotations)
    report["meta"] = _report_meta(
        settings,
        provider,
        cases_path,
        result_k=settings.top_k,
        scope="candidates",
        relevance_filter=False,
        faithfulness=faithfulness,
        cases=len(cases),
    )
    if annotations is not None:
        report["meta"]["multigold"] = {
            "path": str(multigold),
            "cases": len(annotations),
            "cases_with_alternatives": sum(
                1
                for annotation in annotations
                if any(
                    grade == GRADE_ADMISSIBLE for grade in annotation.grades.values()
                )
            ),
        }
    report["meta"].update(
        {
            "advanced_query_limit": settings.advanced_query_limit,
            "super_scope_limit": settings.super_scope_limit,
            "super_refine_enabled": settings.super_refine_enabled,
            "answer_max_citations": settings.answer_max_citations,
        }
    )
    if judge is not None:
        report["meta"]["judge_provider"] = getattr(judge.provider, "name", None)
        report["meta"]["judge_chat_model"] = getattr(judge.provider, "chat_model", None)
        report["meta"]["judge_strict"] = strict_judge
        report["meta"]["judge_strict_v2"] = strict_judge_v2
        report["meta"]["judge_mode"] = (
            "quote"
            if quote_judge
            else "strict_v2"
            if strict_judge_v2
            else "strict"
            if strict_judge
            else "binary"
        )
    report["mode"] = "super"
    return report


def _aggregate(
    rows: list[dict[str, Any]],
    cases: Sequence[Any],
    pages: dict[str, tuple[str, int]],
    annotations: Sequence[MultiGoldCase] | None = None,
) -> dict[str, Any]:
    retrieval: dict[str, Any] = {
        "cases_unmapped": sum(
            1
            for case in cases
            if any(chunk_id not in pages for chunk_id in case.expected_chunk_ids)
        ),
        "page_recall": _mean(rows, "page_recall"),
        "source_page_recall": _mean(rows, "source_page_recall"),
        "citation_precision": _mean(rows, "citation_precision"),
        "citation_book_precision": _mean(rows, "citation_book_precision"),
    }
    if annotations is not None:
        sizes = [len(annotation.grades) for annotation in annotations]
        retrieval["multigold"] = {
            "chunks_per_case": {
                "min": min(sizes),
                "max": max(sizes),
                "mean": round(sum(sizes) / len(sizes), 2),
            },
            "cases_with_alternatives": sum(
                1
                for annotation in annotations
                if any(
                    grade == GRADE_ADMISSIBLE for grade in annotation.grades.values()
                )
            ),
            "citation_precision": _mean_nested(rows, "citation_precision"),
            "page_hit": _mean_nested(rows, "page_hit"),
            "source_page_hit": _mean_nested(rows, "source_page_hit"),
            "page_recall": _mean_nested(rows, "page_recall"),
        }
    return {
        "cases": len(rows),
        "retrieval": retrieval,
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
            "faithfulness": _mean_optional(rows, "faithfulness"),
            "citation_support_precision": _mean_optional(
                rows, "citation_support_precision"
            ),
            "citation_usage": _mean_optional(rows, "citation_usage"),
            "citation_support_rate": _mean_optional(rows, "citation_support_rate"),
            "avg_answer_words": round(_mean(rows, "answer_words"), 1),
            "avg_answer_words_answered": round(
                _mean(
                    [row for row in rows if not row["refused"]],
                    "answer_words",
                ),
                1,
            ),
            "judged_cases": sum(1 for row in rows if row["faithfulness"] is not None),
            "claims_truncated": sum(1 for row in rows if row["claims_truncated"]),
            "faithfulness_errors": sum(1 for row in rows if row["faithfulness_error"]),
        },
        "plan": {
            "avg_queries": round(_mean(rows, "plan"), 2),
            "avg_rounds": round(_mean(rows, "rounds"), 2),
            "avg_extra_queries": round(_mean(rows, "extra_queries"), 2),
        },
        "latency_ms": {
            "p50": _percentile([row["elapsed_ms"] for row in rows], 50),
            "p95": _percentile([row["elapsed_ms"] for row in rows], 95),
        },
        "details": rows,
    }


def _multigold_page_metrics(
    cited_pages: set[tuple[str, int]],
    source_pages: set[tuple[str, int]],
    page_grades: dict[tuple[str, int], int],
) -> dict[str, float]:
    admissible = {page for page, grade in page_grades.items() if grade > 0}
    cited_hits = cited_pages & admissible
    return {
        "citation_precision": len(cited_hits) / len(cited_pages)
        if cited_pages
        else 0.0,
        "page_hit": float(bool(cited_hits)),
        "source_page_hit": float(bool(source_pages & admissible)),
        "page_recall": len(cited_hits) / len(admissible) if admissible else 0.0,
    }


def _mean_nested(rows: list[dict[str, Any]], key: str) -> float:
    values = [
        row["multigold"][key] for row in rows if isinstance(row.get("multigold"), dict)
    ]
    return round(sum(values) / len(values), 4) if values else 0.0


def _mean(rows: list[dict[str, Any]], key: str) -> float:
    return round(sum(row[key] for row in rows) / len(rows), 4) if rows else 0.0


def _mean_optional(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [row[key] for row in rows if row.get(key) is not None]
    return round(sum(values) / len(values), 4) if values else None


__all__ = ["run_agent_eval"]
