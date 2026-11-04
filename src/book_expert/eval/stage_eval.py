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

import hashlib
import json
import os
import subprocess
import time
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from book_expert.catalog import Catalog
from book_expert.config import Settings
from book_expert.eval.faithfulness import FaithfulnessJudge
from book_expert.eval.metrics import (
    hit_at_k,
    load_cases,
    map_at_k,
    ndcg_at_k,
    recall_at_k,
)
from book_expert.eval.multigold import (
    GRADE_ADMISSIBLE,
    MultiGoldCase,
    load_annotations,
)
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
    k: int | None = None,
    judge_provider: str | None = None,
    strict_judge: bool = False,
    strict_judge_v2: bool = False,
    quote_judge: bool = False,
    multigold: str | Path | None = None,
) -> dict[str, Any]:
    cases = load_cases(cases_path)
    if limit:
        cases = cases[:limit]
    annotations: tuple[MultiGoldCase, ...] | None = None
    if multigold is not None:
        annotations = load_annotations(multigold)[: len(cases)]
        if len(annotations) != len(cases):
            raise ValueError(
                "multigold annotations must cover all evaluated cases "
                f"({len(annotations)} < {len(cases)})"
            )
    result_k = settings.top_k if k is None else k
    if result_k <= 0:
        raise ValueError("k must be positive")
    settings.ensure_dirs()
    catalog = Catalog(settings.catalog_path)
    catalog.initialize()
    provider = get_provider(settings)
    searcher = HybridSearcher(settings, provider, catalog)
    router = BookRouter(settings, provider, catalog)
    answerer = GroundedAnswerer(settings, provider, searcher, router)
    judge_client = (
        get_provider(settings.with_provider(judge_provider))
        if faithfulness and judge_provider
        else None
    )
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
                limit=result_k,
            )
            retrieval_ms = (time.perf_counter() - retrieval_start) * 1000
            retrieved_pages = [
                (hit.book_id, hit.pdf_page_index) for hit in outcome.hits
            ]
            top5_pages = retrieved_pages[:5]
            retrieved_ids = [hit.chunk_id for hit in outcome.hits]
            multigold_metrics = (
                _multigold_metrics(
                    annotations[index - 1],
                    retrieved_ids,
                    list(dict.fromkeys(retrieved_pages)),
                    pages,
                    result_k,
                )
                if annotations is not None
                else None
            )
            page_hits = len(expected_pages & set(top5_pages))
            page_hits_k = len(expected_pages & set(retrieved_pages))
            chunk_hits = len(
                expected_chunks & {hit.chunk_id for hit in outcome.hits[:5]}
            )
            chunk_hits_k = len(expected_chunks & {hit.chunk_id for hit in outcome.hits})

            answer_start = time.perf_counter()
            answer = answerer.ask(
                case.question,
                book_ids=search_books,
                limit=result_k,
                relevance_filter=relevance_filter,
            )
            answer_ms = (time.perf_counter() - answer_start) * 1000
            judgement = None
            if judge is not None and not answer.refused:
                judgement = judge.evaluate(
                    case.question,
                    answer.text,
                    tuple(hit.text for hit in answer.sources),
                    citation_count=len(answer.citations),
                )
            cited_pages = {
                (citation.book_id, citation.pdf_page_index)
                for citation in answer.citations
            }
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
                "page_recall@k": page_hits_k / len(expected_pages)
                if expected_pages
                else 0.0,
                "chunk_recall@5": chunk_hits / len(expected_chunks)
                if expected_chunks
                else 0.0,
                "chunk_recall@k": chunk_hits_k / len(expected_chunks)
                if expected_chunks
                else 0.0,
                "page_mrr@5": _page_mrr(top5_pages, expected_pages, 5),
                "page_mrr@k": _page_mrr(retrieved_pages, expected_pages, result_k),
                "retrieval_ms": round(retrieval_ms, 1),
                "answer_ms": round(answer_ms, 1),
                "answer_refused": answer.refused,
                "uncited": answer.uncited,
                "unsupported_claims": len(answer.unsupported_claims),
                "citations": len(answer.citations),
                "citation_precision": citation_hits / len(cited_pages)
                if cited_pages
                else 0.0,
                "citation_recall": citation_hits / len(expected_pages)
                if expected_pages
                else 0.0,
                "faithfulness": judgement.faithfulness if judgement else None,
                "citation_support_precision": (
                    judgement.citation_support_precision if judgement else None
                ),
                "citation_coverage": (
                    judgement.citation_coverage if judgement else None
                ),
                "citation_usage": judgement.citation_usage if judgement else None,
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
                "gate_signal": not outcome.hits,
                "gate_oracle": bool(expected_pages) and page_hits == 0,
            }
            rows.append(row)
            if index % 10 == 0:
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
    report = _aggregate(
        rows,
        cases=cases,
        pages=pages,
        scope=scope,
        relevance_filter=relevance_filter,
        faithfulness=faithfulness,
        annotations=annotations,
    )
    report["k"] = result_k
    report["meta"] = _report_meta(
        settings,
        provider,
        cases_path,
        result_k=result_k,
        scope=scope,
        relevance_filter=relevance_filter,
        faithfulness=faithfulness,
        cases=len(cases),
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
    return report


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
            limit=settings.candidate_limit,
        )
        return [book.id for book in candidates] or None
    return list(expected_books) or None


def _page_mrr(
    retrieved_pages: list[tuple[str, int]],
    expected_pages: set[tuple[str, int]],
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
    cases: Sequence[Any],
    pages: dict[str, tuple[str, int]],
    scope: ScopeMode,
    relevance_filter: bool,
    faithfulness: bool = False,
    annotations: Sequence[MultiGoldCase] | None = None,
) -> dict[str, Any]:
    routed = [row for row in rows if not row["route_fallback"]]
    report: dict[str, Any] = {
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
            "cases_unmapped": sum(
                1
                for case in cases
                if any(chunk_id not in pages for chunk_id in case.expected_chunk_ids)
            ),
            "page_recall_at_5": _mean(rows, "page_recall@5"),
            "page_recall_at_k": _mean(rows, "page_recall@k"),
            "page_mrr_at_5": _mean(rows, "page_mrr@5"),
            "page_mrr_at_k": _mean(rows, "page_mrr@k"),
            "chunk_recall_at_5": _mean(rows, "chunk_recall@5"),
            "chunk_recall_at_k": _mean(rows, "chunk_recall@k"),
        },
        "answers": {
            "refused": sum(bool(row["answer_refused"]) for row in rows),
            "citation_precision": _mean(rows, "citation_precision"),
            "citation_recall": _mean(rows, "citation_recall"),
            "faithfulness": _mean_optional(rows, "faithfulness"),
            "citation_support_precision": _mean_optional(
                rows, "citation_support_precision"
            ),
            "citation_usage": _mean_optional(rows, "citation_usage"),
            "citation_support_rate": _mean_optional(rows, "citation_support_rate"),
            "judged_cases": sum(1 for row in rows if row["faithfulness"] is not None),
            "claims_truncated": sum(1 for row in rows if row["claims_truncated"]),
            "faithfulness_errors": sum(1 for row in rows if row["faithfulness_error"]),
            "no_citations": sum(
                row["citations"] == 0 and not row["answer_refused"] for row in rows
            ),
            "uncited": sum(1 for row in rows if row["uncited"]),
            "unsupported_claims": sum(row["unsupported_claims"] for row in rows),
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
    if annotations is not None:
        report["retrieval_multigold"] = _multigold_summary(rows, annotations)
    return report


def _multigold_metrics(
    annotation: MultiGoldCase,
    retrieved_chunks: Sequence[str],
    retrieved_pages: Sequence[tuple[str, int]],
    pages: dict[str, tuple[str, int]],
    k: int,
) -> dict[str, float]:
    page_grades = annotation.pages(pages)
    chunk_grades = dict(annotation.grades)
    return {
        "chunk_hit@5": hit_at_k(retrieved_chunks, chunk_grades, 5),
        "chunk_hit@k": hit_at_k(retrieved_chunks, chunk_grades, k),
        "chunk_recall@5": recall_at_k(retrieved_chunks, chunk_grades, 5),
        "chunk_recall@k": recall_at_k(retrieved_chunks, chunk_grades, k),
        "chunk_map@k": map_at_k(retrieved_chunks, chunk_grades, k),
        "chunk_ndcg@k": ndcg_at_k(retrieved_chunks, chunk_grades, k),
        "page_hit@5": hit_at_k(retrieved_pages, page_grades, 5),
        "page_hit@k": hit_at_k(retrieved_pages, page_grades, k),
        "page_recall@5": recall_at_k(retrieved_pages, page_grades, 5),
        "page_recall@k": recall_at_k(retrieved_pages, page_grades, k),
        "page_map@k": map_at_k(retrieved_pages, page_grades, k),
        "page_ndcg@k": ndcg_at_k(retrieved_pages, page_grades, k),
    }


def _multigold_summary(
    rows: list[dict[str, Any]], annotations: Sequence[MultiGoldCase]
) -> dict[str, Any]:
    def mean(key: str) -> float:
        values = [
            row["multigold"][key]
            for row in rows
            if isinstance(row.get("multigold"), dict)
        ]
        return round(sum(values) / len(values), 4) if values else 0.0

    sizes = [len(annotation.grades) for annotation in annotations]
    return {
        "cases": len(rows),
        "chunks_per_case": {
            "min": min(sizes),
            "max": max(sizes),
            "mean": round(sum(sizes) / len(sizes), 2),
        },
        "cases_with_alternatives": sum(
            1
            for annotation in annotations
            if any(grade == GRADE_ADMISSIBLE for grade in annotation.grades.values())
        ),
        "chunk_hit_at_5": mean("chunk_hit@5"),
        "chunk_hit_at_k": mean("chunk_hit@k"),
        "chunk_recall_at_5": mean("chunk_recall@5"),
        "chunk_recall_at_k": mean("chunk_recall@k"),
        "chunk_map_at_k": mean("chunk_map@k"),
        "chunk_ndcg_at_k": mean("chunk_ndcg@k"),
        "page_hit_at_5": mean("page_hit@5"),
        "page_hit_at_k": mean("page_hit@k"),
        "page_recall_at_5": mean("page_recall@5"),
        "page_recall_at_k": mean("page_recall@k"),
        "page_map_at_k": mean("page_map@k"),
        "page_ndcg_at_k": mean("page_ndcg@k"),
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


def _report_meta(
    settings: Settings,
    provider: Any,
    cases_path: str | Path,
    *,
    result_k: int,
    scope: ScopeMode,
    relevance_filter: bool,
    faithfulness: bool,
    cases: int,
) -> dict[str, Any]:
    path = Path(cases_path)
    try:
        dataset_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        dataset_sha256 = None
    return {
        "provider": getattr(provider, "name", None),
        "chat_model": getattr(provider, "chat_model", None),
        "embedding_model": getattr(provider, "embedding_model", None),
        "top_k_setting": settings.top_k,
        "candidate_limit": settings.candidate_limit,
        "k": result_k,
        "scope": scope,
        "relevance_filter": relevance_filter,
        "faithfulness": faithfulness,
        "cases": cases,
        "qdrant_collection": settings.qdrant_collection,
        "git_commit": _git_commit(settings.workspace_path),
        "git_dirty": _git_dirty(settings.workspace_path),
        "dataset": path.name,
        "dataset_sha256": dataset_sha256,
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "env_book_expert": sorted(
            key for key in os.environ if key.startswith("BOOK_EXPERT_")
        ),
    }


def _git_commit(workspace: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=workspace,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    commit = result.stdout.strip()
    return commit or None


def _git_dirty(workspace: Path) -> bool | None:
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=workspace,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return bool(result.stdout.strip())


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
