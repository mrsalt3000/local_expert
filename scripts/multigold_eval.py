"""Multi-gold оценка retrieval (T83).

Сравнивает текущие single-gold метрики с разметкой, где у кейса несколько
допустимых фрагментов (та же страница, сосед по разделу, почти дословный
дубль в другой книге). Прод-код поиска не меняется: скрипт только считает.

Пример:

    uv run python scripts/multigold_eval.py \
        data/eval/multi-book-40.local.json \
        --annotations-out data/eval/multi-book-40-multigold.local.json \
        --output data/artifacts/multigold-69.local.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from collections import Counter
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from book_expert.catalog import Catalog
from book_expert.config import Settings
from book_expert.eval.metrics import (
    EvalCase,
    hit_at_k,
    load_cases,
    map_at_k,
    ndcg_at_k,
    recall_at_k,
)
from book_expert.eval.multigold import (
    GRADE_GOLD,
    MultiGoldCase,
    build_multigold,
    find_duplicate_neighbors,
    load_chunks,
)
from book_expert.eval.stage_eval import _candidate_books
from book_expert.providers import get_provider
from book_expert.retrieval.hybrid import HybridSearcher
from book_expert.retrieval.metadata import apply_metadata_filter
from book_expert.retrieval.rewrite import rewrite_query
from book_expert.retrieval.router import BookRouter

ANNOTATION_VERSION = "structural-v2"


def _duplicate_map(
    conn: sqlite3.Connection,
    cases: Sequence[EvalCase],
    chunks_by_id: dict[str, Any],
    *,
    min_jaccard: float,
) -> dict[int, list[str]]:
    result: dict[int, list[str]] = {}
    for index, case in enumerate(cases):
        found: list[str] = []
        for chunk_id in case.expected_chunk_ids:
            gold = chunks_by_id.get(chunk_id)
            if gold is None:
                continue
            for duplicate_id, _score in find_duplicate_neighbors(
                conn,
                gold.text,
                exclude_ids=(*case.expected_chunk_ids, *found),
                exclude_book_ids=(gold.book_id,),
                min_jaccard=min_jaccard,
                limit=2,
            ):
                if duplicate_id not in found:
                    found.append(duplicate_id)
        if found:
            result[index] = found
    return result


def _page_sequence(hits: Sequence[Any]) -> list[tuple[str, int]]:
    seen: set[tuple[str, int]] = set()
    pages: list[tuple[str, int]] = []
    for hit in hits:
        page = (hit.book_id, hit.pdf_page_index)
        if page not in seen:
            seen.add(page)
            pages.append(page)
    return pages


def _metrics(
    retrieved_chunks: Sequence[str],
    retrieved_pages: Sequence[tuple[str, int]],
    *,
    chunk_grades: dict[str, int],
    page_grades: dict[tuple[str, int], int],
    k: int,
) -> dict[str, float]:
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


def _aggregate(
    rows: Sequence[dict[str, Any]],
    annotations: Sequence[MultiGoldCase],
) -> dict[str, Any]:
    def mean(key: str) -> float:
        return round(sum(row[key] for row in rows) / len(rows), 4) if rows else 0.0

    reasons = Counter(
        reason for annotation in annotations for reason in annotation.reasons.values()
    )
    sizes = [len(annotation.grades) for annotation in annotations]
    changed = [
        row
        for row in rows
        if row["multi_page_hit@k"] != row["single_page_hit@k"]
        or row["multi_chunk_hit@k"] != row["single_chunk_hit@k"]
    ]
    return {
        "cases": len(rows),
        "annotation": {
            "version": ANNOTATION_VERSION,
            "chunks_per_case": {
                "min": min(sizes) if sizes else 0,
                "max": max(sizes) if sizes else 0,
                "mean": round(sum(sizes) / len(sizes), 2) if sizes else 0.0,
            },
            "cases_with_alternatives": sum(
                1
                for annotation in annotations
                if any(grade == 1 for grade in annotation.grades.values())
            ),
            "grades": dict(reasons),
        },
        "single": {
            "chunk_recall@5": mean("single_chunk_recall@5"),
            "chunk_recall@k": mean("single_chunk_recall@k"),
            "page_recall@5": mean("single_page_recall@5"),
            "page_recall@k": mean("single_page_recall@k"),
        },
        "multi": {
            "chunk_hit@5": mean("multi_chunk_hit@5"),
            "chunk_hit@k": mean("multi_chunk_hit@k"),
            "chunk_recall@5": mean("multi_chunk_recall@5"),
            "chunk_recall@k": mean("multi_chunk_recall@k"),
            "chunk_map@k": mean("multi_chunk_map@k"),
            "chunk_ndcg@k": mean("multi_chunk_ndcg@k"),
            "page_hit@5": mean("multi_page_hit@5"),
            "page_hit@k": mean("multi_page_hit@k"),
            "page_recall@5": mean("multi_page_recall@5"),
            "page_recall@k": mean("multi_page_recall@k"),
            "page_map@k": mean("multi_page_map@k"),
            "page_ndcg@k": mean("multi_page_ndcg@k"),
        },
        "changed_cases": len(changed),
        "changed_indexes": sorted(row["index"] + 1 for row in changed),
        "details": list(rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument(
        "--min-jaccard",
        type=float,
        default=0.6,
        help="порог для почти дословных дублей между книгами",
    )
    parser.add_argument(
        "--neighbour-min-jaccard",
        type=float,
        default=0.3,
        help="минимальное пересечение с gold для соседа по странице/разделу",
    )
    parser.add_argument(
        "--no-duplicates",
        action="store_true",
        help="не искать дубли по тексту (быстрее, беднее разметка)",
    )
    parser.add_argument(
        "--audit-only",
        action="store_true",
        help="построить разметку и статистику без запуска retrieval",
    )
    parser.add_argument("--annotations-out", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    settings = Settings(top_k=args.k)
    catalog = Catalog(settings.catalog_path)
    catalog.initialize()
    cases = load_cases(args.dataset)
    cases = (
        cases[args.offset : args.offset + args.limit]
        if args.limit
        else cases[args.offset :]
    )
    conn = sqlite3.connect(settings.catalog_path)
    eval_books = {book_id for case in cases for book_id in case.expected_book_ids}
    chunks = load_chunks(conn, eval_books)
    chunks_by_id = {chunk.id: chunk for chunk in chunks}
    duplicates = (
        {}
        if args.no_duplicates
        else _duplicate_map(conn, cases, chunks_by_id, min_jaccard=args.min_jaccard)
    )
    annotations = build_multigold(
        cases,
        chunks,
        duplicates=duplicates,
        neighbour_min_jaccard=args.neighbour_min_jaccard,
    )
    chunk_pages = {chunk.id: (chunk.book_id, chunk.pdf_page_index) for chunk in chunks}

    report: dict[str, Any] = {
        "dataset": args.dataset.name,
        "dataset_sha256": hashlib.sha256(args.dataset.read_bytes()).hexdigest(),
        "k": args.k,
        "offset": args.offset,
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "annotation_version": ANNOTATION_VERSION,
        "neighbour_min_jaccard": args.neighbour_min_jaccard,
        "duplicates": {
            "enabled": not args.no_duplicates,
            "min_jaccard": args.min_jaccard,
            "cases": len(duplicates),
        },
    }

    if args.annotations_out:
        args.annotations_out.parent.mkdir(parents=True, exist_ok=True)
        args.annotations_out.write_text(
            json.dumps(
                {
                    "dataset": args.dataset.name,
                    "version": ANNOTATION_VERSION,
                    "cases": [annotation.to_dict() for annotation in annotations],
                },
                ensure_ascii=False,
                indent=1,
            )
            + "\n",
            encoding="utf-8",
        )

    if args.audit_only:
        report["annotation"] = _aggregate([], annotations)["annotation"]
        print(json.dumps(report, ensure_ascii=False, indent=1))
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(
                json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8"
            )
        return

    provider = get_provider(settings)
    searcher = HybridSearcher(settings, provider, catalog)
    router = BookRouter(settings, provider, catalog)
    ready = [book for book in catalog.list_books() if book.status.value == "ready"]
    rows: list[dict[str, Any]] = []
    try:
        for index, (case, annotation) in enumerate(
            zip(cases, annotations, strict=True), start=1
        ):
            query = (
                rewrite_query(case.question)
                if settings.query_rewrite_enabled
                else case.question
            )
            candidates = _candidate_books(
                query,
                apply_metadata_filter(ready, query),
                catalog,
                profiles=router.profiles,
                provider=provider,
                use_lemmas=settings.lexical_use_lemmas,
                limit=settings.candidate_limit,
            )
            outcome = searcher.search(
                query,
                book_ids=[book.id for book in candidates],
                limit=args.k,
            )
            retrieved_chunks = [hit.chunk_id for hit in outcome.hits]
            retrieved_pages = _page_sequence(outcome.hits)
            single_chunk_grades = {
                chunk_id: GRADE_GOLD for chunk_id in case.expected_chunk_ids
            }
            single_page_grades = {
                chunk_pages[chunk_id]: GRADE_GOLD
                for chunk_id in case.expected_chunk_ids
                if chunk_id in chunk_pages
            }
            single = _metrics(
                retrieved_chunks,
                retrieved_pages,
                chunk_grades=single_chunk_grades,
                page_grades=single_page_grades,
                k=args.k,
            )
            multi_chunk_grades = dict(annotation.grades)
            multi_page_grades = annotation.pages(chunk_pages)
            multi = _metrics(
                retrieved_chunks,
                retrieved_pages,
                chunk_grades=multi_chunk_grades,
                page_grades=multi_page_grades,
                k=args.k,
            )
            rows.append(
                {
                    "index": index - 1,
                    "question": case.question,
                    "gold_chunks": len(case.expected_chunk_ids),
                    "admissible_chunks": len(multi_chunk_grades),
                    "reasons": sorted(set(annotation.reasons.values())),
                    "retrieved": retrieved_chunks[:5],
                    "single_chunk_recall@5": single["chunk_recall@5"],
                    "single_chunk_recall@k": single["chunk_recall@k"],
                    "single_chunk_hit@5": single["chunk_hit@5"],
                    "single_chunk_hit@k": single["chunk_hit@k"],
                    "single_page_recall@5": single["page_recall@5"],
                    "single_page_recall@k": single["page_recall@k"],
                    "single_page_hit@k": single["page_hit@k"],
                    "multi_chunk_hit@5": multi["chunk_hit@5"],
                    "multi_chunk_hit@k": multi["chunk_hit@k"],
                    "multi_chunk_recall@5": multi["chunk_recall@5"],
                    "multi_chunk_recall@k": multi["chunk_recall@k"],
                    "multi_chunk_map@k": multi["chunk_map@k"],
                    "multi_chunk_ndcg@k": multi["chunk_ndcg@k"],
                    "multi_page_hit@5": multi["page_hit@5"],
                    "multi_page_hit@k": multi["page_hit@k"],
                    "multi_page_recall@5": multi["page_recall@5"],
                    "multi_page_recall@k": multi["page_recall@k"],
                    "multi_page_map@k": multi["page_map@k"],
                    "multi_page_ndcg@k": multi["page_ndcg@k"],
                }
            )
            if index % 10 == 0:
                print(f"{index}/{len(cases)}", flush=True)
    finally:
        searcher.close()
        close = getattr(provider, "close", None)
        if callable(close):
            close()

    report["meta"] = {
        "provider": getattr(provider, "name", None),
        "embedding_model": getattr(provider, "embedding_model", None),
        "qdrant_collection": settings.qdrant_collection,
        "candidate_limit": settings.candidate_limit,
        "query_rewrite": settings.query_rewrite_enabled,
    }
    report.update(_aggregate(rows, annotations))
    print(
        json.dumps(
            {k: v for k, v in report.items() if k != "details"},
            ensure_ascii=False,
            indent=1,
        )
    )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8"
        )


if __name__ == "__main__":
    main()
