"""Retrieval A/B harness for T65-T67 experiments.

Variants (all use the same candidate scope and book+page metrics):
  or     - add OR-prefix5 lexical channel to RRF (weight 0.1)
  scope  - add books from global dense top-20 to candidate scope
  expand - LLM query expansion before routing and retrieval
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
from pathlib import Path

from book_expert.catalog import Catalog
from book_expert.config import Settings
from book_expert.eval.metrics import load_cases
from book_expert.eval.stage_eval import _candidate_books
from book_expert.providers import ChatMessage, Provider, get_provider
from book_expert.retrieval.hybrid import RRF_K, HybridSearcher
from book_expert.retrieval.metadata import apply_metadata_filter
from book_expert.retrieval.rewrite import rewrite_query
from book_expert.retrieval.router import BookRouter

EXPAND_PROMPT = (
    "Ты расширяешь поисковый запрос для RAG по книгам. Дай 5–8 ключевых слов "
    "и синонимов, которые могут встретиться в тексте нужной страницы. "
    "Ответ — только слова через запятую, без пояснений."
)


def _fuse(lists: list[tuple[list, float]], limit: int) -> list:
    scores: dict[str, float] = {}
    best: dict[str, object] = {}
    for hits, weight in lists:
        if weight <= 0:
            continue
        for rank, hit in enumerate(hits, start=1):
            scores[hit.chunk_id] = scores.get(hit.chunk_id, 0.0) + weight / (
                RRF_K + rank
            )
            best.setdefault(hit.chunk_id, hit)
    ordered = sorted(scores, key=lambda chunk_id: (-scores[chunk_id], chunk_id))
    return [best[chunk_id] for chunk_id in ordered[:limit]]


def _metrics(rows: list[tuple[bool, bool, int | None]], name: str) -> dict:
    total = len(rows)
    return {
        "variant": name,
        "cases": total,
        "recall@5": round(sum(1 for row in rows if row[0]) / total, 4),
        "recall@20": round(sum(1 for row in rows if row[1]) / total, 4),
        "mrr@20": round(sum(1.0 / row[2] for row in rows if row[2]) / total, 4),
    }


def _expected(
    db: sqlite3.Connection, chunk_ids: tuple[str, ...]
) -> set[tuple[str, int]]:
    if not chunk_ids:
        return set()
    placeholders = ",".join("?" for _ in chunk_ids)
    return {
        (row[0], row[1])
        for row in db.execute(
            f"select book_id, pdf_page_index from chunks where id in ({placeholders})",
            chunk_ids,
        )
    }


def _expand(provider: Provider, question: str) -> str:
    response = provider.chat(
        [
            ChatMessage(role="system", content=EXPAND_PROMPT),
            ChatMessage(role="user", content=question),
        ],
        temperature=0.0,
        max_tokens=80,
    )
    terms = [
        term.strip(" .;:—-")
        for term in re.split(r"[,;\n]", response.text)
        if term.strip()
    ][:8]
    return rewrite_query(question) + " " + " ".join(terms)


def _measure(hits, expected: set[tuple[str, int]]) -> tuple[bool, bool, int | None]:
    pairs = [(hit.book_id, hit.pdf_page_index) for hit in hits]
    return (
        bool(set(pairs[:5]) & expected),
        bool(set(pairs) & expected),
        next((rank for rank, pair in enumerate(pairs, 1) if pair in expected), None),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument(
        "--variant", choices=("or", "scope", "expand", "candidates"), required=True
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--variant-limit",
        type=int,
        default=40,
        help="candidate scope для --variant candidates (baseline всегда 24).",
    )
    args = parser.parse_args()

    settings = Settings(top_k=20)
    catalog = Catalog(settings.catalog_path)
    catalog.initialize()
    provider = get_provider(settings)
    searcher = HybridSearcher(settings, provider, catalog)
    router = BookRouter(settings, provider, catalog)
    ready = [book for book in catalog.list_books() if book.status.value == "ready"]
    cases = load_cases(args.dataset)
    db = sqlite3.connect(settings.catalog_path)
    base: list[tuple[bool, bool, int | None]] = []
    variant: list[tuple[bool, bool, int | None]] = []

    try:
        for case in cases:
            expected = _expected(db, case.expected_chunk_ids)
            if not expected:
                continue
            query = rewrite_query(case.question)
            candidates = _candidate_books(
                query,
                apply_metadata_filter(ready, query),
                catalog,
                profiles=router.profiles,
                provider=provider,
                use_lemmas=settings.lexical_use_lemmas,
                limit=settings.candidate_limit,
            )
            ids = [book.id for book in candidates]
            if args.variant == "or":
                dense = searcher._dense_search(query, ids, 80)
                lex_and = catalog.search_lexical(
                    query, book_ids=ids, limit=80, query_mode="and"
                )
                lex_or = catalog.search_lexical(
                    query, book_ids=ids, limit=80, query_mode="or_prefix5"
                )
                base_hits = _fuse([(dense, 1.0), (lex_and, 0.1)], 20)
                var_hits = _fuse([(dense, 1.0), (lex_and, 0.1), (lex_or, 0.1)], 20)
            elif args.variant == "scope":
                dense_global = searcher._dense_search(query, None, 20)
                wide = sorted(
                    {book.id for book in candidates} | {h.book_id for h in dense_global}
                )
                base_hits = searcher.search(query, book_ids=ids, limit=20).hits
                var_hits = searcher.search(query, book_ids=wide, limit=20).hits
            elif args.variant == "candidates":
                wide_candidates = _candidate_books(
                    query,
                    apply_metadata_filter(ready, query),
                    catalog,
                    profiles=router.profiles,
                    provider=provider,
                    use_lemmas=settings.lexical_use_lemmas,
                    limit=args.variant_limit,
                )
                base_hits = searcher.search(query, book_ids=ids, limit=20).hits
                var_hits = searcher.search(
                    query,
                    book_ids=[book.id for book in wide_candidates],
                    limit=20,
                ).hits
            else:
                expanded = _expand(provider, case.question)
                wide_candidates = _candidate_books(
                    expanded,
                    apply_metadata_filter(ready, expanded),
                    catalog,
                    profiles=router.profiles,
                    provider=provider,
                    use_lemmas=settings.lexical_use_lemmas,
                    limit=settings.candidate_limit,
                )
                base_hits = searcher.search(query, book_ids=ids, limit=20).hits
                var_hits = searcher.search(
                    expanded,
                    book_ids=[book.id for book in wide_candidates],
                    limit=20,
                ).hits
            base.append(_measure(base_hits, expected))
            variant.append(_measure(var_hits, expected))
    finally:
        searcher.close()
        close = getattr(provider, "close", None)
        if callable(close):
            close()

    report = {
        "dataset": args.dataset.name,
        "variant": args.variant,
        "baseline": _metrics(base, "baseline"),
        "variant_metrics": _metrics(variant, args.variant),
    }
    print(json.dumps(report, ensure_ascii=False, indent=1))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8"
        )


if __name__ == "__main__":
    main()
