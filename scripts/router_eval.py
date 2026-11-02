"""Local-only routing eval: вопрос → ожидаемая книга."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from book_expert.catalog import Catalog
from book_expert.config import Settings
from book_expert.providers import get_provider
from book_expert.retrieval.metadata import apply_metadata_filter
from book_expert.retrieval.router import BookRouter, _candidate_books
from book_expert.retrieval.rewrite import rewrite_query

DEFAULT_OUTPUT = Path("data/artifacts/router-eval.local.json")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cases", type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    if not args.output.name.endswith(".local.json"):
        print("error: output must end with .local.json", file=sys.stderr)
        return 1
    try:
        report = evaluate(args.cases)
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if args.json:
        print(json.dumps(report, ensure_ascii=False))
    else:
        print(
            f"cases={report['cases']} top1={report['top1']} "
            f"recall@3={report['recall_at_3']} "
            f"candidate_recall={report['candidate_recall']} "
            f"fallback={report['fallback']}"
        )
    return 0


def evaluate(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not payload:
        raise ValueError("routing eval must be a non-empty JSON list")
    settings = Settings()
    settings.ensure_dirs()
    catalog = Catalog(settings.catalog_path)
    catalog.initialize()
    provider = get_provider(settings)
    router = BookRouter(settings, provider, catalog)
    books = [book for book in catalog.list_books() if book.status.value == "ready"]
    rows: list[dict[str, Any]] = []
    try:
        for index, case in enumerate(payload, start=1):
            if not isinstance(case, dict):
                raise ValueError(f"case {index} must be an object")
            expected = case.get("expected_book_ids")
            question = case.get("question")
            if not isinstance(question, str) or not question.strip():
                raise ValueError(f"case {index}: question is required")
            if (
                not isinstance(expected, list)
                or not expected
                or not all(isinstance(value, str) for value in expected)
            ):
                raise ValueError(f"case {index}: expected_book_ids must be non-empty")
            retrieval_query = (
                rewrite_query(question) if settings.query_rewrite_enabled else question
            )
            candidates = {
                book.id
                for book in _candidate_books(
                    retrieval_query,
                    apply_metadata_filter(books, retrieval_query),
                    catalog,
                    profiles=router.profiles,
                    provider=provider,
                    use_lemmas=settings.lexical_use_lemmas,
                )
            }
            route = router.route(question)
            ids = list(route.book_ids)
            rows.append(
                {
                    "index": index,
                    "topic": case.get("topic", ""),
                    "expected": expected[0],
                    "route": ids,
                    "confidence": route.confidence,
                    "fallback": route.fallback,
                    "top1": bool(ids) and ids[0] == expected[0],
                    "recall_at_3": bool(set(expected) & set(ids[:3])),
                    "candidate_hit": expected[0] in candidates,
                }
            )
    finally:
        provider.close()
    n = len(rows)
    return {
        "cases": n,
        "provider": settings.provider,
        "chat_model": provider.chat_model,
        "profiles": router.profiles is not None,
        "top1": round(sum(row["top1"] for row in rows) / n, 4),
        "recall_at_3": round(
            sum(row["recall_at_3"] for row in rows) / n,
            4,
        ),
        "candidate_recall": round(
            sum(row["candidate_hit"] for row in rows) / n,
            4,
        ),
        "fallback": sum(row["fallback"] for row in rows),
        "details": rows,
    }


if __name__ == "__main__":
    raise SystemExit(main())
