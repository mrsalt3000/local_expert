#!/usr/bin/env python3
"""Local-only parser assessment run (PyMuPDF baseline)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from book_expert.eval.parser_assessment import (
    AssessmentError,
    build_manifest,
    parse_page_range,
)

DEFAULT_OUTPUT = Path("data/eval/parser_assessment.local.json")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdf", type=Path, help="локальный PDF (не коммитится)")
    parser.add_argument(
        "--pages",
        default="",
        help="1-based страницы, например 1-20,25; по умолчанию все",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="путь *.local.json (git-ignored)",
    )
    args = parser.parse_args(argv)

    try:
        pages = parse_page_range(args.pages)
        manifest = build_manifest(args.pdf, pages=pages, output=args.output)
    except (OSError, ValueError, AssessmentError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    records = manifest["records"]
    coverage = sum(record["criteria"]["text_coverage"] == "pass" for record in records)
    empty = len(records) - coverage
    print(
        f"wrote {args.output}: {len(records)} pages, "
        f"text_coverage pass={coverage}, fail={empty}"
    )
    print("manual rubric review is still required")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
