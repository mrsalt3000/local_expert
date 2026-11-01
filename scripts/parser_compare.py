"""Local-only сравнение PyMuPDF и Docling на выбранных страницах."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

from book_expert.eval.parser_assessment import parse_page_range
from book_expert.ingestion.docling import (
    DoclingError,
    docling_available,
    extract_docling,
)
from book_expert.ingestion.pdf import extract_pdf

DEFAULT_OUTPUT = Path("data/artifacts/parser-compare.local.json")
_TOKEN_RE = re.compile(r"[а-яёa-z0-9]{3,}", re.IGNORECASE)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdf", type=Path)
    parser.add_argument(
        "--pages",
        default="1-5",
        help="1-based страницы, например 50-54; по умолчанию 1-5",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)

    try:
        selected = parse_page_range(args.pages)
        if not selected:
            raise ValueError("no pages selected")
        if selected != list(range(min(selected), max(selected) + 1)):
            raise ValueError("Docling comparison requires contiguous pages")
        if not docling_available():
            raise DoclingError("docling is not installed; run uv sync --extra docling")
        report = compare(args.pdf, selected)
    except (OSError, ValueError, DoclingError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    if not args.output.name.endswith(".local.json"):
        print("error: output must end with .local.json", file=sys.stderr)
        return 1
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def compare(pdf: Path, selected: list[int]) -> dict[str, Any]:
    if not pdf.is_file():
        raise FileNotFoundError(pdf)
    selected_set = set(selected)
    started = time.perf_counter()
    pdf_doc = extract_pdf(pdf, pages=selected)
    pdf_ms = int((time.perf_counter() - started) * 1000)
    pdf_pages = {
        page.pdf_page_index: page
        for page in pdf_doc.pages
        if page.pdf_page_index in selected_set
    }

    started = time.perf_counter()
    docling_doc = extract_docling(
        pdf,
        page_range=(min(selected) + 1, max(selected) + 1),
        total_pages=pdf_doc.page_count,
    )
    docling_ms = int((time.perf_counter() - started) * 1000)
    docling_pages = {
        page.pdf_page_index: page
        for page in docling_doc.pages
        if page.pdf_page_index in selected_set
    }

    similarities = []
    recalls = []
    low_recall_pages = []
    for index in sorted(selected_set):
        left = pdf_pages.get(index)
        right = docling_pages.get(index)
        if left is None or right is None:
            continue
        left_tokens = _tokens(left.text)
        right_tokens = _tokens(right.text)
        similarities.append(_jaccard(left_tokens, right_tokens))
        recall = _multiset_recall(left.text, right.text)
        recalls.append(recall)
        if recall < 0.9:
            low_recall_pages.append(
                {"pdf_page_index": index, "token_recall": round(recall, 4)}
            )

    return {
        "document_ref": "sha256:" + _sha256(pdf)[:16],
        "pages": sorted(selected_set),
        "runtime_scope": "pymupdf and docling both timed on selected pages",
        "parsers": {
            "pymupdf": _parser_metrics(pdf_pages, pdf_ms),
            "docling": _parser_metrics(docling_pages, docling_ms),
        },
        "page_linkage": {
            "pymupdf": sorted(pdf_pages),
            "docling": sorted(docling_pages),
        },
        "mean_token_jaccard": round(sum(similarities) / len(similarities), 4)
        if similarities
        else 0.0,
        "mean_token_recall": round(sum(recalls) / len(recalls), 4) if recalls else 0.0,
        "low_recall_pages": low_recall_pages,
        "manual_review": (
            "9-criteria rubric still requires a human review; "
            "this report contains only counts and similarity."
        ),
    }


def _parser_metrics(pages: dict[int, Any], elapsed_ms: int) -> dict[str, Any]:
    nonempty = sum(not page.is_empty for page in pages.values())
    return {
        "pages": len(pages),
        "nonempty_pages": nonempty,
        "coverage": round(nonempty / len(pages), 4) if pages else 0.0,
        "chars": sum(page.char_count for page in pages.values()),
        "blocks": sum(page.block_count for page in pages.values()),
        "elapsed_ms": elapsed_ms,
    }


def _tokens(text: str) -> set[str]:
    return {match.group(0).casefold()[:6] for match in _TOKEN_RE.finditer(text)}


def _multiset_recall(reference: str, candidate: str) -> float:
    expected = Counter(
        match.group(0).casefold()[:6] for match in _TOKEN_RE.finditer(reference)
    )
    actual = Counter(
        match.group(0).casefold()[:6] for match in _TOKEN_RE.finditer(candidate)
    )
    if not expected:
        return 1.0
    overlap = sum((expected & actual).values())
    return overlap / sum(expected.values())


def _jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 1.0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
