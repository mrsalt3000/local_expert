"""Автоматизированный parser assessment (PyMuPDF baseline).

Пишет только local-only manifest: без extracted text, путей и цитат.
Ручные оценки rubric остаются за reviewer-ом; скрипт фиксирует измеримые
сигналы (coverage, runtime, page linkage) и помечает остальное для review.
"""

from __future__ import annotations

import hashlib
import json
import pymupdf
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any, Iterable

from book_expert.ingestion.pdf import detect_printed_label

SCHEMA_VERSION = "1.0.0"
PARSER_ID = "pymupdf"
PARSER_CONFIG = {"ocr": False, "sort": True, "parser": PARSER_ID}
REVIEW_NOTE = "Automated heuristic pass; manual rubric review still required."


class AssessmentError(RuntimeError):
    """Не удалось построить локальный assessment."""


def build_manifest(
    pdf_path: str | Path,
    *,
    pages: Iterable[int] | None = None,
    output: str | Path | None = None,
) -> dict[str, Any]:
    path = Path(pdf_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(path)
    document_ref = "sha256:" + _sha256(path)[:16]
    page_indices = sorted(set(pages)) if pages is not None else None

    records: list[dict[str, Any]] = []
    try:
        document = pymupdf.open(path)
    except RuntimeError as exc:
        raise AssessmentError(f"cannot open PDF: {path.name}") from exc
    try:
        if document.needs_pass:
            raise AssessmentError("encrypted PDF: password required")
        indices = page_indices or list(range(document.page_count))
        for index in indices:
            if index < 0 or index >= document.page_count:
                raise AssessmentError(f"page index out of range: {index}")
            started = perf_counter()
            page = document[index]
            text = page.get_text("text", sort=True)
            elapsed_ms = int((perf_counter() - started) * 1000)
            records.append(
                _page_record(
                    index=index,
                    text=text,
                    elapsed_ms=elapsed_ms,
                    document_ref=document_ref,
                )
            )
    finally:
        document.close()

    if not records:
        raise AssessmentError("no pages selected")

    manifest = {
        "manifest_schema_version": SCHEMA_VERSION,
        "assessment_schema_version": SCHEMA_VERSION,
        "document_ref": document_ref,
        "manifest_note": (
            "Local-only automated PyMuPDF baseline; OCR disabled; "
            "manual rubric review required."
        ),
        "records": records,
    }
    if output is not None:
        write_manifest(manifest, output)
    return manifest


def write_manifest(manifest: dict[str, Any], output: str | Path) -> Path:
    target = Path(output).expanduser()
    if not target.name.endswith(".local.json"):
        raise AssessmentError(
            "refusing to write non-local manifest; use *.local.json (git-ignored)"
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return target


def parse_page_range(value: str) -> list[int]:
    """Парсит '1-5,8' в 0-based индексы; пустая строка — все страницы."""
    if not value.strip():
        return []
    pages: list[int] = []
    for part in value.split(","):
        chunk = part.strip()
        if not chunk:
            continue
        if "-" in chunk:
            start_text, end_text = chunk.split("-", 1)
            start = int(start_text)
            end = int(end_text)
            if start <= 0 or end < start:
                raise ValueError(f"invalid page range: {chunk}")
            pages.extend(range(start - 1, end))
        else:
            number = int(chunk)
            if number <= 0:
                raise ValueError(f"invalid page number: {chunk}")
            pages.append(number - 1)
    return pages


def _page_record(
    *,
    index: int,
    text: str,
    elapsed_ms: int,
    document_ref: str,
) -> dict[str, Any]:
    label = detect_printed_label(text) or "unlabelled"
    page_types = _page_types(text)
    has_text = bool(text.strip())
    return {
        "schema_version": SCHEMA_VERSION,
        "parser": {
            "id": PARSER_ID,
            "version": pymupdf.__version__,
            "configuration_hash": _config_hash(),
        },
        "document_ref": document_ref,
        "page_ref": {
            "pdf_page_index": index,
            "printed_page_label": label,
        },
        "page_types": page_types,
        "criteria": {
            "text_coverage": "pass" if has_text else "fail",
            "reading_order": "warning" if has_text else "not_applicable",
            "heading_structure": "warning" if has_text else "not_applicable",
            "table_fidelity": "warning" if "table" in page_types else "not_applicable",
            "footnote_separation": "warning" if has_text else "not_applicable",
            "header_footer_handling": "warning" if has_text else "not_applicable",
            "image_handling": "warning"
            if "image_or_diagram" in page_types
            else "not_applicable",
            "page_linkage": "pass",
            "runtime": _runtime_status(elapsed_ms),
        },
        "review": {
            "reviewer": "automated-pymupdf",
            "reviewed_at": datetime.now(timezone.utc).isoformat(),
            "notes": [REVIEW_NOTE],
        },
        "runtime": {"elapsed_ms": elapsed_ms},
    }


def _page_types(text: str) -> list[str]:
    stripped = text.strip()
    if not stripped:
        return ["image_or_diagram"]
    lines = [line for line in stripped.splitlines() if line.strip()]
    dotted = sum("..." in line or "…" in line for line in lines)
    if len(lines) >= 5 and dotted >= 3:
        return ["table_of_contents"]
    if any("|" in line for line in lines) or any("\t" in line for line in lines):
        return ["table"]
    return ["body_text"]


def _runtime_status(elapsed_ms: int) -> str:
    if elapsed_ms <= 500:
        return "pass"
    if elapsed_ms <= 2000:
        return "warning"
    return "fail"


def _config_hash() -> str:
    payload = json.dumps(PARSER_CONFIG, sort_keys=True).encode()
    return hashlib.sha256(payload).hexdigest()[:16]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


__all__ = [
    "AssessmentError",
    "SCHEMA_VERSION",
    "build_manifest",
    "parse_page_range",
    "write_manifest",
]
