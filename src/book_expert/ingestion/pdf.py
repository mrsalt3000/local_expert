"""Извлечение PDF: постраничный native text и page map."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import pymupdf

from book_expert.ingestion.document import (
    DocumentError,
    ExtractedDocument,
    ExtractedPage,
)

PdfPage = ExtractedPage
PdfDocument = ExtractedDocument

_LABEL_RE = re.compile(
    r"^(?:[-–—.·]\s*)?(?P<label>\d{1,4}|[ivxlcdmIVXLCDM]{1,7})(?:\s*[-–—.·])?$"
)
_YEAR_RE = re.compile(r"^(?:19|20)\d{2}$")
_STRUCTURAL_RE = re.compile(
    r"^(?:глава|часть|раздел|chapter|part|section)\b", re.IGNORECASE
)
_MARGIN_LINES = 1
_MIN_REPEATS = 4


class PdfError(DocumentError):
    """Базовая ошибка обработки PDF."""


class PdfEncryptedError(PdfError):
    """PDF защищён паролем."""


class PdfParseError(PdfError):
    """PDF не удалось открыть или разобрать."""


def extract_pdf(
    path: str | Path,
    *,
    password: str | None = None,
    strip_running_lines: bool = True,
) -> PdfDocument:
    """Извлекает native text постранично. OCR не выполняется."""
    pdf_path = Path(path).expanduser()
    if not pdf_path.is_file():
        raise FileNotFoundError(pdf_path)

    content_hash = _sha256(pdf_path)
    try:
        document = pymupdf.open(pdf_path)
    except (pymupdf.FileDataError, pymupdf.EmptyFileError, RuntimeError) as exc:
        raise PdfParseError(f"cannot open PDF: {pdf_path.name}") from exc

    try:
        if document.needs_pass:
            if not password:
                raise PdfEncryptedError("PDF is encrypted; password required")
            if not document.authenticate(password):
                raise PdfEncryptedError("invalid PDF password")

        pages = tuple(_extract_page(page, index) for index, page in enumerate(document))
        if strip_running_lines:
            pages = strip_running_lines_from_pages(pages)
        metadata = {
            key: value.strip()
            for key, value in (document.metadata or {}).items()
            if isinstance(value, str) and value.strip()
        }
        return PdfDocument(
            path=pdf_path.resolve(),
            content_hash=content_hash,
            page_count=document.page_count,
            pages=pages,
            title=_optional(metadata.get("title")),
            author=_optional(metadata.get("author")),
            metadata=metadata,
        )
    finally:
        document.close()


def _extract_page(page: Any, index: int) -> PdfPage:
    text = page.get_text("text", sort=True)
    blocks = page.get_text("blocks")
    rect = page.rect
    return PdfPage(
        pdf_page_index=index,
        text=text,
        printed_page_label=_detect_printed_label(text),
        char_count=len(text),
        block_count=len(blocks),
        width=float(rect.width),
        height=float(rect.height),
    )


def detect_printed_label(text: str) -> str | None:
    """Публичная обёртка для эвристики printed page label."""
    return _detect_printed_label(text)


def _detect_printed_label(text: str) -> str | None:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return None
    candidates: list[tuple[int, str]] = []
    for position, line in enumerate(lines[:3]):
        label = _label_from_line(line)
        if label is not None:
            candidates.append((2 + position, label))
    for position, line in enumerate(reversed(lines[-3:])):
        label = _label_from_line(line)
        if label is not None:
            candidates.append((10 + position, label))
    if not candidates:
        return None
    preferred = [item for item in candidates if not _YEAR_RE.match(item[1])]
    if not preferred:
        return None
    return max(preferred, key=lambda item: item[0])[1]


def _label_from_line(line: str) -> str | None:
    match = _LABEL_RE.match(line)
    return match.group("label") if match else None


def strip_running_lines_from_pages(
    pages: Sequence[PdfPage],
) -> tuple[PdfPage, ...]:
    """Убирает повторяющиеся колонтитулы с номерами страниц из текста страниц.

    Срабатывает только на строках в верхнем/нижнем поле страницы, которые
    повторяются минимум на ``_MIN_REPEATS`` страницах и содержат номер
    страницы. Сноски и тело страницы не затрагиваются.
    """
    if len(pages) < _MIN_REPEATS:
        return tuple(pages)

    counts: dict[str, int] = {}
    for page in pages:
        keys = {
            _normalize_margin(line)
            for line in _margin_lines(page.text)
            if _has_edge_number(line) and not _is_structural_heading(line)
        }
        for key in keys:
            counts[key] = counts.get(key, 0) + 1
    repeated = {key for key, count in counts.items() if count >= _MIN_REPEATS}

    max_page = len(pages)
    cleaned: list[PdfPage] = []
    for page in pages:
        lines = page.text.splitlines()
        margins = _margin_positions(lines)
        keep: list[str] = []
        for index, line in enumerate(lines):
            stripped = line.strip()
            if index in margins and stripped:
                if _is_page_number(stripped, max_page):
                    continue
                if (
                    _has_edge_number(stripped)
                    and not _is_structural_heading(stripped)
                    and _normalize_margin(stripped) in repeated
                ):
                    continue
            keep.append(line)
        text = "\n".join(keep)
        cleaned.append(replace(page, text=text, char_count=len(text)))
    return tuple(cleaned)


def _margin_lines(text: str) -> list[str]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[:_MARGIN_LINES] + lines[-_MARGIN_LINES:]


def _margin_positions(lines: Sequence[str]) -> set[int]:
    non_empty = [index for index, line in enumerate(lines) if line.strip()]
    return set(non_empty[:_MARGIN_LINES] + non_empty[-_MARGIN_LINES:])


def _normalize_margin(line: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"\d+", "#", line)).strip().casefold()


def _has_edge_number(line: str) -> bool:
    tokens = line.split()
    if not tokens:
        return False
    for token in (tokens[0], tokens[-1]):
        cleaned = token.strip(".,;:()[]{}—–-\"'«»")
        if cleaned.isdigit() and 1 <= int(cleaned) <= 9999:
            return True
    return False


def _is_page_number(line: str, max_page: int) -> bool:
    match = _LABEL_RE.match(line)
    if match is None:
        return False
    label = match.group("label")
    if label.isdigit():
        return 1 <= int(label) <= max_page
    return True


def _is_structural_heading(line: str) -> bool:
    return _STRUCTURAL_RE.match(line) is not None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _optional(value: str | None) -> str | None:
    return value or None
