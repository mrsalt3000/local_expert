"""Извлечение PDF: постраничный native text и page map."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pymupdf

_LABEL_RE = re.compile(
    r"^(?:[-–—.·]\s*)?(?P<label>\d{1,4}|[ivxlcdmIVXLCDM]{1,7})(?:\s*[-–—.·])?$"
)
_YEAR_RE = re.compile(r"^(?:19|20)\d{2}$")


class PdfError(RuntimeError):
    """Базовая ошибка обработки PDF."""


class PdfEncryptedError(PdfError):
    """PDF защищён паролем."""


class PdfParseError(PdfError):
    """PDF не удалось открыть или разобрать."""


@dataclass(frozen=True, slots=True)
class PdfPage:
    pdf_page_index: int
    text: str
    printed_page_label: str | None
    char_count: int
    block_count: int
    width: float
    height: float

    @property
    def is_empty(self) -> bool:
        return not self.text.strip()


@dataclass(frozen=True, slots=True)
class PdfDocument:
    path: Path
    content_hash: str
    page_count: int
    pages: tuple[PdfPage, ...]
    title: str | None = None
    author: str | None = None
    metadata: dict[str, str] | None = None

    @property
    def empty_page_indices(self) -> tuple[int, ...]:
        return tuple(page.pdf_page_index for page in self.pages if page.is_empty)

    @property
    def has_text_layer(self) -> bool:
        return any(not page.is_empty for page in self.pages)


def extract_pdf(path: str | Path, *, password: str | None = None) -> PdfDocument:
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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _optional(value: str | None) -> str | None:
    return value or None
