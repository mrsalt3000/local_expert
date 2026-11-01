"""Формат-агностичный контракт извлечённого документа."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


class DocumentError(RuntimeError):
    """Базовая ошибка извлечения документа."""


class UnsupportedFormatError(DocumentError):
    """Формат файла не поддерживается."""


@dataclass(frozen=True, slots=True)
class ExtractedPage:
    pdf_page_index: int
    text: str
    printed_page_label: str | None = None
    char_count: int = 0
    block_count: int = 0
    width: float = 0.0
    height: float = 0.0

    @property
    def is_empty(self) -> bool:
        return not self.text.strip()


@dataclass(frozen=True, slots=True)
class ExtractedDocument:
    path: Path
    content_hash: str
    page_count: int
    pages: tuple[ExtractedPage, ...]
    title: str | None = None
    author: str | None = None
    metadata: dict[str, str] | None = None
    format: str = "pdf"

    @property
    def empty_page_indices(self) -> tuple[int, ...]:
        return tuple(page.pdf_page_index for page in self.pages if page.is_empty)

    @property
    def has_text_layer(self) -> bool:
        return any(not page.is_empty for page in self.pages)


__all__ = [
    "DocumentError",
    "ExtractedDocument",
    "ExtractedPage",
    "UnsupportedFormatError",
]
