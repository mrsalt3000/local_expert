"""Безопасный аудит каталога: дубли, мусорные названия и пропавшие файлы."""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from book_expert.models import Book, BookStatus

_EXTENSION_RE = re.compile(r"\.(pdf|epub|fb2|djvu)$", re.IGNORECASE)
_YEAR_PREFIX_RE = re.compile(r"^\s*(?:19|20)\d{2}\s*[-–—:]\s*")
_SUSPICIOUS_RE = re.compile(
    r"(?i)(?:untitled|\.indd\b|^[-_.\s]+$|^\d{1,3}(?:[\s_-]+\d+)*$)"
)


def normalize_title(title: str) -> str:
    """Нормализует название для поиска дублей, не меняя сам каталог."""
    text = title.strip().casefold().replace("ё", "е")
    text = _EXTENSION_RE.sub("", text)
    text = _YEAR_PREFIX_RE.sub("", text)
    text = re.sub(r"[^0-9a-zа-я]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def suspicious_reason(title: str) -> str | None:
    """Возвращает причину, почему название стоит проверить вручную."""
    stripped = title.strip()
    if not stripped:
        return "пустое название"
    if _SUSPICIOUS_RE.search(stripped):
        return "похоже на имя файла или заглушку"
    normalized = normalize_title(stripped)
    if not normalized:
        return "пустое название"
    if len(normalized) < 3:
        return "слишком короткое название"
    return None


def _keep_score(book: Book) -> tuple[int, int, int]:
    return (
        book.chunk_count,
        book.page_count,
        int(book.status is BookStatus.ready),
    )


@dataclass(frozen=True, slots=True)
class DuplicateGroup:
    key: str
    books: tuple[Book, ...]
    keep_id: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "keep_id": self.keep_id,
            "books": [_book_summary(book) for book in self.books],
        }


@dataclass(frozen=True, slots=True)
class CatalogAudit:
    total: int
    ready: int
    pending: int
    indexing: int
    failed: int
    duplicates: tuple[DuplicateGroup, ...]
    suspicious: tuple[tuple[Book, str], ...]
    missing_files: tuple[Book, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "ready": self.ready,
            "pending": self.pending,
            "indexing": self.indexing,
            "failed": self.failed,
            "duplicates": [group.to_dict() for group in self.duplicates],
            "suspicious": [
                {**_book_summary(book), "reason": reason}
                for book, reason in self.suspicious
            ],
            "missing_files": [_book_summary(book) for book in self.missing_files],
        }


def audit_catalog(
    books: list[Book] | tuple[Book, ...],
    *,
    check_files: bool = True,
) -> CatalogAudit:
    """Строит отчёт, ничего не изменяя в каталоге и на диске."""
    duplicate_map: dict[str, list[Book]] = defaultdict(list)
    suspicious: list[tuple[Book, str]] = []
    missing: list[Book] = []
    for book in books:
        key = normalize_title(book.title)
        if len(key) >= 3:
            duplicate_map[key].append(book)
        reason = suspicious_reason(book.title)
        if reason is not None:
            suspicious.append((book, reason))
        if check_files and not Path(book.path).exists():
            missing.append(book)

    duplicates = []
    for key, group in duplicate_map.items():
        if len(group) < 2:
            continue
        ordered = tuple(sorted(group, key=_keep_score, reverse=True))
        duplicates.append(DuplicateGroup(key=key, books=ordered, keep_id=ordered[0].id))
    duplicates.sort(key=lambda item: (-len(item.books), item.key))

    return CatalogAudit(
        total=len(books),
        ready=sum(book.status is BookStatus.ready for book in books),
        pending=sum(book.status is BookStatus.pending for book in books),
        indexing=sum(book.status is BookStatus.indexing for book in books),
        failed=sum(book.status is BookStatus.failed for book in books),
        duplicates=tuple(duplicates),
        suspicious=tuple(suspicious),
        missing_files=tuple(missing),
    )


def _book_summary(book: Book) -> dict[str, Any]:
    return {
        "id": book.id,
        "title": book.title,
        "author": book.author,
        "status": book.status.value,
        "format": book.format,
        "pages": book.page_count,
        "chunks": book.chunk_count,
    }


__all__ = [
    "CatalogAudit",
    "DuplicateGroup",
    "audit_catalog",
    "normalize_title",
    "suspicious_reason",
]
