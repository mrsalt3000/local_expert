"""План дедупликации каталога: оставить самую полную копию, остальные — из RAG."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from book_expert.catalog_audit import DuplicateGroup, audit_catalog
from book_expert.models import Book


@dataclass(frozen=True, slots=True)
class DedupeRemoval:
    group_key: str
    keep_id: str
    book: Book

    def to_dict(self) -> dict[str, Any]:
        return {
            "group_key": self.group_key,
            "keep_id": self.keep_id,
            "book": _book_summary(self.book),
        }


@dataclass(frozen=True, slots=True)
class DedupePlan:
    groups: tuple[DuplicateGroup, ...]
    removals: tuple[DedupeRemoval, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "groups": [group.to_dict() for group in self.groups],
            "removals": [removal.to_dict() for removal in self.removals],
        }


def build_dedupe_plan(books: list[Book] | tuple[Book, ...]) -> DedupePlan:
    """Строит план без изменения каталога."""
    audit = audit_catalog(books, check_files=False)
    removals: list[DedupeRemoval] = []
    for group in audit.duplicates:
        for book in group.books:
            if book.id != group.keep_id:
                removals.append(
                    DedupeRemoval(
                        group_key=group.key,
                        keep_id=group.keep_id,
                        book=book,
                    )
                )
    return DedupePlan(groups=audit.duplicates, removals=tuple(removals))


def _book_summary(book: Book) -> dict[str, Any]:
    return {
        "id": book.id,
        "title": book.title,
        "author": book.author,
        "status": book.status.value,
        "format": book.format,
        "pages": book.page_count,
        "chunks": book.chunk_count,
        "path": str(book.path),
    }


__all__ = ["DedupePlan", "DedupeRemoval", "build_dedupe_plan"]
