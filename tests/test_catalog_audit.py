"""Тесты безопасного аудита каталога."""

from __future__ import annotations

from pathlib import Path

from book_expert.catalog_audit import (
    audit_catalog,
    normalize_title,
    suspicious_reason,
)
from book_expert.models import Book, BookStatus


def _book(
    book_id: str,
    title: str,
    path: Path,
    *,
    chunks: int = 1,
    pages: int = 1,
    status: BookStatus = BookStatus.ready,
) -> Book:
    return Book(
        id=book_id,
        title=title,
        path=path,
        content_hash=f"hash-{book_id}",
        page_count=pages,
        chunk_count=chunks,
        status=status,
    )


def test_normalize_title_strips_year_extension_and_punctuation() -> None:
    assert normalize_title("2020 - Alpha: The Book.pdf") == "alpha the book"
    assert normalize_title("  Ёжик — книга.EPUB ") == "ежик книга"


def test_suspicious_reason_flags_file_names_but_not_short_real_titles() -> None:
    assert suspicious_reason("Untitled-4") == "похоже на имя файла или заглушку"
    assert suspicious_reason("-.indd") == "похоже на имя файла или заглушку"
    assert suspicious_reason("035_1") == "похоже на имя файла или заглушку"
    assert suspicious_reason("1984") is None
    assert suspicious_reason("2666") is None
    assert suspicious_reason("Scrum") is None
    assert suspicious_reason("Zag") is None


def test_audit_groups_duplicates_and_prefers_richer_book(tmp_path: Path) -> None:
    books = [
        _book("a", "2020 - Alpha", tmp_path / "a.pdf", chunks=10, pages=100),
        _book("b", "Alpha", tmp_path / "b.epub", chunks=20, pages=120),
        _book("c", "Beta", tmp_path / "c.pdf"),
    ]

    report = audit_catalog(books, check_files=False)

    assert report.total == 3
    assert report.ready == 3
    assert len(report.duplicates) == 1
    group = report.duplicates[0]
    assert group.key == "alpha"
    assert group.keep_id == "b"
    assert [book.id for book in group.books] == ["b", "a"]


def test_audit_does_not_group_generic_section_titles(tmp_path: Path) -> None:
    books = [
        _book("a", "Вступление", tmp_path / "a.pdf"),
        _book("b", "Вступление", tmp_path / "b.pdf"),
        _book("c", "Alpha", tmp_path / "c.pdf"),
        _book("d", "Alpha", tmp_path / "d.pdf"),
    ]

    report = audit_catalog(books, check_files=False)

    assert [group.key for group in report.duplicates] == ["alpha"]


def test_audit_reports_missing_files_without_deleting(tmp_path: Path) -> None:
    existing = tmp_path / "present.pdf"
    existing.write_text("x", encoding="utf-8")
    indexing = tmp_path / "indexing.pdf"
    indexing.write_text("x", encoding="utf-8")
    books = [
        _book("a", "Present", existing),
        _book("b", "Missing", tmp_path / "missing.epub", status=BookStatus.pending),
        _book("c", "Indexing", indexing, status=BookStatus.indexing),
    ]

    report = audit_catalog(books)

    assert [book.id for book in report.missing_files] == ["b"]
    assert existing.exists()
    assert report.ready == 1
    assert report.pending == 1
    assert report.indexing == 1
