"""Тесты metadata prefilter."""

from __future__ import annotations

from pathlib import Path

from book_expert.models import Book
from book_expert.retrieval.metadata import (
    apply_metadata_filter,
    book_year,
    parse_metadata_filter,
)


def _book(
    book_id: str,
    title: str,
    *,
    author: str | None = None,
    year: int | None = None,
    format: str = "pdf",
    language: str | None = "ru",
) -> Book:
    metadata = {"source": {"date": f"{year}-01-01"}} if year else {}
    return Book(
        id=book_id,
        title=title,
        path=Path(f"{book_id}.{format}"),
        content_hash=f"hash-{book_id}",
        author=author,
        format=format,
        language=language,
        metadata=metadata,
    )


def test_book_year_from_metadata_and_title() -> None:
    assert book_year(_book("a", "Book", year=2020)) == 2020
    assert book_year(_book("b", "2021 - Book")) == 2021
    assert book_year(_book("c", "Book")) is None


def test_parse_explicit_filters() -> None:
    filters = parse_metadata_filter("year:2020 format:epub language:ru author:Иванов")

    assert filters.year == 2020
    assert filters.format == "epub"
    assert filters.language == "ru"
    assert filters.author == "Иванов"


def test_apply_filters_and_fallback() -> None:
    books = [
        _book("a", "A", author="Иванов", year=2020, format="epub"),
        _book("b", "B", author="Петров", year=2022, format="pdf"),
    ]

    assert [book.id for book in apply_metadata_filter(books, "year:2020")] == ["a"]
    assert [book.id for book in apply_metadata_filter(books, "author:Петров")] == ["b"]
    assert [book.id for book in apply_metadata_filter(books, "year:1999")] == [
        "a",
        "b",
    ]
