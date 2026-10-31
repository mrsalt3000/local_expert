"""Тесты SQLite-каталога, FTS5-поиска и каскадного удаления."""

from __future__ import annotations

from pathlib import Path

import pytest

from book_expert.catalog import Catalog
from book_expert.models import Book, BookStatus, Chunk, Section


def _catalog(tmp_path: Path) -> Catalog:
    catalog = Catalog(tmp_path / "catalog.sqlite3")
    catalog.initialize()
    return catalog


def _book(
    tmp_path: Path,
    *,
    book_id: str = "book-1",
    content_hash: str = "hash-1",
    title: str = "Synthetic Book",
    status: BookStatus = BookStatus.ready,
) -> Book:
    return Book(
        id=book_id,
        title=title,
        path=tmp_path / f"{book_id}.pdf",
        content_hash=content_hash,
        page_count=2,
        status=status,
        metadata={"language": "en"},
    )


def _chunk(
    book_id: str,
    ordinal: int,
    text: str,
    *,
    page: int = 0,
    section: str | None = "Chapter 1",
) -> Chunk:
    return Chunk(
        id=f"{book_id}-chunk-{ordinal}",
        book_id=book_id,
        text=text,
        pdf_page_index=page,
        ordinal=ordinal,
        section_path=section,
        printed_page_label=str(page + 1),
        token_count=len(text.split()),
    )


def test_upsert_and_get_book(tmp_path: Path) -> None:
    catalog = _catalog(tmp_path)
    stored = catalog.upsert_book(_book(tmp_path))
    assert stored.id == "book-1"
    assert stored.status is BookStatus.ready
    assert stored.metadata == {"language": "en"}

    fetched = catalog.get_book("book-1")
    assert fetched is not None
    assert fetched.title == "Synthetic Book"
    assert catalog.get_book_by_hash("hash-1") is not None
    assert catalog.get_book_by_path(tmp_path / "book-1.pdf") is not None
    assert catalog.get_book("missing") is None


def test_upsert_is_idempotent(tmp_path: Path) -> None:
    catalog = _catalog(tmp_path)
    catalog.upsert_book(_book(tmp_path))
    updated = catalog.upsert_book(
        _book(tmp_path, title="Updated", status=BookStatus.ready)
    )
    assert updated.title == "Updated"
    assert updated.status is BookStatus.ready
    assert len(catalog.list_books()) == 1


def test_duplicate_content_hash_is_rejected(tmp_path: Path) -> None:
    catalog = _catalog(tmp_path)
    catalog.upsert_book(_book(tmp_path))
    with pytest.raises(ValueError, match="already exists"):
        catalog.upsert_book(_book(tmp_path, book_id="book-2", content_hash="hash-1"))


def test_replace_chunks_and_stats(tmp_path: Path) -> None:
    catalog = _catalog(tmp_path)
    catalog.upsert_book(_book(tmp_path))
    assert (
        catalog.replace_chunks(
            "book-1",
            [
                _chunk("book-1", 0, "alpha beta", page=0),
                _chunk("book-1", 1, "gamma delta", page=1),
            ],
        )
        == 2
    )
    book = catalog.get_book("book-1")
    assert book is not None
    assert book.chunk_count == 2
    assert catalog.stats() == {"books": 1, "ready_books": 1, "chunks": 2}

    catalog.replace_chunks("book-1", [_chunk("book-1", 0, "alpha beta")])
    assert catalog.stats()["chunks"] == 1


def test_sections_and_outline(tmp_path: Path) -> None:
    catalog = _catalog(tmp_path)
    sections = [
        Section("book-1", "Chapter 1", 0, "Chapter 1", 0, 1),
        Section("book-1", "Chapter 1 > Intro", 1, "Intro", 0, 0),
    ]
    catalog.upsert_book(_book(tmp_path), sections)
    outline = catalog.get_outline("book-1")
    assert [section.section_path for section in outline] == [
        "Chapter 1",
        "Chapter 1 > Intro",
    ]

    catalog.upsert_book(_book(tmp_path))
    assert len(catalog.get_outline("book-1")) == 2

    catalog.upsert_book(_book(tmp_path), [])
    assert catalog.get_outline("book-1") == []


def test_lexical_search(tmp_path: Path) -> None:
    catalog = _catalog(tmp_path)
    catalog.upsert_book(_book(tmp_path))
    catalog.upsert_book(
        _book(tmp_path, book_id="book-2", content_hash="hash-2", title="Other Book")
    )
    catalog.replace_chunks(
        "book-1",
        [
            _chunk("book-1", 0, "retrieval augmented generation", page=0),
            _chunk("book-1", 1, "unrelated material", page=1),
        ],
    )
    catalog.replace_chunks(
        "book-2",
        [_chunk("book-2", 0, "retrieval evaluation", page=0)],
    )

    hits = catalog.search_lexical("retrieval augmented")
    assert len(hits) == 1
    assert hits[0].chunk_id == "book-1-chunk-0"
    assert hits[0].book_title == "Synthetic Book"
    assert hits[0].score >= 0

    filtered = catalog.search_lexical("retrieval", book_ids=["book-2"])
    assert [hit.chunk_id for hit in filtered] == ["book-2-chunk-0"]

    assert catalog.search_lexical("retrieval", book_ids=[]) == []
    assert catalog.search_lexical("!!!") == []


def test_delete_book_cascades_and_clears_fts(tmp_path: Path) -> None:
    catalog = _catalog(tmp_path)
    catalog.upsert_book(_book(tmp_path))
    catalog.replace_chunks(
        "book-1", [_chunk("book-1", 0, "unique token retrieval", page=0)]
    )
    assert catalog.delete_book("book-1") is True
    assert catalog.delete_book("book-1") is False
    assert catalog.stats()["chunks"] == 0
    assert catalog.search_lexical("unique") == []


def test_set_status(tmp_path: Path) -> None:
    catalog = _catalog(tmp_path)
    catalog.upsert_book(_book(tmp_path))
    catalog.set_status("book-1", BookStatus.ready, description="indexed")
    book = catalog.get_book("book-1")
    assert book is not None
    assert book.status is BookStatus.ready
    assert book.description == "indexed"
