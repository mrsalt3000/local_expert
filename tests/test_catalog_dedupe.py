"""Тесты плана дедупликации каталога."""

from __future__ import annotations

from pathlib import Path

from book_expert.catalog_dedupe import build_dedupe_plan
from book_expert.models import Book, BookStatus


def _book(
    book_id: str,
    title: str,
    path: Path,
    *,
    chunks: int = 1,
    pages: int = 1,
) -> Book:
    return Book(
        id=book_id,
        title=title,
        path=path,
        content_hash=f"hash-{book_id}",
        page_count=pages,
        chunk_count=chunks,
        status=BookStatus.ready,
    )


def test_dedupe_plan_removes_only_losers(tmp_path: Path) -> None:
    books = [
        _book("keep", "Alpha", tmp_path / "a.pdf", chunks=20, pages=120),
        _book("drop", "2020 - Alpha", tmp_path / "b.epub", chunks=10, pages=100),
        _book("other", "Beta", tmp_path / "c.pdf"),
    ]

    plan = build_dedupe_plan(books)

    assert len(plan.groups) == 1
    assert [removal.book.id for removal in plan.removals] == ["drop"]
    assert plan.removals[0].keep_id == "keep"


def test_dedupe_plan_is_empty_without_duplicates(tmp_path: Path) -> None:
    plan = build_dedupe_plan([_book("one", "Alpha", tmp_path / "a.pdf")])

    assert plan.groups == ()
    assert plan.removals == ()
