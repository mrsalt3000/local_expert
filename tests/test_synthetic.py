"""Тесты генератора synthetic-корпуса."""

from __future__ import annotations

from pathlib import Path

from book_expert.catalog import Catalog
from book_expert.eval.synthetic import BOOK_ID, seed_synthetic_corpus


def test_seed_synthetic_corpus(tmp_path: Path) -> None:
    catalog = Catalog(tmp_path / "catalog.sqlite3")
    catalog.initialize()

    corpus = seed_synthetic_corpus(catalog)

    assert corpus.book.id == BOOK_ID
    assert len(corpus.chunks) == 2
    assert len(corpus.cases) == 3
    assert catalog.stats()["chunks"] == 2
    assert catalog.get_book(BOOK_ID) is not None
    assert catalog.search_lexical("reciprocal rank fusion")
    assert all(case.expected_book_ids == (BOOK_ID,) for case in corpus.cases)
