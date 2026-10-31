"""Генератор synthetic-корпуса для eval без реального текста книги."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from book_expert.catalog import Catalog
from book_expert.eval.metrics import EvalCase
from book_expert.models import Book, BookStatus, Chunk

BOOK_ID = "synthetic-book"
TITLE = "Synthetic Retrieval Handbook"


@dataclass(frozen=True, slots=True)
class SyntheticCorpus:
    book: Book
    chunks: tuple[Chunk, ...]
    cases: tuple[EvalCase, ...]


def seed_synthetic_corpus(catalog: Catalog) -> SyntheticCorpus:
    """Регистрирует вымышленную книгу и eval-вопросы в каталоге."""
    chunks = (
        Chunk(
            id="synthetic-c1",
            book_id=BOOK_ID,
            text=(
                "Hybrid retrieval combines dense vectors with lexical search. "
                "Reciprocal rank fusion merges both rankings."
            ),
            pdf_page_index=0,
            ordinal=0,
            printed_page_label="1",
            section_path="Chapter 1",
        ),
        Chunk(
            id="synthetic-c2",
            book_id=BOOK_ID,
            text=(
                "Evaluation measures recall at k, mean reciprocal rank "
                "and citation accuracy."
            ),
            pdf_page_index=1,
            ordinal=1,
            printed_page_label="2",
            section_path="Chapter 2",
        ),
    )
    book = Book(
        id=BOOK_ID,
        title=TITLE,
        author="Synthetic Author",
        path=Path("synthetic.pdf"),
        content_hash="synthetic-hash",
        status=BookStatus.ready,
        page_count=2,
        chunk_count=len(chunks),
        description="Synthetic corpus for retrieval and citation evaluation.",
        language="en",
    )
    catalog.upsert_book(book)
    catalog.replace_chunks(BOOK_ID, chunks)
    cases = (
        EvalCase(
            "Hybrid retrieval combines dense lexical search.",
            ("synthetic-c1",),
            (BOOK_ID,),
        ),
        EvalCase(
            "Evaluation measures recall and citation accuracy.",
            ("synthetic-c2",),
            (BOOK_ID,),
        ),
        EvalCase(
            "Reciprocal rank fusion merges rankings.",
            ("synthetic-c1",),
            (BOOK_ID,),
        ),
    )
    return SyntheticCorpus(book=book, chunks=chunks, cases=cases)


__all__ = ["SyntheticCorpus", "seed_synthetic_corpus"]
