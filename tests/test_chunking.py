"""Тесты chunking: страницы, overlap, секции и детерминизм."""

from __future__ import annotations

from pathlib import Path

import pytest

from book_expert.ingestion.chunking import chunk_document
from book_expert.ingestion.pdf import PdfDocument, PdfPage


def _page(index: int, text: str, label: str | None = None) -> PdfPage:
    return PdfPage(
        pdf_page_index=index,
        text=text,
        printed_page_label=label,
        char_count=len(text),
        block_count=1 if text.strip() else 0,
        width=595.0,
        height=842.0,
    )


def _document(*pages: PdfPage) -> PdfDocument:
    return PdfDocument(
        path=Path("synthetic.pdf"),
        content_hash="hash",
        page_count=len(pages),
        pages=pages,
    )


def test_short_page_is_single_chunk() -> None:
    result = chunk_document(
        _document(_page(0, "Hello synthetic world.", "1")),
        "book-1",
        chunk_size=100,
        chunk_overlap=10,
    )
    assert len(result.chunks) == 1
    chunk = result.chunks[0]
    assert chunk.text == "Hello synthetic world."
    assert chunk.pdf_page_index == 0
    assert chunk.printed_page_label == "1"
    assert chunk.ordinal == 0
    assert chunk.token_count == 3


def test_empty_page_produces_no_chunks() -> None:
    result = chunk_document(
        _document(_page(0, "   \n\n")),
        "book-1",
        chunk_size=100,
        chunk_overlap=10,
    )
    assert result.chunks == ()
    assert result.sections == ()


def test_long_page_splits_with_overlap() -> None:
    text = " ".join(f"Sentence number {index}." for index in range(40))
    result = chunk_document(
        _document(_page(0, text, "5")),
        "book-1",
        chunk_size=90,
        chunk_overlap=25,
    )
    assert len(result.chunks) > 3
    assert all(len(chunk.text) <= 90 for chunk in result.chunks)
    assert [chunk.ordinal for chunk in result.chunks] == list(range(len(result.chunks)))
    assert all(chunk.pdf_page_index == 0 for chunk in result.chunks)
    assert all(chunk.printed_page_label == "5" for chunk in result.chunks)
    last_word = result.chunks[0].text.split()[-1]
    assert last_word in result.chunks[1].text


def test_chunking_is_deterministic() -> None:
    document = _document(_page(0, "Alpha beta gamma. Delta epsilon."))
    first = chunk_document(document, "book-1", chunk_size=20, chunk_overlap=5)
    second = chunk_document(document, "book-1", chunk_size=20, chunk_overlap=5)
    assert [chunk.id for chunk in first.chunks] == [chunk.id for chunk in second.chunks]


def test_sections_breadcrumb_and_chunk_assignment() -> None:
    document = _document(
        _page(
            0,
            "Chapter 1\nIntro text about things.\n"
            "1.1 Getting Started\nBody of section one.",
            "1",
        ),
        _page(1, "1.2 Next Step\nMore body text.", "2"),
    )
    result = chunk_document(document, "book-1", chunk_size=200, chunk_overlap=20)

    paths = [section.section_path for section in result.sections]
    assert paths == [
        "Chapter 1",
        "Chapter 1 > 1.1 Getting Started",
        "Chapter 1 > 1.2 Next Step",
    ]
    assert result.sections[0].page_end == 0
    assert result.sections[1].page_end == 1
    assert result.sections[2].page_end == 1

    chunk_sections = [chunk.section_path for chunk in result.chunks]
    assert chunk_sections[0] == "Chapter 1"
    assert chunk_sections[1] == "Chapter 1 > 1.1 Getting Started"
    assert chunk_sections[2] == "Chapter 1 > 1.2 Next Step"


def test_year_like_line_is_not_a_section() -> None:
    document = _document(_page(0, "2024 was a year\nBody text follows."))
    result = chunk_document(document, "book-1", chunk_size=200, chunk_overlap=20)
    assert result.sections == ()
    assert result.chunks[0].section_path is None


def test_invalid_overlap_rejected() -> None:
    document = _document(_page(0, "text"))
    with pytest.raises(ValueError, match="chunk_overlap"):
        chunk_document(document, "book-1", chunk_size=100, chunk_overlap=100)
