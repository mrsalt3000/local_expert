"""Тесты parser quality gate."""

from __future__ import annotations

from pathlib import Path

from book_expert.ingestion.document import ExtractedDocument, ExtractedPage
from book_expert.ingestion.quality import QualityThresholds, assess_document


def _doc(pages: list[tuple[str, int]]) -> ExtractedDocument:
    return ExtractedDocument(
        path=Path("book.pdf"),
        content_hash="hash",
        page_count=len(pages),
        pages=tuple(
            ExtractedPage(
                pdf_page_index=index,
                text=text,
                char_count=chars,
            )
            for index, (text, chars) in enumerate(pages)
        ),
    )


def test_quality_ok() -> None:
    report = assess_document(_doc([("первая страница", 100), ("вторая страница", 120)]))

    assert report.status == "ok"
    assert report.coverage == 1.0
    assert report.accepted is True


def test_quality_rejects_low_coverage() -> None:
    report = assess_document(_doc([("текст", 100), ("", 0)]))

    assert report.status == "reject"
    assert report.accepted is False
    assert report.coverage == 0.5
    assert report.issues[0].reason == "empty page"


def test_quality_warns_on_low_text_page() -> None:
    report = assess_document(
        _doc([("полный текст", 100), ("мало", 10)]),
        QualityThresholds(min_coverage=0.5, min_chars_per_page=50),
    )

    assert report.status == "warn"
    assert report.issues[0].reason == "low text coverage"


def test_quality_rejects_empty_document() -> None:
    report = assess_document(_doc([("", 0)]))

    assert report.status == "reject"
    assert "no text layer" in report.reasons
