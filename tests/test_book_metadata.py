"""Извлечение описаний книг из метаданных файла (T84)."""

from __future__ import annotations

import zipfile
from pathlib import Path

import pymupdf

from book_expert.ingestion.metadata import extract_description


def test_epub_description_is_extracted_and_cleaned(tmp_path: Path) -> None:
    path = tmp_path / "book.epub"
    opf = """<?xml version="1.0"?>
<package xmlns:dc="http://purl.org/dc/elements/1.1/">
  <metadata>
    <dc:title>Книга</dc:title>
    <dc:description>&lt;p&gt;Краткое &lt;b&gt;описание&lt;/b&gt; книги про
    управление проектами и командами.&lt;/p&gt;</dc:description>
  </metadata>
</package>"""
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("META-INF/container.xml", "<container/>")
        archive.writestr("OEBPS/content.opf", opf)

    description = extract_description(path, "epub")

    assert description is not None
    assert "<p>" not in description
    assert "Краткое описание книги" in description


def test_pdf_description_uses_subject_and_keywords(tmp_path: Path) -> None:
    path = tmp_path / "book.pdf"
    document = pymupdf.open()
    document.new_page()
    document.set_metadata(
        {"subject": "Практическое руководство по переговорам", "keywords": "бизнес"}
    )
    document.save(path)
    document.close()

    description = extract_description(path, "pdf")

    assert description is not None
    assert "переговорам" in description
    assert "бизнес" in description


def test_fb2_annotation_is_extracted(tmp_path: Path) -> None:
    path = tmp_path / "book.fb2"
    path.write_text(
        """<?xml version="1.0"?>
<FictionBook><description><title-info>
<annotation><p>История одной компании и её основателей.</p></annotation>
</title-info></description></FictionBook>""",
        encoding="utf-8",
    )

    assert (
        extract_description(path, "fb2") == "История одной компании и её основателей."
    )


def test_short_or_filler_description_is_ignored(tmp_path: Path) -> None:
    short = tmp_path / "short.fb2"
    short.write_text(
        "<FictionBook><description><annotation>Статистика</annotation>"
        "</description></FictionBook>",
        encoding="utf-8",
    )
    filler = tmp_path / "filler.fb2"
    filler.write_text(
        "<FictionBook><description><annotation>Лоу Дженет. Книга, книги, "
        "библиотека, диссертации, дипломные, курсовые, рефераты</annotation>"
        "</description></FictionBook>",
        encoding="utf-8",
    )

    assert extract_description(short, "fb2") is None
    assert extract_description(filler, "fb2") is None


def test_missing_file_and_unknown_format_return_none(tmp_path: Path) -> None:
    assert extract_description(tmp_path / "absent.epub", "epub") is None
    assert extract_description(__file__, "txt") is None
