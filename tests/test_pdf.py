"""Тесты извлечения PDF на синтетических документах (без реальной книги)."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pymupdf
import pytest

from book_expert.ingestion.pdf import (
    PdfEncryptedError,
    PdfPage,
    PdfParseError,
    extract_pdf,
    strip_running_lines_from_pages,
)


def _make_pdf(
    path: Path,
    pages: list[tuple[list[str], str | None]],
    *,
    metadata: dict[str, str] | None = None,
    password: str | None = None,
) -> Path:
    document = pymupdf.open()
    if metadata:
        document.set_metadata(metadata)
    for lines, bottom_label in pages:
        page = document.new_page()
        y = 72.0
        for line in lines:
            page.insert_text((72, y), line, fontsize=12)
            y += 24
        if bottom_label is not None:
            page.insert_text((72, page.rect.height - 40), bottom_label, fontsize=10)
    if password:
        document.save(
            path,
            encryption=pymupdf.PDF_ENCRYPT_AES_256,
            owner_pw="owner",
            user_pw=password,
        )
    else:
        document.save(path)
    document.close()
    return path


def test_extract_pages_and_labels(tmp_path: Path) -> None:
    pdf = _make_pdf(
        tmp_path / "synthetic.pdf",
        [
            (["Chapter One", "Body text about retrieval."], "7"),
            ([], None),
            (["Appendix"], "xiv"),
        ],
        metadata={"title": "Synthetic Title", "author": "Synthetic Author"},
    )

    document = extract_pdf(pdf)

    assert document.page_count == 3
    assert len(document.pages) == 3
    assert "Body text about retrieval." in document.pages[0].text
    assert document.pages[0].printed_page_label == "7"
    assert document.pages[0].pdf_page_index == 0
    assert document.pages[1].is_empty
    assert document.pages[1].printed_page_label is None
    assert document.pages[2].printed_page_label == "xiv"
    assert document.empty_page_indices == (1,)
    assert document.has_text_layer is True
    assert document.title == "Synthetic Title"
    assert document.author == "Synthetic Author"
    assert document.path == pdf.resolve()


def test_extract_selected_pages_keeps_absolute_indices(tmp_path: Path) -> None:
    pdf = _make_pdf(
        tmp_path / "selected.pdf",
        [
            (["First page"], None),
            (["Second page"], None),
            (["Third page"], None),
        ],
    )

    document = extract_pdf(pdf, pages=[1, 2])

    assert document.page_count == 3
    assert [page.pdf_page_index for page in document.pages] == [1, 2]
    assert "Second page" in document.pages[0].text


def test_content_hash_matches_file_bytes(tmp_path: Path) -> None:
    pdf = _make_pdf(tmp_path / "synthetic.pdf", [(["alpha"], "1")])
    expected = hashlib.sha256(pdf.read_bytes()).hexdigest()
    assert extract_pdf(pdf).content_hash == expected


def test_year_like_bottom_is_not_treated_as_page_label(tmp_path: Path) -> None:
    pdf = _make_pdf(tmp_path / "synthetic.pdf", [(["Body"], "2024")])
    assert extract_pdf(pdf).pages[0].printed_page_label is None


def test_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        extract_pdf(tmp_path / "missing.pdf")


def test_invalid_pdf(tmp_path: Path) -> None:
    broken = tmp_path / "broken.pdf"
    broken.write_bytes(b"not a pdf at all")
    with pytest.raises(PdfParseError, match="cannot open PDF"):
        extract_pdf(broken)


def test_encrypted_pdf_requires_password(tmp_path: Path) -> None:
    pdf = _make_pdf(
        tmp_path / "encrypted.pdf",
        [(["Top secret synthetic text"], "1")],
        password="secret",
    )
    with pytest.raises(PdfEncryptedError, match="password required"):
        extract_pdf(pdf)
    with pytest.raises(PdfEncryptedError, match="invalid PDF password"):
        extract_pdf(pdf, password="wrong")

    document = extract_pdf(pdf, password="secret")
    assert document.page_count == 1
    assert document.pages[0].printed_page_label == "1"


def test_blank_document_has_no_text_layer(tmp_path: Path) -> None:
    pdf = _make_pdf(tmp_path / "blank.pdf", [([], None)])
    document = extract_pdf(pdf)
    assert document.has_text_layer is False
    assert document.empty_page_indices == (0,)


def _page(index: int, lines: list[str]) -> PdfPage:
    text = "\n".join(lines)
    return PdfPage(
        pdf_page_index=index,
        text=text,
        printed_page_label=str(index + 1),
        char_count=len(text),
        block_count=len(lines),
        width=595.0,
        height=842.0,
    )


def test_running_headers_with_page_numbers_are_removed() -> None:
    pages = [
        _page(
            index,
            [
                f"{12 + index} Глава 1. Синтетика",
                f"Тело страницы {index}.",
                f"Сноска {index} уникальна.",
            ],
        )
        for index in range(6)
    ]
    cleaned = strip_running_lines_from_pages(pages)
    for page in cleaned:
        assert "Глава 1. Синтетика" not in page.text
        assert "Тело страницы" in page.text
        assert "уникальна" in page.text
    assert cleaned[0].printed_page_label == "1"


def test_unique_margin_line_is_kept() -> None:
    pages = [_page(0, ["Заголовок главы", "Текст."])]
    pages += [
        _page(index, [f"{20 + index} Глава 1. Синтетика", "Текст."])
        for index in range(1, 6)
    ]
    cleaned = strip_running_lines_from_pages(pages)
    assert "Заголовок главы" in cleaned[0].text


def test_bare_page_numbers_are_removed() -> None:
    pages = [_page(index, ["Текст.", str(index + 1)]) for index in range(5)]
    cleaned = strip_running_lines_from_pages(pages)
    for page in cleaned:
        assert page.text.strip() == "Текст."


def test_short_documents_are_left_untouched() -> None:
    pages = [
        _page(index, [f"{5 + index} Глава 1. Синтетика", "Текст."])
        for index in range(3)
    ]
    cleaned = strip_running_lines_from_pages(pages)
    assert cleaned == tuple(pages)


def test_structural_headings_are_never_removed() -> None:
    pages = [_page(index, ["Часть 1", f"Текст {index}."]) for index in range(6)]
    cleaned = strip_running_lines_from_pages(pages)
    assert all("Часть 1" in page.text for page in cleaned)
