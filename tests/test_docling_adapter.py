"""Тесты маппинга Docling-документа в ExtractedDocument."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from book_expert.ingestion.docling import _document_to_extracted


def test_document_to_extracted_maps_items_by_page(tmp_path: Path) -> None:
    item_one = SimpleNamespace(
        text="First page text.",
        prov=[SimpleNamespace(page_no=1)],
    )
    item_two = SimpleNamespace(
        text="Second page text.",
        prov=[SimpleNamespace(page_no=2)],
    )

    class PictureItem:
        text = ""
        prov = [SimpleNamespace(page_no=1)]

    picture = PictureItem()
    no_prov = SimpleNamespace(text="Ignored text.")
    document = SimpleNamespace(
        pages={
            1: SimpleNamespace(size=SimpleNamespace(width=100.0, height=200.0)),
            2: SimpleNamespace(size=SimpleNamespace(width=100.0, height=200.0)),
        },
        iterate_items=lambda: [
            (item_one, 0),
            (item_two, 0),
            (picture, 0),
            (no_prov, 0),
        ],
        num_pages=lambda: 2,
    )

    result = _document_to_extracted(
        document,
        tmp_path / "book.pdf",
        "hash",
        page_count=416,
    )

    assert result.page_count == 416
    assert [page.pdf_page_index for page in result.pages] == [0, 1]
    assert result.pages[0].text == "First page text."
    assert result.pages[1].text == "Second page text."
    assert result.pages[0].block_count == 1
    assert result.pages[0].width == 100.0
