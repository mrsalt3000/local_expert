"""Опциональный Docling-адаптер к формату ExtractedDocument."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from book_expert.ingestion.document import (
    DocumentError,
    ExtractedDocument,
    ExtractedPage,
)
from book_expert.ingestion.pdf import detect_printed_label


class DoclingError(DocumentError):
    """Docling недоступен или не смог разобрать документ."""


def docling_available() -> bool:
    try:
        import docling  # noqa: F401
    except ImportError:
        return False
    return True


def extract_docling(
    path: str | Path,
    *,
    page_range: tuple[int, int] | None = None,
    total_pages: int | None = None,
) -> ExtractedDocument:
    """Извлекает PDF через Docling; page_range — 1-based inclusive."""
    try:
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        from docling.document_converter import DocumentConverter, PdfFormatOption
    except ImportError as exc:
        raise DoclingError(
            "docling is not installed; run `uv sync --extra docling`"
        ) from exc

    pdf_path = Path(path).expanduser()
    if not pdf_path.is_file():
        raise FileNotFoundError(pdf_path)
    if page_range is not None:
        start, end = page_range
        if start <= 0 or end < start:
            raise ValueError("page_range must be positive and ordered")

    options = PdfPipelineOptions()
    options.do_ocr = False
    options.do_table_structure = False
    options.generate_page_images = False
    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=options),
        }
    )
    try:
        result = converter.convert(
            pdf_path,
            page_range=page_range or (1, 2**31 - 1),
        )
    except Exception as exc:
        raise DoclingError(f"docling failed: {pdf_path.name}") from exc
    if result.status.value not in {"success", "partial_success"}:
        raise DoclingError(f"docling status: {result.status.value}")
    return _document_to_extracted(
        result.document,
        pdf_path.resolve(),
        _sha256(pdf_path),
        page_count=total_pages,
    )


def _document_to_extracted(
    document: Any,
    path: Path,
    content_hash: str,
    *,
    page_count: int | None = None,
) -> ExtractedDocument:
    by_page: dict[int, list[str]] = {}
    for item in document.iterate_items():
        obj = item[0] if isinstance(item, tuple) else item
        if type(obj).__name__ == "PictureItem":
            continue
        prov = getattr(obj, "prov", None)
        if not prov:
            continue
        page_no = getattr(prov[0], "page_no", None)
        if not isinstance(page_no, int):
            continue
        text = getattr(obj, "text", None)
        if not text and hasattr(obj, "export_to_markdown"):
            try:
                text = obj.export_to_markdown(document)
            except Exception:
                text = None
        if isinstance(text, str) and text.strip():
            by_page.setdefault(page_no, []).append(text.strip())

    pages: list[ExtractedPage] = []
    for page_no in sorted(document.pages):
        page = document.pages[page_no]
        text = "\n\n".join(by_page.get(page_no, []))
        size = getattr(page, "size", None)
        pages.append(
            ExtractedPage(
                pdf_page_index=page_no - 1,
                text=text,
                printed_page_label=detect_printed_label(text),
                char_count=len(text),
                block_count=len(by_page.get(page_no, [])),
                width=float(getattr(size, "width", 0.0) or 0.0),
                height=float(getattr(size, "height", 0.0) or 0.0),
            )
        )
    return ExtractedDocument(
        path=path,
        content_hash=content_hash,
        page_count=page_count or max(document.pages, default=0),
        pages=tuple(pages),
        title=None,
        author=None,
        metadata={"parser": "docling"},
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


__all__ = [
    "DoclingError",
    "docling_available",
    "extract_docling",
]
