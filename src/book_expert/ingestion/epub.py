"""Извлечение EPUB через PyMuPDF (reflow-страницы)."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pymupdf

from book_expert.ingestion.document import ExtractedDocument, ExtractedPage
from book_expert.ingestion.pdf import PdfEncryptedError, PdfParseError


def extract_epub(path: str | Path) -> ExtractedDocument:
    epub_path = Path(path).expanduser()
    if not epub_path.is_file():
        raise FileNotFoundError(epub_path)
    content_hash = _sha256(epub_path)
    try:
        document = pymupdf.open(epub_path)
    except (pymupdf.FileDataError, pymupdf.EmptyFileError, RuntimeError) as exc:
        raise PdfParseError(f"cannot open EPUB: {epub_path.name}") from exc
    try:
        if document.needs_pass:
            raise PdfEncryptedError("EPUB is encrypted; DRM is not supported")
        toc: dict[int, str] = {}
        for _level, title, page_number in document.get_toc():
            if page_number and title.strip():
                toc.setdefault(page_number - 1, title.strip())
        pages = tuple(
            _page(page, index, toc.get(index)) for index, page in enumerate(document)
        )
        if not any(not page.is_empty for page in pages):
            raise PdfParseError("EPUB has no extractable text")
        metadata = {
            key: value.strip()
            for key, value in (document.metadata or {}).items()
            if isinstance(value, str) and value.strip()
        }
        return ExtractedDocument(
            path=epub_path.resolve(),
            content_hash=content_hash,
            page_count=document.page_count,
            pages=pages,
            title=_optional(metadata.get("title")),
            author=_optional(metadata.get("author")),
            metadata=metadata,
            format="epub",
        )
    finally:
        document.close()


def _page(page: Any, index: int, label: str | None) -> ExtractedPage:
    text = page.get_text("text", sort=True)
    rect = page.rect
    return ExtractedPage(
        pdf_page_index=index,
        text=text,
        printed_page_label=label or _heading_label(text),
        char_count=len(text),
        block_count=len(page.get_text("blocks")),
        width=float(rect.width),
        height=float(rect.height),
    )


def _heading_label(text: str) -> str | None:
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if (
            len(line) <= 80
            and len(line.split()) <= 8
            and not line.endswith((".", ",", ";", ":"))
        ):
            return line
        return None
    return None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _optional(value: str | None) -> str | None:
    return value.strip() if value and value.strip() else None


__all__ = ["extract_epub"]
