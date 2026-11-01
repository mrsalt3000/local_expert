"""Определение формата и единая точка извлечения документа."""

from __future__ import annotations

import zipfile
from pathlib import Path

from book_expert.ingestion.document import (
    ExtractedDocument,
    UnsupportedFormatError,
)
from book_expert.ingestion.djvu import extract_djvu
from book_expert.ingestion.epub import extract_epub
from book_expert.ingestion.fb2 import extract_fb2
from book_expert.ingestion.pdf import extract_pdf

FORMAT_BY_SUFFIX = {
    ".pdf": "pdf",
    ".epub": "epub",
    ".fb2": "fb2",
    ".djvu": "djvu",
}
SUPPORTED_SUFFIXES = frozenset(FORMAT_BY_SUFFIX)


def format_for_name(name: str) -> str:
    suffix = Path(name).suffix.lower()
    fmt = FORMAT_BY_SUFFIX.get(suffix)
    if fmt is None:
        raise UnsupportedFormatError(f"unsupported file type: {suffix or name}")
    return fmt


def detect_format(path: str | Path, *, name: str | None = None) -> str:
    file_path = Path(path)
    fmt = format_for_name(name or file_path.name)
    if fmt == "pdf":
        with file_path.open("rb") as handle:
            valid = handle.read(5) == b"%PDF-"
    elif fmt == "epub":
        valid = zipfile.is_zipfile(file_path)
    elif fmt == "fb2":
        with file_path.open("rb") as handle:
            head = handle.read(1024).lstrip(b"\xef\xbb\xbf").lower()
        valid = b"fictionbook" in head
    else:
        with file_path.open("rb") as handle:
            valid = handle.read(8) == b"AT&TFORM"
    if not valid:
        raise UnsupportedFormatError(f"file content does not match {fmt}")
    return fmt


def extract_document(
    path: str | Path,
    *,
    password: str | None = None,
    strip_running_lines: bool = True,
) -> ExtractedDocument:
    file_path = Path(path)
    fmt = detect_format(file_path)
    if fmt == "pdf":
        return extract_pdf(
            file_path,
            password=password,
            strip_running_lines=strip_running_lines,
        )
    if fmt == "epub":
        return extract_epub(file_path)
    if fmt == "fb2":
        return extract_fb2(file_path)
    return extract_djvu(file_path)


__all__ = [
    "FORMAT_BY_SUFFIX",
    "SUPPORTED_SUFFIXES",
    "detect_format",
    "extract_document",
    "format_for_name",
]
