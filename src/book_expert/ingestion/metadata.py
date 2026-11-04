"""Извлечение описания книги из метаданных её файла.

Описание используется только в routing-профилях и метаданных каталога:
оно улучшает выбор книги, но не индексируется как текст.
"""

from __future__ import annotations

import html
import re
import zipfile
from pathlib import Path

import pymupdf

MIN_DESCRIPTION = 40
MAX_DESCRIPTION = 2000
_FILLER = re.compile(
    r"книги, библиотека, диссертации, дипломные, курсовые, рефераты",
    re.IGNORECASE,
)


def extract_description(path: str | Path, fmt: str) -> str | None:
    """Возвращает очищенное описание или ``None``, если его нет."""
    book_path = Path(path)
    if not book_path.is_file():
        return None
    try:
        if fmt == "epub":
            raw = _epub_description(book_path)
        elif fmt == "pdf":
            raw = _pdf_description(book_path)
        elif fmt == "fb2":
            raw = _fb2_description(book_path)
        else:
            return None
    except (OSError, RuntimeError, ValueError, zipfile.BadZipFile):
        return None
    return _clean(raw)


def _epub_description(path: Path) -> str | None:
    with zipfile.ZipFile(path) as archive:
        opf = next(
            (name for name in archive.namelist() if name.lower().endswith(".opf")),
            None,
        )
        if opf is None:
            return None
        xml = archive.read(opf).decode("utf-8", "ignore")
    match = re.search(r"<dc:description[^>]*>(.*?)</dc:description>", xml, re.S | re.I)
    return match.group(1) if match else None


def _pdf_description(path: Path) -> str | None:
    with pymupdf.open(path) as document:
        metadata = document.metadata or {}
    parts = [metadata.get("subject"), metadata.get("keywords")]
    return " ".join(part for part in parts if isinstance(part, str)) or None


def _fb2_description(path: Path) -> str | None:
    xml = path.read_text("utf-8", "ignore")
    match = re.search(r"<annotation>(.*?)</annotation>", xml, re.S | re.I)
    return match.group(1) if match else None


def _clean(raw: str | None) -> str | None:
    if not raw:
        return None
    text = html.unescape(raw)
    text = re.sub(r"<[^>]+>", " ", text)
    text = " ".join(text.split())
    if len(text) < MIN_DESCRIPTION or _FILLER.search(text):
        return None
    return text[:MAX_DESCRIPTION]


__all__ = ["MAX_DESCRIPTION", "MIN_DESCRIPTION", "extract_description"]
