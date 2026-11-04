"""Извлечение описания книги из метаданных её файла.

Описание используется только в routing-профилях и метаданных каталога:
оно улучшает выбор книги, но не индексируется как текст.
"""

from __future__ import annotations

import html
import re
import zipfile
from pathlib import Path
from typing import Any

import pymupdf

MIN_DESCRIPTION = 40
MAX_DESCRIPTION = 2000
_FILLER = re.compile(
    r"книги, библиотека, диссертации, дипломные, курсовые, рефераты",
    re.IGNORECASE,
)
_YEAR_PREFIX = re.compile(r"^\s*((?:1[5-9]|20)\d{2})(?!\d)")
_YEAR_VALUE = re.compile(r"(?<!\d)((?:1[5-9]|20)\d{2})(?!\d)")
_METADATA_YEAR_KEYS = (
    "published_year",
    "publication_year",
    "year",
    "date",
    "creationdate",
    "moddate",
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


def extract_published_year(
    path_name: str,
    title: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> tuple[int | None, str | None]:
    """Возвращает год выпуска и источник: ``filename``, ``title`` или ``metadata``.

    Для курированной библиотеки год надёжно закодирован в имени файла
    (``2014 - Автор - Название.pdf``). Метаданные — запасной вариант:
    ``creationDate`` у сканов часто означает дату файла, поэтому UI показывает
    такой год как приблизительный.
    """
    for source, value in (("filename", path_name), ("title", title)):
        match = _YEAR_PREFIX.match(value or "")
        if match:
            year = _valid_year(match.group(1))
            if year is not None:
                return year, source

    values = _metadata_values(metadata)
    for key in _METADATA_YEAR_KEYS:
        for value in values.get(key, ()):
            year = _year_in_text(value)
            if year is not None:
                return year, "metadata"
    return None, None


def _metadata_values(metadata: dict[str, Any] | None) -> dict[str, tuple[str, ...]]:
    collected: dict[str, list[str]] = {}
    stack: list[Any] = [metadata]
    while stack:
        current = stack.pop()
        if isinstance(current, dict):
            for key, value in current.items():
                if isinstance(value, (str, int)) and not isinstance(value, bool):
                    collected.setdefault(str(key).casefold(), []).append(str(value))
                else:
                    stack.append(value)
        elif isinstance(current, list):
            stack.extend(current)
    return {key: tuple(values) for key, values in collected.items()}


def _year_in_text(value: str) -> int | None:
    # PDF-даты выглядят как D:20200101120000, без разделителя после года.
    compact = re.match(r"^D:((?:1[5-9]|20)\d{2})", value)
    match = compact or _YEAR_VALUE.search(value)
    return _valid_year(match.group(1)) if match else None


def _valid_year(raw: str) -> int | None:
    year = int(raw)
    return year if 1500 <= year <= 2100 else None


__all__ = [
    "MAX_DESCRIPTION",
    "MIN_DESCRIPTION",
    "extract_description",
    "extract_published_year",
]
