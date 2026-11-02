"""Мягкий metadata prefilter для routing и поиска."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from book_expert.models import Book

_YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")
_EXPLICIT_RE = re.compile(
    r"\b(year|год|author|автор|format|формат|language|язык)\s*[:=]\s*"
    r"([^\s,;]+)",
    re.IGNORECASE,
)
_FORMATS = {"pdf", "epub", "fb2", "djvu"}
_LANGUAGES = {
    "ru": "ru",
    "рус": "ru",
    "русский": "ru",
    "en": "en",
    "англ": "en",
    "english": "en",
}


@dataclass(frozen=True, slots=True)
class MetadataFilter:
    year: int | None = None
    format: str | None = None
    author: str | None = None
    language: str | None = None

    @property
    def active(self) -> bool:
        return any((self.year, self.format, self.author, self.language))


def parse_metadata_filter(query: str) -> MetadataFilter:
    values: dict[str, str] = {}
    for key, value in _EXPLICIT_RE.findall(query):
        values[key.casefold()] = value.strip()
    year_text = values.get("year") or values.get("год")
    year = int(year_text) if year_text and year_text.isdigit() else None
    format_name = values.get("format") or values.get("формат")
    if format_name:
        format_name = format_name.casefold().lstrip(".")
    language = values.get("language") or values.get("язык")
    language = _LANGUAGES.get(language.casefold()) if language else None
    author = values.get("author") or values.get("автор")
    return MetadataFilter(
        year=year,
        format=format_name if format_name in _FORMATS else None,
        author=author,
        language=language,
    )


def book_year(book: Book) -> int | None:
    candidates: list[object] = []
    source = book.metadata.get("source") if isinstance(book.metadata, dict) else None
    if isinstance(source, dict):
        for key in ("year", "date", "creationdate", "published", "publication_date"):
            for source_key, value in source.items():
                if source_key.casefold() == key:
                    candidates.append(value)
    candidates.extend((book.title, Path(book.path).stem))
    for value in candidates:
        match = _YEAR_RE.search(str(value))
        if match:
            return int(match.group(0))
    return None


def matches(book: Book, filters: MetadataFilter) -> bool:
    if filters.year is not None and book_year(book) != filters.year:
        return False
    if filters.format is not None and book.format.casefold() != filters.format:
        return False
    if filters.language is not None and (
        not book.language or book.language.casefold() != filters.language
    ):
        return False
    if filters.author is not None and (
        not book.author or filters.author.casefold() not in book.author.casefold()
    ):
        return False
    return True


def apply_metadata_filter(books: list[Book], query: str) -> list[Book]:
    filters = parse_metadata_filter(query)
    if not filters.active:
        return books
    filtered = [book for book in books if matches(book, filters)]
    return filtered or books


__all__ = [
    "MetadataFilter",
    "apply_metadata_filter",
    "book_year",
    "matches",
    "parse_metadata_filter",
]
