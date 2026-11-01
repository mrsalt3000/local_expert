"""Структурный chunking с привязкой к странице и секции."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, replace

from book_expert.ingestion.pdf import PdfDocument
from book_expert.models import Chunk, Section

_CHAPTER_RE = re.compile(
    r"^(?:chapter|part|section|глава|часть|раздел)\s+[\wivxlcdm\d]+",
    re.IGNORECASE,
)
_NUMBERED_RE = re.compile(r"^(?P<num>\d+(?:\.\d+)*)\.?\s+(?P<title>\S.*)$")
_SENTENCE_RE = re.compile(r"(?<=[.!?…])\s+")


@dataclass(frozen=True, slots=True)
class ChunkingResult:
    chunks: tuple[Chunk, ...]
    sections: tuple[Section, ...]


@dataclass(frozen=True, slots=True)
class _Unit:
    text: str
    separator: str


def chunk_document(
    document: PdfDocument,
    book_id: str,
    *,
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
) -> ChunkingResult:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be in [0, chunk_size)")

    chunks: list[Chunk] = []
    sections: list[Section] = []
    stack: list[str] = []
    current_section: str | None = None
    ordinal = 0

    document_format = getattr(document, "format", "pdf")
    for page in document.pages:
        fallback_section = (
            page.printed_page_label if document_format in {"fb2", "epub"} else None
        )
        if document_format == "fb2":
            current_section = fallback_section
        page_had_heading = False
        for heading, segment in _page_segments(page.text):
            if heading is not None:
                page_had_heading = True
                level, title = heading
                stack = stack[: min(level, len(stack) + 1) - 1] + [title]
                current_section = " > ".join(stack)
                sections.append(
                    Section(
                        book_id=book_id,
                        section_path=current_section or fallback_section,
                        ordinal=len(sections),
                        title=title,
                        page_start=page.pdf_page_index,
                    )
                )

            for text in _split_page(segment, chunk_size, chunk_overlap):
                chunks.append(
                    Chunk(
                        id=_chunk_id(book_id, page.pdf_page_index, ordinal, text),
                        book_id=book_id,
                        text=text,
                        pdf_page_index=page.pdf_page_index,
                        printed_page_label=page.printed_page_label,
                        section_path=fallback_section or current_section,
                        ordinal=ordinal,
                        token_count=len(text.split()),
                    )
                )
                ordinal += 1
        if fallback_section and not page_had_heading:
            sections.append(
                Section(
                    book_id=book_id,
                    section_path=fallback_section,
                    ordinal=len(sections),
                    title=fallback_section.split(" > ")[-1],
                    page_start=page.pdf_page_index,
                )
            )

    return ChunkingResult(
        chunks=tuple(chunks),
        sections=tuple(_close_sections(sections, document.page_count)),
    )


def _page_segments(text: str) -> list[tuple[tuple[int, str] | None, str]]:
    segments: list[tuple[tuple[int, str] | None, str]] = []
    heading: tuple[int, str] | None = None
    buffer: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        found = _heading(stripped)
        if found is not None:
            if any(item.strip() for item in buffer):
                segments.append((heading, "\n".join(buffer)))
            heading = found
            buffer = [line]
        else:
            buffer.append(line)
    if any(item.strip() for item in buffer):
        segments.append((heading, "\n".join(buffer)))
    return segments


def _heading(line: str) -> tuple[int, str] | None:
    if not line or len(line) > 100:
        return None
    if _CHAPTER_RE.match(line):
        return 1, line
    numbered = _NUMBERED_RE.match(line)
    if numbered and not line.endswith(".") and numbered.group("title")[:1].isupper():
        title = f"{numbered.group('num')} {numbered.group('title')}".strip()
        return numbered.group("num").count(".") + 1, title
    if (
        line.isupper()
        and 3 <= len(line) <= 60
        and len(line.split()) <= 10
        and not line.endswith((".", ",", ":", ";"))
    ):
        return 1, line
    return None


def _split_page(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    units = _units(text, chunk_size)
    if not units:
        return []
    return [_join(window) for window in _windows(units, chunk_size, chunk_overlap)]


def _units(text: str, chunk_size: int) -> list[_Unit]:
    paragraphs = re.split(r"\n\s*\n", text.strip())
    units: list[_Unit] = []
    for paragraph in paragraphs:
        normalized = " ".join(paragraph.split())
        if not normalized:
            continue
        sentences = [part for part in _SENTENCE_RE.split(normalized) if part]
        for index, sentence in enumerate(sentences):
            separator = "" if not units else ("\n\n" if index == 0 else " ")
            pieces = (
                [sentence]
                if len(sentence) <= chunk_size
                else [
                    sentence[start : start + chunk_size]
                    for start in range(0, len(sentence), chunk_size)
                ]
            )
            for piece_index, piece in enumerate(pieces):
                piece_separator = separator if piece_index == 0 else " "
                units.append(_Unit(piece, piece_separator))
    return units


def _windows(
    units: list[_Unit], chunk_size: int, chunk_overlap: int
) -> list[list[_Unit]]:
    windows: list[list[_Unit]] = []
    start = 0
    while start < len(units):
        end = start
        size = 0
        while end < len(units):
            addition = len(units[end].text) + (
                len(units[end].separator) if end > start else 0
            )
            if end > start and size + addition > chunk_size:
                break
            size += addition
            end += 1
        windows.append(units[start:end])
        if end >= len(units):
            break
        next_start = end
        overlap = 0
        while next_start > start:
            candidate = units[next_start - 1]
            addition = len(candidate.text) + len(candidate.separator)
            if overlap + addition > chunk_overlap:
                break
            next_start -= 1
            overlap += addition
        start = next_start if next_start > start else end
    return windows


def _join(units: list[_Unit]) -> str:
    parts = [
        unit.text if index == 0 else unit.separator + unit.text
        for index, unit in enumerate(units)
    ]
    return "".join(parts).strip()


def _chunk_id(book_id: str, page_index: int, ordinal: int, text: str) -> str:
    payload = f"{book_id}\0{page_index}\0{ordinal}\0{text}".encode()
    return "chk_" + hashlib.sha256(payload).hexdigest()[:24]


def _close_sections(sections: list[Section], page_count: int) -> list[Section]:
    last_page = max(page_count - 1, 0)
    closed: list[Section] = []
    for index, section in enumerate(sections):
        if index + 1 < len(sections):
            page_end = max(sections[index + 1].page_start or 0, section.page_start or 0)
        else:
            page_end = last_page
        closed.append(replace(section, page_end=page_end))
    return closed
