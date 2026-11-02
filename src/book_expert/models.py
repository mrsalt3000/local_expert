"""Доменные модели Book Expert."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Any


class BookStatus(StrEnum):
    pending = "pending"
    indexing = "indexing"
    ready = "ready"
    failed = "failed"


class SearchSource(StrEnum):
    dense = "dense"
    lexical = "lexical"
    fused = "fused"


@dataclass(frozen=True, slots=True)
class Book:
    id: str
    title: str
    path: Path
    content_hash: str
    status: BookStatus = BookStatus.pending
    format: str = "pdf"
    author: str | None = None
    page_count: int = 0
    chunk_count: int = 0
    language: str | None = None
    description: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime | None = None
    updated_at: datetime | None = None


@dataclass(frozen=True, slots=True)
class Section:
    book_id: str
    section_path: str
    ordinal: int
    title: str | None = None
    page_start: int | None = None
    page_end: int | None = None


@dataclass(frozen=True, slots=True)
class Chunk:
    id: str
    book_id: str
    text: str
    pdf_page_index: int
    ordinal: int
    printed_page_label: str | None = None
    section_path: str | None = None
    token_count: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class SearchHit:
    chunk_id: str
    book_id: str
    text: str
    score: float
    pdf_page_index: int
    printed_page_label: str | None = None
    section_path: str | None = None
    book_title: str | None = None
    book_format: str = "pdf"
    source: SearchSource = SearchSource.fused
    context: str | None = None


@dataclass(frozen=True, slots=True)
class Citation:
    book_id: str
    book_title: str
    pdf_page_index: int
    chunk_id: str
    printed_page_label: str | None = None
    section_path: str | None = None
    format: str = "pdf"
    number: int = 0
    quote: str | None = None


@dataclass(frozen=True, slots=True)
class Answer:
    text: str
    citations: tuple[Citation, ...] = ()
    refused: bool = False
    confidence: float | None = None
    book_ids: tuple[str, ...] = ()
    route_reason: str | None = None
    plan: tuple[str, ...] = ()
    rounds: int = 0
    max_rounds: int = 0
    sources: tuple[SearchHit, ...] = ()
