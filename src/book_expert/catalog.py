"""SQLite-каталог книг, секций, chunks и лексического FTS5-индекса."""

from __future__ import annotations

import json
import re
import sqlite3
from collections.abc import Iterable, Sequence
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from book_expert.config import LexicalQueryMode
from book_expert.models import (
    Book,
    BookStatus,
    Chunk,
    SearchHit,
    SearchSource,
    Section,
)
from book_expert.retrieval.lemmas import lemmatize_text, lemmatize_tokens

_SCHEMA = """
CREATE TABLE IF NOT EXISTS books (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    author TEXT,
    path TEXT NOT NULL,
    content_hash TEXT NOT NULL UNIQUE,
    page_count INTEGER NOT NULL DEFAULT 0,
    chunk_count INTEGER NOT NULL DEFAULT 0,
    status TEXT NOT NULL,
    format TEXT NOT NULL DEFAULT 'pdf',
    language TEXT,
    description TEXT,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_books_status ON books(status);

CREATE TABLE IF NOT EXISTS sections (
    book_id TEXT NOT NULL REFERENCES books(id) ON DELETE CASCADE,
    section_path TEXT NOT NULL,
    ordinal INTEGER NOT NULL,
    title TEXT,
    page_start INTEGER,
    page_end INTEGER,
    PRIMARY KEY (book_id, ordinal)
);

CREATE INDEX IF NOT EXISTS idx_sections_path
    ON sections(book_id, section_path);

CREATE TABLE IF NOT EXISTS chunks (
    id TEXT PRIMARY KEY,
    book_id TEXT NOT NULL REFERENCES books(id) ON DELETE CASCADE,
    pdf_page_index INTEGER NOT NULL,
    printed_page_label TEXT,
    section_path TEXT,
    ordinal INTEGER NOT NULL,
    text TEXT NOT NULL,
    lemma_text TEXT,
    token_count INTEGER,
    UNIQUE(book_id, ordinal)
);

CREATE INDEX IF NOT EXISTS idx_chunks_book_page
    ON chunks(book_id, pdf_page_index);

CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    text,
    content='chunks',
    content_rowid='rowid',
    tokenize='unicode61 remove_diacritics 2'
);

CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
    INSERT INTO chunks_fts(rowid, text) VALUES (new.rowid, new.text);
END;

CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, text)
    VALUES ('delete', old.rowid, old.text);
END;

CREATE TRIGGER IF NOT EXISTS chunks_au AFTER UPDATE ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, text)
    VALUES ('delete', old.rowid, old.text);
    INSERT INTO chunks_fts(rowid, text) VALUES (new.rowid, new.text);
END;
"""

_LEMMA_SCHEMA = """
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts_lemma USING fts5(
    lemma_text,
    content='chunks',
    content_rowid='rowid',
    tokenize='unicode61 remove_diacritics 2'
);

CREATE TRIGGER IF NOT EXISTS chunks_lemma_ai AFTER INSERT ON chunks BEGIN
    INSERT INTO chunks_fts_lemma(rowid, lemma_text)
    VALUES (new.rowid, new.lemma_text);
END;

CREATE TRIGGER IF NOT EXISTS chunks_lemma_ad AFTER DELETE ON chunks BEGIN
    INSERT INTO chunks_fts_lemma(chunks_fts_lemma, rowid, lemma_text)
    VALUES ('delete', old.rowid, old.lemma_text);
END;

CREATE TRIGGER IF NOT EXISTS chunks_lemma_au AFTER UPDATE ON chunks BEGIN
    INSERT INTO chunks_fts_lemma(chunks_fts_lemma, rowid, lemma_text)
    VALUES ('delete', old.rowid, old.lemma_text);
    INSERT INTO chunks_fts_lemma(rowid, lemma_text)
    VALUES (new.rowid, new.lemma_text);
END;

CREATE INDEX IF NOT EXISTS chunks_lemma_missing
    ON chunks(lemma_text) WHERE lemma_text IS NULL;
"""


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat()


def _parse_dt(value: str | None) -> datetime | None:
    if not value:
        return None
    return datetime.fromisoformat(value)


def _metadata_json(metadata: dict[str, Any]) -> str:
    return json.dumps(metadata, ensure_ascii=False, sort_keys=True)


def _metadata_load(value: str | None) -> dict[str, Any]:
    if not value:
        return {}
    loaded = json.loads(value)
    return loaded if isinstance(loaded, dict) else {}


class Catalog:
    """Локальный каталог; соединение открывается на каждую операцию."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path, timeout=10)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def initialize(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.executescript(_SCHEMA)
            connection.execute("PRAGMA journal_mode = WAL")
            columns = {
                row["name"] for row in connection.execute("PRAGMA table_info(books)")
            }
            if "format" not in columns:
                connection.execute(
                    "ALTER TABLE books ADD COLUMN format TEXT NOT NULL DEFAULT 'pdf'"
                )
            chunk_columns = {
                row["name"] for row in connection.execute("PRAGMA table_info(chunks)")
            }
            if "lemma_text" not in chunk_columns:
                connection.execute("ALTER TABLE chunks ADD COLUMN lemma_text TEXT")
            connection.executescript(_LEMMA_SCHEMA)

    def upsert_book(
        self, book: Book, sections: Sequence[Section] | None = None
    ) -> Book:
        now = utcnow()
        created_at = book.created_at or now
        updated_at = book.updated_at or now
        path = str(Path(book.path).expanduser().resolve())
        with self._connect() as connection:
            try:
                connection.execute(
                    """
                    INSERT INTO books (
                        id, title, author, path, content_hash, page_count,
                        chunk_count, status, format, language, description,
                        metadata_json, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(id) DO UPDATE SET
                        title = excluded.title,
                        author = excluded.author,
                        path = excluded.path,
                        content_hash = excluded.content_hash,
                        page_count = excluded.page_count,
                        chunk_count = excluded.chunk_count,
                        status = excluded.status,
                        format = excluded.format,
                        language = excluded.language,
                        description = excluded.description,
                        metadata_json = excluded.metadata_json,
                        updated_at = excluded.updated_at
                    """,
                    (
                        book.id,
                        book.title,
                        book.author,
                        path,
                        book.content_hash,
                        book.page_count,
                        book.chunk_count,
                        str(book.status),
                        book.format,
                        book.language,
                        book.description,
                        _metadata_json(book.metadata),
                        _iso(created_at),
                        _iso(updated_at),
                    ),
                )
            except sqlite3.IntegrityError as exc:
                raise ValueError(
                    f"book with content hash {book.content_hash} already exists"
                ) from exc
            if sections is not None:
                connection.execute("DELETE FROM sections WHERE book_id = ?", (book.id,))
                connection.executemany(
                    """
                    INSERT INTO sections (
                        book_id, section_path, ordinal, title, page_start, page_end
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    [
                        (
                            section.book_id,
                            section.section_path,
                            section.ordinal,
                            section.title,
                            section.page_start,
                            section.page_end,
                        )
                        for section in sections
                    ],
                )
        return self.get_book(book.id) or book

    def replace_chunks(self, book_id: str, chunks: Sequence[Chunk]) -> int:
        with self._connect() as connection:
            connection.execute("DELETE FROM chunks WHERE book_id = ?", (book_id,))
            connection.executemany(
                """
                INSERT INTO chunks (
                    id, book_id, pdf_page_index, printed_page_label,
                    section_path, ordinal, text, lemma_text, token_count
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        chunk.id,
                        chunk.book_id,
                        chunk.pdf_page_index,
                        chunk.printed_page_label,
                        chunk.section_path,
                        chunk.ordinal,
                        chunk.text,
                        lemmatize_text(chunk.text),
                        chunk.token_count,
                    )
                    for chunk in chunks
                ],
            )
            connection.execute(
                "UPDATE books SET chunk_count = ?, updated_at = ? WHERE id = ?",
                (len(chunks), _iso(utcnow()), book_id),
            )
        return len(chunks)

    def reindex_lemmas(self) -> int:
        """Заполняет lemma_text для всех chunks и перестраивает FTS5."""
        self.initialize()
        updated = 0
        with self._connect() as connection:
            connection.execute("DROP TRIGGER IF EXISTS chunks_au")
            connection.execute("DROP TRIGGER IF EXISTS chunks_lemma_au")
            cursor = connection.execute("SELECT id, text FROM chunks")
            while True:
                batch = cursor.fetchmany(500)
                if not batch:
                    break
                connection.executemany(
                    "UPDATE chunks SET lemma_text = ? WHERE id = ?",
                    [(lemmatize_text(row["text"]), row["id"]) for row in batch],
                )
                updated += len(batch)
            connection.execute("INSERT INTO chunks_fts(chunks_fts) VALUES ('rebuild')")
            connection.execute(
                "INSERT INTO chunks_fts_lemma(chunks_fts_lemma) VALUES ('rebuild')"
            )
            connection.executescript(_SCHEMA)
            connection.executescript(_LEMMA_SCHEMA)
        return updated

    def _lemma_ready(self) -> bool:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT 1 FROM chunks WHERE lemma_text IS NULL LIMIT 1"
            ).fetchone()
        return row is None

    def page_text(
        self,
        book_id: str,
        pdf_page_index: int,
        *,
        focus_chunk_id: str | None = None,
        max_chars: int = 2000,
    ) -> str:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT id, text FROM chunks
                WHERE book_id = ? AND pdf_page_index = ?
                ORDER BY ordinal
                """,
                (book_id, pdf_page_index),
            ).fetchall()
        full = "\n\n".join(row["text"] for row in rows)
        if len(full) <= max_chars:
            return full
        if focus_chunk_id:
            focus = next(
                (row["text"] for row in rows if row["id"] == focus_chunk_id),
                "",
            )
            position = full.find(focus) if focus else -1
            if position >= 0:
                start = max(0, position - max_chars // 3)
                return full[start : start + max_chars]
        return full[:max_chars]

    def set_status(
        self, book_id: str, status: BookStatus, *, description: str | None = None
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE books
                SET status = ?, description = COALESCE(?, description), updated_at = ?
                WHERE id = ?
                """,
                (str(status), description, _iso(utcnow()), book_id),
            )

    def get_book(self, book_id: str) -> Book | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM books WHERE id = ?", (book_id,)
            ).fetchone()
        return self._row_to_book(row) if row else None

    def get_book_by_path(self, path: Path | str) -> Book | None:
        resolved = str(Path(path).expanduser().resolve())
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM books WHERE path = ?", (resolved,)
            ).fetchone()
        return self._row_to_book(row) if row else None

    def get_book_by_hash(self, content_hash: str) -> Book | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM books WHERE content_hash = ?", (content_hash,)
            ).fetchone()
        return self._row_to_book(row) if row else None

    def list_books(self) -> list[Book]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM books ORDER BY title, id"
            ).fetchall()
        return [self._row_to_book(row) for row in rows]

    def books_by_ids(self, book_ids: Iterable[str]) -> list[Book]:
        ids = list(book_ids)
        if not ids:
            return []
        placeholders = ", ".join("?" for _ in ids)
        with self._connect() as connection:
            rows = connection.execute(
                f"SELECT * FROM books WHERE id IN ({placeholders})", ids
            ).fetchall()
        return [self._row_to_book(row) for row in rows]

    def delete_book(self, book_id: str) -> bool:
        with self._connect() as connection:
            cursor = connection.execute("DELETE FROM books WHERE id = ?", (book_id,))
        return cursor.rowcount > 0

    def get_outline(self, book_id: str) -> list[Section]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM sections WHERE book_id = ? ORDER BY ordinal",
                (book_id,),
            ).fetchall()
        return [
            Section(
                book_id=row["book_id"],
                section_path=row["section_path"],
                ordinal=row["ordinal"],
                title=row["title"],
                page_start=row["page_start"],
                page_end=row["page_end"],
            )
            for row in rows
        ]

    def iter_chunks(self, book_id: str | None = None) -> list[Chunk]:
        query = "SELECT * FROM chunks"
        params: tuple[Any, ...] = ()
        if book_id is not None:
            query += " WHERE book_id = ?"
            params = (book_id,)
        query += " ORDER BY book_id, ordinal"
        with self._connect() as connection:
            rows = connection.execute(query, params).fetchall()
        return [self._row_to_chunk(row) for row in rows]

    def search_lexical(
        self,
        query: str,
        *,
        book_ids: Sequence[str] | None = None,
        limit: int = 20,
        query_mode: LexicalQueryMode = "and",
        use_lemmas: bool = True,
    ) -> list[SearchHit]:
        tokens = _tokens(query)
        if not tokens or limit <= 0:
            return []
        if book_ids is not None and not book_ids:
            return []
        fts_table = "chunks_fts"
        if use_lemmas and self._lemma_ready():
            fts_table = "chunks_fts_lemma"
            tokens = lemmatize_tokens(query)
        if not tokens:
            return []
        if query_mode == "or_prefix5":
            match = " OR ".join(
                f'"{token[:5]}"*' if len(token) >= 5 else f'"{token}"'
                for token in tokens
            )
        else:
            match = " ".join(f'"{token}"' for token in tokens)
        sql = f"""
            SELECT
                c.id, c.book_id, c.text, c.pdf_page_index,
                c.printed_page_label, c.section_path, c.ordinal,
                b.title AS book_title, b.format AS book_format,
                bm25({fts_table}) AS rank
            FROM {fts_table}
            JOIN chunks c ON c.rowid = {fts_table}.rowid
            JOIN books b ON b.id = c.book_id
            WHERE {fts_table} MATCH ? AND b.status = ?
        """
        params: list[Any] = [match, str(BookStatus.ready)]
        if book_ids is not None:
            placeholders = ", ".join("?" for _ in book_ids)
            sql += f" AND c.book_id IN ({placeholders})"
            params.extend(book_ids)
        sql += " ORDER BY rank, c.book_id, c.ordinal LIMIT ?"
        params.append(limit)
        with self._connect() as connection:
            rows = connection.execute(sql, params).fetchall()
        return [
            SearchHit(
                chunk_id=row["id"],
                book_id=row["book_id"],
                text=row["text"],
                score=-float(row["rank"]),
                pdf_page_index=row["pdf_page_index"],
                printed_page_label=row["printed_page_label"],
                section_path=row["section_path"],
                book_title=row["book_title"],
                book_format=row["book_format"],
                source=SearchSource.lexical,
            )
            for row in rows
        ]

    def stats(self) -> dict[str, int]:
        with self._connect() as connection:
            books = connection.execute("SELECT COUNT(*) FROM books").fetchone()[0]
            ready = connection.execute(
                "SELECT COUNT(*) FROM books WHERE status = ?",
                (str(BookStatus.ready),),
            ).fetchone()[0]
            chunks = connection.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        return {"books": books, "ready_books": ready, "chunks": chunks}

    @staticmethod
    def _row_to_book(row: sqlite3.Row) -> Book:
        return Book(
            id=row["id"],
            title=row["title"],
            author=row["author"],
            path=Path(row["path"]),
            content_hash=row["content_hash"],
            page_count=row["page_count"],
            chunk_count=row["chunk_count"],
            status=BookStatus(row["status"]),
            format=row["format"],
            language=row["language"],
            description=row["description"],
            metadata=_metadata_load(row["metadata_json"]),
            created_at=_parse_dt(row["created_at"]),
            updated_at=_parse_dt(row["updated_at"]),
        )

    @staticmethod
    def _row_to_chunk(row: sqlite3.Row) -> Chunk:
        return Chunk(
            id=row["id"],
            book_id=row["book_id"],
            text=row["text"],
            pdf_page_index=row["pdf_page_index"],
            printed_page_label=row["printed_page_label"],
            section_path=row["section_path"],
            ordinal=row["ordinal"],
            token_count=row["token_count"],
        )


def _tokens(query: str, *, limit: int = 32) -> list[str]:
    found = re.findall(r"\w+", query.casefold(), flags=re.UNICODE)
    return found[:limit]
