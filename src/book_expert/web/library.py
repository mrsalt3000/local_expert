"""Управление библиотекой книг: файлы на диске, RAG-индекс и фоновые задачи."""

from __future__ import annotations

import os
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from typing import BinaryIO, Literal, Protocol

from book_expert.catalog import Catalog
from book_expert.config import Settings
from book_expert.ingestion.indexer import BookIndexer
from book_expert.models import Book, BookStatus
from book_expert.providers import get_provider

UPLOAD_CHUNK = 1024 * 1024
MAX_NAME_CHARS = 200
JobStatus = Literal["queued", "running", "done", "failed"]
JobKind = Literal["ingest", "reindex"]


class LibraryError(RuntimeError):
    """Базовая ошибка библиотеки."""


class LibraryNotFoundError(LibraryError):
    """Файл или книга не найдены."""


class LibraryConflictError(LibraryError):
    """Файл уже существует или индексатор занят."""


class LibraryValidationError(LibraryError):
    """Недопустимое имя файла или содержимое."""


class BookIndexerLike(Protocol):
    def index_pdf(
        self, path: str | Path, *, title: str | None = None, force: bool = False
    ) -> Book: ...

    def delete_book(self, book_id: str) -> bool: ...


@dataclass(frozen=True, slots=True)
class LibraryEntry:
    key: str
    name: str
    title: str
    size_bytes: int | None
    file_exists: bool
    in_rag: bool
    status: str
    book_id: str | None = None
    pages: int = 0
    chunks: int = 0
    error: str | None = None


@dataclass(slots=True)
class LibraryJob:
    id: str
    kind: JobKind
    name: str
    status: JobStatus = "queued"
    book_id: str | None = None
    error: str | None = None


class LibraryService:
    """Список книг, загрузка PDF и фоновый ingest/reindex."""

    def __init__(
        self,
        settings: Settings,
        catalog: Catalog | None = None,
        indexer: BookIndexerLike | None = None,
    ) -> None:
        self.settings = settings
        settings.ensure_dirs()
        self.books_dir = settings.books_dir
        self.books_dir.mkdir(parents=True, exist_ok=True)
        self.catalog = catalog or Catalog(settings.catalog_path)
        self.catalog.initialize()
        self._indexer = indexer
        self._provider = None if indexer is not None else get_provider(settings)
        self._lock = RLock()
        self._jobs: dict[str, LibraryJob] = {}
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="library")
        self._closed = False

    def close(self) -> None:
        if self._closed:
            return
        with self._lock:
            self._closed = True
            busy = any(
                job.status in {"queued", "running"} for job in self._jobs.values()
            )
        self._executor.shutdown(wait=False, cancel_futures=True)
        if busy:
            return
        if self._indexer is not None and self._provider is not None:
            self._indexer.close()
        if self._provider is not None:
            close = getattr(self._provider, "close", None)
            if callable(close):
                close()

    def entries(self) -> list[LibraryEntry]:
        entries: list[LibraryEntry] = []
        seen: set[Path] = set()
        for path in self._pdf_files():
            resolved = path.resolve()
            seen.add(resolved)
            entries.append(
                self._entry_for_file(path, self.catalog.get_book_by_path(path))
            )
        for book in self.catalog.list_books():
            if book.path.resolve() in seen:
                continue
            entries.append(self._entry_for_book(book))
        entries.sort(key=lambda item: (not item.in_rag, item.title.lower(), item.name))
        return entries

    def save_upload(
        self,
        filename: str,
        stream: BinaryIO,
        *,
        ingest: bool = False,
    ) -> tuple[LibraryEntry, LibraryJob | None, str | None]:
        name = _safe_filename(filename)
        destination = self.books_dir / name
        if destination.exists():
            raise LibraryConflictError(f"file already exists: {name}")

        limit = self.settings.max_upload_mb * 1024 * 1024
        temp = self.books_dir / f".upload-{uuid.uuid4().hex}.part"
        total = 0
        try:
            with temp.open("wb") as handle:
                while chunk := stream.read(UPLOAD_CHUNK):
                    total += len(chunk)
                    if total > limit:
                        raise LibraryValidationError(
                            f"file is larger than {self.settings.max_upload_mb} MB"
                        )
                    handle.write(chunk)
            if total == 0:
                raise LibraryValidationError("uploaded file is empty")
            with temp.open("rb") as handle:
                if handle.read(5) != b"%PDF-":
                    raise LibraryValidationError("uploaded file is not a PDF")
            os.replace(temp, destination)
        finally:
            temp.unlink(missing_ok=True)

        entry = self._entry_for_file(
            destination, self.catalog.get_book_by_path(destination)
        )
        if not ingest:
            return entry, None, None
        try:
            job = self.start_ingest(name)
        except LibraryConflictError as exc:
            return entry, None, str(exc)
        return entry, job, None

    def start_ingest(self, name: str, *, force: bool = False) -> LibraryJob:
        path = self._resolve_file(name)
        with self._lock:
            if self._closed:
                raise LibraryError("library is shutting down")
            self._ensure_idle_locked()
            job = LibraryJob(
                id=uuid.uuid4().hex,
                kind="reindex" if force else "ingest",
                name=name,
            )
            self._jobs[job.id] = job
            self._prune_jobs_locked()
            self._executor.submit(self._run_job, job, path, force)
        return job

    def get_job(self, job_id: str) -> LibraryJob:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise LibraryNotFoundError(f"unknown job: {job_id}")
            return LibraryJob(
                id=job.id,
                kind=job.kind,
                name=job.name,
                status=job.status,
                book_id=job.book_id,
                error=job.error,
            )

    def remove_from_rag(self, book_id: str) -> None:
        with self._lock:
            if self._closed:
                raise LibraryError("library is shutting down")
            self._ensure_idle_locked()
            book = self.catalog.get_book(book_id)
            if book is None:
                raise LibraryNotFoundError(f"unknown book: {book_id}")
            try:
                self._indexer_instance().delete_book(book_id)
            except Exception as exc:
                raise LibraryError(f"cannot remove from RAG: {exc}") from exc

    def delete_file(self, name: str) -> None:
        path = self._resolve_file(name)
        with self._lock:
            if self._closed:
                raise LibraryError("library is shutting down")
            self._ensure_idle_locked()
            book = self.catalog.get_book_by_path(path)
            if book is not None:
                try:
                    self._indexer_instance().delete_book(book.id)
                except Exception as exc:
                    raise LibraryError(f"cannot remove from RAG: {exc}") from exc
            path.unlink(missing_ok=True)

    def _run_job(self, job: LibraryJob, path: Path, force: bool) -> None:
        with self._lock:
            job.status = "running"
        try:
            book = self._indexer_instance().index_pdf(path, force=force)
        except Exception as exc:
            with self._lock:
                job.status = "failed"
                job.error = _public_error(exc, self.books_dir)
            return
        with self._lock:
            job.status = "done"
            job.book_id = book.id

    def _indexer_instance(self) -> BookIndexerLike:
        with self._lock:
            if self._indexer is None:
                if self._provider is None:
                    self._provider = get_provider(self.settings)
                self._indexer = BookIndexer(self.settings, self._provider, self.catalog)
            return self._indexer

    def _ensure_idle(self) -> None:
        with self._lock:
            self._ensure_idle_locked()

    def _ensure_idle_locked(self) -> None:
        busy = any(job.status in {"queued", "running"} for job in self._jobs.values())
        if busy:
            raise LibraryConflictError("another indexing job is still running")

    def _prune_jobs_locked(self) -> None:
        finished = [
            job for job in self._jobs.values() if job.status in {"done", "failed"}
        ]
        for job in finished[: max(0, len(finished) - 50)]:
            self._jobs.pop(job.id, None)

    def _pdf_files(self) -> list[Path]:
        if not self.books_dir.exists():
            return []
        return sorted(
            path
            for path in self.books_dir.iterdir()
            if path.is_file()
            and not path.is_symlink()
            and path.suffix.lower() == ".pdf"
            and not path.name.startswith(".")
        )

    def _resolve_file(self, name: str) -> Path:
        safe = _safe_filename(name)
        path = (self.books_dir / safe).resolve()
        if path.parent != self.books_dir.resolve():
            raise LibraryValidationError(f"invalid file name: {name}")
        if not path.is_file():
            raise LibraryNotFoundError(f"file not found: {name}")
        return path

    def _entry_for_file(self, path: Path, book: Book | None) -> LibraryEntry:
        in_rag = book is not None and book.status is BookStatus.ready
        return LibraryEntry(
            key=path.name,
            name=path.name,
            title=book.title if book is not None else path.stem,
            size_bytes=_size(path),
            file_exists=True,
            in_rag=in_rag,
            status=book.status.value if book is not None else "not_indexed",
            book_id=book.id if book is not None else None,
            pages=book.page_count if book is not None else 0,
            chunks=book.chunk_count if book is not None else 0,
            error=book.description
            if book is not None and book.status is BookStatus.failed
            else None,
        )

    def _entry_for_book(self, book: Book) -> LibraryEntry:
        exists = book.path.exists()
        return LibraryEntry(
            key=book.id,
            name=book.path.name,
            title=book.title,
            size_bytes=_size(book.path) if exists else None,
            file_exists=exists,
            in_rag=book.status is BookStatus.ready,
            status=book.status.value,
            book_id=book.id,
            pages=book.page_count,
            chunks=book.chunk_count,
            error=book.description if book.status is BookStatus.failed else None,
        )


def _safe_filename(raw: str) -> str:
    name = Path(raw.replace("\\", "/")).name.strip()
    if not name or name in {".", ".."} or "\x00" in name:
        raise LibraryValidationError("invalid file name")
    if name.startswith("."):
        raise LibraryValidationError("hidden file names are not allowed")
    if len(name) > MAX_NAME_CHARS:
        raise LibraryValidationError(f"file name is longer than {MAX_NAME_CHARS} chars")
    if Path(name).suffix.lower() != ".pdf":
        raise LibraryValidationError("only PDF files are supported")
    return name


def _public_error(exc: Exception, books_dir: Path) -> str:
    """Не отдаёт клиенту абсолютные пути из сообщения об ошибке."""
    message = str(exc).replace(str(books_dir), "books")
    return message[:500]


def _size(path: Path) -> int | None:
    try:
        return path.stat().st_size
    except OSError:
        return None


__all__ = [
    "LibraryConflictError",
    "LibraryEntry",
    "LibraryError",
    "LibraryJob",
    "LibraryNotFoundError",
    "LibraryService",
    "LibraryValidationError",
]
