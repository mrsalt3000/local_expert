"""Управление библиотекой книг: файлы на диске, RAG-индекс и фоновые задачи."""

from __future__ import annotations

import os
import re
import shutil
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from threading import RLock
from typing import BinaryIO, Literal, Protocol

from book_expert.catalog import Catalog
from book_expert.config import Settings
from book_expert.ingestion.document import DocumentError
from book_expert.ingestion.extract import (
    SUPPORTED_SUFFIXES,
    detect_format,
)
from book_expert.ingestion.indexer import BookIndexer
from book_expert.models import Book, BookStatus
from book_expert.providers import get_provider

UPLOAD_CHUNK = 1024 * 1024
MAX_NAME_CHARS = 200
JobStatus = Literal["queued", "running", "done", "failed", "cancelled"]
JobKind = Literal["ingest", "reindex", "import"]


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
    format: str = "pdf"
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
    total: int = 0
    remaining: int = 0
    processed: int = 0
    copied: int = 0
    skipped: int = 0
    ingested: int = 0
    failed: int = 0
    current: str | None = None
    errors: list[str] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class ImportPreview:
    pdf_count: int
    non_pdf_count: int
    total_bytes: int
    already_present: int
    skipped: int
    to_copy: int
    batch_size: int
    batch_bytes: int
    remaining: int
    sample: tuple[str, ...]

    @property
    def book_count(self) -> int:
        return self.pdf_count


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
        self._cancelled: set[str] = set()
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
        for path in self._book_files():
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
            try:
                detect_format(temp, name=name)
            except DocumentError as exc:
                raise LibraryValidationError(str(exc)) from exc
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

    def scan_import(self, path: str, *, recursive: bool = True) -> ImportPreview:
        source = self._source_dir(path)
        files, non_pdf = self._collect(source, recursive)
        return self._preview(files, non_pdf)

    def start_import(
        self, path: str, *, recursive: bool = True, ingest: bool = True
    ) -> LibraryJob:
        source = self._source_dir(path)
        files, non_pdf = self._collect(source, recursive)
        preview = self._preview(files, non_pdf)
        if preview.pdf_count == 0:
            raise LibraryValidationError("no supported book files found in the folder")
        eligible = [path for path in files if self._importable(path)]
        if not eligible:
            raise LibraryValidationError("no importable book files found")
        to_import = [path for path in eligible if not self._already_present(path)]
        limit = self.settings.import_max_files
        selected_new = to_import[:limit]
        remaining = max(0, len(to_import) - len(selected_new))
        slots = max(0, limit - len(selected_new))
        already_present = [path for path in eligible if self._already_present(path)][
            :slots
        ]
        batch = selected_new + already_present
        if not batch:
            raise LibraryValidationError("no books to process")
        max_bytes = self.settings.import_max_total_mb * 1024 * 1024
        copy_bytes = sum(_size(path) or 0 for path in selected_new)
        if copy_bytes > max_bytes:
            raise LibraryValidationError(
                f"selected batch is larger than {self.settings.import_max_total_mb} MB"
            )
        free = shutil.disk_usage(self.books_dir).free
        if copy_bytes > free:
            raise LibraryValidationError("not enough free disk space for this import")
        with self._lock:
            if self._closed:
                raise LibraryError("library is shutting down")
            self._ensure_idle_locked()
            job = LibraryJob(
                id=uuid.uuid4().hex,
                kind="import",
                name=source.name,
                total=len(batch),
                remaining=remaining,
            )
            self._jobs[job.id] = job
            self._prune_jobs_locked()
            self._executor.submit(self._run_import, job, batch, ingest)
        return self.get_job(job.id)

    def cancel_job(self, job_id: str) -> LibraryJob:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise LibraryNotFoundError(f"unknown job: {job_id}")
            if job.kind == "import" and job.status in {"queued", "running"}:
                self._cancelled.add(job_id)
        return self.get_job(job_id)

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
                total=job.total,
                remaining=job.remaining,
                processed=job.processed,
                copied=job.copied,
                skipped=job.skipped,
                ingested=job.ingested,
                failed=job.failed,
                current=job.current,
                errors=list(job.errors),
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

    def _run_import(self, job: LibraryJob, files: list[Path], ingest: bool) -> None:
        with self._lock:
            job.status = "running"
        created: set[Path] = set()
        try:
            self._import_loop(job, files, ingest, created)
        except Exception as exc:
            with self._lock:
                job.status = "failed"
                job.error = _public_error(exc, self.books_dir)
                job.current = None
            return
        with self._lock:
            if job.status != "cancelled":
                job.status = "done"
            job.current = None

    def _import_loop(
        self,
        job: LibraryJob,
        files: list[Path],
        ingest: bool,
        created: set[Path],
    ) -> None:
        for src in files:
            if self._is_cancelled(job.id):
                with self._lock:
                    job.status = "cancelled"
                    job.current = None
                return
            with self._lock:
                job.current = src.name
            try:
                destination, copied = self._import_file(src, created)
                if copied:
                    created.add(destination)
            except Exception as exc:
                with self._lock:
                    job.failed += 1
                    job.processed += 1
                    _record_error(
                        job,
                        f"{src.name}: cannot copy ({type(exc).__name__})",
                    )
                continue
            with self._lock:
                if copied:
                    job.copied += 1
                else:
                    job.skipped += 1
                job.current = destination.name
            if ingest:
                existing = self.catalog.get_book_by_path(destination)
                if existing is None or existing.status is not BookStatus.ready:
                    try:
                        self._indexer_instance().index_pdf(destination, force=False)
                    except Exception as exc:
                        with self._lock:
                            job.failed += 1
                            _record_error(
                                job,
                                f"{destination.name}: "
                                f"{_public_error(exc, self.books_dir)}",
                            )
                    else:
                        with self._lock:
                            job.ingested += 1
            with self._lock:
                job.processed += 1

    def _import_file(self, source: Path, created: set[Path]) -> tuple[Path, bool]:
        name = _safe_filename(source.name)
        size = _size(source) or 0
        if size > self.settings.import_max_file_mb * 1024 * 1024:
            raise LibraryValidationError("file is larger than the import limit")
        destination = self.books_dir / name
        if destination.exists() and destination not in created:
            if _size(destination) == _size(source):
                return destination, False
            destination = self._unique_destination(name)
        if destination.exists():
            destination = self._unique_destination(name)
        temp = self.books_dir / f".import-{uuid.uuid4().hex}.part"
        try:
            shutil.copyfile(source, temp)
            os.replace(temp, destination)
        finally:
            temp.unlink(missing_ok=True)
        return destination, True

    def _unique_destination(self, name: str) -> Path:
        stem = Path(name).stem
        suffix = Path(name).suffix
        room = max(1, MAX_NAME_CHARS - len(suffix) - 8)
        for number in range(2, 1000):
            candidate = f"{stem[:room]} ({number}){suffix}"
            destination = self.books_dir / candidate
            if not destination.exists():
                return destination
        raise LibraryConflictError("cannot choose a unique file name")

    def _source_dir(self, raw: str) -> Path:
        path = normalize_source_path(raw)
        if not path.is_absolute():
            path = (self.settings.workspace_path / path).resolve()
        if not path.is_dir():
            raise LibraryValidationError("directory not found or not readable")
        books = self.books_dir.resolve()
        if path == books or books in path.parents:
            raise LibraryValidationError(
                "source folder must be outside the books folder"
            )
        return path

    def _collect(self, source: Path, recursive: bool) -> tuple[list[Path], int]:
        files: list[Path] = []
        non_pdf = 0
        books = self.books_dir.resolve()
        for root, dirs, names in os.walk(source, followlinks=False):
            if not recursive and Path(root) != source:
                dirs[:] = []
                continue
            dirs[:] = [
                name
                for name in dirs
                if not name.startswith(".")
                and not (Path(root) / name).is_symlink()
                and (Path(root) / name).resolve() != books
            ]
            for name in names:
                if name.startswith("."):
                    continue
                path = Path(root) / name
                if path.is_symlink() or not path.is_file():
                    continue
                if path.suffix.lower() not in SUPPORTED_SUFFIXES:
                    non_pdf += 1
                    continue
                files.append(path)
        return sorted(files), non_pdf

    def _preview(self, files: list[Path], non_pdf: int) -> ImportPreview:
        total = 0
        already = 0
        skipped = 0
        copy_sizes: list[int] = []
        sample: list[str] = []
        for path in files:
            if not self._importable(path):
                skipped += 1
                continue
            size = _size(path) or 0
            name = _safe_filename(path.name)
            if self._already_present(path):
                already += 1
                continue
            total += size
            copy_sizes.append(size)
            if len(sample) < 20:
                sample.append(name)
        to_copy = max(0, len(files) - already - skipped)
        limit = self.settings.import_max_files
        batch_size = min(to_copy + already, limit)
        return ImportPreview(
            pdf_count=len(files),
            non_pdf_count=non_pdf,
            total_bytes=total,
            already_present=already,
            skipped=skipped,
            to_copy=to_copy,
            batch_size=batch_size,
            batch_bytes=sum(copy_sizes[:batch_size]),
            remaining=max(0, to_copy - limit),
            sample=tuple(sample),
        )

    def _importable(self, path: Path) -> bool:
        size = _size(path) or 0
        if size > self.settings.import_max_file_mb * 1024 * 1024:
            return False
        try:
            _safe_filename(path.name)
        except LibraryValidationError:
            return False
        return True

    def _already_present(self, path: Path) -> bool:
        name = _safe_filename(path.name)
        destination = self.books_dir / name
        return destination.exists() and _size(destination) == _size(path)

    def _is_cancelled(self, job_id: str) -> bool:
        with self._lock:
            return job_id in self._cancelled

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
            job
            for job in self._jobs.values()
            if job.status in {"done", "failed", "cancelled"}
        ]
        for job in finished[: max(0, len(finished) - 50)]:
            self._jobs.pop(job.id, None)

    def _book_files(self) -> list[Path]:
        if not self.books_dir.exists():
            return []
        return sorted(
            path
            for path in self.books_dir.iterdir()
            if path.is_file()
            and not path.is_symlink()
            and path.suffix.lower() in SUPPORTED_SUFFIXES
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
            format=book.format
            if book is not None
            else path.suffix.lower().lstrip(".") or "pdf",
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
            format=book.format,
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
    if Path(name).suffix.lower() not in SUPPORTED_SUFFIXES:
        raise LibraryValidationError("only PDF, EPUB, FB2 or DJVU files are supported")
    return name


def normalize_source_path(raw: str) -> Path:
    """Принимает WSL-путь или Windows-путь (H:\\... / \\\\wsl.localhost\\...)."""
    text = raw.strip()
    if not text:
        raise LibraryValidationError("path is required")
    if re.match(r"^[A-Za-z]:[\\/]", text):
        drive = text[0].lower()
        rest = text[2:].replace("\\", "/").lstrip("/")
        text = f"/mnt/{drive}/{rest}"
    elif text.startswith("\\\\wsl.localhost\\") or text.startswith("\\\\wsl$\\"):
        parts = [part for part in text.replace("\\", "/").split("/") if part]
        text = "/" + "/".join(parts[2:]) if len(parts) > 2 else "/"
    else:
        text = text.replace("\\", "/")
    path = Path(text).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    return path.resolve()


def _record_error(job: LibraryJob, message: str) -> None:
    if len(job.errors) < 10:
        job.errors.append(message)


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
    "ImportPreview",
    "LibraryConflictError",
    "LibraryEntry",
    "LibraryError",
    "LibraryJob",
    "LibraryNotFoundError",
    "LibraryService",
    "LibraryValidationError",
    "normalize_source_path",
]
