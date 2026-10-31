"""Тесты сервиса библиотеки: файлы, загрузка, ingest-задачи и удаление."""

from __future__ import annotations

import io
import time
from pathlib import Path
from threading import Event

import pytest

from book_expert.catalog import Catalog
from book_expert.config import Settings
from book_expert.models import Book, BookStatus
from book_expert.web.library import (
    LibraryConflictError,
    LibraryError,
    LibraryNotFoundError,
    LibraryService,
    LibraryValidationError,
)

PDF_BYTES = b"%PDF-1.4\n% synthetic test pdf\n"


class FakeIndexer:
    def __init__(self, *, error: Exception | None = None, gate: Event | None = None):
        self.calls: list[tuple[str, bool]] = []
        self.deleted: list[str] = []
        self.error = error
        self.gate = gate

    def index_pdf(
        self, path: str | Path, *, title: str | None = None, force: bool = False
    ) -> Book:
        name = Path(path).name
        self.calls.append((name, force))
        if self.gate is not None:
            self.gate.wait(5)
        if self.error is not None:
            raise self.error
        return Book(
            id="book-new",
            title=title or Path(path).stem,
            path=Path(path),
            content_hash="hash-new",
            status=BookStatus.ready,
            page_count=1,
            chunk_count=2,
        )

    def delete_book(self, book_id: str) -> bool:
        self.deleted.append(book_id)
        return True


def _settings(tmp_path: Path, **overrides: object) -> Settings:
    return Settings(
        _env_file=None,
        workspace=tmp_path,
        data_dir=tmp_path / "data",
        max_upload_mb=1,
        **overrides,
    )


def _catalog(settings: Settings) -> Catalog:
    catalog = Catalog(settings.catalog_path)
    catalog.initialize()
    return catalog


def _pdf(path: Path, content: bytes = PDF_BYTES) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def _indexed_book(settings: Settings, name: str, **overrides: object) -> Book:
    path = settings.books_dir / name
    values: dict[str, object] = {
        "id": "book-indexed",
        "title": "Indexed Book",
        "path": path,
        "content_hash": f"hash-{name}",
        "status": BookStatus.ready,
        "page_count": 10,
        "chunk_count": 20,
    }
    values.update(overrides)
    return Book(**values)  # type: ignore[arg-type]


def _wait(service: LibraryService, job_id: str, timeout: float = 5.0):  # type: ignore[no-untyped-def]
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        job = service.get_job(job_id)
        if job.status in {"done", "failed"}:
            return job
        time.sleep(0.01)
    raise AssertionError("job did not finish")


def test_entries_merge_files_and_catalog(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    catalog = _catalog(settings)
    _pdf(settings.books_dir / "ready.pdf")
    _pdf(settings.books_dir / "new.pdf")
    _pdf(settings.books_dir / "failed.pdf")
    (settings.books_dir / "notes.txt").write_text("skip me", encoding="utf-8")
    catalog.upsert_book(_indexed_book(settings, "ready.pdf"))
    catalog.upsert_book(
        _indexed_book(
            settings,
            "failed.pdf",
            id="book-failed",
            title="Failed Book",
            status=BookStatus.failed,
            description="parse error",
        )
    )
    catalog.upsert_book(
        _indexed_book(
            settings,
            "gone.pdf",
            id="book-gone",
            title="Gone Book",
        )
    )
    service = LibraryService(settings, catalog, FakeIndexer())
    try:
        entries = {entry.name: entry for entry in service.entries()}
    finally:
        service.close()

    assert set(entries) == {"ready.pdf", "new.pdf", "failed.pdf", "gone.pdf"}
    assert entries["ready.pdf"].in_rag is True
    assert entries["ready.pdf"].status == "ready"
    assert entries["new.pdf"].status == "not_indexed"
    assert entries["new.pdf"].file_exists is True
    assert entries["failed.pdf"].status == "failed"
    assert entries["failed.pdf"].error == "parse error"
    assert entries["gone.pdf"].file_exists is False
    assert entries["gone.pdf"].key == "book-gone"


def test_upload_saves_pdf_and_sanitizes_name(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    service = LibraryService(settings, _catalog(settings), FakeIndexer())
    try:
        entry, job, note = service.save_upload("../../evil.pdf", io.BytesIO(PDF_BYTES))
    finally:
        service.close()

    assert job is None and note is None
    assert entry.name == "evil.pdf"
    assert (settings.books_dir / "evil.pdf").is_file()
    assert not (tmp_path / "evil.pdf").exists()


@pytest.mark.parametrize(
    ("name", "content", "message"),
    [
        ("book.txt", PDF_BYTES, "only PDF"),
        ("book.pdf", b"not a pdf", "not a PDF"),
        ("book.pdf", b"", "empty"),
        ("book.pdf", b"%PDF-" + b"x" * (1024 * 1024), "larger than 1 MB"),
        (".hidden.pdf", PDF_BYTES, "hidden"),
    ],
)
def test_upload_rejects_invalid_files(
    tmp_path: Path, name: str, content: bytes, message: str
) -> None:
    settings = _settings(tmp_path)
    service = LibraryService(settings, _catalog(settings), FakeIndexer())
    try:
        with pytest.raises(LibraryValidationError, match=message):
            service.save_upload(name, io.BytesIO(content))
        assert not any(settings.books_dir.iterdir())
    finally:
        service.close()


def test_upload_conflict(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    _pdf(settings.books_dir / "book.pdf")
    service = LibraryService(settings, _catalog(settings), FakeIndexer())
    try:
        with pytest.raises(LibraryConflictError, match="already exists"):
            service.save_upload("book.pdf", io.BytesIO(PDF_BYTES))
    finally:
        service.close()


def test_ingest_job_completes_and_reindex_uses_force(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    _pdf(settings.books_dir / "book.pdf")
    indexer = FakeIndexer()
    service = LibraryService(settings, _catalog(settings), indexer)
    try:
        job = service.start_ingest("book.pdf")
        finished = _wait(service, job.id)
        assert finished.status == "done"
        assert finished.book_id == "book-new"
        assert indexer.calls == [("book.pdf", False)]

        reindex = service.start_ingest("book.pdf", force=True)
        assert _wait(service, reindex.id).status == "done"
        assert indexer.calls[-1] == ("book.pdf", True)
    finally:
        service.close()


def test_ingest_job_records_failure(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    _pdf(settings.books_dir / "book.pdf")
    service = LibraryService(
        settings, _catalog(settings), FakeIndexer(error=RuntimeError("boom"))
    )
    try:
        job = service.start_ingest("book.pdf")
        finished = _wait(service, job.id)
    finally:
        service.close()

    assert finished.status == "failed"
    assert finished.error == "boom"


def test_second_ingest_is_rejected_while_busy(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    _pdf(settings.books_dir / "one.pdf")
    _pdf(settings.books_dir / "two.pdf")
    gate = Event()
    service = LibraryService(settings, _catalog(settings), FakeIndexer(gate=gate))
    try:
        job = service.start_ingest("one.pdf")
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            if service.get_job(job.id).status == "running":
                break
            time.sleep(0.01)
        with pytest.raises(LibraryConflictError, match="still running"):
            service.start_ingest("two.pdf")
        gate.set()
        assert _wait(service, job.id).status == "done"
    finally:
        gate.set()
        service.close()


def test_ingest_unknown_file(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    service = LibraryService(settings, _catalog(settings), FakeIndexer())
    try:
        with pytest.raises(LibraryNotFoundError, match="file not found"):
            service.start_ingest("missing.pdf")
    finally:
        service.close()


def test_upload_with_ingest_when_busy_still_saves_file(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    _pdf(settings.books_dir / "one.pdf")
    gate = Event()
    service = LibraryService(settings, _catalog(settings), FakeIndexer(gate=gate))
    try:
        job = service.start_ingest("one.pdf")
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            if service.get_job(job.id).status == "running":
                break
            time.sleep(0.01)
        entry, queued, note = service.save_upload(
            "two.pdf", io.BytesIO(PDF_BYTES), ingest=True
        )
        assert entry.name == "two.pdf"
        assert queued is None
        assert note and "still running" in note
        assert (settings.books_dir / "two.pdf").is_file()
        gate.set()
        assert _wait(service, job.id).status == "done"
    finally:
        gate.set()
        service.close()


def test_symlinks_are_not_listed(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    settings.ensure_dirs()
    target = _pdf(tmp_path / "outside.pdf")
    (settings.books_dir / "link.pdf").symlink_to(target)
    service = LibraryService(settings, _catalog(settings), FakeIndexer())
    try:
        assert service.entries() == []
    finally:
        service.close()


def test_job_error_does_not_leak_absolute_path(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    _pdf(settings.books_dir / "book.pdf")
    error = RuntimeError(f"{settings.books_dir}/book.pdf disappeared")
    service = LibraryService(settings, _catalog(settings), FakeIndexer(error=error))
    try:
        job = service.start_ingest("book.pdf")
        finished = _wait(service, job.id)
    finally:
        service.close()

    assert finished.status == "failed"
    assert str(settings.books_dir) not in (finished.error or "")
    assert finished.error == "books/book.pdf disappeared"


def test_ingest_after_close_is_rejected(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    _pdf(settings.books_dir / "book.pdf")
    service = LibraryService(settings, _catalog(settings), FakeIndexer())
    service.close()

    with pytest.raises(LibraryError, match="shutting down"):
        service.start_ingest("book.pdf")


def test_remove_from_rag_keeps_file(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    catalog = _catalog(settings)
    _pdf(settings.books_dir / "book.pdf")
    catalog.upsert_book(_indexed_book(settings, "book.pdf"))
    indexer = FakeIndexer()
    service = LibraryService(settings, catalog, indexer)
    try:
        service.remove_from_rag("book-indexed")
    finally:
        service.close()

    assert indexer.deleted == ["book-indexed"]
    assert (settings.books_dir / "book.pdf").is_file()


def test_delete_file_removes_index_too(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    catalog = _catalog(settings)
    _pdf(settings.books_dir / "book.pdf")
    catalog.upsert_book(_indexed_book(settings, "book.pdf"))
    indexer = FakeIndexer()
    service = LibraryService(settings, catalog, indexer)
    try:
        service.delete_file("book.pdf")
    finally:
        service.close()

    assert indexer.deleted == ["book-indexed"]
    assert not (settings.books_dir / "book.pdf").exists()


def test_delete_unindexed_file(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    _pdf(settings.books_dir / "book.pdf")
    indexer = FakeIndexer()
    service = LibraryService(settings, _catalog(settings), indexer)
    try:
        service.delete_file("book.pdf")
    finally:
        service.close()

    assert indexer.deleted == []
    assert not (settings.books_dir / "book.pdf").exists()
