"""Тесты сервиса библиотеки: файлы, загрузка, ingest-задачи и удаление."""

from __future__ import annotations

import io
import time
from datetime import UTC, datetime
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
    normalize_source_path,
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
        if job.status in {"done", "failed", "cancelled"}:
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
        ("book.pdf", b"not a pdf", "does not match pdf"),
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


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("D:\\Books", "/mnt/d/Books"),
        ("C:/temp/books", "/mnt/c/temp/books"),
        ("\\\\wsl.localhost\\Ubuntu\\home\\user", "/home/user"),
        ("/mnt/d/Books", "/mnt/d/Books"),
    ],
)
def test_normalize_source_path(raw: str, expected: str) -> None:
    assert normalize_source_path(raw) == Path(expected)


def test_scan_import_counts_nested_pdfs_and_existing(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    settings.ensure_dirs()
    source = tmp_path / "library"
    _pdf(source / "one.pdf")
    _pdf(source / "sub" / "two.pdf")
    _pdf(source / "existing.pdf", PDF_BYTES)
    (source / "notes.txt").write_text("skip", encoding="utf-8")
    (source / ".hidden.pdf").write_bytes(PDF_BYTES)
    (settings.books_dir / "existing.pdf").write_bytes(PDF_BYTES)

    service = LibraryService(settings, _catalog(settings), FakeIndexer())
    try:
        preview = service.scan_import(str(source))
    finally:
        service.close()

    assert preview.pdf_count == 3
    assert preview.non_pdf_count == 1
    assert preview.already_present == 1
    assert preview.to_copy == 2
    assert preview.skipped == 0


def test_import_copies_flat_and_skips_existing(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    settings.ensure_dirs()
    source = tmp_path / "library"
    _pdf(source / "one.pdf")
    _pdf(source / "sub" / "one.pdf", PDF_BYTES + b"x")
    _pdf(source / "existing.pdf", PDF_BYTES)
    (settings.books_dir / "existing.pdf").write_bytes(PDF_BYTES)
    indexer = FakeIndexer()
    service = LibraryService(settings, _catalog(settings), indexer)
    try:
        job = service.start_import(str(source), ingest=False)
        finished = _wait(service, job.id)
    finally:
        service.close()

    assert finished.status == "done"
    assert finished.total == 3
    assert finished.processed == 3
    assert finished.copied == 2
    assert finished.skipped == 1
    assert finished.ingested == 0
    assert (settings.books_dir / "one.pdf").is_file()
    assert (settings.books_dir / "one (2).pdf").is_file()
    assert not (settings.books_dir / "sub").exists()
    assert indexer.calls == []


def test_import_ingests_copied_files_and_skips_ready(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    settings.ensure_dirs()
    source = tmp_path / "library"
    _pdf(source / "one.pdf")
    _pdf(source / "ready.pdf")
    catalog = _catalog(settings)
    catalog.upsert_book(_indexed_book(settings, "ready.pdf", id="book-ready"))
    indexer = FakeIndexer()
    service = LibraryService(settings, catalog, indexer)
    try:
        job = service.start_import(str(source), ingest=True)
        finished = _wait(service, job.id)
    finally:
        service.close()

    assert finished.status == "done"
    assert finished.copied == 2
    assert finished.processed == 2
    assert finished.ingested == 1
    assert indexer.calls == [("one.pdf", False)]


def test_import_caps_batch_and_resumes_next_run(tmp_path: Path) -> None:
    settings = _settings(tmp_path, import_max_files=1)
    settings.ensure_dirs()
    source = tmp_path / "library"
    _pdf(source / "one.pdf")
    _pdf(source / "two.pdf")
    service = LibraryService(settings, _catalog(settings), FakeIndexer())
    try:
        preview = service.scan_import(str(source))
        assert preview.to_copy == 2
        assert preview.batch_size == 1
        assert preview.remaining == 1

        first = service.start_import(str(source), ingest=False)
        assert first.total == 1
        assert first.remaining == 1
        assert _wait(service, first.id).copied == 1
        assert (settings.books_dir / "one.pdf").is_file()
        assert not (settings.books_dir / "two.pdf").exists()

        second = service.start_import(str(source), ingest=False)
        assert second.total == 1
        assert second.remaining == 0
        assert _wait(service, second.id).copied == 1
        assert (settings.books_dir / "two.pdf").is_file()
    finally:
        service.close()


def test_import_rejects_books_dir_as_source(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    service = LibraryService(settings, _catalog(settings), FakeIndexer())
    try:
        with pytest.raises(LibraryValidationError, match="outside"):
            service.start_import(str(settings.books_dir), ingest=False)
    finally:
        service.close()


def test_import_skips_files_over_size_limit(tmp_path: Path) -> None:
    settings = _settings(tmp_path, import_max_file_mb=1)
    settings.ensure_dirs()
    source = tmp_path / "library"
    _pdf(source / "small.pdf")
    _pdf(source / "big.pdf", PDF_BYTES + b"x" * (1024 * 1024))
    service = LibraryService(settings, _catalog(settings), FakeIndexer())
    try:
        preview = service.scan_import(str(source))
        job = service.start_import(str(source), ingest=False)
        finished = _wait(service, job.id)
    finally:
        service.close()

    assert preview.skipped == 1
    assert preview.to_copy == 1
    assert finished.total == 1
    assert finished.processed == 1
    assert finished.copied == 1
    assert (settings.books_dir / "small.pdf").is_file()
    assert not (settings.books_dir / "big.pdf").exists()


def test_import_limits_ignore_already_present_files(tmp_path: Path) -> None:
    settings = _settings(tmp_path, import_max_total_mb=1)
    settings.ensure_dirs()
    payload = PDF_BYTES + b"x" * 1_500_000
    source = tmp_path / "library"
    _pdf(source / "existing.pdf", payload)
    (settings.books_dir / "existing.pdf").write_bytes(payload)
    service = LibraryService(settings, _catalog(settings), FakeIndexer())
    try:
        preview = service.scan_import(str(source))
        job = service.start_import(str(source), ingest=False)
        finished = _wait(service, job.id)
    finally:
        service.close()

    assert preview.to_copy == 0
    assert preview.total_bytes == 0
    assert finished.status == "done"
    assert finished.copied == 0
    assert finished.skipped == 1


def test_import_keeps_same_name_same_size_in_one_batch(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    settings.ensure_dirs()
    source = tmp_path / "library"
    _pdf(source / "one" / "dup.pdf")
    _pdf(source / "two" / "dup.pdf")
    service = LibraryService(settings, _catalog(settings), FakeIndexer())
    try:
        job = service.start_import(str(source), ingest=False)
        finished = _wait(service, job.id)
    finally:
        service.close()

    assert finished.copied == 2
    assert finished.skipped == 0
    assert (settings.books_dir / "dup.pdf").is_file()
    assert (settings.books_dir / "dup (2).pdf").is_file()


def test_import_copy_error_hides_source_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = _settings(tmp_path)
    settings.ensure_dirs()
    source = tmp_path / "library"
    _pdf(source / "a.pdf")
    service = LibraryService(settings, _catalog(settings), FakeIndexer())

    def fail(path: Path, created: set[Path]) -> tuple[Path, bool]:
        raise OSError(f"permission denied: {source / 'a.pdf'}")

    monkeypatch.setattr(service, "_import_file", fail)
    try:
        job = service.start_import(str(source), ingest=False)
        finished = _wait(service, job.id)
    finally:
        service.close()

    assert finished.failed == 1
    assert finished.processed == 1
    assert finished.errors == ["a.pdf: cannot copy (OSError)"]
    assert str(source) not in finished.errors[0]


def test_import_can_be_cancelled(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    settings.ensure_dirs()
    source = tmp_path / "library"
    _pdf(source / "one.pdf")
    _pdf(source / "two.pdf")
    gate = Event()
    service = LibraryService(settings, _catalog(settings), FakeIndexer(gate=gate))
    try:
        job = service.start_import(str(source), ingest=True)
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            if service.get_job(job.id).status == "running":
                break
            time.sleep(0.01)
        service.cancel_job(job.id)
        gate.set()
        finished = _wait(service, job.id)
    finally:
        gate.set()
        service.close()

    assert finished.status == "cancelled"
    assert finished.ingested <= 1


def test_import_unexpected_error_marks_job_failed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = _settings(tmp_path)
    settings.ensure_dirs()
    source = tmp_path / "library"
    _pdf(source / "a.pdf")
    service = LibraryService(settings, _catalog(settings), FakeIndexer())

    def boom(path: Path) -> None:
        raise RuntimeError("catalog exploded")

    monkeypatch.setattr(service.catalog, "get_book_by_path", boom)
    try:
        job = service.start_import(str(source), ingest=True)
        finished = _wait(service, job.id)
    finally:
        service.close()

    assert finished.status == "failed"
    assert finished.error == "catalog exploded"


def test_entries_expose_topics_year_and_added_date(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    catalog = _catalog(settings)
    path = settings.books_dir / "2014 - Автор - Книга.pdf"
    _pdf(path)
    book = _indexed_book(
        settings,
        path.name,
        title="Книга",
        author="Автор",
        created_at=datetime(2026, 11, 1, 12, tzinfo=UTC),
    )
    catalog.upsert_book(book)
    catalog.set_topics(book.id, ["IT", "Бизнес"])
    service = LibraryService(settings, catalog, FakeIndexer())
    try:
        entry = service.entries()[0]
    finally:
        service.close()

    assert entry.author == "Автор"
    assert entry.topics == (
        "Бизнес и предпринимательство",
        "IT, программирование, архитектура",
    )
    assert entry.published_year == 2014
    assert entry.published_year_source == "filename"
    assert entry.added_at == "2026-11-01"


def test_unindexed_file_gets_year_from_name_and_mtime(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    path = settings.books_dir / "2008 - Неизвестная - Книга.pdf"
    _pdf(path)
    service = LibraryService(settings, _catalog(settings), FakeIndexer())
    try:
        entry = service.entries()[0]
    finally:
        service.close()

    expected_date = datetime.fromtimestamp(path.stat().st_mtime, tz=UTC).date()
    assert entry.published_year == 2008
    assert entry.published_year_source == "filename"
    assert entry.topics == ()
    assert entry.added_at == expected_date.isoformat()
