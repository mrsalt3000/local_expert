"""FastAPI-приложение локального веб-чата."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Callable, Iterator, Sequence
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from queue import Queue
from threading import Lock, Thread
from time import perf_counter
from typing import Annotated, Any, Literal, Protocol
from uuid import uuid4

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field, field_validator

from book_expert.agent.graph import AgentError, AgentTimeoutError
from book_expert.config import Settings
from book_expert.models import Answer, Book
from book_expert.providers import ProviderError, ProviderUnavailableError
from book_expert.topics import TOPICS
from book_expert.web.answer_worker import (
    AnswerTimeoutError,
    WorkerAnswerError,
    answer_in_process,
    stream_answer_in_process,
    stream_error as _stream_error,
)
from book_expert.web.backend import BookChatBackend
from book_expert.web.metrics import RequestMetrics
from book_expert.web.library import (
    ImportPreview,
    LibraryConflictError,
    LibraryEntry,
    LibraryError,
    LibraryJob,
    LibraryNotFoundError,
    LibraryService,
    LibraryValidationError,
)

STATIC_DIR = Path(__file__).parent / "static"
MAX_MESSAGE_CHARS = 4000
MAX_BOOK_FILTERS = 20
MAX_BOOK_ID_CHARS = 128
MAX_FILE_NAME_CHARS = 255


class ChatBackend(Protocol):
    """Минимальный контракт бэкенда; в тестах подменяется fake-объектом."""

    def list_books(self) -> list[Book]: ...

    def ask_rag(
        self,
        message: str,
        book_ids: Sequence[str] | None = None,
        history: Sequence[tuple[str, str]] | None = None,
    ) -> Answer: ...

    def ask_advanced(
        self,
        message: str,
        book_ids: Sequence[str] | None = None,
        progress: Callable[[dict[str, Any]], None] | None = None,
        history: Sequence[tuple[str, str]] | None = None,
    ) -> Answer: ...

    def ask_super(
        self,
        message: str,
        book_ids: Sequence[str] | None = None,
        progress: Callable[[dict[str, Any]], None] | None = None,
        history: Sequence[tuple[str, str]] | None = None,
    ) -> Answer: ...

    def ask_agent(
        self,
        message: str,
        book_ids: Sequence[str] | None = None,
        history: Sequence[tuple[str, str]] | None = None,
    ) -> str: ...

    def close(self) -> None: ...


class ChatTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=MAX_MESSAGE_CHARS)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=MAX_MESSAGE_CHARS)
    history: list[ChatTurn] = Field(default_factory=list, max_length=20)
    mode: Literal["rag", "advanced", "super", "agent"] = "super"
    book_ids: list[Annotated[str, Field(max_length=MAX_BOOK_ID_CHARS)]] = Field(
        default_factory=list, max_length=MAX_BOOK_FILTERS
    )

    @field_validator("message")
    @classmethod
    def _reject_blank(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("message must not be blank")
        return stripped


class BookOut(BaseModel):
    id: str
    title: str
    author: str | None = None
    pages: int
    chunks: int
    status: str
    format: str = "pdf"


class CitationOut(BaseModel):
    book_id: str
    book_title: str
    page: str | int
    chunk_id: str
    section: str | None = None
    format: str = "pdf"
    number: int = 0
    quote: str | None = None


class ChatResponse(BaseModel):
    mode: Literal["rag", "advanced", "super", "agent"]
    answer: str
    refused: bool = False
    uncited: bool = False
    unsupported_claims: list[str] = Field(default_factory=list)
    citations: list[CitationOut] = Field(default_factory=list)
    confidence: float | None = None
    route_reason: str | None = None
    plan: list[str] = Field(default_factory=list)
    rounds: int = 0
    max_rounds: int = 0
    elapsed_ms: int
    request_id: str = ""


class LibraryBookOut(BaseModel):
    key: str
    name: str
    title: str
    size_bytes: int | None = None
    file_exists: bool
    in_rag: bool
    status: str
    book_id: str | None = None
    format: str = "pdf"
    pages: int = 0
    chunks: int = 0
    error: str | None = None
    author: str | None = None
    topics: list[str] = Field(default_factory=list)
    published_year: int | None = None
    published_year_source: str | None = None
    added_at: str | None = None


class TopicsOut(BaseModel):
    topics: list[str] = Field(default_factory=list)


class LibraryJobOut(BaseModel):
    id: str
    kind: Literal["ingest", "reindex", "import"]
    name: str
    status: Literal["queued", "running", "done", "failed", "cancelled"]
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
    errors: list[str] = Field(default_factory=list)


class ImportScanRequest(BaseModel):
    path: str = Field(min_length=1, max_length=4096)
    recursive: bool = True


class ImportStartRequest(ImportScanRequest):
    ingest: bool = True


class ImportPreviewOut(BaseModel):
    book_count: int
    pdf_count: int
    non_pdf_count: int
    total_bytes: int
    already_present: int
    skipped: int
    to_copy: int
    batch_size: int
    batch_bytes: int
    remaining: int
    sample: list[str] = Field(default_factory=list)


class LibraryUploadOut(BaseModel):
    entry: LibraryBookOut
    job: LibraryJobOut | None = None
    note: str | None = None


class LibraryIngestRequest(BaseModel):
    name: str = Field(min_length=1, max_length=MAX_FILE_NAME_CHARS)
    force: bool = False


class LibraryRemoveRequest(BaseModel):
    book_id: str = Field(min_length=1, max_length=MAX_BOOK_ID_CHARS)


class LibraryDeleteRequest(BaseModel):
    name: str = Field(min_length=1, max_length=MAX_FILE_NAME_CHARS)


class OkResponse(BaseModel):
    ok: bool = True


class HealthOut(BaseModel):
    ok: bool = True
    uptime_s: float
    requests: int
    refusals: int
    errors: int
    window: int
    latency_ms: dict[str, int | None]
    tokens: dict[str, int] = Field(default_factory=dict)
    cost_usd: float = 0.0
    usage_coverage: float = 0.0
    modes: dict[str, int]
    by_mode: dict[str, Any] = Field(default_factory=dict)
    by_provider: dict[str, Any] = Field(default_factory=dict)
    slo: dict[str, Any] = Field(default_factory=dict)


class FeedbackRequest(BaseModel):
    chat_id: str = Field(default="", max_length=MAX_BOOK_ID_CHARS)
    message_index: int = Field(default=-1, ge=-1, le=10_000)
    request_id: str = Field(default="", max_length=64)
    rating: Literal["up", "down"]
    mode: str = Field(default="", max_length=32)
    comment: str = Field(default="", max_length=500)


def create_app(
    settings: Settings | None = None,
    backend: ChatBackend | None = None,
    library: LibraryService | None = None,
) -> FastAPI:
    """Создаёт приложение; backend и library можно подменить в тестах."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        yield
        for name in ("backend", "library"):
            active = getattr(app.state, name, None)
            if active is not None:
                active.close()

    app = FastAPI(
        title="Book Expert",
        docs_url="/api/docs",
        openapi_url="/api/openapi.json",
        lifespan=lifespan,
    )
    app.state.settings = settings or Settings()
    if backend is not None:
        app.state.backend = backend
    if library is not None:
        app.state.library = library
    backend_lock = Lock()
    library_lock = Lock()
    feedback_lock = Lock()
    metrics = RequestMetrics(
        path=app.state.settings.artifacts_dir / "metrics.local.jsonl",
        price_input_per_mtok=app.state.settings.price_input_per_mtok,
        price_output_per_mtok=app.state.settings.price_output_per_mtok,
    )

    def write_feedback(entry: dict[str, Any]) -> None:
        path = app.state.settings.artifacts_dir / "feedback.local.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(entry, ensure_ascii=False) + "\n"
        with feedback_lock:
            with path.open("a", encoding="utf-8") as handle:
                handle.write(line)

    def get_backend() -> ChatBackend:
        active = getattr(app.state, "backend", None)
        if active is None:
            with backend_lock:
                active = getattr(app.state, "backend", None)
                if active is None:
                    try:
                        active = BookChatBackend(app.state.settings)
                    except ValueError as exc:
                        raise ProviderUnavailableError(
                            f"backend is not configured: {exc}"
                        ) from exc
                    app.state.backend = active
        return active

    def get_library() -> LibraryService:
        active = getattr(app.state, "library", None)
        if active is None:
            with library_lock:
                active = getattr(app.state, "library", None)
                if active is None:
                    try:
                        active = LibraryService(app.state.settings)
                    except ValueError as exc:
                        raise ProviderUnavailableError(
                            f"library is not configured: {exc}"
                        ) from exc
                    app.state.library = active
        return active

    def raise_library_error(exc: Exception) -> None:
        if isinstance(exc, LibraryNotFoundError):
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        if isinstance(exc, LibraryConflictError):
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        if isinstance(exc, LibraryValidationError):
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if isinstance(exc, LibraryError):
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        if isinstance(exc, ProviderError):
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        if isinstance(exc, OSError):
            raise HTTPException(status_code=503, detail="local storage error") from exc
        raise exc

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(
            STATIC_DIR / "index.html",
            headers={"Cache-Control": "no-store"},
        )

    @app.get("/api/books", response_model=list[BookOut])
    def books() -> list[BookOut]:
        try:
            catalog_books = get_backend().list_books()
        except ProviderError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        return [_book_out(book) for book in catalog_books]

    @app.post("/api/chat", response_model=ChatResponse)
    def chat(payload: ChatRequest) -> ChatResponse:
        book_ids = payload.book_ids or None
        history = [(turn.role, turn.content) for turn in payload.history[-12:]]
        started = perf_counter()
        request_id = uuid4().hex
        refused = False
        failed = False
        usage: dict[str, int] = {}
        citations = 0
        unsupported = 0
        try:
            active = get_backend()
            if payload.mode == "rag":
                answer = active.ask_rag(payload.message, book_ids, history=history)
                response = _rag_response(answer)
                refused = answer.refused
            elif payload.mode in {"advanced", "super"}:
                if app.state.settings.answer_hard_timeout and isinstance(
                    active, BookChatBackend
                ):
                    answer = answer_in_process(
                        app.state.settings,
                        payload.mode,
                        payload.message,
                        book_ids=book_ids,
                        history=history,
                    )
                elif payload.mode == "advanced":
                    answer = active.ask_advanced(
                        payload.message, book_ids, history=history
                    )
                else:
                    answer = active.ask_super(
                        payload.message, book_ids, history=history
                    )
                response = _rag_response(answer, mode=payload.mode)
                refused = answer.refused
            else:
                text = active.ask_agent(payload.message, book_ids, history=history)
                response = ChatResponse(mode="agent", answer=text, elapsed_ms=0)
                usage = dict(getattr(active, "agent_usage", {}) or {})
            if payload.mode != "agent":
                usage = dict(answer.usage)
                citations = len(answer.citations)
                unsupported = len(answer.unsupported_claims)
        except ProviderError as exc:
            failed = True
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except AgentTimeoutError as exc:
            failed = True
            raise HTTPException(status_code=504, detail=str(exc)) from exc
        except AgentError as exc:
            failed = True
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        except ValueError as exc:
            failed = True
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except AnswerTimeoutError as exc:
            failed = True
            raise HTTPException(status_code=504, detail=str(exc)) from exc
        except WorkerAnswerError as exc:
            failed = True
            raise HTTPException(status_code=exc.status, detail=str(exc)) from exc
        finally:
            elapsed_ms = (perf_counter() - started) * 1000
            metrics.record(
                mode=payload.mode,
                elapsed_ms=elapsed_ms,
                refused=refused,
                error=failed,
                usage=usage,
                provider=app.state.settings.provider,
                model=_active_chat_model(app.state.settings),
                request_id=request_id,
                citations=citations,
                unsupported_claims=unsupported,
            )
        response.elapsed_ms = int((perf_counter() - started) * 1000)
        response.request_id = request_id
        return response

    @app.post("/api/chat/stream")
    def chat_stream(payload: ChatRequest) -> StreamingResponse:
        if payload.mode not in {"advanced", "super"}:
            raise HTTPException(
                status_code=400,
                detail="streaming is available for advanced and super modes only",
            )
        book_ids = payload.book_ids or None
        history = [(turn.role, turn.content) for turn in payload.history[-12:]]
        started = perf_counter()
        request_id = uuid4().hex

        def record(answer: Answer | None, *, error: bool = False) -> None:
            elapsed_ms = (perf_counter() - started) * 1000
            metrics.record(
                mode=payload.mode,
                elapsed_ms=elapsed_ms,
                refused=bool(answer and answer.refused),
                error=error,
                usage=dict(answer.usage) if answer is not None else {},
                provider=app.state.settings.provider,
                model=_active_chat_model(app.state.settings),
                request_id=request_id,
                citations=len(answer.citations) if answer is not None else 0,
                unsupported_claims=len(answer.unsupported_claims)
                if answer is not None
                else 0,
            )

        def generate() -> Iterator[str]:
            try:
                backend = get_backend()
            except Exception as exc:  # noqa: BLE001 - SSE обязан вернуть error
                record(None, error=True)
                yield _sse({"type": "error", **_stream_error(exc)})
                return
            if app.state.settings.answer_hard_timeout and isinstance(
                backend, BookChatBackend
            ):
                yield from _hard_timeout_stream(
                    app.state.settings,
                    payload,
                    book_ids,
                    history,
                    started,
                    request_id=request_id,
                    on_done=lambda answer, _elapsed: record(answer),
                    on_error=lambda _elapsed: record(None, error=True),
                )
                return
            events: Queue[dict[str, Any] | None] = Queue()

            def emit(event: dict[str, Any]) -> None:
                events.put(event)

            def worker() -> None:
                try:
                    if payload.mode == "super":
                        answer = backend.ask_super(
                            payload.message,
                            book_ids,
                            progress=emit,
                            history=history,
                        )
                    else:
                        answer = backend.ask_advanced(
                            payload.message,
                            book_ids,
                            progress=emit,
                            history=history,
                        )
                except Exception as exc:
                    record(None, error=True)
                    events.put({"type": "error", **_stream_error(exc)})
                else:
                    response = _rag_response(answer, mode=payload.mode)
                    response.elapsed_ms = int((perf_counter() - started) * 1000)
                    response.request_id = request_id
                    record(answer)
                    events.put(
                        {
                            "type": "done",
                            "response": response.model_dump(),
                        }
                    )
                finally:
                    events.put(None)

            Thread(target=worker, daemon=True).start()
            while True:
                event = events.get()
                if event is None:
                    break
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-store",
                "X-Accel-Buffering": "no",
            },
        )

    @app.get("/api/library", response_model=list[LibraryBookOut])
    def library_books() -> list[LibraryBookOut]:
        try:
            entries = get_library().entries()
        except Exception as exc:
            raise_library_error(exc)
        return [_library_out(entry) for entry in entries]

    @app.get("/api/library/topics", response_model=TopicsOut)
    def library_topics() -> TopicsOut:
        return TopicsOut(topics=list(TOPICS))

    @app.post(
        "/api/library/upload",
        response_model=LibraryUploadOut,
        status_code=201,
    )
    def library_upload(
        file: UploadFile = File(...),
        ingest: bool = Form(False),
    ) -> LibraryUploadOut:
        if not file.filename:
            raise HTTPException(status_code=400, detail="file name is required")
        try:
            entry, job, note = get_library().save_upload(
                file.filename, file.file, ingest=ingest
            )
        except Exception as exc:
            raise_library_error(exc)
        finally:
            file.file.close()
        return LibraryUploadOut(
            entry=_library_out(entry),
            job=_job_out(job),
            note=note,
        )

    @app.post(
        "/api/library/ingest",
        response_model=LibraryJobOut,
        status_code=202,
    )
    def library_ingest(payload: LibraryIngestRequest) -> LibraryJobOut:
        try:
            job = get_library().start_ingest(payload.name, force=payload.force)
        except Exception as exc:
            raise_library_error(exc)
        return _job_out_required(job)

    @app.post("/api/library/remove-from-rag", response_model=OkResponse)
    def library_remove(payload: LibraryRemoveRequest) -> OkResponse:
        try:
            get_library().remove_from_rag(payload.book_id)
        except Exception as exc:
            raise_library_error(exc)
        return OkResponse()

    @app.post("/api/library/delete-file", response_model=OkResponse)
    def library_delete(payload: LibraryDeleteRequest) -> OkResponse:
        try:
            get_library().delete_file(payload.name)
        except Exception as exc:
            raise_library_error(exc)
        return OkResponse()

    @app.post("/api/library/scan", response_model=ImportPreviewOut)
    def library_scan(payload: ImportScanRequest) -> ImportPreviewOut:
        try:
            preview = get_library().scan_import(
                payload.path, recursive=payload.recursive
            )
        except Exception as exc:
            raise_library_error(exc)
        return _preview_out(preview)

    @app.post(
        "/api/library/import",
        response_model=LibraryJobOut,
        status_code=202,
    )
    def library_import(payload: ImportStartRequest) -> LibraryJobOut:
        try:
            job = get_library().start_import(
                payload.path,
                recursive=payload.recursive,
                ingest=payload.ingest,
            )
        except Exception as exc:
            raise_library_error(exc)
        return _job_out_required(job)

    @app.post("/api/jobs/{job_id}/cancel", response_model=LibraryJobOut)
    def library_cancel(job_id: str) -> LibraryJobOut:
        try:
            job = get_library().cancel_job(job_id)
        except Exception as exc:
            raise_library_error(exc)
        return _job_out_required(job)

    @app.get("/api/jobs/{job_id}", response_model=LibraryJobOut)
    def library_job(job_id: str) -> LibraryJobOut:
        try:
            job = get_library().get_job(job_id)
        except Exception as exc:
            raise_library_error(exc)
        return _job_out_required(job)

    @app.get("/api/health", response_model=HealthOut)
    def health() -> HealthOut:
        return HealthOut(**metrics.snapshot())

    @app.post("/api/feedback", response_model=OkResponse)
    def feedback(payload: FeedbackRequest) -> OkResponse:
        write_feedback(
            {
                "chat_id": payload.chat_id,
                "message_index": payload.message_index,
                "request_id": payload.request_id,
                "rating": payload.rating,
                "mode": payload.mode,
                "comment": payload.comment,
                "recorded_at": datetime.now(UTC).isoformat(timespec="seconds"),
            }
        )
        return OkResponse()

    return app


def _library_out(entry: LibraryEntry) -> LibraryBookOut:
    return LibraryBookOut(
        key=entry.key,
        name=entry.name,
        title=entry.title,
        size_bytes=entry.size_bytes,
        file_exists=entry.file_exists,
        in_rag=entry.in_rag,
        status=entry.status,
        book_id=entry.book_id,
        format=entry.format,
        pages=entry.pages,
        chunks=entry.chunks,
        error=entry.error,
        author=entry.author,
        topics=list(entry.topics),
        published_year=entry.published_year,
        published_year_source=entry.published_year_source,
        added_at=entry.added_at,
    )


def _job_out_required(job: LibraryJob | None) -> LibraryJobOut:
    result = _job_out(job)
    if result is None:
        raise HTTPException(status_code=500, detail="job is missing")
    return result


def _job_out(job: LibraryJob | None) -> LibraryJobOut | None:
    if job is None:
        return None
    return LibraryJobOut(
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


def _preview_out(preview: ImportPreview) -> ImportPreviewOut:
    return ImportPreviewOut(
        book_count=preview.book_count,
        pdf_count=preview.pdf_count,
        non_pdf_count=preview.non_pdf_count,
        total_bytes=preview.total_bytes,
        already_present=preview.already_present,
        skipped=preview.skipped,
        to_copy=preview.to_copy,
        batch_size=preview.batch_size,
        batch_bytes=preview.batch_bytes,
        remaining=preview.remaining,
        sample=list(preview.sample),
    )


def _book_out(book: Book) -> BookOut:
    return BookOut(
        id=book.id,
        title=book.title,
        author=book.author,
        pages=book.page_count,
        chunks=book.chunk_count,
        status=book.status.value,
        format=book.format,
    )


def _sse(event: dict[str, Any]) -> str:
    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"


def _hard_timeout_stream(
    settings: Settings,
    payload: ChatRequest,
    book_ids: Sequence[str] | None,
    history: Sequence[tuple[str, str]],
    started: float,
    *,
    request_id: str = "",
    on_done: Callable[[Answer, float], None] | None = None,
    on_error: Callable[[float], None] | None = None,
) -> Iterator[str]:
    """SSE из дочернего процесса: прогресс идёт, зависание убивается по таймауту."""
    try:
        for event in stream_answer_in_process(
            settings,
            payload.mode,
            payload.message,
            book_ids=book_ids,
            history=history,
        ):
            if event["type"] == "done":
                answer = event["answer"]
                response = _rag_response(answer, mode=payload.mode)
                elapsed_ms = int((perf_counter() - started) * 1000)
                response.elapsed_ms = elapsed_ms
                response.request_id = request_id
                if on_done is not None:
                    on_done(answer, float(elapsed_ms))
                yield _sse({"type": "done", "response": response.model_dump()})
            else:
                yield _sse(event)
    except Exception as exc:  # noqa: BLE001 - SSE обязан вернуть error-событие
        if on_error is not None:
            on_error(float((perf_counter() - started) * 1000))
        yield _sse({"type": "error", **_stream_error(exc)})


def _active_chat_model(settings: Settings) -> str:
    if settings.provider == "ollama":
        return settings.chat_model
    if settings.provider == "deepseek":
        return settings.deepseek_chat_model
    return settings.openai_chat_model


def _rag_response(
    answer: Answer,
    *,
    mode: Literal["rag", "advanced", "super", "agent"] = "rag",
) -> ChatResponse:
    return ChatResponse(
        mode=mode,
        answer=answer.text,
        refused=answer.refused,
        uncited=answer.uncited,
        unsupported_claims=list(answer.unsupported_claims),
        citations=[
            CitationOut(
                book_id=citation.book_id,
                book_title=citation.book_title,
                page=citation.printed_page_label or citation.pdf_page_index,
                chunk_id=citation.chunk_id,
                section=citation.section_path,
                format=citation.format,
                number=citation.number,
                quote=citation.quote,
            )
            for citation in answer.citations
        ],
        confidence=answer.confidence,
        route_reason=answer.route_reason,
        plan=list(answer.plan),
        rounds=answer.rounds,
        max_rounds=answer.max_rounds,
        elapsed_ms=0,
    )


__all__ = ["ChatBackend", "ChatRequest", "ChatResponse", "create_app"]
