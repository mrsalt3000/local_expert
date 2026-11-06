"""Жёсткий таймаут для super/advanced: ответ строится в отдельном процессе.

SSE-путь ``POST /api/chat/stream`` — основной путь UI. Раньше ответ считался в
daemon-Thread, который нельзя прервать: зависший провайдер держал запрос
бесконечно. Здесь тот же приём, что у веб-агента (``agent/worker.py``):
дочерний процесс шлёт progress-события через pipe, а по дедлайну родитель
убивает его вместе с сетевыми клиентами.
"""

from __future__ import annotations

import multiprocessing as mp
import time
from collections.abc import Iterator, Sequence
from dataclasses import replace
from typing import Any

from book_expert.agent.graph import AgentError, AgentTimeoutError
from book_expert.agent.trace import TraceRecorder
from book_expert.catalog import Catalog
from book_expert.config import Settings
from book_expert.models import Answer
from book_expert.providers import ProviderError, get_provider
from book_expert.providers.usage import MeteredProvider, finish_usage, start_usage
from book_expert.retrieval.advanced import AdvancedAnswerer
from book_expert.retrieval.hybrid import HybridSearcher
from book_expert.retrieval.router import BookRouter
from book_expert.retrieval.super_advanced import SuperAdvancedAnswerer


class AnswerTimeoutError(TimeoutError):
    """Super/advanced не уложились в отведённое время."""


class WorkerAnswerError(Exception):
    """Ошибка дочернего процесса: detail и статус уже вычислены там."""

    def __init__(self, detail: str, status: int) -> None:
        super().__init__(detail)
        self.status = status


def stream_error(exc: Exception) -> dict[str, Any]:
    """SSE-представление ошибки: detail + HTTP-подобный статус."""
    if isinstance(exc, WorkerAnswerError):
        return {"detail": str(exc), "status": exc.status}
    if isinstance(exc, ProviderError):
        return {"detail": str(exc), "status": 503}
    if isinstance(exc, ValueError):
        return {"detail": str(exc), "status": 400}
    if isinstance(exc, (AnswerTimeoutError, AgentTimeoutError)):
        return {"detail": str(exc), "status": 504}
    if isinstance(exc, AgentError):
        return {"detail": str(exc), "status": 502}
    return {"detail": "internal error", "status": 500}


def stream_answer_in_process(
    settings: Settings,
    mode: str,
    query: str,
    *,
    book_ids: Sequence[str] | None = None,
    history: Sequence[tuple[str, str]] | None = None,
    timeout: float | None = None,
    context: str = "spawn",
    provider: str | None = None,
    model: str | None = None,
) -> Iterator[dict[str, Any]]:
    """Yield progress-события, затем ``{"type": "done", "answer": Answer}``."""
    if provider and model:
        settings = settings.with_model(provider, model)
    limit = timeout if timeout is not None else settings.answer_timeout_s
    ctx: Any = mp.get_context(context)
    parent, child = ctx.Pipe(duplex=False)
    process = ctx.Process(
        target=_answer_child,
        args=(
            child,
            settings,
            mode,
            query,
            list(book_ids) if book_ids else None,
            list(history) if history else None,
        ),
        daemon=True,
    )
    try:
        process.start()
    except Exception:
        parent.close()
        child.close()
        raise
    child.close()
    deadline = time.monotonic() + limit
    completed = False
    try:
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise AnswerTimeoutError(f"answer exceeded {limit:g}s")
            if not parent.poll(remaining):
                raise AnswerTimeoutError(f"answer exceeded {limit:g}s")
            try:
                message = parent.recv()
            except EOFError as exc:
                raise WorkerAnswerError(
                    "answer worker exited unexpectedly", 500
                ) from exc
            kind = message.get("type")
            if kind == "event":
                yield message["event"]
            elif kind == "answer":
                completed = True
                yield {"type": "done", "answer": message["answer"]}
                return
            elif kind == "error":
                raise _error_from_message(message)
    finally:
        if completed:
            process.join(2)
        if process.is_alive():
            process.terminate()
            process.join(2)
            if process.is_alive():
                process.kill()
                process.join(1)
        parent.close()


def answer_in_process(
    settings: Settings,
    mode: str,
    query: str,
    *,
    book_ids: Sequence[str] | None = None,
    history: Sequence[tuple[str, str]] | None = None,
    timeout: float | None = None,
    context: str = "spawn",
    provider: str | None = None,
    model: str | None = None,
) -> Answer:
    """Блокирующий вариант для ``POST /api/chat``: прогресс отбрасывается."""
    answer = None
    for event in stream_answer_in_process(
        settings,
        mode,
        query,
        book_ids=book_ids,
        history=history,
        timeout=timeout,
        context=context,
        provider=provider,
        model=model,
    ):
        if event["type"] == "done":
            answer = event["answer"]
    if answer is None:
        raise WorkerAnswerError("answer worker returned no result", 500)
    return answer


def _answer_child(
    conn: Any,
    settings: Settings,
    mode: str,
    query: str,
    book_ids: list[str] | None,
    history: list[tuple[str, str]] | None,
) -> None:
    provider = None
    searcher = None
    usage_token = start_usage()
    try:
        settings.ensure_dirs()
        catalog = Catalog(settings.catalog_path)
        catalog.initialize()
        provider = MeteredProvider(get_provider(settings))
        searcher = HybridSearcher(settings, provider, catalog)
        router = BookRouter(settings, provider, catalog)
        trace = TraceRecorder.create(settings, prefix="web")
        answerer: AdvancedAnswerer | SuperAdvancedAnswerer
        if mode == "super":
            answerer = SuperAdvancedAnswerer(
                settings, provider, searcher, router, trace=trace
            )
        else:
            answerer = AdvancedAnswerer(
                settings, provider, searcher, router, trace=trace
            )

        def emit(event: dict[str, Any]) -> None:
            conn.send({"type": "event", "event": event})

        answer = answerer.ask(
            query,
            book_ids=book_ids,
            auto_route=True,
            progress=emit,
            history=history,
        )
        answer = replace(answer, usage=finish_usage(usage_token))
        conn.send({"type": "answer", "answer": answer})
    except Exception as exc:  # noqa: BLE001 - процесс обязан вернуть ошибку
        failure = stream_error(exc)
        conn.send(
            {
                "type": "error",
                "detail": failure["detail"],
                "status": failure["status"],
            }
        )
    finally:
        if searcher is not None:
            searcher.close()
        close = getattr(provider, "close", None)
        if callable(close):
            close()
        conn.close()


def _error_from_message(message: dict[str, Any]) -> Exception:
    return WorkerAnswerError(
        str(message.get("detail") or "answer failed"),
        int(message.get("status") or 500),
    )


__all__ = [
    "AnswerTimeoutError",
    "WorkerAnswerError",
    "answer_in_process",
    "stream_answer_in_process",
    "stream_error",
]
