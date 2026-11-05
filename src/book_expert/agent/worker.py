"""Жёсткий таймаут агента: выполнение в отдельном процессе.

ThreadPoolExecutor перестаёт ждать, но не останавливает LangGraph. Здесь задача
уходит в дочерний процесс, который по таймауту убивается вместе со всеми
сетевыми клиентами и инструментами.
"""

from __future__ import annotations

import multiprocessing as mp
from collections.abc import Callable, Sequence
from multiprocessing.connection import Connection
from typing import Any

from book_expert.agent.audit import AuditLogger
from book_expert.agent.graph import AgentError, AgentRunner, AgentTimeoutError
from book_expert.agent.tools.coding import build_coding_tools
from book_expert.agent.tools.rag import build_rag_tools
from book_expert.agent.trace import TraceRecorder
from book_expert.catalog import Catalog
from book_expert.config import Settings
from book_expert.providers import ProviderError, get_provider
from book_expert.retrieval.answer import GroundedAnswerer
from book_expert.retrieval.hybrid import HybridSearcher
from book_expert.retrieval.router import BookRouter


def run_in_process(
    target: Callable[..., None],
    args: Sequence[Any] = (),
    *,
    timeout: float,
    context: str = "spawn",
) -> dict[str, Any]:
    """Запускает target(*args, conn) и убивает процесс по таймауту."""
    if timeout <= 0:
        raise ValueError("timeout must be positive")
    ctx: Any = mp.get_context(context)
    parent, child = ctx.Pipe(duplex=False)
    process = ctx.Process(target=target, args=(*args, child), daemon=True)
    try:
        process.start()
    except Exception:
        parent.close()
        child.close()
        raise
    child.close()
    payload: dict[str, Any] | None = None
    try:
        if not parent.poll(timeout):
            raise AgentTimeoutError(f"agent exceeded {timeout:g}s")
        try:
            payload = parent.recv()
        except EOFError as exc:
            raise AgentError("agent worker exited unexpectedly") from exc
    finally:
        parent.close()
        if process.is_alive():
            _terminate(process)
        process.join(timeout=2)
    if payload is None:
        raise AgentError("agent worker returned no result")
    if "error" in payload:
        _raise_worker_error(payload["error"])
    return payload


def run_agent_in_process(
    settings: Settings,
    query: str,
    book_ids: Sequence[str] | None = None,
    history: Sequence[tuple[str, str]] | None = None,
    *,
    timeout: float | None = None,
) -> dict[str, Any]:
    """Выполняет агента в дочернем процессе под жёстким таймаутом."""
    return run_in_process(
        _agent_child,
        (
            settings,
            query,
            tuple(book_ids or ()),
            tuple(history or ()),
        ),
        timeout=settings.agent_timeout_s if timeout is None else timeout,
    )


def _agent_child(
    settings: Settings,
    query: str,
    book_ids: tuple[str, ...],
    history: tuple[tuple[str, str], ...],
    connection: Connection,
) -> None:
    provider = None
    searcher = None
    try:
        provider = get_provider(settings)
        catalog = Catalog(settings.catalog_path)
        catalog.initialize()
        searcher = HybridSearcher(settings, provider, catalog)
        router = BookRouter(settings, provider, catalog)
        answerer = GroundedAnswerer(settings, provider, searcher, router)
        audit = AuditLogger.create(settings)
        tools = build_rag_tools(searcher, answerer, audit=audit)
        tools.extend(build_coding_tools(settings, audit=audit))
        runner = AgentRunner(
            settings,
            tools,
            trace=TraceRecorder.create(settings),
        )
        result = runner.run(
            query,
            book_ids=list(book_ids) or None,
            history=list(history) or None,
            timeout=None,
        )
        connection.send(
            {
                "text": result.text,
                "steps": result.steps,
                "trace_path": str(result.trace_path),
                "usage": result.usage,
            }
        )
    except BaseException as exc:  # noqa: BLE001 - ошибка уходит родителю
        connection.send(
            {"error": {"type": type(exc).__name__, "message": str(exc)[:500]}}
        )
    finally:
        if searcher is not None:
            searcher.close()
        close = getattr(provider, "close", None)
        if callable(close):
            close()
        connection.close()


def _terminate(process: mp.Process) -> None:
    process.terminate()
    process.join(timeout=2)
    if process.is_alive():
        process.kill()
        process.join(timeout=1)


def _raise_worker_error(error: Any) -> None:
    if not isinstance(error, dict):
        raise AgentError("agent worker failed")
    message = str(error.get("message") or "agent worker failed")
    name = error.get("type")
    if name == "AgentTimeoutError":
        raise AgentTimeoutError(message)
    if name == "ProviderError":
        raise ProviderError(message)
    raise AgentError(message)


__all__ = ["run_agent_in_process", "run_in_process"]
