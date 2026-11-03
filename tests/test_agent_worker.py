"""Тесты жёсткого таймаута: процесс убивается, ошибки доходят до родителя."""

from __future__ import annotations

import time
from typing import Any

import pytest

from book_expert.agent.graph import AgentError, AgentTimeoutError
from book_expert.agent.worker import run_in_process


def _success(value: str, connection: Any) -> None:
    connection.send({"value": value})


def _fail(connection: Any) -> None:
    connection.send({"error": {"type": "AgentError", "message": "boom"}})


def _slow(connection: Any) -> None:
    time.sleep(5)
    connection.send({"value": "late"})


def test_run_in_process_returns_payload() -> None:
    payload = run_in_process(_success, ("ok",), timeout=5, context="fork")

    assert payload == {"value": "ok"}


def test_run_in_process_kills_on_timeout() -> None:
    started = time.monotonic()

    with pytest.raises(AgentTimeoutError, match="exceeded"):
        run_in_process(_slow, (), timeout=0.2, context="fork")

    assert time.monotonic() - started < 3


def test_run_in_process_maps_worker_error() -> None:
    with pytest.raises(AgentError, match="boom"):
        run_in_process(_fail, (), timeout=5, context="fork")


def test_run_in_process_rejects_non_positive_timeout() -> None:
    with pytest.raises(ValueError, match="positive"):
        run_in_process(_success, ("ok",), timeout=0, context="fork")
