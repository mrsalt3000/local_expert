"""Тесты жёсткого таймаута super/advanced и SSE-ошибок."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import pytest

from book_expert.config import Settings
from book_expert.models import Answer
from book_expert.web import app as web_app
from book_expert.web.answer_worker import (
    AnswerTimeoutError,
    WorkerAnswerError,
    answer_in_process,
    stream_answer_in_process,
    stream_error,
)


def _settings(tmp_path: Path) -> Settings:
    return Settings(workspace=tmp_path, data_dir=tmp_path / "data")


def _stream_child(connection: Any, *_: Any) -> None:
    connection.send({"type": "event", "event": {"type": "plan", "queries": ["q"]}})
    connection.send(
        {"type": "answer", "answer": Answer(text="Ответ [1].", plan=("q",))}
    )


def _slow_child(connection: Any, *_: Any) -> None:
    time.sleep(5)


def _slow_child_with_pid(connection: Any, *_: Any) -> None:
    Path(os.environ["T61_PID_FILE"]).write_text(str(os.getpid()))
    time.sleep(5)


def _crash_child(connection: Any, *_: Any) -> None:
    connection.close()


def _error_child(connection: Any, *_: Any) -> None:
    connection.send({"type": "error", "detail": "boom", "status": 503})


def test_stream_forwards_progress_and_answer(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr("book_expert.web.answer_worker._answer_child", _stream_child)

    events = list(
        stream_answer_in_process(
            _settings(tmp_path), "super", "Вопрос", timeout=5, context="fork"
        )
    )

    assert events[0] == {"type": "plan", "queries": ["q"]}
    assert events[-1]["type"] == "done"
    assert events[-1]["answer"].text == "Ответ [1]."


def test_answer_in_process_returns_answer(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr("book_expert.web.answer_worker._answer_child", _stream_child)

    answer = answer_in_process(
        _settings(tmp_path), "advanced", "Вопрос", timeout=5, context="fork"
    )

    assert answer.text == "Ответ [1]."
    assert answer.plan == ("q",)


def test_stream_kills_child_on_timeout(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr("book_expert.web.answer_worker._answer_child", _slow_child)
    started = time.monotonic()

    with pytest.raises(AnswerTimeoutError, match="exceeded"):
        list(
            stream_answer_in_process(
                _settings(tmp_path), "advanced", "Вопрос", timeout=0.2, context="fork"
            )
        )

    assert time.monotonic() - started < 3


def test_stream_maps_worker_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr("book_expert.web.answer_worker._answer_child", _error_child)

    with pytest.raises(WorkerAnswerError, match="boom") as caught:
        list(
            stream_answer_in_process(
                _settings(tmp_path), "super", "Вопрос", timeout=5, context="fork"
            )
        )

    assert caught.value.status == 503
    assert stream_error(caught.value) == {"detail": "boom", "status": 503}


def test_stream_maps_worker_crash_to_500(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr("book_expert.web.answer_worker._answer_child", _crash_child)

    with pytest.raises(WorkerAnswerError) as caught:
        list(
            stream_answer_in_process(
                _settings(tmp_path), "advanced", "Вопрос", timeout=5, context="fork"
            )
        )

    assert caught.value.status == 500
    assert "exited unexpectedly" in str(caught.value)


def test_timeout_kills_and_reaps_child(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    pid_file = tmp_path / "child.pid"
    monkeypatch.setenv("T61_PID_FILE", str(pid_file))
    monkeypatch.setattr(
        "book_expert.web.answer_worker._answer_child", _slow_child_with_pid
    )

    with pytest.raises(AnswerTimeoutError):
        list(
            stream_answer_in_process(
                _settings(tmp_path), "advanced", "Вопрос", timeout=0.3, context="fork"
            )
        )

    child_pid = int(pid_file.read_text())
    with pytest.raises(ProcessLookupError):
        os.kill(child_pid, 0)


def test_timeout_maps_to_504() -> None:
    assert stream_error(AnswerTimeoutError("slow")) == {
        "detail": "slow",
        "status": 504,
    }


def test_hard_timeout_stream_emits_504(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def boom(*_: Any, **__: Any):
        raise AnswerTimeoutError("slow")
        yield  # pragma: no cover

    monkeypatch.setattr(web_app, "stream_answer_in_process", boom)
    payload = web_app.ChatRequest(message="Вопрос", mode="super")

    chunks = list(
        web_app._hard_timeout_stream(_settings(tmp_path), payload, None, [], 0.0)
    )

    assert len(chunks) == 1
    event = json.loads(chunks[0].removeprefix("data: "))
    assert event == {"type": "error", "detail": "slow", "status": 504}
