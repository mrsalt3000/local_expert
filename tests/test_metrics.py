"""Тесты персистентных метрик и SLO (T90)."""

from __future__ import annotations

import json
from pathlib import Path

from book_expert.web.metrics import RequestMetrics


def test_metrics_persist_across_restarts(tmp_path: Path) -> None:
    path = tmp_path / "metrics.local.jsonl"
    first = RequestMetrics(path=path)
    first.record(mode="rag", elapsed_ms=1200)
    first.record(mode="super", elapsed_ms=9000, refused=True)

    second = RequestMetrics(path=path)
    snapshot = second.snapshot()

    assert snapshot["requests"] == 2
    assert snapshot["refusals"] == 1
    assert snapshot["by_mode"]["rag"]["latency_ms"]["p50"] == 1200
    assert snapshot["by_mode"]["super"]["refusal_rate"] == 1.0
    assert snapshot["slo"]["persisted"] is True


def test_metrics_slo_marks_slow_rag_as_failed(tmp_path: Path) -> None:
    metrics = RequestMetrics(path=tmp_path / "metrics.jsonl")
    metrics.record(mode="rag", elapsed_ms=6000)

    status = metrics.snapshot()["by_mode"]["rag"]["slo"]

    assert status["p50_ok"] is False
    assert status["p95_ok"] is False
    assert status["all_ok"] is False


def test_metrics_skip_corrupt_jsonl_lines(tmp_path: Path) -> None:
    path = tmp_path / "metrics.jsonl"
    path.write_text(
        "not json\n"
        + json.dumps(
            {"mode": "rag", "elapsed_ms": 1000, "refused": False, "error": False}
        )
        + "\n",
        encoding="utf-8",
    )

    snapshot = RequestMetrics(path=path).snapshot()

    assert snapshot["requests"] == 1
    assert snapshot["by_mode"]["rag"]["requests"] == 1


def test_metrics_file_has_no_request_text(tmp_path: Path) -> None:
    path = tmp_path / "metrics.jsonl"
    RequestMetrics(path=path).record(mode="rag", elapsed_ms=42)

    entry = json.loads(path.read_text(encoding="utf-8"))

    assert set(entry) == {"recorded_at", "mode", "elapsed_ms", "refused", "error"}
