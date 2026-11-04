"""Лёгкие метрики веб-чата: uptime, отказы, ошибки, latency и SLO.

Счётчики живут в памяти и при наличии ``path`` дописываются в JSONL, поэтому
после рестарта запросы и окно latency восстанавливаются. Полный текст запросов
и ответов не пишется — только режим, задержка и флаги.
"""

from __future__ import annotations

import json
from collections import deque
from datetime import UTC, datetime
from pathlib import Path
from threading import Lock
from time import monotonic
from typing import Any

DEFAULT_WINDOW = 200

# mode -> (p50_ms, p95_ms, max_refusal_rate)
SLO: dict[str, tuple[int, int, float]] = {
    "rag": (2500, 5000, 0.15),
    "advanced": (4000, 8000, 0.15),
    "super": (8000, 15000, 0.20),
    "agent": (8000, 15000, 0.20),
}


class RequestMetrics:
    """Потокобезопасный счётчик с необязательным JSONL-журналом."""

    def __init__(self, window: int = DEFAULT_WINDOW, path: Path | None = None) -> None:
        if window <= 0:
            raise ValueError("window must be positive")
        self._lock = Lock()
        self._started = monotonic()
        self._path = Path(path) if path is not None else None
        self._window = window
        self._latencies: deque[float] = deque(maxlen=window)
        self._requests = 0
        self._refusals = 0
        self._errors = 0
        self._modes: dict[str, int] = {}
        self._mode_latencies: dict[str, deque[float]] = {}
        self._mode_refusals: dict[str, int] = {}
        self._mode_errors: dict[str, int] = {}
        self._load()

    def record(
        self,
        *,
        mode: str,
        elapsed_ms: float,
        refused: bool = False,
        error: bool = False,
    ) -> None:
        entry = {
            "recorded_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "mode": mode,
            "elapsed_ms": round(float(elapsed_ms), 1),
            "refused": bool(refused),
            "error": bool(error),
        }
        with self._lock:
            self._apply(entry)
            self._append(entry)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            latencies = sorted(self._latencies)
            by_mode = {}
            for mode in sorted(self._mode_latencies):
                values = sorted(self._mode_latencies[mode])
                requests = self._modes.get(mode, 0)
                refusals = self._mode_refusals.get(mode, 0)
                by_mode[mode] = {
                    "requests": requests,
                    "refusals": refusals,
                    "errors": self._mode_errors.get(mode, 0),
                    "refusal_rate": round(refusals / requests, 4) if requests else 0.0,
                    "latency_ms": {
                        "p50": _percentile(values, 50),
                        "p95": _percentile(values, 95),
                    },
                    "slo": _slo_status(mode, values, refusals, requests),
                }
            return {
                "ok": True,
                "uptime_s": round(monotonic() - self._started, 1),
                "requests": self._requests,
                "refusals": self._refusals,
                "errors": self._errors,
                "window": len(latencies),
                "latency_ms": {
                    "p50": _percentile(latencies, 50),
                    "p95": _percentile(latencies, 95),
                },
                "modes": dict(sorted(self._modes.items())),
                "by_mode": by_mode,
                "slo": {
                    "targets": {
                        mode: {
                            "p50_ms": target[0],
                            "p95_ms": target[1],
                            "max_refusal_rate": target[2],
                        }
                        for mode, target in SLO.items()
                    },
                    "persisted": self._path is not None,
                },
            }

    def _apply(self, entry: dict[str, Any]) -> None:
        mode = str(entry.get("mode") or "unknown")
        elapsed = float(entry.get("elapsed_ms") or 0.0)
        refused = bool(entry.get("refused"))
        error = bool(entry.get("error"))
        self._requests += 1
        self._latencies.append(elapsed)
        self._modes[mode] = self._modes.get(mode, 0) + 1
        self._mode_latencies.setdefault(mode, deque(maxlen=self._window)).append(
            elapsed
        )
        if refused:
            self._refusals += 1
            self._mode_refusals[mode] = self._mode_refusals.get(mode, 0) + 1
        if error:
            self._errors += 1
            self._mode_errors[mode] = self._mode_errors.get(mode, 0) + 1

    def _append(self, entry: dict[str, Any]) -> None:
        if self._path is None:
            return
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with self._path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except OSError:
            # Метрики не должны ломать запрос из-за диска.
            pass

    def _load(self) -> None:
        if self._path is None or not self._path.exists():
            return
        try:
            lines = self._path.read_text(encoding="utf-8").splitlines()
        except OSError:
            return
        for line in lines[-self._window * 20 :]:
            try:
                entry = json.loads(line)
            except (TypeError, ValueError):
                continue
            if isinstance(entry, dict):
                self._apply(entry)


def _slo_status(
    mode: str, latencies: list[float], refusals: int, requests: int
) -> dict[str, Any] | None:
    target = SLO.get(mode)
    if target is None or not requests:
        return None
    p50_target, p95_target, refusal_target = target
    p50 = _percentile(latencies, 50)
    p95 = _percentile(latencies, 95)
    refusal_rate = refusals / requests
    return {
        "p50_ok": p50 is not None and p50 <= p50_target,
        "p95_ok": p95 is not None and p95 <= p95_target,
        "refusal_ok": refusal_rate <= refusal_target,
        "all_ok": (
            p50 is not None
            and p50 <= p50_target
            and p95 is not None
            and p95 <= p95_target
            and refusal_rate <= refusal_target
        ),
    }


def _percentile(values: list[float], percent: int) -> int | None:
    if not values:
        return None
    index = min(len(values) - 1, round((percent / 100) * (len(values) - 1)))
    return int(values[index])


__all__ = ["DEFAULT_WINDOW", "SLO", "RequestMetrics"]
