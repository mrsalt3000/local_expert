"""Лёгкие метрики веб-чата: uptime, отказы, ошибки и latency."""

from __future__ import annotations

from collections import deque
from threading import Lock
from time import monotonic
from typing import Any

DEFAULT_WINDOW = 200


class RequestMetrics:
    """Потокобезопасный счётчик без внешних зависимостей."""

    def __init__(self, window: int = DEFAULT_WINDOW) -> None:
        if window <= 0:
            raise ValueError("window must be positive")
        self._lock = Lock()
        self._started = monotonic()
        self._latencies: deque[float] = deque(maxlen=window)
        self._requests = 0
        self._refusals = 0
        self._errors = 0
        self._modes: dict[str, int] = {}

    def record(
        self,
        *,
        mode: str,
        elapsed_ms: float,
        refused: bool = False,
        error: bool = False,
    ) -> None:
        with self._lock:
            self._requests += 1
            if refused:
                self._refusals += 1
            if error:
                self._errors += 1
            self._latencies.append(float(elapsed_ms))
            self._modes[mode] = self._modes.get(mode, 0) + 1

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            latencies = sorted(self._latencies)
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
            }


def _percentile(values: list[float], percent: int) -> int | None:
    if not values:
        return None
    index = min(len(values) - 1, round((percent / 100) * (len(values) - 1)))
    return int(values[index])


__all__ = ["DEFAULT_WINDOW", "RequestMetrics"]
