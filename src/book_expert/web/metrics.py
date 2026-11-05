"""Лёгкие метрики веб-чата: uptime, отказы, ошибки, latency, tokens и cost.

Счётчики живут в памяти и при наличии ``path`` дописываются в JSONL, поэтому
после рестарта запросы, окно latency и расход токенов восстанавливаются.
Полный текст запросов и ответов не пишется — только режим, задержка, флаги,
число токенов и стоимость.
"""

from __future__ import annotations

import json
from collections import deque
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from threading import Lock
from time import monotonic
from typing import Any

DEFAULT_WINDOW = 200
MILLION = 1_000_000

# mode -> (p50_ms, p95_ms, max_refusal_rate)
SLO: dict[str, tuple[int, int, float]] = {
    "rag": (2500, 5000, 0.15),
    "advanced": (4000, 8000, 0.15),
    "super": (8000, 15000, 0.20),
    "agent": (8000, 15000, 0.20),
}


class RequestMetrics:
    """Потокобезопасный счётчик с необязательным JSONL-журналом."""

    def __init__(
        self,
        window: int = DEFAULT_WINDOW,
        path: Path | None = None,
        *,
        price_input_per_mtok: float = 0.0,
        price_output_per_mtok: float = 0.0,
    ) -> None:
        if window <= 0:
            raise ValueError("window must be positive")
        self._lock = Lock()
        self._started = monotonic()
        self._path = Path(path) if path is not None else None
        self._window = window
        self._price_input = max(0.0, float(price_input_per_mtok))
        self._price_output = max(0.0, float(price_output_per_mtok))
        self._latencies: deque[float] = deque(maxlen=window)
        self._requests = 0
        self._refusals = 0
        self._errors = 0
        self._tokens_prompt = 0
        self._tokens_completion = 0
        self._cost = 0.0
        self._usage_recorded = 0
        self._modes: dict[str, int] = {}
        self._mode_latencies: dict[str, deque[float]] = {}
        self._mode_refusals: dict[str, int] = {}
        self._mode_errors: dict[str, int] = {}
        self._mode_tokens: dict[str, dict[str, int]] = {}
        self._mode_cost: dict[str, float] = {}
        self._providers: dict[str, dict[str, Any]] = {}
        self._load()

    def record(
        self,
        *,
        mode: str,
        elapsed_ms: float,
        refused: bool = False,
        error: bool = False,
        usage: Mapping[str, Any] | None = None,
        provider: str = "",
        model: str = "",
        request_id: str = "",
        citations: int = 0,
        unsupported_claims: int = 0,
    ) -> None:
        tokens = _normalize_usage(usage)
        cost = self.cost(tokens)
        entry: dict[str, Any] = {
            "recorded_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "mode": mode,
            "elapsed_ms": round(float(elapsed_ms), 1),
            "refused": bool(refused),
            "error": bool(error),
            "provider": provider,
            "model": model,
            "request_id": request_id,
            "citations": int(citations),
            "unsupported_claims": int(unsupported_claims),
        }
        if tokens:
            entry["tokens"] = tokens
            entry["cost_usd"] = round(cost, 8)
        with self._lock:
            self._apply(entry)
            self._append(entry)

    def cost(self, usage: Mapping[str, int] | None) -> float:
        if not usage:
            return 0.0
        prompt = int(usage.get("prompt_tokens") or 0)
        completion = int(usage.get("completion_tokens") or 0)
        return (
            prompt / MILLION * self._price_input
            + completion / MILLION * self._price_output
        )

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
                    "tokens": dict(self._mode_tokens.get(mode, {})),
                    "cost_usd": round(self._mode_cost.get(mode, 0.0), 8),
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
                "tokens": {
                    "prompt": self._tokens_prompt,
                    "completion": self._tokens_completion,
                    "total": self._tokens_prompt + self._tokens_completion,
                },
                "cost_usd": round(self._cost, 8),
                "usage_coverage": round(self._usage_recorded / self._requests, 4)
                if self._requests
                else 0.0,
                "modes": dict(sorted(self._modes.items())),
                "by_mode": by_mode,
                "by_provider": {
                    name: {
                        **values,
                        "cost_usd": round(values["cost_usd"], 8),
                    }
                    for name, values in sorted(self._providers.items())
                },
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
        tokens = _normalize_usage(entry.get("tokens"))
        cost = float(entry.get("cost_usd") or 0.0)
        provider = str(entry.get("provider") or "unknown")
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
        if tokens:
            self._usage_recorded += 1
            self._tokens_prompt += tokens["prompt_tokens"]
            self._tokens_completion += tokens["completion_tokens"]
            self._cost += cost
            mode_tokens = self._mode_tokens.setdefault(
                mode, {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            )
            merge_tokens(mode_tokens, tokens)
            self._mode_cost[mode] = self._mode_cost.get(mode, 0.0) + cost
        provider_stats = self._providers.setdefault(
            provider,
            {
                "requests": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "cost_usd": 0.0,
            },
        )
        provider_stats["requests"] += 1
        if tokens:
            merge_tokens(provider_stats, tokens)
            provider_stats["cost_usd"] += cost

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


def _normalize_usage(usage: Mapping[str, Any] | None) -> dict[str, int]:
    if not usage:
        return {}
    prompt = _to_int(usage.get("prompt_tokens", usage.get("input_tokens")))
    completion = _to_int(usage.get("completion_tokens", usage.get("output_tokens")))
    total = _to_int(usage.get("total_tokens")) or prompt + completion
    if not (prompt or completion or total):
        return {}
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": total,
    }


def _to_int(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0
    return max(0, int(value))


def merge_tokens(target: dict[str, int], usage: Mapping[str, int]) -> None:
    target["prompt_tokens"] = target.get("prompt_tokens", 0) + usage["prompt_tokens"]
    target["completion_tokens"] = (
        target.get("completion_tokens", 0) + usage["completion_tokens"]
    )
    target["total_tokens"] = target.get("total_tokens", 0) + usage["total_tokens"]


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


__all__ = ["DEFAULT_WINDOW", "SLO", "RequestMetrics", "merge_tokens"]
