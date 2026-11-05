"""Корреляция feedback с runtime-признаками качества (T104).

Соединяет `metrics.local.jsonl` и `feedback.local.jsonl` по `request_id` и
считает, связана ли оценка пользователя с числом неподдержанных утверждений
и цитат. Текст вопросов и ответов не читается и не выводится.
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

DEFAULT_METRICS = Path("data/artifacts/metrics.local.jsonl")
DEFAULT_FEEDBACK = Path("data/artifacts/feedback.local.jsonl")
DEFAULT_OUTPUT = Path("data/artifacts/feedback-correlation.json")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--feedback", type=Path, default=DEFAULT_FEEDBACK)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    metrics = _by_request_id(_read_jsonl(args.metrics))
    feedback = _read_jsonl(args.feedback)
    pairs = []
    skipped = 0
    for entry in feedback:
        request_id = str(entry.get("request_id") or "")
        metric = metrics.get(request_id)
        if not request_id or metric is None:
            skipped += 1
            continue
        citations = _int(metric.get("citations"))
        unsupported = _int(metric.get("unsupported_claims"))
        total = citations + unsupported
        pairs.append(
            {
                "request_id": request_id,
                "chat_id": entry.get("chat_id", ""),
                "mode": entry.get("mode", ""),
                "rating": 1.0 if entry.get("rating") == "up" else 0.0,
                "unsupported_claims": unsupported,
                "citations": citations,
                "unsupported_rate": round(unsupported / total, 4) if total else None,
                "tokens": metric.get("tokens", {}),
            }
        )

    usable = [pair for pair in pairs if pair["unsupported_rate"] is not None]
    up = [pair["unsupported_rate"] for pair in usable if pair["rating"] == 1.0]
    down = [pair["unsupported_rate"] for pair in usable if pair["rating"] == 0.0]
    result: dict[str, Any] = {
        "feedback_rows": len(feedback),
        "joined": len(pairs),
        "skipped_without_metrics": skipped,
        "usable": len(usable),
        "ratings": {"up": len(up), "down": len(down)},
        "mean_unsupported_rate": {
            "up": round(sum(up) / len(up), 4) if up else None,
            "down": round(sum(down) / len(down), 4) if down else None,
        },
        "point_biserial": round(
            _pearson(
                [pair["rating"] for pair in usable],
                [pair["unsupported_rate"] for pair in usable],
            ),
            4,
        )
        if len(usable) >= 2
        else None,
        "pairs": pairs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {key: value for key, value in result.items() if key != "pairs"},
            ensure_ascii=False,
            indent=2,
        )
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    rows = []
    for line in lines:
        try:
            value = json.loads(line)
        except (TypeError, ValueError):
            continue
        if isinstance(value, dict):
            rows.append(value)
    return rows


def _by_request_id(rows: Iterable[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    result: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        request_id = str(row.get("request_id") or "")
        if request_id:
            result[request_id] = row
    return result


def _int(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0
    return max(0, int(value))


def _pearson(xs: list[float], ys: list[float]) -> float:
    if len(xs) != len(ys) or len(xs) < 2:
        return 0.0
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    denominator = math.sqrt(
        sum((x - mean_x) ** 2 for x in xs) * sum((y - mean_y) ** 2 for y in ys)
    )
    return numerator / denominator if denominator else 0.0


__all__ = ["_pearson", "_by_request_id", "_read_jsonl", "main"]


if __name__ == "__main__":
    main()
