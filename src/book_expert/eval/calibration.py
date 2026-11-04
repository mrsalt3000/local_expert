"""Метрики калибровки LLM-судьи по независимой разметке (T86)."""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any


def load_labels(path: Path) -> dict[str, bool]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw = payload.get("labels", payload)
    return {str(key): bool(value) for key, value in raw.items()}


def agreement(
    packet: dict[str, Any], judges: dict[str, Any], labels: dict[str, bool]
) -> dict[str, Any]:
    result: dict[str, Any] = {"claims": len(labels), "judges": {}}
    for name, data in judges["judges"].items():
        pairs = [
            (labels[key], bool(value))
            for key, value in data["verdicts"].items()
            if key in labels and value is not None
        ]
        result["judges"][name] = binary_metrics(pairs)
    names = list(result["judges"])
    for left in range(len(names)):
        for right in range(left + 1, len(names)):
            left_values = judges["judges"][names[left]]["verdicts"]
            right_values = judges["judges"][names[right]]["verdicts"]
            pairs = [
                (bool(left_values[key]), bool(right_values[key]))
                for key in left_values.keys() & right_values.keys()
                if left_values[key] is not None and right_values[key] is not None
            ]
            result.setdefault("inter_judge", {})[f"{names[left]}~{names[right]}"] = (
                binary_metrics(pairs)
            )
    return result


def binary_metrics(pairs: list[tuple[bool, bool]]) -> dict[str, Any]:
    if not pairs:
        return {"n": 0}
    tp = sum(1 for gold, pred in pairs if gold and pred)
    tn = sum(1 for gold, pred in pairs if not gold and not pred)
    fp = sum(1 for gold, pred in pairs if not gold and pred)
    fn = sum(1 for gold, pred in pairs if gold and not pred)
    accuracy = (tp + tn) / len(pairs)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    gold_pos = (tp + fn) / len(pairs)
    pred_pos = (tp + fp) / len(pairs)
    expected = gold_pos * pred_pos + (1 - gold_pos) * (1 - pred_pos)
    kappa = (accuracy - expected) / (1 - expected) if expected < 1 else 0.0
    return {
        "n": len(pairs),
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "kappa": round(kappa, 4),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def sample_claims(claims: list[dict[str, Any]], per_case: int) -> list[dict[str, Any]]:
    """Детерминированная выборка с перевесом в сторону цитируемых утверждений."""
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for claim in claims:
        grouped[claim["case"]].append(claim)
    rng = random.Random(42)
    selected: list[dict[str, Any]] = []
    for case_index in sorted(grouped):
        group = grouped[case_index]
        if len(group) <= per_case:
            selected.extend(group)
            continue
        cited = [claim for claim in group if claim["cited"]]
        uncited = [claim for claim in group if not claim["cited"]]
        cited_take = min(len(cited), max(1, per_case // 2 + per_case % 2))
        uncited_take = min(len(uncited), per_case - cited_take)
        if cited_take + uncited_take < per_case:
            cited_take = min(len(cited), per_case - uncited_take)
        chosen = rng.sample(cited, cited_take) + rng.sample(uncited, uncited_take)
        if len(chosen) < per_case:
            remaining = [claim for claim in group if claim not in chosen]
            chosen.extend(rng.sample(remaining, per_case - len(chosen)))
        selected.extend(
            sorted(chosen, key=lambda claim: int(str(claim["id"]).split(":")[1]))
        )
    return selected


__all__ = ["agreement", "binary_metrics", "load_labels", "sample_claims"]
