"""Оценка NLI-скоринга на human-cited и holdout (T97)."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any

from book_expert.eval.calibration import binary_metrics
from book_expert.eval.nli import DEFAULT_MODEL_DIR, NLICitationScorer

THRESHOLDS = (0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--model-file", default="onnx/model_quantized.onnx")
    parser.add_argument(
        "--output", type=Path, default=Path("data/artifacts/nli/results.json")
    )
    parser.add_argument(
        "--types",
        type=Path,
        default=Path("data/artifacts/claim-types/claim_types.csv"),
    )
    parser.add_argument("--cross-lingual", action="store_true")
    parser.add_argument(
        "--packet",
        type=Path,
        default=Path("data/artifacts/cross-lingual-nli/packet.json"),
    )
    parser.add_argument(
        "--labels",
        type=Path,
        default=Path("data/artifacts/cross-lingual-nli/labels.csv"),
    )
    parser.add_argument("--threshold", type=float)
    args = parser.parse_args()
    scorer = NLICitationScorer(args.model_dir, args.model_file)
    if args.cross_lingual:
        if args.threshold is None:
            parser.error("--cross-lingual требует --threshold")
        result = {
            "model": str(args.model_dir),
            "model_file": args.model_file,
            "threshold": args.threshold,
            **_cross_lingual(scorer, args.packet, args.labels, args.threshold),
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    types = _types(args.types)
    result = {
        "model": str(args.model_dir),
        "human_cited": _evaluate(
            scorer,
            Path("data/artifacts/t86-calibration/packet.json"),
            Path("data/artifacts/t86-human/labels.csv"),
            types,
            human=True,
        ),
        "holdout_agent": _evaluate(
            scorer,
            Path("data/artifacts/t96-holdout/packet.json"),
            Path("data/artifacts/t96-holdout/labels.csv"),
            types,
            human=False,
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


def _cross_lingual(
    scorer: NLICitationScorer,
    packet_path: Path,
    labels_path: Path,
    threshold: float,
) -> dict[str, Any]:
    packet = json.loads(packet_path.read_text(encoding="utf-8"))
    labels = _labels(labels_path)
    claims = [claim for claim in packet["claims"] if claim["id"] in labels]
    if not claims:
        raise SystemExit(f"нет размеченных claim: заполните label в {labels_path}")
    scored = []
    for claim in claims:
        excerpts = [
            item["text"] for item in claim.get("excerpts", []) if item.get("text")
        ]
        score = (
            max(scorer.score_many([(text, claim["text"]) for text in excerpts]))
            if excerpts
            else 0.0
        )
        scored.append(
            (
                labels[claim["id"]],
                score,
                claim["id"],
                claim.get("book_title", ""),
                claim.get("page"),
            )
        )
    pairs = [(gold, score >= threshold) for gold, score, _, _, _ in scored]
    sweep = {
        str(candidate): binary_metrics(
            [(gold, score >= candidate) for gold, score, _, _, _ in scored]
        )
        for candidate in THRESHOLDS
    }
    true_scores = [score for gold, score, _, _, _ in scored if gold]
    false_scores = [score for gold, score, _, _, _ in scored if not gold]
    return {
        "claims": len(scored),
        "gold_true": len(true_scores),
        "gold_false": len(false_scores),
        "mean_score_true": round(sum(true_scores) / len(true_scores), 4)
        if true_scores
        else None,
        "mean_score_false": round(sum(false_scores) / len(false_scores), 4)
        if false_scores
        else None,
        "metrics": binary_metrics(pairs),
        "sweep": sweep,
        "scores": [
            {
                "id": claim_id,
                "gold": gold,
                "score": round(score, 4),
                "book": book,
                "page": page,
            }
            for gold, score, claim_id, book, page in scored
        ],
    }


def _evaluate(
    scorer: NLICitationScorer,
    packet_path: Path,
    labels_path: Path,
    types: dict[str, str],
    *,
    human: bool,
) -> dict[str, Any]:
    packet = json.loads(packet_path.read_text(encoding="utf-8"))
    labels = _labels(labels_path)
    claims = [
        claim
        for claim in packet["claims"]
        if claim.get("cited") and _key(claim["id"]) in labels
    ]
    scored = []
    for claim in claims:
        excerpts = [
            item["text"] for item in claim.get("excerpts", []) if item.get("text")
        ]
        score = (
            max(scorer.score_many([(text, claim["text"]) for text in excerpts]))
            if excerpts
            else 0.0
        )
        scored.append(
            (
                labels[_key(claim["id"])],
                score,
                claim["id"],
                types.get(claim["id"], "unknown"),
            )
        )
    factual = [row for row in scored if row[3] == "factual"]
    sweep = {}
    factual_sweep = {}
    for threshold in THRESHOLDS:
        pairs = [(gold, score >= threshold) for gold, score, _, _ in scored]
        sweep[str(threshold)] = {
            **binary_metrics(pairs),
            "predicted_true": sum(1 for _, score, _, _ in scored if score >= threshold),
        }
        factual_pairs = [(gold, score >= threshold) for gold, score, _, _ in factual]
        factual_sweep[str(threshold)] = {
            **binary_metrics(factual_pairs),
            "predicted_true": sum(
                1 for _, score, _, _ in factual if score >= threshold
            ),
        }
    best = max(
        THRESHOLDS,
        key=lambda threshold: (
            sweep[str(threshold)].get("kappa", -1),
            sweep[str(threshold)].get("accuracy", -1),
        ),
    )
    best_factual = max(
        THRESHOLDS,
        key=lambda threshold: (
            factual_sweep[str(threshold)].get("kappa", -1),
            factual_sweep[str(threshold)].get("accuracy", -1),
        ),
    )
    return {
        "human_labels" if human else "agent_labels": len(labels),
        "claims": len(scored),
        "factual_claims": len(factual),
        "meta_claims": sum(1 for row in scored if row[3] == "meta"),
        "best_threshold": best,
        "best": sweep[str(best)],
        "best_threshold_factual": best_factual,
        "best_factual": factual_sweep[str(best_factual)],
        "sweep": sweep,
        "factual_sweep": factual_sweep,
        "scores": [
            {
                "id": claim_id,
                "gold": gold,
                "score": round(score, 4),
                "claim_type": claim_type,
            }
            for gold, score, claim_id, claim_type in scored
        ],
    }


def _sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?…])\s+|\n+", text)
    return [" ".join(part.split()) for part in parts if part.strip()]


def _types(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return {
            str(row["id"]): str(row["claim_type"])
            for row in csv.DictReader(handle, delimiter=";")
        }


def _labels(path: Path) -> dict[str, bool]:
    labels: dict[str, bool] = {}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        sample = handle.read(4096)
        handle.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=";,\t")
        except csv.Error:
            dialect = csv.excel
        for row in csv.DictReader(handle, dialect=dialect):
            raw = (row.get("label") or "").strip().lower()
            if raw not in {"true", "false", "1", "0", "yes", "no", "да", "нет"}:
                continue
            labels[_key(str(row["id"]))] = raw in {"true", "1", "yes", "да"}
    return labels


def _key(value: str) -> str:
    if value.startswith("c") and "_" in value:
        case, claim = value[1:].split("_", 1)
        return f"{case}:{claim}"
    if value.startswith("h") and "_" in value:
        case, claim = value[1:].split("_", 1)
        return f"{case}:{claim}"
    return value


if __name__ == "__main__":
    main()
