"""Оценка judge-профилей на holdout-пакете (T96)."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from book_expert.config import Settings
from book_expert.eval.calibration import binary_metrics
from book_expert.eval.faithfulness import FaithfulnessJudge
from book_expert.providers import get_provider


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--packet", type=Path, default=Path("data/artifacts/t96-holdout/packet.json")
    )
    parser.add_argument(
        "--context", type=Path, default=Path("data/artifacts/t96-holdout/context.json")
    )
    parser.add_argument(
        "--labels", type=Path, default=Path("data/artifacts/t96-holdout/labels.csv")
    )
    parser.add_argument("--provider", default="deepseek")
    parser.add_argument(
        "--output", type=Path, default=Path("data/artifacts/t96-holdout/agreement.json")
    )
    args = parser.parse_args()

    packet = json.loads(args.packet.read_text(encoding="utf-8"))
    context = json.loads(args.context.read_text(encoding="utf-8"))
    labels = _labels(args.labels)
    settings = Settings()
    client = get_provider(settings.with_provider(args.provider))
    result: dict[str, Any] = {
        "labels": len(labels),
        "provider": args.provider,
        "judges": {},
    }
    for name, options in (
        ("binary", {}),
        ("strict", {"strict": True}),
        ("strict_v2", {"strict_v2": True}),
    ):
        judge = FaithfulnessJudge(client, **options)
        verdicts: dict[str, bool | None] = {}
        errors: list[str] = []
        for answer in packet["cases"]:
            case = int(answer["case"])
            if answer["refused"]:
                continue
            sources = tuple(context["context"][str(case)])
            marker_map = {int(k): v for k, v in context["markers"][str(case)].items()}
            try:
                report = judge.evaluate(
                    answer["question"],
                    answer["answer"],
                    sources,
                    marker_map=marker_map,
                    citation_count=len(answer["citations"]),
                )
            except Exception as exc:
                errors.append(f"{case}: {exc}"[:300])
                continue
            if report.error:
                errors.append(f"{case}: {report.error}"[:300])
            for index, claim in enumerate(report.claims, start=1):
                verdicts[f"{case}:{index}"] = (
                    claim.supported_by_cited if claim.judged else None
                )
        pairs = [
            (labels[key], bool(verdicts[key]))
            for key in labels.keys() & verdicts.keys()
            if verdicts[key] is not None
        ]
        result["judges"][name] = {
            **binary_metrics(pairs),
            "errors": len(errors),
            "error_sample": errors[:3],
        }
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


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
            display = str(row["id"])
            key = (
                f"{display[1:].split('_', 1)[0]}:{display[1:].split('_', 1)[1]}"
                if display.startswith("h") and "_" in display
                else display
            )
            labels[key] = raw in {"true", "1", "yes", "да"}
    return labels


if __name__ == "__main__":
    main()
