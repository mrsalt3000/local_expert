"""Слепой human-golden пакет для калибровки судьи (T95).

Пакет и разметка лежат только в ``--output`` (по умолчанию ``/tmp``): в них
есть фрагменты ответов и книг, поэтому в git они не попадают.

    uv run python scripts/human_golden_packet.py \
      --packet /tmp/t86-calibration/packet.json \
      --output /tmp/t86-human --per-case 6

Заполните ``labels.csv`` (колонка ``label``: true/false) и запустите:

    uv run python scripts/human_golden_packet.py \
      --labels /tmp/t86-human/labels.csv \
      --agent-labels /tmp/t86-calibration/labels.json \
      --judges /tmp/t86-deepseek /tmp/t86-ollama-full /tmp/t86-quote-full
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from book_expert.eval.calibration import (
    agreement,
    binary_metrics,
    load_labels,
    sample_claims,
)

INSTRUCTIONS = """\
# Слепая разметка claim-level

Заполните колонку `label` в `labels.csv`: `true` или `false`.

Правило: `true` только если хотя бы один из cited excerpts **прямо**
подтверждает утверждение. Если excerpt лишь по теме, частичный или добавляет
детали, которых в нём нет — `false`. Если цитат нет — `false`.

Не открывайте `judges.json`, `agreement.json` и файлы с вердиктами судьи:
разметка должна быть независимой. Смотрите только `question`, `claim` и
`excerpts`.
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--packet", type=Path, default=Path("/tmp/t86-calibration/packet.json")
    )
    parser.add_argument("--output", type=Path, default=Path("/tmp/t86-human"))
    parser.add_argument("--per-case", type=int, default=6)
    parser.add_argument("--labels", type=Path)
    parser.add_argument("--agent-labels", type=Path)
    parser.add_argument("--judges", nargs="*", type=Path, default=[])
    args = parser.parse_args()

    if args.labels:
        packet = json.loads(args.packet.read_text(encoding="utf-8"))
        human = _load_csv_labels(args.labels)
        result: dict[str, Any] = {
            "human_claims": len(human),
            "agent": None,
            "judges": {},
        }
        if args.agent_labels:
            agent = load_labels(args.agent_labels)
            pairs = [(agent[key], human[key]) for key in human.keys() & agent.keys()]
            result["agent"] = binary_metrics(pairs)
        judge_payloads = {}
        for directory in args.judges:
            path = directory / "judges.json"
            if path.exists():
                judge_payloads[str(directory)] = json.loads(
                    path.read_text(encoding="utf-8")
                )
        result["judges"] = {
            name: agreement(packet, payload, human)["judges"]
            for name, payload in judge_payloads.items()
        }
        args.output.mkdir(parents=True, exist_ok=True)
        path = args.output / "agreement.json"
        path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    packet = json.loads(args.packet.read_text(encoding="utf-8"))
    claims = sample_claims(packet["claims"], args.per_case)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "README.md").write_text(INSTRUCTIONS, encoding="utf-8")
    (args.output / "packet.md").write_text(_markdown(claims), encoding="utf-8")
    _write_csv(args.output / "labels.csv", claims)
    (args.output / "labels.template.json").write_text(
        json.dumps({"labels": {}}, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"packet: {args.output} ({len(claims)} claims)")


def _load_csv_labels(path: Path) -> dict[str, bool]:
    labels: dict[str, bool] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            raw = (row.get("label") or "").strip().lower()
            if raw in {"true", "1", "yes", "да"}:
                labels[str(row["id"])] = True
            elif raw in {"false", "0", "no", "нет"}:
                labels[str(row["id"])] = False
    return labels


def _write_csv(path: Path, claims: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["id", "label", "question", "claim", "cited", "excerpts"],
        )
        writer.writeheader()
        for claim in claims:
            excerpts = "\n---\n".join(
                f"[{excerpt['marker']}] {excerpt['text']}"
                for excerpt in claim["excerpts"]
            )
            writer.writerow(
                {
                    "id": claim["id"],
                    "label": "",
                    "question": claim["question"],
                    "claim": claim["text"],
                    "cited": " ".join(str(number) for number in claim["cited"]),
                    "excerpts": excerpts,
                }
            )


def _markdown(claims: list[dict[str, Any]]) -> str:
    lines = ["# Слепой пакет утверждений", ""]
    for claim in claims:
        lines.extend(
            [
                f"## {claim['id']}",
                f"Вопрос: {claim['question']}",
                f"Утверждение: {claim['text']}",
            ]
        )
        if claim["excerpts"]:
            for excerpt in claim["excerpts"]:
                lines.append(f"- [{excerpt['marker']}] {excerpt['text']}")
        else:
            lines.append("Цитат нет — ожидается false.")
        lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
