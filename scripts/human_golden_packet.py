"""Слепой human-golden пакет для калибровки судьи (T95).

Пакет и разметка лежат только в ``--output`` (по умолчанию проектная папка ``data/artifacts``): в них
есть фрагменты ответов и книг, поэтому в git они не попадают.

    uv run python scripts/human_golden_packet.py \
      --packet data/artifacts/t86-calibration/packet.json \
      --output data/artifacts/t86-human --per-case 6

Заполните ``labels.csv`` (колонка ``label``: true/false) и запустите:

    uv run python scripts/human_golden_packet.py \
      --labels data/artifacts/t86-human/labels.csv \
      --agent-labels data/artifacts/t86-calibration/labels.json \
      --judges data/artifacts/t86-calibration/deepseek data/artifacts/t86-calibration/ollama data/artifacts/t86-calibration/quote
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

В `labels.csv` уже проставлен `false` для строк без цитат (57 из 114):
это негативный контроль, размечать их вручную не нужно. Заполните `label`
только в 57 строках, где `cited` не пустой: `true` или `false`.

Правило: `true` только если хотя бы один из cited excerpts **прямо**
подтверждает утверждение. Если excerpt лишь по теме, частичный или добавляет
детали, которых в нём нет — `false`.

Не открывайте `judges.json`, `agreement.json` и файлы с вердиктами судьи:
разметка должна быть независимой. Смотрите только `question`, `claim` и
`excerpts`.
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--packet",
        type=Path,
        default=Path("data/artifacts/t86-calibration/packet.json"),
    )
    parser.add_argument("--output", type=Path, default=Path("data/artifacts/t86-human"))
    parser.add_argument("--per-case", type=int, default=6)
    parser.add_argument("--labels", type=Path)
    parser.add_argument("--agent-labels", type=Path)
    parser.add_argument("--judges", nargs="*", type=Path, default=[])
    args = parser.parse_args()

    if args.labels:
        packet = json.loads(args.packet.read_text(encoding="utf-8"))
        human = _load_csv_labels(args.labels)
        agent = load_labels(args.agent_labels) if args.agent_labels else None
        judge_payloads = {}
        for directory in args.judges:
            path = directory / "judges.json"
            if path.exists():
                judge_payloads[str(directory)] = json.loads(
                    path.read_text(encoding="utf-8")
                )
        result = _report(packet, human, agent, judge_payloads)
        cited_ids = {claim["id"] for claim in packet["claims"] if claim.get("cited")}
        result["cited_only"] = _report(
            packet,
            {key: value for key, value in human.items() if key in cited_ids},
            agent,
            judge_payloads,
        )
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


def _report(
    packet: dict[str, Any],
    human: dict[str, bool],
    agent: dict[str, bool] | None,
    judge_payloads: dict[str, Any],
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "human_claims": len(human),
        "agent": None,
        "judges": {},
    }
    if agent:
        pairs = [(agent[key], human[key]) for key in human.keys() & agent.keys()]
        result["agent"] = binary_metrics(pairs)
    result["judges"] = {
        name: agreement(packet, payload, human)["judges"]
        for name, payload in judge_payloads.items()
    }
    return result


def _load_csv_labels(path: Path) -> dict[str, bool]:
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
            if raw in {"true", "1", "yes", "да"}:
                labels[str(row["id"])] = True
            elif raw in {"false", "0", "no", "нет"}:
                labels[str(row["id"])] = False
    return labels


def _write_csv(path: Path, claims: list[dict[str, Any]]) -> None:
    # Excel корректно открывает UTF-8 с BOM и разделителем ;.
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["id", "label", "question", "claim", "cited", "excerpts"],
            delimiter=";",
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
                    "label": "false" if not claim["cited"] else "",
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
