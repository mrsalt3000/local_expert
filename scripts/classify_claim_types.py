"""Разметка claim на фактические и мета/заголовочные (T98).

Эвристика, а не human review: нужна для проверки гипотезы «NLI плох из-за
мета-claim». Пишет только типы и claim-текст в gitignored ``data/artifacts``.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from book_expert.eval.claim_types import classify


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--packet",
        action="append",
        default=None,
        help="packet.json; можно несколько",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/artifacts/claim-types/claim_types.csv"),
    )
    args = parser.parse_args()
    packets = args.packet or [
        "data/artifacts/t86-calibration/packet.json",
        "data/artifacts/t96-holdout/packet.json",
    ]
    rows: list[dict[str, str]] = []
    for packet_path in packets:
        packet = json.loads(Path(packet_path).read_text(encoding="utf-8"))
        source = Path(packet_path).parent.name
        for claim in packet["claims"]:
            claim_type, reason = classify(claim["text"])
            rows.append(
                {
                    "id": claim["id"],
                    "source": source,
                    "claim_type": claim_type,
                    "reason": reason,
                    "claim": claim["text"],
                }
            )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["id", "source", "claim_type", "reason", "claim"],
            delimiter=";",
        )
        writer.writeheader()
        writer.writerows(rows)
    print(
        json.dumps({"rows": len(rows), "output": str(args.output)}, ensure_ascii=False)
    )


if __name__ == "__main__":
    main()
