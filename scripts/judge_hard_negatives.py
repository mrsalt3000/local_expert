"""Judge calibration on hard negatives.

Takes a local answers artifact (same shape as t72-cross-judge-40.local.json),
pairs each cited claim with its own source (positive) and with the least
lexically similar source from another book (negative), then reports how often
the judge accepts positives and rejects negatives.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import book_expert.providers.openai_compat as openai_compat
from book_expert.config import Settings
from book_expert.eval.faithfulness import FaithfulnessJudge
from book_expert.providers import get_provider

openai_compat.CHAT_TIMEOUT_S = 15.0


def tokens(text: str) -> set[str]:
    return set(re.findall(r"[а-яёa-z]{4,}", text.casefold()))


def jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def build_pairs(rows: list[dict], limit: int) -> tuple[list[dict], list[dict]]:
    items: list[dict[str, str]] = []
    claims: list[dict[str, str]] = []
    for row in rows:
        if row.get("refused"):
            continue
        sources = {item["n"]: item["text"] for item in row["sources"]}
        books = {c["number"]: c["book"] for c in row["citations"]}
        for item in row["sources"]:
            if item["n"] in books:
                items.append(
                    {
                        "claim": "",
                        "source": item["text"],
                        "book": books[item["n"]],
                    }
                )
        numbers = sorted(books)
        if not numbers:
            continue
        number = numbers[0]
        sentences = [
            part.strip()
            for part in re.split(r"(?<=[.!?])\s+", row["answer"])
            if part.strip()
        ]
        cited = [s for s in sentences if f"[{number}]" in s]
        if not cited:
            continue
        claims.append(
            {
                "question": row["question"],
                "claim": re.sub(r"\[\d+\]", "", cited[0]).strip(),
                "source": sources[number],
                "book": books[number],
            }
        )
    positives = claims[:limit]
    negatives: list[dict[str, str]] = []
    for claim in positives:
        claim_tokens = tokens(claim["claim"])
        best: tuple[float, dict[str, str]] | None = None
        for candidate in items:
            if candidate["book"] == claim["book"]:
                continue
            score = jaccard(claim_tokens, tokens(candidate["source"]))
            if best is None or score < best[0]:
                best = (score, candidate)
        if best is not None:
            negatives.append(
                {
                    "question": claim["question"],
                    "claim": claim["claim"],
                    "source": best[1]["source"],
                    "book": best[1]["book"],
                }
            )
    return positives, negatives


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--pairs", type=int, default=60)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    rows = json.loads(args.input.read_text(encoding="utf-8"))
    positives, negatives = build_pairs(rows, args.pairs)
    settings = Settings(provider="deepseek")
    provider = get_provider(settings)
    judge = FaithfulnessJudge(provider)

    def verdict(item: dict[str, str]) -> bool | None:
        for _ in range(3):
            report = judge.evaluate(
                item["question"], item["claim"] + " [1].", (item["source"],)
            )
            if report.error is None:
                return bool(report.claims[0].supported_by_cited)
            time.sleep(1)
        return None

    try:
        pos_results = [verdict(item) for item in positives]
        neg_results = [verdict(item) for item in negatives]
    finally:
        close = getattr(provider, "close", None)
        if callable(close):
            close()

    pos_ok = sum(1 for value in pos_results if value is True)
    neg_ok = sum(1 for value in neg_results if value is False)
    total_pos = sum(1 for value in pos_results if value is not None)
    total_neg = sum(1 for value in neg_results if value is not None)
    report = {
        "positives": total_pos,
        "positive_supported": pos_ok,
        "positive_rate": round(pos_ok / total_pos, 4) if total_pos else None,
        "negatives": total_neg,
        "negative_rejected": neg_ok,
        "negative_rate": round(neg_ok / total_neg, 4) if total_neg else None,
    }
    print(json.dumps(report, ensure_ascii=False, indent=1))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(
                {
                    "report": report,
                    "positives": pos_results,
                    "negatives": neg_results,
                },
                indent=1,
            ),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
