"""Holdout-пакет для строгого судьи: только cited-утверждения.

Берёт новые кейсы (по умолчанию offset 20), получает ответы «Агента», оставляет
только утверждения с цитатами и отдаёт 50 штук на слепую human-разметку.
Текст ответов и книг пишется только в gitignored ``data/artifacts``.

    uv run python scripts/build_holdout_packet.py
    uv run python scripts/build_holdout_packet.py --offset 40 --limit 20
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from book_expert.catalog import Catalog
from book_expert.config import Settings
from book_expert.eval.faithfulness import split_claims
from book_expert.eval.metrics import load_cases
from book_expert.providers import get_provider
from book_expert.retrieval.hybrid import HybridSearcher
from book_expert.retrieval.router import BookRouter
from book_expert.retrieval.super_advanced import SuperAdvancedAnswerer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="data/eval/combined-120.local.json")
    parser.add_argument("--offset", type=int, default=20)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--target", type=int, default=50)
    parser.add_argument("--per-case", type=int, default=4)
    parser.add_argument(
        "--output", type=Path, default=Path("data/artifacts/t96-holdout")
    )
    args = parser.parse_args()

    settings = Settings()
    settings.ensure_dirs()
    catalog = Catalog(settings.catalog_path)
    catalog.initialize()
    provider = get_provider(settings)
    searcher = HybridSearcher(settings, provider, catalog)
    router = BookRouter(settings, provider, catalog)
    answerer = SuperAdvancedAnswerer(settings, provider, searcher, router)
    cases = load_cases(args.dataset)[args.offset : args.offset + args.limit]
    rows: list[dict[str, Any]] = []
    claims: list[dict[str, Any]] = []
    context_payload: dict[str, dict[str, Any]] = {"context": {}, "markers": {}}
    try:
        for case_index, case in enumerate(cases, start=1):
            answer = answerer.ask(case.question, book_ids=None, auto_route=True)
            rows.append(
                {
                    "case": case_index,
                    "question": case.question,
                    "answer": answer.text,
                    "refused": answer.refused,
                    "citations": [
                        {
                            "number": citation.number,
                            "book_id": citation.book_id,
                            "page": citation.pdf_page_index,
                        }
                        for citation in answer.citations
                    ],
                }
            )
            if answer.refused:
                continue
            context = answer.context or answer.sources
            sources = tuple(hit.text for hit in context)
            index_by_chunk = {
                hit.chunk_id: position for position, hit in enumerate(context, start=1)
            }
            marker_map = {
                citation.number: index_by_chunk.get(citation.chunk_id, citation.number)
                for citation in answer.citations
            }
            context_payload["context"][str(case_index)] = list(sources)
            context_payload["markers"][str(case_index)] = {
                str(key): value for key, value in marker_map.items()
            }
            for claim_index, (text, cited) in enumerate(
                split_claims(answer.text), start=1
            ):
                if not cited:
                    continue
                mapped = tuple(marker_map.get(number, number) for number in cited)
                claims.append(
                    {
                        "id": f"h{case_index}_{claim_index}",
                        "judge_id": f"{case_index}:{claim_index}",
                        "case": case_index,
                        "question": case.question,
                        "text": text,
                        "cited": list(mapped),
                        "excerpts": [
                            {"marker": number, "text": sources[number - 1][:1600]}
                            for number in mapped
                            if 1 <= number <= len(sources)
                        ],
                    }
                )
    finally:
        if hasattr(answerer, "close"):
            answerer.close()

    selected = _sample(claims, args.per_case, args.target)
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    packet = {"claims": selected, "cases": rows}
    (output / "packet.json").write_text(
        json.dumps(packet, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output / "context.json").write_text(
        json.dumps(context_payload, ensure_ascii=False), encoding="utf-8"
    )
    (output / "README.md").write_text(_instructions(), encoding="utf-8")
    (output / "packet.md").write_text(_markdown(selected), encoding="utf-8")
    _write_labels(output / "labels.csv", selected)
    print(
        json.dumps({"claims": len(selected), "output": str(output)}, ensure_ascii=False)
    )


def _sample(
    claims: list[dict[str, Any]], per_case: int, target: int
) -> list[dict[str, Any]]:
    grouped: dict[int, list[dict[str, Any]]] = {}
    for claim in claims:
        grouped.setdefault(int(claim["case"]), []).append(claim)
    selected: list[dict[str, Any]] = []
    for case in sorted(grouped):
        selected.extend(grouped[case][:2])
    for case in sorted(grouped):
        if len(selected) >= target:
            break
        if len(grouped[case]) >= 3:
            selected.append(grouped[case][2])
    return selected[:target]


def _instructions() -> str:
    return """\
# Holdout для строгого судьи

50 cited-утверждений из новых ответов. В этом пакете **нет** строк без цитат:
negative control уже проверен в T95.

Заполните `label` (`true`/`false`) в `labels.csv`.
`true` — хотя бы один cited excerpt прямо и полностью подтверждает claim.
Используйте только excerpt; не открывайте judge-отчёты.
"""


def _write_labels(path: Path, claims: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["id", "label", "question", "claim", "cited", "excerpts"],
            delimiter=";",
        )
        writer.writeheader()
        for claim in claims:
            writer.writerow(
                {
                    "id": claim["id"],
                    "label": "",
                    "question": claim["question"],
                    "claim": claim["text"],
                    "cited": " ".join(str(number) for number in claim["cited"]),
                    "excerpts": "\n---\n".join(
                        f"[{item['marker']}] {item['text']}"
                        for item in claim["excerpts"]
                    ),
                }
            )


def _markdown(claims: list[dict[str, Any]]) -> str:
    lines = ["# Holdout: cited-утверждения", ""]
    for claim in claims:
        lines.extend(
            [
                f"## {claim['id']}",
                f"Вопрос: {claim['question']}",
                f"Утверждение: {claim['text']}",
            ]
        )
        for item in claim["excerpts"]:
            lines.append(f"- [{item['marker']}] {item['text']}")
        lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
