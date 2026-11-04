"""Калибровка LLM-судьи по независимой разметке (T86).

Скрипт собирает ответы «Агента», режет их на утверждения, сохраняет слепой
пакет для разметчика и прогоняет несколько вариантов судьи на тех же данных.
Полный текст книг и ответов пишется только в указанный каталог (по умолчанию
``/tmp/t86-calibration``), а не в репозиторий.

    uv run python scripts/judge_calibration.py --limit 20
    uv run python scripts/judge_calibration.py --labels /tmp/t86/labels.json

Пакет: ``packet.json`` / ``packet.md``. Разметчик заполняет ``labels.json``:
``{"labels": {"1:1": true, ...}}``. После этого повторный запуск с ``--labels``
считает accuracy/precision/recall и Cohen's kappa для каждого судьи.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from book_expert.catalog import Catalog
from book_expert.config import Settings
from book_expert.eval.calibration import (
    agreement,
    load_labels,
    sample_claims,
)
from book_expert.eval.faithfulness import FaithfulnessJudge, split_claims
from book_expert.eval.metrics import load_cases
from book_expert.providers import get_provider
from book_expert.retrieval.hybrid import HybridSearcher
from book_expert.retrieval.router import BookRouter
from book_expert.retrieval.super_advanced import SuperAdvancedAnswerer

MAX_EXCERPT = 1600


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="data/eval/combined-120.local.json")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument(
        "--claims-per-case",
        type=int,
        default=8,
        help="сколько утверждений на ответ попадает в слепой пакет (0 = все)",
    )
    parser.add_argument(
        "--judge-providers", default="deepseek,ollama", help="через запятую"
    )
    parser.add_argument(
        "--judge-modes", default="binary,quote", help="binary,quote или оба"
    )
    parser.add_argument(
        "--judge-answer-limit",
        type=int,
        default=0,
        help="ограничить число ответов для судьи (0 = все)",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("/tmp/t86-calibration"))
    parser.add_argument("--labels", type=Path, help="JSON с независимой разметкой")
    parser.add_argument("--packet-only", action="store_true")
    parser.add_argument("--judges-only", action="store_true")
    args = parser.parse_args()

    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    packet_path = output / "packet.json"
    if args.judges_only:
        packet = json.loads(packet_path.read_text(encoding="utf-8"))
        judges = _run_judges(args, packet)
        (output / "judges.json").write_text(
            json.dumps(judges, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(json.dumps(_judge_summary(judges), ensure_ascii=False, indent=2))
        return
    if args.labels:
        packet = json.loads(packet_path.read_text(encoding="utf-8"))
        judges = json.loads((output / "judges.json").read_text(encoding="utf-8"))
        labels = load_labels(args.labels)
        summary = agreement(packet, judges, labels)
        (output / "agreement.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return

    packet, _ = _build_packet(args)
    packet_path.write_text(
        json.dumps(packet, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output / "packet.md").write_text(_packet_markdown(packet), encoding="utf-8")
    print(f"packet: {packet_path} ({len(packet['claims'])} claims)")
    if args.packet_only:
        return

    judges = _run_judges(args, packet)
    (output / "judges.json").write_text(
        json.dumps(judges, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(_judge_summary(judges), ensure_ascii=False, indent=2))


def _build_packet(args: argparse.Namespace) -> tuple[dict[str, Any], list[Any]]:
    settings = Settings()
    settings.ensure_dirs()
    catalog = Catalog(settings.catalog_path)
    catalog.initialize()
    provider = get_provider(settings)
    searcher = HybridSearcher(settings, provider, catalog)
    router = BookRouter(settings, provider, catalog)
    answerer = SuperAdvancedAnswerer(settings, provider, searcher, router)
    cases = load_cases(args.dataset)[args.offset : args.offset + args.limit]

    claims: list[dict[str, Any]] = []
    answer_rows: list[dict[str, Any]] = []
    context_payload: dict[str, dict[str, Any]] = {"context": {}, "markers": {}}
    try:
        for case_index, case in enumerate(cases, start=1):
            answer = answerer.ask(case.question, book_ids=None, auto_route=True)
            answer_rows.append(
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
            context_index = {
                hit.chunk_id: position for position, hit in enumerate(context, start=1)
            }
            marker_map = {
                citation.number: context_index.get(citation.chunk_id, citation.number)
                for citation in answer.citations
            }
            context_payload["context"][str(case_index)] = list(sources)
            context_payload["markers"][str(case_index)] = {
                str(key): value for key, value in marker_map.items()
            }
            raw_claims = split_claims(answer.text)
            for claim_index, (text, cited) in enumerate(raw_claims, start=1):
                mapped = tuple(marker_map.get(number, number) for number in cited)
                excerpts = [
                    {
                        "marker": number,
                        "text": sources[number - 1][:MAX_EXCERPT],
                    }
                    for number in mapped
                    if 1 <= number <= len(sources)
                ]
                claims.append(
                    {
                        "id": f"{case_index}:{claim_index}",
                        "case": case_index,
                        "question": case.question,
                        "text": text,
                        "cited": list(mapped),
                        "excerpts": excerpts,
                    }
                )
    finally:
        answerer.close() if hasattr(answerer, "close") else None
    (args.output_dir / "context.json").write_text(
        json.dumps(context_payload, ensure_ascii=False), encoding="utf-8"
    )
    if args.claims_per_case:
        claims = sample_claims(claims, args.claims_per_case)
    return (
        {
            "instructions": (
                "Разметь claim-level: supported_by_cited=true только если хотя бы "
                "один из cited excerpts подтверждает утверждение. Без цитат — false."
            ),
            "claims": claims,
            "cases": answer_rows,
        },
        cases,
    )


def _run_judges(args: argparse.Namespace, packet: dict[str, Any]) -> dict[str, Any]:
    settings = Settings()
    context_by_case: dict[int, tuple[str, ...]] = {}
    marker_by_case: dict[int, dict[int, int]] = {}
    # Rebuild judge inputs from the packet is impossible without full context;
    # the packet keeps only cited excerpts, so re-run the answerer is avoided by
    # storing context in a sidecar written by _build_packet.
    sidecar = args.output_dir / "context.json"
    if not sidecar.exists():
        raise SystemExit(
            "context.json missing: run without --labels first in the same process"
        )
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    context_by_case = {
        int(key): tuple(value) for key, value in payload["context"].items()
    }
    marker_by_case = {
        int(key): {int(k): v for k, v in value.items()}
        for key, value in payload["markers"].items()
    }

    providers = [
        part.strip() for part in args.judge_providers.split(",") if part.strip()
    ]
    modes = [part.strip() for part in args.judge_modes.split(",") if part.strip()]
    answers = packet["cases"]
    if args.judge_answer_limit:
        answers = answers[: args.judge_answer_limit]
    judges: dict[str, Any] = {}
    for provider_name in providers:
        client = get_provider(settings.with_provider(provider_name))
        for mode in modes:
            if mode not in {"binary", "quote"}:
                continue
            name = f"{provider_name}/{mode}"
            judge = FaithfulnessJudge(client, quote_mode=(mode == "quote"))
            verdicts: dict[str, bool | None] = {}
            errors: list[str] = []
            for answer in answers:
                case_index = int(answer["case"])
                if answer["refused"]:
                    continue
                try:
                    report = judge.evaluate(
                        answer["question"],
                        answer["answer"],
                        context_by_case.get(case_index, ()),
                        marker_map=marker_by_case.get(case_index),
                        citation_count=len(answer["citations"]),
                    )
                except Exception as exc:  # judge boundary; report and continue
                    errors.append(f"case {case_index}: {exc}"[:300])
                    continue
                if report.error:
                    errors.append(f"case {case_index}: {report.error}"[:300])
                for claim_index, claim in enumerate(report.claims, start=1):
                    verdicts[f"{case_index}:{claim_index}"] = (
                        claim.supported_by_cited if claim.judged else None
                    )
            judges[name] = {"verdicts": verdicts, "errors": errors}
    return {"judges": judges}


def _judge_summary(judges: dict[str, Any]) -> dict[str, Any]:
    summary = {}
    for name, data in judges["judges"].items():
        values = [value for value in data["verdicts"].values() if value is not None]
        summary[name] = {
            "claims": len(values),
            "supported": sum(values),
            "errors": len(data["errors"]),
        }
    return summary


def _packet_markdown(packet: dict[str, Any]) -> str:
    lines = [
        "# Слепая разметка claim-level",
        "",
        packet["instructions"],
        "",
    ]
    for claim in packet["claims"]:
        lines.append(f"## {claim['id']}")
        lines.append(f"Вопрос: {claim['question']}")
        lines.append(f"Утверждение: {claim['text']}")
        if claim["cited"]:
            lines.append(f"Цитируемые маркеры: {claim['cited']}")
            for excerpt in claim["excerpts"]:
                lines.append(f"- [{excerpt['marker']}] {excerpt['text']}")
        else:
            lines.append("Цитат нет — ожидается false.")
        lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
