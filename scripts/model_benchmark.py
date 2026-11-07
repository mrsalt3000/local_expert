"""Сравнение локальных chat-моделей на одном RAG/agent eval (T105).

Запускает `book-expert stage-eval` и `agent-eval` для каждой модели с
фиксированным судьёй `qwen3.5:4b`, складывает отчёты в
`data/artifacts/model-benchmark/` и пишет `summary.json` + `summary.md`.

Пример:

    uv run python scripts/model_benchmark.py --stage-limit 120 --agent-limit 20

Скрипт рассчитан на длинный прогон: запускать в фоне, логи — в `logs/`.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CASES = Path("data/eval/combined-120.local.json")
OUT = Path("data/artifacts/model-benchmark")

MODELS: tuple[dict[str, Any], ...] = (
    {
        "slug": "ministral-3-3b",
        "label": "Ministral 3 3B Instruct",
        "provider": "ollama",
        "model": "ministral-3:3b",
    },
    {
        "slug": "gemma3-4b",
        "label": "Gemma 3 4B IT",
        "provider": "ollama",
        "model": "gemma3:4b",
    },
    {
        "slug": "phi4-mini",
        "label": "Phi-4-mini-instruct",
        "provider": "ollama",
        "model": "phi4-mini:latest",
    },
    {
        "slug": "qwen3-4b-q4",
        "label": "Qwen3-4B-Instruct q4_K_M",
        "provider": "ollama",
        "model": "qwen3:4b-q4_K_M",
    },
    {
        "slug": "qwen3-4b-q8",
        "label": "Qwen3-4B-Instruct q8_0",
        "provider": "ollama",
        "model": "qwen3:4b-q8_0",
    },
    {
        "slug": "deepseek-flash-ref",
        "label": "DeepSeek Flash (reference)",
        "provider": "deepseek",
        "model": "deepseek-flash",
        "stage_limit": 20,
        "agent_limit": 20,
    },
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage-limit", type=int, default=120)
    parser.add_argument("--agent-limit", type=int, default=20)
    parser.add_argument("--judge-model", default="qwen3.5:4b")
    parser.add_argument("--only", nargs="*", default=None)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()

    selected = [
        model
        for model in MODELS
        if args.only is None or model["slug"] in set(args.only)
    ]
    if not selected:
        raise SystemExit("no models selected")
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "logs").mkdir(exist_ok=True)

    for model in selected:
        _run_model(model, args)

    _write_summary(args.out, args.judge_model)


def _run_model(model: dict[str, Any], args: argparse.Namespace) -> None:
    slug = str(model["slug"])
    if model["provider"] == "ollama" and not _ollama_has(str(model["model"])):
        raise SystemExit(f"model is not installed: {model['model']}")
    stage_limit = int(model.get("stage_limit", args.stage_limit))
    agent_limit = int(model.get("agent_limit", args.agent_limit))
    if stage_limit:
        output = args.out / f"{slug}-stage.json"
        if not (args.skip_existing and output.exists()):
            _call(
                model,
                [
                    "stage-eval",
                    str(CASES),
                    "--limit",
                    str(stage_limit),
                    "--scope",
                    "expected",
                    "--faithfulness",
                    "--judge-provider",
                    "ollama",
                    "--judge-model",
                    args.judge_model,
                    "--output",
                    str(output),
                ],
                args.out / "logs" / f"{slug}-stage.log",
            )
    if agent_limit:
        output = args.out / f"{slug}-agent.json"
        if not (args.skip_existing and output.exists()):
            _call(
                model,
                [
                    "agent-eval",
                    str(CASES),
                    "--limit",
                    str(agent_limit),
                    "--faithfulness",
                    "--judge-provider",
                    "ollama",
                    "--judge-model",
                    args.judge_model,
                    "--output",
                    str(output),
                ],
                args.out / "logs" / f"{slug}-agent.log",
            )


def _call(model: dict[str, Any], arguments: list[str], log_path: Path) -> None:
    env = os.environ.copy()
    env.update(
        {
            "BOOK_EXPERT_PROVIDER": str(model["provider"]),
            "BOOK_EXPERT_CHAT_MODEL": str(model["model"]),
            "BOOK_EXPERT_OLLAMA_THINK": "false",
            "BOOK_EXPERT_TOP_K": "20",
        }
    )
    command = ["uv", "run", "book-expert", *arguments]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.run(
            command,
            cwd=ROOT,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if process.returncode != 0:
        raise SystemExit(f"{model['slug']}: command failed, see {log_path}")


def _ollama_has(model: str) -> bool:
    try:
        result = subprocess.run(
            ["ollama", "show", model],
            cwd=ROOT,
            capture_output=True,
            check=False,
        )
    except OSError:
        return False
    return result.returncode == 0


def _write_summary(out: Path, judge_model: str) -> None:
    rows = []
    for model in MODELS:
        stage_path = out / f"{model['slug']}-stage.json"
        agent_path = out / f"{model['slug']}-agent.json"
        rows.append(
            {
                "slug": model["slug"],
                "label": model["label"],
                "provider": model["provider"],
                "model": model["model"],
                "stage": _stage_metrics(stage_path),
                "agent": _agent_metrics(agent_path),
            }
        )
    summary = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "cases": str(CASES),
        "judge": {"provider": "ollama", "model": judge_model},
        "models": rows,
    }
    (out / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out / "summary.md").write_text(_summary_markdown(summary), encoding="utf-8")
    print(json.dumps({"output": str(out / "summary.json")}))


def _stage_metrics(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    report = json.loads(path.read_text(encoding="utf-8"))
    word_path = path.with_name(path.name.replace("-stage.json", "-stage-20.json"))
    word_answers = (
        json.loads(word_path.read_text(encoding="utf-8")).get("answers", {})
        if word_path.exists()
        else {}
    )
    retrieval = report.get("retrieval", {})
    answers = report.get("answers", {})
    latency = report.get("latency_ms", {})
    return {
        "cases": report.get("cases"),
        "page_recall_at_5": retrieval.get("page_recall_at_5"),
        "page_mrr_at_k": retrieval.get("page_mrr_at_k"),
        "citation_precision": answers.get("citation_precision"),
        "citation_recall": answers.get("citation_recall"),
        "faithfulness": answers.get("faithfulness"),
        "citation_support_precision": answers.get("citation_support_precision"),
        "refused": answers.get("refused"),
        "zero_precision": answers.get("zero_precision"),
        "avg_answer_words": word_answers.get(
            "avg_answer_words", answers.get("avg_answer_words")
        ),
        "avg_answer_words_answered": word_answers.get(
            "avg_answer_words_answered", answers.get("avg_answer_words_answered")
        ),
        "answer_p50_ms": latency.get("answer_p50"),
        "answer_p95_ms": latency.get("answer_p95"),
        "meta": report.get("meta", {}),
    }


def _agent_metrics(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    report = json.loads(path.read_text(encoding="utf-8"))
    retrieval = report.get("retrieval", {})
    answers = report.get("answers", {})
    latency = report.get("latency_ms", {})
    plan = report.get("plan", {})
    return {
        "cases": report.get("cases"),
        "page_recall": retrieval.get("page_recall"),
        "source_page_recall": retrieval.get("source_page_recall"),
        "citation_precision": retrieval.get("citation_precision"),
        "faithfulness": answers.get("faithfulness"),
        "citation_support_precision": answers.get("citation_support_precision"),
        "refused": answers.get("refused"),
        "avg_answer_words": answers.get("avg_answer_words"),
        "p50_ms": latency.get("p50"),
        "p95_ms": latency.get("p95"),
        "avg_queries": plan.get("avg_queries"),
        "avg_rounds": plan.get("avg_rounds"),
        "avg_extra_queries": plan.get("avg_extra_queries"),
        "meta": report.get("meta", {}),
    }


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _summary_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Сравнение локальных моделей (RAG + Agent)",
        "",
        f"Кейсы: `{summary['cases']}`; судья: "
        f"`{summary['judge']['provider']}/{summary['judge']['model']}`.",
        "",
        "## RAG (stage-eval)",
        "",
        "| модель | n | page@5 | MRR@k | cit prec | cit rec | faith | support | отказы | p50, ms |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary["models"]:
        stage = row["stage"]
        if not stage:
            lines.append(f"| {row['label']} | — | — | — | — | — | — | — | — | — |")
            continue
        lines.append(
            "| {label} | {cases} | {p5} | {mrr} | {cp} | {cr} | {faith} | "
            "{support} | {refused} | {p50} |".format(
                label=row["label"],
                cases=_fmt(stage["cases"]),
                p5=_fmt(stage["page_recall_at_5"]),
                mrr=_fmt(stage["page_mrr_at_k"]),
                cp=_fmt(stage["citation_precision"]),
                cr=_fmt(stage["citation_recall"]),
                faith=_fmt(stage["faithfulness"]),
                support=_fmt(stage["citation_support_precision"]),
                refused=_fmt(stage["refused"]),
                p50=_fmt(stage["answer_p50_ms"], 0),
            )
        )
    lines.extend(
        [
            "",
            "## Agent (agent-eval)",
            "",
            "| модель | n | page | source | cit prec | faith | support | отказы | p50, ms |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in summary["models"]:
        agent = row["agent"]
        if not agent:
            lines.append(f"| {row['label']} | — | — | — | — | — | — | — | — |")
            continue
        lines.append(
            "| {label} | {cases} | {page} | {source} | {cp} | {faith} | "
            "{support} | {refused} | {p50} |".format(
                label=row["label"],
                cases=_fmt(agent["cases"]),
                page=_fmt(agent["page_recall"]),
                source=_fmt(agent["source_page_recall"]),
                cp=_fmt(agent["citation_precision"]),
                faith=_fmt(agent["faithfulness"]),
                support=_fmt(agent["citation_support_precision"]),
                refused=_fmt(agent["refused"]),
                p50=_fmt(agent["p50_ms"], 0),
            )
        )
    lines.extend(
        [
            "",
            "## Новые метрики (20-кейсовый срез, где есть)",
            "",
            "| модель | RAG words | Agent words | queries | rounds | extra |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in summary["models"]:
        stage = row["stage"] or {}
        agent = row["agent"] or {}
        rounds = agent.get("avg_rounds")
        extra = agent.get("avg_extra_queries")
        lines.append(
            "| {label} | {rw} | {aw} | {q} | {r} | {e} |".format(
                label=row["label"],
                rw=_fmt(stage.get("avg_answer_words"), 1),
                aw=_fmt(agent.get("avg_answer_words"), 1),
                q=_fmt(agent.get("avg_queries"), 2),
                r="n/a (refine off)" if rounds == 0 else _fmt(rounds, 2),
                e="n/a (refine off)" if extra in (0, None) else _fmt(extra, 2),
            )
        )
    lines.extend(
        [
            "",
            "## Ограничения",
            "",
            "- Retrieval одинаков для всех: один индекс, `bge-m3`, `scope=expected`.",
            "- Судья фиксирован, но его kappa на human-holdout ниже .6: это relative ranking, "
            "не абсолютное качество.",
            "- DeepSeek Flash — только reference на 20 кейсах, не кандидат в прод.",
            "- Qwen3-4B q5/q6 не проверялись: в Ollama нет тегов, импорт GGUF не делался.",
        ]
    )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    sys.exit(main())
