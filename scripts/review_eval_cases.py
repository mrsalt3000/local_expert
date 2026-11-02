"""Ревью черновиков eval-кейсов: страница-источник + голоса LLM.

Кейс считается проверенным, если ключевое слово вопроса встречается на
странице-источнике и LLM единогласно (по умолчанию 3 голоса) подтверждает,
что вопрос отвечается этой страницей. Скрипт ничего не удаляет и без
`--apply` не пишет файл.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from book_expert.catalog import Catalog
from book_expert.config import Settings
from book_expert.providers import ChatMessage, Provider, ProviderError, get_provider

VOTE_PROMPT = (
    "Ты проверяешь eval-кейс RAG-системы. Дан вопрос и текст одной страницы "
    "книги. Ответь, можно ли ответить на вопрос, опираясь только на эту "
    "страницу. Не используй внешние знания. "
    'Ответ строго JSON: {"answerable": true} или {"answerable": false}.'
)
PAGE_LIMIT = 6000


def _settings(provider: str | None, model: str | None) -> Settings:
    if provider is None:
        return Settings()
    if model is None and provider == "deepseek":
        model = "deepseek-chat"
    return Settings(provider=provider, **({"chat_model": model} if model else {}))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cases", type=Path)
    parser.add_argument("--apply", action="store_true", help="Записать результат.")
    parser.add_argument("--votes", type=int, default=3, help="Голосов LLM на кейс.")
    parser.add_argument("--provider", default=None, help="ollama | deepseek | openai")
    parser.add_argument("--model", default=None, help="chat-модель провайдера")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    payload = json.loads(args.cases.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise SystemExit("ожидался JSON-список кейсов")
    settings = _settings(args.provider, args.model)
    catalog = Catalog(settings.catalog_path)
    catalog.initialize()
    provider = get_provider(settings) if args.votes > 0 else None
    connection = sqlite3.connect(
        f"file:{settings.catalog_path}?mode=ro", uri=True, check_same_thread=False
    )
    pending = [case for case in payload if not case.get("_reviewed")]
    if args.limit:
        pending = pending[: args.limit]
    print(f"черновиков: {len(pending)} из {len(payload)}; голосов: {args.votes}")

    accepted = rejected = skipped = 0
    try:
        for index, case in enumerate(pending, start=1):
            page = _page_for_case(case, connection)
            if page is None:
                skipped += 1
                print(f"{index}/{len(pending)} skip", flush=True)
                continue
            reason = _check_heuristic(case, catalog, page)
            if reason is None and provider is not None:
                reason = _check_llm(case, catalog, page, provider, args.votes)
            if reason is None:
                case["_reviewed"] = True
                case["_review_note"] = "heuristic + unanimous judge"
                accepted += 1
                verdict = "accept"
            elif reason == "no page":
                skipped += 1
                verdict = "skip"
            else:
                rejected += 1
                verdict = f"reject: {reason}"
            print(f"{index}/{len(pending)} {verdict}", flush=True)
    finally:
        close = getattr(provider, "close", None)
        if callable(close):
            close()

    print(f"accepted {accepted}, rejected {rejected}, skipped {skipped}")
    if args.apply:
        temporary = args.cases.with_suffix(args.cases.suffix + ".tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(args.cases)
        print(f"записано: {args.cases}")
    return 0


def _page_for_case(case: dict[str, Any], connection: sqlite3.Connection) -> int | None:
    page = case.get("_gt_page")
    if isinstance(page, int):
        return page
    for chunk_id in case.get("expected_chunk_ids", []):
        row = connection.execute(
            "SELECT pdf_page_index FROM chunks WHERE id = ?", (chunk_id,)
        ).fetchone()
        if row is not None:
            return int(row[0])
    return None


def _check_heuristic(case: dict[str, Any], catalog: Catalog, page: int) -> str | None:
    book_ids = case.get("expected_book_ids") or []
    if not book_ids:
        return "no page"
    text = catalog.page_text(book_ids[0], page, max_chars=PAGE_LIMIT)
    if not text:
        return "no page text"
    keyword = str(case.get("_gt_kw") or "").casefold().strip()
    if keyword and keyword not in text.casefold():
        return f"keyword {keyword!r} not on page"
    return None


def _check_llm(
    case: dict[str, Any],
    catalog: Catalog,
    page: int,
    provider: Provider,
    votes: int,
) -> str | None:
    book_ids = case.get("expected_book_ids") or []
    if not book_ids:
        return "no page"
    text = (catalog.page_text(book_ids[0], page, max_chars=PAGE_LIMIT) or "")[
        :PAGE_LIMIT
    ]
    messages = [
        ChatMessage(role="system", content=VOTE_PROMPT),
        ChatMessage(
            role="user",
            content=f"Вопрос: {case['question']}\n\nСтраница:\n{text}",
        ),
    ]
    votes_true = 0
    try:
        for _ in range(max(1, votes)):
            response = provider.chat(messages, temperature=0.0, max_tokens=40)
            if _answerable(response.text):
                votes_true += 1
    except ProviderError as exc:
        return f"judge failed: {exc}"
    if votes_true == max(1, votes):
        return None
    return f"judge {votes_true}/{max(1, votes)}"


def _answerable(text: str) -> bool:
    lowered = text.strip().casefold()
    return '"answerable": true' in lowered or "'answerable': true" in lowered


if __name__ == "__main__":
    raise SystemExit(main())
