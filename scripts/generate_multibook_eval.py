"""Черновик eval-набора по 10 книгам: вопрос из страницы + ключевое слово.

Скрипт только создаёт черновики (`_reviewed: false`); проверка — отдельно,
`scripts/review_eval_cases.py`. Реальный текст книги в git не попадает:
результат пишется в `data/eval/*.local.json` (git-ignored).
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from book_expert.catalog import Catalog
from book_expert.config import Settings
from book_expert.models import BookStatus
from book_expert.providers import ChatMessage, Provider, ProviderError, get_provider

GENERATE_PROMPT = (
    "Ты составляешь eval-кейс для RAG-системы по книге. Дан текст одной "
    "страницы. Придумай один конкретный вопрос на русском, ответ на который "
    "явно и полностью есть на этой странице, и одно ключевое слово (2–4 буквы "
    "или больше) из правильного ответа, которое встречается на странице. "
    "Не спрашивай про номер страницы, автора или оглавление. "
    'Ответ строго JSON: {"question": "...", "keyword": "..."}.'
)
PAGE_LIMIT = 5000
MIN_TEXT = 400


def _settings(provider: str | None, model: str | None) -> Settings:
    if provider is None:
        return Settings()
    if model is None and provider == "deepseek":
        model = "deepseek-chat"
    return Settings(provider=provider, **({"chat_model": model} if model else {}))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--books", type=int, default=10)
    parser.add_argument("--per-book", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20261102)
    parser.add_argument("--provider", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/eval/multi-book-40.local.json"),
    )
    parser.add_argument(
        "--exclude",
        type=Path,
        default=Path("data/eval/combined-120.local.json"),
    )
    args = parser.parse_args()

    settings = _settings(args.provider, args.model)
    catalog = Catalog(settings.catalog_path)
    catalog.initialize()
    provider = get_provider(settings)
    excluded = _book_ids(args.exclude)
    books = [
        book
        for book in catalog.list_books()
        if book.status is BookStatus.ready
        and book.id not in excluded
        and book.page_count >= args.per_book * 5
    ]
    random.Random(args.seed).shuffle(books)
    cases: list[dict[str, Any]] = []
    try:
        for book in books[: args.books]:
            pages = _sample_pages(catalog, book.id, book.page_count, args.per_book)
            made = 0
            for page in pages:
                text = (catalog.page_text(book.id, page, max_chars=PAGE_LIMIT) or "")[
                    :PAGE_LIMIT
                ]
                if len(text) < MIN_TEXT:
                    continue
                case = _generate(case_provider=provider, question_text=text)
                if case is None:
                    continue
                keyword, question = case
                if keyword.casefold() not in text.casefold():
                    continue
                chunk = _first_chunk(catalog, book.id, page)
                if chunk is None:
                    continue
                cases.append(
                    {
                        "question": question,
                        "expected_chunk_ids": [chunk],
                        "expected_book_ids": [book.id],
                        "_gt_page": page,
                        "_gt_kw": keyword,
                        "_reviewed": False,
                        "_source": "multibook-draft-2026-11-02",
                    }
                )
                made += 1
                if made >= args.per_book:
                    break
            print(f"{book.title[:50]}: {made} вопросов", flush=True)
    finally:
        close = getattr(provider, "close", None)
        if callable(close):
            close()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(cases, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"черновиков: {len(cases)} → {args.output}")
    return 0


def _book_ids(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    payload = json.loads(path.read_text(encoding="utf-8"))
    ids: set[str] = set()
    for case in payload:
        for book_id in case.get("expected_book_ids", []):
            ids.add(book_id)
    return ids


def _sample_pages(
    catalog: Catalog, book_id: str, page_count: int, count: int
) -> list[int]:
    chunks = list(catalog.iter_chunks(book_id))
    pages = sorted({chunk.pdf_page_index for chunk in chunks})
    body = [page for page in pages if 0 < page < page_count - 1]
    if len(body) <= count:
        return body
    step = max(1, len(body) // count)
    return body[::step][:count]


def _first_chunk(catalog: Catalog, book_id: str, page: int) -> str | None:
    for chunk in catalog.iter_chunks(book_id):
        if chunk.pdf_page_index == page and len(chunk.text) >= MIN_TEXT:
            return chunk.id
    return None


def _generate(*, case_provider: Provider, question_text: str) -> tuple[str, str] | None:
    messages = [
        ChatMessage(role="system", content=GENERATE_PROMPT),
        ChatMessage(role="user", content=f"Страница:\n{question_text}"),
    ]
    try:
        response = case_provider.chat(messages, temperature=0.2, max_tokens=200)
    except ProviderError:
        return None
    payload = _extract_json(response.text)
    question = payload.get("question")
    keyword = payload.get("keyword")
    if not isinstance(question, str) or not isinstance(keyword, str):
        return None
    question = question.strip()
    keyword = keyword.strip()
    if len(question) < 10 or len(keyword) < 2:
        return None
    return keyword, question


def _extract_json(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start < 0 or end < start:
        return {}
    try:
        payload = json.loads(cleaned[start : end + 1])
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


if __name__ == "__main__":
    raise SystemExit(main())
