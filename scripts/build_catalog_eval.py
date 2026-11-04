"""Сборка каталог-wide eval: вопросы по страницам + независимая проверка.

Скрипт не пишет текст книг в репозиторий: датасет сохраняется в указанный
``--output`` (по умолчанию ``/tmp``). Для каждого выбранного чанка локальная
модель пишет вопрос, DeepSeek проверяет, что ответ полностью есть в странице.

    uv run python scripts/build_catalog_eval.py \
      --books-json /tmp/t88_new_books.json \
      --base data/eval/multi-book-40.local.json \
      --output data/eval/catalog-30books-120.local.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

from book_expert.catalog import Catalog
from book_expert.config import Settings
from book_expert.models import Book
from book_expert.providers import ChatMessage, Provider, ProviderError, get_provider

BANNED = (
    "фрагмент",
    "этот текст",
    "данный текст",
    "автор",
    "страниц",
    "книг",
    "отрывок",
)
TARGET_TOTAL = 120


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--books-json", required=True)
    parser.add_argument("--base", default="data/eval/multi-book-40.local.json")
    parser.add_argument("--output", default="data/eval/catalog-30books-120.local.json")
    parser.add_argument("--target-total", type=int, default=TARGET_TOTAL)
    parser.add_argument("--generator-provider", default="ollama")
    parser.add_argument("--verifier-provider", default="deepseek")
    parser.add_argument("--seed", type=int, default=88)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit-books", type=int, default=0)
    args = parser.parse_args()

    books = json.loads(Path(args.books_json).read_text(encoding="utf-8"))
    if args.limit_books:
        books = books[: args.limit_books]
    base = json.loads(Path(args.base).read_text(encoding="utf-8"))
    remaining = max(0, args.target_total - len(base))
    # 2 кейса каждой книге, остаток — самым крупным книгам.
    per_book = {book["id"]: 2 for book in books}
    extra = remaining - 2 * len(books)
    for book in sorted(books, key=lambda item: item["pages"], reverse=True)[:extra]:
        per_book[book["id"]] = 3

    settings = Settings()
    settings.ensure_dirs()
    catalog = Catalog(settings.catalog_path)
    catalog.initialize()
    generator = get_provider(settings.with_provider(args.generator_provider))
    verifier = get_provider(settings.with_provider(args.verifier_provider))
    cases: list[dict[str, Any]] = list(base)
    started = time.monotonic()

    for number, entry in enumerate(books, start=1):
        book = catalog.get_book(entry["id"])
        if book is None:
            print(f"skip missing {entry['id']}", file=sys.stderr)
            continue
        want = per_book[book.id]
        made = 0
        for chunk in _candidate_chunks(catalog, book, want * 4):
            if made >= want:
                break
            question = _generate(generator, chunk.text)
            if question is None:
                continue
            verdict = _verify(verifier, question, chunk.text)
            if verdict is None:
                continue
            keyword = _keyword(verdict, chunk.text)
            if keyword is None:
                continue
            if _normalize(keyword) in _normalize(question):
                continue
            cases.append(
                {
                    "question": question,
                    "expected_chunk_ids": [chunk.id],
                    "expected_book_ids": [book.id],
                    "_gt_page": chunk.pdf_page_index,
                    "_gt_kw": keyword,
                    "_gt_chars": [len(chunk.text)],
                    "_reviewed": True,
                    "_source": "t88-generated-2026-11-04",
                    "_review_note": "page-grounded question + deepseek cross-check",
                }
            )
            made += 1
        elapsed = time.monotonic() - started
        print(
            f"{number}/{len(books)} {book.title[:45]}: {made}/{want} "
            f"({len(cases)} total, {elapsed:.0f}s)",
            file=sys.stderr,
        )
        if not args.dry_run:
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            Path(args.output).write_text(
                json.dumps(cases, ensure_ascii=False, indent=2), encoding="utf-8"
            )

    Path(args.output).write_text(
        json.dumps(cases, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "cases": len(cases),
                "books": len({case["expected_book_ids"][0] for case in cases}),
                "output": args.output,
            },
            ensure_ascii=False,
        )
    )


def _candidate_chunks(catalog: Catalog, book: Book, limit: int):
    by_page: dict[int, Any] = {}
    low = max(8, int(book.page_count * 0.12))
    high = int(book.page_count * 0.95)
    for chunk in catalog.iter_chunks(book.id):
        if not (low <= chunk.pdf_page_index <= high):
            continue
        if not 500 <= len(chunk.text) <= 2400:
            continue
        current = by_page.get(chunk.pdf_page_index)
        if current is None or len(chunk.text) > len(current.text):
            by_page[chunk.pdf_page_index] = chunk
    pages = sorted(by_page)
    if not pages:
        return []
    if len(pages) <= limit:
        return [by_page[page] for page in pages]
    indices = sorted(
        {round(index * (len(pages) - 1) / (limit - 1)) for index in range(limit)}
    )
    return [by_page[pages[index]] for index in indices]


def _generate(provider: Provider, text: str) -> str | None:
    prompt = (
        "По фрагменту книги напиши ОДИН конкретный вопрос на русском языке, "
        "ответ на который полностью содержится в этом фрагменте. Вопрос должен "
        "быть самостоятельным: не упоминай «фрагмент», «текст», «автор», "
        "«страницу» и не копируй ответ в вопрос. Верни строго JSON "
        '{"question": "..."} без пояснений.\n\nФрагмент:\n' + text[:3000]
    )
    try:
        response = provider.chat(
            [ChatMessage(role="user", content=prompt)],
            temperature=0.0,
            max_tokens=220,
        )
    except ProviderError:
        return None
    payload = _json(response.text)
    question = payload.get("question")
    if not isinstance(question, str):
        return None
    question = " ".join(question.split()).strip()
    if not (15 <= len(question) <= 220) or not question.endswith("?"):
        return None
    if any(word in question.casefold() for word in BANNED):
        return None
    return question


def _verify(provider: Provider, question: str, text: str) -> dict[str, Any] | None:
    prompt = (
        "Проверь, можно ли ответить на вопрос ТОЛЬКО по фрагменту. Верни строго "
        'JSON {"answerable": true|false, "answer": "...", "keyword": "..."} '
        "где answer — короткий ответ, keyword — 1–3 слова из фрагмента, по "
        "которым виден ответ. Если ответа нет, answerable=false.\n\n"
        f"Вопрос: {question}\n\nФрагмент:\n{text[:3000]}"
    )
    try:
        response = provider.chat(
            [ChatMessage(role="user", content=prompt)],
            temperature=0.0,
            max_tokens=220,
        )
    except ProviderError:
        return None
    payload = _json(response.text)
    if not payload.get("answerable"):
        return None
    return payload


def _keyword(payload: dict[str, Any], text: str) -> str | None:
    normalized = _normalize(text)
    for key in ("keyword", "answer"):
        value = payload.get(key)
        if not isinstance(value, str):
            continue
        words = [
            word
            for word in re.findall(r"[0-9A-Za-zА-Яа-яЁё-]{3,}", value)
            if _normalize(word) in normalized
        ]
        if words:
            return " ".join(words[:3])
    return None


def _json(text: str) -> dict[str, Any]:
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < start:
        return {}
    try:
        payload = json.loads(text[start : end + 1])
    except (TypeError, ValueError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _normalize(text: str) -> str:
    return re.sub(r"[^0-9a-zа-яё]+", " ", text.casefold()).strip()


if __name__ == "__main__":
    main()
