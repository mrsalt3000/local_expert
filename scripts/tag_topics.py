"""Разметка книг по темам локальной моделью.

Скрипт идемпотентен: книги с непустыми темами пропускаются, ``--force``
перезаписывает разметку. После каждого батча темы сохраняются в каталог,
поэтому прерванный прогон продолжается с места остановки.

    uv run python scripts/tag_topics.py --limit 40 --dry-run
    uv run python scripts/tag_topics.py
    uv run python scripts/tag_topics.py --force --batch-size 16
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

from book_expert.catalog import Catalog
from book_expert.config import Settings
from book_expert.models import Book, BookStatus
from book_expert.providers import ChatMessage, Provider, ProviderError, get_provider
from book_expert.topics import TOPICS, normalize_topics

MAX_DESCRIPTION_CHARS = 400


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, help="сколько книг обработать")
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--provider", default="ollama")
    parser.add_argument("--model", default=None)
    parser.add_argument("--force", action="store_true", help="перезаписать темы")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")

    settings = Settings().with_provider(args.provider)
    if args.model:
        settings = settings.model_copy(update={"chat_model": args.model})
    provider = get_provider(settings)
    catalog = Catalog(settings.catalog_path)
    catalog.initialize()

    existing = catalog.topics_map()
    books = [
        book
        for book in catalog.list_books()
        if book.status is BookStatus.ready and (args.force or not existing.get(book.id))
    ]
    if args.limit:
        books = books[: args.limit]
    if not books:
        print("нечего размечать: все ready-книги уже с темами")
        return

    started = time.monotonic()
    tagged = 0
    failed: list[dict[str, str]] = []
    distribution: Counter[str] = Counter()
    total = len(books)
    junk = [book for book in books if _is_junk_title(book.title)]
    for book in junk:
        tagged += 1
        distribution["Другое"] += 1
        if not args.dry_run:
            catalog.set_topics(book.id, ["Другое"])
    books = [book for book in books if not _is_junk_title(book.title)]
    batches = [
        books[index : index + args.batch_size]
        for index in range(0, len(books), args.batch_size)
    ]
    for number, batch in enumerate(batches, start=1):
        result = _tag_batch(provider, batch)
        for book in batch:
            topics = result.get(book.id)
            if not topics:
                failed.append({"id": book.id, "title": book.title})
                continue
            tagged += 1
            distribution.update(topics)
            if not args.dry_run:
                catalog.set_topics(book.id, topics)
        elapsed = time.monotonic() - started
        done = min(len(junk) + number * args.batch_size, total)
        eta = elapsed / done * (total - done) if done else 0
        print(
            f"{done}/{total} · батч {number}/{len(batches)} · "
            f"{elapsed:.0f}s · ETA {eta:.0f}s · размечено {tagged}, ошибок {len(failed)}",
            file=sys.stderr,
        )

    summary = {
        "books": total,
        "tagged": tagged,
        "failed": failed,
        "dry_run": args.dry_run,
        "model": provider.chat_model,
        "provider": provider.name,
        "elapsed_s": round(time.monotonic() - started, 1),
        "topics": dict(distribution.most_common()),
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    print(json.dumps(summary, ensure_ascii=False))


def _is_junk_title(title: str) -> bool:
    """Имена-заглушки не дают сигнала для темы — не кормим ими модель."""
    text = " ".join(title.split()).strip()
    lowered = text.casefold()
    if len(text) < 3 or lowered in {"untitled", "без названия"}:
        return True
    if re.search(r"\.(?:indd|docx?|rtf|pdf)$", lowered):
        return True
    if re.fullmatch(r"[\d\W_]+", text):
        return True
    return bool(re.fullmatch(r"[a-z0-9_\-]{1,4}", lowered))


def _tag_batch(provider: Provider, books: list[Book]) -> dict[str, tuple[str, ...]]:
    remaining = list(books)
    tagged: dict[str, tuple[str, ...]] = {}
    for _attempt in range(2):
        if not remaining:
            break
        try:
            response = provider.chat(
                [
                    ChatMessage(role="system", content=_system_prompt()),
                    ChatMessage(role="user", content=_books_prompt(remaining)),
                ],
                temperature=0.0,
                max_tokens=max(400, min(2000, 120 * len(remaining))),
            )
        except ProviderError as exc:
            print(f"ошибка провайдера: {exc}", file=sys.stderr)
            continue
        parsed = _parse_response(response.text, remaining)
        tagged.update(parsed)
        remaining = [book for book in remaining if book.id not in tagged]
    return tagged


def _system_prompt() -> str:
    labels = "\n".join(f"- {topic}" for topic in TOPICS)
    return (
        "Ты классифицируешь книги по темам. Выбери 1–3 темы строго из списка, "
        "не придумывай новые формулировки. Если книга не подходит ни к одной "
        "теме, верни «Другое». Верни только JSON без пояснений.\n\n"
        f"Темы:\n{labels}\n\n"
        'Формат: {"books": [{"id": "book-id", "topics": ["тема", "тема"]}]}'
    )


def _books_prompt(books: list[Book]) -> str:
    items = [
        {
            "id": book.id,
            "title": book.title,
            "author": book.author,
            "description": (book.description or "")[:MAX_DESCRIPTION_CHARS],
            "file": book.path.name,
        }
        for book in books
    ]
    return "Книги:\n" + json.dumps(items, ensure_ascii=False)


def _parse_response(text: str, books: list[Book]) -> dict[str, tuple[str, ...]]:
    payload = _extract_json(text)
    raw = payload.get("books")
    if not isinstance(raw, list):
        return {}
    allowed = {book.id for book in books}
    result: dict[str, tuple[str, ...]] = {}
    for item in raw:
        if not isinstance(item, dict):
            continue
        book_id = item.get("id")
        if not isinstance(book_id, str) or book_id not in allowed:
            continue
        topics = item.get("topics")
        if not isinstance(topics, list):
            continue
        normalized = normalize_topics(topics)
        if normalized:
            result[book_id] = normalized
    return result


def _extract_json(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:]
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start < 0 or end < start:
        return {}
    candidate = cleaned[start : end + 1]
    for suffix in ("", "}", "]}", "]"):
        try:
            payload = json.loads(candidate + suffix)
        except (TypeError, ValueError):
            continue
        return payload if isinstance(payload, dict) else {}
    return {}


if __name__ == "__main__":
    main()
