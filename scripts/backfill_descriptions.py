"""Заполнение описаний книг из метаданных файлов (T84).

Описания улучшают routing-профили и метаданные кандидатов; текст книг
не меняется. Скрипт идемпотентен: пустые описания заполняются, существующие
не перезаписываются без ``--force``.

    uv run python scripts/backfill_descriptions.py --dry-run
    uv run python scripts/backfill_descriptions.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from book_expert.catalog import Catalog
from book_expert.config import Settings
from book_expert.ingestion.metadata import extract_description


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true", help="перезаписать описания")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    settings = Settings()
    catalog = Catalog(settings.catalog_path)
    catalog.initialize()
    ready = [book for book in catalog.list_books() if book.status.value == "ready"]
    if args.limit:
        ready = ready[: args.limit]

    updated: list[dict[str, str]] = []
    missing: list[str] = []
    kept = 0
    for book in ready:
        if book.description and not args.force:
            kept += 1
            continue
        description = extract_description(book.path, book.format)
        if description is None:
            missing.append(book.id)
            continue
        if description == book.description:
            kept += 1
            continue
        updated.append(
            {
                "id": book.id,
                "title": book.title,
                "description": description,
            }
        )
        if not args.dry_run:
            catalog.set_status(book.id, book.status, description=description)

    report = {
        "dry_run": args.dry_run,
        "books": len(ready),
        "updated": len(updated),
        "kept": kept,
        "without_description": len(missing),
        "samples": updated[:5],
        "output": str(args.output) if args.output else None,
    }
    print(json.dumps(report, ensure_ascii=False, indent=1))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(
                {**report, "updated_books": updated, "missing": missing},
                ensure_ascii=False,
                indent=1,
            ),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
