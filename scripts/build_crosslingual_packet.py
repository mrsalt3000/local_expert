"""Cross-lingual NLI holdout: русский claim + английский cited-excerpt (T100).

Режим кандидатов выбирает реальные английские фрагменты из каталога:

    uv run python scripts/build_crosslingual_packet.py --candidates

Режим пакета берёт ручные claim'ы (``claims.json``) и делает слепой
``labels.csv``. Текст книг и gold-метки остаются только в gitignored
``data/artifacts``:

    uv run python scripts/build_crosslingual_packet.py --claims claims.json
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
from pathlib import Path
from typing import Any

from book_expert.catalog import Catalog
from book_expert.eval.language import detect_language

STOPWORDS = re.compile(
    r"\b(the|and|that|with|for|from|this|are|was|were|have|has|not|you|"
    r"can|will|which|their|they|its|into|when|then|than|also|such)\b",
    re.IGNORECASE,
)
SKIP = re.compile(
    r"table of contents|\bcontents\b|\bindex\b|bibliograph|references|"
    r"isbn|copyright|all rights reserved",
    re.IGNORECASE,
)
LETTER = re.compile(r"[^\W\d_]", re.UNICODE)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--catalog", type=Path, default=Path("data/artifacts/catalog.sqlite3")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("data/artifacts/cross-lingual-nli")
    )
    parser.add_argument("--per-book", type=int, default=5)
    parser.add_argument("--min-length", type=int, default=500)
    parser.add_argument("--max-length", type=int, default=1600)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--candidates", action="store_true")
    parser.add_argument("--claims", type=Path)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    candidates_path = args.output / "candidates.json"
    if args.claims:
        _build_packet(args.output, args.claims, args.seed)
        return

    catalog = Catalog(args.catalog)
    candidates = _candidates(catalog, args)
    candidates_path.write_text(
        json.dumps({"candidates": candidates}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (args.output / "candidates.md").write_text(
        _candidates_markdown(candidates), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "books": len({item["book_id"] for item in candidates}),
                "candidates": len(candidates),
                "output": str(candidates_path),
            },
            ensure_ascii=False,
        )
    )


def _candidates(catalog: Catalog, args: argparse.Namespace) -> list[dict[str, Any]]:
    rng = random.Random(args.seed)
    result: list[dict[str, Any]] = []
    for book in catalog.list_books():
        if str(book.status) != "ready":
            continue
        chunks = catalog.iter_chunks(book.id)
        if not chunks:
            continue
        sample = _sample(chunks, 24)
        if detect_language(" ".join(chunk.text for chunk in sample)) != "en":
            continue
        if _ascii_share(" ".join(chunk.text for chunk in sample)) < 0.95:
            continue
        eligible = [
            chunk
            for chunk in chunks
            if _eligible(chunk.text, chunk.pdf_page_index, book.page_count, args)
        ]
        ranked = sorted(
            eligible,
            key=lambda chunk: (
                -_factual_score(chunk.text),
                chunk.pdf_page_index,
                chunk.ordinal,
            ),
        )
        picked: list[Any] = []
        pages: set[int] = set()
        for chunk in ranked:
            if chunk.pdf_page_index in pages:
                continue
            picked.append(chunk)
            pages.add(chunk.pdf_page_index)
            if len(picked) >= args.per_book:
                break
        rng.shuffle(picked)
        for chunk in picked:
            result.append(
                {
                    "chunk_id": chunk.id,
                    "book_id": book.id,
                    "book_title": book.title,
                    "author": book.author,
                    "page": chunk.pdf_page_index,
                    "section": chunk.section_path,
                    "text": chunk.text.strip(),
                }
            )
    return result


def _sample(chunks: list[Any], count: int) -> list[Any]:
    if len(chunks) <= count:
        return chunks
    step = (len(chunks) - 1) / (count - 1)
    return [chunks[round(index * step)] for index in range(count)]


def _ascii_share(text: str) -> float:
    letters = LETTER.findall(text)
    if not letters:
        return 0.0
    ascii_letters = sum(1 for char in letters if char.isascii())
    return ascii_letters / len(letters)


def _eligible(text: str, page: int, page_count: int, args: argparse.Namespace) -> bool:
    stripped = text.strip()
    if not args.min_length <= len(stripped) <= args.max_length:
        return False
    if _ascii_share(stripped) < 0.97:
        return False
    if SKIP.search(stripped) or len(STOPWORDS.findall(stripped)) < 4:
        return False
    return 0.12 * page_count <= page <= 0.88 * page_count


def _factual_score(text: str) -> float:
    numbers = len(set(re.findall(r"\d+(?:\.\d+)?", text)))
    copulas = len(
        re.findall(r"\b(is|are|was|were|can|should|must|means|requires)\b", text)
    )
    return numbers + 0.25 * copulas


def _candidates_markdown(candidates: list[dict[str, Any]]) -> str:
    lines = ["# Английские кандидаты для cross-lingual NLI", ""]
    for item in candidates:
        lines.extend(
            [
                f"## {item['chunk_id']}",
                f"Книга: {item['book_title']} — {item['author']}",
                f"Страница: {item['page']} (pdf page index)",
                "",
                item["text"],
                "",
            ]
        )
    return "\n".join(lines)


def _build_packet(output: Path, claims_path: Path, seed: int) -> None:
    candidates = {
        item["chunk_id"]: item
        for item in json.loads(
            (output / "candidates.json").read_text(encoding="utf-8")
        )["candidates"]
    }
    claims = json.loads(claims_path.read_text(encoding="utf-8"))
    random.Random(seed).shuffle(claims)
    packet_claims: list[dict[str, Any]] = []
    gold: dict[str, bool] = {}
    for index, claim in enumerate(claims, start=1):
        chunk = candidates.get(claim["chunk_id"])
        if chunk is None:
            raise SystemExit(f"unknown chunk_id: {claim['chunk_id']}")
        if detect_language(chunk["text"]) != "en":
            raise SystemExit(f"excerpt is not English: {claim['chunk_id']}")
        if detect_language(claim["claim"]) != "ru":
            raise SystemExit(f"claim is not Russian: {claim['claim'][:60]}")
        claim_id = f"x{index:02d}"
        packet_claims.append(
            {
                "id": claim_id,
                "text": claim["claim"],
                "claim": claim["claim"],
                "cited": [1],
                "excerpts": [{"marker": 1, "text": chunk["text"]}],
                "book_id": chunk["book_id"],
                "book_title": chunk["book_title"],
                "page": chunk["page"],
                "chunk_id": chunk["chunk_id"],
                "premise_language": "en",
                "claim_language": "ru",
            }
        )
        gold[claim_id] = bool(claim["label"])
    packet = {"dataset": "t100-cross-lingual", "claims": packet_claims}
    (output / "packet.json").write_text(
        json.dumps(packet, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    private = output / "private"
    private.mkdir(exist_ok=True)
    (private / "gold.local.json").write_text(
        json.dumps({"labels": gold}, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output / "README.md").write_text(
        _instructions(len(packet_claims)), encoding="utf-8"
    )
    (output / "packet.md").write_text(_packet_markdown(packet_claims), encoding="utf-8")
    _write_labels(output / "labels.csv", packet_claims)
    print(
        json.dumps(
            {
                "claims": len(packet_claims),
                "true": sum(gold.values()),
                "false": len(gold) - sum(gold.values()),
                "output": str(output),
            },
            ensure_ascii=False,
        )
    )


def _instructions(count: int) -> str:
    return f"""\
# Cross-lingual NLI holdout (T100)

{count} пар: **русское утверждение** и **английский фрагмент книги**.
Заполните `label` (`true`/`false`) в `labels.csv`.

Правило: `true` только если английский excerpt **прямо** подтверждает
русское утверждение. Перевод смысла допустим, догадки — нет: если excerpt
лишь по теме, частичный или добавляет детали, которых в нём нет — `false`.

Не открывайте папку `private/`: там construction-метки. Разметка должна быть
независимой. Смотрите только `claim`, `excerpt`, `book`, `page`.
"""


def _write_labels(path: Path, claims: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["id", "label", "claim", "book", "page", "excerpt"],
            delimiter=";",
        )
        writer.writeheader()
        for claim in claims:
            writer.writerow(
                {
                    "id": claim["id"],
                    "label": "",
                    "claim": claim["text"],
                    "book": claim["book_title"],
                    "page": claim["page"],
                    "excerpt": " ".join(claim["excerpts"][0]["text"].split()),
                }
            )


def _packet_markdown(claims: list[dict[str, Any]]) -> str:
    lines = ["# Cross-lingual NLI: русский claim + английский excerpt", ""]
    for claim in claims:
        lines.extend(
            [
                f"## {claim['id']}",
                f"Утверждение: {claim['text']}",
                f"Книга: {claim['book_title']} (стр. {claim['page']})",
                "",
                claim["excerpts"][0]["text"],
                "",
            ]
        )
    return "\n".join(lines)


if __name__ == "__main__":
    main()
