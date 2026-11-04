"""Multi-gold разметка retrieval-кейсов (T83).

В eval-наборах на вопрос размечен один (реже 2–4) ``expected_chunk_id``.
Семантически верный, но другой фрагмент — соседняя страница, сосед по
разделу или почти дословный дубль в другом издании — получает recall 0.
Здесь строится расширенная разметка с оценками:

* ``grade 2`` — размеченный вручную gold-фрагмент;
* ``grade 1`` — допустимая альтернатива: другой чанк той же страницы,
  сосед по разделу или почти дословный дубль в другой книге;
* ``grade 0`` — всё остальное.

Правила детерминированы и не используют retrieval, поэтому разметку можно
пересобрать после переиндексации, не меняя ответы поиска.
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from book_expert.eval.metrics import EvalCase
from book_expert.retrieval.lemmas import lemmatize_tokens

GRADE_GOLD = 2
GRADE_ADMISSIBLE = 1

_WORD_RE = re.compile(r"\w+", flags=re.UNICODE)


@dataclass(frozen=True, slots=True)
class ChunkRow:
    id: str
    book_id: str
    pdf_page_index: int
    ordinal: int
    section_path: str | None
    text: str


@dataclass(frozen=True, slots=True)
class MultiGoldCase:
    index: int
    question: str
    grades: Mapping[str, int] = field(default_factory=dict)
    reasons: Mapping[str, str] = field(default_factory=dict)

    def pages(
        self, chunk_pages: Mapping[str, tuple[str, int]]
    ) -> dict[tuple[str, int], int]:
        """Схлопывает оценки чанков в оценки страниц (максимум по странице)."""
        pages: dict[tuple[str, int], int] = {}
        for chunk_id, grade in self.grades.items():
            page = chunk_pages.get(chunk_id)
            if page is None:
                continue
            pages[page] = max(pages.get(page, 0), grade)
        return pages

    def to_dict(self) -> dict[str, object]:
        return {
            "index": self.index,
            "question": self.question,
            "grades": dict(self.grades),
            "reasons": dict(self.reasons),
        }


def load_chunks(
    conn: sqlite3.Connection, book_ids: Iterable[str] | None = None
) -> list[ChunkRow]:
    """Читает чанки (без ``lemma_text``) для указанных книг."""
    sql = "SELECT id, book_id, pdf_page_index, ordinal, section_path, text FROM chunks"
    params: list[object] = []
    if book_ids is not None:
        ids = list(book_ids)
        if not ids:
            return []
        placeholders = ", ".join("?" for _ in ids)
        sql += f" WHERE book_id IN ({placeholders})"
        params.extend(ids)
    sql += " ORDER BY book_id, ordinal"
    return [
        ChunkRow(
            id=row[0],
            book_id=row[1],
            pdf_page_index=row[2],
            ordinal=row[3],
            section_path=row[4],
            text=row[5] or "",
        )
        for row in conn.execute(sql, params)
    ]


def build_multigold(
    cases: Sequence[EvalCase],
    chunks: Sequence[ChunkRow],
    *,
    section_window: int = 1,
    duplicates: Mapping[int, Sequence[str]] | None = None,
    include_same_page: bool = True,
    include_section_neighbour: bool = True,
    neighbour_min_jaccard: float = 0.3,
) -> tuple[MultiGoldCase, ...]:
    """Строит multi-gold разметку по структурным правилам.

    ``duplicates`` — необязательная карта ``index кейса -> chunk_id`` почти
    дословных дублей из других книг (их ищет ``find_duplicate_neighbors``,
    потому что для этого нужен доступ ко всему каталогу).

    Сосед по странице или разделу считается допустимым **только** при
    текстовом пересечении с gold не ниже ``neighbour_min_jaccard``: иначе
    «та же страница» засчитывала бы посторонний абзац и метрика просто
    повторяла бы page recall.
    """
    by_page: dict[tuple[str, int], list[ChunkRow]] = {}
    by_section: dict[tuple[str, str], list[ChunkRow]] = {}
    by_id = {chunk.id: chunk for chunk in chunks}
    for chunk in chunks:
        by_page.setdefault((chunk.book_id, chunk.pdf_page_index), []).append(chunk)
        if chunk.section_path:
            by_section.setdefault((chunk.book_id, chunk.section_path), []).append(chunk)

    result: list[MultiGoldCase] = []
    for index, case in enumerate(cases):
        grades: dict[str, int] = {}
        reasons: dict[str, str] = {}
        for chunk_id in case.expected_chunk_ids:
            grades[chunk_id] = GRADE_GOLD
            reasons[chunk_id] = "gold"

        for gold_id in case.expected_chunk_ids:
            gold = by_id.get(gold_id)
            if gold is None:
                continue
            target = shingles(gold.text)
            if include_same_page:
                for neighbour in by_page.get((gold.book_id, gold.pdf_page_index), ()):
                    if neighbour.id in grades:
                        continue
                    if (
                        jaccard(target, shingles(neighbour.text))
                        >= neighbour_min_jaccard
                    ):
                        grades[neighbour.id] = GRADE_ADMISSIBLE
                        reasons[neighbour.id] = "same_page"
            if not include_section_neighbour or not gold.section_path:
                continue
            section = by_section.get((gold.book_id, gold.section_path), ())
            for neighbour in section:
                if neighbour.id in grades:
                    continue
                if abs(neighbour.ordinal - gold.ordinal) > section_window:
                    continue
                if jaccard(target, shingles(neighbour.text)) >= neighbour_min_jaccard:
                    grades[neighbour.id] = GRADE_ADMISSIBLE
                    reasons[neighbour.id] = "section_neighbor"

        for chunk_id in (duplicates or {}).get(index, ()):
            if chunk_id not in grades:
                grades[chunk_id] = GRADE_ADMISSIBLE
                reasons[chunk_id] = "duplicate_text"

        result.append(
            MultiGoldCase(
                index=index,
                question=case.question,
                grades=grades,
                reasons=reasons,
            )
        )
    return tuple(result)


def shingles(text: str, *, size: int = 3) -> frozenset[int]:
    """Хэши n-грамм слов: устойчивое к правкам сравнение текстов."""
    words = [word.casefold() for word in _WORD_RE.findall(text)]
    if len(words) < size:
        payload = " ".join(words)
        return frozenset({_hash(payload)}) if payload else frozenset()
    return frozenset(
        _hash(" ".join(words[start : start + size]))
        for start in range(len(words) - size + 1)
    )


def jaccard(left: frozenset[int], right: frozenset[int]) -> float:
    if not left or not right:
        return 0.0
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def find_duplicate_neighbors(
    conn: sqlite3.Connection,
    text: str,
    *,
    exclude_ids: Iterable[str] = (),
    exclude_book_ids: Iterable[str] = (),
    min_jaccard: float = 0.6,
    probe_limit: int = 200,
    limit: int = 3,
) -> list[tuple[str, float]]:
    """Ищет почти дословные дубли текста в других чанках каталога.

    Кандидаты берутся из лемма-FTS по редким словам, затем проверяются
    Jaccard'ом 3-грамм. Это нижняя граница: FTS может не вернуть пересказ
    или фрагмент с другой лексикой.
    """
    goal_terms = [token for token in lemmatize_tokens(text) if len(token) >= 6][:10]
    if len(goal_terms) < 3:
        return []
    match = " OR ".join(f'"{term}"' for term in dict.fromkeys(goal_terms))
    excluded = set(exclude_ids)
    excluded_books = set(exclude_book_ids)
    target = shingles(text)
    scored: list[tuple[str, float]] = []
    for row in conn.execute(
        """
        SELECT c.id, c.book_id, c.text
        FROM chunks_fts_lemma
        JOIN chunks c ON c.rowid = chunks_fts_lemma.rowid
        JOIN books b ON b.id = c.book_id
        WHERE chunks_fts_lemma MATCH ? AND b.status = 'ready'
        ORDER BY bm25(chunks_fts_lemma)
        LIMIT ?
        """,
        (match, probe_limit),
    ):
        chunk_id, book_id, candidate_text = row[0], row[1], row[2] or ""
        if chunk_id in excluded or book_id in excluded_books:
            continue
        score = jaccard(target, shingles(candidate_text))
        if score >= min_jaccard:
            scored.append((chunk_id, score))
    scored.sort(key=lambda item: (-item[1], item[0]))
    return scored[:limit]


def _hash(payload: str) -> int:
    digest = hashlib.blake2b(payload.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big")


def load_annotations(path: str | Path) -> tuple[MultiGoldCase, ...]:
    """Читает разметку, сохранённую ``scripts/multigold_eval.py``."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    items = payload.get("cases") if isinstance(payload, dict) else payload
    if not isinstance(items, list) or not items:
        raise ValueError("multigold file must contain a non-empty cases list")
    annotations: list[MultiGoldCase] = []
    for position, item in enumerate(items):
        if not isinstance(item, dict):
            raise ValueError(f"annotation {position} must be an object")
        grades = item.get("grades")
        reasons = item.get("reasons", {})
        if not isinstance(grades, dict) or not grades:
            raise ValueError(f"annotation {position}: grades must be non-empty")
        if not all(
            isinstance(chunk_id, str) and isinstance(grade, int)
            for chunk_id, grade in grades.items()
        ):
            raise ValueError(f"annotation {position}: grades must map str to int")
        annotations.append(
            MultiGoldCase(
                index=int(item.get("index", position)),
                question=str(item.get("question", "")),
                grades=grades,
                reasons=reasons if isinstance(reasons, dict) else {},
            )
        )
    return tuple(annotations)


__all__ = [
    "GRADE_ADMISSIBLE",
    "GRADE_GOLD",
    "ChunkRow",
    "MultiGoldCase",
    "build_multigold",
    "find_duplicate_neighbors",
    "jaccard",
    "load_annotations",
    "load_chunks",
    "shingles",
]
