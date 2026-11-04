"""Тесты таксономии тем библиотеки."""

from __future__ import annotations

from book_expert.topics import (
    MAX_TOPICS_PER_BOOK,
    TOPICS,
    normalize_topics,
    parse_topics_json,
    topics_to_json,
)


def test_normalize_keeps_canonical_order_and_deduplicates() -> None:
    result = normalize_topics(["IT", "Бизнес", "it", "Психология", "Неизвестная тема"])

    assert result == (
        "Бизнес и предпринимательство",
        "Психология и саморазвитие",
        "IT, программирование, архитектура",
    )


def test_normalize_limits_topics_per_book() -> None:
    result = normalize_topics(TOPICS)

    assert len(result) == MAX_TOPICS_PER_BOOK


def test_normalize_accepts_aliases_and_punctuation() -> None:
    assert normalize_topics(["маркетинг", "Продажи."]) == (
        "Маркетинг, продажи, переговоры",
    )
    assert normalize_topics(["машинное обучение"]) == ("Данные, аналитика, ИИ",)


def test_parse_topics_json_handles_broken_payloads() -> None:
    assert parse_topics_json('["IT", "Бизнес"]') == (
        "Бизнес и предпринимательство",
        "IT, программирование, архитектура",
    )
    assert parse_topics_json("not json") == ()
    assert parse_topics_json('{"topics": ["IT"]}') == ()
    assert parse_topics_json(None) == ()


def test_topics_to_json_round_trip() -> None:
    raw = topics_to_json(["IT", "IT", "Другое"])

    assert parse_topics_json(raw) == (
        "IT, программирование, архитектура",
        "Другое",
    )
