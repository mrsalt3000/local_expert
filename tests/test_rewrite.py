"""Тесты query rewriting."""

from __future__ import annotations

from book_expert.retrieval.rewrite import rewrite_query


def test_rewrite_strips_command_prefix() -> None:
    assert (
        rewrite_query("Напиши стих про мои путешествия 2025")
        == "стих про мои путешествия 2025"
    )
    assert rewrite_query("Расскажи, как работает retrieval?") == (
        "как работает retrieval?"
    )
    assert rewrite_query("Пожалуйста, объясни что такое RAG") == "что такое RAG"


def test_rewrite_keeps_question_unchanged() -> None:
    question = "Как устроены пайплайны деплоя ML-моделей?"
    assert rewrite_query(question) == question


def test_rewrite_does_not_return_empty_query() -> None:
    assert rewrite_query("Расскажи") == "Расскажи"
    assert rewrite_query("") == ""
