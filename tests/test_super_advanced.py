"""Тесты супер-продвинутого режима: дополнительные возвраты к плану."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from book_expert.config import Settings
from book_expert.models import SearchHit
from book_expert.providers import (
    ChatMessage,
    ChatResult,
    EmbeddingResult,
    ProviderHealth,
)
from book_expert.retrieval.hybrid import SearchOutcome
from book_expert.retrieval.super_advanced import (
    SuperAdvancedAnswerer,
    _parse_refinement,
)


class FakeProvider:
    name = "fake"
    chat_model = "fake-chat"
    embedding_model = "fake-embed"

    def __init__(self, replies: list[str]) -> None:
        self.replies = list(replies)
        self.calls: list[list[ChatMessage]] = []

    def chat(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> ChatResult:
        self.calls.append(list(messages))
        if not self.replies:
            raise AssertionError("unexpected provider call")
        return ChatResult(text=self.replies.pop(0), model="fake-chat")

    def embed(self, texts: Sequence[str]) -> EmbeddingResult:
        raise NotImplementedError

    def health(self) -> ProviderHealth:
        return ProviderHealth(ok=True, detail="fake")


class FakeSearcher:
    def __init__(self, hits: dict[str, tuple[SearchHit, ...]]) -> None:
        self.hits = hits
        self.calls: list[str] = []
        self.book_ids_seen: list[tuple[str, ...]] = []

    def search(
        self,
        query: str,
        *,
        book_ids: Sequence[str] | None = None,
        limit: int = 8,
        dense: bool = True,
    ) -> SearchOutcome:
        self.calls.append(query)
        self.book_ids_seen.append(tuple(book_ids or ()))
        return SearchOutcome(hits=self.hits.get(query, ()))


def _settings(tmp_path: Path, **overrides: object) -> Settings:
    return Settings(
        _env_file=None,
        workspace=tmp_path,
        data_dir=tmp_path / "data",
        **overrides,
    )


def _hit(chunk_id: str) -> SearchHit:
    return SearchHit(
        chunk_id=chunk_id,
        book_id="book-1",
        book_title="Book One",
        text=f"Текст {chunk_id}",
        score=1.0,
        pdf_page_index=1,
    )


def _answerer(tmp_path: Path, provider, searcher, **overrides: object):
    options = {"super_refine_enabled": True, **overrides}
    return SuperAdvancedAnswerer(_settings(tmp_path, **options), provider, searcher)  # type: ignore[arg-type]


def test_scope_limit_truncates_routed_books(tmp_path: Path) -> None:
    provider = FakeProvider(['{"queries": ["q1"]}', "Ответ [1]."])
    searcher = FakeSearcher({"вопрос": (_hit("c1"),), "q1": (_hit("c2"),)})

    SuperAdvancedAnswerer(
        _settings(tmp_path, super_scope_limit=2), provider, searcher
    ).ask("вопрос", book_ids=None, auto_route=True)

    assert all(len(ids) <= 2 for ids in searcher.book_ids_seen)


def test_refinement_is_off_by_default(tmp_path: Path) -> None:
    provider = FakeProvider(
        ['{"queries": ["q1"]}', '{"done": false, "queries": ["q2"]}', "Ответ [1]."]
    )
    searcher = FakeSearcher({"вопрос": (_hit("c1"),), "q1": (_hit("c2"),)})

    answer = SuperAdvancedAnswerer(_settings(tmp_path), provider, searcher).ask(
        "вопрос", book_ids=["book-1"]
    )

    assert answer.rounds == 0
    assert searcher.calls == ["вопрос", "q1"]


def test_refines_once_then_answers(tmp_path: Path) -> None:
    provider = FakeProvider(
        [
            '{"queries": ["q1", "q2"]}',
            '{"done": false, "queries": ["q3"]}',
            '{"done": true}',
            "Развернутый ответ [1] [2] [3] [4].",
        ]
    )
    searcher = FakeSearcher(
        {
            "вопрос": (_hit("c1"),),
            "q1": (_hit("c2"),),
            "q2": (_hit("c3"),),
            "q3": (_hit("c4"),),
        }
    )
    events: list[dict] = []

    answer = _answerer(tmp_path, provider, searcher).ask(
        "вопрос", book_ids=["book-1"], progress=events.append
    )

    assert answer.plan == ("вопрос", "q1", "q2", "q3")
    assert answer.rounds == 1
    assert answer.max_rounds == 5
    assert searcher.calls == ["вопрос", "q1", "q2", "q3"]
    assert [citation.chunk_id for citation in answer.citations] == [
        "c1",
        "c2",
        "c3",
        "c4",
    ]
    assert "round" in [event["type"] for event in events]
    assert events[-1]["type"] == "synthesizing"


def test_stops_after_max_rounds(tmp_path: Path) -> None:
    provider = FakeProvider(
        [
            '{"queries": ["base"]}',
            '{"done": false, "queries": ["q1"]}',
            '{"done": false, "queries": ["q2"]}',
            '{"done": false, "queries": ["q3"]}',
            '{"done": false, "queries": ["q4"]}',
            '{"done": false, "queries": ["q5"]}',
            "Ответ [1].",
        ]
    )
    searcher = FakeSearcher(
        {
            "вопрос": (_hit("c0"),),
            "base": (_hit("c1"),),
            "q1": (_hit("c2"),),
            "q2": (_hit("c3"),),
            "q3": (_hit("c4"),),
            "q4": (_hit("c5"),),
            "q5": (_hit("c6"),),
        }
    )

    answer = _answerer(tmp_path, provider, searcher).ask("вопрос", book_ids=["book-1"])

    assert answer.rounds == 5
    assert answer.max_rounds == 5
    assert len(searcher.calls) == 7
    assert answer.plan[-1] == "q5"


def test_duplicate_refinement_does_not_loop(tmp_path: Path) -> None:
    provider = FakeProvider(
        [
            '{"queries": ["q1"]}',
            '{"done": false, "queries": ["вопрос", "q1"]}',
            "Ответ [1].",
        ]
    )
    searcher = FakeSearcher({"вопрос": (_hit("c1"),), "q1": (_hit("c2"),)})

    answer = _answerer(tmp_path, provider, searcher).ask("вопрос", book_ids=["book-1"])

    assert answer.rounds == 0
    assert answer.plan == ("вопрос", "q1")


def test_done_refinement_skips_extra_rounds(tmp_path: Path) -> None:
    provider = FakeProvider(['{"queries": ["q1"]}', '{"done": true}', "Ответ [1]."])
    searcher = FakeSearcher({"вопрос": (_hit("c1"),), "q1": (_hit("c2"),)})

    answer = _answerer(tmp_path, provider, searcher).ask("вопрос", book_ids=["book-1"])

    assert answer.rounds == 0
    assert answer.plan == ("вопрос", "q1")


def test_custom_rounds_setting(tmp_path: Path) -> None:
    provider = FakeProvider(['{"queries": ["q1"]}', '{"done": true}', "Ответ [1]."])
    searcher = FakeSearcher({"вопрос": (_hit("c1"),), "q1": (_hit("c2"),)})

    answer = _answerer(tmp_path, provider, searcher, super_max_rounds=2).ask(
        "вопрос", book_ids=["book-1"]
    )

    assert answer.max_rounds == 2


def test_string_false_done_is_not_treated_as_true() -> None:
    assert _parse_refinement('{"done": "false", "queries": ["q1"]}', 3) == (
        False,
        ("q1",),
    )
