"""Тесты продвинутого режима: план запросов → RAG → синтез."""

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
from book_expert.retrieval.advanced import AdvancedAnswerer
from book_expert.retrieval.answer import REFUSAL
from book_expert.retrieval.hybrid import SearchOutcome


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
        self.calls: list[tuple[str, list[str] | None, int, bool]] = []

    def search(
        self,
        query: str,
        *,
        book_ids: Sequence[str] | None = None,
        limit: int = 8,
        dense: bool = True,
    ) -> SearchOutcome:
        self.calls.append(
            (query, list(book_ids) if book_ids is not None else None, limit, dense)
        )
        return SearchOutcome(hits=self.hits.get(query, ()))


def _settings(tmp_path: Path) -> Settings:
    return Settings(_env_file=None, workspace=tmp_path, data_dir=tmp_path / "data")


def _hit(chunk_id: str, page: int = 1) -> SearchHit:
    return SearchHit(
        chunk_id=chunk_id,
        book_id="book-1",
        book_title="Book One",
        text=f"Текст фрагмента {chunk_id}.",
        score=1.0,
        pdf_page_index=page,
        printed_page_label=str(page),
    )


def _answerer(
    tmp_path: Path, provider: FakeProvider, searcher: FakeSearcher
) -> AdvancedAnswerer:
    return AdvancedAnswerer(_settings(tmp_path), provider, searcher)  # type: ignore[arg-type]


def test_plans_then_searches_sequentially_and_synthesizes(tmp_path: Path) -> None:
    provider = FakeProvider(
        [
            '{"queries": ["q1", "q2"]}',
            "Развернутый ответ по двум фрагментам [1] [2].",
        ]
    )
    searcher = FakeSearcher(
        {
            "вопрос": (_hit("c1"),),
            "q1": (_hit("c2"),),
            "q2": (_hit("c1"), _hit("c3")),
        }
    )

    answer = _answerer(tmp_path, provider, searcher).ask("вопрос", book_ids=["book-1"])

    assert answer.plan == ("вопрос", "q1", "q2")
    assert [call[0] for call in searcher.calls] == ["вопрос", "q1", "q2"]
    assert [call[2] for call in searcher.calls] == [4, 4, 4]
    assert [citation.chunk_id for citation in answer.citations] == ["c1", "c2"]
    assert [citation.number for citation in answer.citations] == [1, 2]
    assert "Развернутый ответ" in answer.text


def test_bad_plan_falls_back_to_original_query(tmp_path: Path) -> None:
    provider = FakeProvider(["это не JSON", "Ответ [1]."])
    searcher = FakeSearcher({"вопрос": (_hit("c1"),)})

    answer = _answerer(tmp_path, provider, searcher).ask("вопрос", book_ids=["book-1"])

    assert answer.plan == ("вопрос",)
    assert [call[0] for call in searcher.calls] == ["вопрос"]
    assert answer.citations[0].chunk_id == "c1"


def test_no_hits_refuses_without_answer_call(tmp_path: Path) -> None:
    provider = FakeProvider(['{"queries": ["q1", "q2"]}'])
    searcher = FakeSearcher({})

    answer = _answerer(tmp_path, provider, searcher).ask("вопрос", book_ids=["book-1"])

    assert answer.refused is True
    assert answer.text == REFUSAL
    assert answer.plan == ("вопрос", "q1", "q2")
    assert len(provider.calls) == 1


def test_hits_are_deduplicated_across_queries(tmp_path: Path) -> None:
    provider = FakeProvider(['{"queries": ["q1", "q2"]}', "Ответ [1] [2]."])
    searcher = FakeSearcher(
        {
            "вопрос": (_hit("c1"),),
            "q1": (_hit("c1"), _hit("c2")),
            "q2": (_hit("c2"),),
        }
    )

    answer = _answerer(tmp_path, provider, searcher).ask("вопрос", book_ids=["book-1"])

    assert [citation.chunk_id for citation in answer.citations] == ["c1", "c2"]


def test_plan_is_limited_and_keeps_original_first(tmp_path: Path) -> None:
    provider = FakeProvider(
        [
            '{"queries": ["q1", "q2", "q3", "q4", "q5"]}',
            "Ответ [1].",
        ]
    )
    searcher = FakeSearcher({"вопрос": (_hit("c1"),)})

    answer = _answerer(tmp_path, provider, searcher).ask("вопрос", book_ids=["book-1"])

    assert answer.plan[0] == "вопрос"
    assert len(answer.plan) <= 4


def test_truncated_plan_json_is_repaired(tmp_path: Path) -> None:
    provider = FakeProvider(['{"queries": ["q1", "q2"}', "Ответ [1]."])
    searcher = FakeSearcher({"вопрос": (_hit("c1"),)})

    answer = _answerer(tmp_path, provider, searcher).ask("вопрос", book_ids=["book-1"])

    assert answer.plan == ("вопрос", "q1", "q2")


def test_insufficient_context_refuses(tmp_path: Path) -> None:
    provider = FakeProvider(['{"queries": ["q1"]}', "INSUFFICIENT_CONTEXT"])
    searcher = FakeSearcher({"вопрос": (_hit("c1"),)})

    answer = _answerer(tmp_path, provider, searcher).ask("вопрос", book_ids=["book-1"])

    assert answer.refused is True
    assert answer.citations == ()


def test_progress_events_reflect_plan_and_queries(tmp_path: Path) -> None:
    provider = FakeProvider(['{"queries": ["q1", "q2"]}', "Ответ [1]."])
    searcher = FakeSearcher({"вопрос": (_hit("c1"),), "q1": (_hit("c2"),)})
    events: list[dict] = []

    _answerer(tmp_path, provider, searcher).ask(
        "вопрос", book_ids=["book-1"], progress=events.append
    )

    types = [event["type"] for event in events]
    assert types[0] == "plan"
    assert types[-1] == "synthesizing"
    assert types.count("query_start") == 3
    assert types.count("query_done") == 3
    assert events[0]["total"] == 3


def test_all_plan_queries_run_but_hits_are_capped(tmp_path: Path) -> None:
    provider = FakeProvider(['{"queries": ["q1", "q2", "q3"]}', "Ответ [1]."])
    searcher = FakeSearcher(
        {
            "вопрос": tuple(_hit(f"a{i}") for i in range(4)),
            "q1": tuple(_hit(f"b{i}") for i in range(4)),
            "q2": tuple(_hit(f"c{i}") for i in range(4)),
            "q3": tuple(_hit(f"d{i}") for i in range(4)),
        }
    )

    _answerer(tmp_path, provider, searcher).ask("вопрос", book_ids=["book-1"])

    assert len(searcher.calls) == 4
    prompt = provider.calls[-1][1].content
    assert prompt.count("<source ") == 10
