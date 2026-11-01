"""Тесты grounded answer: контекст, отказ и citations."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import pytest

from book_expert.agent.trace import TraceRecorder
from book_expert.config import Settings
from book_expert.models import Answer, SearchHit
from book_expert.providers import (
    ChatMessage,
    ChatResult,
    EmbeddingResult,
    ProviderError,
    ProviderHealth,
)
from book_expert.retrieval.answer import (
    REFUSAL,
    GroundedAnswerer,
    filter_relevant_hits,
)
from book_expert.retrieval.hybrid import SearchOutcome
from book_expert.retrieval.router import BookRoute


class FakeProvider:
    name = "fake"
    chat_model = "fake-chat"
    embedding_model = "fake-embed"

    def __init__(self, text: str = "Answer [1].", *, error: Exception | None = None):
        self.text = text
        self.error = error
        self.calls = 0
        self.messages: list[ChatMessage] = []

    def chat(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> ChatResult:
        self.calls += 1
        self.messages = list(messages)
        if self.error is not None:
            raise self.error
        return ChatResult(text=self.text, model="fake-chat")

    def embed(self, texts: Sequence[str]) -> EmbeddingResult:
        raise NotImplementedError

    def health(self) -> ProviderHealth:
        return ProviderHealth(ok=True, detail="fake")


class SequenceProvider(FakeProvider):
    def __init__(self, texts: Sequence[str]) -> None:
        super().__init__(texts[0])
        self.texts = list(texts)

    def chat(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> ChatResult:
        self.text = self.texts[min(self.calls, len(self.texts) - 1)]
        return super().chat(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )


class FakeSearcher:
    def __init__(self, hits: Sequence[SearchHit] = ()) -> None:
        self.hits = tuple(hits)
        self.calls: list[dict[str, Any]] = []

    def search(
        self,
        query: str,
        *,
        book_ids: list[str] | None = None,
        limit: int | None = None,
        dense: bool = True,
        lexical: bool = True,
        lexical_query_mode: str | None = None,
    ) -> SearchOutcome:
        self.calls.append(
            {
                "query": query,
                "book_ids": book_ids,
                "limit": limit,
                "dense": dense,
                "lexical": lexical,
                "lexical_query_mode": lexical_query_mode,
            }
        )
        return SearchOutcome(hits=self.hits)


class FakeRouter:
    def __init__(self, route: BookRoute) -> None:
        self.route_value = route
        self.calls = 0

    def route(self, query: str, **kwargs: Any) -> BookRoute:
        self.calls += 1
        return self.route_value


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        _env_file=None,
        workspace=tmp_path,
        data_dir=tmp_path / "data",
        top_k=4,
    )


def _hits() -> tuple[SearchHit, ...]:
    return (
        SearchHit(
            chunk_id="c1",
            book_id="book-1",
            book_title="Book One",
            text="Alpha relevant passage about retrieval.",
            score=0.9,
            pdf_page_index=2,
            printed_page_label="12",
            section_path="Chapter 1",
        ),
        SearchHit(
            chunk_id="c2",
            book_id="book-2",
            book_title="Book Two",
            text="Beta extra passage.",
            score=0.5,
            pdf_page_index=0,
        ),
    )


def _answerer(
    tmp_path: Path,
    provider: FakeProvider,
    searcher: FakeSearcher,
    router: FakeRouter | None = None,
) -> GroundedAnswerer:
    return GroundedAnswerer(
        _settings(tmp_path),
        provider,
        searcher,
        router,
    )


def test_citation_carries_format_and_section(tmp_path: Path) -> None:
    hits = (
        SearchHit(
            chunk_id="c1",
            book_id="book-1",
            book_title="FB2 Book",
            text="Alpha relevant passage.",
            score=0.9,
            pdf_page_index=0,
            printed_page_label="Пролог",
            section_path="Пролог",
            book_format="fb2",
        ),
    )
    provider = FakeProvider("Ответ по фрагменту [1].")
    answerer = _answerer(tmp_path, provider, FakeSearcher(hits))

    answer = answerer.ask("О чём пролог?", book_ids=["book-1"])

    assert answer.citations[0].format == "fb2"
    assert answer.citations[0].section_path == "Пролог"


def test_relevance_filter_keeps_sources_with_rare_query_terms() -> None:
    hits = (
        SearchHit(
            chunk_id="c1",
            book_id="book-1",
            book_title="Book",
            text="Kubernetes deployment pipelines and rollout strategies.",
            score=0.6,
            pdf_page_index=0,
        ),
        SearchHit(
            chunk_id="c2",
            book_id="book-1",
            book_title="Book",
            text="A generic chapter about the history of databases.",
            score=0.5,
            pdf_page_index=0,
        ),
        SearchHit(
            chunk_id="c3",
            book_id="book-1",
            book_title="Book",
            text="Deployment is mentioned once here.",
            score=0.4,
            pdf_page_index=0,
        ),
    )

    kept = filter_relevant_hits(
        "Kubernetes deployment pipelines",
        hits,
        min_ratio=0.8,
        min_keep=1,
    )

    assert [hit.chunk_id for hit in kept] == ["c1"]


def test_relevance_filter_keeps_at_least_three_sources_by_idf() -> None:
    hits = (
        SearchHit(
            chunk_id="c1",
            book_id="book-1",
            book_title="Book",
            text="Kubernetes deployment pipelines and rollout strategies.",
            score=0.6,
            pdf_page_index=0,
        ),
        SearchHit(
            chunk_id="c2",
            book_id="book-1",
            book_title="Book",
            text="A generic chapter about the history of databases.",
            score=0.5,
            pdf_page_index=0,
        ),
        SearchHit(
            chunk_id="c3",
            book_id="book-1",
            book_title="Book",
            text="Deployment is mentioned once here.",
            score=0.4,
            pdf_page_index=0,
        ),
        SearchHit(
            chunk_id="c4",
            book_id="book-1",
            book_title="Book",
            text="Pipelines are mentioned once here.",
            score=0.3,
            pdf_page_index=0,
        ),
        SearchHit(
            chunk_id="c5",
            book_id="book-1",
            book_title="Book",
            text="Another generic passage.",
            score=0.2,
            pdf_page_index=0,
        ),
    )

    kept = filter_relevant_hits(
        "Kubernetes deployment pipelines",
        hits,
        min_ratio=0.9,
        min_keep=3,
    )

    assert [hit.chunk_id for hit in kept] == ["c1", "c3", "c4"]


def test_relevance_filter_keeps_all_without_term_overlap() -> None:
    hits = (
        SearchHit(
            chunk_id="c1",
            book_id="book-1",
            book_title="Book",
            text="Alpha beta gamma.",
            score=0.6,
            pdf_page_index=0,
        ),
        SearchHit(
            chunk_id="c2",
            book_id="book-1",
            book_title="Book",
            text="Delta epsilon zeta.",
            score=0.5,
            pdf_page_index=0,
        ),
    )

    kept = filter_relevant_hits("unrelated question", hits)

    assert [hit.chunk_id for hit in kept] == ["c1", "c2"]


def test_relevance_filter_keeps_single_hit() -> None:
    hit = SearchHit(
        chunk_id="c1",
        book_id="book-1",
        book_title="Book",
        text="Anything.",
        score=0.6,
        pdf_page_index=0,
    )

    assert filter_relevant_hits("query", (hit,)) == (hit,)


def test_answer_relevance_filter_limits_citations(tmp_path: Path) -> None:
    hits = _hits() + (
        SearchHit(
            chunk_id="c3",
            book_id="book-1",
            book_title="Book One",
            text="Gamma generic passage.",
            score=0.4,
            pdf_page_index=3,
        ),
        SearchHit(
            chunk_id="c4",
            book_id="book-1",
            book_title="Book One",
            text="Delta generic passage.",
            score=0.3,
            pdf_page_index=4,
        ),
        SearchHit(
            chunk_id="c5",
            book_id="book-1",
            book_title="Book One",
            text="Epsilon generic passage.",
            score=0.2,
            pdf_page_index=5,
        ),
    )
    provider = FakeProvider("Answer [1].")
    searcher = FakeSearcher(hits)
    answerer = _answerer(tmp_path, provider, searcher)

    answer = answerer.ask(
        "How does retrieval work?",
        book_ids=["book-1"],
        relevance_filter=True,
    )

    assert [citation.chunk_id for citation in answer.citations] == ["c1"]
    assert "Epsilon generic passage." not in provider.messages[1].content
    assert provider.calls == 1


def test_relevance_filter_retries_with_all_hits_on_refusal(tmp_path: Path) -> None:
    hits = _hits() + (
        SearchHit(
            chunk_id="c3",
            book_id="book-1",
            book_title="Book One",
            text="Gamma generic passage.",
            score=0.4,
            pdf_page_index=3,
        ),
        SearchHit(
            chunk_id="c4",
            book_id="book-1",
            book_title="Book One",
            text="Delta generic passage.",
            score=0.3,
            pdf_page_index=4,
        ),
        SearchHit(
            chunk_id="c5",
            book_id="book-1",
            book_title="Book One",
            text="Epsilon generic passage.",
            score=0.2,
            pdf_page_index=5,
        ),
    )
    provider = SequenceProvider(["INSUFFICIENT_CONTEXT", "Fallback answer [5]."])
    searcher = FakeSearcher(hits)
    answerer = _answerer(tmp_path, provider, searcher)

    answer = answerer.ask(
        "How does retrieval work?",
        book_ids=["book-1"],
        relevance_filter=True,
    )

    assert answer.refused is False
    assert answer.text == "Fallback answer [5]."
    assert [citation.chunk_id for citation in answer.citations] == ["c5"]
    assert provider.calls == 2
    assert "Epsilon generic passage." in provider.messages[1].content


def test_answer_adds_missing_selected_book_to_context(tmp_path: Path) -> None:
    class PerBookSearcher(FakeSearcher):
        def __init__(self) -> None:
            super().__init__()
            self.book_hits = {
                "book-1": tuple(
                    SearchHit(
                        chunk_id=f"c{index}",
                        book_id="book-1",
                        book_title="Book One",
                        text=f"Book one passage {index}.",
                        score=0.9 - index / 10,
                        pdf_page_index=index,
                    )
                    for index in range(1, 4)
                ),
                "book-2": (
                    SearchHit(
                        chunk_id="c4",
                        book_id="book-2",
                        book_title="Book Two",
                        text="Book two sleep passage.",
                        score=0.8,
                        pdf_page_index=0,
                    ),
                ),
            }

        def search(
            self,
            query: str,
            *,
            book_ids=None,
            limit=None,
            dense=True,
            lexical=True,
            lexical_query_mode=None,
        ):
            self.calls.append({"book_ids": book_ids, "limit": limit})
            ids = book_ids or ["book-1", "book-2"]
            pool = self.book_hits.get(ids[0], ())
            return SearchOutcome(hits=tuple(pool[: limit or len(pool)]))

    provider = FakeProvider("Answer [3].")
    searcher = PerBookSearcher()
    answerer = _answerer(tmp_path, provider, searcher)

    answer = answerer.ask(
        "How to improve sleep?",
        book_ids=["book-1", "book-2"],
        limit=3,
    )

    assert searcher.calls[-1]["book_ids"] == ["book-2"]
    assert [citation.chunk_id for citation in answer.citations] == ["c4"]


def test_answer_uses_context_and_citations(tmp_path: Path) -> None:
    provider = FakeProvider("Hybrid retrieval combines signals [1].")
    searcher = FakeSearcher(_hits())
    answerer = _answerer(tmp_path, provider, searcher)

    answer = answerer.ask("How does retrieval work?", book_ids=["book-1"])

    assert answer.refused is False
    assert answer.text.startswith("Hybrid retrieval")
    assert [citation.chunk_id for citation in answer.citations] == ["c1"]
    citation = answer.citations[0]
    assert citation.book_title == "Book One"
    assert citation.pdf_page_index == 2
    assert citation.printed_page_label == "12"
    assert citation.quote == "Alpha relevant passage about retrieval."
    assert searcher.calls[0]["book_ids"] == ["book-1"]

    prompt = provider.messages[1].content
    assert '<source id="1" book="Book One" page="12" section="Chapter 1">' in prompt
    assert "Alpha relevant passage about retrieval." in prompt
    assert "How does retrieval work?" in prompt


def test_citations_follow_markers(tmp_path: Path) -> None:
    provider = FakeProvider("First [1], second [2], again [1].")
    answerer = _answerer(tmp_path, provider, FakeSearcher(_hits()))

    answer = answerer.ask("question", book_ids=["book-1", "book-2"])

    assert [citation.chunk_id for citation in answer.citations] == ["c1", "c2"]


def test_missing_markers_fall_back_to_all_hits(tmp_path: Path) -> None:
    provider = FakeProvider("Answer without numeric markers.")
    answerer = _answerer(tmp_path, provider, FakeSearcher(_hits()))

    answer = answerer.ask("question", book_ids=["book-1"])

    assert [citation.chunk_id for citation in answer.citations] == ["c1", "c2"]


def test_out_of_range_markers_produce_no_citations(tmp_path: Path) -> None:
    provider = FakeProvider("Only [99] is invalid.")
    answerer = _answerer(tmp_path, provider, FakeSearcher(_hits()))

    answer = answerer.ask("question", book_ids=["book-1"])

    assert answer.citations == ()


def test_insufficient_context_refuses(tmp_path: Path) -> None:
    provider = FakeProvider("INSUFFICIENT_CONTEXT")
    answerer = _answerer(tmp_path, provider, FakeSearcher(_hits()))

    answer = answerer.ask("question", book_ids=["book-1"])

    assert answer.refused is True
    assert answer.text == REFUSAL
    assert answer.citations == ()


@pytest.mark.parametrize(
    "text",
    [
        "Во фрагментах нет ответа.\n\n**INSUFFICIENT_CONTEXT**",
        "insufficient_context",
        "INSUFFICIENT_CONTEXT — в книге об этом не сказано.",
    ],
)
def test_insufficient_context_marker_is_detected_anywhere(
    tmp_path: Path, text: str
) -> None:
    provider = FakeProvider(text)
    answerer = _answerer(tmp_path, provider, FakeSearcher(_hits()))

    answer = answerer.ask("question", book_ids=["book-1"])

    assert answer.refused is True
    assert answer.text == REFUSAL
    assert answer.citations == ()


def test_empty_response_refuses(tmp_path: Path) -> None:
    provider = FakeProvider("   ")
    answerer = _answerer(tmp_path, provider, FakeSearcher(_hits()))

    answer = answerer.ask("question", book_ids=["book-1"])

    assert answer.refused is True
    assert answer.text == REFUSAL
    assert answer.citations == ()


def test_no_hits_refuses_without_llm(tmp_path: Path) -> None:
    provider = FakeProvider()
    answerer = _answerer(tmp_path, provider, FakeSearcher())

    answer = answerer.ask("question", book_ids=["book-1"])

    assert isinstance(answer, Answer)
    assert answer.refused is True
    assert answer.text == REFUSAL
    assert provider.calls == 0


def test_auto_routing_selects_books(tmp_path: Path) -> None:
    provider = FakeProvider("Answer [1].")
    searcher = FakeSearcher(_hits())
    router = FakeRouter(BookRoute(("book-2",), 0.88, "topic match"))
    answerer = _answerer(tmp_path, provider, searcher, router)

    answer = answerer.ask("question about cooking")

    assert router.calls == 1
    assert searcher.calls[0]["book_ids"] == ["book-2"]
    assert answer.book_ids == ("book-2",)
    assert answer.confidence == 0.88
    assert answer.route_reason == "topic match"


def test_without_router_searches_all(tmp_path: Path) -> None:
    provider = FakeProvider("Answer [1].")
    searcher = FakeSearcher(_hits())
    answerer = _answerer(tmp_path, provider, searcher)

    answer = answerer.ask("question", auto_route=False)

    assert searcher.calls[0]["book_ids"] is None
    assert answer.book_ids == ("book-1", "book-2")


def test_provider_error_propagates(tmp_path: Path) -> None:
    provider = FakeProvider(error=ProviderError("down"))
    answerer = _answerer(tmp_path, provider, FakeSearcher(_hits()))

    with pytest.raises(ProviderError, match="down"):
        answerer.ask("question", book_ids=["book-1"])


def test_answer_writes_trace(tmp_path: Path) -> None:
    trace = TraceRecorder(tmp_path / "trace.jsonl")
    provider = FakeProvider("Answer [1].")
    answerer = GroundedAnswerer(
        _settings(tmp_path),
        provider,
        FakeSearcher(_hits()),
        trace=trace,
    )

    answerer.ask("secret question", book_ids=["book-1"])

    record = json.loads(trace.path.read_text(encoding="utf-8"))
    assert record["event"] == "answer"
    assert record["citation_ids"] == ["c1"]
    assert record["refused"] is False
    assert record["query"] == "<15 chars>"
    assert "secret question" not in trace.path.read_text(encoding="utf-8")


def test_citations_keep_marker_numbers(tmp_path: Path) -> None:
    provider = FakeProvider("Второй источник [2], затем первый [1].")
    answerer = _answerer(tmp_path, provider, FakeSearcher(_hits()))

    answer = answerer.ask("Вопрос?", book_ids=["book-1"])

    assert [citation.number for citation in answer.citations] == [1, 2]


def test_history_is_included_in_answer_prompt(tmp_path: Path) -> None:
    provider = FakeProvider("Ответ [1].")
    answerer = _answerer(tmp_path, provider, FakeSearcher(_hits()))

    answerer.ask(
        "А подробнее?",
        book_ids=["book-1"],
        history=[("user", "Что такое поиск?"), ("assistant", "Это про фрагменты.")],
    )

    prompt = provider.messages[-1].content
    assert "Предыдущий диалог" in prompt
    assert "Что такое поиск?" in prompt
    assert "А подробнее?" in prompt
