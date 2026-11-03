"""Тесты eval режима «Агент»."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from book_expert.config import Settings
from book_expert.eval import agent_eval
from book_expert.eval.faithfulness import ClaimJudgement, FaithfulnessReport
from book_expert.eval.metrics import EvalCase
from book_expert.models import Answer, Citation, SearchHit


class FakeCatalog:
    def __init__(self, chunks: list[object]) -> None:
        self.chunks = chunks

    def initialize(self) -> None:
        pass

    def iter_chunks(self) -> list[object]:
        return self.chunks


class FakeProvider:
    name = "fake"
    chat_model = "fake-chat"
    embedding_model = "fake-embed"

    def close(self) -> None:
        pass


class FakeSearcher:
    def close(self) -> None:
        pass


class FakeRouter:
    profiles = None

    def route(self, query: str) -> None:
        return None


class FakeAgentAnswerer:
    def __init__(self, mapping: dict[str, Answer]) -> None:
        self.mapping = mapping
        self.questions: list[str] = []

    def ask(self, question: str, **kwargs: object) -> Answer:
        self.questions.append(question)
        return self.mapping[question]


def _citation(page: int) -> Citation:
    return Citation(
        book_id="book-1",
        book_title="Book One",
        pdf_page_index=page,
        chunk_id=f"c{page}",
        number=1,
    )


def test_agent_eval_aggregates_citations_and_plan(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    chunks = [
        SimpleNamespace(id="c1", pdf_page_index=10, book_id="book-1"),
        SimpleNamespace(id="c2", pdf_page_index=20, book_id="book-1"),
    ]
    answerer = FakeAgentAnswerer(
        {
            "q1": Answer(
                text="Ответ [1].",
                citations=(_citation(10),),
                plan=("q1", "q1 details"),
                rounds=1,
                max_rounds=5,
            ),
            "q2": Answer(text="мало", refused=True),
        }
    )
    monkeypatch.setattr(agent_eval, "Catalog", lambda path: FakeCatalog(chunks))
    monkeypatch.setattr(agent_eval, "get_provider", lambda settings: FakeProvider())
    monkeypatch.setattr(agent_eval, "HybridSearcher", lambda *a, **k: FakeSearcher())
    monkeypatch.setattr(agent_eval, "BookRouter", lambda *a, **k: FakeRouter())
    monkeypatch.setattr(agent_eval, "SuperAdvancedAnswerer", lambda *a, **k: answerer)
    monkeypatch.setattr(
        agent_eval,
        "load_cases",
        lambda path: (
            EvalCase("q1", ("c1",), ("book-1",)),
            EvalCase("q2", ("c2",), ("book-1",)),
        ),
    )

    report = agent_eval.run_agent_eval(Settings(), tmp_path / "cases.json")

    assert report["cases"] == 2
    assert report["mode"] == "super"
    assert report["retrieval"]["page_recall"] == 0.5
    assert report["answers"]["refused"] == 1
    assert report["answers"]["no_citations"] == 0
    assert report["plan"]["avg_queries"] == 1.0
    assert report["plan"]["avg_rounds"] == 0.5
    assert report["meta"]["provider"] == "fake"
    assert report["meta"]["super_scope_limit"] == 3
    assert report["meta"]["super_refine_enabled"] is False
    assert report["meta"]["answer_max_citations"] == 5
    assert report["details"][0]["citation_precision"] == 1.0
    assert answerer.questions == ["q1", "q2"]


class FakeJudge:
    def __init__(self, provider: object) -> None:
        self.provider = provider
        self.calls = 0

    def evaluate(
        self,
        question: str,
        answer_text: str,
        sources: tuple[str, ...],
        *,
        marker_map: dict[int, int] | None = None,
    ):
        self.calls += 1
        return FaithfulnessReport(
            claims=(ClaimJudgement("claim", (1,), True, True),),
            faithfulness=1.0,
            citation_support_precision=1.0,
            citation_coverage=1.0,
        )


def test_agent_eval_judges_faithfulness(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    chunks = [SimpleNamespace(id="c1", pdf_page_index=10, book_id="book-1")]
    answerer = FakeAgentAnswerer(
        {
            "q1": Answer(text="Ответ [1].", citations=(_citation(10),)),
            "q2": Answer(text="мало", refused=True),
        }
    )
    judge = FakeJudge(FakeProvider())
    monkeypatch.setattr(agent_eval, "Catalog", lambda path: FakeCatalog(chunks))
    monkeypatch.setattr(agent_eval, "get_provider", lambda settings: FakeProvider())
    monkeypatch.setattr(agent_eval, "HybridSearcher", lambda *a, **k: FakeSearcher())
    monkeypatch.setattr(agent_eval, "BookRouter", lambda *a, **k: FakeRouter())
    monkeypatch.setattr(agent_eval, "SuperAdvancedAnswerer", lambda *a, **k: answerer)
    monkeypatch.setattr(agent_eval, "FaithfulnessJudge", lambda provider: judge)
    monkeypatch.setattr(
        agent_eval,
        "load_cases",
        lambda path: (
            EvalCase("q1", ("c1",), ("book-1",)),
            EvalCase("q2", ("c1",), ("book-1",)),
        ),
    )

    report = agent_eval.run_agent_eval(
        Settings(), tmp_path / "cases.json", faithfulness=True
    )

    assert report["answers"]["faithfulness"] == 1.0
    assert report["answers"]["citation_support_precision"] == 1.0
    assert report["answers"]["judged_cases"] == 1
    assert report["meta"]["faithfulness"] is True
    assert judge.calls == 1


def test_source_page_recall_uses_full_context() -> None:
    source = SearchHit(
        chunk_id="c1",
        book_id="book-1",
        book_title="Book One",
        text="cited",
        score=1.0,
        pdf_page_index=10,
    )
    context_hit = SearchHit(
        chunk_id="c2",
        book_id="book-1",
        book_title="Book One",
        text="context",
        score=0.5,
        pdf_page_index=20,
    )
    answer = Answer(
        text="Ответ [1].",
        citations=(_citation(10),),
        sources=(source,),
        context=(source, context_hit),
    )

    assert {hit.pdf_page_index for hit in answer.context} == {10, 20}
    assert answer.sources[0].pdf_page_index == 10


def test_agent_eval_counts_context_pages_not_citations(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    chunks = [
        SimpleNamespace(id="c1", pdf_page_index=10, book_id="book-1"),
        SimpleNamespace(id="c2", pdf_page_index=20, book_id="book-1"),
    ]
    source = SearchHit(
        chunk_id="c1",
        book_id="book-1",
        book_title="Book One",
        text="cited",
        score=1.0,
        pdf_page_index=10,
    )
    context_hit = SearchHit(
        chunk_id="c2",
        book_id="book-1",
        book_title="Book One",
        text="context",
        score=0.5,
        pdf_page_index=20,
    )
    answerer = FakeAgentAnswerer(
        {
            "q1": Answer(
                text="Ответ [1].",
                citations=(_citation(10),),
                sources=(source,),
                context=(source, context_hit),
            )
        }
    )
    monkeypatch.setattr(agent_eval, "Catalog", lambda path: FakeCatalog(chunks))
    monkeypatch.setattr(agent_eval, "get_provider", lambda settings: FakeProvider())
    monkeypatch.setattr(agent_eval, "HybridSearcher", lambda *a, **k: FakeSearcher())
    monkeypatch.setattr(agent_eval, "BookRouter", lambda *a, **k: FakeRouter())
    monkeypatch.setattr(agent_eval, "SuperAdvancedAnswerer", lambda *a, **k: answerer)
    monkeypatch.setattr(
        agent_eval,
        "load_cases",
        lambda path: (EvalCase("q1", ("c2",), ("book-1",)),),
    )

    report = agent_eval.run_agent_eval(Settings(), tmp_path / "cases.json")

    assert report["retrieval"]["page_recall"] == 0.0
    assert report["retrieval"]["source_page_recall"] == 1.0


def test_agent_eval_uses_separate_judge_provider(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    requested: list[str] = []

    def fake_provider(settings: Settings) -> SimpleNamespace:
        requested.append(settings.provider)
        return SimpleNamespace(
            name=f"{settings.provider}-provider",
            chat_model=f"{settings.provider}-chat",
            embedding_model="fake-embed",
            close=lambda: None,
        )

    chunks = [SimpleNamespace(id="c1", pdf_page_index=10, book_id="book-1")]
    answerer = FakeAgentAnswerer(
        {"q1": Answer(text="Ответ [1].", citations=(_citation(10),))}
    )
    monkeypatch.setattr(agent_eval, "Catalog", lambda path: FakeCatalog(chunks))
    monkeypatch.setattr(agent_eval, "get_provider", fake_provider)
    monkeypatch.setattr(agent_eval, "HybridSearcher", lambda *a, **k: FakeSearcher())
    monkeypatch.setattr(agent_eval, "BookRouter", lambda *a, **k: FakeRouter())
    monkeypatch.setattr(agent_eval, "SuperAdvancedAnswerer", lambda *a, **k: answerer)
    monkeypatch.setattr(agent_eval, "FaithfulnessJudge", FakeJudge)
    monkeypatch.setattr(
        agent_eval,
        "load_cases",
        lambda path: (EvalCase("q1", ("c1",), ("book-1",)),),
    )

    report = agent_eval.run_agent_eval(
        Settings(provider="deepseek"),
        tmp_path / "cases.json",
        faithfulness=True,
        judge_provider="ollama",
    )

    assert requested == ["deepseek", "ollama"]
    assert report["meta"]["judge_provider"] == "ollama-provider"
    assert report["meta"]["judge_chat_model"] == "ollama-chat"
