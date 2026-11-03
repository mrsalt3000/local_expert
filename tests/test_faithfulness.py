"""Тесты llm-судьи: разбиение на утверждения, парсинг и метрики."""

from __future__ import annotations

import json
import re
from collections.abc import Sequence

import pytest

from book_expert.eval.faithfulness import (
    JUDGE_BATCH_SIZE,
    MAX_CLAIMS,
    FaithfulnessJudge,
    build_prompt,
    parse_verdicts,
    split_claims,
)
from book_expert.providers import (
    ChatMessage,
    ChatResult,
    ProviderError,
    ProviderHealth,
)


class FakeJudgeProvider:
    name = "fake"
    chat_model = "fake-chat"
    embedding_model = "fake-embed"

    def __init__(self, text: str, *, error: Exception | None = None) -> None:
        self.text = text
        self.error = error
        self.messages: list[ChatMessage] = []
        self.max_tokens: int | None = None

    def chat(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> ChatResult:
        self.messages = list(messages)
        self.max_tokens = max_tokens
        if self.error is not None:
            raise self.error
        return ChatResult(text=self.text, model="fake-chat")

    def embed(self, texts: Sequence[str]):  # pragma: no cover - not used
        raise NotImplementedError

    def health(self) -> ProviderHealth:
        return ProviderHealth(ok=True, detail="fake")


def test_split_claims_extracts_markers_and_deduplicates() -> None:
    text = (
        "Первое утверждение [1]. Второе утверждение [2][3]!\n"
        "Первое утверждение [1].  \n"
        "— Пункт списка без маркера."
    )

    claims = split_claims(text)

    assert [claim for claim, _ in claims] == [
        "Первое утверждение.",
        "Второе утверждение!",
        "Пункт списка без маркера.",
    ]
    assert claims[0][1] == (1,)
    assert claims[1][1] == (2, 3)
    assert claims[2][1] == ()


def test_split_claims_ignores_short_fragments() -> None:
    assert split_claims("Да. [1]") == (("Да.", (1,)),)


def test_split_claims_skips_headings_without_markers() -> None:
    text = "Определение\nТекст утверждения.\nПлан:"

    claims = split_claims(text)

    assert [claim for claim, _ in claims] == ["Текст утверждения."]


def test_parse_verdicts_fills_missing_ids_with_false() -> None:
    verdicts = parse_verdicts(
        '{"claims": [{"id": 1, "supported_by_cited": true, '
        '"supported_by_context": true}]}',
        2,
    )

    assert verdicts[0] == {
        "supported_by_cited": True,
        "supported_by_context": True,
    }
    assert verdicts[1] == {
        "supported_by_cited": False,
        "supported_by_context": False,
    }


def test_parse_verdicts_rejects_empty_payload() -> None:
    with pytest.raises(ValueError):
        parse_verdicts('{"claims": []}', 2)


def test_build_prompt_lists_markers_and_sources() -> None:
    prompt = build_prompt(
        "Что такое RAG?",
        (("RAG ищет фрагменты [1]", (1,)),),
        ("RAG combines retrieval and generation.",),
    )

    assert "Что такое RAG?" in prompt
    assert "1. [1] RAG ищет фрагменты" in prompt
    assert '<source id="1">' in prompt


def test_judge_computes_faithfulness_and_citation_support() -> None:
    provider = FakeJudgeProvider(
        '{"claims": ['
        '{"id": 1, "supported_by_cited": true, "supported_by_context": true},'
        '{"id": 2, "supported_by_cited": false, "supported_by_context": true},'
        '{"id": 3, "supported_by_cited": false, "supported_by_context": false}'
        "]}"
    )

    report = FaithfulnessJudge(provider).evaluate(
        "Вопрос",
        "Первое [1]. Второе [2]. Третье [3].",
        ("Первый фрагмент.", "Второй фрагмент."),
    )

    assert report.faithfulness == pytest.approx(0.6667)
    assert report.citation_support_precision == pytest.approx(0.3333)
    assert report.citation_coverage == pytest.approx(1.0)
    assert len(report.unsupported) == 2
    assert report.error is None
    assert "Вопрос" in provider.messages[1].content


def test_judge_maps_markers_to_context_positions() -> None:
    provider = FakeJudgeProvider(
        '{"claims": [{"id": 1, "supported_by_cited": true, '
        '"supported_by_context": true}]}'
    )

    report = FaithfulnessJudge(provider).evaluate(
        "Вопрос",
        "Утверждение [1].",
        ("Первый фрагмент.", "Второй фрагмент."),
        marker_map={1: 2},
    )

    assert "1. [2] Утверждение." in provider.messages[-1].content
    assert report.citation_support_precision == 1.0


def test_judge_returns_zero_without_sources() -> None:
    report = FaithfulnessJudge(FakeJudgeProvider("{}")).evaluate(
        "Вопрос", "Утверждение [1].", ()
    )

    assert report.faithfulness == 0.0
    assert report.citation_support_precision == 0.0
    assert report.unsupported == ("Утверждение.",)


def test_judge_degrades_on_provider_error() -> None:
    provider = FakeJudgeProvider("", error=ProviderError("boom"))

    report = FaithfulnessJudge(provider).evaluate(
        "Вопрос", "Утверждение [1].", ("Фрагмент.",)
    )

    assert report.faithfulness == 0.0
    assert report.error is not None
    assert report.claims[0].judged is False


def test_judge_handles_refused_and_empty_answers() -> None:
    report = FaithfulnessJudge(FakeJudgeProvider("{}")).evaluate(
        "Вопрос", "", ("Фрагмент.",)
    )

    assert report.claims == ()
    assert report.to_dict()["claims"] == 0


def test_split_claims_respects_explicit_limit() -> None:
    text = " ".join(f"Утверждение {index} [1]." for index in range(5))

    assert len(split_claims(text, limit=2)) == 2
    assert len(split_claims(text, limit=None)) == 5


def test_judge_reports_truncated_long_answers() -> None:
    answer = " ".join(
        f"Утверждение номер {index} [1]." for index in range(MAX_CLAIMS + 7)
    )
    provider = FakeJudgeProvider(
        '{"claims": [{"id": 1, "supported_by_cited": true, '
        '"supported_by_context": true}]}'
    )

    report = FaithfulnessJudge(provider).evaluate("Вопрос", answer, ("Фрагмент.",))

    assert len(report.claims) == MAX_CLAIMS
    assert report.claims_total == MAX_CLAIMS + 7
    assert report.truncated is True
    assert provider.max_tokens is not None
    assert provider.max_tokens >= 32 * JUDGE_BATCH_SIZE
    payload = report.to_dict()
    assert payload["truncated"] is True
    assert payload["claims_total"] == MAX_CLAIMS + 7


def test_judge_marks_short_answers_as_not_truncated() -> None:
    provider = FakeJudgeProvider(
        '{"claims": [{"id": 1, "supported_by_cited": true, '
        '"supported_by_context": true}]}'
    )

    report = FaithfulnessJudge(provider).evaluate(
        "Вопрос", "Утверждение [1].", ("Фрагмент.",)
    )

    assert report.truncated is False
    assert report.claims_total == 1


def test_judge_strict_prompt_is_opt_in() -> None:
    payload = (
        '{"claims": [{"id": 1, "supported_by_cited": true, '
        '"supported_by_context": true}]}'
    )
    default = FakeJudgeProvider(payload)
    strict = FakeJudgeProvider(payload)

    FaithfulnessJudge(default).evaluate("Вопрос", "Утверждение [1].", ("Фрагмент.",))
    FaithfulnessJudge(strict, strict=True).evaluate(
        "Вопрос", "Утверждение [1].", ("Фрагмент.",)
    )

    assert "СТРОГО" not in default.messages[0].content
    assert "СТРОГО" in strict.messages[0].content


def test_quote_mode_requires_substring_from_source() -> None:
    source = "В книге сказано: точная цитата подтверждает утверждение полностью."
    valid = json.dumps(
        {
            "claims": [
                {
                    "id": 1,
                    "quote": "точная цитата подтверждает утверждение",
                    "context_quote": "",
                }
            ]
        }
    )
    missing = json.dumps(
        {
            "claims": [
                {
                    "id": 1,
                    "quote": "этой фразы нет в источнике совсем",
                    "context_quote": "",
                }
            ]
        }
    )

    good = FaithfulnessJudge(FakeJudgeProvider(valid), quote_mode=True).evaluate(
        "Вопрос", "Утверждение [1].", (source,)
    )
    bad = FaithfulnessJudge(FakeJudgeProvider(missing), quote_mode=True).evaluate(
        "Вопрос", "Утверждение [1].", (source,)
    )

    assert good.claims[0].supported_by_cited is True
    assert bad.claims[0].supported_by_cited is False


class BatchEchoProvider(FakeJudgeProvider):
    """Отвечает валидным JSON на каждый батч — проверяет склейку вердиктов."""

    def __init__(self) -> None:
        super().__init__("")
        self.calls = 0
        self.batch_sizes: list[int] = []

    def chat(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> ChatResult:
        self.calls += 1
        prompt = messages[-1].content
        count = len(re.findall(r"(?m)^\d+\. \[", prompt))
        self.batch_sizes.append(count)
        payload = {
            "claims": [
                {
                    "id": index,
                    "supported_by_cited": True,
                    "supported_by_context": True,
                }
                for index in range(1, count + 1)
            ]
        }
        return ChatResult(text=json.dumps(payload), model="fake-chat")


def test_judge_merges_verdicts_from_batches() -> None:
    answer = " ".join(f"Утверждение номер {index} [1]." for index in range(30))
    provider = BatchEchoProvider()

    report = FaithfulnessJudge(provider).evaluate("Вопрос", answer, ("Фрагмент.",))

    assert provider.calls == 3
    assert provider.batch_sizes == [JUDGE_BATCH_SIZE, JUDGE_BATCH_SIZE, 6]
    assert len(report.claims) == 30
    assert report.faithfulness == 1.0
    assert report.citation_support_precision == 1.0
    assert report.truncated is False
