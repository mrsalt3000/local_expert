"""Runtime-проверка claim→citation (T87).

Проверка opt-in: один batched-вызов судьи на ответ. Возвращает утверждения,
которые не подтверждаются цитируемыми фрагментами. Ошибка провайдера не
ломает ответ — проверка просто пропускается.
"""

from __future__ import annotations

from book_expert.eval.faithfulness import FaithfulnessJudge
from book_expert.models import Citation, SearchHit
from book_expert.providers import Provider, ProviderError


def unsupported_claims(
    provider: Provider,
    question: str,
    answer_text: str,
    citations: tuple[Citation, ...],
    hits: tuple[SearchHit, ...],
) -> tuple[str, ...]:
    if not citations or not hits:
        return ()
    marker_map = {citation.number: citation.number for citation in citations}
    try:
        report = FaithfulnessJudge(provider).evaluate(
            question,
            answer_text,
            tuple(hit.text for hit in hits),
            marker_map=marker_map,
            citation_count=len(citations),
        )
    except (ProviderError, ValueError, TypeError):
        return ()
    return report.unsupported


__all__ = ["unsupported_claims"]
