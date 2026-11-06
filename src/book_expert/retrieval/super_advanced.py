"""Супер-продвинутый режим: advanced + возвраты к плану за дополнительными запросами."""

from __future__ import annotations

import json
import re
from collections.abc import Sequence

from book_expert.retrieval.advanced import (
    AdvancedAnswerer,
    ProgressCallback,
    _close_json,
)
from book_expert.retrieval.answer import history_prompt
from book_expert.providers import ChatMessage, ProviderError

REFINE_PROMPT = (
    "Ты — исследователь, который ищет по локальной библиотеке книг. "
    "Ниже вопрос, уже выполненный план и краткие описания найденных фрагментов. "
    "Реши, каких аспектов не хватает для полного ответа. "
    'Если информации уже достаточно, верни ровно {"done": true}. '
    'Иначе верни {"done": false, "queries": ["...", "..."]} с 1–3 новыми '
    "поисковыми запросами, которые не повторяют уже выполненные. "
    "Не добавляй пояснений вне JSON."
)

REFINE_QUERY_LIMIT = 3
SUPER_MAX_HITS = 16


class SuperAdvancedAnswerer(AdvancedAnswerer):
    """До `super_max_rounds` раз возвращается к плану за дополнительными запросами."""

    def ask(
        self,
        query: str,
        *,
        book_ids: list[str] | None = None,
        auto_route: bool = True,
        dense: bool = True,
        progress: ProgressCallback | None = None,
        history: Sequence[tuple[str, str]] | None = None,
    ):
        selected, route = self._route(query, book_ids, auto_route)
        if self.settings.super_scope_limit > 0 and selected:
            selected = selected[: self.settings.super_scope_limit]
        plan = list(self._plan(query, history))
        initial_queries = len(plan)
        if self.trace is not None:
            self.trace.write("advanced_plan", query=query, queries=list(plan))
        if progress is not None:
            progress(
                {
                    "type": "plan",
                    "queries": list(plan),
                    "total": len(plan),
                    "max_rounds": self.settings.super_max_rounds,
                }
            )

        hits, seen = self._search(
            plan,
            selected,
            dense,
            progress,
            start_index=1,
            total=len(plan),
            max_hits=SUPER_MAX_HITS,
        )
        rounds = 0
        max_rounds = self.settings.super_max_rounds
        while (
            self.settings.super_refine_enabled
            and rounds < max_rounds
            and len(hits) < SUPER_MAX_HITS
        ):
            done, proposed = self._refine(query, plan, hits, history)
            if self.trace is not None:
                self.trace.write(
                    "super_refine",
                    round=rounds + 1,
                    done=done,
                    queries=list(proposed),
                )
            if done or not proposed:
                break
            known = {item.casefold() for item in plan}
            new_queries = [item for item in proposed if item.casefold() not in known]
            if not new_queries:
                break
            rounds += 1
            start_index = len(plan) + 1
            plan.extend(new_queries)
            if progress is not None:
                progress(
                    {
                        "type": "round",
                        "round": rounds,
                        "max_rounds": max_rounds,
                        "queries": list(new_queries),
                        "total": len(plan),
                    }
                )
            before = len(hits)
            hits, seen = self._search(
                new_queries,
                selected,
                dense,
                progress,
                start_index=start_index,
                total=len(plan),
                max_hits=SUPER_MAX_HITS,
                hits=hits,
                seen=seen,
            )
            if len(hits) == before:
                break

        return self._synthesize(
            query,
            plan,
            hits,
            selected,
            route,
            progress,
            mode="super",
            rounds=rounds,
            max_rounds=max_rounds,
            initial_queries=initial_queries,
            history=history,
        )

    def _refine(
        self,
        query: str,
        plan: list[str],
        hits,
        history: Sequence[tuple[str, str]] | None = None,
    ) -> tuple[bool, tuple[str, ...]]:
        briefs = []
        for index, hit in enumerate(hits[:SUPER_MAX_HITS], start=1):
            text = " ".join(hit.text.split())[:300]
            briefs.append(
                f"[{index}] {hit.book_title or hit.book_id} "
                f"({hit.section_path or hit.printed_page_label or 'фрагмент'}): {text}"
            )
        plan_text = "\n".join(f"{i}. {item}" for i, item in enumerate(plan, 1))
        prompt = (
            history_prompt(history) + f"Вопрос:\n{query.strip()}\n\n"
            f"Уже выполненные запросы:\n{plan_text}\n\n"
            "Кратко о найденном:\n" + ("\n".join(briefs) if briefs else "пока ничего")
        )
        try:
            response = self.provider.chat(
                [
                    ChatMessage(role="system", content=REFINE_PROMPT),
                    ChatMessage(role="user", content=prompt),
                ],
                temperature=0.0,
                max_tokens=300,
            )
        except ProviderError:
            return True, ()
        return _parse_refinement(response.text, REFINE_QUERY_LIMIT)


def _parse_refinement(text: str, limit: int) -> tuple[bool, tuple[str, ...]]:
    raw = text.strip()
    if "```" in raw:
        fenced = re.findall(r"```(?:json)?\s*(.*?)```", raw, re.S)
        if fenced:
            raw = fenced[0].strip()
    data = None
    for attempt in (raw, _close_json(raw)):
        try:
            data = json.loads(attempt)
            break
        except (json.JSONDecodeError, TypeError):
            continue
    if data is None:
        quoted = re.findall(r'"((?:[^"\\]|\\.)*)"', raw)
        candidates = [
            item
            for item in quoted
            if item.casefold() not in {"done", "queries", "true", "false"}
        ]
        if not candidates:
            return True, ()
        data = {"done": False, "queries": candidates}
    raw_done = data.get("done") if isinstance(data, dict) else False
    done = raw_done is True or (
        isinstance(raw_done, str) and raw_done.strip().casefold() == "true"
    )
    queries = data.get("queries") if isinstance(data, dict) else []
    if not isinstance(queries, list):
        queries = []
    result: list[str] = []
    for item in queries:
        if not isinstance(item, str):
            continue
        value = " ".join(item.split()).strip()
        if not 2 <= len(value) <= 300:
            continue
        if value.casefold() in {existing.casefold() for existing in result}:
            continue
        result.append(value)
        if len(result) >= limit:
            break
    if done:
        return True, ()
    return False, tuple(result)


__all__ = ["SuperAdvancedAnswerer", "SUPER_MAX_HITS"]
