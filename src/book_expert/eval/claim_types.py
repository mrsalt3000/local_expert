"""Эвристическая разметка claim: фактическое утверждение или мета/заголовок."""

from __future__ import annotations

import re

META = re.compile(
    r":\s*$|^(Раздел|Глава|Источники|Согласно фрагменту|Далее|Здесь|Этот|"
    r"Данный|Автор|Текст|Фрагмент)|\b(перечисля|подчёркива|указано|"
    r"сформулирова|описыва|рассказыва|говорит|утверждается|отмечается|"
    r"фрагмент|источник|раздел|глава|текст|автор)\b",
    re.IGNORECASE,
)


def classify(text: str) -> tuple[str, str]:
    cleaned = " ".join(text.split())
    if len(cleaned) < 25:
        return "meta", "too short / heading fragment"
    if cleaned.endswith(":"):
        return "meta", "ends with colon"
    match = META.search(cleaned)
    if match:
        return "meta", f"meta marker: {match.group(0)[:40]}"
    return "factual", "declarative statement"


__all__ = ["META", "classify"]
