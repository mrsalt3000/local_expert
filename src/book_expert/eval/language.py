"""Определение языка фрагмента по письменности (cross-lingual NLI, T100)."""

from __future__ import annotations

import re

_CYRILLIC = re.compile(r"[А-Яа-яЁё]")
_LATIN = re.compile(r"[A-Za-z]")


def cyrillic_ratio(text: str) -> float:
    """Доля кириллицы среди букв кириллицы и латиницы."""
    cyrillic = len(_CYRILLIC.findall(text))
    latin = len(_LATIN.findall(text))
    total = cyrillic + latin
    return cyrillic / total if total else 0.0


def detect_language(text: str) -> str:
    """Грубая письменность фрагмента: ``ru``, ``en`` или ``unknown``."""
    cyrillic = len(_CYRILLIC.findall(text))
    latin = len(_LATIN.findall(text))
    total = cyrillic + latin
    if total < 20:
        return "unknown"
    if cyrillic / total >= 0.5:
        return "ru"
    if latin / total >= 0.9:
        return "en"
    return "unknown"


__all__ = ["cyrillic_ratio", "detect_language"]
