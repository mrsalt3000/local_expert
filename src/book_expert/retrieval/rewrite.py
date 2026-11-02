"""Отделение retrieval query от пользовательской команды."""

from __future__ import annotations

import re

_PREFIX_RE = re.compile(
    r"^\s*(?:(?:пожалуйста|please)[,\s]+)?"
    r"(?:(?:можешь|could you|can you)\s+)?"
    r"(?:напиши|расскажи|объясни|подскажи|опиши|сформулируй|составь|"
    r"приведи|дай|помоги|write|explain|tell me|describe|give me|summarize)"
    r"\b[\s,:-]*(?:мне|me)?[\s,:-]*(?:про|о|об|about|on)?[\s,:-]*",
    re.IGNORECASE,
)
_FRAME_RE = re.compile(
    r"^\s*(?:я\s+хочу\s+узнать|мне\s+интересно|i\s+want\s+to\s+know|"
    r"i'?m\s+interested\s+in|what\s+do\s+you\s+know\s+about)\s*[:,]?\s*",
    re.IGNORECASE,
)


def rewrite_query(query: str) -> str:
    """Убирает командную обёртку, сохраняя смысловой контекст."""
    text = _FRAME_RE.sub("", query).strip()
    text = _PREFIX_RE.sub("", text).strip()
    text = text.lstrip(" ,:-")
    if len(text) < 3 or not re.search(r"\w", text):
        return query
    return text


__all__ = ["rewrite_query"]
