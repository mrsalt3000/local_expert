"""Тесты определения языка для cross-lingual NLI."""

from __future__ import annotations

from book_expert.eval.language import cyrillic_ratio, detect_language


def test_cyrillic_ratio_ignores_punctuation_and_digits() -> None:
    assert cyrillic_ratio("Привет, мир! 123") == 1.0
    assert cyrillic_ratio("hello world 123") == 0.0


def test_detect_language_russian_english_and_unknown() -> None:
    assert detect_language("Это достаточно длинный русский текст для проверки.") == "ru"
    assert detect_language("This is a long enough English text for detection.") == "en"
    assert detect_language("short") == "unknown"


def test_detect_language_rejects_latin1_mojibake() -> None:
    mojibake = "Áåäíûé Ïàïà Áîãàòûé Ïàïà Ðîáåðò Ò. Êèîñàêè Øàðîí Ë. Ëå÷òåð"
    assert detect_language(mojibake) == "unknown"
