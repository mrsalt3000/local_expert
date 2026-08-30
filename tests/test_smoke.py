"""Smoke-тест: пакет book_expert импортируется успешно."""


def test_import_book_expert() -> None:
    import book_expert

    assert book_expert.__name__ == "book_expert"
