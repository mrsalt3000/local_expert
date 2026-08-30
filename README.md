# Book Expert

Локальный наблюдаемый book-only RAG-эксперт: вопросно-ответная система по одной
легально приобретённой книге, работающая локально. Текущий статус — foundation
репозиторий: bootstrap без RAG-логики.

## Правила данных

Реальный PDF, извлечённый текст (extracted text), индексы и secrets нельзя
коммитить — соответствующие пути игнорируются `.gitignore`. В репозитории
разрешены только небольшие вручную созданные metadata/evaluation fixtures без
защищённого текста книги (см. `data/eval/README.md`).

## Разработка

```bash
uv sync
uv run ruff format --check .
uv run ruff check .
uv run pytest
uv run pre-commit run --all-files
```
