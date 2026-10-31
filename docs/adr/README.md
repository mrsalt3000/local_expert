# ADR

Архитектурные решения (Architecture Decision Records).

## Правила

- Каждое решение — отдельный файл `docs/adr/NNNN-short-title.md`.
- Формат: контекст, решение, альтернативы, последствия.
- Статусы: `proposed`, `accepted`, `superseded`, `deprecated`.
- Решение, меняющее предыдущее, ссылается на него и получает новый номер.

## Реестр

| ADR | Решение | Статус |
| --- | --- | --- |
| [0001](0001-stack.md) | Стек: LangGraph + LlamaIndex + Qdrant + Ollama | accepted (retrieval уточнён в 0004) |
| [0002](0002-local-privacy-artifacts.md) | Приватность и артефакты локального прогона | accepted |
| [0003](0003-agent-autonomy.md) | Автономия агента и границы безопасности | accepted |
| [0004](0004-drop-llamaindex.md) | Retrieval без LlamaIndex | accepted |
