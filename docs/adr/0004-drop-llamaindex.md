# ADR-0004: Retrieval без LlamaIndex

- Статус: accepted
- Дата: 2026-10-31
- Уточняет: [ADR-0001](0001-stack.md) в части retrieval-слоя

## Контекст

ADR-0001 предполагал LlamaIndex для node parsing и retriever. В ходе реализации
оказалось, что нужный контур проще и прозрачнее собрать напрямую:

- ingestion уже даёт собственные `Chunk` с page/section anchors;
- dense-поиск — это `qdrant-client` + payload-фильтры;
- lexical-поиск — SQLite FTS5, который уже используется каталогом;
- fusion — RRF на стороне приложения;
- citations строятся из собственных `SearchHit`, без промежуточных node-объектов.

LlamaIndex в `src/` не использовался, но три пакета оставались в зависимостях и
увеличивали lockfile/supply-chain.

## Решение

- Убрать `llama-index-core`, `llama-index-embeddings-ollama` и
  `llama-index-vector-stores-qdrant`.
- Оставить retrieval на `qdrant-client`, SQLite FTS5 и собственном RRF.
- Агент — LangGraph/LangChain, как и было в ADR-0001.

## Последствия

- Меньше зависимостей и меньше слоёв, которые нужно отлаживать.
- Гибридный поиск и citations полностью под контролем проекта.
- Если понадобятся сложные node parsers/retrievers, LlamaIndex можно вернуть
  отдельным ADR, не меняя контракты `SearchHit`/`Citation`.
