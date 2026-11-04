# Гигиена ingestion: stale indexing и byte-aware embeddings (T89, 2026-11-04)

## Что изменено

- `Catalog.reconcile_stale_indexing(older_than_minutes=120)` переводит
  осиротевшие `indexing` в `failed` с причиной `stale indexing: worker did not
  finish; reindex is required`. Вызывается при старте `LibraryService`.
- `BookIndexer._embed` батчит тексты не только по 32 чанкам, но и по суммарному
  UTF-8 объёму (`EMBED_MAX_BYTES = 8 MiB`). Один чанк больше лимита даёт
  `IndexingError` с actionable-сообщением вместо 45-мегабайтного payload'а.
- Retry failed остаётся идемпотентным: кнопка «Повторить» в UI и
  `book-expert ingest <path> --force` идут через один и тот же `BookIndexer`.

## Живая проверка

- На реальном каталоге reconciliation перевёл **1** зависшую книгу («Свобода
  учиться», `indexing` с 2026-11-01) в `failed`.
- `catalog-audit` после: **1160 ready, 0 indexing, 23 failed**, дублей и
  пропавших файлов нет.
- Unit-тесты: `tests/test_catalog.py::test_reconcile_stale_indexing`,
  `tests/test_indexer.py::test_batched_texts_*` — 4 сценария (count/bytes/
  oversized/stale cutoff).
- Живой retry 465-страничной payload-книги не завершился: CLI отдаёт
  обобщённое `qdrant is unavailable`, хотя `/healthz` и прямой `QdrantClient`
  отвечают; ошибка маскирует исходный HTTP-сбой и требует отдельного разбора.
  Это ограничение зафиксировано, байтовый батчинг проверен unit-тестами.

## Остаток

23 failed-книги: 5 без текстового слоя (нужен OCR) и payload-книги, которые
теперь должны проходить после byte-aware batching, но требуют повторного
ingest. Ошибка `qdrant is unavailable` в CLI — кандидат в отдельную задачу:
сейчас она скрывает настоящую причину отказа.
