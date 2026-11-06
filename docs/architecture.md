# Architecture

Целевая архитектура локальной book-only RAG-системы с автономным агентом.
Решения зафиксированы в ADR: [0001](adr/0001-stack.md),
[0002](adr/0002-local-privacy-artifacts.md), [0003](adr/0003-agent-autonomy.md).

## Контекст

- Локальный запуск на Ubuntu (WSL 2), GPU RTX 3060 Laptop 6 GB, без облака по
  умолчанию.
- Knowledge policy v1: book-only (одна или несколько легально приобретённых
  книг).
- Приватный текст книги не покидает машину и не попадает в Git.
- Пользователь может указать книгу явно; иначе система подбирает материалы
  сама.

## Компоненты

```
┌─────────────┐      ┌──────────────────┐      ┌─────────────────┐
│  Typer CLI  │      │   MCP server     │      │  LangGraph      │
│ ingest/ask  │      │ search/ask/list  │      │  agent + tools  │
└──────┬──────┘      └────────┬─────────┘      └────────┬────────┘
       │                      │                         │
       └──────────────┬───────┴─────────────┬───────────┘
                      │                     │
              ┌───────▼────────┐    ┌───────▼─────────┐
              │  Retrieval     │    │  Agent tools    │
              │ router + RRF   │    │ shell/fs/git    │
              └───┬────────┬───┘    │ + audit         │
                  │        │        └─────────────────┘
       ┌──────────▼──┐  ┌──▼──────────┐
       │  Qdrant     │  │ SQLite      │
       │  dense      │  │ FTS5 +      │
       │  vectors    │  │ catalog     │
       └─────────────┘  └─────────────┘
                  ▲
       ┌──────────┴──────────┐
       │  Ingestion          │
       │  PyMuPDF → chunks   │
       │  → embeddings       │
       └─────────────────────┘
```

| Слой | Модуль | Ответственность |
| --- | --- | --- |
| Config | `config.py` | env/`.env`, пути, лимиты, выбор провайдера |
| Domain | `models.py` | `Book`, `Chunk`, `Citation`, `Answer`, `SearchHit` |
| Catalog | `catalog.py` | SQLite: книги, hash, статус, метаданные, оглавление, FTS5 |
| Ingestion | `ingestion/extract.py`, `epub.py`, `fb2.py`, `djvu.py`, `chunking.py`, `indexer.py` | PDF/EPUB/FB2/DJVU → страницы/секции → chunks → embeddings → Qdrant |
| Providers | `providers/` | Ollama и OpenAI-compatible chat/embeddings |
| Retrieval | `retrieval/hybrid.py`, `router.py`, `answer.py` | dense + lexical (AND + OR-prefix5) + RRF, router, grounded answer |
| Agent | `agent/graph.py`, `agent/tools/`, `agent/policy.py` | LangGraph react-цикл, RAG/shell/fs/git, audit |
| Eval | `eval/metrics.py`, `eval/stage_eval.py`, `eval/parser_assessment.py` | recall/MRR/citations, per-stage eval + gate, parser assessment |
| Ops | `doctor.py`, `docker-compose.yml`, `scripts/` | проверки окружения, Qdrant, оценка |
| Interfaces | `cli.py`, `mcp_server.py` | CLI и MCP stdio |
| Web | `web/app.py`, `web/backend.py`, `web/static/` | локальный веб-чат: FastAPI, RAG и агент |

## Потоки

### Ingestion

1. Пользователь: `book-expert ingest <book> [--title ...]`.
2. `extract_document` определяет формат по расширению и сигнатуре:
   PyMuPDF для PDF/EPUB, нативный XML для FB2, `djvutxt` для DJVU.
   Фиксируются `pdf_page_index` (универсальный индекс), `printed_page_label`
   (для FB2 — путь секции) и `format`.
3. Chunker режет текст по структуре (заголовки/абзацы), сохраняя page anchor и
   `section_path`; chunk id детерминирован.
4. Embeddings через выбранный провайдер; запись в Qdrant (dense, payload
   `book_format`) и FTS5 (lexical).
5. Каталог обновляет статус книги, формат и content hash. Повторный ingest
   того же файла идемпотентен; изменённый файл переиндексируется.

### Запрос с явной книгой

`ask --book <id>` → фильтр `book_id` → dense + lexical → RRF → top-k →
grounded answer с citations (book/page/chunk).

### Запрос без книги (auto-routing)

1. Router получает вопрос и каталог (названия, описания, оглавление).
2. LLM возвращает `book_ids` + confidence + причину.
3. При confidence ниже порога — fallback на поиск по всей библиотеке.
4. Решение router-а пишется в трейс.

### Агентный режим

LangGraph react-цикл: `rag_search` / `rag_ask` + `shell` / `read_file` /
`write_file` / `list_dir` / `git`. Shell и git выполняются в `bwrap`-песочнице
(ro-root, writable workspace, private `/tmp`, сеть выключена по умолчанию),
политика и лимиты — ADR-0003. Каждый tool-call попадает в audit JSONL; router
и answer пишут решения в trace.

### MCP

MCP stdio server отдаёт **только RAG-tools** `search_books`, `ask_books`,
`list_books`, `get_outline`, чтобы внешние агенты (Qwen Code, Claude Code)
могли получать знания из библиотеки без собственного ingestion. `ask_books`
по умолчанию использует top-8 источников (`limit` настраивается) и
relevance-фильтр: IDF-покрытие редких терминов вопроса, порог
`BOOK_EXPERT_RELEVANCE_FILTER_MIN_RATIO` от лучшего хита, но не меньше трёх
источников. Если по ним модель вернула `INSUFFICIENT_CONTEXT`, ответ
повторяется на полном top-8. Это не даёт citations захватывать хвостовой шум
retrieval, но и не приводит к отказам из-за слишком узкого контекста. Shell,
файловые и git-инструменты через MCP не публикуются. Для Qwen Code есть
project-scope `.qwen/settings.json` с сервером `book-expert`.

### Веб-чат

`book-expert ui` поднимает FastAPI на loopback. По умолчанию UI и API выбирают
режим `super`. `GET /api/books` отдаёт только `ready` книги.
`GET /api/models` отдаёт локальные chat-модели Ollama (без embedding) и
облачные DeepSeek Flash/V4 Pro, если задан ключ. UI в режиме «Агент» передаёт
выбор как `agent_provider`/`agent_model`; `_super_answerer` и `AgentRunner`
кэшируются по паре провайдер/модель, а hard-timeout worker получает копию
`Settings` с этой моделью.
`POST /api/chat` с `mode=rag` вызывает `GroundedAnswerer.ask()`
и возвращает citations, с `mode=advanced` — `AdvancedAnswerer`: планировщик
строит 2–4 запроса, они последовательно выполняются в `HybridSearcher`,
затем модель синтезирует развёрнутый ответ; в ответе есть `plan` и citations.
Для UI есть `POST /api/chat/stream`: SSE-события `plan`, `query_start`,
`query_done`, `round`, `synthesizing` и `done`, по ним рисуется процент,
текущий запрос и номер раунда. Ответ advanced/super считается в отдельном
процессе (`web/answer_worker.py`, `BOOK_EXPERT_ANSWER_HARD_TIMEOUT=true`):
по `BOOK_EXPERT_ANSWER_TIMEOUT_S` (по умолчанию 180 с) процесс убивается,
клиент получает SSE-событие `error` со статусом 504. Ошибка инициализации
бэкенда приходит тем же событием со статусом 503, аварийный выход воркера —
500; таймаут остаётся 504. Режим `mode=super`
(`SuperAdvancedAnswerer`) после начального плана до
`BOOK_EXPERT_SUPER_MAX_ROUNDS` (по умолчанию 5) раз спрашивает refiner-модель,
каких аспектов не хватает, добавляет новые запросы, снова ищет и только затем
синтезирует ответ.
С `mode=agent` работает `AgentRunner` с теми же tool-политиками и audit.
История диалога живёт в `localStorage` браузера: чаты раздельные, у каждого
своя история, режим и книга. Последние 12 реплик активного чата уходят в
`POST /api/chat(.stream)` как `history` и подмешиваются в промпты RAG, advanced,
super и агента; серверных сессий по-прежнему нет.
ошибки провайдера возвращаются как 503 без traceback.

### Библиотека

`GET /api/library` объединяет книги из `data/raw/books` и записи каталога:
показывает, есть ли файл и попал ли он в RAG. `POST /api/library/upload`
сохраняет книгу (sanitize имени, лимит `BOOK_EXPERT_MAX_UPLOAD_MB`, проверка
сигнатуры формата); `POST /api/library/ingest` ставит ingest/reindex в фоновую очередь
с одним воркером и возвращает id задачи, `GET /api/jobs/{id}` отдаёт её статус.
`POST /api/library/remove-from-rag` удаляет Qdrant points и запись каталога,
оставляя файл; `POST /api/library/delete-file` удаляет и файл, и индекс.
Пока задача выполняется, новые ingest и удаления отклоняются с 409.

Пакетный импорт: `POST /api/library/scan` принимает путь к папке (WSL или
Windows-формат) и возвращает количество книг, объём, число уже имеющихся файлов
и пропущенных; `POST /api/library/import` запускает одну фоновую задачу:
рекурсивный обход без симлинков, плоское копирование в `data/raw/books`
с суффиксом при совпадении имён, пропуск файлов с тем же именем и размером,
затем последовательный ingest. Задача отдаёт `processed/total`, счётчики
`copied/ingested/failed`, `remaining` и текущий файл — по ним UI рисует
прогресс-бар. Если книг больше `IMPORT_MAX_FILES`, берётся первый батч,
`remaining` показывает остаток, а следующий запуск пропустит уже
скопированные файлы и продолжит со следующими.
`POST /api/jobs/{id}/cancel` отменяет импорт после текущего файла.

## Хранение

| Данные | Где | Git |
| --- | --- | --- |
| Книги (PDF/EPUB/FB2/DJVU) | `data/raw/` | ignored |
| Extracted text / chunks | `data/processed/` | ignored |
| Qdrant volume | `data/artifacts/qdrant/` | ignored |
| SQLite catalog + FTS5 | `data/artifacts/catalog.sqlite3` | ignored |
| Трейсы, audit, eval reports | `data/artifacts/` | ignored |
| Synthetic fixtures | `tests/fixtures/` | tracked |

## Ограничения

- Локальность: облачный провайдер только opt-in; книга не отправляется наружу
  по умолчанию.
- Наблюдаемость: route decision, answer (hit ids/scores/citations), agent
  start/end и tool audit; `BOOK_EXPERT_TRACE_TEXT` включает текст в trace.
- Приватность: см. ADR-0002; текст книги не логируется и не коммитится.
- Ресурсы: 6 GB VRAM; одновременно держим одну chat-модель и один embedder,
  лимиты шагов/таймаутов обязательны. Веб-агент выполняется в отдельном
  процессе (`agent/worker.py`, `BOOK_EXPERT_AGENT_HARD_TIMEOUT=true`): по
  `BOOK_EXPERT_AGENT_TIMEOUT_S` процесс убивается вместе с сетевыми
  клиентами и инструментами. SSE-ответы advanced/super — тоже в отдельном
  процессе (`web/answer_worker.py`), дедлайн `BOOK_EXPERT_ANSWER_TIMEOUT_S`.
  CLI-агент по-прежнему использует мягкий caller-side таймаут; side effects
  ограничены `max_steps`, таймаутами инструментов и bwrap.

## Отказы и деградация

- Qdrant недоступен → `doctor` и явная ошибка; CLI не пытается «тихо» искать
  только по FTS5, если пользователь не передал `--lexical-only`.
- Модель/embedder недоступны → ошибка с подсказкой запустить `ollama serve`.
- Пустая/сканированная страница → warning в отчёте ingestion, без падения.
- Недостаточно контекста для ответа → явный отказ вместо галлюцинации.

## Non-goals v1

- Мультиязычный OCR и сложный layout recovery.
- Распределённый Qdrant и multi-user доступ.
- Авторизация и удалённый доступ к веб-чату.
- Автоматический деплой и remote-операции без opt-in.
