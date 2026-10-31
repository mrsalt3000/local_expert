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
| Ingestion | `ingestion/pdf.py`, `chunking.py`, `indexer.py` | PDF → страницы → chunks → embeddings → Qdrant |
| Providers | `providers/` | Ollama и OpenAI-compatible chat/embeddings |
| Retrieval | `retrieval/hybrid.py`, `router.py`, `answer.py` | dense + lexical + RRF, router, grounded answer |
| Agent | `agent/graph.py`, `agent/tools/`, `agent/policy.py` | LangGraph react-цикл, RAG/shell/fs/git, audit |
| Eval | `eval/metrics.py`, `eval/parser_assessment.py` | recall/MRR/citations, parser assessment |
| Ops | `doctor.py`, `docker-compose.yml`, `scripts/` | проверки окружения, Qdrant, оценка |
| Interfaces | `cli.py`, `mcp_server.py` | CLI и MCP stdio |
| Web | `web/app.py`, `web/backend.py`, `web/static/` | локальный веб-чат: FastAPI, RAG и агент |

## Потоки

### Ingestion

1. Пользователь: `book-expert ingest <pdf> [--title ...]`.
2. PyMuPDF извлекает native text постранично; фиксируются `pdf_page_index` и
   `printed_page_label`, если метку удалось определить.
3. Chunker режет текст по структуре (заголовки/абзацы), сохраняя page anchor и
   `section_path`; chunk id детерминирован.
4. Embeddings через выбранный провайдер; запись в Qdrant (dense) и FTS5
   (lexical).
5. Каталог обновляет статус книги и content hash. Повторный ingest того же
   файла идемпотентен; изменённый файл переиндексируется.

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

MCP stdio server отдаёт tools `search_books`, `ask_books`, `list_books`,
`get_outline`, чтобы внешние агенты (Qwen Code, Claude Code) могли получать
знания из библиотеки без собственного ingestion.

### Веб-чат

`book-expert ui` поднимает FastAPI на loopback. `GET /api/books` отдаёт только
`ready` книги; `POST /api/chat` с `mode=rag` вызывает `GroundedAnswerer.ask()`
и возвращает citations, с `mode=agent` — `AgentRunner` с теми же tool-политиками
и audit. История диалога живёт в `localStorage` браузера, серверных сессий нет;
ошибки провайдера возвращаются как 503 без traceback.

## Хранение

| Данные | Где | Git |
| --- | --- | --- |
| PDF | `data/raw/` | ignored |
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
  лимиты шагов/таймаутов обязательны. Таймаут агента — caller-side: уже
  выполняющийся граф может продолжить работу; side effects ограничены
  `max_steps`, таймаутами инструментов и bwrap.

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
