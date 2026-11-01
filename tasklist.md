# Tasklist — Book Expert

План разработки локального book-only RAG-эксперта с агентным режимом.
Статусы: `todo` → `in_progress` → `review` → `accepted` → `done`.

## Цель

Локально запускаемая система, которая:

1. загружает книги (PDF) в RAG-хранилище;
2. отвечает на вопросы пользователя с цитатами на книгу/страницу;
3. позволяет явно указать книгу либо сама подбирает релевантные материалы;
4. работает как автономный coding-агент с доступом к shell/fs/git;
5. отдаёт знания другим агентам через MCP.

## Архитектурные решения

| Область | Решение |
| --- | --- |
| Агент | LangGraph (react-agent), LangChain tools |
| RAG | Собственный retrieval: Qdrant dense + SQLite FTS5 + RRF (ADR-0004) |
| Vector DB | Qdrant в Docker Compose, одна коллекция `books`, изоляция по payload `book_id` |
| Lexical search | SQLite FTS5 + RRF-фузия с dense-результатами |
| LLM / embeddings | Ollama локально (default), OpenAI-совместимый API опционально через `.env` |
| Parsing | PyMuPDF (native text baseline), Docling — отдельный адаптер |
| Интерфейсы | Typer CLI + MCP stdio server |
| Наблюдаемость | структурированные JSONL-трейсы и audit в `data/artifacts/` (git-ignored) |

## Правила данных

- Реальный PDF, extracted/OCR text, embeddings, индексы, трейсы и audit — только
  локально, в git-ignored путях.
- В репозитории — только synthetic fixtures и метаданные без текста книги.
- Реальные прогоны используют `data/raw/*` и `data/eval/*.local.json`.

## Мультиагентный протокол

1. Dev-агент выполняет задачу в границах scope и запускает проверки.
2. Независимый acceptor проверяет diff, acceptance criteria и негативные сценарии.
3. Defects возвращаются dev-агенту; цикл до вердикта `accepted`.
4. Только после `accepted` — commit и push.
5. Каждая задача — отдельный commit; исправления после коммита — новым commit.
6. Push выполняется после `accepted`; remote `origin` настроен
   (`git@github.com:mrsalt3000/local_expert.git`).

## Очередь задач

### T01 — ADR и базовая документация

- [x] Статус: `accepted`
- Scope: `docs/adr/0001-stack.md`, `0002-local-privacy-artifacts.md`,
  `0003-agent-autonomy.md`; обновить `docs/adr/README.md`,
  `docs/architecture.md`, `docs/threat-model.md`, `README.md`.
- Acceptance: решения зафиксированы, границы данных и автономии описаны,
  противоречий с `AGENTS.md` нет.
- Verify: `uv run pre-commit run --all-files`.

### T02 — Конфигурация, доменные модели, каталог книг

- [x] Статус: `accepted`
- Scope: `src/book_expert/config.py`, `models.py`, `catalog.py`, тесты.
- Acceptance: настройки читаются из `.env`/env; SQLite-каталог хранит книгу,
  content hash, статус, метаданные и оглавление; повторная регистрация
  идемпотентна.
- Verify: `uv run pytest`, `uv run ruff check .`.

### T03 — Qdrant в Docker и диагностика окружения

- [x] Статус: `accepted`
- Scope: `docker-compose.yml`, `scripts/doctor`, `.env.example`, тесты/скрипты.
- Acceptance: `docker compose up -d` поднимает Qdrant; `doctor` проверяет
  Qdrant, Ollama, модели, права на `data/` и возвращает понятные ошибки.
- Verify: `docker compose config`, запуск `doctor` при поднятом Qdrant.

### T04 — Провайдеры моделей и embeddings

- [x] Статус: `accepted`
- Scope: `src/book_expert/providers/`, адаптеры Ollama и OpenAI-compatible.
- Acceptance: единый интерфейс chat/embeddings; выбор провайдера через env;
  ошибки недоступного сервера обрабатываются явно; тесты с fake-клиентом.
- Verify: unit-тесты + локальный smoke с Ollama.

### T05 — Извлечение PDF (PyMuPDF)

- [x] Статус: `accepted`
- Scope: `src/book_expert/ingestion/pdf.py`.
- Acceptance: постраничное извлечение native text, page map
  (`pdf_page_index`, label), обработка пустых/сканированных страниц без падения;
  Docling-адаптер выделен отдельно.
- Verify: synthetic PDF fixture в тестах.

### T06 — Chunking с привязкой к странице и секции

- [x] Статус: `accepted`
- Scope: `src/book_expert/ingestion/chunking.py`.
- Acceptance: детерминированные chunk id, page/section anchors, настраиваемый
  размер/overlap; отсутствие текста книги в tracked fixtures.
- Verify: unit-тесты на synthetic тексте.

### T07 — Индексация в Qdrant и CLI `ingest`

- [x] Статус: `accepted`
- Scope: `src/book_expert/ingestion/indexer.py`, `cli.py`.
- Acceptance: dense + FTS5 индексы; повторный ingest по тому же hash не
  дублирует данные; статус книги в каталоге обновляется; `ingest` сообщает
  статистику.
- Verify: integration-тест с Qdrant, unit-тест идемпотентности.

### T08 — Гибридный поиск и цитаты

- [x] Статус: `accepted`
- Scope: `src/book_expert/retrieval/`.
- Acceptance: dense + lexical + RRF; фильтры по `book_id`/page; результат
  содержит book/page/chunk id и score; пустой результат обрабатывается явно.
- Verify: recall@k на synthetic corpus.

### T09 — Авто-routing по библиотеке

- [x] Статус: `accepted`
- Scope: `src/book_expert/retrieval/router.py`.
- Acceptance: LLM выбирает книги по каталогу; при низкой уверенности —
  fallback на всю библиотеку; решение и причина попадают в трейс.
- Verify: тесты с fake LLM, включая fallback.

### T10 — Grounded answer с цитатами

- [x] Статус: `accepted`
- Scope: `src/book_expert/retrieval/answer.py`.
- Acceptance: ответ опирается только на найденные фрагменты; citations
  проверяемы; при недостатке контекста — явный отказ, без галлюцинаций.
- Verify: unit-тесты + e2e на synthetic корпусе.

### T11 — LangGraph агент и RAG-инструменты

- [x] Статус: `accepted`
- Scope: `src/book_expert/agent/graph.py`, `agent/tools/rag.py`.
- Acceptance: react-цикл вызывает `rag_search`/`rag_ask`, сохраняет state,
  пишет трейс; лимит шагов и таймаут.
- Verify: e2e вопрос по synthetic книге с fake LLM.

### T12 — Инструменты shell/fs/git и audit

- [x] Статус: `accepted`
- Scope: `src/book_expert/agent/tools/`, `agent/audit.py`.
- Acceptance: автономный режим без подтверждений, но path confinement в
  workspace, timeout, лимит вывода, audit JSONL; выход за workspace и опасные
  команды блокируются политикой.
- Verify: негативные тесты (path traversal, timeout, запрещённая команда).

### T13 — MCP server

- [x] Статус: `accepted`
- Scope: `src/book_expert/mcp_server.py`.
- Acceptance: tools `search_books`, `ask_books`, `list_books`, `get_outline`;
  stdio-транспорт; ошибки возвращаются как MCP-errors.
- Verify: тест через MCP-клиент.

### T14 — CLI: chat/ask/search/mcp

- [x] Статус: `accepted`
- Scope: `src/book_expert/cli.py`.
- Acceptance: команды `ingest`, `books`, `search`, `ask`, `chat`, `mcp`,
  `doctor`; коды возврата и читаемый вывод; `--json` для машинного режима.
- Verify: CLI smoke-тесты.

### T15 — Synthetic evaluation

- [x] Статус: `accepted`
- Scope: `src/book_expert/eval/`, `tests/fixtures/`.
- Acceptance: генератор synthetic корпуса без реального текста; метрики
  recall@k, MRR, citation accuracy; отчёт в `data/artifacts/` (ignored).
- Verify: `pytest` + прогон eval.

### T16 — E2E smoke и failure modes

- [x] Статус: `accepted`
- Scope: `tests/e2e/`, документация сценариев.
- Acceptance: ingest → search → ask → agent → MCP проходят на synthetic книге;
  описаны и проверены отказы (нет Qdrant, нет модели, пустая книга).
- Verify: полный `pytest`, ручной smoke по README.

### T17 — Parser assessment на реальной книге

- [x] Статус: `accepted`
- Scope: скрипт прогона PyMuPDF/Docling по local-only sample, заполнение
  `data/eval/parser_assessment.local.json`.
- Acceptance: manifest соответствует schema; артефакты не попадают в git;
  отчёт с выводами по 9 критериям.
- Verify: `git status` чистый, `git check-ignore` подтверждает игнор.

### T18 — Hardening и финальная документация

- [x] Статус: `accepted`
- Scope: README, docs, pre-commit, лимиты ресурсов, security review.
- Acceptance: чистый clone → `uv sync` → `doctor` → ingest synthetic → вопрос;
  все проверки `AGENTS.md` зелёные.
- Verify: `uv sync`, `ruff format --check`, `ruff check`, `pytest`,
  `pre-commit run --all-files`.

### T19 — Веб-чат (RAG + агент)

- [x] Статус: `accepted`
- Scope: `src/book_expert/web/` (`app.py`, `backend.py`,
  `static/index.html`), команда `ui` в `cli.py`, `tests/test_web.py`,
  `README.md`, `docs/architecture.md`, `pyproject.toml` + `uv.lock`
  (fastapi/uvicorn).
- Acceptance: `book-expert ui` поднимает сервер на `127.0.0.1:8000`;
  `GET /api/books` отдаёт только `ready` книги; `POST /api/chat` в режимах
  `rag` и `agent` возвращает ответ, в RAG — цитаты; ошибки провайдера → 503
  без traceback; история диалога только в браузере; тесты не требуют
  Qdrant/Ollama; все проверки `AGENTS.md` зелёные.
- Verify: `uv sync`, `ruff format --check`, `ruff check`, `pytest`,
  `pre-commit run --all-files`, ручной smoke из браузера.

## Журнал

| Дата | Задача | Статус | Примечание |
| --- | --- | --- | --- |
| 2026-10-31 | T01–T18 | todo | План утверждён, стек согласован |
| 2026-10-31 | T01 | review | ADR-0001..0003, architecture, threat-model |
| 2026-10-31 | T02 | review | config, models, SQLite catalog + FTS5 |
| 2026-10-31 | T05 | review | PyMuPDF extraction, page labels |
| 2026-10-31 | T06 | review | chunking, sections, overlap |
| 2026-10-31 | T03 | review | Qdrant v1.20.0 healthy, doctor green |
| 2026-10-31 | T04 | review | Ollama/OpenAI providers, 15 tests |
| 2026-10-31 | T07 | review | indexer + ingest, idempotent, Qdrant |
| 2026-10-31 | T08 | review | hybrid search + RRF |
| 2026-10-31 | T09 | review | book router + fallback |
| 2026-10-31 | T10 | review | grounded answer + citations |
| 2026-10-31 | T11 | review | LangGraph agent + trace |
| 2026-10-31 | T12 | review | shell/fs/git policy + audit |
| 2026-10-31 | T13 | review | MCP server |
| 2026-10-31 | T14 | review | CLI search/ask/chat/mcp/doctor/eval |
| 2026-10-31 | T15 | review | eval metrics + synthetic generator |
| 2026-10-31 | T16 | review | e2e synthetic smoke + failure modes |
| 2026-10-31 | T17 | review | real-book local run, manifest ignored |
| 2026-10-31 | T18 | review | docs, think=false fix, full checks |
| 2026-10-31 | T01–T18 | accepted | независимая приёмка: 179 tests, doctor ok |
| 2026-10-31 | T19 | review | FastAPI UI + RAG/агент, 206 tests; live ask упёрся в SW Power Cap GPU |
| 2026-10-31 | T19 | accepted | live smoke в браузере (RAG + агент, 200), 214 tests; приёмка нашла и закрыла 500 в agent-режиме при недоступном провайдере |
| 2026-11-01 | T20 | accepted | book profiles 1150; DeepSeek routing top1 0.533, local 4B 0.500 без профилей; 356 tests |
| 2026-11-01 | T21 | accepted | lemma-FTS5 547946 chunks; `сон/сна`; page recall@5 0.8947; `_lemma_ready` 0.2 ms после partial index |
| 2026-11-01 | T24 | review | 100 reviewed на 3 книгах; third-book page recall@5 0.958, citation precision 0.771 |

### T20 — Book-level routing index

- [x] Статус: `accepted` (2026-11-01; повторная приёмка после D1/D2)
- Scope: `retrieval/book_profiles.py`, `retrieval/router.py`, CLI
  `router-index`, local-only `data/artifacts/book_profiles.local.json`,
  `tests/test_book_profiles.py`, routing eval.
- Acceptance: profile index строится по title/author/section headings;
  `BookRouter` использует его до lexical fallback; routing eval показывает
  candidate recall и top-1 не хуже текущего; без файла — прежнее поведение.
- Verify: targeted tests, `router_eval`, `pytest`, ruff/pre-commit.

### T21 — Лемматизация FTS5 для русской морфологии

- [x] Статус: `accepted` (2026-11-01; partial index `_lemma_ready` 0.2 ms)
- Scope: dependency `pymorphy3`, lemma-normalization query/index, reindex
  strategy, migration tests, eval на 76 кейсах.
- Acceptance: `сон/сна`, `модель/модели` находят друг друга; нет регрессии
  page recall@5; лексический вес остаётся настраиваемым.
- Verify: 76-case eval, targeted tests, full checks.

### T22 — Contextual chunks и parent-child retrieval

- [ ] Статус: `todo`
- Scope: контекст `title + section_path` перед embedding, parent page/section
  в ответе, reindex one book A/B, eval.
- Acceptance: chunk recall@5 и citation precision растут или не падают;
  citations остаются проверяемыми.
- Verify: A/B eval, full checks.

### T23 — Reranking top-20

- [ ] Статус: `todo`
- Scope: пересмотреть reranker после прошлого провала Jina; сравнить
  cross-encoder/DeepSeek listwise на top-20, latency и качество.
- Acceptance: измеримый прирост page recall@5/citation precision без
  неприемлемой задержки; иначе default остаётся off.
- Verify: eval + latency benchmark, full checks.

### T24 — Расширение eval до 5 книг / 150 вопросов

- [ ] Статус: `review` (2026-11-01; 100 reviewed на 3 книгах; 24 third-book draft проверены вручную)
- Scope: human-review вопросы по 3–5 книгам, per-stage routing/retrieval/
  citation metrics, local-only dataset.
- Acceptance: ≥100 reviewed cases, routing и retrieval метрики считаются
  раздельно, draft не смешивается с reviewed.
- Verify: dataset integrity, eval runs, full checks.
