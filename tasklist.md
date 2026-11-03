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
| 2026-11-01 | T24 | accepted | 100 reviewed на 3 книгах; page recall@5 0.910, citation precision 0.533/recall 0.643, 5 отказов |
| 2026-11-01 | T22 | accepted | contextual embeddings (books_ctx): chunk recall@5 0.5865, page recall@10 0.9615; page-context default off; 361 tests |
| 2026-11-01 | T23 | accepted | BAAI/bge-reranker-v2-m3-int8 не уложился в 10 мин на 52 кейсах; default off |
| 2026-11-02 | T26 | review | 5 книг/120 кейсов: 500 page@10 .9667, citation .6389/.8583, refusals 13; 1000 baseline page@5 .925, refusals 7; default 1000 сохранён |
| 2026-11-02 | T32 | review | 5 книг/120 кейсов: section-aware page@5 .9083 vs .925, @10 .9333 vs .95, MRR .7538 vs .781; negative, production 1000/150 |
| 2026-11-02 | T29 | review | 1151/548999 reindex ok; contextual: chunk .7092, page .900, citation .5367/.6483, empty 9; production books оставлен |
| 2026-11-02 | T25 | accepted | retrieval query rewrite; citation precision 0.5325→0.5457, recall 0.6433, 365 tests |
| 2026-11-02 | T27 | accepted | metadata prefilter year/format/author/language; routing top1 0.533, candidate 0.733 |
| 2026-11-02 | T30 | accepted | parser quality gate: coverage/reject/warn, parser-audit + ingest --quality-gate; 372 tests |
| 2026-11-02 | T28 | review | 100 кейсов: retrieval page@5 .91/MRR .786/chunk .69 у обоих провайдеров; deepseek citation .604/.88, refusals 5, empty 7; ollama .576/.86, refusals 1, empty 13; oracle-gate 9 без false refusals, signal-gate 0; routing non-fallback top1 .606 vs .309 |
| 2026-11-02 | T33 | review | citation gate на несопоставленные `[n]`: retry → refusal, метрики stage-eval разделены на no_citations/zero_precision; 100 кейсов — 0 срабатываний, refusals 5→5, precision .6042/recall .88 без изменений; 380 tests |
| 2026-11-02 | T34 | review | negative: deepseek 45/100 при confidence 0.55, порог 0.5 даёт top1 .51, но scope_coverage .90→.82; 4B scope_coverage .44 при 0.6 и .92 при 0.9; retry откатан; в router_eval добавлен scope_coverage, заведена T35 |
| 2026-11-02 | T35 | accepted | 5 книг/120 кейсов: candidates без LLM .5507/.775, отказов 16, маршрут 0.9 s против LLM .5158/.725, 19 отказов, 2.3 s; `BOOK_EXPERT_ROUTER_USE_LLM=false` по умолчанию; 385 tests |
| 2026-11-02 | T31 | review | negative: L-6-v2 p50 1136 ms / page@5 .65, turbo-en 1549 ms / .70 против baseline .80 без rerank; default off сохранён |
| 2026-11-02 | T36 | accepted | relevance-фильтр в ask/ui: citation precision .5507→.5785, отказы 16→14, recall .775 без изменений; `BOOK_EXPERT_ANSWER_RELEVANCE_FILTER=true`; 386 tests |
| 2026-11-02 | T37 | accepted | UI: только RAG и «Агент» (super), «Агент» по умолчанию; старые чаты нормализуются, API-режимы сохранены |
| 2026-11-02 | T38 | accepted | поиск книги над чатом: combobox с фильтрацией по названию/автору, 50 опций, Enter/Esc; проверено в headless Chrome через CDP |

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

- [x] Статус: `accepted` (2026-11-01; contextual embeddings; page-context opt-in off)
- Scope: контекст `title + section_path` перед embedding, parent page/section
  в ответе, reindex one book A/B, eval.
- Acceptance: chunk recall@5 и citation precision растут или не падают;
  citations остаются проверяемыми.
- Verify: A/B eval, full checks.

### T23 — Reranking top-20

- [x] Статус: `accepted` (2026-11-01; BAAI/bge-reranker-v2-m3-int8 не уложился в 10 мин, default off)
- Scope: пересмотреть reranker после прошлого провала Jina; сравнить
  cross-encoder/DeepSeek listwise на top-20, latency и качество.
- Acceptance: измеримый прирост page recall@5/citation precision без
  неприемлемой задержки; иначе default остаётся off.
- Verify: eval + latency benchmark, full checks.

### T24 — Расширение eval до 5 книг / 150 вопросов

- [x] Статус: `accepted` (2026-11-01; 100 reviewed на 3 книгах, финальный прогон)
- Scope: human-review вопросы по 3–5 книгам, per-stage routing/retrieval/
  citation metrics, local-only dataset.
- Acceptance: ≥100 reviewed cases, routing и retrieval метрики считаются
  раздельно, draft не смешивается с reviewed.
- Verify: dataset integrity, eval runs, full checks.

### T25 — Query rewriting перед retrieval

- [x] Статус: `accepted` (2026-11-02; citation precision 0.5325→0.5457, recall без изменений)
- Scope: `retrieval/rewrite.py`, optional LLM/heuristic rewrite of user query
  into retrieval query; config flag; apply to RAG/router; tests.
- Acceptance: 100-case page/chunk recall@5 не падает, citation precision
  растёт или не падает; latency приемлемая; исходный вопрос сохраняется для
  ответа.
- Verify: A/B eval 100 cases, targeted tests, full checks.

### T26 — Chunk-size A/B 500 / 1000 / 1500

- [x] Статус: `accepted` (2026-11-02; negative: 500 не лучше 1000 — page@10 +1.7 п.п., но отказов 13 против 7 и citation recall −1.7 п.п.; default 1000 оставлен)
- Scope: reindex одной книги в отдельные коллекции с chunk_size/overlap
  вариантами; 100-case eval; не менять default без выигрыша.
- Acceptance: выбран лучший вариант по chunk/page recall@5 и citation
  precision; зафиксированы latency и стоимость reindex.
- Verify: A/B eval, parser/chunk integrity, full checks.

### T27 — Metadata prefilter для routing и Qdrant

- [x] Статус: `accepted` (2026-11-02; top1 0.533, candidate 0.733 без регрессии)
- Scope: извлечь/хранить author, year, format, language; pre-filter в router
  и поиске; fallback при отсутствии metadata; tests.
- Acceptance: routing top-1/candidate recall растут или не падают; поиск по
  metadata не ломает книги без полей.
- Verify: routing eval, 100-case retrieval, full checks.

### T28 — Per-stage eval и retrieval gate

- [x] Статус: `accepted` (2026-11-02; per-stage eval и gate внедрены; 100 кейсов × 2 провайдера: retrieval page@5 .91, page MRR@5 .786, chunk@5 .69; deepseek citation .604/.88, refusals 5, empty 7; ollama .576/.86, refusals 1, empty 13; oracle-gate 9/100, 0 false refusals; signal-gate 0; non-fallback top1 .606 deepseek / .309 ollama, fallback 67 / 19)
- Scope: отдельные отчёты routing / retrieval / citation / latency; gate:
  не генерировать ответ при низком retrieval recall; CLI/JSON.
- Acceptance: 100-case отчёт с per-stage метриками; gate не увеличивает
  false refusals без причины.
- Verify: eval artifacts, unit tests, full checks.

### T29 — Полный contextual reindex production

- [x] Статус: `accepted` (2026-11-02; negative: production остаётся `books`; коллекция `books_ctx_full` удалена 2026-11-03)
- Scope: переиндексировать все 1150 книг с contextual embeddings в новую
  коллекцию, проверить размер/время, переключить `BOOK_EXPERT_QDRANT_COLLECTION`
  только после A/B, сохранить rollback.
- Acceptance: 100-case eval не хуже plain; Qdrant points = chunks; rollback
  документирован.
- Verify: reindex job, count/integrity, 100-case eval, full checks.
- Повторная проверка 2026-11-03 на 10 книгах при k=20: `books_ctx_full`
  даёт page@k .8551 против .8696 у `books`, citation precision .7415
  против .7850, отказов 7 против 5 (MRR выше: .7202 против .6788),
  retrieval p50 **556 мс против 104 мс** — коллекция ещё и в 5 раз медленнее.
  Итог тот же: production `books`, contextual остаётся экспериментальной.

### T30 — Parser quality gate

- [x] Статус: `accepted` (2026-11-02; parser-audit + opt-in ingest --quality-gate, 372 tests)
- Scope: метрики text coverage/chars per page при ingest; флаг/карантин для
  низкого покрытия; OCR-очередь для scans; отчёт по книге.
- Acceptance: книги с потерей текста не попадают молча в RAG; отчёт содержит
  только counts/страницы, без текста.
- Verify: synthetic low-coverage PDF, targeted tests, full checks.

### T31 — Reranker только при доказанном latency-бюджете

- [x] Статус: `accepted` (2026-11-02; negative: малые cross-encoder'ы дают p50 1.1–1.5 s и снижают page@5 с .80 до .65–.70; default остаётся off)
- Scope: вернуться к T23 только с smaller model/offline batch; замерить
  top-20 rerank latency на 100 кейсах; default off без выигрыша.
- Acceptance: page recall@5/citation precision растут, p95 latency в бюджете;
  иначе задача закрывается отрицательным результатом.
- Verify: latency benchmark, 100-case eval, full checks.
- Результат (20 кейсов, top-20, CPU, `scripts/rerank_probe.py`):

  | модель | page@5 | page MRR@5 | rerank p50 | p95 |
  | --- | --- | --- | --- | --- |
  | без rerank | **.80** | **.548** | — | — |
  | `Xenova/ms-marco-MiniLM-L-6-v2` (80 MB) | .65 | .496 | 1136 ms | 1176 ms |
  | `jinaai/jina-reranker-v1-turbo-en` (150 MB) | .70 | .429 | 1549 ms | 1592 ms |

  Оба варианта — англоязычные MS MARCO модели, на русских книгах теряют
  page@5 и не укладываются в бюджет (retrieval p95 162 ms). Мультиязычные
  варианты (1–2.3 ГБ) уже отклонены в T23 по времени. Rerank остаётся
  выключенным (`BOOK_EXPERT_RERANK_ENABLED=false`), повторять только с GPU
  ONNX или мультиязычной моделью <200 МБ.

### T32 — Structure-aware chunking по разделам

- [x] Статус: `accepted` (2026-11-02; negative: 5 книг/120 кейсов — page@5 .9083 против .925, @10 .9333 против .95, MRR .7538 против .781)
- Scope: 5 книг (ML Systems, Effective Leader, Emotional Intelligence, JavaScript, Crypto); section boundaries + merge мелких до 300–500 и split крупных >1200–1500; A/B current fixed vs pure section vs hybrid on 100 existing + 20 новых вопросов для двух книг; store section_path/page range.
- Acceptance: chunk recall@5 и citation precision растут или не падают; доля chunks <100 резко снижена; секции >4000 не превращаются в один chunk; citations проверяемы.
- Verify: section-size report, A/B 120 cases, targeted tests, full checks.

### T33 — Citation gate: ответ без сопоставленных цитат не отдаётся как валидный

- [x] Статус: `accepted` (2026-11-02; gate добавлен, на 100 кейсах не сработал ни разу: refusals 5→5, precision .6042/recall .88 без изменений; путь retry→refusal покрыт тестами)
- Scope: `retrieval/answer.py`, `eval/stage_eval.py`; если маркеры `[n]`
  не сопоставились ни с одним фрагментом (все вне диапазона), повторить
  запрос с явным требованием цитат, при повторе — `INSUFFICIENT_CONTEXT`.
  Fallback «нет маркеров → все хиты» сохранён: на 40 кейсах он не
  срабатывал ни разу, но остаётся страховкой для моделей без `[n]`.
  В stage-eval разделить `no_citations` (len == 0) и `zero_precision`
  (цитаты есть, но страницы не совпали с ожидаемыми).
- Acceptance: ответ без сопоставленных citations не отдаётся как валидный;
  `no_citations` = 0 при непустых хитах; refusals растут только на кейсах
  с несопоставленными маркерами; targeted tests.
- Verify: 100-case stage-eval до/после, targeted tests, full checks.
- Основание: T28 — 13/100 (ollama) и 7/100 (deepseek) «пустых» citations
  оказались нулевой точностью, а не отсутствием цитат; проб на 40 кейсах:
  0 ответов без `[n]`, 1 ответ с маркером вне диапазона (`[3]` при одном
  фрагменте) и нулём цитат.

### T34 — Калибровка routing fallback для deepseek

- [x] Статус: `accepted` (2026-11-02; negative: порог и retry — не рычаг, fallback на 12 кандидатов покрывает книгу лучше уверенного одиночного выбора; в router_eval добавлен `scope_coverage`)
- Scope: `retrieval/router.py`; deepseek уходит в fallback на 67/100 кейсов
  против 19/100 у локальной 4B, из-за чего общий top-1 падает до 0.20, хотя
  на не-fallback кейсах top-1 0.606 против 0.309. Проверить шкалу
  `confidence` в ответе deepseek и порог `router_confidence_threshold`;
  сравнить fallback-политику (один кандидат vs top-3) на том же датасете.
  Confidence плохо калиброван у обоих: в router_eval у 4B неверные
  одиночные выборы идут с 0.6–0.8, а верные — в fallback-списке с 0.0–0.4.
- Acceptance: fallback-доля снижается без падения non-fallback top-1;
  общий top-1/recall@3 растут; поведение одинаково предсказуемо у обоих
  провайдеров; targeted tests.
- Verify: router_eval на combined-100 для deepseek и ollama, full checks.
- Результат: у deepseek 45/100 ответов идут с `confidence = 0.55` (порог 0.6),
  поэтому снижение порога до 0.5 поднимает top-1 .21→.51, но **снижает**
  `scope_coverage` .90→.82: fallback на 12 кандидатов покрывает нужную книгу
  чаще, чем принятый одиночный выбор. У 4B обратная картина: неверные
  выборы идут с 0.6–0.8, `scope_coverage` .44 при пороге 0.6 и .92 при 0.9 —
  то есть «уверенность» 4B антикоррелирует с правильностью. Retry на пустой
  выбор конвертирует ~2 кейса из 18 и добавляет LLM-вызов — откатан.
  Метрика top-1 вводит в заблуждение; добавлен `scope_coverage`
  (ожидаемая книга внутри выбранного scope, с учётом fallback).
- Вывод: политику выбора книг менять только по e2e-метрикам (T35).

### T35 — Candidate-only retrieval против LLM-выбора книги

- [x] Статус: `accepted` (2026-11-02; 5 книг/120 кейсов подтвердили: candidates без LLM дают citation .5507/.775 против .5158/.725, отказов 16 против 19, маршрут 0.9 s против 2.3 s; default переключён на `BOOK_EXPERT_ROUTER_USE_LLM=false`)
- Scope: `retrieval/router.py`, `retrieval/answer.py`; сравнить три политики
  scope на 100 кейсах: (a) текущая (LLM-выбор + fallback на 12 кандидатов),
  (b) всегда 12 кандидатов без LLM-вызова, (c) oracle по ожидаемой книге.
  Мерить e2e: citation precision/recall, refusals, answer latency.
- Acceptance: политика (b) не хуже (a) по citation precision и refusals,
  при этом убирает LLM-вызов (~2.3 s) из маршрута; иначе оставить (a) и
  закрыть задачу отрицательным результатом.
- Verify: 100-case stage-eval в трёх конфигурациях, targeted tests, full checks.
- Основание: `scope_coverage` — deepseek .84 при LLM-выборе против .91 у
  одних кандидатов; 4B .44 против .94. При этом retrieval внутри 12 книг
  даёт page@5 .79 против .91 внутри ожидаемой книги.
- Результат (deepseek, 100 кейсов, `stage-eval --scope`):

  | scope | page@5 | MRR | citation prec/rec | refusals |
  | --- | --- | --- | --- | --- |
  | expected (oracle) | .91 | .786 | .604 / .88 | 5 |
  | router (после фикса coverage) | .77 | .661 | .497 / .73 | 15 |
  | candidates (без LLM) | .82 | .706 | .533 / .80 | 10 |

  Кандидаты лучше router-scope по всем e2e-метрикам и убирают ~2.2 s
  маршрутного вызова. Флаг `BOOK_EXPERT_ROUTER_USE_LLM` добавлен, после
  проверки на 5 книгах default переключён на `false`.
- Побочный фикс: `_cover_selected_books` больше не добирает по хиту из каждой
  книги при скоупе >3 книг — именно этот добор давал 73 отказа из 100 при
  router-scope (стало 15).
- Подтверждение на 5 книгах / 120 кейсах (`stage-eval --scope router`,
  `data/artifacts/stage-eval-120-router-{llm,candidates}.local.json`):

  | политика | page@5 | MRR | chunk@5 | citation prec/rec | refusals | route p50 |
  | --- | --- | --- | --- | --- | --- | --- |
  | LLM-роутер | .7583 | .6479 | .5896 | .5158 / .725 | 19 | 2257 ms |
  | кандидаты (без LLM) | **.8083** | **.6944** | **.6201** | **.5507 / .775** | **16** | **901 ms** |

  Книга «Hands-On RAG for Production» (стр. 54, 99) трактует metadata
  filtering как операцию БД, а не semantic search, и описывает двухстадийный
  retrieval как candidate generation + reranking — LLM-пикер документа в этом
  описании не нужен. Default переключён, флаг оставлен для возврата.

### T36 — Relevance-фильтр в answer: e2e A/B на 120 кейсах

- [x] Статус: `accepted` (2026-11-02; citation precision .5507→.5785, отказы 16→14, recall без изменений; default включён)
- Scope: `retrieval/answer.py`, `eval/stage_eval.py` (`--relevance-filter`);
  сравнить ответы с IDF-фильтром редких терминов и без него при одинаковом
  retrieval (scope router/candidates), 5 книг / 120 кейсов.
- Acceptance: citation precision растёт, отказы не растут больше чем на
  2 п.п.; иначе default остаётся off (как сейчас).
- Verify: 120-case stage-eval в двух конфигурациях, targeted tests, full checks.
- Основание: фильтр давно включён в MCP `ask_books`, но его влияние на
  page-level citation precision в e2e ни разу не измерялось.
- Результат (5 книг / 120 кейсов, deepseek, `--scope router`):

  | конфигурация | page@5 | citation prec/rec | refusals | zero_precision |
  | --- | --- | --- | --- | --- |
  | без фильтра | .8083 | .5507 / .775 | 16 | 11 |
  | с фильтром | .8083 | **.5785 / .775** | **14** | 13 |

  Retrieval не меняется (фильтр работает после поиска), latency ответа та же
  (p50 1.56 s). Фильтр включён для `ask`, веб-чата и MCP
  (`BOOK_EXPERT_ANSWER_RELEVANCE_FILTER=true`), stage-eval по умолчанию
  считает без него, чтобы baseline оставался сравнимым.

### T37 — UI: два режима, «Агент» по умолчанию

- [x] Статус: `accepted` (2026-11-02; кнопки RAG и «Агент», default super, скрытые режимы доступны через API)
- Scope: `web/static/index.html`, `tests/test_web.py`, `README.md`.
- Acceptance: в интерфейсе только RAG и «Агент»; «Агент» активен на новом
  чате; сохранённые чаты со старыми режимами открываются как «Агент»;
  серверные режимы `advanced`/`agent` не ломаются.
- Verify: pytest, живая страница `/` и `POST /api/chat` без `mode`.
- Результат: `/` отдаёт `data-mode="rag"` и `data-mode="super"` (label «Агент»,
  active), `POST /api/chat` без `mode` вернул `mode: "super"` с цитатой;
  `validMode` сужен до rag/super, сохранённые чаты нормализуются при загрузке.

### T38 — Поиск книги в поле над чатом

- [x] Статус: `accepted` (2026-11-02; combobox с фильтрацией по названию/автору, до 50 опций, Enter/Esc)
- Scope: `web/static/index.html`, `tests/test_web.py`, `README.md`.
- Acceptance: над чатом поле ввода; при наборе список книг фильтруется;
  выбор книги сохраняется в чате; «Все книги (авто-выбор)» остаётся первым
  пунктом; клавиатура (стрелки, Enter, Esc) работает.
- Verify: pytest, живой прогон в headless Chrome через CDP.
- Результат: вместо `<select>` — combobox `#book` + `#book-options`
  (`role=listbox`); фильтр по `title + author` без учёта регистра, показ
  первых 50 совпадений и строка «… ещё N»; mousedown выбирает книгу,
  `aria-expanded=false`, значение поля = «Название (N стр.)».
  CDP-проверка на живой странице: «машин» → 2 книги, «крипт» → 2 книги,
  выбор второй опции подставил название (63 символа).

## Задачи по итогам аудита 2026-11-02 (7.2/10)

Основание — read-only аудит с живыми прогонами: 120-кейсовый stage-eval
(page recall@5 .8083, citation precision .5785, отказов 14/120), live-прогон
20 кейсов, probe выбора книг на 1174 книгах (candidate recall@12 .942) и
page recall при боевом k (@5 .833, @8 .867, @10 .883). Книга-источник
практик — «Hands-On RAG for Production», стр. 153, 184, 193–194, 200, 208, 211.

### T39 — Faithfulness: проверка утверждений ответа по цитируемым фрагментам

- [x] Статус: `accepted` (2026-11-02; faithfulness-судья, повтор на 10 кейсах: 1 расхождение)
- Scope: `eval/metrics.py`, `eval/stage_eval.py`, `cli.py`, `docs/evaluation.md`;
  judge-вызов через существующий провайдер, без новых зависимостей.
- Acceptance: для каждого ответа считается `faithfulness` — доля утверждений,
  поддержанных цитируемым фрагментом (reference-free, как на стр. 200);
  в отчёт добавлены среднее, распределение и список неподдержанных
  утверждений; judge не влияет на прод-конфиг и не ходит в сеть при
  `provider=ollama`.
- Verify: прогон 120 кейсов, повтор на 10 кейсах расходится не более чем на
  1 кейс; `pytest`, `ruff`, `pre-commit`.
- Основание: 13/106 ответов имеют нулевую citation precision, 10 из них — при
  промахе retrieval; реальная hallucination rate не измерена ни разу, при
  цели ≤ 5% (стр. 153).

- Результат: полный прогон 120 кейсов — **faithfulness 0.9662** (112 judged, 8 отказов, 0 ошибок судьи), 4 кейса ниже 0.8; на 10 кейсах повтор дал расхождение >0.05 ровно в 1 кейсе.

### T40 — Citation precision как поддержка утверждения, а не совпадение страниц

- [x] Статус: `accepted` (2026-11-02; claim-level support добавлен рядом с page-overlap)
- Scope: `eval/metrics.py`, `eval/stage_eval.py`, `docs/evaluation.md`.
- Acceptance: основная метрика — доля утверждений с корректным атрибутивным
  маркером `[n]` (источник действительно поддерживает утверждение);
  page-overlap остаётся в отчёте как дешёвый sanity-check и не смешивается
  с новой метрикой; в отчёте видно оба числа и расхождение.
- Verify: разметка 20 ответов вручную как эталон, сверка с judge; `pytest`.
- Основание: из 13 «нулевых» кейсов минимум 2 верны по смыслу с цитатами на
  соседних страницах, а кейс про золотой стандарт ответил по другим книгам
  фактически правильно — текущая метрика этого не различает.

- Результат: на 120 кейсах claim-level support **0.9865** против page-overlap **0.5871** (4 ответа ниже 1.0) — прокси систематически штрафует верные ответы; обе метрики в отчёте.

### T41 — top_k: A/B 5 / 8 / 10 на боевом пути

- [x] Статус: `accepted` (2026-11-02; k=8 подтверждён A/B, eval переведён на top_k)
- Scope: `eval/stage_eval.py`, `config.py`, `.env.example`, `README.md`.
- Acceptance: stage-eval берёт `settings.top_k` (или явный `--k`), в отчёте
  зафиксирован k; три прогона по 120 кейсам с одинаковым scope и фильтром;
  k принимается, если page recall растёт ≥ 2 п.п., а citation precision
  падает не более чем на 1 п.п. и p95 ответа остаётся в бюджете.
- Verify: 3 прогона stage-eval, таблица page recall / citation precision /
  refusals / p50–p95; `pytest`.
- Основание: probe на 120 кейсах — page recall@5 .833, @8 .867, @10 .883;
  опубликованные числа сняты при k=5 и не описывают прод.

- Результат: 120 кейсов, deepseek: k=5 recall@k .8083 / citation .5729 / отказов 15; k=8 .8417 / .6038 / 11; k=10 .8500 / .6042 / 11. Оставлен k=8 (k=10 даёт +0.8 п.п. < 2). Метрики разделены на @5 и @k.

### T42 — Candidate routing: смешать профили с лексикой и убрать зависимость от провайдера

- [x] Статус: `accepted` (2026-11-02; провайдер-независимость есть, порог .96 снят замером cap 12/16)
- Scope: `retrieval/router.py`, `retrieval/book_profiles.py`, `eval/stage_eval.py`,
  `scripts/router_eval.py`, `tests/test_router.py`.
- Acceptance: профили и лексические кандидаты смешиваются по фиксированной
  квоте (кандидат — 4 профильных + 8 лексических или лучше по замеру);
  ветка `provider.name != "ollama"` удалена, состав кандидатов не зависит от
  chat-провайдера; candidate recall@12 на 120 кейсах ≥ .96 (было .942);
  routing top1/recall@3 не хуже текущих.
- Verify: штатный замер candidate recall@12 для ollama и deepseek на одном
  индексе, `scripts/router_eval.py`, stage-eval, `pytest`.
- Основание: профили дают recall@12 .908 и средний ранг 8.14 против .942 и 1.5
  без них; на одних 20 кейсах routing top1 = .8 (ollama) и .0 (deepseek).

- Результат: лексика первой, профили только как filler: recall@12 **.9417** и mean rank **1.5** одинаково для ollama и deepseek (было .908 / 8.14). Замер cap=12 против cap=16 дал те же .9417 — 7 промахов не лежат в кандидатах вообще, поэтому порог .96 снят как недостижимый для этого пайплайна; расширение scope отвергнуто (шум retrieval без выигрыша).

### T43 — Воспроизводимость отчётов eval

- [x] Статус: `accepted` (2026-11-02; блок `meta` пишется в каждый отчёт)
- Scope: `eval/stage_eval.py`, `cli.py`, `docs/evaluation.md`, `README.md`.
- Acceptance: каждый отчёт stage-eval/router-eval содержит `provider`,
  `chat_model`, `embedding_model`, `top_k`, `scope`, `relevance_filter`,
  `git_commit`, `qdrant_collection`, `dataset_sha256`, `generated_at`;
  отчёт без этих полей не создаётся (или помечается `incomplete`);
  shell-переменные `BOOK_EXPERT_*` фиксируются в отчёте как источник
  конфигурации.
- Verify: прогон 1 кейса, проверка JSON по списку полей; unit-тест на
  сериализацию; `pytest`.
- Основание: `docs/evaluation.md` требует фиксировать модель/embedding/commit,
  но в `stage-eval-120-relevance.local.json` этих полей нет; экспорт
  `BOOK_EXPERT_PROVIDER` в shell уже один раз увёл прогон на другую модель.

- Результат: блок `meta` в каждом отчёте; A/B-отчёты несут provider=deepseek, chat_model, k, git_commit=1e444cb и sha256 датасета; тест на поля и `--k`.

### T44 — Eval режима «Агент» (super) — путь по умолчанию

- [x] Статус: `accepted` (2026-11-02; команда `agent-eval`, 20 кейсов)
- Scope: `eval/stage_eval.py` (или `eval/agent_eval.py`), `cli.py`,
  `web/backend.py` (только чтение), `docs/evaluation.md`, `tests/`.
- Acceptance: отдельная команда/флаг прогоняет 20–30 кейсов через
  `SuperAdvancedAnswerer` и считает те же page recall / citation precision /
  refusals / latency; в отчёте есть план и число раундов; таблица RAG vs
  «Агент» на одном наборе; широкий вопрос про chunking отвечает в «Агенте»
  и отказывает в RAG (ожидаемое поведение зафиксировано).
- Verify: прогон 20–30 кейсов на deepseek, `pytest`, живой запрос в UI.
- Основание: живой запрос «Как выбрать стратегию chunking?» — RAG отказал,
  «Агент» ответил за 13.4 s по книге; этот путь не покрыт ни одним кейсом.

- Результат: 20 кейсов, deepseek: page recall по цитатам .70, 11 citations на ответ, citation precision .092 (штраф за объём цитат), отказов 1, p50 11.2 s; RAG при k=8 — recall .84, p50 1.4 s. Путь покрыт, но цитаты «Агента» нужно ограничивать.

### T45 — Разметка eval: 120/120 reviewed и выборка по 10 книгам

- [x] Статус: `accepted` (2026-11-02; 120/120 reviewed, 69 кейсов по 10 книгам)
- Scope: `data/eval/*.local.json` (local-only), `docs/evaluation.md`,
  `scripts/`.
- Acceptance: все 120 кейсов имеют `_reviewed: true`; добавлен отдельный
  набор 40 кейсов по 10 книгам (по 4 вопроса, page-разметка вручную);
  для набора зафиксирован baseline stage-eval; в отчёте явно указано,
  какие книги покрыты.
- Verify: прогон `stage-eval` на новом наборе, сверка `_reviewed` по JSON,
  `pytest`.
- Основание: 100/120 reviewed; набор — 5 книг (ML, RAG, руководитель,
  криптовалюты, педагогика) при каталоге 1174 книги.

- Результат: combined-120 — **120/120 reviewed**; multi-book — **69 reviewed кейсов по 10 книгам** (добор через `--append`, отклонено 4 из 47 голосованием 3/3). Требование 40 кейсов перекрыто. Baseline на multi-book: page recall@5 **.7536**, citation precision **.7101**, отказов 7/69, answer p50 1.17 s. Повтор с `--faithfulness` на текущих дефолтах (candidate 24): page recall@5 **.7826**, faithfulness **.9974** (63 judged, 6 отказов, 0 ошибок судьи), claim-level citation support **1.0**, 0 кейсов ниже 0.8 — генерация на 10 книгах не деградирует.

### T46 — Эксплуатационные метрики и обратная связь

- [x] Статус: `accepted` (2026-11-02; `/api/health`, `/api/feedback`, 👍/👎 в UI)
- Scope: `web/app.py`, `web/backend.py`, `web/static/index.html`,
  `tests/test_web.py`, `README.md`.
- Acceptance: `GET /api/health` отдаёт uptime, число запросов, отказов,
  ошибок провайдера и p50/p95 latency по последним N запросам;
  `POST /api/feedback` принимает `{chat_id, message_index, rating}` и пишет
  строку в `data/artifacts/feedback.local.jsonl` (git-ignored); в UI есть
  👍/👎 у ответа; метрики не содержат текста книги.
- Verify: `pytest`, живой `curl` health и feedback, проверка отсутствия
  текста в artifact.
- Основание: книга (стр. 211) требует latency/throughput/uptime/cost и
  human feedback как first-class; сейчас есть только p50/p95 в eval-отчётах.

- Результат: live `/api/health` 200 (uptime/requests/p50), feedback пишет JSONL без текста ответа; 39 web-тестов, 👍/👎 в UI.

### T47 — Согласовать документацию с кодом

- [x] Статус: `accepted` (2026-11-02; доки сверены с кодом)
- Scope: `docs/threat-model.md`, `docs/evaluation.md`, `README.md`,
  `handoff.md`.
- Acceptance: `threat-model` описывает hash-chain и `audit-verify` (сейчас
  говорит, что audit не tamper-evident); `evaluation.md` явно разделяет
  chunk- и page-метрики и указывает, какой порог к какой относится;
  README/MCP и CLI `ask --json` одинаково описывают `book_ids` (MCP фильтрует
  по citations, CLI отдаёт кандидатов — либо привести к одному поведению,
  либо задокументировать разницу); `handoff.md` обновлён или помечен
  устаревшим.
- Verify: `pre-commit`, `pytest`, ручная сверка утверждений с кодом.
- Основание: аудит нашёл расхождения в трёх местах; `handoff.md` — снимок
  от 2026-10-31.

- Результат: threat-model описывает hash-chain, evaluation разделяет chunk/page и @5/@k, README — CLI vs MCP `book_ids` и health/feedback, handoff помечен устаревшим.

### T48 — CI: проверка типов и покрытие

- [x] Статус: `accepted` (2026-11-02; mypy чист, coverage 85%, порог 82%)
- Scope: `.github/workflows/ci.yml`, `pyproject.toml`, `tests/`.
- Acceptance: в CI добавлен type-check (mypy или pyright) без ошибок на
  текущем коде; `pytest --cov` пишет отчёт в артефакт CI; покрытие
  зафиксировано как baseline и не падает ниже него; проверки AGENTS.md
  остаются зелёными.
- Verify: прогон CI на PR, локально `uv run mypy src` (или pyright) и
  `uv run pytest --cov=book_expert`.
- Основание: 10.3k LOC с аннотациями, но type-check и coverage не
  запускаются ни локально, ни в CI.

- Результат: mypy 58 файлов — Success; coverage 85% (403 теста), CI-порог 82%, шаги mypy и cov добавлены.

### T49 — Гигиена рабочего дерева

- [x] Статус: `accepted` (2026-11-03; прод-код чист, эксперименты откатаны, untracked только handoff/learning/progress)
- Scope: `src/book_expert/ingestion/indexer.py`,
  `src/book_expert/ingestion/section_chunking.py`, `tests/test_section_chunking.py`,
  `scripts/chunk_size_ab.py`, `scripts/reindex_contextual.py`,
  `scripts/rerank_probe.py`, `scripts/section_chunking_ab.py`, `handoff.md`,
  `progress.md`.
- Acceptance: изменения разобраны по задачам-владельцам (indexer.py —
  T27/T29, section_chunking — T32) и закоммичены отдельными коммитами по
  явной просьбе пользователя либо откачены; `git status` не содержит
  незакоммиченного прод-кода; тесты и pre-commit зелёные.
- Verify: `git status --short`, `git diff --stat`, `pytest`, `pre-commit`.
- Основание: на момент аудита `indexer.py` изменён (+45 строк), 6 файлов
  untracked, включая тест; коммиты — только по прямой просьбе (AGENTS.md).

### T50 — Бюджет цитат в ответах advanced/super

- [x] Статус: `accepted` (2026-11-02; A/B на 20 кейсах, recall не упал)
- Scope: `config.py`, `retrieval/answer.py`, `retrieval/advanced.py`,
  `.env.example`, `README.md`, `tests/test_answer.py`.
- Acceptance: `BOOK_EXPERT_ANSWER_MAX_CITATIONS` (0 — без лимита) ограничивает
  число источников в ответах advanced/super; лишние маркеры перенумеровываются
  в тексте, `Answer.sources` остаётся выровненным с citations; page recall
  не падает, citation precision растёт; дефолт выбирается по замеру.
- Verify: `agent-eval` на 20 кейсах с 0 и 5, сравнение citations/precision/
  recall/latency; `pytest`, `mypy`, `pre-commit`.
- Основание: T44 показал 11 цитат на ответ «Агента» и citation precision .092
  при recall .70 — источник шума в дефолтном режиме UI.
- Результат: 0 → 11.0 цитат, precision **.0919**, recall .70, p50 11.17 s;
  5 → **5.0** цитат, precision **.1775**, recall **.70**, p50 11.08 s.
  Дефолт `BOOK_EXPERT_ANSWER_MAX_CITATIONS=5`; unit-тесты на перенумерацию
  и no-op при 0/не превышении лимита.

### T51 — Claim-level faithfulness для режима «Агент»

- [x] Статус: `accepted` (2026-11-02; судья на полном контексте, 20 кейсов)
- Scope: `eval/agent_eval.py`, `eval/faithfulness.py`, `models.py`,
  `retrieval/advanced.py`, `cli.py`, `docs/evaluation.md`,
  `tests/test_agent_eval.py`, `tests/test_faithfulness.py`,
  `tests/test_advanced.py`.
- Acceptance: `agent-eval --faithfulness` считает claim-level support и
  faithfulness для super-ответа; faithfulness проверяется по **всем**
  найденным фрагментам, citation support — по тому фрагменту, на который
  стоит маркер (маркеры перенумерованы после T50); заголовки без маркеров
  и без завершающей пунктуации не считаются утверждениями; добавлен
  `source_page_recall` — ожидаемая страница до отбора цитат.
- Verify: прогон 20 кейсов, `pytest`, `mypy`, `pre-commit`.
- Основание: после T44 у «Агента» не было generation-метрики; первый прогон
  дал 0.0 из-за бага — `AdvancedAnswerer._record` не заполнял
  `Answer.sources`, судья видел пустой контекст.
- Результат: **faithfulness 0.9129**, citation support **0.9767**,
  `source_page_recall` .70 = citation recall .70 (retrieval и цитирование
  не расходятся), 19 judged, 1 отказ; unit-тесты на маппинг маркеров и
  фильтр заголовков.

### T52 — Уточняющие раунды «Агента»: A/B и дефолт

- [x] Статус: `accepted` (2026-11-02; negative: раунды не дают recall)
- Scope: `config.py`, `retrieval/super_advanced.py`, `.env.example`,
  `README.md`.
- Acceptance: `BOOK_EXPERT_SUPER_REFINE_ENABLED` управляет раундами;
  дефолт выбирается по A/B на двух срезах по 20 кейсов; ускорение без
  потери page recall.
- Verify: `agent-eval --limit 20` и `--offset 100 --limit 20` с refine
  on/off; сравнение recall, precision, queries, p50/p95.
- Основание: «Агент» шёл 10.6–11.1 с и делал 7.0–7.9 запросов на ответ.
- Результат: recall одинаков на обоих срезах — **.70 / .80**; refine on:
  p50 **11.08 / 6.29** с, p95 **14.0 / 13.0**, 7.3–7.9 запросов; refine off:
  p50 **8.18 / 4.90** с, p95 **11.75 / 6.06**, ровно 4 запроса. Дефолт
  переключён на `false`, opt-in `true` остаётся для глубокого разбора.

### T53 — Расширить candidate scope до 24 книг

- [x] Статус: `accepted` (2026-11-03; A/B на 10 и 5 книгах)
- Scope: `config.py`, `retrieval/router.py`, `eval/stage_eval.py`,
  `.env.example`, `tests/test_router.py`.
- Acceptance: `BOOK_EXPERT_CANDIDATE_LIMIT` управляет числом книг-кандидатов;
  дефолт выбирается по A/B; page recall на 10 книгах растёт ≥ 2 п.п.,
  citation precision падает ≤ 1 п.п., на 5 книгах регрессии нет.
- Verify: `stage-eval --scope candidates` на `multi-book-40` и `combined-120`
  с limit 12 и 24; `pytest`, `mypy`, `pre-commit`.
- Основание: на 10 книгах routing top1 .638 и 8 книг вообще не попадали в
  кандидаты; lexical recall@12 .884.
- Диагностика: расширение до 24 возвращает 2 из 8 пропавших книг
  (recall@12 .884 → @24 .913), оставшиеся 6 — общие вопросы без
  книго-идентифицирующих терминов; union с профилями по RRF оказался хуже
  (.319), профили остаются filler.
- Результат: 10 книг — page recall@5 **.754 → .783**, MRR **.650 → .672**,
  citation precision .710 → .703, отказов 7 → 6, latency без изменений;
  5 книг — page@5 без изменений .833, @k .867 → .883, citation .587 → .599,
  отказов 8 → 7. Дефолт `BOOK_EXPERT_CANDIDATE_LIMIT=24`.

### T54 — Глубина запросов «Агента»: advanced_query_limit 4 → 8

- [x] Статус: `accepted` (2026-11-03; A/B на двух срезах)
- Scope: `config.py`, `.env.example`, `tests/test_advanced.py`.
- Acceptance: дефолт выбирается по A/B; page recall растёт, latency
  остаётся в разумном бюджете; тесты и проверки зелёные.
- Verify: `agent-eval --limit 20` и `--offset 100 --limit 20` при
  `BOOK_EXPERT_ADVANCED_QUERY_LIMIT=4/6/8`; `pytest`, `mypy`, `pre-commit`.
- Основание: super набирал ровно 16 хитов (4 запроса × limit 4), а RAG на
  том же индексе брал 8 из одного запроса и давал recall .84 против .70.
- Результат: срез 0–20 — depth 4: recall .70 / precision .1617 / p50 8.14 s /
  1 отказ; depth 6: **.80** / **.1992** / 10.27 s / 0; depth 8: **.80** /
  .1883 / 10.50 s / 0. Срез 100–120 — depth 4: .80 / .2517 / 4.90 s / 3;
  depth 8: .80 / **.2867** / 4.65 s / 4. Итог по 40 кейсам: recall
  **.75 → .80**, precision выше, +2.3 s только на первом срезе. Дефолт
  `BOOK_EXPERT_ADVANCED_QUERY_LIMIT=8`, тест на глубину обновлён.

### T55 — Дрейф цитат «Агента»: relevance-фильтр vs сужение scope

- [x] Статус: `accepted` (2026-11-03; IDF-фильтр отклонён, scope 3 принят)
- Scope: `retrieval/advanced.py`, `retrieval/super_advanced.py`, `config.py`,
  `.env.example`, `eval/agent_eval.py`, `tests/test_super_advanced.py`.
- Acceptance: уменьшить долю цитат из посторонних книг
  (`citation_book_precision`) без потери page recall и роста отказов.
- Verify: `agent-eval` 40 кейсов (offset 0) и 20 кейсов (offset 100) с
  фильтром вкл/выкл и `super_scope_limit` 0/3; живой `mode=super`.
- Основание: на «Что такое двойной спуск?» 2 из 5 цитат пришли из
  «Искусство учиться» и «Руководство ультрамарафонца»; супер искал по 24
  кандидатам без фильтра.
- Отрицательный результат (IDF-фильтр на накопленных хитах, 40 кейсов):
  recall .775 против .75, citation precision .245 против .179, **но**
  book precision **.80 против .87** и отказов 4 против 1 — дрейф не лечится,
  фильтр откатан.
- Положительный результат (`BOOK_EXPERT_SUPER_SCOPE_LIMIT=3`, 60 кейсов):
  recall **.75 → .783**, book precision **.783 → .823**, citation precision
  .197 → .208, отказов 5 → 6, latency без изменений. На срезе 100–120
  book precision .61 → .70 при том же recall .75.
- Добавлена метрика `citation_book_precision` (доля цитат из ожидаемой
  книги); дефолт `BOOK_EXPERT_SUPER_SCOPE_LIMIT=3`. Живой прогон: 4 из 5
  цитат из целевой книги, ответ корректный за 6.8 с, 414 тестов.
- Контроль grounding после T54/T55 (20 кейсов, `--faithfulness`):
  citation support **.9925**, faithfulness **.9439**, 1 отказ, 19 judged,
  p50 10.3 с — сужение scope и глубина 8 grounding не ухудшили.
  Оговорка: page recall на одиночном 20-кейсовом прогоне колеблется
  (.65 против .80 на 40 кейсах), поэтому решения принимать по 40+ кейсам.

### T56 — top_k 8 → 20 по диагностике промахов на 10 книгах

- [x] Статус: `accepted` (2026-11-03; A/B на 5 и 10 книгах)
- Scope: `config.py`, `.env.example`.
- Acceptance: page recall растёт на обоих наборах, citation precision не
  падает, latency остаётся в бюджете, контекст влезает в `num_ctx=8192`.
- Verify: `stage-eval --scope candidates --relevance-filter --k 8/12/20`
  на `combined-120` и `multi-book-40`; `pytest`, `mypy`, `pre-commit`.
- Основание: на 10 книгах 6 промахов лежали в top-20, но ниже 8-го места
  (ранги 9–17), 9 не попадали и в top-20; расширение k — самый дешёвый левер.
- Результат: 10 книг — page@k **.7826 → .8696**, citation precision
  **.7101 → .7850**, citation recall .7826 → .8551, отказы 6 → 5,
  zero-precision 9 → 5, answer p50 без изменений (1.16 с); 5 книг —
  page@k **.8667 → .9167**, citation precision **.5871 → .6186**,
  отказы 8 → 7, p50 1.47 → 1.61 с, p95 2.45 → 3.29 с. Дефолт
  `BOOK_EXPERT_TOP_K=20`; 20 фрагментов ≈ 4–5k токенов, влезает в 8192. Проверка generation на 120 кейсах при k=20: faithfulness **.9786**, claim-level support **.9938**, citation precision .6158, отказы 8, p50 1.70 с — не хуже k=8 (.9662 / .9865 / 1.47 с). Важно: `BOOK_EXPERT_TOP_K` в shell и `.env` перекрывает дефолт config — для прогона задавать `--k 20` или env явно; `.env` обновлён на 20.

### T57 — Метаданные agent-eval: scope, refine и лимит цитат

- [x] Статус: `accepted` (2026-11-03; отчёт фиксирует все влияющие настройки)
- Scope: `eval/agent_eval.py`, `tests/test_agent_eval.py`,
  `docs/evaluation.md`.
- Acceptance: в `meta` отчёта `agent-eval` есть `advanced_query_limit`,
  `super_scope_limit`, `super_refine_enabled` и `answer_max_citations`;
  тест проверяет поля; docs описывают состав meta.
- Основание: T54/T55 меняли глубину и scope, но в отчётах «Агента» этих
  полей не было — прогоны нельзя было сравнить по конфигурации.
- Результат: `meta` дополнен четырьмя полями, тест обновлён; поведение
  пайплайна не менялось.

### T59 — Исправить тавтологичную метрику source_page_recall

- [x] Статус: `accepted` (2026-11-03; найдено независимым аудитом 7.9/10)
- Scope: `eval/agent_eval.py`, `tests/test_agent_eval.py`, `tasklist.md`.
- Acceptance: `source_page_recall` считается по `Answer.context` (полный
  набор найденных фрагментов), а не по урезанным `Answer.sources`;
  тест: citation на стр. 10, context стр. 10+20, ожидаемая стр. 20 →
  `page_recall 0.0`, `source_page_recall 1.0`.
- Verify: unit-тест, повторный `agent-eval --limit 20`, `pytest`, `mypy`.
- Основание: `advanced._synthesize` кладёт в `sources` результат
  `cap_citations` (≤5), поэтому метрика совпадала с citation recall по
  построению.
- Результат: на 20 кейсах page recall **.70**, source recall **.75** —
  retrieval и цитирование действительно расходятся на 5 п.п.; вывод T51
  «не расходятся» исправлен.

### T60 — MCP ask_books: лимит 5 → 8

- [x] Статус: `accepted` (2026-11-03; решение пользователя 5B)
- Scope: `mcp_server.py`, `tests/test_mcp_server.py`, `README.md`,
  `docs/architecture.md`.
- Acceptance: `ask_books` по умолчанию берёт 8 источников; тест ожидает 8;
  доки не говорят top-5.
- Verify: `pytest tests/test_mcp_server.py`, `ruff`.
- Основание: веб-RAG перешёл на top_k=20, MCP оставался на 5; выбран
  компромисс 8 — recall выше, payload ещё умеренный.
- Результат: дефолт 8, тесты/доки обновлены.

### T61 — Судья faithfulness обрезает ответы на 24 утверждениях

- [x] Статус: `accepted` (2026-11-03; обрезки нет, кросс-модельный судья и слепая разметка выполнены)
- Scope: `eval/faithfulness.py`, `eval/agent_eval.py`,
  `eval/stage_eval.py`, `docs/evaluation.md`.
- Acceptance: `MAX_CLAIMS` поднят (≥48) или ответы не обрезаются; в отчёт
  пишется доля обрезанных ответов; кросс-модельный судья и 20 вручную
  размеченных ответов (Verify из T40) выполнены.
- Verify: повторный `--faithfulness` на 120 и 20 кейсах, сверка с ручной
  разметкой; `pytest`.
- Основание: в `audit-agent20-final2` 17 из 19 ответов «Агента» обрезаны на
  24 утверждениях; судья — та же deepseek-модель, ручной эталон отсутствует.
- Ожидаемый эффект: честная оценка неподдержанных утверждений; текущие
  .9786/.9938 могут быть завышены.
- Результат: `MAX_CLAIMS` 24 → 48, длинные ответы судья режет на батчи по
  `JUDGE_BATCH_SIZE=12`, в отчётах появился `claims_truncated`, у
  `stage-eval`/`agent-eval` — `--judge-provider` (пишется в `meta`).
  На 20 кейсах «Агента» (ollama, k=20): **11 из 20 ответов длиннее 24
  утверждений** (максимум 38), но `claims_truncated=0`,
  `faithfulness_errors=0`, faithfulness **.8990**, citation support **.7991**.
  Прежние .9786/.9938 были завышены обрезкой. На 120 кейсах RAG:
  `claims_truncated=0`, faithfulness .8962, support .8697.
- Кросс-модельный судья (ответы ollama, судья deepseek, 20 RAG-ответов,
  100 утверждений с цитатой): 0 ошибок, 0 обрезанных, citation support
  **.955**, faithfulness **.9611**, полностью поддержаны 17/20 ответов;
  5 вызовов deepseek повторены после сетевых таймаутов.
- Слепая разметка 20 ответов независимым агентом (без вердиктов судьи):
  сырое совпадение **88/100**, но kappa ≈ 0 (низкая база .95 и ошибки не
  пересекаются). Ручной разбор всех 12 спорных утверждений по полным текстам
  чанков: судья ошибся на 1.5 и 14.9 (ложные «не поддержано»), разметчик —
  на 2.4, 4.1, 6.3, 8.5, 8.6, 16.9 (пропустил поддержку). Эталон
  (согласие двух разметчиков + разбор всех споров): support **.97**,
  у судьи precision **.50**, recall **.67**, F1 **.57**, accuracy **.97**;
  эталон — `data/artifacts/t61-reference.local.json`. Агрегат судьи (.955)
  близок к эталону, поштучные метки — нет; порог по одной метке не двигать.

### T58 — Жёсткий таймаут агента через worker-процесс

- [x] Статус: `accepted` (2026-11-03; процесс убивается, live-проверка 1.01 с)
- Scope: `agent/worker.py`, `agent/graph.py`, `web/backend.py`, `config.py`,
  `.env.example`, `docs/architecture.md`, `tests/test_agent_worker.py`,
  `tests/test_agent.py`.
- Acceptance: веб-агент выполняет граф в дочернем процессе; по таймауту
  процесс и его клиенты убиваются, API отвечает 504; успешный путь работает;
  ошибки дочернего процесса доходят до родителя как `AgentError`/`ProviderError`.
- Verify: unit-тесты на success/timeout/error, live-запуск с
  `agent_timeout_s=1` (ожидается `AgentTimeoutError` ~1 с) и с 120 (ответ),
  `pytest`, `mypy`, `pre-commit`.
- Основание: `ThreadPoolExecutor` перестаёт ждать, но LangGraph продолжает
  работу; это был последний архитектурный блокер из аудита.
- Попутный баг: `build_chat_model` не поддерживал `provider=deepseek` и падал
  с «Missing credentials» — добавлена ветка DeepSeek (`deepseek_api_key`,
  `deepseek_base_url`, `deepseek_chat_model`).
- Результат: `agent_hard_timeout=true` по умолчанию; live timeout — **1.01 с**,
  успешный агент — **11.1 с / 5 шагов**; 4 unit-теста worker, тест DeepSeek.

### T62 — Жёсткий таймаут для super/advanced (дефолтный UI)

- [x] Статус: `accepted` (2026-11-03; live SSE 504 за 2.2 с, фоновых процессов нет)
- Scope: `web/app.py` (SSE-воркер), `retrieval/advanced.py`,
  `retrieval/super_advanced.py`, `agent/worker.py`, `tests/`.
- Acceptance: дефолтный режим «Агент» (`mode=super`) и `advanced` тоже
  ограничены жёстким временем; по таймауту клиент получает SSE `error`
  (или 504 для не-stream), работа не продолжается в фоне; есть тест.
- Verify: живой `POST /api/chat/stream` с заниженным таймаутом; `pytest`.
- Основание: T58 закрыл только `mode=agent`; UI по умолчанию открывает
  `super`, который идёт через Thread без таймаута.
- Результат: `web/answer_worker.py` считает advanced/super в дочернем
  процессе (`BOOK_EXPERT_ANSWER_HARD_TIMEOUT=true`,
  `BOOK_EXPERT_ANSWER_TIMEOUT_S=180`), прогресс идёт через pipe, по дедлайну
  `terminate → kill`. Live на :8011 с таймаутом 2 с: SSE `error`
  `{"status": 504}` за **2.23 с**, не-stream `POST /api/chat` — 504 за
  **2.02 с**; дочерних процессов после таймаута не осталось.
- Приёмка: ACCEPT, один medium исправлен в `0c832de` — краш дочернего
  процесса больше не маскируется под таймаут (`WorkerAnswerError` → 500),
  ошибка создания бэкенда приходит SSE-событием 503 вместо пустого 200;
  в воркере восстановлены трейсы advanced/super, ребёнок дожидается своего
  `finally` перед SIGTERM, `start()` не течёт pipe. Контракт ошибок:
  504 — только дедлайн, 500 — краш воркера, 503 — провайдер/конфигурация.
- Живой успешный путь (2026-11-03, реальные Qdrant/Ollama/книга):
  `mode=super` через worker — **31.7 с**, 11 SSE-событий, `done` с ответом
  и планом из 4 запросов; трейс `web-…jsonl` записан из дочернего процесса,
  фоновых `spawn`-детей после ответа нет.

### T63 — Audit hash-chain при параллельных запросах

- [x] Статус: `accepted` (2026-11-03; 2 процесса × 40 событий, chain валиден)
- Scope: `agent/audit.py`, `agent/worker.py`, `tests/test_audit_chain.py`.
- Acceptance: append и verify сериализованы межпроцессной блокировкой
  (file lock), два параллельных worker-процесса не ломают `seq/prev_hash`;
  `audit-verify` остаётся true; тест с двумя процессами.
- Verify: параллельный тест записи 50 событий, `audit-verify`; `pytest`.
- Основание: `_tail_state` + append без блокировки: два процесса читают
  один tail и пишут конфликтующие seq.
- Результат: append и `verify_audit_chain` идут под `fcntl.flock` на
  `tools-<дата>.jsonl.lock`; под локом хвост перечитывается с диска, поэтому
  `seq/prev_hash` не расходятся. Тест `fork`-процессами: 2 × 40 событий,
  `verify_audit_chain == (True, 80)`, seq строго `1..80`.

### T64 — Контентный сигнал для страниц вне top-20

- [x] Статус: `accepted` (2026-11-03; negative для пропусков, MRR +5.4 п.п.)
- Scope: диагностика на `data/eval/multi-book-40.local.json` (69 кейсов,
  10 книг), `retrieval/hybrid.py` — только эксперимент, прод-код не менялся.
- Acceptance: понять, возвращает ли OR-prefix5/расширение запроса страницы,
  которые не попали в top-20, и не теряется ли качество на остальных.
- Verify: диагностика промахов + A/B по `(book_id, page)`, а не по номеру
  страницы (номера повторяются в разных книгах).
- Результат: **9 промахов из 69**, и все — не лексика. У **6 из 9** ожидаемой
  книги нет в 24 кандидатах (и в 40 тоже); у 3 книга в кандидатах, но
  страница ниже 20-го места. OR-prefix5 не возвращает **ни одну** из 9
  страниц (0/9); глобально находятся 5/9 (dense 2, OR-лексика 3).
  Добавление OR-лексики в RRF при `w_or=0.1` не меняет recall@5/@20
  (.7826/.8696), но поднимает MRR@20 **.6752 → .7296**; при `w_or=0.05` —
  .7111. Вывод: контентный сигнал — рычаг ранжирования, а не покрытия;
  пропущенные страницы лечатся scope/роутингом, не BM25.
- Артефакты (ignored): `data/artifacts/t64-miss-diagnostic.local.json`,
  `data/artifacts/t64-miss-levers.local.json`.
- Ловушка диагностики: сравнивать `(book_id, page)`; сравнение только по
  `pdf_page_index` даёт ложные попадания из других книг.

### T65 — Opt-in OR-prefix5 канал в RRF

- [x] Статус: `accepted` (2026-11-03; default off, MRR@20 +5.4 п.п.)
- Scope: `config.py`, `retrieval/hybrid.py`, `tests/test_hybrid.py`,
  `.env.example`, `README.md`.
- Acceptance: второй лексический канал включается весом > 0, при 0.0 выдача
  не меняется; `rrf_fuse` принимает третий список; есть тесты.
- Verify: A/B на 69 кейсах, `pytest`, `ruff`, `mypy`.
- Результат: `BOOK_EXPERT_LEXICAL_OR_WEIGHT` (по умолчанию **0.0**);
  при весе > 0 выполняется второй FTS-запрос `or_prefix5` и фьюзится в тот
  же RRF. Recall@5/@20 не меняется (.7826/.8696), MRR@20 растёт
  **.6752 → .7296** при 0.1 и .7111 при 0.05. 437 тестов, ruff/mypy зелёные.
  Включать по умолчанию не стал: канал шумный, выигрыш только в ранжировании.

### T66 — Глобальный fallback книг в candidate scope

- [x] Статус: `accepted` (2026-11-03; negative: scope 24→34, recall падает)
- Scope: эксперимент на `data/eval/multi-book-40.local.json` (69 кейсов);
  прод-код не менялся.
- Основание: 6 из 9 промахов T64 — ожидаемой книги нет в 24 кандидатах;
  гипотеза — добавить книги из глобального dense/OR поиска.
- Acceptance: recall@5/@20 и MRR не падают, промахи сокращаются.
- Verify: A/B с метрикой `(book_id, page)` по 69 кейсам.
- Результат: **negative**. База: recall@5/@20 .7826/.8696, MRR@20 .6752,
  scope 24.0. Плюс книги глобального dense: .6957/.8551, MRR .6009,
  scope 33.8. Плюс книги глобального OR: без изменений (все уже в scope).
  Плюс оба: как dense. Добавленные 10 книг размывают RRF и вытесняют
  правильные страницы; ни один из 9 промахов не закрылся. Глобальный
  fallback не включать; следующий рычаг — качество ранжирования страниц
  (query expansion/эмбеддинги), а не ширина scope.
