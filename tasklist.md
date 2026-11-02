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

- [ ] Статус: `review` (2026-11-02; 500: page@10 +1.7pp, но refusals 13 vs 7, citation recall −1.7pp; default 1000 оставлен)
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

- [ ] Статус: `review` (2026-11-02; 100 кейсов × 2 провайдера: retrieval page@5 .91, page MRR@5 .786, chunk@5 .69; deepseek citation .604/.88, refusals 5, empty 7; ollama .576/.86, refusals 1, empty 13; oracle-gate 9/100, 0 false refusals; signal-gate 0. Routing: non-fallback top1 .606 deepseek / .309 ollama, fallback 67 / 19)
- Scope: отдельные отчёты routing / retrieval / citation / latency; gate:
  не генерировать ответ при низком retrieval recall; CLI/JSON.
- Acceptance: 100-case отчёт с per-stage метриками; gate не увеличивает
  false refusals без причины.
- Verify: eval artifacts, unit tests, full checks.

### T29 — Полный contextual reindex production

- [ ] Статус: `review` (2026-11-02; reindex 1151/548999 ok; eval: chunk +1.9pp, page −1.0pp, empty citations 9; production switch не одобрен)
- Scope: переиндексировать все 1150 книг с contextual embeddings в новую
  коллекцию, проверить размер/время, переключить `BOOK_EXPERT_QDRANT_COLLECTION`
  только после A/B, сохранить rollback.
- Acceptance: 100-case eval не хуже plain; Qdrant points = chunks; rollback
  документирован.
- Verify: reindex job, count/integrity, 100-case eval, full checks.

### T30 — Parser quality gate

- [x] Статус: `accepted` (2026-11-02; parser-audit + opt-in ingest --quality-gate, 372 tests)
- Scope: метрики text coverage/chars per page при ingest; флаг/карантин для
  низкого покрытия; OCR-очередь для scans; отчёт по книге.
- Acceptance: книги с потерей текста не попадают молча в RAG; отчёт содержит
  только counts/страницы, без текста.
- Verify: synthetic low-coverage PDF, targeted tests, full checks.

### T31 — Reranker только при доказанном latency-бюджете

- [ ] Статус: `review` (2026-11-02; negative: малые cross-encoder'ы дают p50 1.1–1.5 s и снижают page@5 с .80 до .65–.70; default остаётся off)
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

- [ ] Статус: `review` (2026-11-02; 5 книг/120 кейсов: page@5 .9083 vs .925, @10 .9333 vs .95, MRR .7538 vs .781; negative)
- Scope: 5 книг (ML Systems, Effective Leader, Emotional Intelligence, JavaScript, Crypto); section boundaries + merge мелких до 300–500 и split крупных >1200–1500; A/B current fixed vs pure section vs hybrid on 100 existing + 20 новых вопросов для двух книг; store section_path/page range.
- Acceptance: chunk recall@5 и citation precision растут или не падают; доля chunks <100 резко снижена; секции >4000 не превращаются в один chunk; citations проверяемы.
- Verify: section-size report, A/B 120 cases, targeted tests, full checks.

### T33 — Citation gate: ответ без сопоставленных цитат не отдаётся как валидный

- [ ] Статус: `review` (2026-11-02; gate добавлен, на 100 кейсах не сработал ни разу: refusals 5→5, precision .6042/recall .88 без изменений; путь retry→refusal покрыт тестами)
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

- [ ] Статус: `review` (2026-11-02; negative: порог и retry — не рычаг, fallback на 12 кандидатов покрывает книгу лучше уверенного одиночного выбора; в router_eval добавлен `scope_coverage`)
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
