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

- [ ] Статус: `todo`
- Scope: вернуться к T23 только с smaller model/offline batch; замерить
  top-20 rerank latency на 100 кейсах; default off без выигрыша.
- Acceptance: page recall@5/citation precision растут, p95 latency в бюджете;
  иначе задача закрывается отрицательным результатом.
- Verify: latency benchmark, 100-case eval, full checks.

### T32 — Structure-aware chunking по разделам

- [ ] Статус: `review` (2026-11-02; 5 книг/120 кейсов: page@5 .9083 vs .925, @10 .9333 vs .95, MRR .7538 vs .781; negative)
- Scope: 5 книг (ML Systems, Effective Leader, Emotional Intelligence, JavaScript, Crypto); section boundaries + merge мелких до 300–500 и split крупных >1200–1500; A/B current fixed vs pure section vs hybrid on 100 existing + 20 новых вопросов для двух книг; store section_path/page range.
- Acceptance: chunk recall@5 и citation precision растут или не падают; доля chunks <100 резко снижена; секции >4000 не превращаются в один chunk; citations проверяемы.
- Verify: section-size report, A/B 120 cases, targeted tests, full checks.

### T33 — Citation gate: ответ без цитат не отдаётся как валидный

- [ ] Статус: `todo`
- Scope: `retrieval/answer.py`, `eval/stage_eval.py`; если модель вернула
  текст без маркеров `[n]` и `refused=false`, повторять запрос с явным
  требованием цитат, а при повторе — отдавать `INSUFFICIENT_CONTEXT`;
  в отчёте stage-eval разделять «нет цитат при промахе retrieval» и «нет
  цитат при верной странице в top-5».
- Acceptance: `empty_citations` в stage-eval падает без роста refusals на
  кейсах, где retrieval нашёл ожидаемую страницу; ответ без citations больше
  не считается успешным; targeted tests.
- Verify: 100-case stage-eval до/после, targeted tests, full checks.
- Основание: T28 — ollama 13/100 ответов без citations, из них 5 при
  page recall@5 = 1.0; deepseek 7, из них 1 при верной странице.

### T34 — Калибровка routing fallback для deepseek

- [ ] Статус: `todo`
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
