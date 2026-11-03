# Book Expert

Локальный наблюдаемый book-only RAG-эксперт: загружает книги в локальный индекс,
отвечает на вопросы с цитатами и работает как автономный coding-агент. Книга не
покидает машину; облачный провайдер — только явный opt-in. Для `deepseek`
chat уходит в DeepSeek API, а embeddings остаются локальными (`bge-m3`).

## Статус

Работающий MVP: ingestion PDF/EPUB/FB2/DJVU → Qdrant + SQLite FTS5 → гибридный
поиск → grounded answer → LangGraph-агент → MCP stdio server. План и прогресс —
[tasklist.md](tasklist.md).

## Форматы

- **PDF** — основной формат: native text постранично, пустые страницы
  пропускаются; OCR не включён, зашифрованные PDF отклоняются.
- **EPUB** — через PyMuPDF (reflow-страницы); DRM не поддерживается, цитаты
  показываются по разделу/главе.
- **FB2** — нативный XML-парсер: секции становятся виртуальными страницами,
  поддержаны UTF-8 и Windows-1251; цитаты по разделу.
- **DJVU** — только при наличии текстового слоя (`djvutxt` из
  `djvulibre-bin`); сканы без текста отклоняются, OCR не выполняется.

Остальные форматы (MOBI, DOCX, HTML, TXT) не поддерживаются.

## Требования

- Python 3.12, [uv](https://docs.astral.sh/uv/)
- Docker + Compose (Qdrant)
- `bubblewrap` (`bwrap`) — песочница shell/git-инструментов агента
- `djvulibre-bin` — опционально, только для DJVU (нужен `djvutxt`)
- Ollama с моделями `qwen3.5:4b` (chat) и `bge-m3` (embeddings)
- Linux/WSL 2; GPU не обязателен, но рекомендуется

## Быстрый старт

```bash
uv sync
docker compose up -d
ollama serve            # в отдельном терминале
ollama pull qwen3.5:4b
ollama pull bge-m3

cp .env.example .env
uv run book-expert doctor

uv run book-expert ingest data/raw/book.pdf --title "My Book"
uv run book-expert ask "О чём глава 3?"
uv run book-expert search "ключевой термин" --book <book_id>
uv run book-expert chat
```

MCP-сервер для внешних агентов — **только RAG**:

```bash
uv run book-expert mcp
```

Инструменты: `list_books`, `get_outline`, `search_books`, `ask_books`.
`ask_books` по умолчанию берёт top-8 источников (параметр `limit` можно
задать явно) и отбрасывает источники, слабо покрывающие редкие термины
вопроса: остаются хиты с IDF-score не ниже
`BOOK_EXPERT_RELEVANCE_FILTER_MIN_RATIO` (по умолчанию 0.5) от лучшего, но
не меньше трёх. Если по отфильтрованным фрагментам модель отвечает
`INSUFFICIENT_CONTEXT`, ответ повторяется на полном top-8 — фильтр не должен
отнимать ответ. Так в citations не попадает хвостовой шум retrieval.
В ответе MCP `book_ids` содержит только книги, реально попавшие в citations
(при отказе — не более пяти), чтобы MCP-payload не раздувался. Это отличается
от CLI `ask --json`: там `book_ids` — весь выбранный scope (до 24 книг
префильтра), а не только процитированные.
Shell, файлы и git через MCP не отдаются. Для Qwen Code из корня проекта:

```bash
qwen mcp add -s project book-expert uv run book-expert mcp
```

Готовый project-scope конфиг лежит в `.qwen/settings.json`; после изменения
конфигурации перезапустите Qwen Code или выполните `qwen mcp reconnect`.
При первом подключении Qwen Code пометит сервер как «Pending approval» —
подтвердите его командой `qwen mcp approve book-expert` (или в интерфейсе).

## Веб-чат

```bash
uv run book-expert ui
```

Откройте `http://127.0.0.1:8000`. По умолчанию выбран **Агент**. В интерфейсе
два режима: **RAG** — ответ только по найденным фрагментам с цитатами на
книгу/страницу; **Агент** — планировщик строит 2–4 поисковых запроса,
выполняет их последовательно, до `BOOK_EXPERT_SUPER_MAX_ROUNDS` (по умолчанию
5) раз возвращается к плану, добавляет запросы по найденному и синтезирует
развёрнутый ответ. API по-прежнему принимает `mode=rag|advanced|super|agent`:
скрытые в интерфейсе «Продвинутый» и «Простой» остаются доступны запросом, а
сохранённые чаты с этими режимами открываются как «Агент».
Прогресс режима «Агент» идёт потоком (`POST /api/chat/stream`):
видны план, текущий запрос, номер раунда и процент. Книгу можно выбрать явно
или оставить авто-выбор по библиотеке: в поле над чатом начните вводить
название или автора — список книг фильтруется на месте (показывается до 50
совпадений, `Enter` выбирает первую, `Esc` закрывает). Для длинных ответов
агента Ollama запускается с контекстом `BOOK_EXPERT_OLLAMA_NUM_CTX` (по
умолчанию 8192); если ответ всё равно обрывается, увеличьте значение.

Слева — список **раздельных чатов**: у каждого своя история, режим и выбранная
книга, поэтому можно вести несколько независимых диалогов и возвращаться к ним.
Последние 12 реплик активного чата передаются на сервер (`history`), так что
агент и продвинутые режимы видят контекст беседы и понимают уточняющие вопросы.
У каждого блока ответа есть кнопка **«Подробнее»**: она отправляет уточняющий
запрос по тексту этого блока в текущем режиме, без ручного ввода вопроса.

Параметры продвинутого/супер-режимов:

```dotenv
# фрагментов на каждый поисковый запрос (advanced и super)
BOOK_EXPERT_ADVANCED_QUERY_LIMIT=8
# сколько раз super может вернуться к плану за доп. запросами
BOOK_EXPERT_SUPER_MAX_ROUNDS=5
# жёсткий таймаут SSE-ответа advanced/super: процесс убивается по дедлайну
BOOK_EXPERT_ANSWER_HARD_TIMEOUT=true
BOOK_EXPERT_ANSWER_TIMEOUT_S=180
```

Сервер слушает только loopback и не имеет авторизации. История диалога
хранится в `localStorage` браузера и в модель не передаётся; серверных
сессий нет. `--host`/`--port` меняют адрес, но выставлять UI в сеть без
дополнительной защиты не стоит. Ответы ассистента рендерятся как markdown,
а источники в конце ответа пронумерованы так же, как маркеры `[n]` в тексте,
и по умолчанию свёрнуты в блок «Источники (N)» — раскрываются кликом.
У каждого ответа есть 👍/👎: оценка уходит в `POST /api/feedback` и пишется
строкой в `data/artifacts/feedback.local.jsonl` (без текста ответа).
`GET /api/health` отдаёт uptime, число запросов/отказов/ошибок и p50/p95
latency по последним 200 запросам. В режимах «Продвинутый» и «Агент» ответ
оставляет не больше `BOOK_EXPERT_ANSWER_MAX_CITATIONS` источников
(по умолчанию 5): лишние маркеры перенумеровываются, а не висят в тексте.
«Агент» по умолчанию строит один план запросов без уточняющих раундов
(`BOOK_EXPERT_SUPER_REFINE_ENABLED=false`): на 40 кейсах раунды не дали
прироста recall, но добавляли ~3 запроса и 2–3 с.

### Библиотека

Вторая вкладка показывает книги из `data/raw/books` и записи каталога: статус
(«в RAG», «не в RAG», ошибка индексации), размер, страницы и число чанков.
Действия: загрузить книгу (опционально сразу в RAG), «В RAG»,
«Переиндексировать», «Убрать из RAG» (файл остаётся) и «Удалить файл»
(удаляет и файл, и индекс). Индексация выполняется фоновой задачей, интерфейс
опрашивает `/api/jobs/{id}`; одновременно идёт не более одного ingest.
Лимит загрузки — `BOOK_EXPERT_MAX_UPLOAD_MB` (по умолчанию 200 MB).

Есть пакетный импорт из папки: укажите путь в WSL-виде (`/mnt/d/Books`)
или Windows-виде (`D:\Books`), выберите «рекурсивно» и при необходимости
«сразу в RAG», затем «Проверить» — интерфейс покажет количество книг, объём и
сколько файлов уже есть. Импорт копирует книги плоско в `data/raw/books`
(совпадения имён получают суффикс), пропускает файлы с тем же именем и
размером, не следует по симлинкам и идёт одной фоновой задачей с прогрессом
и кнопкой «Отменить». Во время импорта показывается прогресс-бар с процентом,
текущим файлом, числом скопированных/проиндексированных книг и ошибок.
Лимиты — `BOOK_EXPERT_IMPORT_MAX_FILES`, `BOOK_EXPERT_IMPORT_MAX_FILE_MB`,
`BOOK_EXPERT_IMPORT_MAX_TOTAL_MB`. Если файлов больше лимита, ошибки нет:
за один запуск берётся первые N, а остальные остаются на следующий — уже
скопированные файлы пропускаются по имени и размеру.

## Команды

| Команда | Назначение |
| --- | --- |
| `doctor` | Проверка Python, Docker, Qdrant, Ollama, моделей и git-ignore |
| `ingest <book>` | Извлечение, chunking, embeddings, запись в Qdrant и FTS5 |
| `books` | Список книг, статусов и размеров индекса |
| `parser-audit` | Проверить text coverage документа до индексации; `--quality-gate` у `ingest` |
| `catalog-audit` | Дубли по названию, подозрительные имена и пропавшие файлы; ничего не удаляет |
| `catalog-dedupe` | План/применение удаления дублей из RAG; файлы остаются, `--apply` делает backup |
| `router-index` | Построить book-level профили для выбора книг |
| `reindex-lemmas` | Заполнить lemma-FTS5 для русского поиска |
| `audit-verify` | Проверка hash chain audit-лога tool-calls |
| `search <query>` | Гибридный поиск (dense + lexical + RRF); `--lexical-only` без Qdrant |
| `ask <question>` | Ответ с citations; авто-routing; `--lexical-only` без Qdrant |
| `chat` | Интерактивный агент: RAG + shell/fs/git в bwrap-песочнице |
| `ui` | Веб-чат в браузере: RAG-ответы с цитатами или агент |
| `mcp` | MCP stdio server (`search_books`, `ask_books`, `list_books`, `get_outline`) |
| `eval` | `--synthetic` или `<cases.json>`: recall@k, MRR, citations |
| `stage-eval` | Per-stage routing/retrieval/citations/latency + gate-симуляция |

`reindex-lemmas` — maintenance-команда: она перестраивает FTS и блокирует
параллельную запись в каталог; запускайте её при остановленном ingest.

Флаги `--json` дают машинный вывод; `--book <id>` ограничивает поиск книгой.

## Выбор книг

По умолчанию книги для поиска не выбирает LLM: берутся кандидаты префильтра
(профили, лексика, метаданные) — до 24 книг. На 120 кейсах по 5 книгам это
точнее и быстрее LLM-роутера: citation precision 0.5507 против 0.5158,
recall 0.775 против 0.725, отказов 16 против 19, маршрут 0.9 s против 2.3 s.
Вернуть прежнее поведение — `BOOK_EXPERT_ROUTER_USE_LLM=true`; порог
`BOOK_EXPERT_ROUTER_CONFIDENCE_THRESHOLD` действует только в этом режиме.

Ответы `ask` и веб-чата по умолчанию проходят relevance-фильтр (IDF редких
терминов): на 120 кейсах он поднял citation precision с 0.5507 до 0.5785 и
сократил отказы с 16 до 14. Отключить — `BOOK_EXPERT_ANSWER_RELEVANCE_FILTER=false`.

## Данные и приватность

Реальный PDF, extracted text, эмбеддинги, индексы, трейсы и audit нельзя
коммитить: `data/raw/`, `data/processed/`, `data/artifacts/` игнорируются.
Локальные manifests именуются `*.local.json`. Подробности —
[docs/adr/0002](docs/adr/0002-local-privacy-artifacts.md).

## Агент

Агент автономен внутри workspace, но с границами: bwrap-песочница (ro-root,
writable workspace, private `/tmp`, сеть выключена), path confinement, таймауты,
denylist опасных команд, audit JSONL, запрет remote-операций без
`BOOK_EXPERT_ALLOW_REMOTE=1`. Если `bwrap` не установлен, shell/git-инструменты
отключены. Политика —
[docs/adr/0003](docs/adr/0003-agent-autonomy.md).

## Веса RRF

Dense- и лексическая выдача смешиваются через взвешенный RRF:

```dotenv
BOOK_EXPERT_RRF_DENSE_WEIGHT=1.0
BOOK_EXPERT_RRF_LEXICAL_WEIGHT=0.1
BOOK_EXPERT_LEXICAL_OR_WEIGHT=0.0
```

Лексика по умолчанию весит меньше: на реальной книге равные веса размывали
dense-результаты. Снижение веса до 0.1 подняло page MRR@5 с 0.520 до 0.543
на 22 размеченных вопросах (эффект меньше одного кейса, 1/22 = 0.045) и не
ухудшило page recall@5. На расширенном наборе из 52 проверенных вопросов
page MRR@5 вырос с 0.670 до 0.680, recall не изменился.

`BOOK_EXPERT_LEXICAL_OR_WEIGHT` добавляет второй лексический канал
(`or_prefix5`) в тот же RRF. На 69 кейсах/10 книгах вес 0.1 не менял
page recall@5/@20 (.7826/.8696), но поднимал page MRR@20 с .675 до .730;
вес 0.05 даёт .711. По умолчанию включён **0.1**; 0.0 выключает канал.

Форма лексического запроса задаётся `BOOK_EXPERT_LEXICAL_QUERY_MODE`:
`and` (по умолчанию, все токены должны встретиться) или `or_prefix5`
(OR по 5-символьным префиксам). На 52 кейсах `or_prefix5` поднял общий
chunk recall@5 с 0.558 до 0.587, но на ручных 22 вопросах chunk MRR@5
просел с 0.464 до 0.422, поэтому режим оставлен opt-in. FTS5 trigram
не внедрён: против `unicode61 + or_prefix5` различимого прироста нет,
а индекс растёт примерно в 5 раз.

## Reranking

Гибридный поиск можно дополнительно переранжировать cross-encoder-моделью
через `fastembed` (ONNX, без torch). По умолчанию выключено:

```dotenv
BOOK_EXPERT_RERANK_ENABLED=true
BOOK_EXPERT_RERANK_MODEL=jinaai/jina-reranker-v2-base-multilingual
BOOK_EXPERT_RERANK_TOP_N=20
```

Модель (~1.1 ГБ) скачивается при первом поиске в `data/artifacts/models`
(git-ignored); `doctor` показывает, есть ли она в кэше. Если модель не
загрузилась или инференс упал, поиск возвращается к обычному RRF-порядку и
помечает это в `note`.

## Очистка колонтитулов

Опционально (`BOOK_EXPERT_STRIP_RUNNING_HEADERS=true`) из текста перед chunking
удаляются повторяющиеся колонтитулы с номерами страниц. Сноски и тело страницы
остаются; page map и `printed_page_label` не меняются. На реальной книге
выигрыша в recall/MRR не дало, поэтому по умолчанию выключено.

## Оценка и тесты

```bash
uv run pytest
uv run book-expert eval --synthetic
uv run book-expert eval path/to/cases.json --k 5
uv run book-expert stage-eval data/eval/combined-100.local.json
uv run python scripts/router_eval.py data/eval/router-eval.local.json
```

`router_eval` проверяет выбор книги без явного `book_ids`: top-1, recall@3,
`scope_coverage` (ожидаемая книга внутри выбранного scope — с учётом fallback
на кандидатов) и попадание target в prefilter-кандидаты. Датасет — local-only
`*.local.json`.

`stage-eval` считает стадии отдельно (routing, retrieval, citations, latency) и
симулирует retrieval gate. Ответ генерируется всегда, иначе не измерить
false refusals: gate `signal` — наблюдаемый сигнал «поиск не вернул хитов»,
gate `oracle` — в top-k нет ожидаемой страницы (верхняя граница экономии,
требует разметки). Метрики цитат считаются по страницам, потому что chunk id
размеченного набора привязан к конкретной сборке индекса и расходится после
переиндексации.

`--k` задаёт число хитов в ответе (по умолчанию `BOOK_EXPERT_TOP_K`, то есть
боевой путь), `--faithfulness` включает LLM-судью: он проверяет, поддержано ли
каждое утверждение цитируемым фрагментом (`citation_support_precision`) и хоть
каким-то найденным фрагментом (`faithfulness`). Page-overlap
`citation_precision` остаётся дешёвым sanity-check. Каждый отчёт содержит блок
`meta` с провайдером, моделями, `k`, scope, commit и sha256 датасета — по нему
сверяются прогоны, потому что экспортированные `BOOK_EXPERT_*` перекрывают
`.env`.

```bash
uv run book-expert stage-eval data/eval/combined-120.local.json \
  --scope candidates --relevance-filter --k 10 --faithfulness
```

Parser assessment (Docling/PyMuPDF) описан в
[docs/parsing-assessment.md](docs/parsing-assessment.md). Для локального
сравнения на выбранных страницах:

```bash
uv sync --extra docling
uv run python scripts/parser_compare.py data/raw/books/<book>.pdf --pages 50-54
```

Отчёт пишется в `data/artifacts/parser-compare.local.json` (git-ignored) и
содержит counts, runtime, page indices и token similarity; текста книги и
путей в нём нет. Ручной review по 9 критериям остаётся обязательным. Локальный прогон:

```bash
uv run python scripts/parser_assessment.py data/raw/book.pdf --pages 1-20
```

## Разработка

```bash
uv sync
uv run ruff format --check .
uv run ruff check .
uv run mypy src
uv run pytest --cov=book_expert --cov-report=term
uv run pre-commit run --all-files
```

CI повторяет те же шаги и падает при покрытии ниже 82% (baseline на
2026-11-02 — 85%).

Архитектура — [docs/architecture.md](docs/architecture.md), решения —
[docs/adr/](docs/adr/README.md).
