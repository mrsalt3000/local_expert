# Book Expert

Локальный наблюдаемый book-only RAG-эксперт: загружает книги в локальный индекс,
отвечает на вопросы с цитатами и работает как автономный coding-агент. Книга не
покидает машину; облачный провайдер — только явный opt-in.

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
`ask_books` по умолчанию берёт top-5 источников (параметр `limit` можно
задать явно) и отбрасывает источники, слабо покрывающие редкие термины
вопроса: остаются хиты с IDF-score не ниже
`BOOK_EXPERT_RELEVANCE_FILTER_MIN_RATIO` (по умолчанию 0.5) от лучшего, но
не меньше трёх. Если по отфильтрованным фрагментам модель отвечает
`INSUFFICIENT_CONTEXT`, ответ повторяется на полном top-5 — фильтр не должен
отнимать ответ. Так в citations не попадает хвостовой шум retrieval.
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

Откройте `http://127.0.0.1:8000`. По умолчанию выбран **Супер**. В интерфейсе
четыре режима: **RAG** — ответ
только по найденным фрагментам с цитатами на книгу/страницу; **Продвинутый** —
планировщик строит 2–4 поисковых запроса, выполняет их последовательно в RAG
и синтезирует развёрнутый ответ; **Супер** — после продвинутого может до
`BOOK_EXPERT_SUPER_MAX_ROUNDS` (по умолчанию 5) раз вернуться к плану, добавить
запросы по найденному и затем синтезировать ответ; **Простой** — react-цикл
с `rag_search`/`rag_ask` и coding-инструментами (если доступен `bwrap`).
Прогресс продвинутого и супер-режимов идёт потоком (`POST /api/chat/stream`):
видны план, текущий запрос, номер раунда и процент. Книгу можно выбрать явно
или оставить авто-выбор по библиотеке. Для длинных ответов супер-режима
Ollama запускается с контекстом `BOOK_EXPERT_OLLAMA_NUM_CTX` (по умолчанию
8192); если ответ всё равно обрывается, увеличьте значение.

Слева — список **раздельных чатов**: у каждого своя история, режим и выбранная
книга, поэтому можно вести несколько независимых диалогов и возвращаться к ним.
Последние 12 реплик активного чата передаются на сервер (`history`), так что
агент и продвинутые режимы видят контекст беседы и понимают уточняющие вопросы.
У каждого блока ответа есть кнопка **«Подробнее»**: она отправляет уточняющий
запрос по тексту этого блока в текущем режиме, без ручного ввода вопроса.

Параметры продвинутого/супер-режимов:

```dotenv
# фрагментов на каждый поисковый запрос (advanced и super)
BOOK_EXPERT_ADVANCED_QUERY_LIMIT=4
# сколько раз super может вернуться к плану за доп. запросами
BOOK_EXPERT_SUPER_MAX_ROUNDS=5
```

Сервер слушает только loopback и не имеет авторизации. История диалога
хранится в `localStorage` браузера и в модель не передаётся; серверных
сессий нет. `--host`/`--port` меняют адрес, но выставлять UI в сеть без
дополнительной защиты не стоит. Ответы ассистента рендерятся как markdown,
а источники в конце ответа пронумерованы так же, как маркеры `[n]` в тексте,
и по умолчанию свёрнуты в блок «Источники (N)» — раскрываются кликом.

### Библиотека

Вторая вкладка показывает книги из `data/raw/books` и записи каталога: статус
(«в RAG», «не в RAG», ошибка индексации), размер, страницы и число чанков.
Действия: загрузить книгу (опционально сразу в RAG), «В RAG»,
«Переиндексировать», «Убрать из RAG» (файл остаётся) и «Удалить файл»
(удаляет и файл, и индекс). Индексация выполняется фоновой задачей, интерфейс
опрашивает `/api/jobs/{id}`; одновременно идёт не более одного ingest.
Лимит загрузки — `BOOK_EXPERT_MAX_UPLOAD_MB` (по умолчанию 200 MB).

Есть пакетный импорт из папки: укажите путь в WSL-виде (`/mnt/h/DISK/Книги`)
или Windows-виде (`H:\DISK\Книги`), выберите «рекурсивно» и при необходимости
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
| `catalog-audit` | Дубли по названию, подозрительные имена и пропавшие файлы; ничего не удаляет |
| `search <query>` | Гибридный поиск (dense + lexical + RRF); `--lexical-only` без Qdrant |
| `ask <question>` | Ответ с citations; авто-routing; `--lexical-only` без Qdrant |
| `chat` | Интерактивный агент: RAG + shell/fs/git в bwrap-песочнице |
| `ui` | Веб-чат в браузере: RAG-ответы с цитатами или агент |
| `mcp` | MCP stdio server (`search_books`, `ask_books`, `list_books`, `get_outline`) |
| `eval` | `--synthetic` или `<cases.json>`: recall@k, MRR, citations |

Флаги `--json` дают машинный вывод; `--book <id>` ограничивает поиск книгой.

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
```

Лексика по умолчанию весит меньше: на реальной книге равные веса размывали
dense-результаты. Снижение веса до 0.1 подняло page MRR@5 с 0.520 до 0.543
на 22 размеченных вопросах (эффект меньше одного кейса, 1/22 = 0.045) и не
ухудшило page recall@5. На расширенном наборе из 52 проверенных вопросов
page MRR@5 вырос с 0.670 до 0.680, recall не изменился.

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
```

Parser assessment (Docling/PyMuPDF) описан в
[docs/parsing-assessment.md](docs/parsing-assessment.md). Локальный прогон:

```bash
uv run python scripts/parser_assessment.py data/raw/book.pdf --pages 1-20
```

## Разработка

```bash
uv sync
uv run ruff format --check .
uv run ruff check .
uv run pytest
uv run pre-commit run --all-files
```

Архитектура — [docs/architecture.md](docs/architecture.md), решения —
[docs/adr/](docs/adr/README.md).
