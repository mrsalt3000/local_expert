# Book Expert

Локальный наблюдаемый book-only RAG-эксперт: загружает книги в локальный индекс,
отвечает на вопросы с цитатами и работает как автономный coding-агент. Книга не
покидает машину; облачный провайдер — только явный opt-in.

## Статус

Работающий MVP: ingestion PDF → Qdrant + SQLite FTS5 → гибридный поиск →
grounded answer → LangGraph-агент → MCP stdio server. План и прогресс —
[tasklist.md](tasklist.md).

## Требования

- Python 3.12, [uv](https://docs.astral.sh/uv/)
- Docker + Compose (Qdrant)
- `bubblewrap` (`bwrap`) — песочница shell/git-инструментов агента
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

MCP-сервер для внешних агентов:

```bash
uv run book-expert mcp
```

## Веб-чат

```bash
uv run book-expert ui
```

Откройте `http://127.0.0.1:8000`. В интерфейсе два режима: **RAG** — ответ
только по найденным фрагментам с цитатами на книгу/страницу; **Агент** —
react-цикл с `rag_search`/`rag_ask` и coding-инструментами (если доступен
`bwrap`). Книгу можно выбрать явно или оставить авто-выбор по библиотеке.

Сервер слушает только loopback и не имеет авторизации. История диалога
хранится в `localStorage` браузера и в модель не передаётся; серверных
сессий нет. `--host`/`--port` меняют адрес, но выставлять UI в сеть без
дополнительной защиты не стоит.

### Библиотека

Вторая вкладка показывает PDF из `data/raw/books` и записи каталога: статус
(«в RAG», «не в RAG», ошибка индексации), размер, страницы и число чанков.
Действия: загрузить PDF (опционально сразу в RAG), «В RAG»,
«Переиндексировать», «Убрать из RAG» (файл остаётся) и «Удалить файл»
(удаляет и файл, и индекс). Индексация выполняется фоновой задачей, интерфейс
опрашивает `/api/jobs/{id}`; одновременно идёт не более одного ingest.
Лимит загрузки — `BOOK_EXPERT_MAX_UPLOAD_MB` (по умолчанию 200 MB).

## Команды

| Команда | Назначение |
| --- | --- |
| `doctor` | Проверка Python, Docker, Qdrant, Ollama, моделей и git-ignore |
| `ingest <pdf>` | Извлечение, chunking, embeddings, запись в Qdrant и FTS5 |
| `books` | Список книг, статусов и размеров индекса |
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
