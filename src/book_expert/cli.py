"""CLI Book Expert: ingest, books, search, ask, chat, mcp."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import typer
from rich.console import Console

from book_expert.agent.audit import AuditLogger, verify_audit_chain
from book_expert.agent.graph import AgentRunner
from book_expert.agent.tools.coding import build_coding_tools
from book_expert.agent.tools.rag import build_rag_tools
from book_expert.agent.trace import TraceRecorder
from book_expert.catalog import Catalog
from book_expert.catalog_audit import audit_catalog
from book_expert.catalog_dedupe import build_dedupe_plan
from book_expert.config import Settings
from book_expert.eval.metrics import load_cases, run_answer_eval, run_retrieval_eval
from book_expert.eval.stage_eval import ScopeMode, run_stage_eval, write_report
from book_expert.eval.synthetic import seed_synthetic_corpus
from book_expert.ingestion.document import DocumentError
from book_expert.ingestion.extract import extract_document
from book_expert.ingestion.indexer import BookIndexer, IndexingError
from book_expert.ingestion.pdf import PdfError
from book_expert.ingestion.quality import QualityThresholds, assess_document
from book_expert.models import Answer, Book, SearchHit
from book_expert.providers import Provider, ProviderError, get_provider
from book_expert.retrieval.advanced import AdvancedAnswerer
from book_expert.retrieval.answer import GroundedAnswerer
from book_expert.retrieval.book_profiles import build_profiles
from book_expert.retrieval.super_advanced import SuperAdvancedAnswerer
from book_expert.retrieval.hybrid import HybridSearcher, SearchOutcome
from book_expert.retrieval.router import BookRouter

app = typer.Typer(
    no_args_is_help=True,
    help="Локальный book-only RAG-эксперт.",
)
console = Console()


@dataclass
class RetrievalServices:
    catalog: Catalog
    provider: Provider
    searcher: HybridSearcher
    router: BookRouter
    answerer: GroundedAnswerer
    advanced: AdvancedAnswerer
    super_advanced: SuperAdvancedAnswerer
    trace: TraceRecorder

    def close(self) -> None:
        self.searcher.close()
        close = getattr(self.provider, "close", None)
        if callable(close):
            close()


def _make_services(settings: Settings) -> tuple[Catalog, Provider, BookIndexer]:
    catalog = Catalog(settings.catalog_path)
    catalog.initialize()
    provider = get_provider(settings)
    return catalog, provider, BookIndexer(settings, provider, catalog)


def _make_retrieval(settings: Settings) -> RetrievalServices:
    catalog = Catalog(settings.catalog_path)
    catalog.initialize()
    provider = get_provider(settings)
    searcher = HybridSearcher(settings, provider, catalog)
    trace = TraceRecorder.create(settings, prefix="retrieval")
    router = BookRouter(settings, provider, catalog, trace=trace)
    answerer = GroundedAnswerer(settings, provider, searcher, router, trace=trace)
    advanced = AdvancedAnswerer(settings, provider, searcher, router, trace=trace)
    super_advanced = SuperAdvancedAnswerer(
        settings, provider, searcher, router, trace=trace
    )
    return RetrievalServices(
        catalog, provider, searcher, router, answerer, advanced, super_advanced, trace
    )


def _load_catalog(settings: Settings) -> Catalog:
    catalog = Catalog(settings.catalog_path)
    catalog.initialize()
    return catalog


@app.command()
def ingest(
    path: Path = typer.Argument(..., exists=True, dir_okay=False, readable=True),
    title: str | None = typer.Option(None, "--title", help="Название книги."),
    force: bool = typer.Option(False, "--force", help="Переиндексировать книгу."),
    quality_gate: bool = typer.Option(
        False,
        "--quality-gate",
        help="Не индексировать книгу с потерей текста.",
    ),
    json_output: bool = typer.Option(False, "--json", help="Машинный вывод."),
) -> None:
    """Загрузить книгу (PDF/EPUB/FB2/DJVU) в локальный индекс."""
    settings = Settings()
    settings.ensure_dirs()
    provider: Provider | None = None
    indexer: BookIndexer | None = None
    payload: dict[str, Any] | None = None
    try:
        if quality_gate:
            document = extract_document(
                path,
                strip_running_lines=settings.strip_running_headers,
            )
            quality = assess_document(document)
            if quality.status == "reject":
                _fail(
                    "quality gate: " + ", ".join(quality.reasons),
                    json_output,
                )
            if quality.status == "warn" and not json_output:
                console.print(
                    f"[yellow]quality warning[/yellow] {len(quality.issues)} pages"
                )
        _, provider, indexer = _make_services(settings)
        book = indexer.index_pdf(path, title=title, force=force)
        payload = _book_payload(book, indexer)
    except (
        DocumentError,
        PdfError,
        ProviderError,
        IndexingError,
        ValueError,
        OSError,
    ) as exc:
        _fail(str(exc), json_output)
        return
    finally:
        if indexer is not None:
            indexer.close()
        if provider is not None and hasattr(provider, "close"):
            provider.close()

    if payload is None:
        return
    if json_output:
        typer.echo(json.dumps(payload, ensure_ascii=False))
    else:
        console.print(
            f"[green]indexed[/green] {payload['title']} "
            f"({payload['pages']} pages, {payload['chunks']} chunks)"
        )


@app.command("parser-audit")
def parser_audit(
    path: Path = typer.Argument(..., exists=True, dir_okay=False, readable=True),
    min_coverage: float = typer.Option(0.9, "--min-coverage"),
    min_chars: int = typer.Option(50, "--min-chars"),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    """Проверить качество извлечённого текста без индексации."""
    if not 0.0 <= min_coverage <= 1.0:
        _fail("min-coverage must be in [0, 1]", json_output)
    settings = Settings()
    try:
        document = extract_document(
            path,
            strip_running_lines=settings.strip_running_headers,
        )
        report = assess_document(
            document,
            QualityThresholds(
                min_coverage=min_coverage,
                min_chars_per_page=min_chars,
            ),
        )
    except (DocumentError, PdfError, ValueError, OSError) as exc:
        _fail(str(exc), json_output)
        return
    payload = report.to_dict()
    if json_output:
        typer.echo(json.dumps(payload, ensure_ascii=False))
    else:
        console.print(
            f"{report.status} pages={report.total_pages} "
            f"nonempty={report.nonempty_pages} chars={report.total_chars} "
            f"coverage={report.coverage:.3f}"
        )
        for issue in report.issues[:20]:
            console.print(
                f"  page {issue.pdf_page_index}: {issue.reason} ({issue.chars} chars)"
            )
    if report.status == "reject":
        raise typer.Exit(code=1)


@app.command()
def books(json_output: bool = typer.Option(False, "--json")) -> None:
    """Показать книги в каталоге."""
    settings = Settings()
    settings.ensure_dirs()
    items = _load_catalog(settings).list_books()
    if json_output:
        typer.echo(
            json.dumps([_book_payload(book) for book in items], ensure_ascii=False)
        )
        return
    if not items:
        console.print("каталог пуст")
        return
    for book in items:
        console.print(
            f"{book.id[:8]}  {book.status.value:<8}  {book.title}  "
            f"({book.page_count} pages, {book.chunk_count} chunks)"
        )


@app.command("catalog-audit")
def catalog_audit_report(
    limit: int = typer.Option(20, "--limit", min=1, max=200),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    """Показать дубли, подозрительные названия и пропавшие файлы."""
    settings = Settings()
    settings.ensure_dirs()
    report = audit_catalog(_load_catalog(settings).list_books())
    if json_output:
        typer.echo(json.dumps(report.to_dict(), ensure_ascii=False))
        return
    console.print(
        f"книг: {report.total}, ready: {report.ready}, "
        f"pending: {report.pending}, indexing: {report.indexing}, "
        f"failed: {report.failed}"
    )
    console.print(
        f"дублей: {len(report.duplicates)}, "
        f"подозрительных названий: {len(report.suspicious)}, "
        f"пропавших файлов: {len(report.missing_files)}"
    )
    for group in report.duplicates[:limit]:
        console.print(f"[yellow]дубль[/yellow] {group.key}")
        for book in group.books:
            marker = "*" if book.id == group.keep_id else " "
            console.print(
                f" {marker} {book.id[:8]}  {book.title}  "
                f"({book.page_count} pages, {book.chunk_count} chunks)"
            )
    for book, reason in report.suspicious[:limit]:
        console.print(
            f"[yellow]название[/yellow] {book.id[:8]}  {book.title}  ({reason})"
        )
    for book in report.missing_files[:limit]:
        console.print(f"[red]нет файла[/red] {book.id[:8]}  {book.title}")
    if (
        len(report.duplicates) > limit
        or len(report.suspicious) > limit
        or len(report.missing_files) > limit
    ):
        console.print(f"[dim]показаны первые {limit}; полный отчёт — --json[/dim]")


@app.command("reindex-lemmas")
def reindex_lemmas(
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    """Заполнить lemma-индекс FTS5 для всех chunks."""
    settings = Settings()
    settings.ensure_dirs()
    catalog = _load_catalog(settings)
    try:
        count = catalog.reindex_lemmas()
    except (OSError, ValueError) as exc:
        _fail(str(exc), json_output)
    if json_output:
        typer.echo(json.dumps({"chunks": count}, ensure_ascii=False))
    else:
        console.print(f"лемматизировано chunks: {count}")


@app.command("router-index")
def router_index(
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    """Построить book-level профили для routing."""
    settings = Settings()
    settings.ensure_dirs()
    catalog = _load_catalog(settings)
    provider = get_provider(settings)
    try:
        if provider.name == "ollama":
            console.print(
                "[yellow]профили используются только облачным router; "
                "для local Ollama они не загружаются[/yellow]"
            )
        count = build_profiles(settings, catalog, provider)
    except (ProviderError, ValueError, OSError) as exc:
        _fail(str(exc), json_output)
    finally:
        close = getattr(provider, "close", None)
        if callable(close):
            close()
    if json_output:
        typer.echo(
            json.dumps(
                {"books": count, "path": str(settings.book_profiles_path)},
                ensure_ascii=False,
            )
        )
    else:
        console.print(f"профилей: {count}; {settings.book_profiles_path}")


@app.command("catalog-dedupe")
def catalog_dedupe(
    apply: bool = typer.Option(
        False,
        "--apply",
        help="Убрать проигравшие копии из RAG; файлы остаются на диске.",
    ),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    """Показать или применить план дедупликации каталога."""
    settings = Settings()
    settings.ensure_dirs()
    catalog = _load_catalog(settings)
    plan = build_dedupe_plan(catalog.list_books())
    if not apply:
        if json_output:
            typer.echo(json.dumps(plan.to_dict(), ensure_ascii=False))
            return
        console.print(
            f"групп дублей: {len(plan.groups)}, к удалению из RAG: {len(plan.removals)}"
        )
        for removal in plan.removals:
            console.print(
                f"[yellow]убрать[/yellow] {removal.book.id[:8]}  "
                f"{removal.book.title}  "
                f"(keep {removal.keep_id[:8]})"
            )
        console.print("[dim]это dry-run; для применения добавьте --apply[/dim]")
        return

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    backup = catalog.path.with_name(f"{catalog.path.name}.bak-{stamp}")
    shutil.copy2(catalog.path, backup)
    provider: Provider | None = None
    indexer: BookIndexer | None = None
    removed: list[dict[str, str]] = []
    errors: list[dict[str, str]] = []
    try:
        provider = get_provider(settings)
        indexer = BookIndexer(settings, provider, catalog)
        for removal in plan.removals:
            try:
                deleted = indexer.delete_book(removal.book.id)
            except Exception as exc:
                errors.append({"id": removal.book.id, "error": str(exc)[:300]})
            else:
                if not deleted:
                    errors.append({"id": removal.book.id, "error": "book not found"})
                    continue
                removed.append({"id": removal.book.id, "title": removal.book.title})
    finally:
        if indexer is not None:
            indexer.close()
        close = getattr(provider, "close", None)
        if callable(close):
            close()

    report = settings.artifacts_dir / f"catalog-dedupe-{stamp}.json"
    report.write_text(
        json.dumps(
            {
                "backup": str(backup),
                "removed": removed,
                "errors": errors,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    if json_output:
        typer.echo(
            json.dumps(
                {
                    "backup": str(backup),
                    "report": str(report),
                    "removed": len(removed),
                    "errors": errors,
                },
                ensure_ascii=False,
            )
        )
    else:
        console.print(
            f"убрано из RAG: {len(removed)}, ошибок: {len(errors)}; "
            f"backup: {backup}; отчёт: {report}"
        )
    if errors:
        raise typer.Exit(code=1)


@app.command("audit-verify")
def audit_verify(
    path: Path | None = typer.Argument(
        None, exists=True, dir_okay=False, readable=True
    ),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    """Проверить hash chain audit-лога."""
    settings = Settings()
    settings.ensure_dirs()
    target = path
    if target is None:
        files = sorted(settings.audit_dir.glob("tools-*.jsonl"))
        if not files:
            _fail("audit log not found", json_output)
        target = files[-1]
    ok, records = verify_audit_chain(target)
    if json_output:
        typer.echo(
            json.dumps(
                {"ok": ok, "records": records, "path": str(target)},
                ensure_ascii=False,
            )
        )
    else:
        status = "[green]ok[/green]" if ok else "[red]broken[/red]"
        console.print(f"{status} {target} ({records} records)")
    if not ok:
        raise typer.Exit(code=1)


@app.command()
def search(
    query: str = typer.Argument(...),
    book: list[str] | None = typer.Option(None, "--book", help="Ограничить книгами."),
    limit: int = typer.Option(8, "--limit", min=1, max=20),
    lexical_only: bool = typer.Option(
        False, "--lexical-only", help="Только SQLite FTS5, без Qdrant."
    ),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    """Гибридный поиск по библиотеке."""
    settings = Settings()
    settings.ensure_dirs()
    services: RetrievalServices | None = None
    payload: dict[str, Any] | None = None
    try:
        services = _make_retrieval(settings)
        outcome = services.searcher.search(
            query,
            book_ids=book or None,
            limit=limit,
            dense=not lexical_only,
        )
        payload = _outcome_payload(query, outcome)
    except (ProviderError, ValueError, OSError) as exc:
        _fail(str(exc), json_output)
        return
    finally:
        if services is not None:
            services.close()

    if payload is None:
        return
    if json_output:
        typer.echo(json.dumps(payload, ensure_ascii=False))
        return
    _print_hits(payload)


@app.command()
def ask(
    question: str = typer.Argument(...),
    book: list[str] | None = typer.Option(None, "--book", help="Ограничить книгами."),
    no_route: bool = typer.Option(False, "--no-route", help="Не выбирать книги."),
    lexical_only: bool = typer.Option(
        False, "--lexical-only", help="Только SQLite FTS5, без Qdrant."
    ),
    advanced: bool = typer.Option(
        False, "--advanced", help="План запросов + несколько RAG-поисков."
    ),
    super_mode: bool = typer.Option(
        False, "--super", help="Advanced + до N возвратов к плану."
    ),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    """Ответ по книгам с цитатами."""
    settings = Settings()
    settings.ensure_dirs()
    services: RetrievalServices | None = None
    answer: Answer | None = None
    try:
        services = _make_retrieval(settings)
        if super_mode:
            answer = services.super_advanced.ask(
                question,
                book_ids=book or None,
                auto_route=not no_route,
                dense=not lexical_only,
            )
        elif advanced:
            answer = services.advanced.ask(
                question,
                book_ids=book or None,
                auto_route=not no_route,
                dense=not lexical_only,
            )
        else:
            answer = services.answerer.ask(
                question,
                book_ids=book or None,
                auto_route=not no_route,
                dense=not lexical_only,
                relevance_filter=settings.answer_relevance_filter,
            )
    except (ProviderError, ValueError, OSError) as exc:
        _fail(str(exc), json_output)
        return
    finally:
        if services is not None:
            services.close()

    if answer is None:
        return
    if json_output:
        typer.echo(json.dumps(_answer_payload(answer), ensure_ascii=False))
        return
    if answer.max_rounds:
        console.print(f"[dim]Раунды: {answer.rounds}/{answer.max_rounds}[/dim]")
    if answer.plan:
        console.print("[dim]План поиска:[/dim]")
        for index, query in enumerate(answer.plan, start=1):
            console.print(f"[dim]{index}. {query}[/dim]")
    console.print(answer.text)
    for citation in answer.citations:
        page = citation.printed_page_label or str(citation.pdf_page_index)
        if citation.format in {"pdf", "djvu"}:
            locator = f"стр. {page}"
        elif citation.section_path:
            locator = citation.section_path
        else:
            locator = f"фрагмент {page}"
        console.print(
            f"[dim]— {citation.book_title}, {locator} ({citation.chunk_id})[/dim]"
        )


@app.command()
def chat(
    book: list[str] | None = typer.Option(None, "--book", help="Ограничить книгами."),
) -> None:
    """Интерактивный агент с RAG и coding-инструментами."""
    settings = Settings()
    settings.ensure_dirs()
    services = _make_retrieval(settings)
    audit = AuditLogger.create(settings)
    tools = build_rag_tools(
        services.searcher,
        services.answerer,
        default_book_ids=book,
        audit=audit,
    )
    tools.extend(build_coding_tools(settings, audit=audit))
    runner = AgentRunner(settings, tools, trace=services.trace)
    console.print("[bold]book-expert chat[/bold] — /exit для выхода")
    try:
        while True:
            try:
                line = typer.prompt("you").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if line in {"/exit", "/quit"}:
                break
            if not line:
                continue
            result = runner.run(line, book_ids=book or None)
            console.print(result.text)
            console.print(f"[dim]trace: {result.trace_path}[/dim]")
    finally:
        services.close()


@app.command()
def ui(
    host: str = typer.Option("127.0.0.1", "--host", help="Адрес прослушивания."),
    port: int = typer.Option(8000, "--port", min=1, max=65535),
) -> None:
    """Веб-чат в браузере (RAG и агент)."""
    import uvicorn

    from book_expert.web.app import create_app

    settings = Settings()
    settings.ensure_dirs()
    if host not in {"127.0.0.1", "localhost", "::1"}:
        console.print(
            "[yellow]warning[/yellow] у UI нет авторизации: не открывайте его "
            "в недоверенной сети."
        )
    console.print(f"[bold]book-expert ui[/bold] — http://{host}:{port}")
    uvicorn.run(
        create_app(settings),
        host=host,
        port=port,
        log_level=settings.log_level.lower(),
    )


@app.command()
def doctor(
    json_output: bool = typer.Option(False, "--json"),
    skip_qdrant: bool = typer.Option(False, "--skip-qdrant"),
    skip_ollama: bool = typer.Option(False, "--skip-ollama"),
) -> None:
    """Проверить локальное окружение."""
    from book_expert.doctor import run_doctor

    code = run_doctor(
        Settings(),
        as_json=json_output,
        skip_qdrant=skip_qdrant,
        skip_ollama=skip_ollama,
    )
    raise typer.Exit(code=code)


@app.command()
def mcp() -> None:
    """Запустить MCP stdio server."""
    from book_expert.mcp_server import create_server

    runtime = create_server(Settings())
    try:
        runtime.server.run("stdio")
    finally:
        runtime.close()


@app.command("stage-eval")
def stage_eval(
    cases: Path = typer.Argument(..., exists=True, dir_okay=False, readable=True),
    limit: int | None = typer.Option(None, "--limit", min=1),
    scope: ScopeMode = typer.Option(
        "expected",
        "--scope",
        help="expected — по разметке, router — выбор роутера, candidates — без LLM.",
    ),
    relevance_filter: bool = typer.Option(
        False,
        "--relevance-filter",
        help="Отбрасывать хиты, слабо покрывающие редкие термины вопроса.",
    ),
    faithfulness: bool = typer.Option(
        False,
        "--faithfulness",
        help="LLM-судья: поддержка утверждений ответа цитируемыми фрагментами.",
    ),
    k: int | None = typer.Option(
        None,
        "--k",
        min=1,
        max=20,
        help="Сколько хитов брать в ответ; по умолчанию BOOK_EXPERT_TOP_K.",
    ),
    judge_provider: str | None = typer.Option(
        None,
        "--judge-provider",
        help="Провайдер LLM-судьи (ollama|openai|deepseek); по умолчанию — как у ответов.",
    ),
    strict_judge: bool = typer.Option(
        False,
        "--strict-judge",
        help="Строгий судья: частичное совпадение темы не считается поддержкой.",
    ),
    output: Path = typer.Option(
        Path("data/artifacts/stage-eval.local.json"),
        "--output",
    ),
) -> None:
    """Per-stage eval: routing, retrieval, citations, latency, gate."""
    settings = Settings()
    report = run_stage_eval(
        settings,
        cases,
        limit=limit,
        scope=scope,
        relevance_filter=relevance_filter,
        faithfulness=faithfulness,
        k=k,
        judge_provider=judge_provider,
        strict_judge=strict_judge,
    )
    write_report(report, output)
    typer.echo(
        json.dumps(
            {
                "cases": report["cases"],
                "k": report["k"],
                "scope": report["scope"],
                "relevance_filter": report["relevance_filter"],
                "faithfulness_enabled": report["faithfulness_enabled"],
                "meta": report["meta"],
                "gate": report["gate"],
                "routing": report["routing"],
                "retrieval": report["retrieval"],
                "answers": report["answers"],
                "latency_ms": report["latency_ms"],
            },
            ensure_ascii=False,
        )
    )
    console.print(f"отчёт: {output}")


@app.command("agent-eval")
def agent_evaluate(
    cases: Path = typer.Argument(..., exists=True, dir_okay=False, readable=True),
    limit: int | None = typer.Option(None, "--limit", min=1),
    offset: int = typer.Option(0, "--offset", min=0),
    faithfulness: bool = typer.Option(
        False,
        "--faithfulness",
        help="LLM-судья: поддержка утверждений ответа цитируемыми фрагментами.",
    ),
    judge_provider: str | None = typer.Option(
        None,
        "--judge-provider",
        help="Провайдер LLM-судьи (ollama|openai|deepseek); по умолчанию — как у ответов.",
    ),
    strict_judge: bool = typer.Option(
        False,
        "--strict-judge",
        help="Строгий судья: частичное совпадение темы не считается поддержкой.",
    ),
    output: Path = typer.Option(
        Path("data/artifacts/agent-eval.local.json"),
        "--output",
    ),
) -> None:
    """Eval режима «Агент»: тот же набор, но ответ строит super-answerer."""
    from book_expert.eval.agent_eval import run_agent_eval

    report = run_agent_eval(
        Settings(),
        cases,
        limit=limit,
        offset=offset,
        faithfulness=faithfulness,
        judge_provider=judge_provider,
        strict_judge=strict_judge,
    )
    write_report(report, output)
    typer.echo(
        json.dumps(
            {
                "cases": report["cases"],
                "mode": report["mode"],
                "meta": report["meta"],
                "retrieval": report["retrieval"],
                "answers": report["answers"],
                "plan": report["plan"],
                "latency_ms": report["latency_ms"],
            },
            ensure_ascii=False,
        )
    )
    console.print(f"отчёт: {output}")


@app.command("eval")
def evaluate(
    cases: Path | None = typer.Argument(
        None, exists=True, dir_okay=False, readable=True
    ),
    k: int = typer.Option(5, "--k", min=1, max=20),
    route: bool = typer.Option(False, "--route", help="Включить авто-routing."),
    synthetic: bool = typer.Option(
        False, "--synthetic", help="Засеять synthetic-корпус и прогнать его."
    ),
    lexical_only: bool = typer.Option(
        False, "--lexical-only", help="Только SQLite FTS5, без Qdrant."
    ),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    """Прогнать eval: recall@k, MRR и citations."""
    settings = Settings()
    settings.ensure_dirs()
    services: RetrievalServices | None = None
    synthetic_book_id: str | None = None
    payload: dict[str, Any] | None = None
    try:
        services = _make_retrieval(settings)
        if synthetic:
            corpus = seed_synthetic_corpus(services.catalog)
            items = corpus.cases
            synthetic_book_id = corpus.book.id
        elif cases is None:
            raise ValueError("укажите cases.json или --synthetic")
        else:
            items = load_cases(cases)
        if not services.catalog.list_books():
            raise ValueError("каталог пуст: сначала выполните ingest")
        dense = not (lexical_only or synthetic)
        retrieval = run_retrieval_eval(services.searcher, items, k=k, dense=dense)
        answers = run_answer_eval(
            services.answerer,
            items,
            k=k,
            auto_route=route,
            dense=dense,
        )
        payload = {
            "retrieval": retrieval.to_dict(),
            "answers": answers.to_dict(),
        }
    except (ProviderError, ValueError, OSError) as exc:
        _fail(str(exc), json_output)
        return
    finally:
        if synthetic_book_id is not None and services is not None:
            services.catalog.delete_book(synthetic_book_id)
        if services is not None:
            services.close()

    if payload is None:
        return
    if json_output:
        typer.echo(json.dumps(payload, ensure_ascii=False))
        return
    retrieval = payload["retrieval"]
    answers = payload["answers"]
    console.print(f"recall@{retrieval['k']}: {retrieval['recall_at_k']:.3f}")
    console.print(f"MRR: {retrieval['mrr']:.3f}")
    console.print(f"citation precision: {answers['citation_precision']:.3f}")
    console.print(f"citation recall: {answers['citation_recall']:.3f}")


def _book_payload(book: Book, indexer: BookIndexer | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "id": book.id,
        "title": book.title,
        "author": book.author,
        "status": book.status.value,
        "format": book.format,
        "pages": book.page_count,
        "chunks": book.chunk_count,
        "content_hash": book.content_hash,
    }
    if indexer is not None:
        payload["points"] = indexer.count_points(book.id)
    return payload


def _outcome_payload(query: str, outcome: SearchOutcome) -> dict[str, Any]:
    return {
        "query": query,
        "sources": list(outcome.sources),
        "note": outcome.note,
        "hits": [_hit_payload(hit) for hit in outcome.hits],
    }


def _hit_payload(hit: SearchHit) -> dict[str, Any]:
    return {
        "chunk_id": hit.chunk_id,
        "book_id": hit.book_id,
        "book_title": hit.book_title,
        "page": hit.printed_page_label or hit.pdf_page_index,
        "section": hit.section_path,
        "format": hit.book_format,
        "score": round(hit.score, 6),
        "text": hit.text,
    }


def _answer_payload(answer: Answer) -> dict[str, Any]:
    return {
        "answer": answer.text,
        "refused": answer.refused,
        "confidence": answer.confidence,
        "book_ids": list(answer.book_ids),
        "route_reason": answer.route_reason,
        "plan": list(answer.plan),
        "rounds": answer.rounds,
        "max_rounds": answer.max_rounds,
        "citations": [
            {
                "book_id": citation.book_id,
                "book_title": citation.book_title,
                "page": citation.printed_page_label or citation.pdf_page_index,
                "chunk_id": citation.chunk_id,
                "section": citation.section_path,
                "format": citation.format,
                "number": citation.number,
                "quote": citation.quote,
            }
            for citation in answer.citations
        ],
    }


def _print_hits(payload: dict[str, Any]) -> None:
    hits = payload["hits"]
    if not hits:
        console.print(payload.get("note") or "ничего не найдено")
        return
    for index, hit in enumerate(hits, start=1):
        console.print(
            f"[bold]{index}. {hit['book_title']}[/bold] "
            f"p.{hit['page']} ({hit['score']:.4f})"
        )
        console.print(hit["text"][:300])


def _fail(message: str, json_output: bool) -> None:
    if json_output:
        typer.echo(json.dumps({"error": message}, ensure_ascii=False))
    else:
        console.print(f"[red]error[/red] {message}")
    raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
