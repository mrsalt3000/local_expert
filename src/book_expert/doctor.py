"""Диагностика локального окружения Book Expert.

Проверяет Python, каталоги данных, Qdrant, Ollama, Docker и git-ignore для
приватных путей. Запуск::

    python -m book_expert.doctor
    python -m book_expert.doctor --json
    scripts/doctor

Exit code: 0 — нет fail-проверок, 1 — есть хотя бы одна fail-проверка.
"""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import subprocess
import sys
from collections.abc import Callable, Sequence
from contextlib import AbstractContextManager, nullcontext, suppress
from dataclasses import asdict, dataclass
from typing import Any

import httpx
from pydantic import ValidationError
from rich.console import Console
from rich.table import Table

from book_expert.config import Settings

OK = "ok"
WARN = "warn"
FAIL = "fail"

STATUS_COLORS = {OK: "green", WARN: "yellow", FAIL: "red"}
HTTP_TIMEOUT_S = 3.0
COMMAND_TIMEOUT_S = 15
QDRANT_CONTAINER = "book-expert-qdrant"
REQUIRED_PYTHON = (3, 12)

CommandRunner = Callable[[Sequence[str]], subprocess.CompletedProcess[str]]
Which = Callable[[str], "str | None"]


@dataclass(frozen=True)
class CheckResult:
    """Результат одной проверки."""

    name: str
    status: str
    detail: str
    hint: str | None = None


def _run_command(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(command),
        capture_output=True,
        text=True,
        timeout=COMMAND_TIMEOUT_S,
        check=False,
    )


def _call(
    runner: CommandRunner, command: Sequence[str]
) -> subprocess.CompletedProcess[str]:
    """Вызывает внешнюю команду, превращая ошибки запуска в failed-результат."""
    try:
        return runner(command)
    except (OSError, subprocess.SubprocessError) as exc:
        return subprocess.CompletedProcess(
            list(command), 1, "", f"{type(exc).__name__}: {exc}"
        )


def _first_line(text: str) -> str:
    for line in text.strip().splitlines():
        if line.strip():
            return line.strip()
    return ""


def check_python() -> CheckResult:
    current = sys.version_info[:2]
    version = platform.python_version()
    if current >= REQUIRED_PYTHON:
        return CheckResult("python", OK, f"{version} (нужен >= 3.12)")
    return CheckResult(
        "python",
        FAIL,
        f"{version} < 3.12",
        "установите Python 3.12+ и пересоздайте окружение: uv sync",
    )


def check_directories(settings: Settings) -> CheckResult:
    try:
        settings.ensure_dirs()
    except OSError as exc:
        return CheckResult(
            "data dirs",
            FAIL,
            f"каталоги не создаются: {exc}",
            f"проверьте права на {settings.data_path}",
        )

    for directory in (
        settings.raw_dir,
        settings.processed_dir,
        settings.artifacts_dir,
        settings.audit_dir,
    ):
        probe = directory / ".doctor-write-probe"
        try:
            probe.write_text("ok", encoding="utf-8")
        except OSError as exc:
            return CheckResult(
                "data dirs",
                FAIL,
                f"{directory} недоступен на запись: {exc}",
                "проверьте владельца и права каталога",
            )
        finally:
            with suppress(OSError):
                probe.unlink(missing_ok=True)

    return CheckResult(
        "data dirs",
        OK,
        f"raw/processed/artifacts/audit доступны на запись в {settings.data_path}",
    )


def check_provider(settings: Settings) -> CheckResult:
    if settings.provider == "openai":
        return CheckResult(
            "provider",
            WARN,
            "включён облачный OpenAI-compatible провайдер",
            "фрагменты книг уйдут наружу; используйте только с явного согласия",
        )
    return CheckResult("provider", OK, "локальный ollama")


def check_bwrap(which: Which) -> CheckResult:
    if which("bwrap") is None:
        return CheckResult(
            "sandbox",
            FAIL,
            "bubblewrap (bwrap) не найден",
            "установите bubblewrap: shell/git-инструменты агента отключены",
        )
    return CheckResult("sandbox", OK, "bubblewrap доступен")


def check_qdrant(
    url: str, client: httpx.Client, timeout: float = HTTP_TIMEOUT_S
) -> CheckResult:
    base = url.rstrip("/")
    errors: list[str] = []
    for path in ("/healthz", "/collections"):
        try:
            response = client.get(f"{base}{path}", timeout=timeout)
        except httpx.HTTPError as exc:
            errors.append(f"{path}: {type(exc).__name__}")
            continue
        if response.status_code == 200:
            return CheckResult("qdrant", OK, f"{base} отвечает на {path}")
        errors.append(f"{path}: HTTP {response.status_code}")
    return CheckResult(
        "qdrant",
        FAIL,
        f"недоступен {base} ({'; '.join(errors)})",
        "запустите: docker compose up -d",
    )


def _model_names(payload: Any) -> set[str]:
    if not isinstance(payload, dict):
        return set()
    names: set[str] = set()
    for item in payload.get("models", []):
        if not isinstance(item, dict):
            continue
        for key in ("model", "name"):
            value = item.get(key)
            if isinstance(value, str) and value:
                names.add(value)
    return names


def _has_model(names: set[str], model: str) -> bool:
    if model in names:
        return True
    return ":" not in model and f"{model}:latest" in names


def check_ollama(
    settings: Settings, client: httpx.Client, timeout: float = HTTP_TIMEOUT_S
) -> list[CheckResult]:
    base = settings.ollama_base_url.rstrip("/")
    try:
        response = client.get(f"{base}/api/tags", timeout=timeout)
    except httpx.HTTPError as exc:
        return [
            CheckResult(
                "ollama",
                FAIL,
                f"нет соединения с {base} ({type(exc).__name__})",
                "запустите: ollama serve",
            )
        ]
    if response.status_code != 200:
        return [
            CheckResult(
                "ollama",
                FAIL,
                f"HTTP {response.status_code} на {base}/api/tags",
                "проверьте: ollama serve",
            )
        ]
    try:
        names = _model_names(response.json())
    except ValueError:
        return [
            CheckResult(
                "ollama",
                FAIL,
                "ответ /api/tags не является JSON",
                "проверьте версию ollama: ollama --version",
            )
        ]

    results = [CheckResult("ollama", OK, f"{base}: моделей {len(names)}")]
    for role, model in (
        ("chat", settings.chat_model),
        ("embedding", settings.embedding_model),
    ):
        if _has_model(names, model):
            results.append(CheckResult(f"ollama:{role}", OK, f"{model} доступна"))
        else:
            results.append(
                CheckResult(
                    f"ollama:{role}",
                    FAIL,
                    f"{model} не найдена",
                    f"установите: ollama pull {model}",
                )
            )
    return results


def check_docker(which: Which, runner: CommandRunner) -> CheckResult:
    if which("docker") is None:
        return CheckResult(
            "docker",
            FAIL,
            "команда docker не найдена",
            "установите Docker Engine и Compose plugin",
        )
    result = _call(runner, ["docker", "version", "--format", "{{.Server.Version}}"])
    if result.returncode != 0:
        detail = _first_line(result.stderr) or "daemon недоступен"
        return CheckResult(
            "docker",
            FAIL,
            f"docker CLI есть, daemon недоступен: {detail}",
            "запустите Docker Engine",
        )
    return CheckResult("docker", OK, f"server {result.stdout.strip()}")


def check_docker_compose(which: Which, runner: CommandRunner) -> CheckResult:
    if which("docker") is None:
        return CheckResult(
            "docker compose",
            FAIL,
            "docker не найден",
            "установите Docker Engine и Compose plugin",
        )
    result = _call(runner, ["docker", "compose", "version"])
    if result.returncode != 0:
        return CheckResult(
            "docker compose",
            FAIL,
            "плагин compose недоступен",
            "установите docker-compose-plugin",
        )
    return CheckResult("docker compose", OK, _first_line(result.stdout) or "доступен")


def check_qdrant_container(runner: CommandRunner) -> CheckResult:
    fmt = (
        "{{.State.Status}}|"
        "{{if .State.Health}}{{.State.Health.Status}}{{else}}none{{end}}"
    )
    result = _call(runner, ["docker", "inspect", "--format", fmt, QDRANT_CONTAINER])
    if result.returncode != 0:
        return CheckResult(
            "qdrant container",
            WARN,
            f"контейнер {QDRANT_CONTAINER} не запущен",
            "запустите: docker compose up -d",
        )
    status, _, health = result.stdout.strip().partition("|")
    detail = f"{QDRANT_CONTAINER}: {status}"
    if health and health != "none":
        detail += f", health {health}"
    if status == "running" and health in ("healthy", "none"):
        return CheckResult("qdrant container", OK, detail)
    return CheckResult(
        "qdrant container",
        WARN,
        detail,
        "проверьте: docker compose logs qdrant",
    )


def _private_probes(settings: Settings) -> list[str] | None:
    """Относительные пути приватных файлов для проверки через git check-ignore."""
    probes: list[str] = []
    pairs = (
        (settings.raw_dir, "book.pdf"),
        (settings.processed_dir, "extracted.txt"),
        (settings.artifacts_dir, "catalog.sqlite3"),
    )
    for directory, name in pairs:
        try:
            probes.append(str((directory / name).relative_to(settings.workspace_path)))
        except ValueError:
            return None
    return probes


def check_gitignored(
    settings: Settings, which: Which, runner: CommandRunner
) -> CheckResult:
    if which("git") is None:
        return CheckResult("gitignore", WARN, "git не найден, проверка пропущена")
    repo = _call(
        runner,
        [
            "git",
            "-C",
            str(settings.workspace_path),
            "rev-parse",
            "--is-inside-work-tree",
        ],
    )
    if repo.returncode != 0 or repo.stdout.strip() != "true":
        return CheckResult("gitignore", WARN, "workspace не является git-репозиторием")
    probes = _private_probes(settings)
    if probes is None:
        return CheckResult(
            "gitignore", WARN, "data_dir вне workspace, проверка пропущена"
        )
    leaked = [
        probe
        for probe in probes
        if _call(
            runner,
            ["git", "-C", str(settings.workspace_path), "check-ignore", "-q", probe],
        ).returncode
        != 0
    ]
    if leaked:
        return CheckResult(
            "gitignore",
            WARN,
            "не игнорируются приватные пути: " + ", ".join(leaked),
            "добавьте их в .gitignore",
        )
    return CheckResult(
        "gitignore", OK, "data/raw, data/processed, data/artifacts игнорируются"
    )


class Doctor:
    """Набор проверок с внедряемыми зависимостями (используется в тестах)."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        client: httpx.Client | None = None,
        runner: CommandRunner | None = None,
        which: Which | None = None,
    ) -> None:
        self.settings = settings if settings is not None else Settings()
        self._client = client
        self._runner = runner or _run_command
        self._which = which or shutil.which

    def _client_scope(self) -> AbstractContextManager[httpx.Client]:
        if self._client is not None:
            return nullcontext(self._client)
        return httpx.Client(timeout=HTTP_TIMEOUT_S)

    def run(
        self, *, skip_qdrant: bool = False, skip_ollama: bool = False
    ) -> list[CheckResult]:
        results = [
            check_python(),
            check_directories(self.settings),
            check_provider(self.settings),
            check_bwrap(self._which),
            check_docker(self._which, self._runner),
        ]
        compose = check_docker_compose(self._which, self._runner)
        results.append(compose)
        if compose.status == OK:
            results.append(check_qdrant_container(self._runner))

        if skip_qdrant:
            results.append(CheckResult("qdrant", WARN, "пропущено: --skip-qdrant"))
        else:
            with self._client_scope() as client:
                results.append(check_qdrant(self.settings.qdrant_url, client))

        if skip_ollama:
            results.append(CheckResult("ollama", WARN, "пропущено: --skip-ollama"))
        else:
            with self._client_scope() as client:
                results.extend(check_ollama(self.settings, client))

        results.append(check_gitignored(self.settings, self._which, self._runner))
        return results


def overall_status(results: Sequence[CheckResult]) -> str:
    statuses = {result.status for result in results}
    if FAIL in statuses:
        return FAIL
    if WARN in statuses:
        return WARN
    return OK


def exit_code(results: Sequence[CheckResult]) -> int:
    return 0 if overall_status(results) != FAIL else 1


def render_table(results: Sequence[CheckResult], console: Console) -> None:
    table = Table(title="Book Expert doctor")
    table.add_column("Проверка", no_wrap=True)
    table.add_column("Статус", no_wrap=True)
    table.add_column("Детали", overflow="fold")
    table.add_column("Подсказка", overflow="fold")
    for result in results:
        color = STATUS_COLORS.get(result.status, "white")
        table.add_row(
            result.name,
            f"[{color}]{result.status}[/]",
            result.detail,
            result.hint or "",
        )
    console.print(table)
    failures = sum(result.status == FAIL for result in results)
    warnings = sum(result.status == WARN for result in results)
    console.print(
        f"Итог: {len(results)} проверок, {failures} fail, {warnings} warn — "
        f"{overall_status(results)}"
    )


def render_json(results: Sequence[CheckResult]) -> str:
    payload = {
        "status": overall_status(results),
        "checks": [asdict(result) for result in results],
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def run_doctor(
    settings: Settings,
    *,
    as_json: bool = False,
    skip_qdrant: bool = False,
    skip_ollama: bool = False,
    console: Console | None = None,
    client: httpx.Client | None = None,
    runner: CommandRunner | None = None,
    which: Which | None = None,
) -> int:
    doctor = Doctor(settings, client=client, runner=runner, which=which)
    results = doctor.run(skip_qdrant=skip_qdrant, skip_ollama=skip_ollama)
    if as_json:
        print(render_json(results))
    else:
        render_table(results, console or Console())
    return exit_code(results)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="book_expert.doctor",
        description="Проверка локального окружения Book Expert.",
    )
    parser.add_argument("--json", action="store_true", help="вывести результат в JSON")
    parser.add_argument(
        "--skip-qdrant", action="store_true", help="не проверять Qdrant"
    )
    parser.add_argument(
        "--skip-ollama", action="store_true", help="не проверять Ollama"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        settings = Settings()
    except ValidationError as exc:
        message = f"некорректная конфигурация: {exc}"
        if args.json:
            print(json.dumps({"status": FAIL, "error": message}, ensure_ascii=False))
        else:
            Console(stderr=True).print(f"[red]{FAIL}[/] {message}")
        return 1
    return run_doctor(
        settings,
        as_json=args.json,
        skip_qdrant=args.skip_qdrant,
        skip_ollama=args.skip_ollama,
    )


if __name__ == "__main__":
    raise SystemExit(main())
