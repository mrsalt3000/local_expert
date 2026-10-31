"""Тесты doctor: без живых Qdrant/Ollama и без Docker."""

from __future__ import annotations

import json
import subprocess
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import httpx
import pytest

from book_expert import doctor
from book_expert.config import Settings

REPO_ROOT = Path(__file__).resolve().parents[1]


def _completed(
    command: Sequence[str], code: int = 0, stdout: str = "", stderr: str = ""
) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(list(command), code, stdout, stderr)


class FakeRunner:
    """Подменяет docker/git CLI, записывая вызовы."""

    def __init__(
        self,
        *,
        container: tuple[int, str] = (0, "running|healthy\n"),
        not_ignored: tuple[str, ...] = (),
        has_git_repo: bool = True,
    ) -> None:
        self.calls: list[tuple[str, ...]] = []
        self.container = container
        self.not_ignored = not_ignored
        self.has_git_repo = has_git_repo

    def __call__(self, command: Sequence[str]) -> subprocess.CompletedProcess[str]:
        key = tuple(command)
        self.calls.append(key)
        if key[:2] == ("docker", "version"):
            return _completed(command, stdout="29.5.2\n")
        if key[:3] == ("docker", "compose", "version"):
            return _completed(command, stdout="Docker Compose version v5.1.4\n")
        if key[:3] == ("docker", "inspect", "--format"):
            code, output = self.container
            return _completed(command, code, output)
        if "rev-parse" in key:
            if not self.has_git_repo:
                return _completed(command, 128, stderr="not a git repository")
            return _completed(command, stdout="true\n")
        if "check-ignore" in key:
            return _completed(command, 1 if key[-1] in self.not_ignored else 0)
        return _completed(command)


def _which(_name: str) -> str:
    return "/usr/bin/tool"


def _client(
    *,
    qdrant_status: int = 200,
    qdrant_error: bool = False,
    ollama_models: tuple[str, ...] = ("qwen3.5:4b", "bge-m3:latest"),
    ollama_status: int = 200,
) -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.port == 6333:
            if qdrant_error:
                raise httpx.ConnectError("connection refused", request=request)
            return httpx.Response(qdrant_status, json={"title": "qdrant"})
        if request.url.path == "/api/tags":
            if ollama_status != 200:
                return httpx.Response(ollama_status, json={"error": "boom"})
            models = [{"name": name} for name in ollama_models]
            return httpx.Response(200, json={"models": models})
        return httpx.Response(404)

    return httpx.Client(transport=httpx.MockTransport(handler))


def _doctor(tmp_path: Path, **kwargs: Any) -> doctor.Doctor:
    settings = Settings(_env_file=None, workspace=tmp_path)
    return doctor.Doctor(
        settings,
        client=_client(**kwargs),
        runner=FakeRunner(),
        which=_which,
    )


def _by_name(results: Sequence[doctor.CheckResult]) -> dict[str, doctor.CheckResult]:
    return {result.name: result for result in results}


def test_success_scenario(tmp_path: Path) -> None:
    results = _doctor(tmp_path).run()

    assert doctor.exit_code(results) == 0
    assert [result for result in results if result.status == doctor.FAIL] == []
    checks = _by_name(results)
    assert checks["python"].status == doctor.OK
    assert checks["data dirs"].status == doctor.OK
    assert checks["qdrant"].status == doctor.OK
    assert checks["ollama:chat"].status == doctor.OK
    assert checks["ollama:embedding"].status == doctor.OK
    assert checks["gitignore"].status == doctor.OK
    assert (tmp_path / "data" / "raw").is_dir()
    assert (tmp_path / "data" / "artifacts" / "audit").is_dir()


def test_qdrant_unavailable_fails(tmp_path: Path) -> None:
    results = _doctor(tmp_path, qdrant_status=503).run()

    qdrant = _by_name(results)["qdrant"]
    assert qdrant.status == doctor.FAIL
    assert "docker compose up -d" in (qdrant.hint or "")
    assert "HTTP 503" in qdrant.detail
    assert doctor.exit_code(results) == 1


def test_qdrant_connection_error_fails(tmp_path: Path) -> None:
    results = _doctor(tmp_path, qdrant_error=True).run()

    qdrant = _by_name(results)["qdrant"]
    assert qdrant.status == doctor.FAIL
    assert "/healthz" in qdrant.detail
    assert "/collections" in qdrant.detail
    assert doctor.exit_code(results) == 1


def test_missing_ollama_model_fails(tmp_path: Path) -> None:
    results = _doctor(tmp_path, ollama_models=("qwen3.5:4b",)).run()

    checks = _by_name(results)
    assert checks["ollama"].status == doctor.OK
    assert checks["ollama:chat"].status == doctor.OK
    embedding = checks["ollama:embedding"]
    assert embedding.status == doctor.FAIL
    assert "bge-m3" in embedding.detail
    assert "ollama pull bge-m3" in (embedding.hint or "")
    assert doctor.exit_code(results) == 1


def test_ollama_unreachable_fails(tmp_path: Path) -> None:
    results = _doctor(tmp_path, ollama_status=500).run()

    ollama = _by_name(results)["ollama"]
    assert ollama.status == doctor.FAIL
    assert "ollama serve" in (ollama.hint or "")


def test_json_output_and_exit_code(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    settings = Settings(_env_file=None, workspace=tmp_path)
    code = doctor.run_doctor(
        settings,
        as_json=True,
        skip_qdrant=True,
        skip_ollama=True,
        runner=FakeRunner(),
        which=_which,
    )

    payload = json.loads(capsys.readouterr().out)
    assert code == 0
    assert payload["status"] == doctor.WARN
    names = [check["name"] for check in payload["checks"]]
    assert "qdrant" in names
    assert "ollama" in names
    skipped = {check["name"]: check for check in payload["checks"]}
    assert "skip-qdrant" in skipped["qdrant"]["detail"]


def test_json_output_reports_failure(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    settings = Settings(_env_file=None, workspace=tmp_path)
    code = doctor.run_doctor(
        settings,
        as_json=True,
        client=_client(ollama_models=()),
        runner=FakeRunner(),
        which=_which,
    )

    payload = json.loads(capsys.readouterr().out)
    assert code == 1
    assert payload["status"] == doctor.FAIL


def test_gitignore_warning_is_not_failure(tmp_path: Path) -> None:
    settings = Settings(_env_file=None, workspace=tmp_path)
    results = doctor.Doctor(
        settings,
        client=_client(),
        runner=FakeRunner(not_ignored=("data/raw/book.pdf",)),
        which=_which,
    ).run()

    gitignore = _by_name(results)["gitignore"]
    assert gitignore.status == doctor.WARN
    assert "data/raw/book.pdf" in gitignore.detail
    assert doctor.exit_code(results) == 0


def test_gitignore_without_repo_is_warning(tmp_path: Path) -> None:
    settings = Settings(_env_file=None, workspace=tmp_path)
    results = doctor.Doctor(
        settings,
        client=_client(),
        runner=FakeRunner(has_git_repo=False),
        which=_which,
    ).run()

    assert _by_name(results)["gitignore"].status == doctor.WARN
    assert doctor.exit_code(results) == 0


def test_container_not_running_is_warning(tmp_path: Path) -> None:
    settings = Settings(_env_file=None, workspace=tmp_path)
    results = doctor.Doctor(
        settings,
        client=_client(),
        runner=FakeRunner(container=(1, "")),
        which=_which,
    ).run()

    container = _by_name(results)["qdrant container"]
    assert container.status == doctor.WARN
    assert "docker compose up -d" in (container.hint or "")


def test_parser_flags() -> None:
    args = doctor.build_parser().parse_args(["--json", "--skip-qdrant"])
    assert args.json is True
    assert args.skip_qdrant is True
    assert args.skip_ollama is False


def test_env_example_matches_settings() -> None:
    settings = Settings(_env_file=REPO_ROOT / ".env.example")
    assert settings.provider == "ollama"
    assert settings.qdrant_collection == "books"
    assert settings.embedding_dim == 1024
