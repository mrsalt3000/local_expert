"""Тесты политики, confinement, shell/fs/git-инструментов и audit."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest

from book_expert.agent.audit import AuditLogger
from book_expert.agent.policy import (
    CommandError,
    PolicyViolation,
    Workspace,
    check_command,
    check_git,
    run_command,
)
from book_expert.agent.tools.coding import build_coding_tools
from book_expert.config import Settings


def _settings(tmp_path: Path, **overrides: Any) -> Settings:
    values: dict[str, Any] = {
        "workspace": tmp_path,
        "data_dir": tmp_path / "data",
        "command_timeout_s": 1,
        "max_output_bytes": 1000,
    }
    values.update(overrides)
    return Settings(_env_file=None, **values)


def _tool(tools: list[Any], name: str) -> Any:
    return next(tool for tool in tools if tool.name == name)


def test_workspace_confines_paths(tmp_path: Path) -> None:
    workspace = Workspace(_settings(tmp_path))
    workspace.write_text("nested/file.txt", "hello")

    assert workspace.read_text("nested/file.txt", 100) == "hello"
    assert workspace.list_dir("nested")[0]["name"] == "file.txt"
    with pytest.raises(PolicyViolation, match="escapes workspace"):
        workspace.read_text("../outside.txt", 100)


def test_workspace_blocks_symlink_escape(tmp_path: Path) -> None:
    outside = tmp_path.parent / "outside-synthetic.txt"
    outside.write_text("secret", encoding="utf-8")
    link = tmp_path / "link.txt"
    os.symlink(outside, link)

    workspace = Workspace(_settings(tmp_path))
    with pytest.raises(PolicyViolation, match="escapes workspace"):
        workspace.read_text("link.txt", 100)


def test_command_policy_denies_dangerous_commands(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    for command in (
        "sudo ls",
        "rm -rf /",
        "rm -rf .",
        "ls | cat",
        "bash -c 'echo hi'",
        "env sh -c id",
        "timeout 1 sh -c id",
        "find . -exec sh -c id \\;",
        "git push origin main",
    ):
        with pytest.raises(PolicyViolation):
            check_command(command, settings)


def test_git_remote_operations_require_flag(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    for args in (
        ["push"],
        ["fetch"],
        ["pull"],
        ["clone", "https://example.invalid/repo.git"],
        ["remote", "add", "origin", "https://example.invalid/repo.git"],
    ):
        with pytest.raises(PolicyViolation, match="ALLOW_REMOTE"):
            check_git(args, settings)
    assert check_git(["status", "--short"], settings) == ["status", "--short"]


def test_command_policy_allows_safe_command(tmp_path: Path) -> None:
    tokens = check_command("printf hello", _settings(tmp_path))
    assert tokens == ["printf", "hello"]


def test_git_policy(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    with pytest.raises(PolicyViolation, match="ALLOW_REMOTE"):
        check_git(["push"], settings)
    with pytest.raises(PolicyViolation, match="force"):
        check_git(["push", "--force"], _settings(tmp_path, allow_remote=True))
    with pytest.raises(PolicyViolation, match="reset option"):
        check_git(["reset", "--hard"], settings)
    with pytest.raises(PolicyViolation, match="reset option"):
        check_git(["reset", "--har"], settings)
    with pytest.raises(PolicyViolation, match="clean"):
        check_git(["clean", "-fdx"], settings)
    assert check_git(["status", "--short"], settings) == ["status", "--short"]


def test_git_policy_blocks_force_refspec_and_global_options(tmp_path: Path) -> None:
    allow = _settings(tmp_path, allow_remote=True)
    for args in (
        ["push", "origin", "+main"],
        ["push", "--mirror"],
        ["push", "--mirro"],
        ["push", "--force-with-lease=main", "origin", "main"],
        ["push", "--force-with-leas", "origin", "main"],
        ["push", "-uf", "origin", "main"],
    ):
        with pytest.raises(PolicyViolation, match="force"):
            check_git(args, allow)
    for args in (
        ["-C", "repo", "push"],
        ["-c", "core.hooksPath=/tmp", "push"],
        ["--git-dir=.git", "push"],
        ["--work-tree", ".", "push"],
        ["--config-env=FOO=bar", "push"],
        ["--bare", "push"],
        ["--attr-source", "HEAD", "push", "--force"],
        ["--attr-source=HEAD", "push", "--force"],
    ):
        with pytest.raises(PolicyViolation, match="denied"):
            check_git(args, allow)
    with pytest.raises(PolicyViolation, match="ALLOW_REMOTE"):
        check_git(["--no-pager", "push"], _settings(tmp_path))
    assert check_git(["stash", "push", "-m", "wip"], _settings(tmp_path)) == [
        "stash",
        "push",
        "-m",
        "wip",
    ]
    assert check_git(["log", "--grep", "push"], _settings(tmp_path)) == [
        "log",
        "--grep",
        "push",
    ]


def test_run_command_output_and_truncation(tmp_path: Path) -> None:
    result = run_command("printf abc", _settings(tmp_path))
    assert result.output == "abc"
    assert result.exit_code == 0
    assert result.truncated is False

    truncated = run_command(
        "printf abcdefghijklmnop",
        _settings(tmp_path, max_output_bytes=5),
    )
    assert truncated.truncated is True
    assert truncated.output.startswith("abcde")
    assert "<truncated>" in truncated.output


def test_run_command_timeout(tmp_path: Path) -> None:
    with pytest.raises(CommandError, match="timed out"):
        run_command("sleep 2", _settings(tmp_path), timeout=0.1)


def test_fs_tools(tmp_path: Path) -> None:
    tools = build_coding_tools(_settings(tmp_path))
    write_tool = _tool(tools, "write_file")
    read_tool = _tool(tools, "read_file")
    list_tool = _tool(tools, "list_dir")

    written = json.loads(write_tool.invoke({"path": "notes/a.txt", "content": "hello"}))
    assert written["ok"] is True
    assert written["chars"] == 5

    read = json.loads(read_tool.invoke({"path": "notes/a.txt"}))
    assert read["text"] == "hello"

    listing = json.loads(list_tool.invoke({"path": "notes"}))
    assert listing["entries"][0]["name"] == "a.txt"


def test_shell_tool_returns_json(tmp_path: Path) -> None:
    tools = build_coding_tools(_settings(tmp_path))
    shell_tool = _tool(tools, "shell")

    payload = json.loads(shell_tool.invoke({"command": "printf hi"}))

    assert payload["ok"] is True
    assert payload["output"] == "hi"
    assert payload["exit_code"] == 0


def test_shell_tool_reports_policy_error(tmp_path: Path) -> None:
    tools = build_coding_tools(_settings(tmp_path))
    shell_tool = _tool(tools, "shell")

    payload = json.loads(shell_tool.invoke({"command": "sudo ls"}))

    assert payload["ok"] is False
    assert "denied" in payload["error"]


def test_git_tool_in_workspace(tmp_path: Path) -> None:
    subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
    tools = build_coding_tools(_settings(tmp_path))
    git_tool = _tool(tools, "git")

    payload = json.loads(git_tool.invoke({"arguments": ["status", "--short"]}))

    assert payload["ok"] is True
    assert payload["exit_code"] == 0


def test_audit_redacts_content(tmp_path: Path) -> None:
    audit = AuditLogger(tmp_path / "audit" / "tools.jsonl")
    tools = build_coding_tools(_settings(tmp_path), audit=audit)
    write_tool = _tool(tools, "write_file")

    write_tool.invoke({"path": "a.txt", "content": "super secret synthetic text"})

    record = json.loads(audit.path.read_text(encoding="utf-8").splitlines()[0])
    assert record["tool"] == "write_file"
    assert record["ok"] is True
    assert record["args"]["content"] == "<27 chars>"
    assert "super secret" not in audit.path.read_text(encoding="utf-8")


def test_write_file_rejects_oversized_content(tmp_path: Path) -> None:
    tools = build_coding_tools(_settings(tmp_path, max_output_bytes=10))
    write_tool = _tool(tools, "write_file")

    payload = json.loads(write_tool.invoke({"path": "big.txt", "content": "x" * 20}))

    assert payload["ok"] is False
    assert "too large" in payload["error"]
    assert not (tmp_path / "big.txt").exists()


def test_autonomy_off_exposes_read_only_tools(tmp_path: Path) -> None:
    tools = build_coding_tools(_settings(tmp_path, autonomy=False))
    assert {tool.name for tool in tools} == {"read_file", "list_dir"}


@pytest.mark.skipif(shutil.which("bwrap") is None, reason="bwrap is not installed")
def test_sandbox_blocks_write_outside_workspace(tmp_path: Path) -> None:
    target = Path.home() / f".book-expert-escape-{os.getpid()}"
    target.unlink(missing_ok=True)

    result = run_command(
        f"python3 -c \"open('{target}', 'w').write('escape')\"",
        _settings(tmp_path),
    )

    assert result.exit_code != 0
    assert target.exists() is False


@pytest.mark.skipif(shutil.which("bwrap") is None, reason="bwrap is not installed")
def test_sandbox_blocks_network(tmp_path: Path) -> None:
    result = run_command(
        "curl -sS --max-time 2 http://example.com",
        _settings(tmp_path),
    )
    assert result.exit_code != 0


@pytest.mark.skipif(shutil.which("bwrap") is None, reason="bwrap is not installed")
def test_sandbox_allows_write_inside_workspace(tmp_path: Path) -> None:
    result = run_command(
        "python3 -c \"open('inside.txt', 'w').write('ok')\"",
        _settings(tmp_path),
    )
    assert result.exit_code == 0
    assert (tmp_path / "inside.txt").read_text(encoding="utf-8") == "ok"
