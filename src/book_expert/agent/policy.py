"""Политика безопасности shell/fs/git-инструментов (ADR-0003)."""

from __future__ import annotations

import os
import shlex
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from book_expert.config import Settings

SHELLS = {"sh", "bash", "zsh", "dash", "ksh", "fish", "csh", "tcsh"}
DENIED = {
    "sudo",
    "su",
    "doas",
    "mkfs",
    "mkfs.ext4",
    "mkswap",
    "shutdown",
    "reboot",
    "poweroff",
    "halt",
    "init",
    "systemctl",
    "service",
    "dd",
    "chroot",
    "mount",
    "umount",
    "nsenter",
    "unshare",
    "modprobe",
    "insmod",
    "env",
    "timeout",
    "nice",
    "nohup",
    "setsid",
    "xargs",
    "watch",
    "chrt",
    "taskset",
    "stdbuf",
    "ionice",
    "flock",
}
DENIED_TOKENS = {"|", "||", "&&", ";", ">", ">>", "<", "<<"}
REMOTE_GIT = {
    "push",
    "fetch",
    "pull",
    "clone",
    "remote",
    "submodule",
    "ls-remote",
    "send-pack",
    "fetch-pack",
}
FORBIDDEN_GIT_OPTIONS = {
    "-C",
    "-c",
    "--git-dir",
    "--work-tree",
    "--namespace",
    "--exec-path",
    "--config-env",
    "--bare",
    "--super-prefix",
    "--attr-source",
}
PUSH_ALLOWED_FLAGS = {
    "-u",
    "--set-upstream",
    "--tags",
    "--follow-tags",
    "--all",
    "--branches",
    "--dry-run",
    "-n",
    "--verbose",
    "-v",
    "--progress",
    "--no-progress",
    "--atomic",
    "--porcelain",
    "--",
}
RESET_ALLOWED_FLAGS = {"--soft", "--mixed", "--keep", "-q", "--quiet", "--"}
EXEC_FLAGS = {"-exec", "-execdir", "-ok", "-okdir"}


class PolicyViolation(RuntimeError):
    """Действие запрещено политикой."""


class CommandError(RuntimeError):
    """Команда завершилась ошибкой или превысила лимит."""


@dataclass(frozen=True, slots=True)
class CommandResult:
    output: str
    exit_code: int
    truncated: bool


class Workspace:
    """Ограничивает файловые операции корнем workspace."""

    def __init__(self, settings: Settings) -> None:
        self.root = settings.workspace_path
        self.root.mkdir(parents=True, exist_ok=True)

    def resolve(self, path: str | Path) -> Path:
        candidate = Path(path).expanduser()
        candidate = candidate if candidate.is_absolute() else self.root / candidate
        resolved = candidate.resolve()
        if resolved != self.root and self.root not in resolved.parents:
            raise PolicyViolation(f"path escapes workspace: {path}")
        return resolved

    def read_text(self, path: str | Path, max_bytes: int) -> str:
        target = self.resolve(path)
        if not target.is_file():
            raise PolicyViolation(f"not a file: {path}")
        with target.open("rb") as handle:
            data = handle.read(max_bytes + 1)
        truncated = len(data) > max_bytes
        text = data[:max_bytes].decode("utf-8", errors="replace")
        return text + ("\n<truncated>" if truncated else "")

    def write_text(self, path: str | Path, content: str) -> Path:
        target = self.resolve(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return target

    def list_dir(self, path: str | Path = ".") -> list[dict[str, object]]:
        target = self.resolve(path)
        if not target.is_dir():
            raise PolicyViolation(f"not a directory: {path}")
        entries: list[dict[str, object]] = []
        for entry in sorted(target.iterdir(), key=lambda item: item.name):
            resolved = entry.resolve()
            if resolved != self.root and self.root not in resolved.parents:
                continue
            entries.append(
                {
                    "name": entry.name,
                    "type": "dir" if entry.is_dir() else "file",
                    "size": entry.stat().st_size if entry.is_file() else None,
                }
            )
        return entries


def check_command(
    command: str, settings: Settings, *, workspace: Workspace | None = None
) -> list[str]:
    tokens = shlex.split(command)
    if not tokens:
        raise PolicyViolation("empty command")
    workspace = workspace or Workspace(settings)
    name = Path(tokens[0]).name
    if name in SHELLS:
        raise PolicyViolation(f"shell interpreters are not allowed: {name}")
    if name in DENIED:
        raise PolicyViolation(f"command is denied: {name}")
    if any(token in DENIED_TOKENS for token in tokens):
        raise PolicyViolation("shell control operators are not allowed")
    if tokens[0].startswith("-"):
        raise PolicyViolation("command must start with an executable")
    if name == "git":
        check_git(tokens[1:], settings)
    if name == "find" and any(token in EXEC_FLAGS for token in tokens):
        raise PolicyViolation("find -exec is not allowed")
    if name == "rm":
        _check_rm(tokens, workspace)
    _check_path_tokens(tokens, workspace)
    return tokens


def _check_rm(tokens: Sequence[str], workspace: Workspace) -> None:
    recursive = any(
        token in {"-r", "-R", "--recursive"}
        or (token.startswith("-") and not token.startswith("--") and "r" in token)
        for token in tokens[1:]
    )
    if not recursive:
        return
    for token in tokens[1:]:
        if token.startswith("-"):
            continue
        try:
            target = workspace.resolve(token)
        except PolicyViolation:
            continue
        if target == workspace.root:
            raise PolicyViolation("recursive removal of workspace root is denied")


def check_git(args: Sequence[str], settings: Settings) -> list[str]:
    values = list(args)
    if not values:
        raise PolicyViolation("git args are empty")
    if any(not isinstance(value, str) for value in values):
        raise PolicyViolation("git args must be strings")
    for value in values:
        if value.startswith(("-C", "-c")) or any(
            value == option or value.startswith(f"{option}=")
            for option in FORBIDDEN_GIT_OPTIONS
            if option.startswith("--")
        ):
            raise PolicyViolation(f"git option is denied: {value}")

    subcommand_index = next(
        (index for index, value in enumerate(values) if not value.startswith("-")),
        None,
    )
    subcommand = values[subcommand_index] if subcommand_index is not None else None
    if subcommand in REMOTE_GIT and not settings.allow_remote:
        raise PolicyViolation(f"git {subcommand} requires BOOK_EXPERT_ALLOW_REMOTE=1")
    if (
        subcommand == "archive"
        and any(value.startswith("--remote") for value in values)
        and not settings.allow_remote
    ):
        raise PolicyViolation(
            "git archive --remote requires BOOK_EXPERT_ALLOW_REMOTE=1"
        )

    tail = values[subcommand_index + 1 :] if subcommand_index is not None else []
    if subcommand == "push":
        for value in tail:
            if value.startswith("+"):
                raise PolicyViolation("force push is denied")
            if value.startswith("-") and value not in PUSH_ALLOWED_FLAGS:
                raise PolicyViolation("force push is denied")
    if subcommand == "reset":
        for value in tail:
            if value.startswith("-") and value not in RESET_ALLOWED_FLAGS:
                raise PolicyViolation("git reset option is denied")
    if subcommand == "clean" and any(
        value.startswith("-") and "f" in value for value in tail
    ):
        raise PolicyViolation("git clean -f is denied")
    if subcommand == "rebase" and any(
        value.startswith("--inter") or value == "-i" for value in tail
    ):
        raise PolicyViolation("interactive rebase is denied")
    return values


def run_command(
    command: str,
    settings: Settings,
    *,
    timeout: float | None = None,
) -> CommandResult:
    workspace = Workspace(settings)
    tokens = check_command(command, settings, workspace=workspace)
    executable = _resolve_executable(tokens[0], workspace)
    effective_timeout = settings.command_timeout_s if timeout is None else timeout
    return _run_sandboxed(
        [executable, *tokens[1:]],
        settings,
        workspace,
        timeout=effective_timeout,
        label="command",
    )


def run_git(
    args: Sequence[str],
    settings: Settings,
    *,
    timeout: float | None = None,
) -> CommandResult:
    workspace = Workspace(settings)
    values = check_git(args, settings)
    git = shutil.which("git")
    if git is None:
        raise PolicyViolation("git not found")
    effective_timeout = settings.command_timeout_s if timeout is None else timeout
    return _run_sandboxed(
        [git, *values],
        settings,
        workspace,
        timeout=effective_timeout,
        label="git",
    )


def _run_sandboxed(
    argv: Sequence[str],
    settings: Settings,
    workspace: Workspace,
    *,
    timeout: float,
    label: str,
) -> CommandResult:
    prefix = _sandbox_prefix(settings, workspace)
    command = [*prefix, *argv]
    try:
        completed = subprocess.run(
            command,
            cwd=workspace.root,
            env=_environment(settings),
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise CommandError(f"{label} timed out after {timeout:g}s") from exc
    raw = (completed.stdout or "") + (completed.stderr or "")
    encoded = raw.encode("utf-8")
    truncated = len(encoded) > settings.max_output_bytes
    return CommandResult(
        output=encoded[: settings.max_output_bytes].decode("utf-8", errors="replace")
        + ("\n<truncated>" if truncated else ""),
        exit_code=completed.returncode,
        truncated=truncated,
    )


def _sandbox_prefix(settings: Settings, workspace: Workspace) -> list[str]:
    """bwrap: ro-root, writable workspace, private /tmp, no network by default."""
    bwrap = shutil.which("bwrap")
    if bwrap is None:
        raise PolicyViolation(
            "bubblewrap (bwrap) is required for shell/git tools; install it"
        )
    root = str(workspace.root)
    prefix = [
        bwrap,
        "--die-with-parent",
        "--unshare-pid",
        "--unshare-ipc",
        "--unshare-uts",
        "--new-session",
        "--ro-bind",
        "/",
        "/",
        "--dev",
        "/dev",
        "--proc",
        "/proc",
        "--tmpfs",
        "/tmp",
        "--bind",
        root,
        root,
        "--chdir",
        root,
        "--clearenv",
    ]
    if not settings.allow_remote:
        prefix.append("--unshare-net")
    for key, value in _environment(settings).items():
        prefix.extend(["--setenv", key, value])
    prefix.append("--")
    return prefix


def _check_path_tokens(tokens: Sequence[str], workspace: Workspace) -> None:
    for token in tokens[1:]:
        if token.startswith("-") or "=" in token:
            continue
        if not (token.startswith(("/", ".", "..")) or "/" in token):
            continue
        workspace.resolve(token)


def _resolve_executable(token: str, workspace: Workspace) -> str:
    if "/" in token:
        resolved = workspace.resolve(token)
        if not resolved.is_file() or not os.access(resolved, os.X_OK):
            raise PolicyViolation(f"executable is not allowed: {token}")
        return str(resolved)
    found = shutil.which(token)
    if found is None:
        raise PolicyViolation(f"command not found: {token}")
    return found


def _environment(settings: Settings) -> Mapping[str, str]:
    home = settings.artifacts_dir / "home"
    home.mkdir(parents=True, exist_ok=True)
    return {
        "PATH": os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin"),
        "HOME": str(home),
        "LANG": os.environ.get("LANG", "C.UTF-8"),
        "LC_ALL": os.environ.get("LC_ALL", "C.UTF-8"),
        "TERM": "dumb",
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_AUTHOR_NAME": "Book Expert",
        "GIT_AUTHOR_EMAIL": "book-expert@localhost",
        "GIT_COMMITTER_NAME": "Book Expert",
        "GIT_COMMITTER_EMAIL": "book-expert@localhost",
    }


__all__ = [
    "CommandError",
    "CommandResult",
    "PolicyViolation",
    "Workspace",
    "check_command",
    "check_git",
    "run_command",
    "run_git",
]
