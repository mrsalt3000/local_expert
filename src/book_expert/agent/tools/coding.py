"""Инструменты coding-агента: shell, fs, git с audit и политикой."""

from __future__ import annotations

import json
import time
from typing import Any, Callable

from langchain_core.tools import BaseTool, tool

from book_expert.agent.audit import AuditLogger
from book_expert.agent.policy import (
    CommandError,
    PolicyViolation,
    Workspace,
    run_command,
    run_git,
)
from book_expert.config import Settings

READ_LIMIT = 20_000


def build_coding_tools(
    settings: Settings,
    *,
    audit: AuditLogger | None = None,
) -> list[BaseTool]:
    workspace = Workspace(settings)
    logger = audit or AuditLogger.create(settings)

    @tool
    def read_file(path: str, max_bytes: int = READ_LIMIT) -> str:
        """Читает текстовый файл внутри workspace.

        path — относительный путь от корня workspace. Большие файлы усекаются.
        """

        def action() -> dict[str, Any]:
            limit = max(1, min(max_bytes, settings.max_output_bytes))
            return {
                "path": path,
                "text": workspace.read_text(path, limit),
            }

        return _call(
            "read_file", {"path": path, "max_bytes": max_bytes}, logger, action
        )

    @tool
    def list_dir(path: str = ".") -> str:
        """Показывает содержимое каталога внутри workspace."""

        def action() -> dict[str, Any]:
            return {"path": path, "entries": workspace.list_dir(path)}

        return _call("list_dir", {"path": path}, logger, action)

    tools: list[BaseTool] = [read_file, list_dir]
    if not settings.autonomy:
        return tools

    @tool
    def shell(command: str) -> str:
        """Выполняет одну команду в workspace без shell-операторов.

        Не используй sudo, устройства, системные службы и удаление корня.
        Команда запускается с таймаутом и ограничением вывода.
        """

        def action() -> dict[str, Any]:
            result = run_command(command, settings)
            return {
                "command": command,
                "exit_code": result.exit_code,
                "output": result.output,
                "truncated": result.truncated,
            }

        return _call("shell", {"command": command}, logger, action)

    @tool
    def write_file(path: str, content: str) -> str:
        """Записывает текстовый файл внутри workspace, создавая каталоги."""

        def action() -> dict[str, Any]:
            size = len(content.encode("utf-8"))
            if size > settings.max_output_bytes:
                raise PolicyViolation(
                    f"file too large: {size} bytes > {settings.max_output_bytes}"
                )
            target = workspace.write_text(path, content)
            return {"path": str(target), "chars": len(content)}

        return _call(
            "write_file",
            {"path": path, "content": content},
            logger,
            action,
        )

    @tool
    def git(arguments: list[str]) -> str:
        """Выполняет git-команду в workspace.

        arguments — список аргументов, например ["status", "--short"].
        Push запрещён без явного разрешения; force-push и reset --hard запрещены.
        """

        def action() -> dict[str, Any]:
            result = run_git(arguments, settings)
            return {
                "args": arguments,
                "exit_code": result.exit_code,
                "output": result.output,
                "truncated": result.truncated,
            }

        return _call("git", {"args": arguments}, logger, action)

    tools.extend([shell, write_file, git])
    return tools


def _call(
    name: str,
    args: dict[str, Any],
    logger: AuditLogger,
    action: Callable[[], dict[str, Any]],
) -> str:
    started = time.perf_counter()
    try:
        payload = action()
    except (PolicyViolation, CommandError, OSError, ValueError) as exc:
        duration = (time.perf_counter() - started) * 1000
        logger.log(name, args, ok=False, duration_ms=duration, error=str(exc))
        return json.dumps(
            {"ok": False, "tool": name, "error": str(exc)},
            ensure_ascii=False,
        )
    duration = (time.perf_counter() - started) * 1000
    logger.log(
        name,
        args,
        ok=True,
        exit_code=payload.get("exit_code"),
        duration_ms=duration,
    )
    return json.dumps({"ok": True, "tool": name, **payload}, ensure_ascii=False)


__all__ = ["build_coding_tools"]
