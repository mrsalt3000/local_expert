"""Тесты конфигурации: env-префикс, пути, валидация и confinement."""

from __future__ import annotations

from pathlib import Path

import pytest

from book_expert.config import Settings


def test_defaults(tmp_path: Path) -> None:
    settings = Settings(_env_file=None, workspace=tmp_path)
    assert settings.provider == "ollama"
    assert settings.chunk_size == 1000
    assert settings.chunk_overlap == 150
    assert settings.qdrant_url == "http://127.0.0.1:6333"
    assert settings.ollama_num_ctx == 8192
    assert settings.lexical_query_mode == "and"
    assert settings.autonomy is True
    assert settings.allow_remote is False


def test_env_prefix_is_applied(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BOOK_EXPERT_TOP_K", "3")
    monkeypatch.setenv("BOOK_EXPERT_PROVIDER", "openai")
    settings = Settings(_env_file=None, workspace=tmp_path)
    assert settings.top_k == 3
    assert settings.provider == "openai"


def test_derived_paths(tmp_path: Path) -> None:
    settings = Settings(_env_file=None, workspace=tmp_path)
    assert settings.raw_dir == tmp_path / "data" / "raw"
    assert settings.processed_dir == tmp_path / "data" / "processed"
    assert settings.artifacts_dir == tmp_path / "data" / "artifacts"
    assert settings.catalog_path == tmp_path / "data" / "artifacts" / "catalog.sqlite3"
    assert settings.qdrant_storage_path == tmp_path / "data" / "artifacts" / "qdrant"


def test_absolute_data_dir_is_kept(tmp_path: Path) -> None:
    data_dir = tmp_path / "elsewhere"
    settings = Settings(_env_file=None, workspace=tmp_path, data_dir=data_dir)
    assert settings.data_path == data_dir


def test_ensure_dirs(tmp_path: Path) -> None:
    settings = Settings(_env_file=None, workspace=tmp_path)
    settings.ensure_dirs()
    assert settings.raw_dir.is_dir()
    assert settings.processed_dir.is_dir()
    assert settings.artifacts_dir.is_dir()
    assert settings.audit_dir.is_dir()


def test_resolve_in_workspace_allows_inside(tmp_path: Path) -> None:
    settings = Settings(_env_file=None, workspace=tmp_path)
    assert settings.resolve_in_workspace("data/raw/book.pdf") == (
        tmp_path / "data" / "raw" / "book.pdf"
    )


def test_resolve_in_workspace_rejects_escape(tmp_path: Path) -> None:
    settings = Settings(_env_file=None, workspace=tmp_path)
    with pytest.raises(ValueError, match="escapes workspace"):
        settings.resolve_in_workspace("../outside.pdf")


def test_overlap_must_be_smaller_than_chunk_size(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="chunk_overlap"):
        Settings(
            _env_file=None,
            workspace=tmp_path,
            chunk_size=100,
            chunk_overlap=100,
        )


def test_ollama_num_ctx_must_be_positive(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="ollama_num_ctx"):
        Settings(_env_file=None, workspace=tmp_path, ollama_num_ctx=0)


def test_threshold_range(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="router_confidence_threshold"):
        Settings(_env_file=None, workspace=tmp_path, router_confidence_threshold=1.5)


def test_with_provider_picks_that_provider_chat_model(tmp_path: Path) -> None:
    settings = Settings(
        _env_file=None,
        workspace=tmp_path,
        provider="deepseek",
        chat_model="deepseek-chat",
    )

    assert settings.with_provider("openai").chat_model == "gpt-4o-mini"
    assert settings.with_provider("deepseek").chat_model == "deepseek-chat"
    ollama = settings.with_provider("ollama")
    assert ollama.provider == "ollama"
    assert ollama.chat_model == "qwen3.5:4b"


def test_with_provider_keeps_settings_for_same_provider(tmp_path: Path) -> None:
    settings = Settings(
        _env_file=None, workspace=tmp_path, provider="ollama", chat_model="custom:1b"
    )

    assert settings.with_provider("ollama") is settings
    assert settings.with_provider("ollama").chat_model == "custom:1b"
