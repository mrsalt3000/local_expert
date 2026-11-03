"""Конфигурация Book Expert: env/.env, пути и лимиты."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

ProviderName = Literal["ollama", "openai", "deepseek"]
LexicalQueryMode = Literal["and", "or_prefix5"]


class Settings(BaseSettings):
    """Настройки приложения. Все переменные окружения имеют префикс BOOK_EXPERT_."""

    model_config = SettingsConfigDict(
        env_prefix="BOOK_EXPERT_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    workspace: Path = Field(default_factory=Path.cwd)
    data_dir: Path = Path("data")

    qdrant_url: str = "http://127.0.0.1:6333"
    qdrant_api_key: SecretStr | None = None
    qdrant_collection: str = "books"

    provider: ProviderName = "ollama"
    ollama_base_url: str = "http://127.0.0.1:11434"
    ollama_think: bool = False
    ollama_num_ctx: int = 8192
    chat_model: str = "qwen3.5:4b"
    embedding_model: str = "bge-m3"
    embedding_dim: int = 1024

    openai_api_key: SecretStr | None = None
    openai_base_url: str | None = None
    openai_chat_model: str = "gpt-4o-mini"
    openai_embedding_model: str = "text-embedding-3-small"

    deepseek_api_key: SecretStr | None = None
    deepseek_base_url: str = "https://api.deepseek.com/v1"
    deepseek_chat_model: str = "deepseek-chat"

    chunk_size: int = 1000
    chunk_overlap: int = 150
    strip_running_headers: bool = False
    top_k: int = 20
    router_confidence_threshold: float = 0.6
    router_use_llm: bool = False
    relevance_filter_min_ratio: float = 0.5
    answer_relevance_filter: bool = True
    answer_max_citations: int = 5
    candidate_limit: int = 24
    advanced_query_limit: int = 8
    super_max_rounds: int = 5
    super_refine_enabled: bool = False
    super_scope_limit: int = 3
    rrf_dense_weight: float = 1.0
    rrf_lexical_weight: float = 0.1
    lexical_query_mode: LexicalQueryMode = "and"
    lexical_use_lemmas: bool = True
    page_context_enabled: bool = False
    query_rewrite_enabled: bool = True
    max_upload_mb: int = 200
    import_max_files: int = 500
    import_max_file_mb: int = 500
    import_max_total_mb: int = 50_000

    rerank_enabled: bool = False
    rerank_model: str = "jinaai/jina-reranker-v2-base-multilingual"
    rerank_top_n: int = 20

    agent_max_steps: int = 12
    agent_timeout_s: int = 120
    command_timeout_s: int = 30
    max_output_bytes: int = 100_000
    autonomy: bool = True
    allow_remote: bool = False
    trace_text: bool = False
    log_level: str = "INFO"

    @field_validator("workspace", "data_dir", mode="after")
    @classmethod
    def _expand_path(cls, value: Path) -> Path:
        return value.expanduser()

    @field_validator("router_confidence_threshold", "relevance_filter_min_ratio")
    @classmethod
    def _check_threshold(cls, value: float) -> float:
        if not 0.0 <= value <= 1.0:
            raise ValueError("threshold must be in [0, 1]")
        return value

    @field_validator("rrf_dense_weight", "rrf_lexical_weight")
    @classmethod
    def _check_weight(cls, value: float) -> float:
        if value < 0.0:
            raise ValueError("rrf weights must be non-negative")
        return value

    @field_validator(
        "chunk_size",
        "top_k",
        "rerank_top_n",
        "ollama_num_ctx",
        "agent_max_steps",
        "agent_timeout_s",
        "command_timeout_s",
        "advanced_query_limit",
        "super_max_rounds",
        "max_upload_mb",
        "import_max_files",
        "import_max_file_mb",
        "import_max_total_mb",
    )
    @classmethod
    def _check_positive(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("value must be positive")
        return value

    @field_validator("super_scope_limit")
    @classmethod
    def _check_super_scope_limit(cls, value: int) -> int:
        if value < 0:
            raise ValueError("value must be non-negative")
        return value

    @field_validator("candidate_limit")
    @classmethod
    def _check_candidate_limit(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("value must be positive")
        return value

    @field_validator("answer_max_citations")
    @classmethod
    def _check_citations_non_negative(cls, value: int) -> int:
        if value < 0:
            raise ValueError("value must be non-negative")
        return value

    @field_validator("chunk_overlap")
    @classmethod
    def _check_non_negative(cls, value: int) -> int:
        if value < 0:
            raise ValueError("chunk_overlap must be non-negative")
        return value

    @model_validator(mode="after")
    def _check_chunking(self) -> Settings:
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        if self.rrf_dense_weight == 0.0 and self.rrf_lexical_weight == 0.0:
            raise ValueError("at least one RRF weight must be positive")
        return self

    @property
    def workspace_path(self) -> Path:
        return self.workspace.resolve()

    @property
    def data_path(self) -> Path:
        return self._resolve(self.data_dir)

    @property
    def raw_dir(self) -> Path:
        return self.data_path / "raw"

    @property
    def books_dir(self) -> Path:
        return self.raw_dir / "books"

    @property
    def processed_dir(self) -> Path:
        return self.data_path / "processed"

    @property
    def artifacts_dir(self) -> Path:
        return self.data_path / "artifacts"

    @property
    def eval_dir(self) -> Path:
        return self.data_path / "eval"

    @property
    def catalog_path(self) -> Path:
        return self.artifacts_dir / "catalog.sqlite3"

    @property
    def book_profiles_path(self) -> Path:
        return self.artifacts_dir / "book_profiles.local.json"

    @property
    def qdrant_storage_path(self) -> Path:
        return self.artifacts_dir / "qdrant"

    @property
    def audit_dir(self) -> Path:
        return self.artifacts_dir / "audit"

    def _resolve(self, path: Path) -> Path:
        return path if path.is_absolute() else self.workspace_path / path

    def ensure_dirs(self) -> None:
        for directory in (
            self.raw_dir,
            self.books_dir,
            self.processed_dir,
            self.artifacts_dir,
            self.eval_dir,
            self.audit_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)

    def resolve_in_workspace(self, path: str | Path) -> Path:
        """Резолвит путь и запрещает выход за workspace."""
        candidate = Path(path).expanduser()
        candidate = (
            candidate if candidate.is_absolute() else self.workspace_path / candidate
        )
        resolved = candidate.resolve()
        if (
            resolved != self.workspace_path
            and self.workspace_path not in resolved.parents
        ):
            raise ValueError(f"path escapes workspace: {path}")
        return resolved
