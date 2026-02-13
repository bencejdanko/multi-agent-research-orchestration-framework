"""Centralised configuration loaded from .env / environment."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, Field

# Load .env from project root
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(_PROJECT_ROOT / ".env")


class Settings(BaseModel):
    """Application settings — all sourced from env vars with sane defaults."""

    # LLM
    openai_api_key: str = Field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    openai_model: str = Field(default_factory=lambda: os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
    openai_base_url: str | None = Field(
        default_factory=lambda: os.getenv("OPENAI_BASE_URL") or None
    )

    # Web search
    tavily_api_key: str = Field(
        default_factory=lambda: os.getenv("TAVILY_API_KEY", "")
    )

    # Discord
    discord_webhook_url: str = Field(
        default_factory=lambda: os.getenv("DISCORD_WEBHOOK_URL", "")
    )

    # Directories (relative to project root)
    tasks_dir: Path = Field(
        default_factory=lambda: _PROJECT_ROOT / os.getenv("TASKS_DIR", "tasks")
    )
    reports_dir: Path = Field(
        default_factory=lambda: _PROJECT_ROOT / os.getenv("REPORTS_DIR", "reports")
    )
    data_dir: Path = Field(
        default_factory=lambda: _PROJECT_ROOT / os.getenv("DATA_DIR", "data")
    )

    # S3 (for PDF uploads)
    s3_api_url: str = Field(default_factory=lambda: os.getenv("S3_API_URL", ""))
    s3_access_key_id: str = Field(default_factory=lambda: os.getenv("S3_ACCESS_KEY_ID", ""))
    s3_secret_access_key: str = Field(default_factory=lambda: os.getenv("S3_SECRET_ACCESS_KEY", ""))
    s3_bucket_name: str = Field(default_factory=lambda: os.getenv("S3_BUCKET_NAME", ""))
    s3_public_url: str = Field(default_factory=lambda: os.getenv("S3_PUBLIC_URL", ""))

    # Embedding model
    embedding_model: str = Field(
        default_factory=lambda: os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    )

    # arXiv pipeline defaults
    max_papers_per_run: int = Field(
        default_factory=lambda: int(os.getenv("MAX_PAPERS_PER_RUN", "50"))
    )
    citation_depth: int = Field(
        default_factory=lambda: int(os.getenv("CITATION_DEPTH", "1"))
    )
    max_citation_expansions: int = Field(
        default_factory=lambda: int(os.getenv("MAX_CITATION_EXPANSIONS", "10"))
    )

    def ensure_dirs(self) -> None:
        self.tasks_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)


def get_settings() -> Settings:
    """Return a fresh Settings instance."""
    return Settings()
