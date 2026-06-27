"""
Central configuration using Pydantic Settings.

All configuration is read from environment variables / .env file.
Never hardcode secrets — use .env.example as a template.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings
from pydantic import Field


# Project root = parent of config/
PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # --- Anthropic LLM ---
    anthropic_api_key: str = Field(default="", description="Anthropic API key")
    anthropic_url: str = Field(default="", description="Custom Anthropic API base URL (optional)")
    anthropic_model: str = Field(default="claude-sonnet-4-20250514", description="Anthropic model name")

    # --- LangSmith ---
    langchain_api_key: str = Field(default="", description="LangSmith API key")
    langchain_tracing_v2: bool = Field(default=True, description="Enable LangSmith tracing")
    langchain_project: str = Field(default="Enterprise Multi Agent Demo", description="LangSmith project name")

    # --- Optional Search ---
    tavily_api_key: str = Field(default="", description="Tavily API key (optional, uses mock if empty)")

    # --- Paths ---
    documents_dir: Path = Field(default=PROJECT_ROOT / "data" / "documents")
    vectorstore_dir: Path = Field(default=PROJECT_ROOT / "vectorstore")

    # --- Embeddings ---
    embedding_model: str = Field(default="sentence-transformers/all-MiniLM-L6-v2")
    embedding_device: str = Field(default="cpu")

    # --- RAG ---
    chunk_size: int = Field(default=500, description="Document chunk size for splitting")
    chunk_overlap: int = Field(default=50, description="Overlap between chunks")
    retrieval_k: int = Field(default=4, description="Number of documents to retrieve")

    # --- Server ---
    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000)

    # --- Logging ---
    log_level: str = Field(default="INFO")

    model_config = {
        "env_file": str(PROJECT_ROOT / ".env"),
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }

    @property
    def has_tavily(self) -> bool:
        """Check if Tavily search is configured."""
        return bool(self.tavily_api_key)

    @property
    def has_anthropic_url(self) -> bool:
        """Check if a custom Anthropic URL is configured."""
        return bool(self.anthropic_url)

    def configure_langsmith_env(self) -> None:
        """Push LangSmith settings into environment variables.

        LangSmith SDK reads from env vars directly, so we ensure
        our Pydantic-loaded values are available there.
        """
        if self.langchain_api_key:
            os.environ["LANGCHAIN_API_KEY"] = self.langchain_api_key
        os.environ["LANGCHAIN_TRACING_V2"] = str(self.langchain_tracing_v2).lower()
        os.environ["LANGCHAIN_PROJECT"] = self.langchain_project


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Get cached application settings (singleton)."""
    settings = Settings()
    settings.configure_langsmith_env()
    return settings
