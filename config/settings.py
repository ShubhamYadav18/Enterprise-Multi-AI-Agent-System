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

    # --- DefectDojo ---
    defectdojo_url: str = Field(default="", description="DefectDojo URL")
    defectdojo_api_key: str = Field(default="", description="DefectDojo API token")
    defectdojo_product: str = Field(default="", description="DefectDojo product identifier (name or ID)")
    defectdojo_engagement: str = Field(default="", description="DefectDojo engagement identifier (name or ID)")

    # --- Paths ---
    documents_dir: Path = Field(default=PROJECT_ROOT / "data" / "documents")
    vectorstore_dir: Path = Field(default=PROJECT_ROOT / "vectorstore")

    # --- Embeddings ---
    embedding_model: str = Field(default="sentence-transformers/all-MiniLM-L6-v2")
    embedding_device: str = Field(default="cpu")

    # --- RAG ---
    chunk_size: int = Field(default=400, description="Document chunk size for splitting")
    chunk_overlap: int = Field(default=50, description="Overlap between chunks")
    retrieval_k: int = Field(default=4, description="Number of documents to retrieve")

    # --- Server ---
    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000)

    # --- Logging ---
    log_level: str = Field(default="INFO")

    # --- Datadog ---
    datadog_enabled: bool = Field(default=False, description="Enable Datadog metrics publishing")
    datadog_site: str = Field(default="datadoghq.com", description="Datadog site (e.g. datadoghq.com, datadoghq.eu)")
    datadog_api_key: str = Field(default="", description="Datadog API key")
    datadog_app_key: str = Field(default="", description="Datadog Application key")
    datadog_metric_prefix: str = Field(default="ai.enterprise", description="Metric name prefix")

    # --- Monitoring Thresholds ---
    groundedness_min: float = Field(default=0.80, description="Min acceptable groundedness score (0-1)")
    hallucination_max: float = Field(default=0.20, description="Max acceptable hallucination score (0-1)")
    latency_max_ms: float = Field(default=20000.0, description="Max acceptable latency in ms")
    total_cost_max: float = Field(default=0.50, description="Max acceptable total cost per query (USD)")
    token_limit: int = Field(default=50000, description="Max acceptable tokens per query")

    # --- Incident Provider ---
    incident_provider: str = Field(default="noop", description="Incident provider: noop | servicenow | jira | pagerduty")

    # --- ServiceNow ---
    servicenow_enabled: bool = Field(default=False, description="Enable ServiceNow incident creation")
    servicenow_instance_url: str = Field(default="", description="ServiceNow instance URL (https://xxx.service-now.com)")
    servicenow_username: str = Field(default="", description="ServiceNow username")
    servicenow_password: str = Field(default="", description="ServiceNow password")
    servicenow_table: str = Field(default="incident", description="ServiceNow table name")
    servicenow_category: str = Field(default="AI", description="ServiceNow incident category")
    servicenow_subcategory: str = Field(default="LLM", description="ServiceNow incident subcategory")
    servicenow_assignment_group: str = Field(default="", description="ServiceNow assignment group")
    servicenow_impact: int = Field(default=2, description="ServiceNow impact (1=High, 2=Medium, 3=Low)")
    servicenow_urgency: int = Field(default=2, description="ServiceNow urgency (1=High, 2=Medium, 3=Low)")

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
