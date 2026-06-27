"""
LLM factory for creating configured Anthropic chat model instances.

Centralizes LLM creation so all agents use consistent configuration,
including proper handling of Azure-hosted Anthropic endpoints.
"""

from __future__ import annotations

from typing import Any

from langchain_anthropic import ChatAnthropic

from config.settings import get_settings
from utils.logger import get_logger

logger = get_logger(__name__)


def create_llm(
    max_tokens: int = 2048,
    temperature: float = 0.1,
    **kwargs: Any,
) -> ChatAnthropic:
    """Create a configured ChatAnthropic instance.

    Handles both standard Anthropic API and Azure-hosted endpoints.
    Azure endpoints use the `api-key` header instead of `x-api-key`.

    Args:
        max_tokens: Maximum tokens in the response.
        temperature: Sampling temperature (0.0 = deterministic).
        **kwargs: Additional kwargs passed to ChatAnthropic.

    Returns:
        Configured ChatAnthropic instance.
    """
    settings = get_settings()

    llm_kwargs: dict[str, Any] = {
        "model": settings.anthropic_model,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "api_key": settings.anthropic_api_key,
        **kwargs,
    }

    if settings.has_anthropic_url:
        # Strip trailing /v1/messages, /messages, or /v1 because the SDK automatically appends /v1/messages
        base_url = settings.anthropic_url.rstrip("/")
        if base_url.endswith("/v1/messages"):
            base_url = base_url[:-12]
        elif base_url.endswith("/messages"):
            base_url = base_url[:-9]
        elif base_url.endswith("/v1"):
            base_url = base_url[:-3]

        llm_kwargs["base_url"] = base_url
        # Azure-hosted Anthropic endpoints use `api-key` header
        llm_kwargs["default_headers"] = {
            "api-key": settings.anthropic_api_key,
        }
        logger.info(
            f"Using Azure Anthropic endpoint: {base_url} (original: {settings.anthropic_url}) "
            f"(model: {settings.anthropic_model})"
        )

    return ChatAnthropic(**llm_kwargs)
