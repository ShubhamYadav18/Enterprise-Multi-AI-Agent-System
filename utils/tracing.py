"""
LangSmith tracing helpers.

Provides utilities to create RunnableConfig objects with
session-correlated metadata so all traces for a single user
request appear in one execution tree in LangSmith.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from langchain_core.runnables import RunnableConfig


def create_session_config(
    query: str,
    session_id: str | None = None,
    conversation_id: str = "",
    user_id: str = "demo-user",
    query_type: str = "",
    extra_metadata: dict[str, Any] | None = None,
) -> RunnableConfig:
    """Create a RunnableConfig with session-correlated metadata.

    All downstream LangChain/LangGraph calls that receive this config
    will have their traces grouped under the same session in LangSmith.

    Args:
        query: The user's query text.
        session_id: Unique session ID. Auto-generated if not provided.
        conversation_id: Conversation identifier for multi-turn tracking.
        user_id: User identifier (mock for demo).
        query_type: Classified query type (filled after orchestrator runs).
        extra_metadata: Additional metadata to attach to traces.

    Returns:
        RunnableConfig with metadata and tags for LangSmith correlation.
    """
    if session_id is None:
        session_id = str(uuid.uuid4())

    metadata: dict[str, Any] = {
        "session_id": session_id,
        "conversation_id": conversation_id or str(uuid.uuid4()),
        "user_id": user_id,
        "query": query,
        "query_type": query_type,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "environment": "demo",
    }

    if extra_metadata:
        metadata.update(extra_metadata)

    config: RunnableConfig = {
        "metadata": metadata,
        "tags": [
            f"session:{session_id}",
            f"user:{user_id}",
        ],
        "run_name": f"Session {session_id[:8]}",
    }

    # Add configurable for session thread tracking
    config["configurable"] = {
        "session_id": session_id,
        "thread_id": session_id,
    }

    return config


def update_config_metadata(
    config: RunnableConfig,
    **kwargs: Any,
) -> RunnableConfig:
    """Update metadata in an existing RunnableConfig.

    Args:
        config: Existing config to update.
        **kwargs: Key-value pairs to add/update in metadata.

    Returns:
        Updated RunnableConfig (mutated in-place and returned).
    """
    if "metadata" not in config:
        config["metadata"] = {}
    config["metadata"].update(kwargs)
    return config
