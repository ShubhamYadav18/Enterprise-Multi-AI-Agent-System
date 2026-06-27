"""
Session management for LangSmith trace correlation.

Each user request gets a unique session. All traces for that request
are grouped under the same session in LangSmith, enabling full
execution tree inspection.
"""

from __future__ import annotations

import uuid
import time
from datetime import datetime, timezone
from typing import Any, Optional

from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field

from config.settings import get_settings
from utils.logger import get_logger

logger = get_logger(__name__)


class TraceSession(BaseModel):
    """Represents a traced execution session."""

    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    conversation_id: str = Field(default="")
    user_id: str = Field(default="demo-user")
    query: str = Field(default="")
    query_type: str = Field(default="")
    start_time: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    end_time: Optional[datetime] = Field(default=None)
    agents_invoked: list[str] = Field(default_factory=list)
    agents_skipped: list[str] = Field(default_factory=list)
    parallel_groups: list[list[str]] = Field(default_factory=list)
    tool_count: int = Field(default=0)
    total_tokens: int = Field(default=0)
    total_duration_ms: float = Field(default=0.0)
    langsmith_url: Optional[str] = Field(default=None)

    def get_runnable_config(self) -> RunnableConfig:
        """Create a RunnableConfig for LangSmith trace correlation.

        Returns:
            RunnableConfig with session metadata that propagates
            to all downstream LangChain/LangGraph calls.
        """
        settings = get_settings()

        metadata: dict[str, Any] = {
            "session_id": self.session_id,
            "conversation_id": self.conversation_id,
            "user_id": self.user_id,
            "query": self.query[:200],  # Truncate for metadata
            "query_type": self.query_type,
            "timestamp": self.start_time.isoformat(),
            "environment": "demo",
            "project": settings.langchain_project,
        }

        config: RunnableConfig = {
            "metadata": metadata,
            "tags": [
                f"session:{self.session_id}",
                f"user:{self.user_id}",
                "enterprise-multi-agent",
            ],
            "run_name": "User Query",
            "configurable": {
                "session_id": self.session_id,
                "thread_id": self.session_id,
            },
        }

        return config

    def finalize(
        self,
        query_type: str = "",
        agents_invoked: list[str] | None = None,
        agents_skipped: list[str] | None = None,
        parallel_groups: list[list[str]] | None = None,
        tool_count: int = 0,
        total_tokens: int = 0,
    ) -> None:
        """Finalize the session with execution results.

        Args:
            query_type: Classified query type.
            agents_invoked: List of agents that ran.
            agents_skipped: List of agents that were skipped.
            parallel_groups: Groups of agents that ran in parallel.
            tool_count: Total number of tool invocations.
            total_tokens: Total tokens used.
        """
        self.end_time = datetime.now(timezone.utc)
        self.query_type = query_type or self.query_type
        self.agents_invoked = agents_invoked or self.agents_invoked
        self.agents_skipped = agents_skipped or self.agents_skipped
        self.parallel_groups = parallel_groups or self.parallel_groups
        self.tool_count = tool_count
        self.total_tokens = total_tokens

        if self.start_time and self.end_time:
            delta = (self.end_time - self.start_time).total_seconds() * 1000
            self.total_duration_ms = delta

        # Build LangSmith URL
        settings = get_settings()
        if settings.langchain_api_key:
            project_name = settings.langchain_project.replace(" ", "+")
            self.langsmith_url = (
                f"https://smith.langchain.com/projects/"
                f"?filter=metadata.session_id%3D%22{self.session_id}%22"
            )

        logger.info(
            f"Session {self.session_id[:8]} finalized: "
            f"type={self.query_type}, agents={self.agents_invoked}, "
            f"duration={self.total_duration_ms:.0f}ms"
        )


def create_session(
    query: str,
    user_id: str = "demo-user",
    conversation_id: str = "",
) -> TraceSession:
    """Create a new trace session for a user request.

    Args:
        query: The user's query text.
        user_id: User identifier.
        conversation_id: Conversation identifier.

    Returns:
        TraceSession with a unique session_id and RunnableConfig.
    """
    session = TraceSession(
        query=query,
        user_id=user_id,
        conversation_id=conversation_id or str(uuid.uuid4()),
    )

    logger.info(
        f"Created session {session.session_id[:8]} for query: '{query[:50]}...'"
    )

    return session
