"""
Custom LangSmith callback handler for structured agent tracing.

Captures detailed lifecycle events and attaches them to the
session trace tree for full observability in LangSmith.
"""

from __future__ import annotations

import time
from typing import Any, Optional
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult

from utils.logger import get_logger

logger = get_logger(__name__)


class AgentTraceCallback(BaseCallbackHandler):
    """Callback handler that logs structured agent lifecycle events.

    Captures:
    - LLM call start/end with token counts
    - Tool invocations with timing
    - Retriever queries
    - Chain start/end
    - Errors
    """

    def __init__(self, session_id: str = "") -> None:
        super().__init__()
        self.session_id = session_id
        self._timers: dict[str, float] = {}
        self._run_names: dict[str, str] = {}
        self.node_durations: dict[str, float] = {}
        self.total_tokens: int = 0
        self.prompt_tokens: int = 0
        self.completion_tokens: int = 0
        self.tool_calls: int = 0
        self.llm_calls: int = 0
        self.retriever_calls: int = 0

    # --- LLM Events ---

    def on_llm_start(
        self,
        serialized: dict[str, Any],
        prompts: list[str],
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        """Called when an LLM call starts."""
        self._timers[str(run_id)] = time.time()
        self.llm_calls += 1
        model = serialized.get("kwargs", {}).get("model", "unknown")
        logger.info(
            f"LLM call started (model={model})",
            extra={"session_id": self.session_id, "event_type": "llm_start"},
        )

    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        """Called when an LLM call finishes."""
        duration = (time.time() - self._timers.pop(str(run_id), time.time())) * 1000

        # Extract token usage if available
        token_usage = {}
        if response.llm_output:
            token_usage = response.llm_output.get("usage", {})
            if token_usage:
                total = token_usage.get("total_tokens", 0)
                if total:
                    self.total_tokens += total
                prompt = token_usage.get("prompt_tokens", token_usage.get("input_tokens", 0))
                completion = token_usage.get("completion_tokens", token_usage.get("output_tokens", 0))
                self.prompt_tokens += prompt
                self.completion_tokens += completion

        logger.info(
            f"LLM call completed ({duration:.0f}ms)",
            extra={
                "session_id": self.session_id,
                "event_type": "llm_end",
                "duration_ms": duration,
                "token_count": token_usage.get("total_tokens", 0),
            },
        )


    def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        """Called when an LLM call errors."""
        logger.error(
            f"LLM call failed: {error}",
            extra={"session_id": self.session_id, "event_type": "llm_error"},
        )

    # --- Tool Events ---

    def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        """Called when a tool is invoked."""
        self._timers[str(run_id)] = time.time()
        self.tool_calls += 1
        tool_name = serialized.get("name", "unknown")
        logger.info(
            f"Tool invoked: {tool_name}",
            extra={
                "session_id": self.session_id,
                "event_type": "tool_start",
                "tool": tool_name,
            },
        )

    def on_tool_end(
        self,
        output: str,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        """Called when a tool finishes."""
        duration = (time.time() - self._timers.pop(str(run_id), time.time())) * 1000
        logger.info(
            f"Tool completed ({duration:.0f}ms)",
            extra={
                "session_id": self.session_id,
                "event_type": "tool_end",
                "duration_ms": duration,
            },
        )

    def on_tool_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        """Called when a tool errors."""
        logger.error(
            f"Tool failed: {error}",
            extra={"session_id": self.session_id, "event_type": "tool_error"},
        )

    # --- Retriever Events ---

    def on_retriever_start(
        self,
        serialized: dict[str, Any],
        query: str,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        """Called when a retriever is invoked."""
        self._timers[str(run_id)] = time.time()
        self.retriever_calls += 1
        logger.info(
            f"Retriever query: '{query[:80]}'",
            extra={
                "session_id": self.session_id,
                "event_type": "retriever_start",
            },
        )

    def on_retriever_end(
        self,
        documents: Any,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        """Called when a retriever finishes."""
        duration = (time.time() - self._timers.pop(str(run_id), time.time())) * 1000
        doc_count = len(documents) if documents else 0
        logger.info(
            f"Retriever returned {doc_count} documents ({duration:.0f}ms)",
            extra={
                "session_id": self.session_id,
                "event_type": "retriever_end",
                "duration_ms": duration,
            },
        )

    # --- Chain Events ---

    def on_chain_start(
        self,
        serialized: dict[str, Any],
        inputs: dict[str, Any],
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        """Called when a chain/graph node starts."""
        self._timers[str(run_id)] = time.time()
        name = serialized.get("name", kwargs.get("name", "unknown"))
        self._run_names[str(run_id)] = name
        logger.debug(
            f"Chain started: {name}",
            extra={"session_id": self.session_id, "event_type": "chain_start"},
        )

    def on_chain_end(
        self,
        outputs: dict[str, Any],
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        """Called when a chain/graph node finishes."""
        duration = (time.time() - self._timers.pop(str(run_id), time.time())) * 1000
        name = self._run_names.pop(str(run_id), "unknown")
        self.node_durations[name] = duration
        logger.debug(
            f"Chain completed ({duration:.0f}ms)",
            extra={
                "session_id": self.session_id,
                "event_type": "chain_end",
                "duration_ms": duration,
            },
        )

    def get_summary(self) -> dict[str, Any]:
        """Get a summary of all traced events.

        Returns:
            Dict with event counts and total tokens.
        """
        return {
            "llm_calls": self.llm_calls,
            "tool_calls": self.tool_calls,
            "retriever_calls": self.retriever_calls,
            "total_tokens": self.total_tokens,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
        }
