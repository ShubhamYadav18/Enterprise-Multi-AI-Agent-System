"""
Structured logging for the Enterprise Multi-Agent AI System.

Provides consistent, structured log output for:
- Agent start/finish with timing
- Routing decisions
- Tool invocations
- LLM calls
- Retriever queries
- Errors with context
"""

from __future__ import annotations

import logging
import sys
import json
from datetime import datetime, timezone
from functools import lru_cache
from typing import Any


class StructuredFormatter(logging.Formatter):
    """JSON-structured log formatter for machine-readable output."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Include extra fields if attached to the record
        for key in ("session_id", "agent", "tool", "query_type",
                     "duration_ms", "token_count", "event_type"):
            value = getattr(record, key, None)
            if value is not None:
                log_entry[key] = value

        # Include exception info if present
        if record.exc_info and record.exc_info[0] is not None:
            log_entry["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_entry, default=str)


@lru_cache(maxsize=16)
def get_logger(name: str) -> logging.Logger:
    """Get a named logger with structured formatting.

    Args:
        name: Logger name (typically __name__ of the calling module).

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(name)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(StructuredFormatter())
        logger.addHandler(handler)

    # Avoid duplicate logs from root logger
    logger.propagate = False

    return logger


def log_agent_event(
    logger: logging.Logger,
    event_type: str,
    agent: str,
    session_id: str = "",
    **kwargs: Any,
) -> None:
    """Log a structured agent lifecycle event.

    Args:
        logger: Logger instance.
        event_type: Event type (e.g., "agent_start", "agent_finish", "tool_call").
        agent: Agent name.
        session_id: Session identifier.
        **kwargs: Additional context fields.
    """
    extra = {
        "event_type": event_type,
        "agent": agent,
        "session_id": session_id,
        **kwargs,
    }
    logger.info(
        f"{event_type}: {agent}",
        extra=extra,
    )


def configure_root_logging(level: str = "INFO") -> None:
    """Configure the root logger level for the application.

    Args:
        level: Logging level string (DEBUG, INFO, WARNING, ERROR).
    """
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logging.getLogger().setLevel(numeric_level)

    # Also set our module loggers
    for name in list(logging.Logger.manager.loggerDict.keys()):
        if name.startswith("enterprise_multi_agent") or name.startswith("agents") \
                or name.startswith("graph") or name.startswith("tools"):
            logging.getLogger(name).setLevel(numeric_level)
