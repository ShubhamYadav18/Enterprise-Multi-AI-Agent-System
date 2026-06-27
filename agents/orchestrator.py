"""
Orchestrator Agent.

Analyzes user intent and decides which specialized agents should handle
the request. Never answers the user's question directly.
"""

from __future__ import annotations

import json
import time
from typing import Any


from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langsmith import traceable


from prompts.orchestrator import ORCHESTRATOR_SYSTEM_PROMPT, ORCHESTRATOR_HUMAN_PROMPT
from utils.logger import get_logger, log_agent_event

logger = get_logger(__name__)





def run_orchestrator(
    query: str,
    config: RunnableConfig | None = None,
) -> dict[str, Any]:
    """Run the orchestrator to determine agent routing.

    Args:
        query: User query to analyze.
        config: LangSmith tracing config for session correlation.

    Returns:
        Dict with keys: query_type, agents_to_invoke, reasoning,
        parallel_groups, refined_query.
    """
    session_id = (config or {}).get("metadata", {}).get("session_id", "")
    log_agent_event(logger, "agent_start", "orchestrator", session_id)
    start_time = time.time()

    from utils.llm import create_llm
    llm = create_llm(max_tokens=1024, temperature=0.0)

    messages = [
        SystemMessage(content=ORCHESTRATOR_SYSTEM_PROMPT),
        HumanMessage(content=ORCHESTRATOR_HUMAN_PROMPT.format(query=query)),
    ]

    response = llm.invoke(messages, config=config)
    raw_content = response.content

    # Parse the JSON response
    try:
        # Handle markdown code blocks if present
        text = raw_content
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0]
        elif "```" in text:
            text = text.split("```")[1].split("```")[0]

        routing = json.loads(text.strip())
    except (json.JSONDecodeError, IndexError):
        logger.warning(f"Failed to parse orchestrator response as JSON: {raw_content[:200]}")
        # Fallback: route everything through RAG
        routing = {
            "query_type": "rag_only",
            "agents_to_invoke": ["rag_agent"],
            "reasoning": "Failed to parse routing — defaulting to RAG agent.",
            "parallel_groups": [["rag_agent"]],
            "refined_query": query,
        }

    # Ensure required fields
    routing.setdefault("query_type", "rag_only")
    routing.setdefault("agents_to_invoke", ["rag_agent"])
    routing.setdefault("reasoning", "")
    routing.setdefault("parallel_groups", [routing["agents_to_invoke"]])
    routing.setdefault("refined_query", query)

    duration_ms = (time.time() - start_time) * 1000
    log_agent_event(
        logger, "agent_finish", "orchestrator", session_id,
        duration_ms=duration_ms,
        query_type=routing["query_type"],
        agents_to_invoke=routing["agents_to_invoke"],
    )

    return routing
