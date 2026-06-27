"""
Research Agent.

Uses search tools to find external information, industry benchmarks,
and market data. Structured for easy provider replacement.
"""

from __future__ import annotations

import time
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langsmith import traceable

from prompts.research_agent import RESEARCH_SYSTEM_PROMPT, RESEARCH_HUMAN_PROMPT
from tools.search_tools import search as search_tool, get_search_provider
from utils.logger import get_logger, log_agent_event
from utils.models import AgentOutput, ToolCallRecord

logger = get_logger(__name__)





def run_research_agent(
    query: str,
    config: RunnableConfig | None = None,
) -> AgentOutput:
    """Run the research agent to find external information.

    Args:
        query: Research query to investigate.
        config: LangSmith tracing config for session correlation.

    Returns:
        AgentOutput with research findings, sources, and data points.
    """
    session_id = (config or {}).get("metadata", {}).get("session_id", "")
    log_agent_event(logger, "agent_start", "research_agent", session_id)
    start_time = time.time()

    try:
        # Step 1: Search for relevant information
        search_start = time.time()
        search_results_text = search_tool.invoke({"query": query, "max_results": 5}, config=config)
        search_duration = (time.time() - search_start) * 1000

        tool_calls = [
            ToolCallRecord(
                tool_name="search",
                tool_input={"query": query, "max_results": 5},
                tool_output=search_results_text[:500],  # Truncate for metadata
                duration_ms=search_duration,
            )
        ]

        # Step 2: Synthesize findings via LLM
        from utils.llm import create_llm
        llm = create_llm(max_tokens=2048, temperature=0.2)
        messages = [
            SystemMessage(content=RESEARCH_SYSTEM_PROMPT),
            HumanMessage(content=(
                f"{RESEARCH_HUMAN_PROMPT.format(query=query)}\n\n"
                f"## Search Results\n\n{search_results_text}"
            )),
        ]

        response = llm.invoke(messages, config=config)
        content = response.content

        # Extract confidence
        confidence = 0.7
        if "HIGH" in content.upper().split("CONFIDENCE")[-1][:20] if "CONFIDENCE" in content.upper() else "":
            confidence = 0.9
        elif "LOW" in content.upper().split("CONFIDENCE")[-1][:20] if "CONFIDENCE" in content.upper() else "":
            confidence = 0.4

        duration_ms = (time.time() - start_time) * 1000

        log_agent_event(
            logger, "agent_finish", "research_agent", session_id,
            duration_ms=duration_ms,
        )

        return AgentOutput(
            agent_name="research_agent",
            output=content,
            confidence=confidence,
            sources=["external_search"],
            tool_calls=tool_calls,
            metadata={
                "search_duration_ms": search_duration,
                "total_duration_ms": duration_ms,
                "provider": type(get_search_provider()).__name__,
            },
        )

    except Exception as e:
        logger.error(f"Research agent error: {e}", exc_info=True)
        return AgentOutput(
            agent_name="research_agent",
            output="",
            error=str(e),
            metadata={"total_duration_ms": (time.time() - start_time) * 1000},
        )
