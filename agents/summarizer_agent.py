"""
Summarizer Agent.

Merges outputs from multiple specialized agents into one
coherent, well-structured response.
"""

from __future__ import annotations

import time
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langsmith import traceable

from prompts.summarizer_agent import SUMMARIZER_SYSTEM_PROMPT, SUMMARIZER_HUMAN_PROMPT
from utils.logger import get_logger, log_agent_event
from utils.models import AgentOutput

logger = get_logger(__name__)





def _format_agent_outputs(outputs: list[AgentOutput]) -> str:
    """Format multiple agent outputs for the summarizer prompt."""
    parts = []
    for output in outputs:
        if output.error:
            parts.append(f"### {output.agent_name} (ERROR)\n{output.error}")
        else:
            parts.append(f"### {output.agent_name}\n{output.output}")

            if output.sources:
                parts.append(f"Sources: {', '.join(output.sources)}")

            if output.tool_calls:
                tool_summary = ", ".join(
                    f"{tc.tool_name}: {tc.tool_output}"
                    for tc in output.tool_calls
                )
                parts.append(f"Tool Results: {tool_summary}")

        parts.append("")  # Blank line separator

    return "\n".join(parts)


def run_summarizer_agent(
    query: str,
    agent_outputs: list[AgentOutput],
    config: RunnableConfig | None = None,
) -> str:
    """Merge multiple agent outputs into a single coherent response.

    Args:
        query: Original user query.
        agent_outputs: List of outputs from specialized agents.
        config: LangSmith tracing config.

    Returns:
        Unified summary string.
    """
    session_id = (config or {}).get("metadata", {}).get("session_id", "")
    log_agent_event(logger, "agent_start", "summarizer_agent", session_id)
    start_time = time.time()

    try:
        from utils.llm import create_llm
        llm = create_llm(max_tokens=3000, temperature=0.3)
        formatted_outputs = _format_agent_outputs(agent_outputs)

        messages = [
            SystemMessage(content=SUMMARIZER_SYSTEM_PROMPT),
            HumanMessage(content=SUMMARIZER_HUMAN_PROMPT.format(
                query=query,
                agent_outputs=formatted_outputs,
            )),
        ]

        response = llm.invoke(messages, config=config)
        summary = response.content

        duration_ms = (time.time() - start_time) * 1000
        log_agent_event(
            logger, "agent_finish", "summarizer_agent", session_id,
            duration_ms=duration_ms,
            agents_merged=len(agent_outputs),
        )

        return summary

    except Exception as e:
        logger.error(f"Summarizer agent error: {e}", exc_info=True)
        # Fallback: concatenate raw outputs
        fallback = "\n\n".join(
            f"**{o.agent_name}**: {o.output}" for o in agent_outputs if o.output
        )
        return fallback or f"Error generating summary: {str(e)}"
