"""
Calculator Agent.

Performs mathematical calculations using dedicated tools.
Shows step-by-step work and validates results.
"""

from __future__ import annotations

import time
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langsmith import traceable

from prompts.calculator_agent import CALCULATOR_SYSTEM_PROMPT, CALCULATOR_HUMAN_PROMPT
from tools.calculator_tools import CALCULATOR_TOOLS
from utils.logger import get_logger, log_agent_event
from utils.models import AgentOutput, ToolCallRecord

logger = get_logger(__name__)





def run_calculator_agent(
    query: str,
    config: RunnableConfig | None = None,
) -> AgentOutput:
    """Run the calculator agent to perform mathematical computations.

    Uses an agentic loop: the LLM decides which calculator tools to invoke,
    executes them, and synthesizes the results.

    Args:
        query: The calculation request.
        config: LangSmith tracing config for session correlation.

    Returns:
        AgentOutput with calculation results and tool call records.
    """
    session_id = (config or {}).get("metadata", {}).get("session_id", "")
    log_agent_event(logger, "agent_start", "calculator_agent", session_id)
    start_time = time.time()

    try:
        from utils.llm import create_llm
        llm = create_llm(max_tokens=2048, temperature=0.0).bind_tools(CALCULATOR_TOOLS)
        tool_map = {t.name: t for t in CALCULATOR_TOOLS}
        tool_calls_record: list[ToolCallRecord] = []

        messages = [
            SystemMessage(content=CALCULATOR_SYSTEM_PROMPT),
            HumanMessage(content=CALCULATOR_HUMAN_PROMPT.format(query=query)),
        ]

        # Agentic tool-calling loop (max 5 iterations to prevent infinite loops)
        for iteration in range(5):
            response = llm.invoke(messages, config=config)
            messages.append(response)

            # Check for tool calls
            if not response.tool_calls:
                break

            # Execute each tool call
            from langchain_core.messages import ToolMessage

            for tool_call in response.tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]
                tool_id = tool_call["id"]

                log_agent_event(
                    logger, "tool_call", "calculator_agent", session_id,
                    tool=tool_name,
                )

                tool_start = time.time()

                if tool_name in tool_map:
                    tool_result = tool_map[tool_name].invoke(tool_args, config=config)
                else:
                    tool_result = f"Unknown tool: {tool_name}"

                tool_duration = (time.time() - tool_start) * 1000

                tool_calls_record.append(ToolCallRecord(
                    tool_name=tool_name,
                    tool_input=tool_args,
                    tool_output=tool_result,
                    duration_ms=tool_duration,
                ))

                messages.append(ToolMessage(
                    content=str(tool_result),
                    tool_call_id=tool_id,
                ))

        # Final response is the last assistant message without tool calls
        final_content = response.content if response.content else "Calculation completed."

        duration_ms = (time.time() - start_time) * 1000

        log_agent_event(
            logger, "agent_finish", "calculator_agent", session_id,
            duration_ms=duration_ms,
            tool_count=len(tool_calls_record),
        )

        return AgentOutput(
            agent_name="calculator_agent",
            output=final_content,
            confidence=0.95,
            tool_calls=tool_calls_record,
            metadata={
                "tool_count": len(tool_calls_record),
                "iterations": iteration + 1,
                "total_duration_ms": duration_ms,
            },
        )

    except Exception as e:
        logger.error(f"Calculator agent error: {e}", exc_info=True)
        return AgentOutput(
            agent_name="calculator_agent",
            output="",
            error=str(e),
            metadata={"total_duration_ms": (time.time() - start_time) * 1000},
        )
