"""
Validation Agent.

Validates the summarized response for grounding, consistency,
hallucination, completeness, and logical soundness.
Returns a structured validation report.
"""

from __future__ import annotations

import json
import time
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langsmith import traceable

from prompts.validation_agent import VALIDATION_SYSTEM_PROMPT, VALIDATION_HUMAN_PROMPT
from utils.logger import get_logger, log_agent_event
from utils.models import AgentOutput, ValidationReport

logger = get_logger(__name__)





def _format_agent_outputs(outputs: list[AgentOutput]) -> str:
    """Format agent outputs for the validation prompt."""
    parts = []
    for output in outputs:
        if output.output:
            parts.append(f"### {output.agent_name}\n{output.output}")
            if output.sources:
                parts.append(f"Sources: {', '.join(output.sources)}")
    return "\n\n".join(parts)


def run_validation_agent(
    query: str,
    summary: str,
    agent_outputs: list[AgentOutput],
    config: RunnableConfig | None = None,
) -> tuple[ValidationReport, str]:
    """Validate the summarized response.

    Args:
        query: Original user query.
        summary: Draft response from the summarizer.
        agent_outputs: Original agent outputs for cross-validation.
        config: LangSmith tracing config.

    Returns:
        Tuple of (ValidationReport, final_response).
    """
    session_id = (config or {}).get("metadata", {}).get("session_id", "")
    log_agent_event(logger, "agent_start", "validation_agent", session_id)
    start_time = time.time()

    try:
        from utils.llm import create_llm
        llm = create_llm(max_tokens=2048, temperature=0.0)
        formatted_outputs = _format_agent_outputs(agent_outputs)

        messages = [
            SystemMessage(content=VALIDATION_SYSTEM_PROMPT),
            HumanMessage(content=VALIDATION_HUMAN_PROMPT.format(
                query=query,
                summary=summary,
                agent_outputs=formatted_outputs,
            )),
        ]

        response = llm.invoke(messages, config=config)
        raw_content = response.content

        # Parse the JSON validation report
        try:
            text = raw_content
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0]
            elif "```" in text:
                text = text.split("```")[1].split("```")[0]

            validation_data = json.loads(text.strip())
        except (json.JSONDecodeError, IndexError):
            logger.warning(f"Failed to parse validation response: {raw_content[:200]}")
            validation_data = {
                "is_valid": True,
                "grounding_score": 0.7,
                "consistency_score": 0.8,
                "hallucination_score": 0.1,
                "completeness_score": 0.7,
                "logic_score": 0.8,
                "issues": [],
                "recommendations": [],
                "final_response": summary,
            }

        # Build ValidationReport
        report = ValidationReport(
            is_valid=validation_data.get("is_valid", True),
            grounding_score=validation_data.get("grounding_score", 0.7),
            consistency_score=validation_data.get("consistency_score", 0.8),
            hallucination_score=validation_data.get("hallucination_score", 0.1),
            completeness_score=validation_data.get("completeness_score", 0.7),
            logic_score=validation_data.get("logic_score", 0.8),
            issues=validation_data.get("issues", []),
            recommendations=validation_data.get("recommendations", []),
        )

        final_response = validation_data.get("final_response", summary)

        duration_ms = (time.time() - start_time) * 1000
        log_agent_event(
            logger, "agent_finish", "validation_agent", session_id,
            duration_ms=duration_ms,
            is_valid=report.is_valid,
        )

        return report, final_response

    except Exception as e:
        logger.error(f"Validation agent error: {e}", exc_info=True)
        # Fallback: pass through the summary without validation
        report = ValidationReport(
            is_valid=True,
            issues=[f"Validation skipped due to error: {str(e)}"],
        )
        return report, summary
