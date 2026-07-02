"""
MetricBuilder — assembles AIMetricEvent from ChatResponse + evaluation results.

Reads application metadata (git revision, model name, version) from the
AI Manifest where available, falling back to settings and defaults.
"""

from __future__ import annotations

import hashlib
from typing import Any, Optional

from monitoring.models import AgentMetric, AIMetricEvent
from utils.logger import get_logger

logger = get_logger(__name__)

# Evaluator key → numeric score (or "N/A" / "Not Invoked")
_NUMERIC_SCORE_SENTINEL = ("N/A", "Not Invoked", None)


def _to_float(value: Any) -> Optional[float]:
    """Convert an evaluator score to float, returning None for non-numeric values."""
    if value in _NUMERIC_SCORE_SENTINEL:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _query_hash(query: str) -> str:
    """Return a short SHA-256 prefix of the query for tagging."""
    return hashlib.sha256(query.encode("utf-8")).hexdigest()[:16]


def _get_app_metadata() -> dict[str, str]:
    """
    Read application version and git revision from the AI manifest.
    Falls back to defaults if the manifest is unavailable.
    """
    try:
        from metadata.ai_manifest_generator import load_manifest
        manifest = load_manifest()
        if manifest:
            return {
                "application_version": manifest.application.version,
                "git_revision": (manifest.application.git_commit_hash or "unknown")[:12],
                "model_name": manifest.llm.model_name or "unknown",
            }
    except Exception:
        pass

    try:
        from config.settings import get_settings
        s = get_settings()
        return {
            "application_version": "0.1.0",
            "git_revision": "unknown",
            "model_name": s.anthropic_model or "unknown",
        }
    except Exception:
        return {"application_version": "0.1.0", "git_revision": "unknown", "model_name": "unknown"}


class MetricBuilder:
    """
    Builds AIMetricEvent from a completed chat response and evaluation results.

    Usage::

        event = MetricBuilder.build(
            session_id=session_id,
            query=query,
            latency_ms=latency_ms,
            total_tokens=total_tokens,
            cost=cost,
            workflow_evals=workflow_evals,
            agent_evals=agent_evals,
            agents_invoked=agents_invoked,
            conversation_id=conversation_id,
            root_run_id=root_run_id,
            langsmith_url=langsmith_url,
        )
    """

    @staticmethod
    def build(
        session_id: str,
        query: str,
        latency_ms: float,
        total_tokens: int,
        cost: float,
        workflow_evals: dict[str, Any],
        agent_evals: dict[str, Any],
        agents_invoked: list[str],
        conversation_id: str = "",
        root_run_id: str = "",
        langsmith_url: Optional[str] = None,
        input_tokens: int = 0,
        output_tokens: int = 0,
    ) -> AIMetricEvent:
        """Assemble a complete AIMetricEvent from all available data."""

        app_meta = _get_app_metadata()

        # Build per-agent metrics
        agent_metrics: list[AgentMetric] = []

        # Known agent name mappings (display name → internal name fragment)
        _agent_map = {
            "RAG Agent": "rag",
            "Calculator Agent": "calculator",
            "Research Agent": "research",
            "Summarizer": "summarizer",
            "Validator": "validator",
        }

        for display_name, evals in agent_evals.items():
            if isinstance(evals, dict) and evals.get("status") == "Not Invoked":
                agent_metrics.append(
                    AgentMetric(
                        agent_name=display_name,
                        execution_status="skipped",
                    )
                )
                continue

            if not isinstance(evals, dict):
                continue

            agent_metrics.append(
                AgentMetric(
                    agent_name=display_name,
                    execution_status="executed",
                    groundedness=_to_float(evals.get("groundedness")),
                    hallucination=_to_float(evals.get("hallucination")),
                    context_relevance=_to_float(evals.get("context_relevance")),
                    retriever_quality=_to_float(evals.get("retriever_quality")),
                    consistency=_to_float(evals.get("consistency")),
                    accuracy=_to_float(evals.get("accuracy")),
                )
            )

        event = AIMetricEvent(
            session_id=session_id,
            thread_id=session_id,
            conversation_id=conversation_id,
            user_query_hash=_query_hash(query),
            model_name=app_meta["model_name"],
            application_version=app_meta["application_version"],
            git_revision=app_meta["git_revision"],
            langsmith_run_id=root_run_id,
            langsmith_url=langsmith_url,
            query=query[:500],
            agents_invoked=agents_invoked,
            # Workflow scores
            groundedness=_to_float(workflow_evals.get("groundedness")),
            context_relevance=_to_float(workflow_evals.get("context_relevance")),
            answer_correctness=_to_float(workflow_evals.get("answer_correctness")),
            hallucination=None,   # workflow-level hallucination via validator
            task_completion=_to_float(workflow_evals.get("task_completion")),
            tool_selection=_to_float(workflow_evals.get("tool_selection")),
            # Runtime
            latency_ms=latency_ms,
            total_tokens=total_tokens,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost=cost,
            agents=agent_metrics,
        )

        logger.debug(
            f"MetricBuilder assembled event for session={session_id[:8]} "
            f"latency={latency_ms:.0f}ms tokens={total_tokens} cost=${cost:.6f}"
        )
        return event
