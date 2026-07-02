"""
Pydantic models for the AI Monitoring Framework.

These models are the data contracts between:
  - MetricBuilder (constructs events from chat response)
  - DatadogPublisher (sends events to Datadog)
  - PolicyEngine (evaluates thresholds)
  - IncidentManager (creates incidents from policy violations)
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


# ============================================================
# Policy Severity
# ============================================================

class PolicyStatus(str, Enum):
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"


# ============================================================
# Per-Agent Metric
# ============================================================

class AgentMetric(BaseModel):
    """Monitoring metrics for a single executed agent."""

    agent_name: str
    latency_ms: float = 0.0
    groundedness: Optional[float] = None
    hallucination: Optional[float] = None
    context_relevance: Optional[float] = None
    retriever_quality: Optional[float] = None
    consistency: Optional[float] = None
    accuracy: Optional[float] = None
    execution_status: str = "executed"   # "executed" | "skipped"


# ============================================================
# AI Metric Event — the canonical monitoring payload
# ============================================================

class AIMetricEvent(BaseModel):
    """
    Complete AI metric event for one user query session.

    Published to Datadog and evaluated by the Policy Engine.
    All values read from the completed ChatResponse + evaluations.
    """

    # Session identity tags
    session_id: str
    thread_id: str = ""
    conversation_id: str = ""
    user_query_hash: str = ""
    model_name: str = "unknown"
    application_version: str = "0.1.0"
    git_revision: str = "unknown"
    langsmith_run_id: str = ""
    langsmith_url: Optional[str] = None
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    # Original query (truncated to 500 chars for safety)
    query: str = ""

    # Agents invoked
    agents_invoked: list[str] = Field(default_factory=list)

    # Workflow evaluation scores (None = evaluator not run / N/A)
    groundedness: Optional[float] = None
    context_relevance: Optional[float] = None
    answer_correctness: Optional[float] = None
    hallucination: Optional[float] = None
    task_completion: Optional[float] = None
    tool_selection: Optional[float] = None

    # Runtime metrics
    latency_ms: float = 0.0
    total_tokens: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost: float = 0.0

    # Per-agent breakdown
    agents: list[AgentMetric] = Field(default_factory=list)


# ============================================================
# Policy Violation
# ============================================================

class PolicyViolation(BaseModel):
    """A single threshold breach detected by the Policy Engine."""

    metric: str
    actual: float
    threshold: float
    severity: PolicyStatus
    description: str = ""


# ============================================================
# Policy Result
# ============================================================

class PolicyResult(BaseModel):
    """Result of evaluating an AIMetricEvent against configured thresholds."""

    status: PolicyStatus = PolicyStatus.HEALTHY
    violations: list[PolicyViolation] = Field(default_factory=list)
    session_id: str = ""
    evaluated_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    @property
    def has_violations(self) -> bool:
        return len(self.violations) > 0

    @property
    def critical_violations(self) -> list[PolicyViolation]:
        return [v for v in self.violations if v.severity == PolicyStatus.CRITICAL]

    @property
    def warning_violations(self) -> list[PolicyViolation]:
        return [v for v in self.violations if v.severity == PolicyStatus.WARNING]

    def summary_text(self) -> str:
        if not self.violations:
            return "All metrics within thresholds."
        lines = []
        for v in self.violations:
            lines.append(
                f"[{v.severity.upper()}] {v.metric}: {v.actual:.3f} "
                f"(threshold: {v.threshold:.3f})"
            )
        return "\n".join(lines)


# ============================================================
# Session Monitoring Result
# ============================================================

class SessionMonitoringResult(BaseModel):
    """
    Combined result of one monitoring cycle for a session.

    Stored in Streamlit session state to display status in the UI.
    """

    session_id: str
    policy_result: PolicyResult
    datadog_published: bool = False
    datadog_error: Optional[str] = None
    incident_created: bool = False
    incident_number: Optional[str] = None
    incident_url: Optional[str] = None
    incident_error: Optional[str] = None
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


# ============================================================
# Datadog publish result
# ============================================================

class DatadogPublishResult(BaseModel):
    """Result of a Datadog metric publish attempt."""

    success: bool
    metrics_count: int = 0
    error: Optional[str] = None
    http_status: Optional[int] = None
