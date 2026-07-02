"""
Policy Engine — evaluates AIMetricEvent against configured thresholds.

All thresholds are read from get_settings() (env vars / .env).
No values are hardcoded.

Severity rules:
  - groundedness < GROUNDEDNESS_MIN    → CRITICAL
  - hallucination > HALLUCINATION_MAX  → CRITICAL
  - latency_ms > LATENCY_MAX_MS        → WARNING
  - cost > TOTAL_COST_MAX              → WARNING
  - total_tokens > TOKEN_LIMIT         → WARNING
"""

from __future__ import annotations

from monitoring.models import (
    AIMetricEvent,
    PolicyResult,
    PolicyStatus,
    PolicyViolation,
)
from utils.logger import get_logger

logger = get_logger(__name__)


class PolicyEngine:
    """
    Evaluates an AIMetricEvent and returns a PolicyResult.

    All threshold values come from the Settings singleton so they
    can be changed via .env without touching code.
    """

    def evaluate(self, event: AIMetricEvent) -> PolicyResult:
        """
        Run all threshold checks against the metric event.

        Returns:
            PolicyResult with overall status and list of violations.
            Never raises.
        """
        try:
            from config.settings import get_settings
            s = get_settings()
        except Exception as exc:
            logger.warning(f"PolicyEngine: could not load settings, skipping: {exc}")
            return PolicyResult(status=PolicyStatus.HEALTHY, session_id=event.session_id)

        violations: list[PolicyViolation] = []

        # ── CRITICAL: groundedness below minimum ────────────────
        if event.groundedness is not None and event.groundedness < s.groundedness_min:
            violations.append(PolicyViolation(
                metric="groundedness",
                actual=event.groundedness,
                threshold=s.groundedness_min,
                severity=PolicyStatus.CRITICAL,
                description=(
                    f"Groundedness {event.groundedness:.3f} is below "
                    f"minimum {s.groundedness_min:.3f}. "
                    "Response may not be grounded in retrieved context."
                ),
            ))

        # ── CRITICAL: hallucination above maximum ───────────────
        if event.hallucination is not None and event.hallucination > s.hallucination_max:
            violations.append(PolicyViolation(
                metric="hallucination",
                actual=event.hallucination,
                threshold=s.hallucination_max,
                severity=PolicyStatus.CRITICAL,
                description=(
                    f"Hallucination risk {event.hallucination:.3f} exceeds "
                    f"maximum {s.hallucination_max:.3f}."
                ),
            ))

        # ── WARNING: latency over limit ─────────────────────────
        if event.latency_ms > s.latency_max_ms:
            violations.append(PolicyViolation(
                metric="latency_ms",
                actual=event.latency_ms,
                threshold=s.latency_max_ms,
                severity=PolicyStatus.WARNING,
                description=(
                    f"Latency {event.latency_ms:.0f}ms exceeds "
                    f"threshold {s.latency_max_ms:.0f}ms."
                ),
            ))

        # ── WARNING: cost over limit ────────────────────────────
        if event.cost > s.total_cost_max:
            violations.append(PolicyViolation(
                metric="cost_usd",
                actual=event.cost,
                threshold=s.total_cost_max,
                severity=PolicyStatus.WARNING,
                description=(
                    f"Query cost ${event.cost:.6f} exceeds "
                    f"threshold ${s.total_cost_max:.6f}."
                ),
            ))

        # ── WARNING: token usage over limit ─────────────────────
        if event.total_tokens > s.token_limit:
            violations.append(PolicyViolation(
                metric="total_tokens",
                actual=float(event.total_tokens),
                threshold=float(s.token_limit),
                severity=PolicyStatus.WARNING,
                description=(
                    f"Token usage {event.total_tokens} exceeds "
                    f"limit {s.token_limit}."
                ),
            ))

        # ── Determine overall status ────────────────────────────
        if any(v.severity == PolicyStatus.CRITICAL for v in violations):
            overall = PolicyStatus.CRITICAL
        elif violations:
            overall = PolicyStatus.WARNING
        else:
            overall = PolicyStatus.HEALTHY

        result = PolicyResult(
            status=overall,
            violations=violations,
            session_id=event.session_id,
        )

        if violations:
            logger.warning(
                f"[PolicyEngine] session={event.session_id[:8]} "
                f"status={overall.upper()} violations={len(violations)}"
            )
        else:
            logger.info(
                f"[PolicyEngine] session={event.session_id[:8]} status=HEALTHY"
            )

        return result
