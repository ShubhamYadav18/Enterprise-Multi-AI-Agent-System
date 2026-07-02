"""
NoOpProvider — default incident provider.

Logs the incident event but creates no external tickets.
Safe to use in development / CI / when no incident platform is configured.
"""
from __future__ import annotations
from incident.base import IncidentProvider
from incident.models import IncidentEvent, IncidentResult
from utils.logger import get_logger

logger = get_logger(__name__)


class NoOpProvider(IncidentProvider):
    """Incident provider that only logs. Never calls external systems."""

    @property
    def provider_name(self) -> str:
        return "noop"

    def create_incident(self, event: IncidentEvent) -> IncidentResult:
        pr = event.policy_result
        status = getattr(pr, "status", "unknown")
        violations = getattr(pr, "violations", [])
        logger.info(
            f"[NoOpProvider] Incident event received: session={event.session_id[:8]} "
            f"status={status} violations={len(violations)} "
            f"(no ticket created — provider=noop)"
        )
        return IncidentResult(created=False, provider="noop")

    def health_check(self) -> bool:
        return True
