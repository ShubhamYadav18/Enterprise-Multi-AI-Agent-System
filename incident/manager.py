"""
IncidentManager — holds the active provider and handles incident creation.

The active provider is determined by the INCIDENT_PROVIDER env variable:
  - "noop"        → NoOpProvider (default, safe for all environments)
  - "servicenow"  → ServiceNowProvider (requires SERVICENOW_* env vars)
  - "jira"        → JiraProvider (placeholder — not yet implemented)
  - "pagerduty"   → PagerDutyProvider (placeholder — not yet implemented)

Switching providers requires only a .env change and server restart.
No code changes required.
"""
from __future__ import annotations

import threading
from typing import Optional

from incident.base import IncidentProvider
from incident.models import IncidentEvent, IncidentResult
from utils.logger import get_logger

logger = get_logger(__name__)


def _build_provider(provider_name: str) -> IncidentProvider:
    """Instantiate the correct IncidentProvider by name."""
    name = provider_name.lower().strip()

    if name == "servicenow":
        from incident.providers.servicenow import ServiceNowProvider
        logger.info("[IncidentManager] Provider: ServiceNow")
        return ServiceNowProvider()

    if name == "jira":
        from incident.providers.jira import JiraProvider
        logger.info("[IncidentManager] Provider: Jira (placeholder)")
        return JiraProvider()

    if name == "pagerduty":
        from incident.providers.pagerduty import PagerDutyProvider
        logger.info("[IncidentManager] Provider: PagerDuty (placeholder)")
        return PagerDutyProvider()

    # Default / "noop"
    from incident.providers.noop import NoOpProvider
    logger.info(f"[IncidentManager] Provider: NoOp (requested='{provider_name}')")
    return NoOpProvider()


class IncidentManager:
    """
    Manages incident creation through a pluggable provider.

    handle() is the sole entry point. It only calls create_incident()
    when the policy result has violations. Never raises.
    """

    def __init__(self, provider: IncidentProvider) -> None:
        self._provider = provider

    @property
    def provider(self) -> IncidentProvider:
        return self._provider

    def handle(self, event: IncidentEvent) -> IncidentResult:
        """
        Create an incident if the policy has violations.

        Returns IncidentResult. Never raises.
        """
        try:
            pr = event.policy_result
            violations = getattr(pr, "violations", [])
            status = getattr(pr, "status", None)

            if not violations:
                logger.debug(
                    f"[IncidentManager] No violations for session={event.session_id[:8]} — skipping"
                )
                return IncidentResult(created=False, provider=self._provider.provider_name)

            logger.info(
                f"[IncidentManager] Creating incident via {self._provider.provider_name} "
                f"for session={event.session_id[:8]} status={status}"
            )
            return self._provider.create_incident(event)

        except Exception as exc:
            logger.error(f"[IncidentManager] handle() raised: {exc}")
            return IncidentResult(
                created=False,
                provider=self._provider.provider_name,
                error=str(exc),
            )


# ============================================================
# Singleton accessor
# ============================================================

_manager_instance: Optional[IncidentManager] = None
_manager_lock = threading.Lock()


def get_incident_manager() -> IncidentManager:
    """Return the process-wide IncidentManager singleton."""
    global _manager_instance
    if _manager_instance is None:
        with _manager_lock:
            if _manager_instance is None:
                try:
                    from config.settings import get_settings
                    provider_name = get_settings().incident_provider
                except Exception:
                    provider_name = "noop"
                provider = _build_provider(provider_name)
                _manager_instance = IncidentManager(provider)
    return _manager_instance
