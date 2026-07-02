"""
JiraProvider — placeholder implementation.

Set INCIDENT_PROVIDER=jira in .env to enable (requires implementation).
"""
from incident.base import IncidentProvider
from incident.models import IncidentEvent, IncidentResult
from utils.logger import get_logger

logger = get_logger(__name__)


class JiraProvider(IncidentProvider):
    """Jira Service Management provider — placeholder."""

    @property
    def provider_name(self) -> str:
        return "jira"

    def create_incident(self, event: IncidentEvent) -> IncidentResult:
        logger.warning(
            "[JiraProvider] Jira integration is not yet implemented. "
            "Set INCIDENT_PROVIDER=noop or INCIDENT_PROVIDER=servicenow."
        )
        return IncidentResult(
            created=False,
            provider="jira",
            error="JiraProvider is not yet implemented.",
        )

    def health_check(self) -> bool:
        return False
