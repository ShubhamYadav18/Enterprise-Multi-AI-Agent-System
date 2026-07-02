"""
IncidentProvider — abstract base class for all incident backends.

Any new provider (Jira, PagerDuty, OpsGenie, Azure DevOps, etc.)
only needs to implement this interface. The rest of the system
never changes.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from incident.models import IncidentEvent, IncidentResult


class IncidentProvider(ABC):
    """Abstract base class for incident management providers."""

    @abstractmethod
    def create_incident(self, event: IncidentEvent) -> IncidentResult:
        """Create a new incident. Must never raise."""
        ...

    def update_incident(self, incident_number: str, notes: str) -> bool:
        """Add notes / update an existing incident. Optional. Returns True on success."""
        return False

    def resolve_incident(self, incident_number: str) -> bool:
        """Resolve / close an incident. Optional. Returns True on success."""
        return False

    def health_check(self) -> bool:
        """Test connectivity to the provider. Returns True if reachable."""
        return True

    @property
    def provider_name(self) -> str:
        """Human-readable provider name."""
        return self.__class__.__name__
