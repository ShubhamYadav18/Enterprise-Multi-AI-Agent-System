"""
Incident data models.
"""
from __future__ import annotations
from typing import Optional, TYPE_CHECKING
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from monitoring.models import PolicyResult, AIMetricEvent


class IncidentEvent(BaseModel):
    """Payload passed to IncidentProvider.create_incident()."""

    session_id: str
    query: str = ""
    policy_result: object   # PolicyResult — typed as object to avoid circular import
    metric_event: object    # AIMetricEvent
    langsmith_url: Optional[str] = None

    model_config = {"arbitrary_types_allowed": True}


class IncidentResult(BaseModel):
    """Result returned by IncidentProvider.create_incident()."""

    created: bool = False
    incident_number: Optional[str] = None
    incident_url: Optional[str] = None
    provider: str = "noop"
    error: Optional[str] = None
