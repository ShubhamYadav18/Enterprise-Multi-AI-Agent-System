"""
ServiceNowProvider — creates incidents in ServiceNow via the Table API.

Configuration (all from .env, no hardcoded values):
    SERVICENOW_ENABLED
    SERVICENOW_INSTANCE_URL
    SERVICENOW_USERNAME
    SERVICENOW_PASSWORD
    SERVICENOW_TABLE
    SERVICENOW_CATEGORY
    SERVICENOW_SUBCATEGORY
    SERVICENOW_ASSIGNMENT_GROUP
    SERVICENOW_IMPACT
    SERVICENOW_URGENCY
"""
from __future__ import annotations

import json
import time
import urllib.request
import urllib.error
from base64 import b64encode
from typing import Any, Optional

from incident.base import IncidentProvider
from incident.models import IncidentEvent, IncidentResult
from utils.logger import get_logger

logger = get_logger(__name__)

_TIMEOUT_SECONDS = 15
_MAX_RETRIES = 3


def _basic_auth_header(username: str, password: str) -> str:
    token = b64encode(f"{username}:{password}".encode()).decode()
    return f"Basic {token}"


def _build_description(event: IncidentEvent) -> str:
    """Build a rich incident description from the monitoring event."""
    pr = event.policy_result
    me = event.metric_event

    violations_text = "\n".join(
        f"  [{v.severity.upper()}] {v.metric}: actual={v.actual:.4f}, threshold={v.threshold:.4f}\n  {v.description}"
        for v in getattr(pr, "violations", [])
    ) or "  None"

    agents_text = ", ".join(getattr(me, "agents_invoked", [])) or "N/A"

    gnd = getattr(me, "groundedness", None)
    hal = getattr(me, "hallucination", None)
    ctr = getattr(me, "context_relevance", None)
    anc = getattr(me, "answer_correctness", None)
    lat = getattr(me, "latency_ms", 0)
    tok = getattr(me, "total_tokens", 0)
    cst = getattr(me, "cost", 0)

    def fmt(v: Optional[float]) -> str:
        return f"{v:.4f}" if v is not None else "N/A"

    return f"""AI Evaluation Threshold Breached
{"=" * 60}

Session ID:           {event.session_id}
Timestamp:            {getattr(me, "timestamp", "N/A")}
Model:                {getattr(me, "model_name", "N/A")}
Application Version:  {getattr(me, "application_version", "N/A")}
Git Revision:         {getattr(me, "git_revision", "N/A")}
Agents Invoked:       {agents_text}

User Query (truncated):
  {event.query[:300]}

--- Evaluation Metrics ---
Groundedness:         {fmt(gnd)}
Hallucination:        {fmt(hal)}
Context Relevance:    {fmt(ctr)}
Answer Correctness:   {fmt(anc)}
Latency:              {lat:.0f} ms
Total Tokens:         {tok}
Cost:                 ${cst:.6f}

--- Policy Violations ---
{violations_text}

--- Links ---
LangSmith Trace:      {event.langsmith_url or "N/A"}

--- Recommended Investigation Steps ---
1. Open the LangSmith trace link above to inspect the full execution tree.
2. Review agent outputs for hallucinated or unsupported content.
3. Check RAG retrieval quality — consider increasing retrieval_k or chunk_size.
4. Review model temperature settings if hallucination is high.
5. If latency is high, check for external API rate limiting or slow embeddings.
"""


class ServiceNowProvider(IncidentProvider):
    """
    Full ServiceNow Table API implementation.

    Reads all configuration from get_settings(). No hardcoded values.
    """

    def __init__(self) -> None:
        from config.settings import get_settings
        self._s = get_settings()
        self._base_url = self._s.servicenow_instance_url.rstrip("/")
        self._auth = _basic_auth_header(
            self._s.servicenow_username,
            self._s.servicenow_password,
        )
        self._table = self._s.servicenow_table
        self._headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": self._auth,
        }

    @property
    def provider_name(self) -> str:
        return "servicenow"

    def _request(
        self,
        method: str,
        path: str,
        data: Optional[dict[str, Any]] = None,
    ) -> tuple[int, dict[str, Any]]:
        """Make an HTTP request to ServiceNow. Returns (status_code, body_dict)."""
        url = f"{self._base_url}{path}"
        payload = json.dumps(data).encode("utf-8") if data else None
        req = urllib.request.Request(
            url=url, data=payload, method=method, headers=self._headers
        )
        with urllib.request.urlopen(req, timeout=_TIMEOUT_SECONDS) as resp:
            body = json.loads(resp.read().decode("utf-8"))
            return resp.status, body

    def create_incident(self, event: IncidentEvent) -> IncidentResult:
        """Create a ServiceNow incident. Returns IncidentResult. Never raises."""
        if not self._s.servicenow_enabled:
            return IncidentResult(created=False, provider="servicenow",
                                  error="SERVICENOW_ENABLED=false")

        pr = event.policy_result
        status_str = str(getattr(pr, "status", "unknown")).upper()
        violations_count = len(getattr(pr, "violations", []))

        payload = {
            "short_description": (
                f"[AI Control Tower] Threshold Breached — "
                f"{status_str} — Session {event.session_id[:8]}"
            ),
            "description": _build_description(event),
            "category": self._s.servicenow_category,
            "subcategory": self._s.servicenow_subcategory,
            "impact": str(self._s.servicenow_impact),
            "urgency": str(self._s.servicenow_urgency),
            "work_notes": (
                f"Automated incident raised by Enterprise AI Control Tower.\n"
                f"Violations: {violations_count}\n"
                f"Policy status: {status_str}\n"
                f"LangSmith URL: {event.langsmith_url or 'N/A'}"
            ),
        }
        if self._s.servicenow_assignment_group:
            payload["assignment_group"] = self._s.servicenow_assignment_group

        path = f"/api/now/table/{self._table}"
        last_error: Optional[str] = None

        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                status_code, body = self._request("POST", path, payload)
                result_data = body.get("result", {})
                inc_number = result_data.get("number", "")
                sys_id = result_data.get("sys_id", "")
                inc_url = (
                    f"{self._base_url}/nav_to.do?uri=incident.do?sys_id={sys_id}"
                    if sys_id else None
                )
                logger.info(
                    f"[ServiceNowProvider] Created incident {inc_number} "
                    f"for session={event.session_id[:8]} (HTTP {status_code})"
                )
                return IncidentResult(
                    created=True,
                    incident_number=inc_number,
                    incident_url=inc_url,
                    provider="servicenow",
                )
            except urllib.error.HTTPError as exc:
                body_text = exc.read().decode("utf-8", errors="replace")[:300]
                last_error = f"HTTP {exc.code}: {body_text}"
                logger.warning(
                    f"[ServiceNowProvider] Attempt {attempt} failed: {last_error}"
                )
            except Exception as exc:
                last_error = str(exc)
                logger.warning(
                    f"[ServiceNowProvider] Attempt {attempt} exception: {exc}"
                )

            if attempt < _MAX_RETRIES:
                time.sleep(2 ** (attempt - 1))

        logger.error(
            f"[ServiceNowProvider] All {_MAX_RETRIES} attempts failed: {last_error}"
        )
        return IncidentResult(created=False, provider="servicenow", error=last_error)

    def update_incident(self, incident_number: str, notes: str) -> bool:
        """Append work notes to an existing incident."""
        try:
            # Resolve sys_id from number
            path = f"/api/now/table/{self._table}?sysparm_query=number={incident_number}&sysparm_fields=sys_id"
            _, body = self._request("GET", path)
            records = body.get("result", [])
            if not records:
                return False
            sys_id = records[0].get("sys_id", "")
            _, _ = self._request(
                "PATCH",
                f"/api/now/table/{self._table}/{sys_id}",
                {"work_notes": notes},
            )
            return True
        except Exception as exc:
            logger.warning(f"[ServiceNowProvider] update_incident failed: {exc}")
            return False

    def resolve_incident(self, incident_number: str) -> bool:
        """Set incident state to Resolved (6) and close it."""
        try:
            path = f"/api/now/table/{self._table}?sysparm_query=number={incident_number}&sysparm_fields=sys_id"
            _, body = self._request("GET", path)
            records = body.get("result", [])
            if not records:
                return False
            sys_id = records[0].get("sys_id", "")
            _, _ = self._request(
                "PATCH",
                f"/api/now/table/{self._table}/{sys_id}",
                {
                    "state": "6",  # 6 = Resolved in ServiceNow
                    "close_code": "Solved (Permanently)",
                    "close_notes": "Resolved by Enterprise AI Control Tower.",
                },
            )
            logger.info(f"[ServiceNowProvider] Resolved incident {incident_number}")
            return True
        except Exception as exc:
            logger.warning(f"[ServiceNowProvider] resolve_incident failed: {exc}")
            return False

    def health_check(self) -> bool:
        """Check connectivity to the ServiceNow instance."""
        try:
            status_code, _ = self._request("GET", "/api/now/table/incident?sysparm_limit=1")
            return status_code == 200
        except Exception as exc:
            logger.debug(f"[ServiceNowProvider] health_check failed: {exc}")
            return False
