"""
MonitoringService — orchestrates the full monitoring pipeline for each session.

Pipeline:
  1. MetricBuilder.build()           — assemble AIMetricEvent
  2. MetricsPublisher.publish()      — send to Datadog (async, best-effort)
  3. PolicyEngine.evaluate()         — check thresholds
  4. IncidentManager.handle()        — create incident if policy breached

All steps are wrapped in individual try/except blocks.
The user response is NEVER affected by monitoring failures.

process_session_async() runs the pipeline in a daemon thread.
The result is stored in a shared dict keyed by session_id so the
Streamlit UI can display it when the user views their session.
"""

from __future__ import annotations

import threading
from typing import Any, Optional

from monitoring.metrics import MetricBuilder
from monitoring.models import (
    AIMetricEvent,
    PolicyResult,
    PolicyStatus,
    SessionMonitoringResult,
)
from monitoring.policy_engine import PolicyEngine
from monitoring.publisher import get_publisher
from utils.logger import get_logger

logger = get_logger(__name__)

# In-memory store: session_id → SessionMonitoringResult
# This is process-scoped. Streamlit reads it via the API.
_monitoring_results: dict[str, SessionMonitoringResult] = {}
_results_lock = threading.Lock()


def _store_result(result: SessionMonitoringResult) -> None:
    with _results_lock:
        _monitoring_results[result.session_id] = result
        # Keep only the last 200 results to avoid unbounded memory growth
        if len(_monitoring_results) > 200:
            oldest = list(_monitoring_results.keys())[0]
            del _monitoring_results[oldest]


def get_monitoring_result(session_id: str) -> Optional[SessionMonitoringResult]:
    """Retrieve the monitoring result for a specific session (thread-safe)."""
    with _results_lock:
        return _monitoring_results.get(session_id)


def get_all_monitoring_results() -> dict[str, SessionMonitoringResult]:
    """Return a snapshot of all stored monitoring results."""
    with _results_lock:
        return dict(_monitoring_results)


class MonitoringService:
    """
    Orchestrates the monitoring pipeline for completed query sessions.

    Instantiate once (singleton via get_monitoring_service()) and call
    process_session_async() after every successful chat response.
    """

    def __init__(self) -> None:
        self._publisher = get_publisher()
        self._policy_engine = PolicyEngine()

    def _run_pipeline(
        self,
        session_id: str,
        query: str,
        latency_ms: float,
        total_tokens: int,
        cost: float,
        workflow_evals: dict[str, Any],
        agent_evals: dict[str, Any],
        agents_invoked: list[str],
        conversation_id: str,
        root_run_id: str,
        langsmith_url: Optional[str],
        input_tokens: int,
        output_tokens: int,
    ) -> SessionMonitoringResult:
        """Run the full pipeline synchronously. Called from a daemon thread."""

        # Step 1: Build metric event
        try:
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
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )
        except Exception as exc:
            logger.error(f"[MonitoringService] MetricBuilder failed: {exc}")
            return SessionMonitoringResult(
                session_id=session_id,
                policy_result=PolicyResult(
                    status=PolicyStatus.HEALTHY, session_id=session_id
                ),
            )

        # Step 2: Publish to Datadog
        datadog_published = False
        datadog_error: Optional[str] = None
        try:
            pub_result = self._publisher.publish(event)
            datadog_published = pub_result.success
            if not pub_result.success:
                datadog_error = pub_result.error
        except Exception as exc:
            logger.error(f"[MonitoringService] Publisher raised: {exc}")
            datadog_error = str(exc)

        # Step 3: Evaluate policy
        try:
            policy_result = self._policy_engine.evaluate(event)
        except Exception as exc:
            logger.error(f"[MonitoringService] PolicyEngine raised: {exc}")
            policy_result = PolicyResult(
                status=PolicyStatus.HEALTHY, session_id=session_id
            )

        # Step 4: Create incident if policy breached
        incident_created = False
        incident_number: Optional[str] = None
        incident_url: Optional[str] = None
        incident_error: Optional[str] = None

        if policy_result.has_violations:
            try:
                from incident.manager import get_incident_manager
                from incident.models import IncidentEvent
                incident_manager = get_incident_manager()
                inc_event = IncidentEvent(
                    session_id=session_id,
                    query=query,
                    policy_result=policy_result,
                    metric_event=event,
                    langsmith_url=langsmith_url,
                )
                inc_result = incident_manager.handle(inc_event)
                incident_created = inc_result.created
                incident_number = inc_result.incident_number
                incident_url = inc_result.incident_url
                if inc_result.error:
                    incident_error = inc_result.error
            except Exception as exc:
                logger.error(f"[MonitoringService] IncidentManager raised: {exc}")
                incident_error = str(exc)

        result = SessionMonitoringResult(
            session_id=session_id,
            policy_result=policy_result,
            datadog_published=datadog_published,
            datadog_error=datadog_error,
            incident_created=incident_created,
            incident_number=incident_number,
            incident_url=incident_url,
            incident_error=incident_error,
        )

        _store_result(result)
        logger.info(
            f"[MonitoringService] Completed pipeline for session={session_id[:8]} "
            f"status={policy_result.status.upper()} "
            f"datadog={'OK' if datadog_published else 'SKIP/FAIL'} "
            f"incident={'YES' if incident_created else 'NO'}"
        )
        return result

    def process_session_async(
        self,
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
    ) -> None:
        """
        Fire-and-forget: run the monitoring pipeline in a daemon thread.

        Returns immediately. Never blocks the caller. Never raises.
        The result is stored in _monitoring_results[session_id].
        """
        kwargs = dict(
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
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )
        t = threading.Thread(target=self._run_pipeline, kwargs=kwargs, daemon=True)
        t.start()
        logger.debug(f"[MonitoringService] Background thread started for session={session_id[:8]}")


# ============================================================
# Singleton accessor
# ============================================================

_service_instance: Optional[MonitoringService] = None
_service_lock = threading.Lock()


def get_monitoring_service() -> MonitoringService:
    """Return the process-wide MonitoringService singleton."""
    global _service_instance
    if _service_instance is None:
        with _service_lock:
            if _service_instance is None:
                _service_instance = MonitoringService()
    return _service_instance
