"""
DatadogClient — low-level HTTP client for the Datadog Metrics API v2.

Sends gauge metrics in batches. Uses only the standard library (urllib)
to avoid adding a hard dependency on the datadog-api-client package.

Configuration via get_settings():
    DATADOG_API_KEY   — required
    DATADOG_APP_KEY   — required
    DATADOG_SITE      — e.g. datadoghq.com
"""

from __future__ import annotations

import json
import time
import urllib.request
import urllib.error
from typing import Any

from monitoring.models import AIMetricEvent, DatadogPublishResult
from utils.logger import get_logger

logger = get_logger(__name__)

_TIMEOUT_SECONDS = 10


def _build_series(event: AIMetricEvent, prefix: str) -> list[dict[str, Any]]:
    """
    Convert an AIMetricEvent into a list of Datadog v2 metric series payloads.

    Each series is a gauge with session-level tags. Only numeric values
    are published (None values are omitted to avoid polluting Datadog).
    """
    now = int(time.time())

    # Base tags for every metric
    tags = [
        f"session_id:{event.session_id[:16]}",
        f"model:{event.model_name}",
        f"app_version:{event.application_version}",
        f"git_rev:{event.git_revision}",
        f"conversation_id:{event.conversation_id[:16] if event.conversation_id else 'none'}",
    ]
    for agent in event.agents_invoked:
        tags.append(f"agent:{agent}")

    def gauge(name: str, value: float) -> dict[str, Any]:
        return {
            "metric": f"{prefix}.{name}",
            "type": 3,  # GAUGE
            "points": [{"timestamp": now, "value": value}],
            "tags": tags,
        }

    series = []

    # Workflow scores
    workflow_metrics = {
        "groundedness": event.groundedness,
        "context_relevance": event.context_relevance,
        "answer_correctness": event.answer_correctness,
        "hallucination": event.hallucination,
        "task_completion": event.task_completion,
        "tool_selection": event.tool_selection,
    }
    for metric_name, value in workflow_metrics.items():
        if value is not None:
            series.append(gauge(f"workflow.{metric_name}", value))

    # Runtime metrics
    series.append(gauge("runtime.latency_ms", event.latency_ms))
    series.append(gauge("runtime.total_tokens", float(event.total_tokens)))
    series.append(gauge("runtime.cost_usd", event.cost))
    if event.input_tokens:
        series.append(gauge("runtime.input_tokens", float(event.input_tokens)))
    if event.output_tokens:
        series.append(gauge("runtime.output_tokens", float(event.output_tokens)))

    # Per-agent metrics
    for agent_m in event.agents:
        if agent_m.execution_status == "skipped":
            continue
        safe_name = agent_m.agent_name.lower().replace(" ", "_")
        agent_tags = tags + [f"agent_name:{safe_name}"]

        def agent_gauge(name: str, value: float) -> dict[str, Any]:
            return {
                "metric": f"{prefix}.agent.{name}",
                "type": 3,
                "points": [{"timestamp": now, "value": value}],
                "tags": agent_tags,
            }

        if agent_m.latency_ms:
            series.append(agent_gauge(f"{safe_name}.latency_ms", agent_m.latency_ms))
        for attr in ("groundedness", "hallucination", "context_relevance",
                     "retriever_quality", "consistency", "accuracy"):
            val = getattr(agent_m, attr)
            if val is not None:
                series.append(agent_gauge(f"{safe_name}.{attr}", val))

    return series


class DatadogClient:
    """
    Sends metrics to Datadog via the v2 Metrics Submit API.

    Authentication uses DD-API-KEY and DD-APPLICATION-KEY headers.
    No third-party SDK required.
    """

    def __init__(self, api_key: str, app_key: str, site: str = "datadoghq.com") -> None:
        self._api_key = api_key
        self._app_key = app_key
        self._base_url = f"https://api.{site}/api/v2/series"

    def submit_metrics(self, series: list[dict[str, Any]]) -> DatadogPublishResult:
        """
        POST a batch of series to the Datadog v2 metrics endpoint.

        Returns DatadogPublishResult regardless of success/failure.
        """
        if not series:
            return DatadogPublishResult(success=True, metrics_count=0)

        payload = json.dumps({"series": series}).encode("utf-8")
        req = urllib.request.Request(
            url=self._base_url,
            data=payload,
            method="POST",
            headers={
                "Content-Type": "application/json",
                "DD-API-KEY": self._api_key,
                "DD-APPLICATION-KEY": self._app_key,
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=_TIMEOUT_SECONDS) as resp:
                status = resp.status
                logger.info(
                    f"Datadog metrics submitted: {len(series)} series, HTTP {status}"
                )
                return DatadogPublishResult(
                    success=True,
                    metrics_count=len(series),
                    http_status=status,
                )
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")[:300]
            logger.warning(f"Datadog HTTP {exc.code}: {body}")
            return DatadogPublishResult(
                success=False,
                http_status=exc.code,
                error=f"HTTP {exc.code}: {body}",
            )
        except Exception as exc:
            logger.warning(f"Datadog submit failed: {exc}")
            return DatadogPublishResult(success=False, error=str(exc))

    def health_check(self) -> bool:
        """
        Validate Datadog connectivity by submitting a 0-value ping gauge.

        The v1/validate endpoint requires additional permissions not always
        granted to API-only keys. Sending a metric to v2/series is the
        most reliable connectivity test — a 202 response means the key is valid.
        """
        import time as _time
        series = [{
            "metric": "ai.enterprise.health.ping",
            "type": 3,
            "points": [{"timestamp": int(_time.time()), "value": 0.0}],
            "tags": ["source:ai_control_tower", "check:health"],
        }]
        result = self.submit_metrics(series)
        return result.success

