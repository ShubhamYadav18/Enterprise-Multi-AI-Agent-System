"""
Metrics Publisher — abstract base + DatadogPublisher implementation.

Design:
  - MetricsPublisher  — abstract base class (plug in any monitoring backend)
  - DatadogPublisher  — publishes AIMetricEvent to Datadog with retry

Publishing failures NEVER raise. They return DatadogPublishResult with
success=False and an error message.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Any

from monitoring.datadog_client import DatadogClient, _build_series
from monitoring.models import AIMetricEvent, DatadogPublishResult
from utils.logger import get_logger

logger = get_logger(__name__)

_DEFAULT_MAX_RETRIES = 3
_DEFAULT_BACKOFF_BASE = 1.0   # seconds


# ============================================================
# Abstract Base
# ============================================================

class MetricsPublisher(ABC):
    """Abstract base class for metrics publishers."""

    @abstractmethod
    def publish(self, event: AIMetricEvent) -> DatadogPublishResult:
        """Publish a metric event. Must never raise."""
        ...

    def health_check(self) -> bool:
        """Optional connectivity check. Returns True by default."""
        return True


# ============================================================
# NoOp Publisher (default when Datadog disabled)
# ============================================================

class NoOpPublisher(MetricsPublisher):
    """No-operation publisher — logs the event but does not send anything."""

    def publish(self, event: AIMetricEvent) -> DatadogPublishResult:
        logger.info(
            f"[NoOpPublisher] Metric event for session={event.session_id[:8]} "
            f"(Datadog disabled)"
        )
        return DatadogPublishResult(success=True, metrics_count=0)


# ============================================================
# Datadog Publisher
# ============================================================

class DatadogPublisher(MetricsPublisher):
    """
    Publishes AIMetricEvent to Datadog with exponential-backoff retry.

    Configuration is read from get_settings() on construction.
    """

    def __init__(
        self,
        api_key: str,
        app_key: str,
        site: str = "datadoghq.com",
        metric_prefix: str = "ai.enterprise",
        max_retries: int = _DEFAULT_MAX_RETRIES,
        backoff_base: float = _DEFAULT_BACKOFF_BASE,
    ) -> None:
        self._client = DatadogClient(api_key=api_key, app_key=app_key, site=site)
        self._prefix = metric_prefix
        self._max_retries = max_retries
        self._backoff_base = backoff_base

    def publish(self, event: AIMetricEvent) -> DatadogPublishResult:
        """
        Build metric series from the event and POST to Datadog with retry.

        Retries up to max_retries times with exponential backoff.
        Never raises — returns DatadogPublishResult on all paths.
        """
        try:
            series = _build_series(event, self._prefix)
        except Exception as exc:
            logger.warning(f"MetricBuilder series construction failed: {exc}")
            return DatadogPublishResult(success=False, error=str(exc))

        last_result: DatadogPublishResult = DatadogPublishResult(
            success=False, error="No attempts made"
        )

        for attempt in range(1, self._max_retries + 1):
            try:
                result = self._client.submit_metrics(series)
                if result.success:
                    logger.info(
                        f"[DatadogPublisher] Published {result.metrics_count} metrics "
                        f"for session={event.session_id[:8]} (attempt {attempt})"
                    )
                    return result
                last_result = result
                logger.warning(
                    f"[DatadogPublisher] Attempt {attempt} failed: {result.error}"
                )
            except Exception as exc:
                last_result = DatadogPublishResult(success=False, error=str(exc))
                logger.warning(f"[DatadogPublisher] Attempt {attempt} exception: {exc}")

            if attempt < self._max_retries:
                sleep_time = self._backoff_base * (2 ** (attempt - 1))
                logger.debug(f"[DatadogPublisher] Retrying in {sleep_time:.1f}s")
                time.sleep(sleep_time)

        logger.error(
            f"[DatadogPublisher] All {self._max_retries} attempts failed "
            f"for session={event.session_id[:8]}: {last_result.error}"
        )
        return last_result

    def health_check(self) -> bool:
        return self._client.health_check()


# ============================================================
# Publisher Factory
# ============================================================

def get_publisher() -> MetricsPublisher:
    """
    Return the configured MetricsPublisher.

    Returns DatadogPublisher if DATADOG_ENABLED=true and keys are set,
    otherwise returns NoOpPublisher.
    """
    try:
        from config.settings import get_settings
        s = get_settings()

        if s.datadog_enabled and s.datadog_api_key and s.datadog_app_key:
            logger.info("[Publisher] Using DatadogPublisher")
            return DatadogPublisher(
                api_key=s.datadog_api_key,
                app_key=s.datadog_app_key,
                site=s.datadog_site,
                metric_prefix=s.datadog_metric_prefix,
            )
    except Exception as exc:
        logger.warning(f"[Publisher] Failed to build DatadogPublisher: {exc}")

    logger.info("[Publisher] Using NoOpPublisher (Datadog disabled or not configured)")
    return NoOpPublisher()
