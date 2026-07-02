"""
Security Service coordinating SBOM generation, scanning, and history.
"""

from __future__ import annotations

import json
import time
import uuid
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from config.settings import get_settings
from security.models import (
    SBOMResult,
    VulnerabilityFinding,
    SecurityScanResult,
    SecurityHistoryEntry,
)
from security.sbom import generate_sbom
from security.defectdojo import DefectDojoClient
from security.utils import parse_dependency_tree
from utils.logger import get_logger

logger = get_logger(__name__)


class SecurityService:
    """Orchestrator service for all Software Supply Chain security workflows."""

    _instance: Optional[SecurityService] = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return
            
        self.project_root = Path(__file__).resolve().parent.parent
        self.data_dir = self.project_root / "data"
        self.scans_dir = self.data_dir / "scans"
        self.history_file = self.data_dir / "security_history.json"
        
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.scans_dir.mkdir(parents=True, exist_ok=True)
        
        self._scan_lock = threading.Lock()
        self.active_scan_id: Optional[str] = None
        self._initialized = True

    def get_history(self) -> list[SecurityHistoryEntry]:
        """Load scan history list from disk."""
        if not self.history_file.exists():
            return []
        try:
            with open(self.history_file, encoding="utf-8") as f:
                data = json.load(f)
            return [SecurityHistoryEntry(**item) for item in data]
        except Exception as e:
            logger.error(f"Failed to load security history: {e}")
            return []

    def _save_history(self, history: list[SecurityHistoryEntry]) -> None:
        """Save scan history list to disk."""
        try:
            data = [item.model_dump() for item in history]
            with open(self.history_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to save security history: {e}")

    def get_scan_result(self, scan_id: str) -> Optional[SecurityScanResult]:
        """Load full scan result for a given scan ID from disk."""
        scan_file = self.scans_dir / f"{scan_id}.json"
        if not scan_file.exists():
            return None
        try:
            with open(scan_file, encoding="utf-8") as f:
                data = json.load(f)
            return SecurityScanResult(**data)
        except Exception as e:
            logger.error(f"Failed to load scan result {scan_id}: {e}")
            return None

    def _save_scan_result(self, result: SecurityScanResult) -> None:
        """Save full scan result to disk."""
        scan_file = self.scans_dir / f"{result.scan_id}.json"
        try:
            with open(scan_file, "w", encoding="utf-8") as f:
                json.dump(result.model_dump(), f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to save scan result {result.scan_id}: {e}")

    def get_latest_scan_result(self) -> Optional[SecurityScanResult]:
        """Get the detailed scan result of the most recent run."""
        history = self.get_history()
        if not history:
            return None
        # Retrieve the first one (latest is prepended or sorted)
        latest_entry = history[0]
        return self.get_scan_result(latest_entry.scan_id)

    def get_dependency_tree(self) -> dict[str, Any]:
        """Parse and return the current dependency tree from the last generated SBOM."""
        sbom_file = self.project_root / "sbom" / "bom.json"
        return parse_dependency_tree(sbom_file)

    def run_scan(self) -> SecurityScanResult:
        """Execute a full Software Supply Chain scan.

        Generates SBOM, uploads to DefectDojo, parses findings, updates history.
        This call is thread-safe and executes sequentially.
        """
        scan_id = str(uuid.uuid4())
        
        # Acquire lock to ensure only one scan runs at a time
        acquired = self._scan_lock.acquire(blocking=False)
        if not acquired:
            logger.warning("Another security scan is already in progress.")
            return SecurityScanResult(
                scan_id=scan_id,
                status="Failed",
                error="Another security scan is already in progress. Please wait.",
            )
            
        self.active_scan_id = scan_id
        start_time = time.time()
        logger.info(f"Starting security scan run: {scan_id}")

        try:
            # 1. Generate CycloneDX SBOM
            logger.info("Step 1: Generating CycloneDX SBOM...")
            sbom_res = generate_sbom()
            
            if sbom_res.status != "success":
                logger.error(f"SBOM generation failed: {sbom_res.error}")
                result = SecurityScanResult(
                    scan_id=scan_id,
                    status="Failed",
                    error=f"SBOM generation failed: {sbom_res.error}",
                )
                self._save_scan_result(result)
                self._add_to_history(result)
                return result

            # Parse packages count from SBOM
            packages_count = self._count_sbom_packages(sbom_res.output_file)

            # 2. Upload to DefectDojo
            settings = get_settings()
            if not settings.defectdojo_url or not settings.defectdojo_api_key:
                logger.warning("DefectDojo settings are missing. Completing in offline mode.")
                result = SecurityScanResult(
                    scan_id=scan_id,
                    timestamp=datetime.utcnow(),
                    total_packages=packages_count,
                    status="Completed (Offline)",
                    error="DefectDojo URL or API Key is missing. Scan completed in local offline mode.",
                )
                self._save_scan_result(result)
                self._add_to_history(result)
                return result

            logger.info("Step 2: Uploading to DefectDojo and running analysis...")
            client = DefectDojoClient(
                url=settings.defectdojo_url,
                api_key=settings.defectdojo_api_key,
                product_identifier=settings.defectdojo_product,
                engagement_identifier=settings.defectdojo_engagement,
            )

            # Resolve product and engagement
            product_id = client.get_product_id()
            engagement_id = client.get_engagement_id(product_id)

            # Upload scan
            upload_res = client.upload_sbom(sbom_res.output_file, engagement_id)
            test_id = upload_res.get("test")
            
            if not test_id:
                raise ValueError("DefectDojo response did not include a valid test ID.")

            # 3. Retrieve findings
            logger.info(f"Step 3: Fetching vulnerability findings for test ID {test_id}...")
            findings = client.get_findings(test_id)

            # Analyze findings severities
            critical = sum(1 for f in findings if f.severity.lower() == "critical")
            high = sum(1 for f in findings if f.severity.lower() == "high")
            medium = sum(1 for f in findings if f.severity.lower() == "medium")
            low = sum(1 for f in findings if f.severity.lower() == "low")
            info = sum(1 for f in findings if f.severity.lower() in ("info", "informational"))

            result = SecurityScanResult(
                scan_id=scan_id,
                test_id=test_id,
                timestamp=datetime.utcnow(),
                total_packages=packages_count,
                critical_count=critical,
                high_count=high,
                medium_count=medium,
                low_count=low,
                info_count=info,
                findings=findings,
                status="Completed",
            )
            
            self._save_scan_result(result)
            self._add_to_history(result)
            logger.info(f"Security scan run {scan_id} completed successfully. Found {len(findings)} issues.")
            return result

        except Exception as e:
            logger.error(f"Exception during security scan run {scan_id}: {e}", exc_info=True)
            result = SecurityScanResult(
                scan_id=scan_id,
                status="Failed (DefectDojo Offline)",
                error=f"DefectDojo integration failed: {str(e)}",
            )
            self._save_scan_result(result)
            self._add_to_history(result)
            return result
            
        finally:
            self.active_scan_id = None
            self._scan_lock.release()

    def run_scan_async(self) -> str:
        """Run a security scan asynchronously in a background thread.

        Returns:
            The scan_id that was created for the run.
        """
        scan_id = str(uuid.uuid4())
        
        # Start a thread to run the scan
        thread = threading.Thread(
            target=self._run_scan_thread,
            args=(scan_id,),
            daemon=True,
        )
        thread.start()
        
        return scan_id

    def _run_scan_thread(self, scan_id: str) -> None:
        """Worker thread entry point for scan execution."""
        # Acquire lock to prevent overlapping runs
        acquired = self._scan_lock.acquire(blocking=True, timeout=5)
        if not acquired:
            logger.warning("Async scan aborted: another scan in progress.")
            return
            
        self.active_scan_id = scan_id
        start_time = time.time()
        logger.info(f"Starting async security scan run: {scan_id}")

        try:
            sbom_res = generate_sbom()
            if sbom_res.status != "success":
                result = SecurityScanResult(
                    scan_id=scan_id,
                    status="Failed",
                    error=f"SBOM generation failed: {sbom_res.error}",
                )
                self._save_scan_result(result)
                self._add_to_history(result)
                return

            packages_count = self._count_sbom_packages(sbom_res.output_file)

            settings = get_settings()
            if not settings.defectdojo_url or not settings.defectdojo_api_key:
                result = SecurityScanResult(
                    scan_id=scan_id,
                    timestamp=datetime.utcnow(),
                    total_packages=packages_count,
                    status="Completed (Offline)",
                    error="DefectDojo settings are missing. Completed in local offline mode.",
                )
                self._save_scan_result(result)
                self._add_to_history(result)
                return

            client = DefectDojoClient(
                url=settings.defectdojo_url,
                api_key=settings.defectdojo_api_key,
                product_identifier=settings.defectdojo_product,
                engagement_identifier=settings.defectdojo_engagement,
            )

            product_id = client.get_product_id()
            engagement_id = client.get_engagement_id(product_id)

            upload_res = client.upload_sbom(sbom_res.output_file, engagement_id)
            test_id = upload_res.get("test")
            
            if not test_id:
                raise ValueError("DefectDojo response did not include a valid test ID.")

            findings = client.get_findings(test_id)

            critical = sum(1 for f in findings if f.severity.lower() == "critical")
            high = sum(1 for f in findings if f.severity.lower() == "high")
            medium = sum(1 for f in findings if f.severity.lower() == "medium")
            low = sum(1 for f in findings if f.severity.lower() == "low")
            info = sum(1 for f in findings if f.severity.lower() in ("info", "informational"))

            result = SecurityScanResult(
                scan_id=scan_id,
                test_id=test_id,
                timestamp=datetime.utcnow(),
                total_packages=packages_count,
                critical_count=critical,
                high_count=high,
                medium_count=medium,
                low_count=low,
                info_count=info,
                findings=findings,
                status="Completed",
            )
            
            self._save_scan_result(result)
            self._add_to_history(result)
            logger.info(f"Async security scan run {scan_id} completed successfully.")

        except Exception as e:
            logger.error(f"Async security scan run {scan_id} failed: {e}", exc_info=True)
            result = SecurityScanResult(
                scan_id=scan_id,
                status="Failed (DefectDojo Offline)",
                error=f"DefectDojo integration failed: {str(e)}",
            )
            self._save_scan_result(result)
            self._add_to_history(result)
            
        finally:
            self.active_scan_id = None
            self._scan_lock.release()

    def _add_to_history(self, result: SecurityScanResult) -> None:
        """Add a scan result metadata entry to history log."""
        history = self.get_history()
        
        # Format timestamp nicely
        ts_str = result.timestamp.strftime("%Y-%m-%d %H:%M:%S UTC")
        
        entry = SecurityHistoryEntry(
            scan_id=result.scan_id,
            timestamp=ts_str,
            packages=result.total_packages,
            critical=result.critical_count,
            high=result.high_count,
            medium=result.medium_count,
            low=result.low_count,
            status=result.status,
        )
        
        # Prepend to make newest first
        history.insert(0, entry)
        self._save_history(history)

    def _count_sbom_packages(self, sbom_path: Optional[str]) -> int:
        """Safely parse SBOM file to count components."""
        if not sbom_path:
            return 0
        try:
            with open(sbom_path, encoding="utf-8") as f:
                data = json.load(f)
            # CycloneDX components plus 1 (for metadata component)
            count = len(data.get("components", []))
            if "metadata" in data and "component" in data["metadata"]:
                count += 1
            return count
        except Exception:
            return 0
