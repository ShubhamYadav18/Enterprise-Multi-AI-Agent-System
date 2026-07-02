"""
DefectDojo REST API Client.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional
import requests
from security.models import VulnerabilityFinding
from utils.logger import get_logger

logger = get_logger(__name__)


class DefectDojoClient:
    """Client for interacting with DefectDojo API v2."""

    def __init__(
        self,
        url: str,
        api_key: str,
        product_identifier: str,
        engagement_identifier: str,
    ) -> None:
        """Initialize the client.

        Args:
            url: Base URL of DefectDojo (e.g. http://localhost:8080)
            api_key: API authorization token
            product_identifier: Product ID (int) or Product Name (str)
            engagement_identifier: Engagement ID (int) or Engagement Name (str)
        """
        # Ensure URL doesn't have trailing slash
        self.base_url = url.rstrip("/")
        self.api_key = api_key
        self.product_identifier = product_identifier
        self.engagement_identifier = engagement_identifier
        
        self.headers = {
            "Authorization": f"Token {self.api_key}",
        }

    def _get(self, endpoint: str, params: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        """Perform a GET request to DefectDojo API."""
        url = f"{self.base_url}/api/v2/{endpoint.lstrip('/')}"
        logger.info(f"DefectDojo GET request to: {url} with params {params}")
        
        response = requests.get(url, headers=self.headers, params=params, timeout=30)
        
        if response.status_code not in (200, 201):
            logger.error(f"DefectDojo GET {endpoint} failed ({response.status_code}): {response.text}")
            response.raise_for_status()
            
        return response.json()

    def _post(self, endpoint: str, json_data: Optional[dict[str, Any]] = None, files: Optional[dict[str, Any]] = None, data: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        """Perform a POST request to DefectDojo API."""
        url = f"{self.base_url}/api/v2/{endpoint.lstrip('/')}"
        
        # If files are present, header content-type will be multipart/form-data (requests handles this)
        headers = self.headers.copy()
        
        logger.info(f"DefectDojo POST request to: {url}")
        
        if files:
            response = requests.post(url, headers=headers, files=files, data=data, timeout=60)
        else:
            response = requests.post(url, headers=headers, json=json_data, timeout=30)
            
        if response.status_code not in (200, 201):
            logger.error(f"DefectDojo POST {endpoint} failed ({response.status_code}): {response.text}")
            response.raise_for_status()
            
        return response.json()

    def get_product_id(self) -> int:
        """Resolve product ID from name or return it directly if integer."""
        ident = str(self.product_identifier).strip()
        if ident.isdigit():
            return int(ident)
            
        # Search by product name
        logger.info(f"Resolving Product ID for product name: '{ident}'")
        res = self._get("products/", params={"name": ident})
        results = res.get("results", [])
        
        if not results:
            raise ValueError(f"Product '{ident}' not found in DefectDojo. Please create it first.")
            
        product_id = results[0]["id"]
        logger.info(f"Resolved Product ID: {product_id}")
        return product_id

    def get_engagement_id(self, product_id: int) -> int:
        """Resolve engagement ID or create it if name is not found."""
        ident = str(self.engagement_identifier).strip()
        if ident.isdigit():
            return int(ident)

        # Search by engagement name and product ID
        logger.info(f"Resolving Engagement ID for engagement: '{ident}' under Product ID: {product_id}")
        res = self._get("engagements/", params={"name": ident, "product": product_id})
        results = res.get("results", [])
        
        if results:
            engagement_id = results[0]["id"]
            logger.info(f"Resolved existing Engagement ID: {engagement_id}")
            return engagement_id
            
        # If not found, create a new engagement
        logger.info(f"Engagement '{ident}' not found. Creating a new engagement under Product ID {product_id}...")
        
        today = datetime.now()
        start_date = today.strftime("%Y-%m-%d")
        end_date = (today + timedelta(days=90)).strftime("%Y-%m-%d")
        
        payload = {
            "name": ident,
            "product": product_id,
            "target_start": start_date,
            "target_end": end_date,
            "status": "In Progress",
            "engagement_type": "CI/CD",
            "active": True,
        }
        
        created = self._post("engagements/", json_data=payload)
        new_engagement_id = created["id"]
        logger.info(f"Created new engagement '{ident}' with ID: {new_engagement_id}")
        return new_engagement_id

    def upload_sbom(self, file_path: str | Path, engagement_id: int) -> dict[str, Any]:
        """Upload CycloneDX SBOM to DefectDojo.

        Args:
            file_path: Path to the bom.json file
            engagement_id: Targeting DefectDojo engagement ID

        Returns:
            JSON response from the scan import endpoint.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"SBOM file not found at {path}")

        logger.info(f"Uploading SBOM {path.name} to DefectDojo engagement {engagement_id}...")

        # Multipart form parameters
        data = {
            "scan_type": "CycloneDX Scan",
            "engagement": str(engagement_id),
            "close_old_findings": "true",
            "active": "true",
            "verified": "true",
            "minimum_severity": "Info",
        }

        with open(path, "rb") as f:
            files = {
                "file": (path.name, f, "application/json"),
            }
            res = self._post("import-scan/", files=files, data=data)

        logger.info(f"DefectDojo upload successful. Test ID: {res.get('test')}")
        return res

    def get_findings(self, test_id: int) -> list[VulnerabilityFinding]:
        """Retrieve vulnerability findings for a specific test ID.

        Handles pagination automatically.

        Args:
            test_id: DefectDojo test ID.

        Returns:
            List of VulnerabilityFinding objects.
        """
        logger.info(f"Retrieving findings for test ID: {test_id}...")
        findings: list[VulnerabilityFinding] = []
        
        endpoint = "findings/"
        params: dict[str, Any] = {
            "test": test_id,
            "limit": 100,
            "offset": 0,
        }

        while True:
            res = self._get(endpoint, params=params)
            results = res.get("results", [])
            
            for item in results:
                # Resolve CVE field safely
                cve_val = "N/A"
                # Sometimes CVE is a simple string, sometimes it's inside vulnerability_ids list
                vuln_ids = item.get("vulnerability_ids", [])
                if vuln_ids:
                    cve_val = vuln_ids[0].get("vulnerability_id", "N/A") if isinstance(vuln_ids[0], dict) else str(vuln_ids[0])
                elif item.get("cve"):
                    cve_val = str(item.get("cve"))

                # Create finding model
                finding = VulnerabilityFinding(
                    id=item.get("id"),
                    package=item.get("component_name") or "unknown",
                    version=item.get("component_version") or "unknown",
                    severity=item.get("severity", "Info"),
                    cve=cve_val,
                    description=item.get("description") or "No description provided.",
                    recommendation=item.get("mitigation") or "No recommendation provided.",
                    status="Active" if item.get("active") else "Inactive",
                )
                findings.append(finding)

            # Check pagination
            next_url = res.get("next")
            if not next_url:
                break
                
            # Increment offset
            params["offset"] += params["limit"]

        logger.info(f"Retrieved {len(findings)} findings in total for test ID: {test_id}")
        return findings
