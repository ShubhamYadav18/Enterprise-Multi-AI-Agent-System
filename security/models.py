"""
Pydantic models for Security and Software Supply Chain modules.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional
from pydantic import BaseModel, Field


class SBOMResult(BaseModel):
    """Result of CycloneDX SBOM generation."""

    status: str = Field(description="Generation status (e.g., 'success', 'failed')")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Time of generation")
    output_file: Optional[str] = Field(default=None, description="Path to the generated JSON SBOM file")
    duration_ms: float = Field(default=0.0, description="Time taken to generate in milliseconds")
    error: Optional[str] = Field(default=None, description="Error message if generation failed")


class VulnerabilityFinding(BaseModel):
    """A single vulnerability finding parsed from DefectDojo."""

    id: Optional[int] = Field(default=None, description="DefectDojo finding ID")
    package: str = Field(description="Name of the vulnerable package/component")
    version: str = Field(description="Version of the vulnerable package")
    severity: str = Field(description="Severity (Critical, High, Medium, Low, Info)")
    cve: str = Field(description="CVE identifier or vulnerability code")
    description: str = Field(description="Detailed vulnerability description")
    recommendation: str = Field(description="Recommendation or mitigation steps")
    status: str = Field(description="Finding status (e.g., 'Active', 'Verified', 'Inactive')")


class SecurityScanResult(BaseModel):
    """Complete summary of a security scan execution."""

    scan_id: str = Field(description="Unique scan run identifier")
    test_id: Optional[int] = Field(default=None, description="DefectDojo test ID")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Scan execution timestamp")
    total_packages: int = Field(default=0, description="Total count of packages in the SBOM")
    critical_count: int = Field(default=0, description="Number of critical vulnerabilities")
    high_count: int = Field(default=0, description="Number of high vulnerabilities")
    medium_count: int = Field(default=0, description="Number of medium vulnerabilities")
    low_count: int = Field(default=0, description="Number of low vulnerabilities")
    info_count: int = Field(default=0, description="Number of informational vulnerabilities")
    findings: list[VulnerabilityFinding] = Field(default_factory=list, description="List of findings")
    status: str = Field(description="Overall scan status (e.g., 'Completed', 'Failed')")
    error: Optional[str] = Field(default=None, description="Scan error message if applicable")


class SecurityHistoryEntry(BaseModel):
    """A summary entry for scan history representation."""

    scan_id: str = Field(description="Scan identifier")
    timestamp: str = Field(description="Formatted scan timestamp")
    packages: int = Field(description="Total packages scanned")
    critical: int = Field(description="Critical findings count")
    high: int = Field(description="High findings count")
    medium: int = Field(description="Medium findings count")
    low: int = Field(description="Low findings count")
    status: str = Field(description="Status of the scan")
