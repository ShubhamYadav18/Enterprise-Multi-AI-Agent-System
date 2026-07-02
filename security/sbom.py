"""
CycloneDX SBOM Generator using cdxgen wrapper.
"""

from __future__ import annotations

import os
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Optional
from security.models import SBOMResult
from security.utils import is_cdxgen_available, is_npm_available
from utils.logger import get_logger

logger = get_logger(__name__)


def generate_sbom(output_path: Optional[str | Path] = None) -> SBOMResult:
    """Scan the repository and generate a CycloneDX JSON SBOM using cdxgen.

    Args:
        output_path: Explicit file path to save the SBOM. Defaults to `sbom/bom.json`.

    Returns:
        SBOMResult with execution metadata.
    """
    start_time = time.time()
    
    # Resolve default output path: project_root/sbom/bom.json
    project_root = Path(__file__).resolve().parent.parent
    if output_path is None:
        output_dir = project_root / "sbom"
        output_file = output_dir / "bom.json"
    else:
        output_file = Path(output_path)
        output_dir = output_file.parent

    # Create target directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Determine execution command
    cmd = []
    if is_cdxgen_available():
        cmd = ["cdxgen"]
        logger.info("cdxgen found globally, executing directly.")
    elif is_npm_available():
        cmd = ["npx", "@cyclonedx/cdxgen"]
        logger.info("cdxgen not found globally, falling back to execution via npx.")
    else:
        error_msg = (
            "cdxgen not found.\n"
            "Please install cdxgen.\n"
            "npm install -g @cyclonedx/cdxgen"
        )
        logger.error("Neither cdxgen nor npm are available on the system PATH.")
        return SBOMResult(
            status="failed",
            timestamp=datetime.utcnow(),
            error=error_msg,
            duration_ms=(time.time() - start_time) * 1000,
        )

    # 2. Add cdxgen CLI parameters
    # -t python targets Python dependency resolution (requirements.txt, lockfiles, virtual environments)
    # -o specifies output destination
    cmd.extend(["-t", "python", "-o", str(output_file)])

    logger.info(f"Running SBOM generation command: {' '.join(cmd)}")

    try:
        # Run command with a timeout to avoid hangs
        res = subprocess.run(
            cmd,
            cwd=str(project_root),
            shell=True,  # Necessary on Windows for finding cmd scripts / npx
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=120,  # 2-minute timeout limit
        )

        duration_ms = (time.time() - start_time) * 1000

        if res.returncode != 0:
            error_details = res.stderr or res.stdout or f"Exit code {res.returncode}"
            logger.error(f"cdxgen SBOM generation failed: {error_details}")
            return SBOMResult(
                status="failed",
                timestamp=datetime.utcnow(),
                error=f"cdxgen exited with code {res.returncode}: {error_details}",
                duration_ms=duration_ms,
            )

        if not output_file.exists() or output_file.stat().st_size == 0:
            logger.error("cdxgen finished but output file was not created or is empty.")
            return SBOMResult(
                status="failed",
                timestamp=datetime.utcnow(),
                error="SBOM file was not generated or is empty.",
                duration_ms=duration_ms,
            )

        logger.info(f"CycloneDX SBOM generated successfully: {output_file} ({output_file.stat().st_size} bytes)")
        return SBOMResult(
            status="success",
            timestamp=datetime.utcnow(),
            output_file=str(output_file),
            duration_ms=duration_ms,
        )

    except subprocess.TimeoutExpired as e:
        duration_ms = (time.time() - start_time) * 1000
        logger.error("cdxgen SBOM generation timed out.")
        return SBOMResult(
            status="failed",
            timestamp=datetime.utcnow(),
            error=f"SBOM generation timed out after 120 seconds: {str(e)}",
            duration_ms=duration_ms,
        )
    except Exception as e:
        duration_ms = (time.time() - start_time) * 1000
        logger.error(f"Failed to execute cdxgen subprocess: {e}", exc_info=True)
        return SBOMResult(
            status="failed",
            timestamp=datetime.utcnow(),
            error=f"Exception during SBOM generation: {str(e)}",
            duration_ms=duration_ms,
        )
