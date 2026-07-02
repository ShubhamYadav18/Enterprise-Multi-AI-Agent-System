"""
AI Configuration Manifest Generator.

Orchestrates all metadata collectors into a single AIManifest,
serialises it to JSON, and persists it to disk.

Usage:
    from metadata.ai_manifest_generator import generate_manifest, load_manifest

    manifest = generate_manifest()     # always fresh
    manifest = load_manifest()         # read last written manifest from disk
"""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from utils.logger import get_logger
from metadata.ai_manifest_models import AIManifest
from metadata.metadata_collector import (
    collect_application_meta,
    collect_framework_meta,
    collect_llm_config,
    collect_embedding_config,
    collect_rag_config,
    collect_prompt_config,
    collect_agents_meta,
    collect_tools,
    collect_observability_config,
    collect_security_config,
)

logger = get_logger(__name__)

# Project root and canonical output path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_MANIFEST_PATH = _PROJECT_ROOT / "metadata" / "ai_manifest.json"

# Thread lock so concurrent regenerate requests don't corrupt the file
_generate_lock = threading.Lock()


# ============================================================
# Public helpers
# ============================================================

def get_manifest_path() -> Path:
    """Return the canonical path to ai_manifest.json."""
    return _MANIFEST_PATH


def generate_manifest() -> AIManifest:
    """
    Collect all AI configuration metadata and write ai_manifest.json.

    This function is idempotent — calling it multiple times overwrites
    the previous manifest with the latest configuration snapshot.

    Returns:
        The freshly generated AIManifest instance.
    """
    with _generate_lock:
        logger.info("Generating AI Configuration Manifest (AI-BOM)...")

        # ── Collect all sections ──────────────────────────────────
        application = collect_application_meta()
        framework   = collect_framework_meta()
        llm         = collect_llm_config()
        embedding   = collect_embedding_config()
        rag         = collect_rag_config()
        prompts     = collect_prompt_config()
        agents      = collect_agents_meta()
        tools       = collect_tools()
        observability = collect_observability_config()
        security    = collect_security_config()

        # ── Assemble root manifest ────────────────────────────────
        manifest = AIManifest(
            manifest_version="1.0.0",
            schema_version="1.0",
            generated_at=datetime.now(timezone.utc).isoformat(),
            application=application,
            framework=framework,
            llm=llm,
            embedding=embedding,
            rag=rag,
            prompts=prompts,
            agents=agents,
            tools=tools,
            observability=observability,
            security=security,
        )

        # ── Write to disk ─────────────────────────────────────────
        try:
            _MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
            manifest_json = manifest.model_dump(mode="json")
            with open(_MANIFEST_PATH, "w", encoding="utf-8") as f:
                json.dump(manifest_json, f, indent=2, ensure_ascii=False)
            logger.info(f"AI Manifest written to: {_MANIFEST_PATH}")
        except OSError as exc:
            logger.error(f"Failed to write AI Manifest to disk: {exc}")

        return manifest


def load_manifest() -> Optional[AIManifest]:
    """
    Load the last written AI Manifest from disk.

    Returns:
        AIManifest if the file exists, otherwise None.
    """
    if not _MANIFEST_PATH.exists():
        logger.debug("ai_manifest.json not found — call generate_manifest() first.")
        return None

    try:
        with open(_MANIFEST_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return AIManifest(**data)
    except Exception as exc:
        logger.warning(f"Failed to load AI Manifest from disk: {exc}")
        return None


def generate_manifest_background() -> None:
    """
    Trigger manifest generation in a daemon background thread.

    Safe to call at application startup — will never block request serving.
    Errors inside the thread are logged but not propagated.
    """
    def _run() -> None:
        try:
            generate_manifest()
        except Exception as exc:
            logger.error(f"Background manifest generation failed: {exc}", exc_info=True)

    thread = threading.Thread(target=_run, daemon=True, name="ai-manifest-generator")
    thread.start()
    logger.info("AI Manifest generation started in background thread.")
