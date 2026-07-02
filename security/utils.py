"""
Utility functions for Security Module.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any, Optional
from utils.logger import get_logger

logger = get_logger(__name__)


def check_executable(cmd: str) -> bool:
    """Check if a command-line executable exists in the system PATH.

    On Windows, uses shell=True to check for `.cmd`, `.bat` or other wrappers.
    """
    if shutil.which(cmd) is not None:
        return True
    try:
        # Fallback check using subprocess execution
        res = subprocess.run(
            f"{cmd} --version",
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=3,
        )
        return res.returncode == 0
    except Exception:
        return False


def is_cdxgen_available() -> bool:
    """Check if cdxgen is installed and executable."""
    return check_executable("cdxgen")


def is_npm_available() -> bool:
    """Check if npm is installed and executable."""
    return check_executable("npm")


def parse_dependency_tree(sbom_path: str | Path) -> dict[str, Any]:
    """Parse a CycloneDX SBOM file and construct a hierarchical dependency tree.

    Args:
        sbom_path: Path to the bom.json file.

    Returns:
        A dictionary representing the root of the dependency tree.
    """
    path = Path(sbom_path)
    if not path.exists():
        logger.warning(f"SBOM file not found for tree parsing: {path}")
        return {"name": "No SBOM generated", "version": "0.0.0", "dependencies": []}

    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        logger.error(f"Failed to read/parse SBOM JSON: {e}")
        return {"name": "Failed to parse SBOM", "version": "0.0.0", "dependencies": []}

    # 1. Gather all components
    components: dict[str, dict[str, Any]] = {}
    
    # Check if there is a root metadata component
    metadata = data.get("metadata", {})
    root_component = metadata.get("component", {})
    root_ref = root_component.get("bom-ref")
    
    if root_component:
        root_name = root_component.get("name", "project")
        root_version = root_component.get("version", "1.0.0")
        if root_ref:
            components[root_ref] = {
                "name": root_name,
                "version": root_version,
                "dependencies": []
            }
    else:
        root_name = "project"
        root_version = "1.0.0"
        root_ref = "root"
        components[root_ref] = {
            "name": root_name,
            "version": root_version,
            "dependencies": []
        }

    # Parse components list
    for comp in data.get("components", []):
        ref = comp.get("bom-ref")
        if ref:
            components[ref] = {
                "name": comp.get("name", "unknown"),
                "version": comp.get("version", "unknown"),
                "dependencies": []
            }

    # 2. Map dependencies
    adjacency_list: dict[str, list[str]] = {}
    for dep in data.get("dependencies", []):
        ref = dep.get("ref")
        depends_on = dep.get("dependsOn", [])
        if ref:
            adjacency_list[ref] = depends_on

    # If the root reference has no dependency listing but there are others,
    # let's find orphan nodes (nodes that nothing depends on) and attach them to root
    all_children = set()
    for children in adjacency_list.values():
        all_children.update(children)
        
    orphans = []
    for ref in components.keys():
        if ref != root_ref and ref not in all_children:
            orphans.append(ref)

    if root_ref in adjacency_list:
        adjacency_list[root_ref] = list(set(adjacency_list[root_ref] + orphans))
    else:
        adjacency_list[root_ref] = orphans

    # 3. Recursively build the tree
    visited: set[str] = set()

    def build_node(ref: str) -> dict[str, Any]:
        if ref in visited:
            # Handle cycle detection
            comp_info = components.get(ref, {"name": ref, "version": "circular"})
            return {
                "name": comp_info.get("name"),
                "version": comp_info.get("version"),
                "dependencies": [],
                "circular": True,
            }
        
        visited.add(ref)
        comp_info = components.get(ref, {"name": "unknown-ref", "version": "unknown"})
        
        child_refs = adjacency_list.get(ref, [])
        children_nodes = []
        for child_ref in child_refs:
            if child_ref in components:
                children_nodes.append(build_node(child_ref))
                
        visited.remove(ref)
        
        # Sort children alphabetically for consistent presentation
        children_nodes.sort(key=lambda x: x.get("name", ""))
        
        return {
            "name": comp_info.get("name"),
            "version": comp_info.get("version"),
            "dependencies": children_nodes,
        }

    return build_node(root_ref)
