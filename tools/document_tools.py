"""
Document management tools.

Provides tools to list and load knowledge base documents.
"""

from __future__ import annotations

from langchain_core.tools import tool

from config.settings import get_settings
from utils.logger import get_logger

logger = get_logger(__name__)


@tool
def list_documents() -> str:
    """List all available documents in the enterprise knowledge base.

    Returns:
        A formatted list of document names and their sizes.
    """
    settings = get_settings()
    docs_path = settings.documents_dir

    if not docs_path.exists():
        return "No documents directory found."

    files = sorted(docs_path.glob("*.md"))

    if not files:
        return "No documents found in the knowledge base."

    formatted = ["Available Knowledge Base Documents:", ""]
    for i, f in enumerate(files, 1):
        size_kb = f.stat().st_size / 1024
        title = f.stem.replace("_", " ").title()
        formatted.append(f"  {i}. {title} ({f.name}) — {size_kb:.1f} KB")

    formatted.append(f"\nTotal: {len(files)} documents")
    return "\n".join(formatted)


@tool
def load_document(name: str) -> str:
    """Load the full content of a specific knowledge base document.

    Args:
        name: Document filename (e.g., 'leave_policy.md') or stem (e.g., 'leave_policy').

    Returns:
        The full document content.
    """
    settings = get_settings()
    docs_path = settings.documents_dir

    # Try exact filename first
    doc_path = docs_path / name
    if not doc_path.exists():
        # Try adding .md extension
        doc_path = docs_path / f"{name}.md"

    if not doc_path.exists():
        available = [f.name for f in docs_path.glob("*.md")]
        return (
            f"Document '{name}' not found.\n"
            f"Available documents: {', '.join(available)}"
        )

    content = doc_path.read_text(encoding="utf-8")
    return f"Document: {doc_path.name}\n\n{content}"


DOCUMENT_TOOLS = [list_documents, load_document]
