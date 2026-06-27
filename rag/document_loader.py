"""
Document loading and splitting for the knowledge base.

Loads markdown files from the data/documents/ directory,
splits them into chunks suitable for embedding and retrieval.
"""

from __future__ import annotations

from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config.settings import get_settings
from utils.logger import get_logger

logger = get_logger(__name__)


def load_documents(documents_dir: Path | None = None) -> list[Document]:
    """Load all markdown documents from the knowledge base directory.

    Args:
        documents_dir: Path to documents directory. Uses settings default if None.

    Returns:
        List of Document objects with source metadata.
    """
    settings = get_settings()
    docs_path = documents_dir or settings.documents_dir

    if not docs_path.exists():
        logger.warning(f"Documents directory not found: {docs_path}")
        return []

    documents: list[Document] = []

    for md_file in sorted(docs_path.glob("*.md")):
        content = md_file.read_text(encoding="utf-8")
        doc = Document(
            page_content=content,
            metadata={
                "source": md_file.name,
                "source_path": str(md_file),
                "title": md_file.stem.replace("_", " ").title(),
            },
        )
        documents.append(doc)
        logger.info(f"Loaded document: {md_file.name} ({len(content)} chars)")

    logger.info(f"Total documents loaded: {len(documents)}")
    return documents


def split_documents(documents: list[Document]) -> list[Document]:
    """Split documents into chunks for embedding.

    Uses RecursiveCharacterTextSplitter with markdown-aware separators
    to preserve document structure as much as possible.

    Args:
        documents: List of full documents to split.

    Returns:
        List of document chunks with preserved metadata + chunk index.
    """
    settings = get_settings()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=[
            "\n## ",   # H2 headers first
            "\n### ",  # H3 headers
            "\n#### ", # H4 headers
            "\n\n",    # Double newlines (paragraphs)
            "\n",      # Single newlines
            ". ",      # Sentences
            " ",       # Words
        ],
        length_function=len,
    )

    chunks = splitter.split_documents(documents)

    # Add chunk index to metadata
    source_chunk_counts: dict[str, int] = {}
    for chunk in chunks:
        source = chunk.metadata.get("source", "unknown")
        idx = source_chunk_counts.get(source, 0)
        chunk.metadata["chunk_index"] = idx
        source_chunk_counts[source] = idx + 1

    logger.info(
        f"Split {len(documents)} documents into {len(chunks)} chunks "
        f"(chunk_size={settings.chunk_size}, overlap={settings.chunk_overlap})"
    )

    return chunks
