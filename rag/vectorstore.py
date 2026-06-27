"""
FAISS vector store management.

Handles building, saving, loading, and querying the FAISS index.
Persists to disk so the index doesn't need to be rebuilt on every startup.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.runnables import RunnableConfig
from langsmith import traceable

from config.settings import get_settings
from rag.embeddings import get_embeddings
from rag.document_loader import load_documents, split_documents
from utils.logger import get_logger

logger = get_logger(__name__)

# Module-level reference to the loaded vectorstore
_vectorstore: FAISS | None = None


def build_vectorstore(
    documents: list[Document] | None = None,
    save: bool = True,
) -> FAISS:
    """Build a FAISS vector store from documents.

    Args:
        documents: Pre-split document chunks. If None, loads and splits from disk.
        save: Whether to save the index to disk.

    Returns:
        FAISS vector store instance.
    """
    global _vectorstore
    settings = get_settings()
    embeddings = get_embeddings()

    if documents is None:
        raw_docs = load_documents()
        documents = split_documents(raw_docs)

    if not documents:
        raise ValueError("No documents provided or found. Cannot build vector store.")

    logger.info(f"Building FAISS index from {len(documents)} chunks...")

    vectorstore = FAISS.from_documents(documents, embeddings)

    if save:
        save_path = settings.vectorstore_dir
        save_path.mkdir(parents=True, exist_ok=True)
        vectorstore.save_local(str(save_path))
        logger.info(f"FAISS index saved to: {save_path}")

    _vectorstore = vectorstore
    logger.info("FAISS vector store built successfully.")
    return vectorstore


def load_vectorstore() -> FAISS:
    """Load a previously saved FAISS index from disk.

    Returns:
        FAISS vector store instance.

    Raises:
        FileNotFoundError: If no saved index exists.
    """
    global _vectorstore
    settings = get_settings()
    index_path = settings.vectorstore_dir

    if not (index_path / "index.faiss").exists():
        raise FileNotFoundError(
            f"No FAISS index found at {index_path}. Run ingest.py first."
        )

    embeddings = get_embeddings()

    logger.info(f"Loading FAISS index from: {index_path}")
    vectorstore = FAISS.load_local(
        str(index_path),
        embeddings,
        allow_dangerous_deserialization=True,
    )

    _vectorstore = vectorstore
    logger.info("FAISS index loaded successfully.")
    return vectorstore


def get_vectorstore() -> FAISS:
    """Get the vector store, loading from disk or building if needed.

    Returns:
        FAISS vector store instance (cached).
    """
    global _vectorstore

    if _vectorstore is not None:
        return _vectorstore

    settings = get_settings()

    # Try loading from disk first
    if (settings.vectorstore_dir / "index.faiss").exists():
        return load_vectorstore()

    # Build from scratch if no saved index
    logger.info("No existing index found. Building from documents...")
    return build_vectorstore()


@traceable(name="VectorStoreRetriever", run_type="retriever")
def similarity_search(
    query: str,
    k: int | None = None,
    config: RunnableConfig | None = None,
) -> list[tuple[Document, float]]:
    """Search the vector store for similar documents.

    Args:
        query: Search query text.
        k: Number of results to return. Uses settings default if None.

    Returns:
        List of (Document, score) tuples sorted by relevance.
    """
    settings = get_settings()
    k = k or settings.retrieval_k
    vectorstore = get_vectorstore()

    results = vectorstore.similarity_search_with_score(query, k=k)

    logger.info(
        f"Vector search for '{query[:50]}...' returned {len(results)} results"
    )

    return results
