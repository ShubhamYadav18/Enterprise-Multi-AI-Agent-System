"""
Retriever tools for vector store search.

Provides LangChain tools for searching the FAISS knowledge base.
"""

from __future__ import annotations

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool

from rag.vectorstore import similarity_search
from utils.logger import get_logger

logger = get_logger(__name__)


@tool
def vector_search(query: str, k: int = 4, config: RunnableConfig | None = None) -> str:
    """Search the enterprise knowledge base using semantic similarity.

    Searches the FAISS vector store for documents relevant to the query.

    Args:
        query: The search query describing what information to find.
        k: Number of results to return (default 4).

    Returns:
        Retrieved document chunks with sources and relevance scores.
    """
    try:
        results = similarity_search(query, k=k, config=config)
    except Exception as e:
        logger.error(f"Vector search failed: {e}")
        return f"Error performing vector search: {str(e)}"

    if not results:
        return f"No relevant documents found for: '{query}'"

    formatted = []
    for i, (doc, score) in enumerate(results, 1):
        source = doc.metadata.get("source", "unknown")
        chunk_idx = doc.metadata.get("chunk_index", 0)
        # FAISS returns L2 distance; convert to similarity (lower distance = higher similarity)
        similarity = max(0, 1 - (score / 2))

        formatted.append(
            f"[Document {i}]\n"
            f"  Source: {source} (chunk {chunk_idx})\n"
            f"  Relevance: {similarity:.1%}\n"
            f"  Content: {doc.page_content}"
        )

    return "\n\n".join(formatted)


RETRIEVER_TOOLS = [vector_search]
