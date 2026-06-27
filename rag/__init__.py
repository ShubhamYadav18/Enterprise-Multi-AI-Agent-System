"""RAG (Retrieval-Augmented Generation) module."""

from rag.embeddings import get_embeddings
from rag.document_loader import load_documents, split_documents
from rag.vectorstore import get_vectorstore, build_vectorstore

__all__ = [
    "get_embeddings",
    "load_documents",
    "split_documents",
    "get_vectorstore",
    "build_vectorstore",
]
