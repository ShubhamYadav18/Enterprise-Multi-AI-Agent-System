"""
Document ingestion script.

Loads knowledge base documents, splits them into chunks,
embeds them, and builds/saves a FAISS vector index.

Usage:
    python ingest.py
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config.settings import get_settings
from rag.document_loader import load_documents, split_documents
from rag.vectorstore import build_vectorstore
from utils.logger import get_logger, configure_root_logging


def main() -> None:
    """Run the document ingestion pipeline."""
    settings = get_settings()
    configure_root_logging(settings.log_level)
    logger = get_logger(__name__)

    logger.info("=" * 60)
    logger.info("Starting document ingestion...")
    logger.info("=" * 60)

    # Step 1: Load documents
    logger.info(f"Loading documents from: {settings.documents_dir}")
    documents = load_documents()

    if not documents:
        logger.error("No documents found. Please add markdown files to data/documents/")
        sys.exit(1)

    logger.info(f"Loaded {len(documents)} documents.")

    # Step 2: Split into chunks
    chunks = split_documents(documents)
    logger.info(f"Created {len(chunks)} chunks.")

    # Step 3: Build and save FAISS index
    vectorstore = build_vectorstore(documents=chunks, save=True)

    # Verify with a test query
    test_results = vectorstore.similarity_search("leave policy", k=2)
    logger.info(f"Verification search returned {len(test_results)} results.")
    for i, doc in enumerate(test_results):
        logger.info(f"  Result {i+1}: {doc.metadata.get('source', 'unknown')} — {doc.page_content[:80]}...")

    logger.info("=" * 60)
    logger.info("Ingestion complete!")
    logger.info(f"  Documents: {len(documents)}")
    logger.info(f"  Chunks: {len(chunks)}")
    logger.info(f"  Index saved to: {settings.vectorstore_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
