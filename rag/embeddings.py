"""
Embedding model initialization.

Uses HuggingFace sentence-transformers for free, local embeddings.
No API key required — the model is downloaded on first use (~80MB).
"""

from __future__ import annotations

from functools import lru_cache

from langchain_huggingface import HuggingFaceEmbeddings

from config.settings import get_settings
from utils.logger import get_logger

logger = get_logger(__name__)


@lru_cache(maxsize=1)
def get_embeddings() -> HuggingFaceEmbeddings:
    """Get the singleton embedding model instance.

    Returns:
        HuggingFaceEmbeddings configured with the model from settings.
    """
    settings = get_settings()

    logger.info(
        f"Initializing embedding model: {settings.embedding_model} "
        f"on device: {settings.embedding_device}"
    )

    embeddings = HuggingFaceEmbeddings(
        model_name=settings.embedding_model,
        model_kwargs={"device": settings.embedding_device},
        encode_kwargs={"normalize_embeddings": True},
    )

    logger.info("Embedding model loaded successfully.")
    return embeddings
