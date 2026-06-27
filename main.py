"""
Main entry point for the Enterprise Multi-Agent AI System.

Starts the FastAPI server with Uvicorn.

Usage:
    python main.py
"""

import sys
from pathlib import Path

# Ensure project root is in path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import uvicorn

from config.settings import get_settings
from rag.vectorstore import get_vectorstore
from utils.logger import get_logger, configure_root_logging


def main() -> None:
    """Initialize the system and start the API server."""
    settings = get_settings()
    configure_root_logging(settings.log_level)
    logger = get_logger(__name__)

    logger.info("=" * 60)
    logger.info("Enterprise Multi-Agent AI System — Starting")
    logger.info("=" * 60)

    # Validate required configuration
    if not settings.anthropic_api_key:
        logger.error("ANTHROPIC_API_KEY is not set. Please configure .env file.")
        print("\n  ERROR: ANTHROPIC_API_KEY is not set.")
        print("  Copy .env.example to .env and add your API key.\n")
        sys.exit(1)

    # Initialize vector store (build if needed)
    try:
        logger.info("Initializing vector store...")
        vs = get_vectorstore()
        logger.info("Vector store ready.")
    except Exception as e:
        logger.warning(f"Vector store initialization failed: {e}")
        logger.info("Run 'python ingest.py' to build the vector store.")

    # Log configuration summary
    logger.info(f"  Model: {settings.anthropic_model}")
    logger.info(f"  LangSmith Tracing: {settings.langchain_tracing_v2}")
    logger.info(f"  LangSmith Project: {settings.langchain_project}")
    logger.info(f"  Tavily Search: {'enabled' if settings.has_tavily else 'disabled (using mock)'}")
    logger.info(f"  API: http://{settings.api_host}:{settings.api_port}")
    logger.info(f"  Docs: http://{settings.api_host}:{settings.api_port}/docs")

    # Start the server
    uvicorn.run(
        "api.app:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=False,
        log_level="info",
    )


if __name__ == "__main__":
    main()
