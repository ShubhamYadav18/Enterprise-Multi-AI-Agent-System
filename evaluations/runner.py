"""
Evaluation runner.

Runs the multi-agent system against the LangSmith evaluation dataset
using all custom evaluators. Results are stored in LangSmith.

Usage:
    python -m evaluations.runner
"""

from __future__ import annotations

import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langsmith import Client, evaluate

from config.settings import get_settings
from evaluations.dataset import create_dataset, DATASET_NAME
from evaluations.evaluators import ALL_EVALUATORS
from graph.builder import get_compiled_graph
from traces.session import create_session
from utils.logger import get_logger, configure_root_logging

logger = get_logger(__name__)


def run_system(inputs: dict[str, Any]) -> dict[str, Any]:
    """Target function for evaluation — runs the full multi-agent pipeline.

    Args:
        inputs: Dict with 'query' key from the evaluation dataset.

    Returns:
        Dict with system outputs for evaluator inspection.
    """
    query = inputs["query"]

    # Create a session for this evaluation run
    session = create_session(query=query, user_id="evaluator")
    config = session.get_runnable_config()

    # Run the graph
    graph = get_compiled_graph()
    initial_state = {
        "query": query,
        "session_id": session.session_id,
        "user_id": "evaluator",
        "conversation_id": session.conversation_id,
        "query_type": "",
        "agents_to_invoke": [],
        "routing_reasoning": "",
        "parallel_groups": [],
        "refined_query": "",
        "agent_outputs": [],
        "summary": "",
        "validation_report": None,
        "final_response": "",
        "execution_metadata": None,
    }

    result = graph.invoke(initial_state, config=config)

    # Determine which agents were invoked vs skipped
    all_agents = {"rag_agent", "research_agent", "calculator_agent"}
    invoked = set(result.get("agents_to_invoke", []))
    skipped = all_agents - invoked

    # Build response dict for evaluators
    agent_outputs_serialized = []
    for ao in result.get("agent_outputs", []):
        if hasattr(ao, "model_dump"):
            agent_outputs_serialized.append(ao.model_dump())
        elif isinstance(ao, dict):
            agent_outputs_serialized.append(ao)

    return {
        "response": result.get("final_response", ""),
        "final_response": result.get("final_response", ""),
        "query_type": result.get("query_type", ""),
        "summary": result.get("summary", ""),
        "validation_report": result.get("validation_report", {}),
        "agent_outputs": agent_outputs_serialized,
        "execution_metadata": {
            "session_id": session.session_id,
            "agents_invoked": list(invoked),
            "agents_skipped": list(skipped),
            "query_type": result.get("query_type", ""),
        },
    }


def run_evaluations(
    dataset_name: str = DATASET_NAME,
    experiment_prefix: str | None = None,
) -> dict[str, Any]:
    """Run evaluations against the dataset.

    Args:
        dataset_name: LangSmith dataset name.
        experiment_prefix: Prefix for the experiment name.

    Returns:
        Summary of evaluation results.
    """
    settings = get_settings()

    if not settings.langchain_api_key:
        logger.warning("LANGCHAIN_API_KEY not set. Cannot run LangSmith evaluations.")
        return {"error": "LANGCHAIN_API_KEY not configured"}

    client = Client()

    # Ensure dataset exists
    create_dataset(client)

    if experiment_prefix is None:
        ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        experiment_prefix = f"multi-agent-eval-{ts}"

    logger.info(f"Starting evaluation run: {experiment_prefix}")
    logger.info(f"Dataset: {dataset_name}")
    logger.info(f"Evaluators: {len(ALL_EVALUATORS)}")

    start_time = time.time()

    results = evaluate(
        run_system,
        data=dataset_name,
        evaluators=ALL_EVALUATORS,
        experiment_prefix=experiment_prefix,
        max_concurrency=1,  # Sequential for reliable tracing
    )

    duration = time.time() - start_time

    logger.info(f"Evaluation complete in {duration:.1f}s")
    logger.info(f"Experiment: {experiment_prefix}")

    return {
        "experiment_name": experiment_prefix,
        "dataset_name": dataset_name,
        "duration_seconds": duration,
        "evaluators_used": [e.__name__ for e in ALL_EVALUATORS],
    }


def main() -> None:
    """CLI entry point for running evaluations."""
    settings = get_settings()
    configure_root_logging(settings.log_level)

    print("=" * 60)
    print("  Enterprise Multi-Agent AI System — Evaluations")
    print("=" * 60)
    print()

    if not settings.langchain_api_key:
        print("ERROR: LANGCHAIN_API_KEY is not set in .env")
        print("Evaluations require a LangSmith account.")
        sys.exit(1)

    if not settings.anthropic_api_key:
        print("ERROR: ANTHROPIC_API_KEY is not set in .env")
        sys.exit(1)

    results = run_evaluations()

    print()
    print("=" * 60)
    print("  Results")
    print("=" * 60)
    for key, value in results.items():
        print(f"  {key}: {value}")
    print()
    print("View detailed results in LangSmith:")
    print(f"  https://smith.langchain.com")
    print("=" * 60)


if __name__ == "__main__":
    main()
