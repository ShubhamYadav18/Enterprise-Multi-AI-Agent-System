"""
Evaluation dataset for LangSmith.

Creates a dataset of test examples covering different query types
and expected behaviors for the multi-agent system.
"""

from __future__ import annotations

from langsmith import Client

from utils.logger import get_logger

logger = get_logger(__name__)

DATASET_NAME = "enterprise-multi-agent-eval"

# Evaluation examples covering all routing scenarios
EVAL_EXAMPLES = [
    {
        "input": {"query": "What is the company leave policy?"},
        "expected_output": {
            "answer_contains": ["20 days", "annual leave", "sick leave"],
            "expected_agents": ["rag_agent"],
            "expected_query_type": "rag_only",
        },
    },
    {
        "input": {"query": "How many sick days do employees get?"},
        "expected_output": {
            "answer_contains": ["12 days", "sick leave"],
            "expected_agents": ["rag_agent"],
            "expected_query_type": "rag_only",
        },
    },
    {
        "input": {"query": "Calculate the growth rate from $1,000,000 to $1,500,000"},
        "expected_output": {
            "answer_contains": ["50%", "growth"],
            "expected_agents": ["calculator_agent"],
            "expected_query_type": "calculator_only",
        },
    },
    {
        "input": {"query": "What percentage is 12 sick days out of 260 working days?"},
        "expected_output": {
            "answer_contains": ["4.6", "%"],
            "expected_agents": ["calculator_agent"],
            "expected_query_type": "calculator_only",
        },
    },
    {
        "input": {"query": "Compare our leave policy with industry standards"},
        "expected_output": {
            "answer_contains": ["leave", "industry", "benchmark"],
            "expected_agents": ["rag_agent", "research_agent"],
            "expected_query_type": "rag_research",
        },
    },
    {
        "input": {"query": "What is the IT security policy regarding MFA?"},
        "expected_output": {
            "answer_contains": ["multi-factor", "MFA", "authentication"],
            "expected_agents": ["rag_agent"],
            "expected_query_type": "rag_only",
        },
    },
    {
        "input": {"query": "How many sick days do we get and what percentage of working days is that?"},
        "expected_output": {
            "answer_contains": ["12", "sick", "percentage"],
            "expected_agents": ["rag_agent", "calculator_agent"],
            "expected_query_type": "rag_calculator",
        },
    },
    {
        "input": {
            "query": "Compare our leave policy with industry standards and calculate annual leave utilization rate"
        },
        "expected_output": {
            "answer_contains": ["leave", "industry", "utilization"],
            "expected_agents": ["rag_agent", "research_agent", "calculator_agent"],
            "expected_query_type": "full",
        },
    },
    {
        "input": {"query": "What are the expense reimbursement procedures?"},
        "expected_output": {
            "answer_contains": ["Concur", "receipt", "reimbursement"],
            "expected_agents": ["rag_agent"],
            "expected_query_type": "rag_only",
        },
    },
    {
        "input": {"query": "Calculate revenue growth for quarterly revenues: $1M, $1.2M, $1.5M, $1.8M"},
        "expected_output": {
            "answer_contains": ["growth", "%"],
            "expected_agents": ["calculator_agent"],
            "expected_query_type": "calculator_only",
        },
    },
]


def create_dataset(client: Client | None = None) -> str:
    """Create or update the evaluation dataset in LangSmith.

    Args:
        client: LangSmith client. Creates one if not provided.

    Returns:
        Dataset name.
    """
    if client is None:
        client = Client()

    # Check if dataset already exists
    try:
        existing = client.read_dataset(dataset_name=DATASET_NAME)
        logger.info(f"Dataset '{DATASET_NAME}' already exists (id={existing.id})")
        return DATASET_NAME
    except Exception:
        pass  # Dataset doesn't exist, create it

    logger.info(f"Creating evaluation dataset: {DATASET_NAME}")

    dataset = client.create_dataset(
        dataset_name=DATASET_NAME,
        description="Evaluation dataset for the Enterprise Multi-Agent AI System. "
                    "Covers RAG, Calculator, Research, and mixed-agent routing scenarios.",
    )

    for example in EVAL_EXAMPLES:
        client.create_example(
            inputs=example["input"],
            outputs=example["expected_output"],
            dataset_id=dataset.id,
        )

    logger.info(
        f"Created dataset '{DATASET_NAME}' with {len(EVAL_EXAMPLES)} examples."
    )

    return DATASET_NAME
