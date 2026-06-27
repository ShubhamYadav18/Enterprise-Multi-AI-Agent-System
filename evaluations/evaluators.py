"""
Custom evaluators for LangSmith evaluations.

Implements 5 evaluators:
1. Task Completion — Did the system produce a final response?
2. Answer Correctness — Does the output contain expected information?
3. Groundedness — Are claims supported by source material?
4. Tool Selection Accuracy — Were the correct agents invoked?
5. Context Relevance — Are retrieved documents relevant?
"""

from __future__ import annotations

from typing import Any


def task_completion_evaluator(run, example) -> dict[str, Any]:
    """Evaluate whether the system produced a complete response.

    Checks:
    - Response is non-empty
    - Response has reasonable length
    - No error messages in output
    """
    outputs = run.outputs or {}
    response = outputs.get("response", "") or outputs.get("final_response", "")

    score = 0.0
    feedback = []

    if response and len(response.strip()) > 20:
        score += 0.5
        feedback.append("Response is non-empty and has reasonable length.")
    else:
        feedback.append("Response is missing or too short.")

    # Check for error indicators
    error_indicators = ["error", "failed", "exception", "could not"]
    has_error = any(ind in response.lower() for ind in error_indicators)
    if not has_error:
        score += 0.3
        feedback.append("No error indicators found.")
    else:
        feedback.append("Response contains error indicators.")

    # Check response has structure
    if any(c in response for c in [".", "\n", "-", "*"]):
        score += 0.2
        feedback.append("Response has proper formatting.")

    return {
        "key": "task_completion",
        "score": min(score, 1.0),
        "comment": " | ".join(feedback),
    }


def answer_correctness_evaluator(run, example) -> dict[str, Any]:
    """Evaluate whether the response contains expected key information.

    Compares the output against expected_output.answer_contains
    to check if key terms are present.
    """
    outputs = run.outputs or {}
    response = (
        outputs.get("response", "") or outputs.get("final_response", "")
    ).lower()

    expected = (example.outputs or {}).get("answer_contains", [])

    if not expected:
        return {"key": "answer_correctness", "score": 1.0, "comment": "No expected terms to check."}

    matches = sum(1 for term in expected if term.lower() in response)
    score = matches / len(expected) if expected else 0.0

    return {
        "key": "answer_correctness",
        "score": score,
        "comment": f"Matched {matches}/{len(expected)} expected terms.",
    }


def groundedness_evaluator(run, example) -> dict[str, Any]:
    """Evaluate whether the response is grounded in source material.

    Checks the validation report's grounding score if available,
    otherwise estimates from source citations.
    """
    outputs = run.outputs or {}

    # Try to get grounding from validation report
    validation = outputs.get("validation_report", {})
    if isinstance(validation, dict) and "grounding_score" in validation:
        score = validation["grounding_score"]
        return {
            "key": "groundedness",
            "score": score,
            "comment": f"Grounding score from validation agent: {score:.2f}",
        }

    # Fallback: check if response mentions sources
    response = outputs.get("response", "") or outputs.get("final_response", "")
    has_sources = any(
        indicator in response.lower()
        for indicator in ["source:", "according to", "based on", ".md", "policy", "handbook"]
    )

    score = 0.8 if has_sources else 0.4
    return {
        "key": "groundedness",
        "score": score,
        "comment": f"Source references {'found' if has_sources else 'not found'} in response.",
    }


def tool_selection_evaluator(run, example) -> dict[str, Any]:
    """Evaluate whether the correct agents were invoked.

    Compares the actually invoked agents against the expected agents
    from the evaluation dataset.
    """
    outputs = run.outputs or {}
    expected_agents = set((example.outputs or {}).get("expected_agents", []))

    # Try to get invoked agents from execution metadata
    metadata = outputs.get("execution_metadata", {})
    if isinstance(metadata, dict):
        actual_agents = set(metadata.get("agents_invoked", []))
    else:
        actual_agents = set()

    if not expected_agents:
        return {"key": "tool_selection", "score": 1.0, "comment": "No expected agents specified."}

    if not actual_agents:
        # Try to infer from query type
        query_type = outputs.get("query_type", "")
        return {
            "key": "tool_selection",
            "score": 0.5,
            "comment": f"Could not determine actual agents. Query type: {query_type}",
        }

    # Calculate overlap score
    correct = expected_agents & actual_agents
    extra = actual_agents - expected_agents
    missing = expected_agents - actual_agents

    precision = len(correct) / len(actual_agents) if actual_agents else 0
    recall = len(correct) / len(expected_agents) if expected_agents else 0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0

    comment_parts = [f"Expected: {expected_agents}", f"Actual: {actual_agents}"]
    if extra:
        comment_parts.append(f"Extra: {extra}")
    if missing:
        comment_parts.append(f"Missing: {missing}")

    return {
        "key": "tool_selection",
        "score": f1,
        "comment": " | ".join(comment_parts),
    }


def context_relevance_evaluator(run, example) -> dict[str, Any]:
    """Evaluate whether retrieved documents are relevant to the query.

    Checks if retrieved document sources align with the query topic.
    """
    outputs = run.outputs or {}
    query = (run.inputs or {}).get("query", "").lower()

    # Try to find agent outputs with retrieved documents
    agent_outputs = outputs.get("agent_outputs", [])

    total_docs = 0
    relevant_docs = 0

    for agent_out in agent_outputs:
        if isinstance(agent_out, dict):
            docs = agent_out.get("retrieved_documents", [])
            for doc in docs:
                total_docs += 1
                if isinstance(doc, dict):
                    score = doc.get("relevance_score", 0)
                    if score > 0.3:  # Threshold for relevance
                        relevant_docs += 1

    if total_docs == 0:
        # No documents retrieved — could be a calculator-only query
        expected_agents = (example.outputs or {}).get("expected_agents", [])
        if "rag_agent" not in expected_agents:
            return {
                "key": "context_relevance",
                "score": 1.0,
                "comment": "No retrieval expected for this query type.",
            }
        return {
            "key": "context_relevance",
            "score": 0.0,
            "comment": "No documents retrieved when retrieval was expected.",
        }

    score = relevant_docs / total_docs
    return {
        "key": "context_relevance",
        "score": score,
        "comment": f"{relevant_docs}/{total_docs} retrieved documents were relevant.",
    }


# All evaluators for easy import
ALL_EVALUATORS = [
    task_completion_evaluator,
    answer_correctness_evaluator,
    groundedness_evaluator,
    tool_selection_evaluator,
    context_relevance_evaluator,
]
