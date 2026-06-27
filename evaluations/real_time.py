"""
Real-time evaluation service for executing metrics during user query requests.
"""

from __future__ import annotations

import datetime
import threading
from typing import Any, Optional
from uuid import UUID

from langsmith import Client

from evaluations.evaluators import (
    get_field,
    answer_correctness_evaluator,
    groundedness_evaluator,
    context_relevance_evaluator,
    task_completion_evaluator,
    tool_selection_evaluator,
    rag_groundedness_evaluator,
    rag_context_relevance_evaluator,
    rag_retriever_quality_evaluator,
    calculator_groundedness_evaluator,
    calculator_hallucination_evaluator,
    calculator_latency_evaluator,
    research_hallucination_evaluator,
    research_latency_evaluator,
    summarizer_consistency_evaluator,
    validator_accuracy_evaluator,
)
from utils.logger import get_logger

logger = get_logger(__name__)


class MockRun:
    """Mock LangSmith Run tree node to reuse offline evaluator logic in real-time."""

    def __init__(
        self,
        name: str,
        inputs: dict[str, Any] | None = None,
        outputs: dict[str, Any] | None = None,
        child_runs: list[MockRun] | None = None,
        start_time: Optional[datetime.datetime] = None,
        end_time: Optional[datetime.datetime] = None,
    ):
        self.name = name
        self.inputs = inputs or {}
        self.outputs = outputs or {}
        self.child_runs = child_runs or []
        self.start_time = start_time or datetime.datetime.now(datetime.timezone.utc)
        self.end_time = end_time or datetime.datetime.now(datetime.timezone.utc)


def run_evaluator_safely(evaluator_fn: Any, run: MockRun, example: Any = None) -> Any:
    """Execute an evaluator safely, returning 'N/A' on exception or 'Not Invoked' on skip."""
    try:
        res = evaluator_fn(run, example)
        if res is not None:
            if res.get("score") is None:
                return "Not Invoked"
            return res.get("score")
        return "N/A"
    except Exception as e:
        logger.error(f"Error running evaluator {evaluator_fn.__name__}: {e}", exc_info=True)
        return "N/A"


def submit_feedback_to_langsmith(root_run_id: str, workflow_evals: dict, agent_evals: dict):
    """Worker function to push real-time evaluations as trace feedback in LangSmith."""
    try:
        client = Client()
        
        # Submit workflow-level feedback
        for key, val in workflow_evals.items():
            if isinstance(val, (int, float)):
                client.create_feedback(
                    run_id=root_run_id,
                    key=key,
                    score=val,
                )

        # Submit agent-level feedback
        for agent_name, metrics in agent_evals.items():
            if isinstance(metrics, dict) and "status" not in metrics:
                for key, val in metrics.items():
                    if isinstance(val, (int, float)):
                        feedback_key = f"{agent_name.lower().replace(' ', '_')}_{key}"
                        client.create_feedback(
                            run_id=root_run_id,
                            key=feedback_key,
                            score=val,
                        )
        logger.info(f"Successfully submitted real-time evaluations feedback for run {root_run_id}")
    except Exception as e:
        logger.warning(f"Failed to submit real-time evaluations feedback to LangSmith: {e}")


class RealTimeEvaluationService:
    """Service to execute evaluations on completed query execution state in real-time."""

    @staticmethod
    def evaluate_run(
        state: dict[str, Any],
        root_run_id: str,
        elapsed_time_ms: float,
        callback: Any,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Run all workflow and agent-level evaluations for the executed query.

        Args:
            state: The final LangGraph state.
            root_run_id: The UUID run identifier of the graph execution trace.
            elapsed_time_ms: Total latency of the run.
            callback: The AgentTraceCallback handler containing node timers.

        Returns:
            Tuple of (workflow_evaluations, agent_evaluations) dicts.
        """
        query = state.get("query", "")
        final_response = state.get("final_response", "")
        invoked = set(state.get("agents_to_invoke", []))

        # 1. Build the mock run tree in memory
        child_runs = []

        # RAG Agent Mock Child Run
        rag_out = None
        for ao in state.get("agent_outputs", []):
            if get_field(ao, "agent_name") == "rag_agent":
                rag_out = ao
                break
        if rag_out:
            rag_dur = callback.node_durations.get("RAG Agent", 0.0)
            start_time = datetime.datetime.now(datetime.timezone.utc)
            end_time = start_time + datetime.timedelta(milliseconds=rag_dur)
            child_runs.append(MockRun(
                name="RAG Agent",
                inputs={"query": query},
                outputs={"agent_outputs": [rag_out]},
                start_time=start_time,
                end_time=end_time,
            ))

        # Research Agent Mock Child Run
        res_out = None
        for ao in state.get("agent_outputs", []):
            if get_field(ao, "agent_name") == "research_agent":
                res_out = ao
                break
        if res_out:
            res_dur = callback.node_durations.get("Research Agent", 0.0)
            start_time = datetime.datetime.now(datetime.timezone.utc)
            end_time = start_time + datetime.timedelta(milliseconds=res_dur)
            child_runs.append(MockRun(
                name="Research Agent",
                inputs={"query": query},
                outputs={"agent_outputs": [res_out]},
                start_time=start_time,
                end_time=end_time,
            ))

        # Calculator Agent Mock Child Run
        calc_out = None
        for ao in state.get("agent_outputs", []):
            if get_field(ao, "agent_name") == "calculator_agent":
                calc_out = ao
                break
        if calc_out:
            calc_dur = callback.node_durations.get("Calculator Agent", 0.0)
            start_time = datetime.datetime.now(datetime.timezone.utc)
            end_time = start_time + datetime.timedelta(milliseconds=calc_dur)
            child_runs.append(MockRun(
                name="Calculator Agent",
                inputs={"query": query},
                outputs={"agent_outputs": [calc_out]},
                start_time=start_time,
                end_time=end_time,
            ))

        # Summarizer Mock Child Run
        summary = state.get("summary", "")
        if summary:
            sum_dur = callback.node_durations.get("Summarizer", 0.0)
            start_time = datetime.datetime.now(datetime.timezone.utc)
            end_time = start_time + datetime.timedelta(milliseconds=sum_dur)
            child_runs.append(MockRun(
                name="Summarizer",
                inputs={"state": {"agent_outputs": state.get("agent_outputs", [])}},
                outputs={"summary": summary},
                start_time=start_time,
                end_time=end_time,
            ))

        # Validator Mock Child Run
        val_report = state.get("validation_report", {})
        if val_report:
            val_dur = callback.node_durations.get("Validator", 0.0)
            start_time = datetime.datetime.now(datetime.timezone.utc)
            end_time = start_time + datetime.timedelta(milliseconds=val_dur)
            child_runs.append(MockRun(
                name="Validator",
                inputs={"state": {"summary": summary, "agent_outputs": state.get("agent_outputs", [])}},
                outputs={"validation_report": val_report, "final_response": final_response},
                start_time=start_time,
                end_time=end_time,
            ))

        # Build execution metadata for tool selection evaluator
        invoked_metadata = list(invoked)
        if invoked_metadata:
            invoked_metadata.append("summarizer")
            invoked_metadata.append("validator")

        root_start = datetime.datetime.now(datetime.timezone.utc)
        root_end = root_start + datetime.timedelta(milliseconds=elapsed_time_ms)

        mock_run = MockRun(
            name="User Query",
            inputs={"query": query},
            outputs={
                "response": final_response,
                "final_response": final_response,
                "execution_metadata": {"agents_invoked": invoked_metadata},
            },
            child_runs=child_runs,
            start_time=root_start,
            end_time=root_end,
        )

        # 2. Run Workflow Evaluations
        workflow_evals = {
            "answer_correctness": run_evaluator_safely(answer_correctness_evaluator, mock_run),
            "groundedness": run_evaluator_safely(groundedness_evaluator, mock_run),
            "context_relevance": run_evaluator_safely(context_relevance_evaluator, mock_run),
            "task_completion": run_evaluator_safely(task_completion_evaluator, mock_run),
            "tool_selection": run_evaluator_safely(tool_selection_evaluator, mock_run),
        }

        # 3. Run Agent Evaluations
        agent_evals = {}

        # RAG evaluations
        if "rag_agent" in invoked:
            agent_evals["RAG Agent"] = {
                "groundedness": run_evaluator_safely(rag_groundedness_evaluator, mock_run),
                "context_relevance": run_evaluator_safely(rag_context_relevance_evaluator, mock_run),
                "retriever_quality": run_evaluator_safely(rag_retriever_quality_evaluator, mock_run),
            }
        else:
            agent_evals["RAG Agent"] = {"status": "Not Invoked"}

        # Calculator evaluations
        if "calculator_agent" in invoked:
            agent_evals["Calculator Agent"] = {
                "groundedness": run_evaluator_safely(calculator_groundedness_evaluator, mock_run),
                "hallucination": run_evaluator_safely(calculator_hallucination_evaluator, mock_run),
                "latency": run_evaluator_safely(calculator_latency_evaluator, mock_run),
            }
        else:
            agent_evals["Calculator Agent"] = {"status": "Not Invoked"}

        # Research evaluations
        if "research_agent" in invoked:
            agent_evals["Research Agent"] = {
                "hallucination": run_evaluator_safely(research_hallucination_evaluator, mock_run),
                "latency": run_evaluator_safely(research_latency_evaluator, mock_run),
            }
        else:
            agent_evals["Research Agent"] = {"status": "Not Invoked"}

        # Summarizer evaluations
        if invoked:
            agent_evals["Summarizer"] = {
                "consistency": run_evaluator_safely(summarizer_consistency_evaluator, mock_run),
            }
        else:
            agent_evals["Summarizer"] = {"status": "Not Invoked"}

        # Validator evaluations
        if invoked:
            agent_evals["Validator"] = {
                "accuracy": run_evaluator_safely(validator_accuracy_evaluator, mock_run),
            }
        else:
            agent_evals["Validator"] = {"status": "Not Invoked"}

        # 4. Asynchronously push to LangSmith trace feedback
        if root_run_id:
            thread = threading.Thread(
                target=submit_feedback_to_langsmith,
                args=(root_run_id, workflow_evals, agent_evals),
                daemon=True,
            )
            thread.start()

        return workflow_evals, agent_evals
