"""
Graph node functions.

Each node wraps an agent call and updates the graph state.
All nodes are @traceable for LangSmith visibility.
"""

from __future__ import annotations

import time
from typing import Any

from langchain_core.runnables import RunnableConfig
from langsmith import traceable

from agents.orchestrator import run_orchestrator
from agents.rag_agent import run_rag_agent
from agents.research_agent import run_research_agent
from agents.calculator_agent import run_calculator_agent
from agents.summarizer_agent import run_summarizer_agent
from agents.validation_agent import run_validation_agent
from graph.state import GraphState
from utils.logger import get_logger, log_agent_event
from utils.models import AgentOutput

logger = get_logger(__name__)


# ============================================================
# Orchestrator Node
# ============================================================

@traceable(name="Orchestrator", run_type="chain")
def orchestrator_node(state: GraphState, config: RunnableConfig) -> dict[str, Any]:
    """Analyze user intent and determine agent routing.

    Updates state with routing decision: which agents to invoke,
    query type, reasoning, and parallel groups.
    """
    query = state["query"]
    session_id = state.get("session_id", "")

    log_agent_event(logger, "node_start", "orchestrator_node", session_id)
    start_time = time.time()

    routing = run_orchestrator(query, config=config)

    duration_ms = (time.time() - start_time) * 1000
    log_agent_event(
        logger, "routing_decision", "orchestrator_node", session_id,
        query_type=routing.get("query_type"),
        agents=routing.get("agents_to_invoke"),
        duration_ms=duration_ms,
    )

    return {
        "query_type": routing.get("query_type", "rag_only"),
        "agents_to_invoke": routing.get("agents_to_invoke", ["rag_agent"]),
        "routing_reasoning": routing.get("reasoning", ""),
        "parallel_groups": routing.get("parallel_groups", []),
        "refined_query": routing.get("refined_query", query),
    }


# ============================================================
# RAG Node
# ============================================================

@traceable(name="RAG Agent", run_type="chain")
def rag_node(state: GraphState, config: RunnableConfig) -> dict[str, Any]:
    """Retrieve and synthesize from the knowledge base."""
    query = state.get("refined_query") or state["query"]
    result = run_rag_agent(query, config=config)

    return {
        "agent_outputs": [result],
    }


# ============================================================
# Research Node
# ============================================================

@traceable(name="Research Agent", run_type="chain")
def research_node(state: GraphState, config: RunnableConfig) -> dict[str, Any]:
    """Search for external information and benchmarks."""
    query = state.get("refined_query") or state["query"]
    result = run_research_agent(query, config=config)

    return {
        "agent_outputs": [result],
    }


# ============================================================
# Calculator Node
# ============================================================

@traceable(name="Calculator Agent", run_type="chain")
def calculator_node(state: GraphState, config: RunnableConfig) -> dict[str, Any]:
    """Perform mathematical calculations."""
    query = state.get("refined_query") or state["query"]
    result = run_calculator_agent(query, config=config)

    return {
        "agent_outputs": [result],
    }


# ============================================================
# Summarizer Node
# ============================================================

@traceable(name="Summarizer", run_type="chain")
def summarizer_node(state: GraphState, config: RunnableConfig) -> dict[str, Any]:
    """Merge all agent outputs into a single response."""
    query = state["query"]
    agent_outputs = state.get("agent_outputs", [])

    summary = run_summarizer_agent(query, agent_outputs, config=config)

    return {
        "summary": summary,
    }


# ============================================================
# Validation Node
# ============================================================

@traceable(name="Validator", run_type="chain")
def validation_node(state: GraphState, config: RunnableConfig) -> dict[str, Any]:
    """Validate the summarized response."""
    query = state["query"]
    summary = state.get("summary", "")
    agent_outputs = state.get("agent_outputs", [])

    report, final_response = run_validation_agent(
        query, summary, agent_outputs, config=config
    )

    return {
        "validation_report": report.model_dump(),
        "final_response": final_response,
    }
