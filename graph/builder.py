"""
LangGraph builder.

Constructs the StateGraph with:
- Conditional routing from the orchestrator
- Parallel execution branches for concurrent agents
- Fan-in to summarizer and validator

Architecture:
    START → orchestrator → route_agents (conditional)
        → [parallel branch: rag / research / calculator]
        → fan_in → summarizer → validator → END
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal, Any, Union

from langgraph.graph import StateGraph, START, END

from graph.state import GraphState
from graph.nodes import (
    orchestrator_node,
    rag_node,
    research_node,
    calculator_node,
    summarizer_node,
    validation_node,
)
from utils.logger import get_logger

logger = get_logger(__name__)


# ============================================================
# Routing Function
# ============================================================

def route_after_orchestrator(
    state: GraphState,
) -> Union[list[str], str]:
    """Determine which agent nodes to execute based on orchestrator routing.

    This function is used with add_conditional_edges to dynamically
    route the graph. When multiple agents are returned, LangGraph
    executes them in parallel (same super-step).

    Args:
        state: Current graph state with routing info from orchestrator.

    Returns:
        List of node names for parallel execution, or single node name.
    """
    agents = state.get("agents_to_invoke", ["rag_agent"])

    # Map agent names to node names
    agent_to_node = {
        "rag_agent": "rag",
        "research_agent": "research",
        "calculator_agent": "calculator",
    }

    nodes_to_execute = []
    for agent in agents:
        node = agent_to_node.get(agent)
        if node:
            nodes_to_execute.append(node)

    if not nodes_to_execute:
        nodes_to_execute = ["rag"]  # Fallback

    logger.info(f"Routing to nodes: {nodes_to_execute} (parallel execution)")

    # Return list for parallel execution
    return nodes_to_execute


# ============================================================
# Graph Builder
# ============================================================

def build_graph() -> StateGraph:
    """Build the multi-agent LangGraph.

    The graph structure:
        START
          │
          ▼
        orchestrator
          │
          ▼ (conditional routing)
        ┌─────────┬──────────┐
        │         │          │
        ▼         ▼          ▼
       rag    research    calculator    (parallel super-step)
        │         │          │
        └─────────┴──────────┘
                  │
                  ▼
              summarizer
                  │
                  ▼
              validator
                  │
                  ▼
                 END

    Returns:
        Compiled StateGraph ready for invocation.
    """
    logger.info("Building multi-agent LangGraph...")

    graph = StateGraph(GraphState)

    # --- Add nodes ---
    graph.add_node("orchestrator", orchestrator_node)
    graph.add_node("rag", rag_node)
    graph.add_node("research", research_node)
    graph.add_node("calculator", calculator_node)
    graph.add_node("summarizer", summarizer_node)
    graph.add_node("validator", validation_node)

    # --- Entry point ---
    graph.add_edge(START, "orchestrator")

    # --- Conditional routing with parallel branches ---
    # When route_after_orchestrator returns multiple nodes,
    # LangGraph executes them in the same super-step (parallel).
    graph.add_conditional_edges(
        "orchestrator",
        route_after_orchestrator,
        {
            "rag": "rag",
            "research": "research",
            "calculator": "calculator",
        },
    )

    # --- Fan-in: all agent nodes converge to summarizer ---
    # The operator.add reducer on agent_outputs merges parallel results
    graph.add_edge("rag", "summarizer")
    graph.add_edge("research", "summarizer")
    graph.add_edge("calculator", "summarizer")

    # --- Sequential: summarizer → validator → END ---
    graph.add_edge("summarizer", "validator")
    graph.add_edge("validator", END)

    logger.info("LangGraph built successfully.")

    return graph


@lru_cache(maxsize=1)
def get_compiled_graph():
    """Get the compiled graph (cached singleton).

    Returns:
        Compiled LangGraph ready for .invoke() or .ainvoke().
    """
    graph = build_graph()
    compiled = graph.compile()
    logger.info("LangGraph compiled and ready.")
    return compiled
