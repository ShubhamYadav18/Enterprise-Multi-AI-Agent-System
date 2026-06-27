"""
Graph state schema.

Defines the TypedDict state that flows through the LangGraph.
Uses Annotated types with reducer functions to merge parallel
branch outputs correctly.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, Optional

from typing_extensions import TypedDict

from utils.models import AgentOutput, ExecutionMetadata, ValidationReport


class GraphState(TypedDict):
    """State schema for the multi-agent LangGraph.

    Fields with Annotated reducers (operator.add) correctly merge
    updates from parallel branches — each branch appends to the list
    rather than overwriting.
    """

    # --- Input ---
    query: str
    session_id: str
    user_id: str
    conversation_id: str

    # --- Routing (set by orchestrator) ---
    query_type: str
    agents_to_invoke: list[str]
    routing_reasoning: str
    parallel_groups: list[list[str]]
    refined_query: str

    # --- Agent outputs (merged from parallel branches via reducer) ---
    agent_outputs: Annotated[list[AgentOutput], operator.add]

    # --- Pipeline outputs ---
    summary: str
    validation_report: Optional[dict[str, Any]]
    final_response: str

    # --- Execution metadata ---
    execution_metadata: Optional[dict[str, Any]]
