"""
Pydantic models used across the Enterprise Multi-Agent AI System.

Provides strongly-typed data structures for:
- Agent inputs/outputs
- API requests/responses
- Execution metadata
- Tracing records
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field


# ============================================================
# Retrieved Document
# ============================================================

class RetrievedDocument(BaseModel):
    """A document chunk retrieved from the vector store."""

    content: str = Field(description="Document text content")
    source: str = Field(description="Source filename")
    chunk_index: int = Field(default=0, description="Chunk index within the source")
    relevance_score: float = Field(default=0.0, description="Similarity score (0-1)")

    class Config:
        json_schema_extra = {
            "example": {
                "content": "Annual leave is 20 days per year...",
                "source": "leave_policy.md",
                "chunk_index": 0,
                "relevance_score": 0.89,
            }
        }


# ============================================================
# Tool Call Record
# ============================================================

class ToolCallRecord(BaseModel):
    """Record of a tool invocation."""

    tool_name: str = Field(description="Name of the tool called")
    tool_input: dict[str, Any] = Field(default_factory=dict, description="Input arguments")
    tool_output: Any = Field(default=None, description="Tool output/result")
    duration_ms: float = Field(default=0.0, description="Execution time in milliseconds")


# ============================================================
# Agent Output
# ============================================================

class AgentOutput(BaseModel):
    """Structured output from any agent."""

    agent_name: str = Field(description="Name of the agent that produced this output")
    output: str = Field(default="", description="Agent's textual output")
    confidence: float = Field(default=0.0, description="Confidence score (0-1)")
    sources: list[str] = Field(default_factory=list, description="Source references")
    retrieved_documents: list[RetrievedDocument] = Field(
        default_factory=list, description="Documents retrieved (RAG agent)"
    )
    tool_calls: list[ToolCallRecord] = Field(
        default_factory=list, description="Tools invoked by this agent"
    )
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    error: Optional[str] = Field(default=None, description="Error message if agent failed")


# ============================================================
# Validation Report
# ============================================================

class ValidationReport(BaseModel):
    """Output from the Validation Agent."""

    is_valid: bool = Field(default=True, description="Overall validation pass/fail")
    grounding_score: float = Field(default=0.0, description="How well grounded in sources (0-1)")
    consistency_score: float = Field(default=0.0, description="Internal consistency (0-1)")
    hallucination_score: float = Field(default=0.0, description="Hallucination risk (0=none, 1=high)")
    completeness_score: float = Field(default=0.0, description="Answer completeness (0-1)")
    logic_score: float = Field(default=0.0, description="Logical soundness (0-1)")
    issues: list[str] = Field(default_factory=list, description="Identified issues")
    recommendations: list[str] = Field(default_factory=list, description="Improvement suggestions")


# ============================================================
# Execution Metadata
# ============================================================

class ExecutionMetadata(BaseModel):
    """Metadata about a complete execution run."""

    session_id: str = Field(default="", description="Unique session identifier")
    conversation_id: str = Field(default="", description="Conversation identifier")
    user_id: str = Field(default="demo-user", description="User identifier (mock)")
    query_type: str = Field(default="", description="Classified query type")
    agents_invoked: list[str] = Field(default_factory=list, description="Agents that ran")
    agents_skipped: list[str] = Field(default_factory=list, description="Agents that were skipped")
    parallel_groups: list[list[str]] = Field(
        default_factory=list, description="Groups of agents that ran in parallel"
    )
    tool_count: int = Field(default=0, description="Total tool invocations")
    total_tokens: int = Field(default=0, description="Total tokens used")
    total_duration_ms: float = Field(default=0.0, description="Total execution time (ms)")
    start_time: Optional[datetime] = Field(default=None, description="Execution start time")
    end_time: Optional[datetime] = Field(default=None, description="Execution end time")
    langsmith_url: Optional[str] = Field(default=None, description="LangSmith trace URL")
    routing_reasoning: str = Field(default="", description="Why the orchestrator chose this route")


# ============================================================
# API Models
# ============================================================

class ChatRequest(BaseModel):
    """Request body for POST /chat."""

    query: str = Field(description="User query text")
    user_id: str = Field(default="demo-user", description="User identifier")
    conversation_id: str = Field(default="", description="Conversation identifier")

    class Config:
        json_schema_extra = {
            "example": {
                "query": "What is the company leave policy?",
                "user_id": "demo-user",
                "conversation_id": "conv-001",
            }
        }


class ChatResponse(BaseModel):
    """Response body for POST /chat."""

    session_id: str = Field(description="Session identifier for this request")
    query: str = Field(description="Original query")
    response: str = Field(description="Final response text")
    query_type: str = Field(description="Classified query type")
    execution_metadata: ExecutionMetadata = Field(description="Execution details")
    agent_outputs: list[AgentOutput] = Field(default_factory=list, description="Individual agent outputs")
    validation_report: Optional[ValidationReport] = Field(default=None, description="Validation results")
    execution_summary: Optional[dict[str, Any]] = Field(default=None, description="Token and cost execution summary")
    workflow_evaluation: Optional[dict[str, Any]] = Field(default=None, description="Workflow-level evaluations")
    agent_evaluations: Optional[dict[str, Any]] = Field(default=None, description="Agent-level evaluations")
    langsmith_trace_url: Optional[str] = Field(default=None, description="LangSmith trace URL")



class EvaluationRequest(BaseModel):
    """Request body for POST /evaluate."""

    dataset_name: str = Field(
        default="enterprise-multi-agent-eval",
        description="LangSmith dataset name to evaluate against",
    )
    experiment_prefix: str = Field(
        default="eval-run",
        description="Prefix for the evaluation experiment",
    )


class EvaluationResponse(BaseModel):
    """Response body for POST /evaluate."""

    experiment_name: str = Field(description="Name of the evaluation experiment")
    dataset_name: str = Field(description="Dataset evaluated against")
    num_examples: int = Field(default=0, description="Number of examples evaluated")
    results: list[dict[str, Any]] = Field(default_factory=list, description="Evaluation results summary")


class HealthResponse(BaseModel):
    """Response body for GET /health."""

    status: str = Field(default="healthy")
    version: str = Field(default="1.0.0")
    vectorstore_ready: bool = Field(default=False)
    agents_available: list[str] = Field(default_factory=list)


class AgentInfo(BaseModel):
    """Information about an available agent."""

    name: str = Field(description="Agent name")
    description: str = Field(description="Agent description")
    tools: list[str] = Field(default_factory=list, description="Available tools")
