"""
FastAPI application for the Enterprise Multi-Agent AI System.

Endpoints:
    POST /chat       — Submit a query and get the full agent pipeline response
    POST /evaluate   — Trigger a LangSmith evaluation run
    GET  /health     — Health check
    GET  /agents     — List available agents
    GET  /documents  — List knowledge base documents
    GET  /trace/{session_id} — Get trace metadata for a session
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from config.settings import get_settings
from graph.builder import get_compiled_graph
from rag.vectorstore import get_vectorstore
from traces.session import create_session
from utils.logger import get_logger
from utils.models import (
    AgentInfo,
    AgentOutput,
    ChatRequest,
    ChatResponse,
    EvaluationRequest,
    EvaluationResponse,
    ExecutionMetadata,
    HealthResponse,
    ValidationReport,
)

logger = get_logger(__name__)

# ============================================================
# FastAPI App
# ============================================================

app = FastAPI(
    title="Enterprise Multi-Agent AI System",
    description="Production-quality multi-agent orchestration with LangGraph, LangSmith tracing, and RAG.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS — allow Streamlit and local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# POST /chat
# ============================================================

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    """Process a user query through the full multi-agent pipeline.

    The orchestrator analyzes intent, routes to appropriate agents
    (with parallel execution where possible), summarizes, validates,
    and returns the final response with full execution metadata.
    """
    start_time = time.time()
    logger.info(f"Chat request: '{request.query[:80]}'")

    try:
        # Create session for trace correlation
        session = create_session(
            query=request.query,
            user_id=request.user_id,
            conversation_id=request.conversation_id,
        )
        config = session.get_runnable_config()

        # Build initial graph state
        initial_state = {
            "query": request.query,
            "session_id": session.session_id,
            "user_id": request.user_id,
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

        # Setup callbacks to capture token usage, latency, and LangSmith root run ID
        from traces.callbacks import AgentTraceCallback
        from langchain_core.callbacks import BaseCallbackHandler
        from uuid import UUID
        from evaluations.real_time import RealTimeEvaluationService

        class RootRunIdCallback(BaseCallbackHandler):
            def __init__(self) -> None:
                super().__init__()
                self.root_run_id: Optional[UUID] = None

            def on_chain_start(
                self,
                serialized: dict[str, Any],
                inputs: dict[str, Any],
                *,
                run_id: UUID,
                **kwargs: Any,
            ) -> None:
                if self.root_run_id is None:
                    self.root_run_id = run_id

        trace_callback = AgentTraceCallback(session_id=session.session_id)
        root_callback = RootRunIdCallback()

        config["callbacks"] = config.get("callbacks", []) + [trace_callback, root_callback]

        # Execute the graph via execute_query wrapper to log session_id as root input/output
        from traces.session import state_var, config_var, evals_var, execute_query
        
        state_token = state_var.set(initial_state)
        config_token = config_var.set(config)
        evals_token = evals_var.set(({}, {}))
        try:
            execute_query(session.session_id)
            result = initial_state
            workflow_evals, agent_evals = evals_var.get()
        finally:
            state_var.reset(state_token)
            config_var.reset(config_token)
            evals_var.reset(evals_token)

        # Compute execution metadata
        total_duration_ms = (time.time() - start_time) * 1000

        all_agents = {"rag_agent", "research_agent", "calculator_agent"}
        invoked = set(result.get("agents_to_invoke", []))
        
        # Build corrected list of invoked agents to include Summarizer and Validator if any domain agent ran
        invoked_list = list(invoked)
        if invoked_list:
            invoked_list.append("summarizer")
            invoked_list.append("validator")
            
        skipped = list(all_agents - invoked)
        parallel_groups = result.get("parallel_groups", [])

        # Count total tool calls across all agents
        agent_outputs: list[AgentOutput] = result.get("agent_outputs", [])
        total_tool_calls = sum(
            len(ao.tool_calls) if hasattr(ao, "tool_calls") else 0
            for ao in agent_outputs
        )

        # Finalize session
        session.finalize(
            query_type=result.get("query_type", ""),
            agents_invoked=invoked_list,
            agents_skipped=skipped,
            parallel_groups=parallel_groups,
            tool_count=total_tool_calls,
        )

        # Build execution metadata
        exec_metadata = ExecutionMetadata(
            session_id=session.session_id,
            conversation_id=session.conversation_id,
            user_id=request.user_id,
            query_type=result.get("query_type", ""),
            agents_invoked=invoked_list,
            agents_skipped=skipped,
            parallel_groups=parallel_groups,
            tool_count=total_tool_calls,
            total_tokens=trace_callback.total_tokens,
            total_duration_ms=total_duration_ms,
            start_time=session.start_time,
            end_time=datetime.now(timezone.utc),
            langsmith_url=session.langsmith_url,
            routing_reasoning=result.get("routing_reasoning", ""),
        )

        # Build validation report
        val_data = result.get("validation_report")
        validation_report = None
        if isinstance(val_data, dict):
            validation_report = ValidationReport(**val_data)

        # Serialize agent outputs
        serialized_outputs = []
        for ao in agent_outputs:
            if isinstance(ao, AgentOutput):
                serialized_outputs.append(ao)
            elif isinstance(ao, dict):
                serialized_outputs.append(AgentOutput(**ao))

        # Real-time evaluations are already run inside the User Query root trace context

        # Compute cost estimation (standard pricing for Claude 3.5 Sonnet)
        cost = (trace_callback.prompt_tokens * 3.0 + trace_callback.completion_tokens * 15.0) / 1_000_000
        execution_summary = {
            "agents_invoked": [agent.replace("_", " ").title() for agent in invoked_list],
            "latency": round(total_duration_ms, 2),
            "tokens": trace_callback.total_tokens,
            "cost": round(cost, 6),
        }

        return ChatResponse(
            session_id=session.session_id,
            query=request.query,
            response=result.get("final_response", ""),
            query_type=result.get("query_type", ""),
            execution_metadata=exec_metadata,
            agent_outputs=serialized_outputs,
            validation_report=validation_report,
            execution_summary=execution_summary,
            workflow_evaluation=workflow_evals,
            agent_evaluations=agent_evals,
            langsmith_trace_url=session.langsmith_url,
        )

    except Exception as e:
        logger.error(f"Chat error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# POST /evaluate
# ============================================================

@app.post("/evaluate", response_model=EvaluationResponse)
async def run_evaluation(request: EvaluationRequest) -> EvaluationResponse:
    """Trigger a LangSmith evaluation run against the evaluation dataset."""
    try:
        from evaluations.runner import run_evaluations

        results = run_evaluations(
            dataset_name=request.dataset_name,
            experiment_prefix=request.experiment_prefix,
        )

        return EvaluationResponse(
            experiment_name=results.get("experiment_name", ""),
            dataset_name=results.get("dataset_name", ""),
            num_examples=10,
            results=[results],
        )

    except Exception as e:
        logger.error(f"Evaluation error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# GET /health
# ============================================================

@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Health check endpoint."""
    try:
        vs = get_vectorstore()
        vectorstore_ready = vs is not None
    except Exception:
        vectorstore_ready = False

    return HealthResponse(
        status="healthy",
        version="1.0.0",
        vectorstore_ready=vectorstore_ready,
        agents_available=[
            "orchestrator",
            "rag_agent",
            "research_agent",
            "calculator_agent",
            "summarizer_agent",
            "validation_agent",
        ],
    )


# ============================================================
# GET /agents
# ============================================================

@app.get("/agents", response_model=list[AgentInfo])
async def list_agents() -> list[AgentInfo]:
    """List all available agents with their descriptions and tools."""
    return [
        AgentInfo(
            name="orchestrator",
            description="Analyzes user intent and routes to specialized agents. Never answers directly.",
            tools=[],
        ),
        AgentInfo(
            name="rag_agent",
            description="Retrieves and synthesizes information from the enterprise knowledge base.",
            tools=["vector_search"],
        ),
        AgentInfo(
            name="research_agent",
            description="Searches for external information, industry benchmarks, and market data.",
            tools=["search"],
        ),
        AgentInfo(
            name="calculator_agent",
            description="Performs mathematical calculations with step-by-step work.",
            tools=[
                "calculate", "percentage", "average", "growth_rate",
                "revenue_growth", "date_difference", "token_counter",
            ],
        ),
        AgentInfo(
            name="summarizer_agent",
            description="Merges outputs from multiple agents into one coherent response.",
            tools=[],
        ),
        AgentInfo(
            name="validation_agent",
            description="Validates responses for grounding, consistency, and hallucination.",
            tools=[],
        ),
    ]


# ============================================================
# GET /documents
# ============================================================

@app.get("/documents")
async def list_documents() -> list[dict[str, Any]]:
    """List all knowledge base documents."""
    settings = get_settings()
    docs_path = settings.documents_dir

    if not docs_path.exists():
        return []

    documents = []
    for md_file in sorted(docs_path.glob("*.md")):
        documents.append({
            "name": md_file.name,
            "title": md_file.stem.replace("_", " ").title(),
            "size_kb": round(md_file.stat().st_size / 1024, 1),
        })

    return documents


# ============================================================
# GET /trace/{session_id}
# ============================================================

@app.get("/trace/{session_id}")
async def get_trace(session_id: str) -> dict[str, Any]:
    """Get trace metadata for a session.

    Returns the LangSmith URL where the full execution trace
    can be inspected.
    """
    settings = get_settings()

    langsmith_url = None
    if settings.langchain_api_key:
        langsmith_url = (
            f"https://smith.langchain.com/projects/"
            f"?filter=metadata.session_id%3D%22{session_id}%22"
        )

    return {
        "session_id": session_id,
        "langsmith_url": langsmith_url,
        "project": settings.langchain_project,
    }
