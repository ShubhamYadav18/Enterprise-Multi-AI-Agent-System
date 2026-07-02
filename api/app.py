"""
FastAPI application for the Enterprise Multi-Agent AI System.

Endpoints:
    POST /chat                    — Submit a query and get the full agent pipeline response
    POST /evaluate                — Trigger a LangSmith evaluation run
    GET  /health                  — Health check
    GET  /agents                  — List available agents
    GET  /documents               — List knowledge base documents
    GET  /trace/{session_id}      — Get trace metadata for a session
    GET  /monitoring/status       — Monitoring configuration and health status
    GET  /monitoring/result/{id}  — Monitoring result for a specific session
    GET  /monitoring/history      — All stored monitoring results
    POST /monitoring/test         — Send a test metric event to verify connectivity
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager
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

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan — generate AI manifest on startup."""
    try:
        from metadata.ai_manifest_generator import generate_manifest_background
        generate_manifest_background()
    except Exception as exc:
        logger.warning(f"AI Manifest startup generation skipped: {exc}")
    yield  # application runs here


app = FastAPI(
    title="Enterprise Multi-Agent AI System",
    description="Production-quality multi-agent orchestration with LangGraph, LangSmith tracing, and RAG.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
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

        # Setup callbacks to capture token usage and latency
        from traces.callbacks import AgentTraceCallback
        from evaluations.real_time import RealTimeEvaluationService

        trace_callback = AgentTraceCallback(session_id=session.session_id)
        config["callbacks"] = config.get("callbacks", []) + [trace_callback]

        # Execute the graph via execute_query wrapper to log session_id as root input/output
        from traces.session import state_var, config_var, evals_var, trace_id_var, execute_query
        
        state_token = state_var.set(initial_state)
        config_token = config_var.set(config)
        evals_token = evals_var.set(({}, {}))
        try:
            execute_query(session.session_id)
            result = initial_state
            workflow_evals, agent_evals = evals_var.get()
            root_run_id_str = trace_id_var.get()
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
            sbom_version=session.sbom_version,
            security_scan_timestamp=session.security_scan_timestamp,
            security_status=session.security_status,
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

        # Compute tokens and cost (fallback to local if LangSmith fails)
        total_tokens = trace_callback.total_tokens
        cost = (trace_callback.prompt_tokens * 3.0 + trace_callback.completion_tokens * 15.0) / 1_000_000

        if root_run_id_str:
            try:
                from langsmith import Client
                import time
                ls_client = Client()
                # Give LangSmith backend a moment to aggregate token usage
                time.sleep(2.0)
                runs = list(ls_client.list_runs(trace_id=root_run_id_str))
                
                ls_total_tokens = 0
                ls_prompt_tokens = 0
                ls_completion_tokens = 0
                for r in runs:
                    if r.run_type == "llm":
                        ls_prompt_tokens += getattr(r, "prompt_tokens", 0)
                        ls_completion_tokens += getattr(r, "completion_tokens", 0)
                        ls_total_tokens += getattr(r, "total_tokens", 0)
                
                if ls_prompt_tokens > 0 or ls_completion_tokens > 0:
                    if ls_total_tokens == 0:
                        ls_total_tokens = ls_prompt_tokens + ls_completion_tokens
                    total_tokens = ls_total_tokens
                    cost = (ls_prompt_tokens * 3.0 + ls_completion_tokens * 15.0) / 1_000_000
                    
            except Exception:
                pass

        execution_summary = {
            "agents_invoked": [agent.replace("_", " ").title() for agent in invoked_list],
            "latency": round(total_duration_ms, 2),
            "tokens": total_tokens,
            "cost": round(cost, 6),
        }

        chat_response = ChatResponse(
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

        # ── Fire-and-forget monitoring pipeline ─────────────────
        # Never blocks the response. Failures are swallowed here.
        try:
            from monitoring.monitoring_service import get_monitoring_service
            get_monitoring_service().process_session_async(
                session_id=session.session_id,
                query=request.query,
                latency_ms=total_duration_ms,
                total_tokens=total_tokens,
                cost=cost,
                workflow_evals=workflow_evals,
                agent_evals=agent_evals,
                agents_invoked=invoked_list,
                conversation_id=session.conversation_id,
                root_run_id=root_run_id_str or "",
                langsmith_url=session.langsmith_url,
                input_tokens=getattr(trace_callback, "prompt_tokens", 0),
                output_tokens=getattr(trace_callback, "completion_tokens", 0),
            )
        except Exception as _mon_err:
            logger.debug(f"Monitoring hook skipped: {_mon_err}")
        # ────────────────────────────────────────────────────────

        return chat_response

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


# ============================================================
# Security API Endpoints
# ============================================================

from security.service import SecurityService

@app.post("/api/security/sbom")
async def generate_sbom_endpoint() -> dict:
    """Generate a CycloneDX SBOM for the current project (does NOT upload to DefectDojo)."""
    try:
        from security.sbom import generate_sbom
        result = generate_sbom()
        return {
            "status": result.status,
            "output_file": result.output_file,
            "timestamp": str(result.timestamp),
            "duration_ms": result.duration_ms,
            "error": result.error,
        }
    except Exception as e:
        logger.error(f"Failed to generate SBOM: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/security/upload")
async def trigger_security_scan() -> dict:
    """Generate SBOM, upload to DefectDojo, and retrieve findings asynchronously."""
    try:
        service = SecurityService()
        scan_id = service.run_scan_async()
        return {"status": "success", "scan_id": scan_id}
    except Exception as e:
        logger.error(f"Failed to trigger scan: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/security/summary")
async def get_security_summary() -> dict:
    """Get high-level security summary (from the latest scan)."""
    try:
        service = SecurityService()
        result = service.get_latest_scan_result()
        if not result:
            return {"status": "No scan run yet"}
        return result.model_dump()
    except Exception as e:
        logger.error(f"Failed to fetch summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/security/findings")
async def get_security_findings(scan_id: Optional[str] = None) -> list:
    """Get vulnerabilities for a specific scan ID, or the latest if omitted."""
    try:
        service = SecurityService()
        if scan_id:
            result = service.get_scan_result(scan_id)
        else:
            result = service.get_latest_scan_result()
            
        if not result:
            return []
            
        return [f.model_dump() for f in result.findings]
    except Exception as e:
        logger.error(f"Failed to fetch findings: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/security/dependencies")
async def get_dependency_tree() -> dict:
    """Get the parsed dependency tree from the generated SBOM."""
    try:
        service = SecurityService()
        return service.get_dependency_tree()
    except Exception as e:
        logger.error(f"Failed to fetch dependencies: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/security/history")
async def get_security_history() -> list:
    """Get the history of all security scans."""
    try:
        service = SecurityService()
        return [h.model_dump() for h in service.get_history()]
    except Exception as e:
        logger.error(f"Failed to fetch history: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================
# AI Manifest Endpoints
# ============================================================

@app.get("/metadata/manifest")
async def get_ai_manifest() -> dict:
    """
    Return the current AI Configuration Manifest (AI-BOM).

    If no manifest has been generated yet, generates one synchronously.
    """
    try:
        from metadata.ai_manifest_generator import load_manifest, generate_manifest
        manifest = load_manifest()
        if manifest is None:
            logger.info("No cached manifest found — generating synchronously.")
            manifest = generate_manifest()
        return manifest.model_dump(mode="json")
    except Exception as exc:
        logger.error(f"Failed to retrieve AI manifest: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/metadata/regenerate")
async def regenerate_ai_manifest() -> dict:
    """
    Force-regenerate the AI Configuration Manifest and return the updated version.

    Use this after changing model, RAG settings, prompts, or any other
    AI configuration to get a fresh snapshot.
    """
    try:
        from metadata.ai_manifest_generator import generate_manifest
        manifest = generate_manifest()
        return {
            "status": "regenerated",
            "generated_at": manifest.generated_at,
            "manifest": manifest.model_dump(mode="json"),
        }
    except Exception as exc:
        logger.error(f"Failed to regenerate AI manifest: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


# ============================================================
# Monitoring API Endpoints
# ============================================================

@app.get("/monitoring/status")
async def get_monitoring_status() -> dict:
    """
    Return monitoring configuration and connectivity health.

    Used by the Streamlit dashboard to render the 📡 Monitoring Config panel.
    """
    try:
        settings = get_settings()
        from monitoring.publisher import get_publisher
        from incident.manager import get_incident_manager

        publisher = get_publisher()
        incident_mgr = get_incident_manager()

        # Datadog health (only if enabled — avoid network call when disabled)
        dd_healthy: Optional[bool] = None
        if settings.datadog_enabled and settings.datadog_api_key:
            try:
                dd_healthy = publisher.health_check()
            except Exception:
                dd_healthy = False

        # ServiceNow health (only if enabled)
        sn_healthy: Optional[bool] = None
        if settings.servicenow_enabled and settings.servicenow_instance_url:
            try:
                sn_healthy = incident_mgr.provider.health_check()
            except Exception:
                sn_healthy = False

        return {
            "datadog": {
                "enabled": settings.datadog_enabled,
                "site": settings.datadog_site,
                "metric_prefix": settings.datadog_metric_prefix,
                "api_key_configured": bool(settings.datadog_api_key),
                "app_key_configured": bool(settings.datadog_app_key),
                "healthy": dd_healthy,
            },
            "incident": {
                "provider": settings.incident_provider,
                "servicenow_enabled": settings.servicenow_enabled,
                "servicenow_instance_url": settings.servicenow_instance_url or None,
                "healthy": sn_healthy,
            },
            "thresholds": {
                "groundedness_min": settings.groundedness_min,
                "hallucination_max": settings.hallucination_max,
                "latency_max_ms": settings.latency_max_ms,
                "total_cost_max": settings.total_cost_max,
                "token_limit": settings.token_limit,
            },
        }
    except Exception as exc:
        logger.error(f"Monitoring status error: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/monitoring/result/{session_id}")
async def get_monitoring_result(session_id: str) -> dict:
    """Return the monitoring result for a specific session."""
    try:
        from monitoring.monitoring_service import get_monitoring_result
        result = get_monitoring_result(session_id)
        if result is None:
            return {"found": False, "session_id": session_id}
        return {"found": True, **result.model_dump(mode="json")}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/monitoring/history")
async def get_monitoring_history() -> list:
    """Return all stored monitoring results (latest 200 sessions)."""
    try:
        from monitoring.monitoring_service import get_all_monitoring_results
        results = get_all_monitoring_results()
        return [
            {"session_id": k, **v.model_dump(mode="json")}
            for k, v in list(results.items())[-50:]
        ]
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/monitoring/test")
async def test_monitoring() -> dict:
    """
    Send a synthetic test metric event through the full monitoring pipeline.

    Uses realistic but clearly synthetic data so no real threshold is breached.
    Returns the full SessionMonitoringResult so the Streamlit test button
    can display connectivity results.
    """
    import uuid
    from monitoring.monitoring_service import MonitoringService
    from monitoring.metrics import MetricBuilder
    from monitoring.policy_engine import PolicyEngine
    from monitoring.publisher import get_publisher
    from monitoring.models import PolicyStatus

    test_session_id = f"test-{uuid.uuid4().hex[:8]}"

    try:
        # Build a synthetic event that should be HEALTHY
        event = MetricBuilder.build(
            session_id=test_session_id,
            query="[TEST] Monitoring connectivity check",
            latency_ms=1200.0,
            total_tokens=500,
            cost=0.001,
            workflow_evals={
                "groundedness": 0.95,
                "context_relevance": 0.90,
                "answer_correctness": 0.92,
                "task_completion": 0.88,
                "tool_selection": 1.0,
            },
            agent_evals={
                "RAG Agent": {
                    "groundedness": 0.95,
                    "context_relevance": 0.90,
                    "retriever_quality": 0.88,
                },
            },
            agents_invoked=["rag_agent"],
            root_run_id="test-run-id",
        )

        # Test Datadog publish
        publisher = get_publisher()
        pub_result = publisher.publish(event)

        # Test Policy Engine
        policy_result = PolicyEngine().evaluate(event)

        # Test provider health
        settings = get_settings()
        dd_healthy = publisher.health_check() if settings.datadog_enabled else None
        from incident.manager import get_incident_manager
        inc_mgr = get_incident_manager()
        sn_healthy = inc_mgr.provider.health_check() if settings.servicenow_enabled else None

        return {
            "test_session_id": test_session_id,
            "datadog": {
                "enabled": settings.datadog_enabled,
                "published": pub_result.success,
                "metrics_count": pub_result.metrics_count,
                "error": pub_result.error,
                "healthy": dd_healthy,
            },
            "policy_engine": {
                "status": policy_result.status,
                "violations": len(policy_result.violations),
            },
            "incident_provider": {
                "name": inc_mgr.provider.provider_name,
                "healthy": sn_healthy,
            },
        }
    except Exception as exc:
        logger.error(f"Monitoring test error: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
