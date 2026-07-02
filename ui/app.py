"""
Streamlit Dashboard for the Enterprise Multi-Agent AI System.

Professional UI displaying:
- Chat interface
- Agent routing decisions
- Parallel execution visualization
- Retrieved documents
- Tool calls
- Validation reports
- Execution metrics
- LangSmith trace links

Usage:
    streamlit run ui/app.py
"""

import sys
from pathlib import Path

# Ensure project root is in path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from typing import Any, Optional
import time
from datetime import datetime, timezone

import streamlit as st

from config.settings import get_settings
from graph.builder import get_compiled_graph
from rag.vectorstore import get_vectorstore
from traces.session import create_session
from utils.models import AgentOutput, ValidationReport


# ============================================================
# Page Configuration
# ============================================================

st.set_page_config(
    page_title="Enterprise Multi-Agent AI System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# Custom CSS
# ============================================================

st.markdown("""
<style>
    /* Main header styling */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        color: white;
        margin-bottom: 1.5rem;
    }
    .main-header h1 {
        color: white;
        margin: 0;
        font-size: 1.8rem;
    }
    .main-header p {
        color: rgba(255,255,255,0.85);
        margin: 0.3rem 0 0 0;
        font-size: 0.95rem;
    }

    /* Agent cards */
    .agent-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        border-radius: 10px;
        padding: 0.8rem 1rem;
        margin: 0.3rem 0;
        border-left: 4px solid #667eea;
    }
    .agent-card.invoked {
        border-left-color: #27ae60;
        background: linear-gradient(135deg, #d4efdf 0%, #a9dfbf 100%);
    }
    .agent-card.skipped {
        border-left-color: #95a5a6;
        background: linear-gradient(135deg, #ecf0f1 0%, #d5dbdb 100%);
        opacity: 0.7;
    }

    /* Metric cards */
    .metric-card {
        background: white;
        border-radius: 10px;
        padding: 1rem;
        border: 1px solid #e0e0e0;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .metric-value {
        font-size: 1.5rem;
        font-weight: bold;
        color: #667eea;
    }
    .metric-label {
        font-size: 0.8rem;
        color: #666;
        text-transform: uppercase;
    }

    /* Validation scores */
    .score-high { color: #27ae60; font-weight: bold; }
    .score-medium { color: #f39c12; font-weight: bold; }
    .score-low { color: #e74c3c; font-weight: bold; }

    /* Chat messages */
    .user-message {
        background: #667eea;
        color: white;
        padding: 1rem;
        border-radius: 12px 12px 2px 12px;
        margin: 0.5rem 0;
    }
    .assistant-message {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 2px 12px 12px 12px;
        margin: 0.5rem 0;
        border: 1px solid #e0e0e0;
    }

    /* Sidebar styling */
    .sidebar-section {
        background: rgba(102, 126, 234, 0.05);
        border-radius: 8px;
        padding: 0.8rem;
        margin: 0.5rem 0;
        border: 1px solid rgba(102, 126, 234, 0.15);
    }

    /* Pipeline visualization */
    .pipeline-step {
        display: inline-block;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        margin: 0.2rem;
        font-size: 0.8rem;
        font-weight: 500;
    }
    .pipeline-active {
        background: #27ae60;
        color: white;
    }
    .pipeline-inactive {
        background: #bdc3c7;
        color: white;
    }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ============================================================
# Session State Initialization
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = []
if "current_session_id" not in st.session_state:
    st.session_state.current_session_id = None
if "execution_results" not in st.session_state:
    st.session_state.execution_results = []
if "selected_session_data" not in st.session_state:
    st.session_state.selected_session_data = None


# ============================================================
# Sidebar
# ============================================================

with st.sidebar:
    st.markdown("### 🤖 Enterprise Multi-Agent AI")
    st.markdown("---")
    
    view_mode = st.radio("Navigation", ["💬 Chat", "📊 Sessions", "🛡️ Security", "🧬 AI Config", "📡 Monitoring"], index=0, label_visibility="collapsed")
    st.markdown("---")

    # Session Info
    st.markdown("#### 📋 Session Info")
    if st.session_state.current_session_id:
        st.code(st.session_state.current_session_id[:16] + "...", language=None)
    else:
        st.caption("No active session")

    st.markdown("---")

    # Agent Registry
    st.markdown("#### 🔧 Available Agents")
    agents_info = {
        "🎯 Orchestrator": "Intent analysis & routing",
        "📚 RAG Agent": "Knowledge base retrieval",
        "🔍 Research Agent": "External search & benchmarks",
        "🔢 Calculator Agent": "Mathematical computations",
        "📝 Summarizer Agent": "Multi-agent output merger",
        "✅ Validation Agent": "Response quality validation",
    }
    for name, desc in agents_info.items():
        st.markdown(f"**{name}**")
        st.caption(desc)

    st.markdown("---")

    # Demo Scenarios
    st.markdown("#### 🎬 Demo Scenarios")
    demo_prompts = {
        "📚 RAG Only": "What is the company leave policy?",
        "🔢 Calculator Only": "Calculate the growth rate from $1,000,000 to $1,500,000",
        "📚🔍 RAG + Research": "Compare our leave policy with industry standards",
        "📚🔢 RAG + Calculator": "How many sick days do we get and what percentage of total working days is that?",
        "🌟 Full Pipeline": "Compare our leave policy with industry standards and calculate annual leave utilization rate",
    }

    for label, prompt in demo_prompts.items():
        if st.button(label, key=f"demo_{label}", use_container_width=True):
            st.session_state.demo_input = prompt

    st.markdown("---")

    # Knowledge Base
    st.markdown("#### 📁 Knowledge Base")
    settings = get_settings()
    docs_path = settings.documents_dir
    if docs_path.exists():
        for md_file in sorted(docs_path.glob("*.md")):
            title = md_file.stem.replace("_", " ").title()
            size_kb = md_file.stat().st_size / 1024
            st.caption(f"📄 {title} ({size_kb:.1f} KB)")
    else:
        st.caption("No documents found")

    st.markdown("---")

    # Clear chat
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.execution_results = []
        st.session_state.current_session_id = None
        st.session_state.selected_session_data = None
        st.rerun()


# ============================================================
# Global Helper Functions
# ============================================================

def _fmt(v):
    if isinstance(v, (int, float)):
        return f"{v:.0%}" if v <= 1.0 else str(v)
    return str(v) if v not in (None, "N/A", "Not Invoked") else "—"

def _color(v):
    try:
        s = float(v)
        if s >= 0.8: return "🟢"
        if s >= 0.5: return "🟡"
        return "🔴"
    except Exception:
        return "⚪"

def fetch_session_from_langsmith(trace_id: str) -> dict:
    from langsmith import Client
    from config.settings import get_settings
    from types import SimpleNamespace
    
    ls_client = Client()
    settings = get_settings()
    
    # 1. Fetch root run
    root_run = ls_client.read_run(trace_id)
    
    # 2. Fetch all spans in the trace
    spans = list(ls_client.list_runs(trace_id=trace_id))
    
    # 3. Find the LangGraph/graph run to extract query and response
    langgraph_run = None
    for s in spans:
        if s.name == "LangGraph":
            langgraph_run = s
            break
    if not langgraph_run:
        langgraph_run = root_run
        
    inputs = langgraph_run.inputs or {}
    outputs = langgraph_run.outputs or {}
    
    # Extract query
    query = inputs.get("query", "")
    if not query:
        for s in spans:
            if s.inputs and "query" in s.inputs:
                query = s.inputs["query"]
                break
    if not query:
        query = root_run.extra.get("metadata", {}).get("query", "")
                
    # Extract final response
    final_response = outputs.get("final_response", "")
    if not final_response:
        for s in spans:
            if s.outputs and "final_response" in s.outputs:
                final_response = s.outputs["final_response"]
                break
                
    meta = root_run.extra.get("metadata", {})
    query_type = meta.get("query_type", outputs.get("query_type", ""))
    routing_reasoning = meta.get("routing_reasoning", outputs.get("routing_reasoning", ""))
    
    # Parse invoked agents
    agents_invoked = meta.get("agents_invoked", [])
    if not agents_invoked:
        invoked_set = set()
        for s in spans:
            if s.name in ("RAG Agent", "rag"):
                invoked_set.add("Rag Agent")
            elif s.name in ("Calculator Agent", "calculator"):
                invoked_set.add("Calculator Agent")
            elif s.name in ("Research Agent", "research_agent"):
                invoked_set.add("Research Agent")
            elif s.name in ("Summarizer", "summarizer"):
                invoked_set.add("Summarizer")
            elif s.name in ("Validator", "validator"):
                invoked_set.add("Validator")
        agents_invoked = list(invoked_set)
        
    all_agents = {"rag_agent", "research_agent", "calculator_agent"}
    invoked_lower = {a.lower().replace(" ", "_") for a in agents_invoked}
    agents_skipped = list(all_agents - invoked_lower)
    
    # Duration / Latency
    duration_ms = 0.0
    if root_run.end_time and root_run.start_time:
        duration_ms = (root_run.end_time - root_run.start_time).total_seconds() * 1000
        
    # Cost & Tokens
    total_tokens = root_run.total_tokens or 0
    prompt_tokens = root_run.prompt_tokens or 0
    completion_tokens = root_run.completion_tokens or 0
    
    if not total_tokens:
        for s in spans:
            if s.run_type == "llm":
                prompt_tokens += getattr(s, "prompt_tokens", 0) or 0
                completion_tokens += getattr(s, "completion_tokens", 0) or 0
                total_tokens += getattr(s, "total_tokens", 0) or 0
        if total_tokens == 0:
            total_tokens = prompt_tokens + completion_tokens
            
    cost = float(root_run.total_cost) if getattr(root_run, "total_cost", None) is not None else 0.0
    if cost == 0.0 and (prompt_tokens or completion_tokens):
        cost = (prompt_tokens * 3.0 + completion_tokens * 15.0) / 1_000_000

    # Parse feedback evaluations
    workflow_eval = {}
    feedback_stats = getattr(root_run, "feedback_stats", {}) or {}
    for key in ["answer_correctness", "groundedness", "context_relevance", "task_completion", "tool_selection"]:
        if key in feedback_stats:
            workflow_eval[key] = feedback_stats[key].get("avg")
            
    agent_evals = {}
    agent_names = ["RAG Agent", "Calculator Agent", "Research Agent", "Summarizer", "Validator"]
    for agent in agent_names:
        prefix = agent.lower().replace(" ", "_")
        agent_metrics = {}
        for k, v in feedback_stats.items():
            if k.startswith(f"{prefix}_"):
                metric_name = k[len(prefix)+1:]
                agent_metrics[metric_name] = v.get("avg")
        if agent_metrics:
            agent_evals[agent] = agent_metrics
        else:
            ran = False
            for s in spans:
                if s.name.lower().replace(" ", "_") == prefix or (prefix == "rag_agent" and s.name == "RAG Agent"):
                    ran = True
                    break
            if ran:
                agent_evals[agent] = {}
            else:
                agent_evals[agent] = {"status": "Not Invoked"}

    # Extract agent outputs
    agent_outputs = []
    for s in spans:
        if s.name in ("RAG Agent", "rag", "Calculator Agent", "calculator", "Research Agent", "research_agent", "Summarizer", "summarizer", "Validator", "validator"):
            agent_name_normalized = s.name.lower().replace(" ", "_")
            if agent_name_normalized == "rag":
                agent_name_normalized = "rag_agent"
            elif agent_name_normalized == "calculator":
                agent_name_normalized = "calculator_agent"
            elif agent_name_normalized == "research_agent":
                agent_name_normalized = "research_agent"
            
            s_outputs = s.outputs or {}
            agent_output_data = None
            if s_outputs:
                state_agent_outputs = s_outputs.get("agent_outputs", []) or []
                if isinstance(state_agent_outputs, list):
                    for ao in state_agent_outputs:
                        if isinstance(ao, dict) and ao.get("agent_name") == agent_name_normalized:
                            agent_output_data = ao
                            break
            
            if not agent_output_data:
                output_text = s_outputs.get("summary", "") or s_outputs.get("final_response", "") or ""
                
                tool_calls = []
                for child in spans:
                    if child.parent_run_id == s.id and child.run_type == "tool":
                        tool_calls.append({
                            "tool_name": child.name,
                            "tool_input": child.inputs or {},
                            "tool_output": child.outputs or {},
                            "duration_ms": (child.end_time - child.start_time).total_seconds() * 1000 if child.end_time and child.start_time else 0.0
                        })
                
                retrieved_documents = []
                for child in spans:
                    if child.parent_run_id == s.id and child.run_type == "retriever":
                        docs = child.outputs.get("documents", []) or []
                        for d in docs:
                            retrieved_documents.append({
                                "source": d.get("metadata", {}).get("source", "unknown") if isinstance(d, dict) else "unknown",
                                "chunk_index": d.get("metadata", {}).get("chunk", 0) if isinstance(d, dict) else 0,
                                "relevance_score": d.get("metadata", {}).get("score", 0.0) if isinstance(d, dict) else 0.0,
                                "content": d.get("page_content", "") if isinstance(d, dict) else str(d)
                            })
                
                agent_output_data = {
                    "agent_name": agent_name_normalized,
                    "output": output_text or str(s_outputs),
                    "confidence": 1.0,
                    "sources": [],
                    "tool_calls": tool_calls,
                    "retrieved_documents": retrieved_documents
                }
            
            agent_outputs.append(agent_output_data)

    total_tool_calls = 0
    for s in spans:
        if s.run_type == "tool":
            total_tool_calls += 1

    validation_report = outputs.get("validation_report", {}) or []
    parallel_groups = outputs.get("parallel_groups", []) or []

    session_obj = SimpleNamespace(
        session_id=trace_id,
        conversation_id=meta.get("conversation_id", ""),
        langsmith_url=f"https://smith.langchain.com/projects/p/{root_run.session_id}/r/{trace_id}?power_user_mode=true" if hasattr(root_run, "session_id") else f"https://smith.langchain.com/runs/{trace_id}",
        total_tokens=total_tokens
    )

    return {
        "result": {
            "query": query,
            "final_response": final_response,
            "query_type": query_type,
            "routing_reasoning": routing_reasoning,
            "parallel_groups": parallel_groups,
            "validation_report": validation_report,
        },
        "session": session_obj,
        "duration_ms": duration_ms,
        "agents_invoked": [a.replace("_", " ").title() for a in agents_invoked],
        "agents_skipped": agents_skipped,
        "agent_outputs": agent_outputs,
        "total_tool_calls": total_tool_calls,
        "total_tokens": total_tokens,
        "cost": cost,
        "workflow_evaluation": workflow_eval,
        "agent_evaluations": agent_evals,
    }


# ============================================================
# Main Content
# ============================================================

if view_mode == '💬 Chat':
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🤖 Enterprise Multi-Agent AI System</h1>
        <p>LangGraph Orchestration • Parallel Execution • LangSmith Observability • RAG</p>
    </div>
    """, unsafe_allow_html=True)


    # ============================================================
    # Process Query Function
    # ============================================================

    def process_query(query: str) -> dict:
        """Process a user query through the multi-agent pipeline."""
        start_time = time.time()

        # Initialize vectorstore if needed
        try:
            get_vectorstore()
        except Exception:
            from rag.vectorstore import build_vectorstore
            build_vectorstore()

        # Create session
        session = create_session(query=query)
        st.session_state.current_session_id = session.session_id
        config = session.get_runnable_config()

        # Build initial state
        initial_state = {
            "query": query,
            "session_id": session.session_id,
            "user_id": "demo-user",
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

        # Execute graph via execute_query wrapper to log session_id as root input/output
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

        # Compute metadata
        total_duration_ms = (time.time() - start_time) * 1000

        all_agents = {"rag_agent", "research_agent", "calculator_agent"}
        invoked = set(result.get("agents_to_invoke", []))
    
        # Build corrected list of invoked agents to include Summarizer and Validator if any domain agent ran
        invoked_list = list(invoked)
        if invoked_list:
            invoked_list.append("summarizer")
            invoked_list.append("validator")
        
        skipped = list(all_agents - invoked)

        # Count tool calls
        agent_outputs = result.get("agent_outputs", [])
        total_tool_calls = sum(
            len(ao.tool_calls) if hasattr(ao, "tool_calls") else 0
            for ao in agent_outputs
        )

        # Finalize session
        session.finalize(
            query_type=result.get("query_type", ""),
            agents_invoked=invoked_list,
            agents_skipped=skipped,
            parallel_groups=result.get("parallel_groups", []),
            tool_count=total_tool_calls,
        )

        # Compute tokens and cost (fallback to local if LangSmith fails)
        total_tokens = trace_callback.total_tokens
        cost = (trace_callback.prompt_tokens * 3.0 + trace_callback.completion_tokens * 15.0) / 1_000_000

        if root_run_id_str:
            try:
                from langsmith import Client
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

        return {
            "result": result,
            "session": session,
            "duration_ms": total_duration_ms,
            "agents_invoked": [agent.replace("_", " ").title() for agent in invoked_list],
            "agents_skipped": skipped,
            "agent_outputs": agent_outputs,
            "total_tool_calls": total_tool_calls,
            "total_tokens": total_tokens,
            "cost": cost,
            "workflow_evaluation": workflow_evals,
            "agent_evaluations": agent_evals,
        }


    # ============================================================
    # Display Execution Details
    # ============================================================

    def display_execution_details(exec_data: dict, idx: int) -> None:
        """Display detailed execution information for a query."""
        result = exec_data["result"]
        session = exec_data["session"]
        duration_ms = exec_data["duration_ms"]
        agents_invoked = exec_data["agents_invoked"]
        agents_skipped = exec_data["agents_skipped"]
        agent_outputs = exec_data["agent_outputs"]

        # --- Metrics Row ---
        cols = st.columns(4)
        with cols[0]:
            st.metric("⏱️ Duration", f"{duration_ms:.0f}ms")
        with cols[1]:
            st.metric("🎯 Query Type", result.get("query_type", "unknown"))
        with cols[2]:
            st.metric("🤖 Agents", f"{len(agents_invoked)} invoked")
        with cols[3]:
            st.metric("🔧 Tool Calls", str(exec_data["total_tool_calls"]))

        # --- Expandable Details ---

        # 1. Routing Decision
        with st.expander("🎯 Routing Decision", expanded=True):
            st.markdown(f"**Query Type:** `{result.get('query_type', 'unknown')}`")
            st.markdown(f"**Reasoning:** {result.get('routing_reasoning', 'N/A')}")

            col1, col2 = st.columns(2)
            with col1:
                for agent in agents_invoked:
                    emoji = {
                        "Rag Agent": "📚", 
                        "Research Agent": "🔍", 
                        "Calculator Agent": "🔢",
                        "Summarizer": "✍️",
                        "Validator": "🛡️"
                    }.get(agent, "🤖")
                    st.markdown(f"  {emoji} `{agent}`")
            with col2:
                st.markdown("**⏭️ Skipped Agents:**")
                for agent in agents_skipped:
                    display_skipped = agent.replace("_", " ").title()
                    st.markdown(f"  ⬜ `{display_skipped}`")

        # 2. Parallel Execution
        parallel_groups = result.get("parallel_groups", [])
        with st.expander("⚡ Parallel Execution"):
            if parallel_groups:
                for i, group in enumerate(parallel_groups):
                    if len(group) > 1:
                        st.success(f"**Parallel Group {i+1}:** {', '.join(group)} — executed simultaneously")
                    else:
                        st.info(f"**Sequential:** {', '.join(group)}")
            else:
                st.info("No parallel groups detected.")

            # Pipeline visualization
            st.markdown("**Execution Pipeline:**")
            pipeline = "Orchestrator → "
            if len(agents_invoked) > 1:
                pipeline += f"[{' | '.join(agents_invoked)}] (parallel)"
            elif agents_invoked:
                pipeline += agents_invoked[0]
            pipeline += " → Summarizer → Validator"
            st.code(pipeline, language=None)

        # 3. Agent Outputs
        with st.expander("🤖 Agent Outputs"):
            for ao in agent_outputs:
                if isinstance(ao, AgentOutput):
                    agent_name = ao.agent_name
                    output = ao.output
                    confidence = ao.confidence
                    sources = ao.sources
                    error = ao.error
                elif isinstance(ao, dict):
                    agent_name = ao.get("agent_name", "unknown")
                    output = ao.get("output", "")
                    confidence = ao.get("confidence", 0)
                    sources = ao.get("sources", [])
                    error = ao.get("error")
                else:
                    continue

                emoji = {"rag_agent": "📚", "research_agent": "🔍", "calculator_agent": "🔢"}.get(agent_name, "🤖")
                st.markdown(f"### {emoji} {agent_name}")

                if error:
                    st.error(f"Error: {error}")
                else:
                    st.markdown(output[:1500] if output else "No output")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.caption(f"Confidence: {confidence:.0%}")
                    with col2:
                        if sources:
                            st.caption(f"Sources: {', '.join(sources)}")

                st.markdown("---")

        # 4. Retrieved Documents
        with st.expander("📄 Retrieved Documents"):
            has_docs = False
            for ao in agent_outputs:
                docs = ao.retrieved_documents if isinstance(ao, AgentOutput) else ao.get("retrieved_documents", []) if isinstance(ao, dict) else []
                for doc in docs:
                    has_docs = True
                    if hasattr(doc, "source"):
                        st.markdown(f"**📄 {doc.source}** (chunk {doc.chunk_index}) — Relevance: {doc.relevance_score:.0%}")
                        st.text(doc.content[:300] + "..." if len(doc.content) > 300 else doc.content)
                    elif isinstance(doc, dict):
                        st.markdown(f"**📄 {doc.get('source', 'unknown')}** — Relevance: {doc.get('relevance_score', 0):.0%}")
                        content = doc.get("content", "")
                        st.text(content[:300] + "..." if len(content) > 300 else content)
                    st.markdown("---")

            if not has_docs:
                st.info("No documents retrieved (query may not require RAG).")

        # 5. Tool Calls
        with st.expander("🔧 Tool Calls"):
            has_tools = False
            for ao in agent_outputs:
                tools = ao.tool_calls if isinstance(ao, AgentOutput) else ao.get("tool_calls", []) if isinstance(ao, dict) else []
                for tc in tools:
                    has_tools = True
                    if hasattr(tc, "tool_name"):
                        st.markdown(f"**🔧 {tc.tool_name}** ({tc.duration_ms:.0f}ms)")
                        st.json({"input": tc.tool_input, "output": str(tc.tool_output)[:500]})
                    elif isinstance(tc, dict):
                        st.markdown(f"**🔧 {tc.get('tool_name', 'unknown')}** ({tc.get('duration_ms', 0):.0f}ms)")
                        st.json({"input": tc.get("tool_input", {}), "output": str(tc.get("tool_output", ""))[:500]})

            if not has_tools:
                st.info("No tools invoked.")

        # 6. Validation Report
        with st.expander("✅ Validation Report"):
            val_data = result.get("validation_report")
            if val_data and isinstance(val_data, dict):
                def score_color(score: float) -> str:
                    if score >= 0.8: return "🟢"
                    elif score >= 0.5: return "🟡"
                    return "🔴"

                cols = st.columns(5)
                scores = [
                    ("Grounding", val_data.get("grounding_score", 0)),
                    ("Consistency", val_data.get("consistency_score", 0)),
                    ("Hallucination", val_data.get("hallucination_score", 0)),
                    ("Completeness", val_data.get("completeness_score", 0)),
                    ("Logic", val_data.get("logic_score", 0)),
                ]
                for col, (label, score) in zip(cols, scores):
                    with col:
                        # For hallucination, lower is better
                        indicator = score_color(1 - score) if label == "Hallucination" else score_color(score)
                        st.metric(f"{indicator} {label}", f"{score:.0%}")

                if val_data.get("is_valid"):
                    st.success("✅ Response validated successfully")
                else:
                    st.warning("⚠️ Response has validation issues")

                issues = val_data.get("issues", [])
                if issues:
                    st.markdown("**Issues:**")
                    for issue in issues:
                        st.markdown(f"- {issue}")
            else:
                st.info("No validation report available.")

        # 7. Workflow Evaluation
        workflow_eval = exec_data.get("workflow_evaluation")
        if workflow_eval:
            with st.expander("⚖️ Workflow Evaluation", expanded=True):
                def eval_indicator(score: Any) -> str:
                    if score == "N/A": return "⚪ N/A"
                    if score == "Not Invoked": return "⚪ Not Invoked"
                    try:
                        s = float(score)
                        if s >= 0.8: return "🟢"
                        elif s >= 0.5: return "🟡"
                        return "🔴"
                    except Exception:
                        return "⚪"

                def format_score(score: Any) -> str:
                    if isinstance(score, (int, float)):
                        return f"{score:.0%}"
                    return str(score)

                cols = st.columns(5)
                metrics = [
                    ("Answer Correctness", workflow_eval.get("answer_correctness")),
                    ("Groundedness", workflow_eval.get("groundedness")),
                    ("Context Relevance", workflow_eval.get("context_relevance")),
                    ("Task Completion", workflow_eval.get("task_completion")),
                    ("Tool Selection", workflow_eval.get("tool_selection")),
                ]
                for col, (label, score) in zip(cols, metrics):
                    with col:
                        indicator = eval_indicator(score)
                        st.metric(f"{indicator} {label}", format_score(score))

        # 8. Agent-Level Evaluations
        agent_evals = exec_data.get("agent_evaluations")
        if agent_evals:
            with st.expander("🕵️ Agent-Level Evaluations", expanded=True):
                for agent_name, metrics in agent_evals.items():
                    if isinstance(metrics, dict):
                        # Only show if not skipped/Not Invoked
                        if metrics.get("status") == "Not Invoked":
                            continue
                    
                        st.markdown(f"#### {agent_name}")
                        cols = st.columns(len(metrics))
                        for col, (metric_name, score) in zip(cols, metrics.items()):
                            with col:
                                display_metric_name = metric_name.replace("_", " ").title()
                                indicator = eval_indicator(score)
                            
                                if "latency" in metric_name.lower():
                                    val_str = f"{score:.1f}ms" if isinstance(score, (int, float)) else str(score)
                                else:
                                    val_str = format_score(score)
                                
                                st.metric(f"{indicator} {display_metric_name}", val_str)
                        st.markdown("---")

        # 9. Execution Metrics
        with st.expander("📊 Execution Metrics"):
            st.markdown(f"**Session ID:** `{session.session_id}`")
            st.markdown(f"**Total Duration:** {duration_ms:.0f}ms")
            st.markdown(f"**Agents Invoked:** {len(agents_invoked)}")
            st.markdown(f"**Agents Skipped:** {len(agents_skipped)}")
            st.markdown(f"**Total Tool Calls:** {exec_data['total_tool_calls']}")
            st.markdown(f"**Total Tokens:** {exec_data.get('total_tokens', 0)}")
            st.markdown(f"**Total Cost:** ${exec_data.get('cost', 0.0):.6f}")
            parallel_groups = result.get("parallel_groups", [])
            st.markdown(f"**Parallel Execution:** {'Yes' if any(len(g) > 1 for g in parallel_groups) else 'No'}")

            if session.langsmith_url:
                st.markdown(f"**🔗 [View in LangSmith]({session.langsmith_url})**")
            else:
                st.caption("LangSmith URL not available (check LANGCHAIN_API_KEY)")


    # ============================================================
    # Chat Interface
    # ============================================================

    # Display chat history
    for i, msg in enumerate(st.session_state.messages):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

            # Show execution details for assistant messages
            if msg["role"] == "assistant" and i // 2 < len(st.session_state.execution_results):
                exec_idx = i // 2
                display_execution_details(st.session_state.execution_results[exec_idx], exec_idx)

    # Handle demo input
    if "demo_input" in st.session_state:
        demo_query = st.session_state.demo_input
        del st.session_state.demo_input
        # Process demo query
        st.session_state.messages.append({"role": "user", "content": demo_query})
        with st.chat_message("user"):
            st.markdown(demo_query)

        with st.chat_message("assistant"):
            with st.spinner("🤖 Processing through multi-agent pipeline..."):
                try:
                    exec_data = process_query(demo_query)
                    response = exec_data["result"].get("final_response", "No response generated.")
                    st.markdown(response)
                    display_execution_details(exec_data, len(st.session_state.execution_results))

                    st.session_state.messages.append({"role": "assistant", "content": response})
                    st.session_state.execution_results.append(exec_data)
                except Exception as e:
                    st.error(f"Error: {str(e)}")
                    st.session_state.messages.append({"role": "assistant", "content": f"Error: {str(e)}"})

    # Chat input
    if prompt := st.chat_input("Ask the Enterprise Knowledge Assistant..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("🤖 Processing through multi-agent pipeline..."):
                try:
                    exec_data = process_query(prompt)
                    response = exec_data["result"].get("final_response", "No response generated.")
                    st.markdown(response)
                    display_execution_details(exec_data, len(st.session_state.execution_results))

                    st.session_state.messages.append({"role": "assistant", "content": response})
                    st.session_state.execution_results.append(exec_data)
                except Exception as e:
                    st.error(f"Error: {str(e)}")
                    st.session_state.messages.append({"role": "assistant", "content": f"Error: {str(e)}"})

# ============================================================
# Sessions Page
# ============================================================

elif view_mode == '📊 Sessions':
    st.markdown("""
    <div class="main-header">
        <h1>📊 AI Control Tower — Sessions</h1>
        <p>Every AI execution session fetched directly from LangSmith</p>
    </div>
    """, unsafe_allow_html=True)

    from langsmith import Client
    ls_client = Client()
    settings = get_settings()

    # ----------------------------------------------------------
    # Session Details View (drill-down)
    # ----------------------------------------------------------
    if st.session_state.get("selected_session_data") is not None:
        exec_data = st.session_state.selected_session_data
        result = exec_data["result"]
        session = exec_data["session"]
        duration_ms = exec_data["duration_ms"]
        agents_invoked = exec_data["agents_invoked"]
        agent_outputs = exec_data.get("agent_outputs", [])
        workflow_eval = exec_data.get("workflow_evaluation", {})
        agent_evals = exec_data.get("agent_evaluations", {})

        # Back button
        if st.button("← Back to Sessions", key="back_to_sessions"):
            st.session_state.selected_session_data = None
            st.rerun()

        st.markdown(f"### Session `{session.session_id[:8]}`")
        st.markdown("---")

        # Original Query
        st.markdown("#### 💬 Original Query")
        st.info(result.get("query", ""))

        # Final Response
        st.markdown("#### 📝 Final Response")
        st.success(result.get("final_response", "No response generated."))

        st.markdown("---")

        # Agents Invoked
        st.markdown("#### 🤖 Agents Invoked")
        agent_emojis = {
            "Orchestrator": "🎯", "Rag Agent": "📚", "Research Agent": "🔍",
            "Calculator Agent": "🔢", "Summarizer": "✍️", "Validator": "🛡️",
        }
        agent_cols = st.columns(min(len(agents_invoked), 6)) if agents_invoked else []
        for i, agent in enumerate(agents_invoked):
            with agent_cols[i % len(agent_cols)]:
                emoji = agent_emojis.get(agent, "🤖")
                st.markdown(f"✅ {emoji} **{agent}**")

        st.markdown("---")

        # Metrics row
        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric("⏱️ Latency", f"{duration_ms / 1000:.2f}s")
        mc2.metric("💰 Cost", f"${exec_data.get('cost', 0):.4f}")
        mc3.metric("📊 Total Tokens", f"{getattr(session, 'total_tokens', exec_data.get('total_tokens', 0)):,}")
        mc4.metric("🔧 Tool Calls", str(exec_data.get("total_tool_calls", 0)))

        st.markdown("---")

        # Workflow Evaluation
        if workflow_eval:
            st.markdown("#### ⚖️ Workflow Evaluation")
            wf_keys = list(workflow_eval.keys())
            wf_cols = st.columns(len(wf_keys)) if wf_keys else []
            for col, key in zip(wf_cols, wf_keys):
                val = workflow_eval[key]
                with col:
                    label = key.replace("_", " ").title()
                    st.metric(f"{_color(val)} {label}", _fmt(val))

            st.markdown("---")

        # Per-Agent Evaluation Cards
        if agent_evals:
            st.markdown("#### 🕵️ Agent Evaluations")
            for agent_name, metrics in agent_evals.items():
                if isinstance(metrics, dict) and metrics.get("status") == "Not Invoked":
                    continue

                with st.expander(f"{agent_emojis.get(agent_name, '🤖')} {agent_name}", expanded=False):
                    if isinstance(metrics, dict):
                        eval_cols = st.columns(len(metrics))
                        for col, (mk, mv) in zip(eval_cols, metrics.items()):
                            with col:
                                display_name = mk.replace("_", " ").title()
                                if "latency" in mk.lower():
                                    val_str = f"{mv:.1f}ms" if isinstance(mv, (int, float)) else str(mv)
                                else:
                                    val_str = _fmt(mv)
                                st.metric(f"{_color(mv)} {display_name}", val_str)

                    # Agent output detail
                    for ao in agent_outputs:
                        ao_name = ao.get("agent_name", "") if isinstance(ao, dict) else ""
                        if ao_name.replace("_", " ").title() == agent_name or ao_name == agent_name.lower().replace(" ", "_"):
                            st.markdown("**Output:**")
                            output_text = ao.get("output", "") if isinstance(ao, dict) else ""
                            if output_text:
                                st.markdown(output_text[:2000])

                            # Tool calls
                            tools = ao.get("tool_calls", []) if isinstance(ao, dict) else []
                            if tools:
                                st.markdown("**🔧 Tool Calls:**")
                                for tc in tools:
                                    tc_name = tc.get("tool_name", "unknown") if isinstance(tc, dict) else "unknown"
                                    tc_dur = tc.get("duration_ms", 0) if isinstance(tc, dict) else 0
                                    tc_input = tc.get("tool_input", {}) if isinstance(tc, dict) else {}
                                    tc_output = tc.get("tool_output", "") if isinstance(tc, dict) else ""
                                    st.markdown(f"**`{tc_name}`** ({tc_dur:.0f}ms)")
                                    st.json({"input": tc_input, "output": str(tc_output)[:500]})

                            # Retrieved documents (for RAG)
                            docs = ao.get("retrieved_documents", []) if isinstance(ao, dict) else []
                            if docs:
                                st.markdown("**📄 Retrieved Documents:**")
                                for doc in docs:
                                    if isinstance(doc, dict):
                                        st.markdown(f"📄 **{doc.get('source', 'unknown')}** (chunk {doc.get('chunk_index', 0)}) — Relevance: {doc.get('relevance_score', 0):.0%}")
                                        content = doc.get("content", "")
                                        st.text(content[:300] + "..." if len(content) > 300 else content)

                            break

            st.markdown("---")

        # Validation Report
        val_data = result.get("validation_report")
        if val_data and isinstance(val_data, dict):
            st.markdown("#### ✅ Validation Report")
            vr_cols = st.columns(5)
            vr_scores = [
                ("Grounding", val_data.get("grounding_score", 0)),
                ("Consistency", val_data.get("consistency_score", 0)),
                ("Hallucination", val_data.get("hallucination_score", 0)),
                ("Completeness", val_data.get("completeness_score", 0)),
                ("Logic", val_data.get("logic_score", 0)),
            ]
            for col, (label, score) in zip(vr_cols, vr_scores):
                with col:
                    indicator = _color(1 - score) if label == "Hallucination" else _color(score)
                    st.metric(f"{indicator} {label}", f"{score:.0%}")

            if val_data.get("is_valid"):
                st.success("✅ Response validated successfully")
            else:
                st.warning("⚠️ Response has validation issues")

            st.markdown("---")

        # LangSmith Link
        st.markdown("#### 🔗 LangSmith Trace")
        if session.langsmith_url:
            st.markdown(f"[**Open Trace in LangSmith →**]({session.langsmith_url})")
        else:
            st.caption("LangSmith URL not available (check LANGCHAIN_API_KEY)")

    # ----------------------------------------------------------
    # Sessions Table View
    # ----------------------------------------------------------
    else:
        # Load from LangSmith automatically (cached in session state)
        if "langsmith_root_runs" not in st.session_state or st.button("🔄 Refresh Sessions from LangSmith", use_container_width=True):
            with st.spinner("Fetching latest sessions from LangSmith..."):
                try:
                    runs = list(ls_client.list_runs(
                        project_name=settings.langchain_project,
                        is_root=True,
                        limit=20
                    ))
                    st.session_state.langsmith_root_runs = runs
                except Exception as e:
                    st.error(f"Failed to fetch runs from LangSmith: {e}")
                    st.session_state.langsmith_root_runs = []

        runs = st.session_state.get("langsmith_root_runs", [])
        if not runs:
            st.info("No sessions found in LangSmith. Go to **💬 Chat** and submit some queries first!")
        else:
            st.markdown(f"Showing last **{len(runs)}** session(s) from LangSmith.")
            st.markdown("")

            # Column headers
            h1, h2, h3, h4, h5, h6 = st.columns([1.2, 3, 3, 1, 1, 1])
            h1.markdown("**Session ID**")
            h2.markdown("**Query**")
            h3.markdown("**Agents Invoked**")
            h4.markdown("**Cost**")
            h5.markdown("**Latency**")
            h6.markdown("**Actions**")
            st.markdown("<hr style='margin:4px 0; border:none; border-top:2px solid #ccc;'>", unsafe_allow_html=True)

            # Build rows
            for i, r in enumerate(runs):
                sid = str(r.id)[:8]
                
                # Fetch query from root run metadata
                meta = r.extra.get("metadata", {})
                query_text = meta.get("query", "")
                if not query_text:
                    query_text = r.inputs.get("query", "") if r.inputs else ""
                if not query_text:
                    query_text = "Session execution trace"

                agents = ", ".join(meta.get("agents_invoked", [])) or "Orchestrator"
                
                # Latency
                latency_s = 0.0
                if r.end_time and r.start_time:
                    latency_s = (r.end_time - r.start_time).total_seconds()
                latency = f"{latency_s:.1f}s"
                
                # Cost
                cost_val = float(r.total_cost) if getattr(r, "total_cost", None) is not None else 0.0
                if cost_val == 0.0 and r.total_tokens:
                    prompt_tok = getattr(r, "prompt_tokens", 0) or 0
                    comp_tok = getattr(r, "completion_tokens", 0) or 0
                    cost_val = (prompt_tok * 3.0 + comp_tok * 15.0) / 1_000_000
                cost = f"${cost_val:.4f}"
                
                status = "✅" if r.status == "success" else "❌"

                with st.container():
                    c1, c2, c3, c4, c5, c6 = st.columns([1.2, 3, 3, 1, 1, 1])
                    with c1:
                        st.code(sid, language=None)
                    with c2:
                        st.markdown(f"**{query_text[:60]}**{'...' if len(query_text) > 60 else ''}")
                    with c3:
                        st.caption(agents)
                    with c4:
                        st.markdown(cost)
                    with c5:
                        st.markdown(latency)
                    with c6:
                        if st.button(f"View {status}", key=f"view_ls_session_{r.id}"):
                            with st.spinner("Fetching full trace details from LangSmith..."):
                                try:
                                    exec_data = fetch_session_from_langsmith(str(r.id))
                                    st.session_state.selected_session_data = exec_data
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"Error fetching trace details: {e}")
                    st.markdown("<hr style='margin:2px 0;border:none;border-top:1px solid #eee;'>", unsafe_allow_html=True)

elif view_mode == '🛡️ Security':
    from security.service import SecurityService
    from security.sbom import generate_sbom

    st.markdown("""
    <div class="main-header">
        <h1>🛡️ Security & Software Supply Chain</h1>
        <p>SBOM Generation • Vulnerability Scanning • DefectDojo Integration</p>
    </div>
    """, unsafe_allow_html=True)

    service = SecurityService()

    # ---------------------------------------------------------
    # Action Buttons
    # ---------------------------------------------------------
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 📦 Generate SBOM")
        st.caption("Generate a CycloneDX SBOM using cdxgen.")
        if st.button("Generate SBOM", use_container_width=True, key="btn_gen_sbom"):
            with st.spinner("Generating SBOM..."):
                res = generate_sbom()
                if res.status == "success":
                    st.success("SBOM Generated Successfully!")
                    st.markdown(f"**Time:** {res.timestamp}")
                    st.markdown(f"**Output Path:** `{res.output_file}`")
                    st.markdown(f"**Duration:** {res.duration_ms:.0f}ms")
                else:
                    st.error(f"Generation Failed: {res.error}")

    with col2:
        st.markdown("### ☁️ Upload & Scan")
        st.caption("Upload SBOM to DefectDojo and retrieve findings.")
        if st.button("Run Full Security Scan", use_container_width=True, key="btn_full_scan"):
            with st.spinner("Running security scan (this may take a moment)..."):
                result = service.run_scan()
                if "Completed" in result.status:
                    st.success(f"Scan {result.status}!")
                    st.markdown(f"**Scan ID:** `{result.scan_id}`")
                elif "Failed" in result.status:
                    st.error(f"Scan {result.status}: {result.error}")
                else:
                    st.warning(f"Scan Status: {result.status}")
                    if result.error:
                        st.caption(result.error)

    st.markdown("---")

    # ---------------------------------------------------------
    # Security Summary Metrics
    # ---------------------------------------------------------
    latest = service.get_latest_scan_result()
    if not latest:
        st.info("No scans have been run yet. Generate an SBOM and run a scan to see results here.")
    else:
        st.markdown("## 📊 Security Summary")
        st.caption(f"Latest Scan ID: `{latest.scan_id}` — {latest.timestamp} — Status: **{latest.status}**")

        m1, m2, m3, m4, m5, m6 = st.columns(6)
        m1.metric("📦 Packages", latest.total_packages)
        m2.metric("🔴 Critical", latest.critical_count)
        m3.metric("🟠 High", latest.high_count)
        m4.metric("🟡 Medium", latest.medium_count)
        m5.metric("🔵 Low", latest.low_count)
        m6.metric("ℹ️ Info", latest.info_count)

        st.markdown("---")

        # Tabs
        tab1, tab2, tab3 = st.tabs(["🐛 Vulnerabilities", "🌳 Dependency Tree", "📜 Scan History"])

        with tab1:
            st.markdown("### Vulnerability Findings")
            if not latest.findings:
                st.success("No vulnerabilities found in the latest scan!")
            else:
                import pandas as pd

                # Search / Filter
                search_term = st.text_input("🔍 Search vulnerabilities", "", key="vuln_search")
                severity_filter = st.multiselect(
                    "Filter by Severity",
                    ["Critical", "High", "Medium", "Low", "Info"],
                    default=[],
                    key="vuln_severity_filter",
                )

                data = []
                for f in latest.findings:
                    data.append({
                        "Package": f.package,
                        "Version": f.version,
                        "Severity": f.severity,
                        "CVE": f.cve,
                        "Description": f.description[:120] + "..." if len(f.description) > 120 else f.description,
                        "Recommendation": f.recommendation[:100] + "..." if len(f.recommendation) > 100 else f.recommendation,
                        "Status": f.status,
                    })
                df = pd.DataFrame(data)

                # Apply filters
                if severity_filter:
                    df = df[df["Severity"].isin(severity_filter)]
                if search_term:
                    mask = df.apply(lambda row: search_term.lower() in str(row).lower(), axis=1)
                    df = df[mask]

                # Sort by severity
                sev_order = {"Critical": 0, "High": 1, "Medium": 2, "Low": 3, "Info": 4, "Informational": 4}
                df["_sort"] = df["Severity"].map(sev_order).fillna(5)
                df = df.sort_values("_sort").drop(columns=["_sort"])

                st.dataframe(df, use_container_width=True, hide_index=True)
                st.caption(f"Showing {len(df)} of {len(latest.findings)} findings.")

        with tab2:
            st.markdown("### Dependency Tree")
            tree = service.get_dependency_tree()
            if tree.get("name") in ("No SBOM generated", "Failed to parse SBOM"):
                st.warning("Dependency tree not available. Run SBOM generation first.")
            else:
                # Render as expandable tree
                def render_tree_node(node: dict, depth: int = 0) -> None:
                    indent = "&nbsp;" * (depth * 6)
                    name = node.get("name", "?")
                    version = node.get("version", "?")
                    children = node.get("dependencies", [])
                    circular = node.get("circular", False)

                    icon = "📦" if depth == 0 else "├─" if depth > 0 else ""
                    circ_tag = " 🔄 *(circular)*" if circular else ""
                    st.markdown(f"{indent}{icon} **{name}** `{version}`{circ_tag}", unsafe_allow_html=True)

                    for child in children:
                        render_tree_node(child, depth + 1)

                render_tree_node(tree)

        with tab3:
            st.markdown("### Scan History")
            history = service.get_history()
            if not history:
                st.info("No scan history available.")
            else:
                import pandas as pd
                h_data = [h.model_dump() for h in history]
                df_hist = pd.DataFrame(h_data)

                # Allow selecting a previous scan
                selected_idx = st.selectbox(
                    "Select a scan to view details",
                    range(len(history)),
                    format_func=lambda i: f"{history[i].timestamp} — {history[i].status} ({history[i].packages} pkgs)",
                    key="hist_select",
                )
                st.dataframe(df_hist, use_container_width=True, hide_index=True)

                selected_entry = history[selected_idx]
                selected_result = service.get_scan_result(selected_entry.scan_id)
                if selected_result:
                    st.markdown(f"**Selected Scan:** `{selected_entry.scan_id}`")
                    sc1, sc2, sc3, sc4 = st.columns(4)
                    sc1.metric("🔴 Critical", selected_result.critical_count)
                    sc2.metric("🟠 High", selected_result.high_count)
                    sc3.metric("🟡 Medium", selected_result.medium_count)
                    sc4.metric("🔵 Low", selected_result.low_count)

        st.markdown("---")

        # Download buttons
        d_col1, d_col2 = st.columns(2)
        with d_col1:
            sbom_path = service.project_root / "sbom" / "bom.json"
            if sbom_path.exists():
                with open(sbom_path, "rb") as sbom_f:
                    st.download_button(
                        "📥 Download SBOM JSON",
                        data=sbom_f,
                        file_name="bom.json",
                        mime="application/json",
                        key="dl_sbom",
                    )
            else:
                st.caption("No SBOM file available yet.")

        with d_col2:
            report_data = latest.model_dump_json(indent=2)
            st.download_button(
                "📥 Download Scan Report",
                data=report_data,
                file_name=f"scan_{latest.scan_id}.json",
                mime="application/json",
                key="dl_report",
            )
# ============================================================
# AI Config Page
# ============================================================

elif view_mode == '\U0001f9ec AI Config':
    st.markdown("""
    <div class="main-header">
        <h1>\U0001f9ec AI Configuration Registry</h1>
        <p>Complete AI Bill of Materials (AI-BOM) \u2014 deterministically generated from runtime configuration</p>
    </div>
    """, unsafe_allow_html=True)

    import json
    from metadata.ai_manifest_generator import load_manifest, generate_manifest, get_manifest_path

    _mb1, _mb2, _mb3 = st.columns([1, 1, 5])
    with _mb1:
        if st.button("\U0001f504 Regenerate", key="regenerate_manifest", use_container_width=True):
            with st.spinner("Regenerating AI Manifest..."):
                _m = generate_manifest()
                st.session_state["_ai_manifest"] = _m
                st.success("Manifest regenerated!")
    with _mb2:
        _mp = get_manifest_path()
        if _mp.exists():
            with open(_mp, "r", encoding="utf-8") as _mf:
                _mj = _mf.read()
            st.download_button(
                "\U0001f4e5 Download JSON",
                data=_mj,
                file_name="ai_manifest.json",
                mime="application/json",
                key="dl_manifest",
                use_container_width=True,
            )

    if "_ai_manifest" not in st.session_state or st.session_state["_ai_manifest"] is None:
        manifest = load_manifest()
        if manifest is None:
            with st.spinner("Generating AI Manifest for the first time..."):
                manifest = generate_manifest()
        st.session_state["_ai_manifest"] = manifest
    else:
        manifest = st.session_state["_ai_manifest"]

    if manifest is None:
        st.error("Failed to generate AI Manifest. Check server logs for details.")
    else:
        st.caption(
            "Generated at: **" + manifest.generated_at + "** | "
            "Manifest v" + manifest.manifest_version + " | Schema v" + manifest.schema_version
        )
        st.markdown("---")

        # 1. Application
        with st.expander("\U0001f4e6 Application", expanded=True):
            _am = manifest.application
            _c1, _c2, _c3 = st.columns(3)
            _c1.metric("Application", _am.name)
            _c2.metric("Version", _am.version)
            _c3.metric("Environment", _am.environment.upper())
            _c4, _c5 = st.columns(2)
            _c4.metric("Git Branch", _am.git_branch or "N/A")
            _c5.metric("Git Commit", (_am.git_commit_hash or "N/A")[:12])
            st.caption("Python: " + _am.python_version.split()[0] + " | OS: " + _am.operating_system)
            st.caption("Build Timestamp: " + _am.build_timestamp)

        # 2. Framework
        with st.expander("\U0001f527 Framework Versions", expanded=False):
            _fw = manifest.framework
            _fw_items = {
                "LangChain": _fw.langchain_version,
                "LangGraph": _fw.langgraph_version,
                "LangSmith": _fw.langsmith_version,
                "LangChain-Anthropic": _fw.langchain_anthropic_version,
                "LangChain-Community": _fw.langchain_community_version,
                "LangChain-HuggingFace": _fw.langchain_huggingface_version,
                "FastAPI": _fw.fastapi_version,
                "Streamlit": _fw.streamlit_version,
                "Pydantic": _fw.pydantic_version,
                "Uvicorn": _fw.uvicorn_version,
            }
            _fcols = st.columns(5)
            for _fi, (_pkg, _ver) in enumerate(_fw_items.items()):
                with _fcols[_fi % 5]:
                    st.metric(_pkg, _ver or "\u2014")

        # 3. LLM
        with st.expander("\U0001f916 LLM Configuration", expanded=True):
            _llm = manifest.llm
            _lc1, _lc2, _lc3, _lc4 = st.columns(4)
            _lc1.metric("Provider", _llm.provider)
            _lc2.metric("Model", _llm.model_name or "\u2014")
            _lc3.metric("Temperature", str(_llm.temperature) if _llm.temperature is not None else "\u2014")
            _lc4.metric("Max Tokens", str(_llm.max_tokens) if _llm.max_tokens is not None else "\u2014")
            _lc5, _lc6, _lc7 = st.columns(3)
            _lc5.metric("API Type", _llm.api_type or "\u2014")
            _lc6.metric("Streaming", "Yes" if _llm.streaming_enabled else "No")
            _lc7.metric("Region", _llm.region or "N/A")
            if _llm.endpoint_url:
                st.caption("Endpoint: `" + _llm.endpoint_url + "`")

        # 4. Embedding
        with st.expander("Embedding Configuration", expanded=False):
            _emb = manifest.embedding
            _ec1, _ec2, _ec3, _ec4 = st.columns(4)
            _ec1.metric("Provider", _emb.provider)
            _ec2.metric("Model", _emb.model_name or "\u2014")
            _ec3.metric("Dimensions", str(_emb.embedding_dimensions) if _emb.embedding_dimensions else "\u2014")
            _ec4.metric("Device", _emb.device or "\u2014")
            _norm = "Yes" if _emb.normalize_embeddings else "No"
            _bs = str(_emb.batch_size) if _emb.batch_size else "default"
            st.caption("Normalize: " + _norm + " | Batch Size: " + _bs)

        # 5. RAG
        with st.expander("RAG Configuration", expanded=False):
            _rag = manifest.rag
            _rc1, _rc2, _rc3, _rc4 = st.columns(4)
            _rc1.metric("Vector DB", _rag.vector_database)
            _rc2.metric("Chunk Size", str(_rag.chunk_size) if _rag.chunk_size else "\u2014")
            _rc3.metric("Chunk Overlap", str(_rag.chunk_overlap) if _rag.chunk_overlap else "\u2014")
            _rc4.metric("Retrieval K", str(_rag.retriever_top_k) if _rag.retriever_top_k else "\u2014")
            _rc5, _rc6, _rc7 = st.columns(3)
            _rc5.metric("Text Splitter", _rag.text_splitter_type or "\u2014")
            _rc6.metric("Search Strategy", _rag.search_strategy)
            _rc7.metric("Similarity Metric", _rag.similarity_metric)
            _rag_flags = []
            if _rag.mmr_enabled:
                _rag_flags.append("MMR")
            if _rag.compression_retriever_enabled:
                _rag_flags.append("Compression Retriever")
            if _rag.reranker_enabled:
                _rag_flags.append("Re-ranker")
            st.caption("Active Features: " + (", ".join(_rag_flags) if _rag_flags else "None"))

        # 6. Prompts
        with st.expander("Prompt Configuration", expanded=False):
            _prm = manifest.prompts
            _so = "Yes" if _prm.structured_output_enabled else "No"
            _fs = "Yes" if _prm.few_shot_enabled else "No"
            st.caption("Directory: `" + _prm.prompt_directory + "` | Version: " + _prm.system_prompt_version + " | Few-Shot: " + _fs + " | Structured Output: " + _so)
            if _prm.prompts:
                import pandas as pd
                _prows = [{"Prompt": p.prompt_name, "File": p.file_path, "SHA-256 (first 16)": p.sha256_hash[:16] + "...", "Constants": ", ".join(p.constants[:4])} for p in _prm.prompts]
                st.dataframe(pd.DataFrame(_prows), use_container_width=True, hide_index=True)

        # 7. Agents
        with st.expander("Multi-Agent Architecture", expanded=True):
            _agts = manifest.agents
            _ag1, _ag2, _ag3 = st.columns(3)
            _ag1.metric("Framework", _agts.framework)
            _ag2.metric("Total Agents", str(_agts.num_agents))
            _ag3.metric("Graph Topology", _agts.graph_topology)
            _aflags = []
            if _agts.parallel_execution_enabled:
                _aflags.append("Parallel (max " + str(_agts.max_parallel_agents) + ")")
            if _agts.validation_layer_enabled:
                _aflags.append("Validation Layer")
            if _agts.summarization_enabled:
                _aflags.append("Summarization")
            st.caption("Features: " + ", ".join(_aflags))
            import pandas as pd
            _arows = [{"Agent": a.display_name, "Type": a.agent_type} for a in _agts.agents]
            st.dataframe(pd.DataFrame(_arows), use_container_width=True, hide_index=True)

        # 8. Tools
        with st.expander("Registered Tools", expanded=False):
            if manifest.tools:
                import pandas as pd
                _trows = [{"Tool": t.tool_name, "Output Type": t.output_type, "Description": (t.description or "")[:90]} for t in manifest.tools]
                st.dataframe(pd.DataFrame(_trows), use_container_width=True, hide_index=True)
            else:
                st.caption("No tools registered or tool introspection unavailable.")

        # 9. Observability
        with st.expander("Observability", expanded=False):
            _obs = manifest.observability
            _oc1, _oc2, _oc3 = st.columns(3)
            _oc1.metric("LangSmith", "Enabled" if _obs.langsmith_enabled else "Disabled")
            _oc2.metric("Tracing", "Yes" if _obs.tracing_enabled else "No")
            _oc3.metric("OpenTelemetry", "Yes" if _obs.opentelemetry_enabled else "No")
            _oc4, _oc5, _oc6 = st.columns(3)
            _oc4.metric("Real-time Eval", "Yes" if _obs.realtime_evaluation_enabled else "No")
            _oc5.metric("Offline Eval", "Yes" if _obs.offline_evaluation_enabled else "No")
            _oc6.metric("Project", _obs.project_name or "\u2014")

        # 10. Security
        with st.expander("Security", expanded=False):
            _sec = manifest.security
            _sc1, _sc2, _sc3 = st.columns(3)
            _sc1.metric("CycloneDX SBOM", "Yes" if _sec.cyclonedx_enabled else "No")
            _sc2.metric("DefectDojo", "Configured" if _sec.defectdojo_enabled else "Not configured")
            _sc3.metric("SBOM Generation", "Yes" if _sec.sbom_generation_enabled else "No")
            st.caption("SBOM Path: `" + str(_sec.sbom_file_path or "\u2014") + "`")
            st.caption("Secrets: " + _sec.secrets_loaded_from)
            st.caption("Env Source: " + _sec.environment_variables_source)

        # Raw JSON
        with st.expander("Raw Manifest JSON", expanded=False):
            st.code(json.dumps(manifest.model_dump(mode="json"), indent=2, ensure_ascii=False), language="json")


# ============================================================
# Monitoring Page
# ============================================================

elif view_mode == '\U0001f4e1 Monitoring':
    import requests as _req
    import pandas as _pd

    API = "http://localhost:8000"

    st.markdown("""
    <div class="main-header">
        <h1>&#x1F4E1; AI Incident Management</h1>
        <p>Real-time policy enforcement, metric publishing and incident management</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Helper ──────────────────────────────────────────────────
    def _status_badge(status: str) -> str:
        s = str(status).lower()
        if s == "healthy":   return "🟢 Healthy"
        if s == "warning":   return "🟡 Warning"
        if s == "critical":  return "🔴 Critical"
        return f"⚪ {status}"

    def _bool_icon(v) -> str:
        return "✅ Yes" if v else "❌ No"

    # ── Monitoring Config ────────────────────────────────────────
    st.markdown("### ⚙️ Monitoring Configuration")

    try:
        _cfg = _req.get(f"{API}/monitoring/status", timeout=5).json()
        _dd  = _cfg.get("datadog", {})
        _inc = _cfg.get("incident", {})
        _thr = _cfg.get("thresholds", {})

        _c1, _c2 = st.columns(2)

        with _c1:
            st.markdown("#### Datadog")
            st.metric("Status", "Enabled" if _dd.get("enabled") else "Disabled")
            sub1, sub2 = st.columns(2)
            sub1.metric("API Key", "✅ Set" if _dd.get("api_key_configured") else "❌ Missing")
            sub2.metric("App Key", "✅ Set" if _dd.get("app_key_configured") else "❌ Missing")
            st.caption(f"Site: `{_dd.get('site', '—')}` | Prefix: `{_dd.get('metric_prefix', '—')}`")
            if _dd.get("healthy") is not None:
                st.info(f"Connectivity: {_bool_icon(_dd['healthy'])}")

        with _c2:
            st.markdown("#### Incident Provider")
            provider_name = str(_inc.get("provider", "noop")).upper()
            st.metric("Provider", provider_name)
            sn_enabled = _inc.get("servicenow_enabled", False)
            sn_url = _inc.get("servicenow_instance_url") or "Not configured"
            st.metric("ServiceNow", "Enabled" if sn_enabled else "Disabled")
            st.caption(f"Instance: `{sn_url}`")
            if _inc.get("healthy") is not None:
                st.info(f"Connectivity: {_bool_icon(_inc['healthy'])}")

        st.markdown("#### Policy Thresholds")
        _th_rows = [
            {"Metric": "Groundedness",  "Threshold": f">= {_thr.get('groundedness_min', 0.80):.2f}",  "Severity": "🔴 Critical"},
            {"Metric": "Hallucination",  "Threshold": f"<= {_thr.get('hallucination_max', 0.20):.2f}",  "Severity": "🔴 Critical"},
            {"Metric": "Latency (ms)",  "Threshold": f"<= {_thr.get('latency_max_ms', 20000):.0f}",    "Severity": "🟡 Warning"},
            {"Metric": "Cost (USD)",    "Threshold": f"<= ${_thr.get('total_cost_max', 0.50):.4f}",   "Severity": "🟡 Warning"},
            {"Metric": "Total Tokens",  "Threshold": f"<= {_thr.get('token_limit', 50000):,}",         "Severity": "🟡 Warning"},
        ]
        st.dataframe(_pd.DataFrame(_th_rows), use_container_width=True, hide_index=True)

    except Exception as _cfg_err:
        st.error(f"Could not load monitoring config: {_cfg_err}")

    st.markdown("---")

    # ── Test Monitoring Button ───────────────────────────────────
    st.markdown("### 🧪 Test Monitoring Connectivity")
    st.caption("Sends a synthetic metric event through the full pipeline to verify Datadog and incident provider connectivity.")

    if st.button("▶ Run Connectivity Test", key="btn_mon_test", type="primary"):
        with st.spinner("Running monitoring test..."):
            try:
                _test = _req.post(f"{API}/monitoring/test", timeout=30).json()
                st.success("Test complete!")

                _tc1, _tc2, _tc3 = st.columns(3)
                _dd_t = _test.get("datadog", {})
                _pe_t = _test.get("policy_engine", {})
                _ip_t = _test.get("incident_provider", {})

                with _tc1:
                    st.markdown("**Datadog**")
                    st.metric("Published", _bool_icon(_dd_t.get("published")))
                    st.metric("Metrics Sent", str(_dd_t.get("metrics_count", 0)))
                    if _dd_t.get("error"):
                        st.warning(_dd_t["error"])

                with _tc2:
                    st.markdown("**Policy Engine**")
                    pe_status = _pe_t.get("status", "healthy")
                    st.metric("Status", _status_badge(pe_status))
                    st.metric("Violations", str(_pe_t.get("violations", 0)))

                with _tc3:
                    st.markdown("**Incident Provider**")
                    st.metric("Provider", str(_ip_t.get("name", "noop")).upper())
                    h = _ip_t.get("healthy")
                    if h is None:
                        st.metric("Connectivity", "N/A (disabled)")
                    else:
                        st.metric("Connectivity", _bool_icon(h))

            except Exception as _te:
                st.error(f"Test failed: {_te}")

    st.markdown("---")

    # ── Session Monitoring History ───────────────────────────────
    st.markdown("### 📋 Session Monitoring History")
    st.caption("Monitoring results for recent sessions (last 50, process lifetime).")

    try:
        _hist = _req.get(f"{API}/monitoring/history", timeout=5).json()
        if not _hist:
            st.info("No monitoring results yet. Submit a query to generate them.")
        else:
            _hist_rows = []
            for _h in reversed(_hist):
                _pr = _h.get("policy_result", {})
                _viols = _pr.get("violations", [])
                _hist_rows.append({
                    "Session": _h.get("session_id", "")[:16] + "...",
                    "Status": _status_badge(_pr.get("status", "healthy")),
                    "Violations": len(_viols),
                    "Datadog": "✅" if _h.get("datadog_published") else "—",
                    "Incident": _h.get("incident_number") or "—",
                    "Timestamp": (_h.get("timestamp", "")[:19]).replace("T", " "),
                })
            st.dataframe(_pd.DataFrame(_hist_rows), use_container_width=True, hide_index=True)

            # Detailed view of selected session
            _session_ids = [h.get("session_id", "") for h in reversed(_hist)]
            _sel_sid = st.selectbox("🔍 Inspect Session", _session_ids,
                                     format_func=lambda x: x[:16] + "...",
                                     key="mon_session_select")
            if _sel_sid:
                try:
                    _detail = _req.get(f"{API}/monitoring/result/{_sel_sid}", timeout=5).json()
                    if _detail.get("found"):
                        _dpr = _detail.get("policy_result", {})
                        _dviols = _dpr.get("violations", [])

                        _d1, _d2, _d3 = st.columns(3)
                        _d1.metric("Policy Status", _status_badge(_dpr.get("status", "healthy")))
                        _d2.metric("Datadog", "Published" if _detail.get("datadog_published") else "Skipped")
                        _d3.metric("Incident", _detail.get("incident_number") or "None")

                        if _detail.get("datadog_error"):
                            st.warning(f"Datadog error: {_detail['datadog_error']}")
                        if _detail.get("incident_error"):
                            st.warning(f"Incident error: {_detail['incident_error']}")
                        if _detail.get("incident_url"):
                            st.markdown(f"[🎫 Open in ServiceNow]({_detail['incident_url']})")

                        if _dviols:
                            st.markdown("#### Policy Violations")
                            _vrows = []
                            for _v in _dviols:
                                _vrows.append({
                                    "Metric": _v.get("metric", ""),
                                    "Actual": round(float(_v.get("actual", 0)), 4),
                                    "Threshold": round(float(_v.get("threshold", 0)), 4),
                                    "Severity": _status_badge(_v.get("severity", "warning")),
                                    "Detail": _v.get("description", "")[:120],
                                })
                            st.dataframe(_pd.DataFrame(_vrows), use_container_width=True, hide_index=True)
                        else:
                            st.success("No policy violations for this session.")
                except Exception as _de:
                    st.warning(f"Could not fetch detail: {_de}")
    except Exception as _he:
        st.error(f"Could not fetch monitoring history: {_he}")
