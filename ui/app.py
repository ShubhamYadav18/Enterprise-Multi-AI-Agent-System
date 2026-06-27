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


# ============================================================
# Sidebar
# ============================================================

with st.sidebar:
    st.markdown("### 🤖 Enterprise Multi-Agent AI")
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
        st.rerun()


# ============================================================
# Main Content
# ============================================================

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

    # Execute graph
    graph = get_compiled_graph()
    result = graph.invoke(initial_state, config=config)

    # Compute metadata
    total_duration_ms = (time.time() - start_time) * 1000

    all_agents = {"rag_agent", "research_agent", "calculator_agent"}
    invoked = set(result.get("agents_to_invoke", []))
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
        agents_invoked=list(invoked),
        agents_skipped=skipped,
        parallel_groups=result.get("parallel_groups", []),
        tool_count=total_tool_calls,
    )

    return {
        "result": result,
        "session": session,
        "duration_ms": total_duration_ms,
        "agents_invoked": list(invoked),
        "agents_skipped": skipped,
        "agent_outputs": agent_outputs,
        "total_tool_calls": total_tool_calls,
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
            st.markdown("**✅ Invoked Agents:**")
            for agent in agents_invoked:
                emoji = {"rag_agent": "📚", "research_agent": "🔍", "calculator_agent": "🔢"}.get(agent, "🤖")
                st.markdown(f"  {emoji} `{agent}`")
        with col2:
            st.markdown("**⏭️ Skipped Agents:**")
            for agent in agents_skipped:
                st.markdown(f"  ⬜ `{agent}`")

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

    # 7. Execution Metrics
    with st.expander("📊 Execution Metrics"):
        st.markdown(f"**Session ID:** `{session.session_id}`")
        st.markdown(f"**Total Duration:** {duration_ms:.0f}ms")
        st.markdown(f"**Agents Invoked:** {len(agents_invoked)}")
        st.markdown(f"**Agents Skipped:** {len(agents_skipped)}")
        st.markdown(f"**Total Tool Calls:** {exec_data['total_tool_calls']}")
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
