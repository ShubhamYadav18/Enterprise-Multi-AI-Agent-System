import dotenv
dotenv.load_dotenv()
from config.settings import get_settings
from langsmith import Client
from types import SimpleNamespace
import datetime

def fetch_session_from_langsmith(trace_id: str) -> dict:
    ls_client = Client()
    settings = get_settings()
    
    root_run = ls_client.read_run(trace_id)
    spans = list(ls_client.list_runs(trace_id=trace_id))
    
    langgraph_run = None
    for s in spans:
        if s.name == "LangGraph":
            langgraph_run = s
            break
    if not langgraph_run:
        langgraph_run = root_run
        
    inputs = langgraph_run.inputs or {}
    outputs = langgraph_run.outputs or {}
    
    query = inputs.get("query", "")
    if not query:
        for s in spans:
            if s.inputs and "query" in s.inputs:
                query = s.inputs["query"]
                break
                
    final_response = outputs.get("final_response", "")
    if not final_response:
        for s in spans:
            if s.outputs and "final_response" in s.outputs:
                final_response = s.outputs["final_response"]
                break
                
    meta = root_run.extra.get("metadata", {})
    query_type = meta.get("query_type", outputs.get("query_type", ""))
    routing_reasoning = meta.get("routing_reasoning", outputs.get("routing_reasoning", ""))
    
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
    
    duration_ms = 0.0
    if root_run.end_time and root_run.start_time:
        duration_ms = (root_run.end_time - root_run.start_time).total_seconds() * 1000
        
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

# Test parsing the latest root run
ls_client = Client()
runs = list(ls_client.list_runs(project_name=get_settings().langchain_project, is_root=True, limit=1))
if runs:
    parsed = fetch_session_from_langsmith(str(runs[0].id))
    print("Parsed output successfully!")
    print("Cost:", parsed["cost"])
    print("Tokens:", parsed["total_tokens"])
    print("Duration (s):", parsed["duration_ms"]/1000)
    print("Workflow Eval Keys:", list(parsed["workflow_evaluation"].keys()))
    print("Agent outputs count:", len(parsed["agent_outputs"]))
else:
    print("No runs found to test.")
