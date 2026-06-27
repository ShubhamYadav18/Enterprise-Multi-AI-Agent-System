"""
Custom agent-level evaluators for LangSmith.

Replaces workflow-level metrics with 10 agent-specific metrics:
1. RAG Agent: Groundedness, Context Relevance, Retriever Quality
2. Calculator Agent: Groundedness, Hallucination, Latency
3. Research Agent: Hallucination, Latency
4. Summarizer: Consistency
5. Validator: Validation Accuracy
"""

from __future__ import annotations

from typing import Any
from utils.llm import create_llm
from langchain_core.messages import HumanMessage


def find_child_run(run, name: str) -> Any | None:
    """Recursively search for a child run by name in the run tree."""
    if getattr(run, "name", None) == name:
        return run
    if hasattr(run, "child_runs") and run.child_runs:
        for child in run.child_runs:
            res = find_child_run(child, name)
            if res:
                return res
    return None


def get_field(obj: Any, key: str, default: Any = None) -> Any:
    """Safely get a field from a dictionary or an object attribute."""
    if obj is None:
        return default
    if hasattr(obj, key):
        return getattr(obj, key)
    if isinstance(obj, dict):
        return obj.get(key, default)
    return default


# ============================================================
# 1. RAG Agent Evaluators
# ============================================================

def rag_groundedness_evaluator(run, example) -> dict[str, Any]:
    """Evaluate if the RAG Agent's response is grounded in retrieved context."""
    rag_run = find_child_run(run, "RAG Agent")
    if not rag_run:
        return {
            "key": "rag_agent_groundedness",
            "score": None,
            "comment": "Skipped: RAG Agent was not invoked."
        }

    outputs = rag_run.outputs or {}
    agent_outputs = get_field(outputs, "agent_outputs", [])
    if not agent_outputs:
        return {"key": "rag_agent_groundedness", "score": 0.0, "comment": "No agent outputs found."}

    rag_out = agent_outputs[0]
    output_text = get_field(rag_out, "output", "")
    retrieved_docs = get_field(rag_out, "retrieved_documents", [])

    if not retrieved_docs:
        return {"key": "rag_agent_groundedness", "score": 0.0, "comment": "No documents retrieved."}

    context = "\n\n".join(get_field(doc, "content", "") for doc in retrieved_docs)

    llm = create_llm(max_tokens=10, temperature=0.0)
    judge_prompt = f"""Evaluate whether the Answer is fully grounded in and supported by the Context.
Do not use any external knowledge. If there are facts in the Answer that are not in the Context, rate as 0.0.
Otherwise, rate as 1.0. Output ONLY the score (0.0 or 1.0) and nothing else.

Context:
{context}

Answer:
{output_text}

Score:"""
    try:
        res = llm.invoke([HumanMessage(content=judge_prompt)])
        score = float(res.content.strip())
    except Exception:
        score = 1.0 if "1" in res.content else 0.0

    return {
        "key": "rag_agent_groundedness",
        "score": score,
        "comment": f"Groundedness score: {score:.1f}",
    }


def rag_context_relevance_evaluator(run, example) -> dict[str, Any]:
    """Evaluate if retrieved context is relevant to the input query."""
    rag_run = find_child_run(run, "RAG Agent")
    if not rag_run:
        return {
            "key": "rag_agent_context_relevance",
            "score": None,
            "comment": "Skipped: RAG Agent was not invoked."
        }

    outputs = rag_run.outputs or {}
    agent_outputs = get_field(outputs, "agent_outputs", [])
    if not agent_outputs:
        return {
            "key": "rag_agent_context_relevance",
            "score": None,
            "comment": "Skipped: No agent outputs."
        }

    rag_out = agent_outputs[0]
    retrieved_docs = get_field(rag_out, "retrieved_documents", [])
    query = get_field(rag_run.inputs, "query", "") or get_field(run.inputs, "query", "")

    if not retrieved_docs:
        return {"key": "rag_agent_context_relevance", "score": 0.0, "comment": "No documents retrieved."}

    context = "\n\n".join(get_field(doc, "content", "") for doc in retrieved_docs)

    llm = create_llm(max_tokens=10, temperature=0.0)
    judge_prompt = f"""Evaluate whether the retrieved Context is highly relevant to answering the Query.
Rate from 0.0 (completely irrelevant) to 1.0 (perfectly relevant and sufficient).
Output ONLY a float number between 0.0 and 1.0 and nothing else.

Query: {query}

Context:
{context}

Score:"""
    try:
        res = llm.invoke([HumanMessage(content=judge_prompt)])
        score = float(res.content.strip())
    except Exception:
        score = 0.5

    return {
        "key": "rag_agent_context_relevance",
        "score": score,
        "comment": f"Context relevance score: {score:.2f}",
    }


def rag_retriever_quality_evaluator(run, example) -> dict[str, Any]:
    """Evaluate retrieval quality based on semantic search score metrics."""
    rag_run = find_child_run(run, "RAG Agent")
    if not rag_run:
        return {
            "key": "rag_agent_retriever_quality",
            "score": None,
            "comment": "Skipped: RAG Agent was not invoked."
        }

    outputs = rag_run.outputs or {}
    agent_outputs = get_field(outputs, "agent_outputs", [])
    if not agent_outputs:
        return {
            "key": "rag_agent_retriever_quality",
            "score": None,
            "comment": "Skipped: No agent outputs."
        }

    rag_out = agent_outputs[0]
    retrieved_docs = get_field(rag_out, "retrieved_documents", [])

    if not retrieved_docs:
        return {"key": "rag_agent_retriever_quality", "score": 0.0, "comment": "No documents retrieved."}

    scores = [get_field(doc, "relevance_score", 0.0) for doc in retrieved_docs]
    score = sum(scores) / len(scores) if scores else 0.0

    return {
        "key": "rag_agent_retriever_quality",
        "score": score,
        "comment": f"Average retrieval confidence: {score:.2%}",
    }


# ============================================================
# 2. Calculator Agent Evaluators
# ============================================================

def calculator_groundedness_evaluator(run, example) -> dict[str, Any]:
    """Evaluate if calculation output matches tool call records."""
    calc_run = find_child_run(run, "Calculator Agent")
    if not calc_run:
        return {
            "key": "calculator_agent_groundedness",
            "score": None,
            "comment": "Skipped: Calculator Agent was not invoked."
        }

    outputs = calc_run.outputs or {}
    agent_outputs = get_field(outputs, "agent_outputs", [])
    if not agent_outputs:
        return {
            "key": "calculator_agent_groundedness",
            "score": None,
            "comment": "Skipped: No agent outputs."
        }

    calc_out = agent_outputs[0]
    output_text = get_field(calc_out, "output", "")
    tool_calls = get_field(calc_out, "tool_calls", [])

    if not tool_calls:
        return {
            "key": "calculator_agent_groundedness",
            "score": 0.5,
            "comment": "Calculator ran but did no tool calls.",
        }

    tool_history = []
    for tc in tool_calls:
        tool_history.append(
            f"Tool {get_field(tc, 'tool_name')}({get_field(tc, 'tool_input')}) -> {get_field(tc, 'tool_output')}"
        )
    tool_context = "\n".join(tool_history)

    llm = create_llm(max_tokens=10, temperature=0.0)
    judge_prompt = f"""Evaluate whether the calculation Answer is fully grounded in and correctly uses the mathematical Tool Calls.
Rate 1.0 if the final response reflects the numbers returned by the tools. Rate 0.0 if there is a contradiction.
Output ONLY the score (0.0 or 1.0) and nothing else.

Tool Calls:
{tool_context}

Answer:
{output_text}

Score:"""
    try:
        res = llm.invoke([HumanMessage(content=judge_prompt)])
        score = float(res.content.strip())
    except Exception:
        score = 1.0 if "1" in res.content else 0.0

    return {
        "key": "calculator_agent_groundedness",
        "score": score,
        "comment": f"Calculator groundedness score: {score:.1f}",
    }


def calculator_hallucination_evaluator(run, example) -> dict[str, Any]:
    """Evaluate if the calculator agent outputted unverified numbers."""
    calc_run = find_child_run(run, "Calculator Agent")
    if not calc_run:
        return {
            "key": "calculator_agent_hallucination",
            "score": None,
            "comment": "Skipped: Calculator Agent was not invoked."
        }

    outputs = calc_run.outputs or {}
    agent_outputs = get_field(outputs, "agent_outputs", [])
    if not agent_outputs:
        return {
            "key": "calculator_agent_hallucination",
            "score": None,
            "comment": "Skipped: No agent outputs."
        }

    calc_out = agent_outputs[0]
    output_text = get_field(calc_out, "output", "")
    tool_calls = get_field(calc_out, "tool_calls", [])

    tool_history = []
    for tc in tool_calls:
        tool_history.append(f"Result: {get_field(tc, 'tool_output')}")
    tool_context = "\n".join(tool_history)

    llm = create_llm(max_tokens=10, temperature=0.0)
    judge_prompt = f"""Does the Answer hallucinate or fabricate mathematical facts/numbers NOT supported by the Tool Results?
Answer with 1.0 if the answer is clean and has NO hallucinations. Answer with 0.0 if it has hallucinations.
Output ONLY the score (0.0 or 1.0) and nothing else.

Tool Results:
{tool_context}

Answer:
{output_text}

Score:"""
    try:
        res = llm.invoke([HumanMessage(content=judge_prompt)])
        score = float(res.content.strip())
    except Exception:
        score = 1.0 if "1" in res.content else 0.0

    return {
        "key": "calculator_agent_hallucination",
        "score": score,
        "comment": "No hallucination detected." if score == 1.0 else "Hallucination detected.",
    }


def calculator_latency_evaluator(run, example) -> dict[str, Any]:
    """Evaluate the execution latency of the Calculator Agent."""
    calc_run = find_child_run(run, "Calculator Agent")
    if not calc_run:
        return {
            "key": "calculator_agent_latency",
            "score": None,
            "comment": "Skipped: Calculator Agent was not invoked."
        }

    if calc_run.end_time and calc_run.start_time:
        duration_ms = (calc_run.end_time - calc_run.start_time).total_seconds() * 1000
    else:
        duration_ms = 0.0

    return {
        "key": "calculator_agent_latency",
        "score": duration_ms,
        "comment": f"Calculator latency: {duration_ms:.1f}ms",
    }


# ============================================================
# 3. Research Agent Evaluators
# ============================================================

def research_hallucination_evaluator(run, example) -> dict[str, Any]:
    """Evaluate if research findings contain hallucinated facts."""
    res_run = find_child_run(run, "Research Agent")
    if not res_run:
        return {
            "key": "research_agent_hallucination",
            "score": None,
            "comment": "Skipped: Research Agent was not invoked."
        }

    outputs = res_run.outputs or {}
    agent_outputs = get_field(outputs, "agent_outputs", [])
    if not agent_outputs:
        return {
            "key": "research_agent_hallucination",
            "score": None,
            "comment": "Skipped: No agent outputs."
        }

    res_out = agent_outputs[0]
    output_text = get_field(res_out, "output", "")
    tool_calls = get_field(res_out, "tool_calls", [])

    tool_history = []
    for tc in tool_calls:
        tool_history.append(f"Search Result: {get_field(tc, 'tool_output')}")
    tool_context = "\n".join(tool_history)

    llm = create_llm(max_tokens=10, temperature=0.0)
    judge_prompt = f"""Does the Answer hallucinate or fabricate facts NOT supported by the Search Results?
Answer with 1.0 if the answer is clean and has NO hallucinations. Answer with 0.0 if it has hallucinations.
Output ONLY the score (0.0 or 1.0) and nothing else.

Search Results:
{tool_context}

Answer:
{output_text}

Score:"""
    try:
        res = llm.invoke([HumanMessage(content=judge_prompt)])
        score = float(res.content.strip())
    except Exception:
        score = 1.0 if "1" in res.content else 0.0

    return {
        "key": "research_agent_hallucination",
        "score": score,
        "comment": "No hallucination detected." if score == 1.0 else "Hallucination detected.",
    }


def research_latency_evaluator(run, example) -> dict[str, Any]:
    """Evaluate the execution latency of the Research Agent."""
    res_run = find_child_run(run, "Research Agent")
    if not res_run:
        return {
            "key": "research_agent_latency",
            "score": None,
            "comment": "Skipped: Research Agent was not invoked."
        }

    if res_run.end_time and res_run.start_time:
        duration_ms = (res_run.end_time - res_run.start_time).total_seconds() * 1000
    else:
        duration_ms = 0.0

    return {
        "key": "research_agent_latency",
        "score": duration_ms,
        "comment": f"Research latency: {duration_ms:.1f}ms",
    }


# ============================================================
# 4. Summarizer Evaluator
# ============================================================

def summarizer_consistency_evaluator(run, example) -> dict[str, Any]:
    """Evaluate if the summary is consistent with input agent outputs."""
    sum_run = find_child_run(run, "Summarizer")
    if not sum_run:
        return {
            "key": "summarizer_consistency",
            "score": None,
            "comment": "Skipped: Summarizer was not invoked."
        }

    outputs = sum_run.outputs or {}
    summary_text = get_field(outputs, "summary", "")
    inputs = sum_run.inputs or {}
    state_agent_outputs = get_field(get_field(inputs, "state", {}), "agent_outputs", [])

    if not state_agent_outputs:
        return {
            "key": "summarizer_consistency",
            "score": None,
            "comment": "Skipped: No agent outputs."
        }

    agent_outputs_text = ""
    for ao in state_agent_outputs:
        agent_outputs_text += f"\n### {get_field(ao, 'agent_name')}\n{get_field(ao, 'output')}"

    llm = create_llm(max_tokens=10, temperature=0.0)
    judge_prompt = f"""Evaluate if the Summary is fully consistent with the Agent Outputs.
The Summary must not contradict any information in the Agent Outputs and must not introduce new claims.
Rate 1.0 if perfectly consistent, otherwise 0.0. Output ONLY the score (0.0 or 1.0) and nothing else.

Agent Outputs:
{agent_outputs_text}

Summary:
{summary_text}

Score:"""
    try:
        res = llm.invoke([HumanMessage(content=judge_prompt)])
        score = float(res.content.strip())
    except Exception:
        score = 1.0 if "1" in res.content else 0.0

    return {
        "key": "summarizer_consistency",
        "score": score,
        "comment": f"Consistency score: {score:.1f}",
    }


# ============================================================
# 5. Validator Evaluator
# ============================================================

def validator_accuracy_evaluator(run, example) -> dict[str, Any]:
    """Evaluate validator output completeness."""
    val_run = find_child_run(run, "Validator")
    if not val_run:
        return {
            "key": "validator_accuracy",
            "score": None,
            "comment": "Skipped: Validator was not invoked."
        }

    outputs = val_run.outputs or {}
    val_report = get_field(outputs, "validation_report", {})

    score = 0.0
    if val_report:
        has_scores = get_field(val_report, "grounding_score") is not None and get_field(val_report, "consistency_score") is not None
        score = 1.0 if has_scores else 0.5

    return {
        "key": "validator_accuracy",
        "score": score,
        "comment": f"Validator outputted validation report with scores: {score:.1f}",
    }


ALL_EVALUATORS = [
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
]
