"""
RAG Agent.

Retrieves information from the vector knowledge base and
synthesizes answers grounded in the retrieved documents.
"""

from __future__ import annotations

import time
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langsmith import traceable

from prompts.rag_agent import RAG_SYSTEM_PROMPT, RAG_HUMAN_PROMPT
from rag.vectorstore import similarity_search
from utils.logger import get_logger, log_agent_event
from utils.models import AgentOutput, RetrievedDocument, ToolCallRecord

logger = get_logger(__name__)





def run_rag_agent(
    query: str,
    config: RunnableConfig | None = None,
) -> AgentOutput:
    """Run the RAG agent to retrieve and synthesize knowledge base information.

    Args:
        query: User query to answer from the knowledge base.
        config: LangSmith tracing config for session correlation.

    Returns:
        AgentOutput with synthesized answer, sources, and retrieved documents.
    """
    session_id = (config or {}).get("metadata", {}).get("session_id", "")
    log_agent_event(logger, "agent_start", "rag_agent", session_id)
    start_time = time.time()

    try:
        # Step 1: Retrieve relevant documents
        retrieval_start = time.time()
        search_results = similarity_search(query, config=config)
        retrieval_duration = (time.time() - retrieval_start) * 1000

        retrieved_docs: list[RetrievedDocument] = []
        context_parts: list[str] = []

        for i, (doc, score) in enumerate(search_results):
            source = doc.metadata.get("source", "unknown")
            similarity = max(0, 1 - (score / 2))

            retrieved_docs.append(RetrievedDocument(
                content=doc.page_content,
                source=source,
                chunk_index=doc.metadata.get("chunk_index", 0),
                relevance_score=round(similarity, 3),
            ))

            context_parts.append(
                f"[Source: {source}]\n{doc.page_content}"
            )

        context = "\n\n---\n\n".join(context_parts) if context_parts else "No relevant documents found."

        tool_calls = [
            ToolCallRecord(
                tool_name="vector_search",
                tool_input={"query": query, "k": len(search_results)},
                tool_output=f"Retrieved {len(search_results)} documents",
                duration_ms=retrieval_duration,
            )
        ]

        # Step 2: Generate answer from context
        from utils.llm import create_llm
        llm = create_llm(max_tokens=2048, temperature=0.1)
        system_prompt = RAG_SYSTEM_PROMPT.format(context=context)
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=RAG_HUMAN_PROMPT.format(query=query)),
        ]

        response = llm.invoke(messages, config=config)

        # Determine confidence from response
        content = response.content
        confidence = 0.9
        if "LOW" in content.upper():
            confidence = 0.4
        elif "MEDIUM" in content.upper():
            confidence = 0.7

        sources = list(set(d.source for d in retrieved_docs))
        duration_ms = (time.time() - start_time) * 1000

        log_agent_event(
            logger, "agent_finish", "rag_agent", session_id,
            duration_ms=duration_ms,
            documents_retrieved=len(retrieved_docs),
        )

        return AgentOutput(
            agent_name="rag_agent",
            output=content,
            confidence=confidence,
            sources=sources,
            retrieved_documents=retrieved_docs,
            tool_calls=tool_calls,
            metadata={
                "documents_retrieved": len(retrieved_docs),
                "retrieval_duration_ms": retrieval_duration,
                "total_duration_ms": duration_ms,
            },
        )

    except Exception as e:
        logger.error(f"RAG agent error: {e}", exc_info=True)
        return AgentOutput(
            agent_name="rag_agent",
            output="",
            error=str(e),
            metadata={"total_duration_ms": (time.time() - start_time) * 1000},
        )
