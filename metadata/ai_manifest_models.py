"""
Pydantic v2 models for the AI Configuration Manifest (AI-BOM).

Each model represents one section of the manifest JSON.
The root AIManifest wraps all sections and is serialised to ai_manifest.json.

Schema version is bumped only on breaking structural changes, allowing
stable version-to-version comparison of manifests.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field


# ============================================================
# 1. Application Metadata
# ============================================================

class ApplicationMeta(BaseModel):
    """Application-level identity and runtime environment."""

    name: str = "Enterprise Multi-Agent AI System"
    version: str
    build_timestamp: str
    git_branch: Optional[str] = None
    git_commit_hash: Optional[str] = None
    python_version: str
    operating_system: str
    environment: str  # "development" | "production"


# ============================================================
# 2. Framework Metadata
# ============================================================

class FrameworkMeta(BaseModel):
    """Installed Python framework versions — read from importlib.metadata."""

    langchain_version: Optional[str] = None
    langgraph_version: Optional[str] = None
    langsmith_version: Optional[str] = None
    langchain_anthropic_version: Optional[str] = None
    langchain_community_version: Optional[str] = None
    langchain_huggingface_version: Optional[str] = None
    fastapi_version: Optional[str] = None
    streamlit_version: Optional[str] = None
    pydantic_version: Optional[str] = None
    uvicorn_version: Optional[str] = None
    python_version: str


# ============================================================
# 3. LLM Configuration
# ============================================================

class LLMConfig(BaseModel):
    """LLM provider and model configuration — read from ChatAnthropic instance."""

    provider: str  # e.g. "Anthropic" | "Azure Anthropic"
    model_name: Optional[str] = None
    model_version: Optional[str] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    streaming_enabled: bool = False
    endpoint_url: Optional[str] = None
    api_type: Optional[str] = None   # "standard" | "azure"
    region: Optional[str] = None
    timeout_seconds: Optional[float] = None


# ============================================================
# 4. Embedding Configuration
# ============================================================

class EmbeddingConfig(BaseModel):
    """Embedding model configuration — read from HuggingFaceEmbeddings instance."""

    provider: str = "HuggingFace"
    model_name: Optional[str] = None
    embedding_dimensions: Optional[int] = None  # detected via lookup table
    device: Optional[str] = None
    normalize_embeddings: Optional[bool] = None
    batch_size: Optional[int] = None


# ============================================================
# 5. RAG Configuration
# ============================================================

class RAGConfig(BaseModel):
    """Retrieval-Augmented Generation pipeline configuration."""

    vector_database: str = "FAISS"
    vector_store_type: str = "FAISS"
    embedding_model: Optional[str] = None
    chunk_size: Optional[int] = None
    chunk_overlap: Optional[int] = None
    text_splitter_type: Optional[str] = None
    retriever_type: str = "similarity"
    retriever_top_k: Optional[int] = None
    search_strategy: str = "similarity"
    similarity_metric: str = "L2"      # FAISS default
    mmr_enabled: Optional[bool] = False
    compression_retriever_enabled: Optional[bool] = False
    reranker_enabled: Optional[bool] = False
    context_window_size: Optional[int] = None


# ============================================================
# 6. Prompt Configuration
# ============================================================

class PromptFileInfo(BaseModel):
    """Metadata for a single prompt file."""

    prompt_name: str
    file_path: str
    sha256_hash: str
    constants: list[str]   # names of string constants defined in the module


class PromptConfig(BaseModel):
    """Prompt template inventory with content-addressable hashes."""

    system_prompt_version: str = "1.0.0"
    prompt_directory: str
    prompts: list[PromptFileInfo] = Field(default_factory=list)
    few_shot_enabled: bool = False
    structured_output_enabled: bool = False


# ============================================================
# 7. Multi-Agent Architecture
# ============================================================

class AgentSpec(BaseModel):
    """Single agent description."""

    name: str
    display_name: str
    agent_type: str   # "orchestrator" | "domain" | "utility"


class AgentsMeta(BaseModel):
    """Complete multi-agent architecture description."""

    framework: str = "LangGraph"
    orchestrator_name: str = "orchestrator"
    num_agents: int
    agents: list[AgentSpec]
    parallel_execution_enabled: bool = True
    max_parallel_agents: Optional[int] = None
    validation_layer_enabled: bool = True
    summarization_enabled: bool = True
    graph_topology: str = "DAG"   # Directed Acyclic Graph


# ============================================================
# 8. Tools
# ============================================================

class ToolSpec(BaseModel):
    """Metadata for a single registered LangChain tool."""

    tool_name: str
    description: Optional[str] = None
    input_schema: Optional[dict[str, Any]] = None
    output_type: str = "str"


# ============================================================
# 9. Observability
# ============================================================

class ObservabilityConfig(BaseModel):
    """Observability and monitoring configuration."""

    langsmith_enabled: bool
    project_name: Optional[str] = None
    tracing_enabled: bool
    evaluation_enabled: bool
    realtime_evaluation_enabled: bool
    offline_evaluation_enabled: bool
    opentelemetry_enabled: bool = False


# ============================================================
# 10. Security
# ============================================================

class SecurityConfig(BaseModel):
    """Security scanning and supply-chain configuration."""

    cyclonedx_enabled: bool
    defectdojo_enabled: bool
    sbom_generation_enabled: bool
    sbom_file_path: Optional[str] = None
    secrets_loaded_from: str = ".env"
    environment_variables_source: str = ".env / OS environment"


# ============================================================
# Root: AI Manifest
# ============================================================

class AIManifest(BaseModel):
    """
    AI Configuration Manifest (AI-BOM).

    Root model that wraps all configuration sections.
    Schema version bumps only on breaking structural changes —
    this guarantees stable manifest-to-manifest comparison.
    """

    manifest_version: str = "1.0.0"
    schema_version: str = "1.0"
    generated_at: str   # ISO-8601 UTC timestamp
    application: ApplicationMeta
    framework: FrameworkMeta
    llm: LLMConfig
    embedding: EmbeddingConfig
    rag: RAGConfig
    prompts: PromptConfig
    agents: AgentsMeta
    tools: list[ToolSpec]
    observability: ObservabilityConfig
    security: SecurityConfig
