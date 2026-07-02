"""
Metadata collectors for each AI-BOM section.

Each collector function is independent and fully fault-tolerant.
If introspection fails, collectors log a warning and return sensible defaults.
Collectors NEVER raise — application startup must never be blocked.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from utils.logger import get_logger
from metadata.ai_manifest_models import (
    AgentSpec,
    AgentsMeta,
    ApplicationMeta,
    EmbeddingConfig,
    FrameworkMeta,
    LLMConfig,
    ObservabilityConfig,
    PromptConfig,
    PromptFileInfo,
    RAGConfig,
    SecurityConfig,
    ToolSpec,
)

logger = get_logger(__name__)

# Project root — two levels up from metadata/
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ─────────────────────────────────────────────────────────────
# Known embedding dimension lookup (avoids runtime test-embed)
# ─────────────────────────────────────────────────────────────
_EMBEDDING_DIMS: dict[str, int] = {
    "sentence-transformers/all-MiniLM-L6-v2": 384,
    "sentence-transformers/all-MiniLM-L12-v2": 384,
    "sentence-transformers/all-mpnet-base-v2": 768,
    "sentence-transformers/paraphrase-MiniLM-L6-v2": 384,
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2": 384,
    "BAAI/bge-small-en-v1.5": 384,
    "BAAI/bge-base-en-v1.5": 768,
    "BAAI/bge-large-en-v1.5": 1024,
    "text-embedding-ada-002": 1536,
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
}


def _safe_pkg_version(pkg: str) -> Optional[str]:
    """Return installed package version or None."""
    try:
        return importlib.metadata.version(pkg)
    except importlib.metadata.PackageNotFoundError:
        return None


def _git_info() -> tuple[Optional[str], Optional[str]]:
    """Return (branch, commit_hash) from git, or (None, None) on failure."""
    try:
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=str(PROJECT_ROOT),
            stderr=subprocess.DEVNULL,
            timeout=5,
        ).decode().strip()
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(PROJECT_ROOT),
            stderr=subprocess.DEVNULL,
            timeout=5,
        ).decode().strip()
        return branch or None, commit or None
    except Exception:
        logger.debug("Git metadata unavailable (not a git repo or git not on PATH).")
        return None, None


def _sha256_of_file(path: Path) -> str:
    """Compute SHA-256 hex digest of a file's content."""
    h = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
    except OSError:
        pass
    return h.hexdigest()


# ============================================================
# 1. Application Metadata
# ============================================================

def collect_application_meta() -> ApplicationMeta:
    """Collect application identity and runtime environment."""
    try:
        git_branch, git_hash = _git_info()

        # Attempt to read version from package metadata or fallback
        try:
            app_version = importlib.metadata.version("enterprise-multi-agent")
        except importlib.metadata.PackageNotFoundError:
            app_version = "0.1.0"

        # Read environment hint from env var
        env_hint = os.environ.get("APP_ENV", os.environ.get("ENVIRONMENT", "development"))

        return ApplicationMeta(
            name="Enterprise Multi-Agent AI System",
            version=app_version,
            build_timestamp=datetime.now(timezone.utc).isoformat(),
            git_branch=git_branch,
            git_commit_hash=git_hash,
            python_version=sys.version,
            operating_system=f"{platform.system()} {platform.release()} ({platform.machine()})",
            environment=env_hint,
        )
    except Exception as exc:
        logger.warning(f"collect_application_meta failed: {exc}")
        return ApplicationMeta(
            version="unknown",
            build_timestamp=datetime.now(timezone.utc).isoformat(),
            python_version=sys.version,
            operating_system=platform.system(),
            environment="unknown",
        )


# ============================================================
# 2. Framework Metadata
# ============================================================

def collect_framework_meta() -> FrameworkMeta:
    """Collect installed framework package versions via importlib.metadata."""
    try:
        return FrameworkMeta(
            langchain_version=_safe_pkg_version("langchain"),
            langgraph_version=_safe_pkg_version("langgraph"),
            langsmith_version=_safe_pkg_version("langsmith"),
            langchain_anthropic_version=_safe_pkg_version("langchain-anthropic"),
            langchain_community_version=_safe_pkg_version("langchain-community"),
            langchain_huggingface_version=_safe_pkg_version("langchain-huggingface"),
            fastapi_version=_safe_pkg_version("fastapi"),
            streamlit_version=_safe_pkg_version("streamlit"),
            pydantic_version=_safe_pkg_version("pydantic"),
            uvicorn_version=_safe_pkg_version("uvicorn"),
            python_version=sys.version,
        )
    except Exception as exc:
        logger.warning(f"collect_framework_meta failed: {exc}")
        return FrameworkMeta(python_version=sys.version)


# ============================================================
# 3. LLM Configuration
# ============================================================

def collect_llm_config() -> LLMConfig:
    """Read LLM configuration from the ChatAnthropic factory."""
    try:
        from utils.llm import create_llm
        from config.settings import get_settings

        settings = get_settings()

        # Instantiate with default values to read configuration
        llm = create_llm()

        # Determine provider / api_type
        has_custom_url = bool(settings.anthropic_url)
        api_type = "azure" if has_custom_url else "standard"
        provider = "Azure Anthropic" if has_custom_url else "Anthropic"

        # Extract base_url safely
        endpoint_url = getattr(llm, "anthropic_api_url", None)
        if endpoint_url is None:
            endpoint_url = getattr(llm, "_base_url", None)
        if endpoint_url is None and has_custom_url:
            endpoint_url = settings.anthropic_url

        # Read model attributes from the object
        model_name = getattr(llm, "model", None) or settings.anthropic_model
        temperature = getattr(llm, "temperature", None)
        max_tokens = getattr(llm, "max_tokens", None)
        streaming = getattr(llm, "streaming", False)

        return LLMConfig(
            provider=provider,
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            streaming_enabled=bool(streaming),
            endpoint_url=str(endpoint_url) if endpoint_url else None,
            api_type=api_type,
        )
    except Exception as exc:
        logger.warning(f"collect_llm_config failed: {exc}")
        try:
            from config.settings import get_settings
            s = get_settings()
            return LLMConfig(
                provider="Anthropic",
                model_name=s.anthropic_model,
                api_type="standard" if not s.anthropic_url else "azure",
            )
        except Exception:
            return LLMConfig(provider="Unknown")


# ============================================================
# 4. Embedding Configuration
# ============================================================

def collect_embedding_config() -> EmbeddingConfig:
    """Read embedding configuration from the HuggingFaceEmbeddings singleton."""
    try:
        from rag.embeddings import get_embeddings
        from config.settings import get_settings

        settings = get_settings()
        emb = get_embeddings()

        model_name = getattr(emb, "model_name", settings.embedding_model)
        model_kwargs = getattr(emb, "model_kwargs", {})
        encode_kwargs = getattr(emb, "encode_kwargs", {})

        device = model_kwargs.get("device", settings.embedding_device)
        normalize = encode_kwargs.get("normalize_embeddings", True)

        # Lookup embedding dimensions by model name
        dims = _EMBEDDING_DIMS.get(model_name)

        return EmbeddingConfig(
            provider="HuggingFace Sentence Transformers",
            model_name=model_name,
            embedding_dimensions=dims,
            device=device,
            normalize_embeddings=normalize,
            batch_size=encode_kwargs.get("batch_size"),
        )
    except Exception as exc:
        logger.warning(f"collect_embedding_config failed: {exc}")
        try:
            from config.settings import get_settings
            s = get_settings()
            return EmbeddingConfig(
                provider="HuggingFace",
                model_name=s.embedding_model,
                device=s.embedding_device,
                embedding_dimensions=_EMBEDDING_DIMS.get(s.embedding_model),
            )
        except Exception:
            return EmbeddingConfig(provider="Unknown")


# ============================================================
# 5. RAG Configuration
# ============================================================

def collect_rag_config() -> RAGConfig:
    """Collect RAG pipeline configuration from settings."""
    try:
        from config.settings import get_settings
        settings = get_settings()

        return RAGConfig(
            vector_database="FAISS",
            vector_store_type="FAISS (langchain_community)",
            embedding_model=settings.embedding_model,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            text_splitter_type="RecursiveCharacterTextSplitter",
            retriever_type="similarity",
            retriever_top_k=settings.retrieval_k,
            search_strategy="similarity_search_with_score",
            similarity_metric="L2 (Euclidean)",
            mmr_enabled=False,
            compression_retriever_enabled=False,
            reranker_enabled=False,
            context_window_size=None,
        )
    except Exception as exc:
        logger.warning(f"collect_rag_config failed: {exc}")
        return RAGConfig()


# ============================================================
# 6. Prompt Configuration
# ============================================================

def collect_prompt_config() -> PromptConfig:
    """Scan the prompts/ directory and collect prompt hashes and metadata."""
    prompts_dir = PROJECT_ROOT / "prompts"
    prompt_infos: list[PromptFileInfo] = []

    try:
        for py_file in sorted(prompts_dir.glob("*.py")):
            if py_file.name.startswith("_"):
                continue

            file_hash = _sha256_of_file(py_file)

            # Read the source to find top-level string constant names
            try:
                source = py_file.read_text(encoding="utf-8")
                constants = []
                for line in source.splitlines():
                    stripped = line.strip()
                    # Match lines like: SOME_CONSTANT = """...""" or = "..."
                    if (
                        "=" in stripped
                        and not stripped.startswith("#")
                        and stripped.split("=")[0].strip().isupper()
                    ):
                        const_name = stripped.split("=")[0].strip()
                        if const_name.isidentifier():
                            constants.append(const_name)
            except OSError:
                constants = []

            prompt_infos.append(
                PromptFileInfo(
                    prompt_name=py_file.stem,
                    file_path=str(py_file.relative_to(PROJECT_ROOT)),
                    sha256_hash=file_hash,
                    constants=constants,
                )
            )

        return PromptConfig(
            system_prompt_version="1.0.0",
            prompt_directory=str(prompts_dir.relative_to(PROJECT_ROOT)),
            prompts=prompt_infos,
            few_shot_enabled=False,
            structured_output_enabled=True,  # orchestrator uses JSON structured output
        )
    except Exception as exc:
        logger.warning(f"collect_prompt_config failed: {exc}")
        return PromptConfig(
            prompt_directory=str(prompts_dir.relative_to(PROJECT_ROOT)),
        )


# ============================================================
# 7. Multi-Agent Architecture
# ============================================================

def collect_agents_meta() -> AgentsMeta:
    """Read agent definitions from the LangGraph builder."""
    try:
        from graph.builder import build_graph

        # Build the raw (uncompiled) StateGraph to read registered nodes
        graph = build_graph()
        node_names: list[str] = list(getattr(graph, "nodes", {}).keys())

        # Remove built-in LangGraph special nodes
        agent_names = [n for n in node_names if n not in ("__start__", "__end__")]

        agent_specs: list[AgentSpec] = []
        for node in agent_names:
            display = node.replace("_", " ").title()
            
            # Dynamically infer type
            if node == "orchestrator":
                atype = "orchestrator"
            elif node in ("summarizer", "validator"):
                atype = "utility"
            else:
                atype = "domain"

            agent_specs.append(
                AgentSpec(
                    name=node,
                    display_name=display,
                    agent_type=atype,
                )
            )

        return AgentsMeta(
            framework="LangGraph",
            orchestrator_name="orchestrator",
            num_agents=len(agent_specs),
            agents=agent_specs,
            parallel_execution_enabled=True,
            max_parallel_agents=3,
            validation_layer_enabled=True,
            summarization_enabled=True,
            graph_topology="DAG",
        )

    except Exception as exc:
        logger.warning(f"collect_agents_meta failed: {exc}")
        # Dynamic fallback from static lists (but kept basic/safe)
        fallback_agents = ["orchestrator", "rag", "research", "calculator", "summarizer", "validator"]
        agent_specs = []
        for node in fallback_agents:
            display = node.replace("_", " ").title()
            if node == "orchestrator":
                atype = "orchestrator"
            elif node in ("summarizer", "validator"):
                atype = "utility"
            else:
                atype = "domain"
            agent_specs.append(AgentSpec(name=node, display_name=display, agent_type=atype))

        return AgentsMeta(
            framework="LangGraph",
            orchestrator_name="orchestrator",
            num_agents=len(agent_specs),
            agents=agent_specs,
        )


# ============================================================
# 8. Tools
# ============================================================

def collect_tools() -> list[ToolSpec]:
    """Introspect all registered LangChain tools via tools.__all__."""
    tool_specs: list[ToolSpec] = []

    try:
        import tools as tools_pkg

        all_names: list[str] = getattr(tools_pkg, "__all__", [])
        for name in all_names:
            try:
                tool_obj = getattr(tools_pkg, name)

                # Only process actual LangChain tool objects
                if not callable(tool_obj):
                    continue

                tool_name = getattr(tool_obj, "name", name)
                description = getattr(tool_obj, "description", None)

                # Extract JSON Schema from the tool's args schema
                input_schema: Optional[dict] = None
                args_schema = getattr(tool_obj, "args_schema", None)
                if args_schema is not None:
                    try:
                        if hasattr(args_schema, "model_json_schema"):
                            input_schema = args_schema.model_json_schema()
                        elif hasattr(args_schema, "schema"):
                            input_schema = args_schema.schema()
                    except Exception:
                        input_schema = None

                tool_specs.append(
                    ToolSpec(
                        tool_name=tool_name,
                        description=(description or "").strip(),
                        input_schema=input_schema,
                        output_type="str",
                    )
                )
            except Exception as inner_exc:
                logger.debug(f"Could not introspect tool '{name}': {inner_exc}")
                continue

    except Exception as exc:
        logger.warning(f"collect_tools failed: {exc}")

    return tool_specs


# ============================================================
# 9. Observability Configuration
# ============================================================

def collect_observability_config() -> ObservabilityConfig:
    """Read observability configuration from settings and env vars."""
    try:
        from config.settings import get_settings
        settings = get_settings()

        langsmith_enabled = bool(settings.langchain_api_key and settings.langchain_tracing_v2)
        otel_enabled = bool(os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT"))

        return ObservabilityConfig(
            langsmith_enabled=langsmith_enabled,
            project_name=settings.langchain_project or None,
            tracing_enabled=settings.langchain_tracing_v2,
            evaluation_enabled=True,     # evaluations/runner.py exists
            realtime_evaluation_enabled=True,   # evaluations/real_time.py exists
            offline_evaluation_enabled=True,    # evaluations/runner.py exists
            opentelemetry_enabled=otel_enabled,
        )
    except Exception as exc:
        logger.warning(f"collect_observability_config failed: {exc}")
        return ObservabilityConfig(
            langsmith_enabled=False,
            tracing_enabled=False,
            evaluation_enabled=False,
            realtime_evaluation_enabled=False,
            offline_evaluation_enabled=False,
        )


# ============================================================
# 10. Security Configuration
# ============================================================

def collect_security_config() -> SecurityConfig:
    """Collect security integration configuration."""
    try:
        from config.settings import get_settings
        settings = get_settings()

        sbom_path = PROJECT_ROOT / "sbom" / "bom.json"
        defectdojo_enabled = bool(settings.defectdojo_url and settings.defectdojo_api_key)

        return SecurityConfig(
            cyclonedx_enabled=True,     # security/sbom.py + cdxgen is wired in
            defectdojo_enabled=defectdojo_enabled,
            sbom_generation_enabled=True,
            sbom_file_path=str(sbom_path.relative_to(PROJECT_ROOT)) if sbom_path.exists() else str(sbom_path.relative_to(PROJECT_ROOT)),
            secrets_loaded_from=".env (pydantic-settings)",
            environment_variables_source=".env / OS environment variables",
        )
    except Exception as exc:
        logger.warning(f"collect_security_config failed: {exc}")
        return SecurityConfig(
            cyclonedx_enabled=False,
            defectdojo_enabled=False,
            sbom_generation_enabled=False,
        )
