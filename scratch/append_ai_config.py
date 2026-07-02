"""Script to append AI Config page to ui/app.py"""
import sys
from pathlib import Path

root = Path(r"d:\MULTI AGENT\enterprise_multi_agent")

ai_config_code = r'''

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
            _arows = [{"Agent": a.display_name, "Type": a.agent_type, "Parallel": "Yes" if a.parallel_capable else "No", "Tools": ", ".join(a.tools) if a.tools else "\u2014"} for a in _agts.agents]
            st.dataframe(pd.DataFrame(_arows), use_container_width=True, hide_index=True)

        # 8. Tools
        with st.expander("Registered Tools", expanded=False):
            if manifest.tools:
                import pandas as pd
                _trows = [{"Tool": t.tool_name, "Owner Agent": t.agent_owner or "\u2014", "Output Type": t.output_type, "Description": (t.description or "")[:90]} for t in manifest.tools]
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
'''

app_file = root / "ui" / "app.py"
content = app_file.read_text(encoding="utf-8").rstrip()
content = content + "\n" + ai_config_code.strip() + "\n"
app_file.write_text(content, encoding="utf-8")
print(f"Done. Total lines: {len(content.splitlines())}")
