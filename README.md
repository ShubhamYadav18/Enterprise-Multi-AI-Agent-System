# 🤖 Enterprise Multi-Agent AI System

> Production-quality multi-agent orchestration with **LangGraph**, **LangChain**, **LangSmith**, **FAISS**, **FastAPI**, and **Streamlit**.

---

## Architecture

```
                         USER
                           │
                           ▼
                 ORCHESTRATOR AGENT
            ┌──────────┼───────────┐
            │          │           │
            ▼          ▼           ▼
       RAG Agent   Research    Calculator     ← Parallel Execution
            │          │           │
            └──────┬───┴───────────┘
                   ▼
           Summarizer Agent
                   ▼
           Validation Agent
                   ▼
             Final Response
```

### Key Features

- **LangGraph Orchestration** — StateGraph with conditional routing and parallel super-steps
- **Dynamic Routing** — Orchestrator analyzes intent and invokes only needed agents
- **Parallel Execution** — Multiple agents run simultaneously in the same super-step
- **RAG** — FAISS vector store with HuggingFace embeddings over enterprise knowledge base
- **Tool Calling** — Calculator, search, retriever, and document tools
- **LangSmith Tracing** — Full end-to-end observability with session correlation
- **LangSmith Evaluations** — 5 custom evaluators across 10 test examples
- **Streamlit Dashboard** — Professional UI with execution visualization
- **FastAPI Backend** — REST API with 6 endpoints

---

## Quick Start

### 1. Setup

```bash
# Clone/copy the project
cd enterprise_multi_agent

# Run setup (creates venv, installs dependencies)
setup.bat
```

### 2. Configure

```bash
# Copy environment template
copy .env.example .env

# Edit .env and add your API keys:
#   ANTHROPIC_API_KEY=your-key-here
#   LANGCHAIN_API_KEY=your-langsmith-key
```

### 3. Run

```bash
run_demo.bat
```

This will:
1. Build the FAISS vector index (first run only)
2. Start the FastAPI server on `http://localhost:8000`
3. Launch the Streamlit dashboard on `http://localhost:8501`

---

## Demo Scenarios

| # | Scenario | Query | Expected Route |
|---|----------|-------|---------------|
| 1 | RAG Only | "What is the company leave policy?" | RAG → Summarizer → Validator |
| 2 | Calculator Only | "Calculate growth rate from $1M to $1.5M" | Calculator → Summarizer → Validator |
| 3 | RAG + Research (Parallel) | "Compare our leave policy with industry standards" | [RAG ‖ Research] → Summarizer → Validator |
| 4 | RAG + Calculator (Parallel) | "How many sick days and what % of working days?" | [RAG ‖ Calculator] → Summarizer → Validator |
| 5 | Full Pipeline | "Compare leave policy with industry and calculate utilization" | [RAG ‖ Research ‖ Calculator] → Summarizer → Validator |

Each demo visibly shows: routing decision, invoked/skipped agents, parallel execution, tool usage, retrieved documents, validation scores, and LangSmith trace link.

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/chat` | Submit query, returns full response + execution metadata |
| `POST` | `/evaluate` | Trigger LangSmith evaluation run |
| `GET` | `/health` | Health check with system status |
| `GET` | `/agents` | List all available agents with tools |
| `GET` | `/documents` | List knowledge base documents |
| `GET` | `/trace/{session_id}` | Get LangSmith trace URL for a session |

Interactive API docs: `http://localhost:8000/docs`

---

## Project Structure

```
enterprise_multi_agent/
│
├── agents/                  # Agent implementations
│   ├── orchestrator.py      # Intent analysis & routing
│   ├── rag_agent.py         # Knowledge base retrieval
│   ├── research_agent.py    # External search & benchmarks
│   ├── calculator_agent.py  # Mathematical computations
│   ├── summarizer_agent.py  # Multi-agent output merger
│   └── validation_agent.py  # Response quality validation
│
├── graph/                   # LangGraph orchestration
│   ├── state.py             # Graph state with reducers
│   ├── nodes.py             # Node functions
│   └── builder.py           # StateGraph construction
│
├── tools/                   # Reusable LangChain tools
│   ├── calculator_tools.py  # Math operations (7 tools)
│   ├── search_tools.py      # Pluggable search provider
│   ├── retriever_tools.py   # FAISS vector search
│   └── document_tools.py    # Document management
│
├── rag/                     # RAG pipeline
│   ├── embeddings.py        # HuggingFace sentence-transformers
│   ├── document_loader.py   # Markdown loader & splitter
│   └── vectorstore.py       # FAISS index management
│
├── prompts/                 # Agent prompt templates
├── config/                  # Pydantic Settings
├── traces/                  # LangSmith session & callbacks
├── evaluations/             # LangSmith evaluators & dataset
├── api/                     # FastAPI application
├── ui/                      # Streamlit dashboard
├── data/documents/          # Knowledge base (8 markdown files)
│
├── main.py                  # Server entry point
├── ingest.py                # Document ingestion script
├── requirements.txt         # Python dependencies
├── setup.bat                # Environment setup
├── run_demo.bat             # Demo launcher
├── .env.example             # Environment template
└── .gitignore
```

---

## LangSmith Observability

Every user request generates a correlated trace tree:

```
User Query
  └── Session (unique ID)
       └── Graph Execution
            ├── Orchestrator Node
            │    └── LLM Call (routing)
            ├── RAG Node (parallel)
            │    ├── Vector Search (tool)
            │    └── LLM Call (synthesis)
            ├── Research Node (parallel)
            │    ├── Search (tool)
            │    └── LLM Call (synthesis)
            ├── Calculator Node (parallel)
            │    ├── Tool Calls (calculate, growth_rate, etc.)
            │    └── LLM Call (synthesis)
            ├── Summarizer Node
            │    └── LLM Call (merge)
            └── Validator Node
                 └── LLM Call (validation)
```

### Metadata Attached to Every Trace

- Session ID, Conversation ID, User ID
- Query Type, Invoked Agents, Tool Count
- Execution Time, Token Usage, Latency

---

## Evaluations

Run evaluations with:

```bash
python -m evaluations.runner
```

### Evaluators

| Evaluator | What it Measures |
|-----------|-----------------|
| Task Completion | Did the system produce a valid response? |
| Answer Correctness | Does the output contain expected key terms? |
| Groundedness | Are claims supported by source material? |
| Tool Selection | Were the correct agents invoked? |
| Context Relevance | Are retrieved documents relevant? |

Results are stored in LangSmith for comparison across runs.

---

## Technology Stack

| Component | Technology |
|-----------|-----------|
| Orchestration | LangGraph (StateGraph) |
| LLM | Anthropic Claude via langchain-anthropic |
| Embeddings | HuggingFace sentence-transformers (local, free) |
| Vector Store | FAISS |
| Observability | LangSmith |
| API | FastAPI + Uvicorn |
| UI | Streamlit |
| Validation | Pydantic v2 |
| Configuration | python-dotenv + pydantic-settings |
| Supply Chain Security | CycloneDX (cdxgen) & DefectDojo (See [SECURITY.md](file:///d:/MULTI%20AGENT/enterprise_multi_agent/SECURITY.md)) |


---

## Extending the System

### Adding a New Agent

1. Create `agents/new_agent.py` with a `run_new_agent()` function
2. Add a prompt in `prompts/new_agent.py`
3. Add tools in `tools/new_tools.py` (if needed)
4. Add a node in `graph/nodes.py`
5. Register the node in `graph/builder.py`
6. Update the orchestrator prompt to include the new agent

The system is designed so new agents can be added without modifying existing agent code.

---

## License

Internal use only. Enterprise demonstration project.
