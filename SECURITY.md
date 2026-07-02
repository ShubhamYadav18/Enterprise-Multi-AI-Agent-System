# 🛡️ Software Supply Chain Security Integration

This module integrates Software Bill of Materials (SBOM) generation and vulnerability tracking directly into the Enterprise Multi-Agent AI System's Control Tower.

---

## 🏗️ Architecture

The security module runs independently and does not block the primary LangGraph AI routing execution workflow.

```
                  CODEBASE / REPOSITORY
                            │
                            ▼
                    CycloneDX cdxgen
                            │
                            ▼
                    CycloneDX SBOM JSON
                            │
                            ▼
                DefectDojo REST API Upload
                            │
                            ▼
                Parsed Vulnerability Findings
                            │
                            ▼
          FastAPI endpoints ◄───► Streamlit UI
```

1.  **SBOM Generation**: When triggered, `cdxgen` scans the repository and generates a CycloneDX JSON representation of Python dependencies (including transitive dependencies and package versions).
2.  **DefectDojo Scan Import**: The generated SBOM is uploaded to DefectDojo using the official API v2.
3.  **Vulnerability Reporting**: The module pulls active findings from DefectDojo, parsing and classifying them by severity levels (Critical, High, Medium, Low, Info).
4.  **Trace Session Decorator**: Every AI conversation session is tagged with the current active software version (SBOM version) and security status, ensuring auditability.

---

## ⚙️ Setup & Installation

### 1. Install CycloneDX Generator (cdxgen)
SBOM generation requires `cdxgen`. Ensure you have Node.js installed, then install `cdxgen` globally:

```bash
npm install -g @cyclonedx/cdxgen
```

*If `cdxgen` is not installed globally, the system will dynamically attempt to execute it using `npx @cyclonedx/cdxgen` as a fallback.*

### 2. Configure DefectDojo Credentials
Add the following configuration lines to your `.env` file:

```env
# --- DefectDojo Integration ---
DEFECTDOJO_URL=http://localhost:8080
DEFECTDOJO_API_KEY=your_defectdojo_personal_api_token
DEFECTDOJO_PRODUCT=enterprise-multi-agent
DEFECTDOJO_ENGAGEMENT=multi-agent-sbom-scan
```

*Note: If the product or engagement names do not exist, the module will dynamically query the DefectDojo API to create them. If credentials are left blank, the module will operate in local offline mode.*

---

## 🔗 FastAPI REST API Reference

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `POST` | `/api/security/sbom` | Generate a CycloneDX SBOM locally (does not upload). |
| `POST` | `/api/security/upload` | Generate SBOM + upload to DefectDojo + retrieve findings. |
| `GET` | `/api/security/findings` | Retrieve all parsed vulnerability findings. |
| `GET` | `/api/security/summary` | Fetch summary count of vulnerabilities (Critical, High, etc.). |
| `GET` | `/api/security/dependencies` | Retrieve hierarchical dependency tree. |
| `GET` | `/api/security/history` | List scan run history log. |

---

## 🖥️ User Interface Views

The Streamlit dashboard has been updated to include three primary control screens:

1.  **💬 Chat Assistant**: The original LangGraph agentic chat interface.
2.  **📊 AI Control Tower**: Unified views including:
    *   **Execution Sessions**: Real-time tokens, duration, cost, and validation status.
    *   **Evaluators**: Batch LangSmith evaluator statistics.
    *   **Security Summary**: Global codebase security status metrics.
3.  **🛡️ Security Dashboard**: Interactive pane to:
    *   Generate a new SBOM.
    *   Execute scans (via DefectDojo).
    *   Filter, sort, and search findings in the vulnerability table.
    *   Browse the hierarchical dependency tree.
    *   Download report logs.
