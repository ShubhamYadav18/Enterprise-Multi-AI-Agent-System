@echo off
echo ============================================================
echo  Enterprise Multi-Agent AI System - Demo
echo ============================================================
echo.

:: Check for .env
if not exist ".env" (
    echo  ERROR: .env file not found!
    echo  Copy .env.example to .env and fill in your API keys.
    echo.
    pause
    exit /b 1
)

:: Activate virtual environment
if not exist ".venv\Scripts\activate.bat" (
    echo  ERROR: Virtual environment not found!
    echo  Run setup.bat first.
    echo.
    pause
    exit /b 1
)

call .venv\Scripts\activate.bat

:: Run document ingestion if vectorstore doesn't exist
if not exist "vectorstore" (
    echo [1/3] Running initial document ingestion...
    python ingest.py
) else (
    echo [1/3] Vector store already exists. Skipping ingestion.
)

:: Start FastAPI server in background
echo [2/3] Starting FastAPI server on http://localhost:8000 ...
start /b python main.py

:: Give the server a moment to start
timeout /t 3 /nobreak >nul

:: Start Streamlit UI
echo [3/3] Starting Streamlit dashboard...
echo.
echo  Dashboard: http://localhost:8501
echo  API Docs:  http://localhost:8000/docs
echo.
streamlit run ui/app.py --server.port 8501
