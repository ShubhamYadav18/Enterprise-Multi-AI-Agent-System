@echo off
echo ============================================================
echo  Enterprise Multi-Agent AI System - Setup
echo ============================================================
echo.

:: Create virtual environment
if not exist ".venv" (
    echo [1/4] Creating virtual environment...
    python -m venv .venv
) else (
    echo [1/4] Virtual environment already exists.
)

:: Activate virtual environment
echo [2/4] Activating virtual environment...
call .venv\Scripts\activate.bat

:: Upgrade pip
echo [3/4] Upgrading pip...
python -m pip install --upgrade pip --quiet

:: Install dependencies
echo [4/4] Installing dependencies from requirements.txt...
pip install -r requirements.txt --quiet

echo.
echo ============================================================
echo  Setup complete!
echo ============================================================
echo.
echo  Next steps:
echo    1. Copy .env.example to .env
echo    2. Fill in your API keys in .env
echo    3. Run: run_demo.bat
echo.
pause
