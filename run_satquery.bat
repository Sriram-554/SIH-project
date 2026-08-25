@echo off
title SatQuery AI - Launching...
cd /d "%~dp0"

echo.
echo  ======================================
echo    SatQuery AI - Agentic RS Platform
echo  ======================================
echo.

:: Activate virtual environment
call .venv\Scripts\activate.bat

:: Open browser after a short delay (runs in background)
start "" timeout /t 3 >nul && start "" "http://localhost:8501"

:: Start Streamlit
echo  Starting Streamlit on http://localhost:8501 ...
echo  Press Ctrl+C to stop the server.
echo.
python -m streamlit run app.py --server.port 8501 --browser.gatherUsageStats false

pause
