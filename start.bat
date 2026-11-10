@echo off
title Render3D AI Engine
echo =======================================================
echo          Render3D AI Engine - Starting Server
echo =======================================================
echo.

python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python not found. Please install Python and add to PATH.
    pause
    exit /b 1
)

if not exist "storage\assets" (
    mkdir storage\assets
)

echo [1/2] Checking dependencies...
python -m pip install -r requirements.txt --quiet

echo [2/2] Opening Swagger Docs in browser...
start http://localhost:8000/docs

echo.
echo Server running at: http://localhost:8000
echo Interactive Docs:  http://localhost:8000/docs
echo Press Ctrl+C to stop.
echo.

python -m uvicorn services.gateway:app --host 0.0.0.0 --port 8000 --reload
pause
