@echo off
title RVC Cover Service (127.0.0.1:8000)
setlocal

rem ==============================================================
rem  TEMPLATE ONLY.
rem  Running install.bat already generates a ready-to-use launcher at
rem      runtime\start-backend.bat
rem  with every path filled in - use that one unless you moved things.
rem
rem  If you do need this template, edit the three paths below.
rem ==============================================================
set "PROJECT_DIR=C:\path\to\Retrieval-based-Voice-Conversion-FastAPI"
set "PYTHON_EXE=C:\path\to\.venv\Scripts\python.exe"
set "WATCHER=C:\path\to\ai-cover-bot\tools\watch_and_open.py"
rem ==============================================================

if not exist "%PYTHON_EXE%" (
    echo [ERROR] Python not found: %PYTHON_EXE%
    echo Please edit PROJECT_DIR / PYTHON_EXE at the top of this file.
    pause
    exit /b 1
)

rem If port 8000 is already listening, the service is up: just open the docs.
netstat -ano | findstr ":8000" | findstr "LISTENING" >nul
if %errorlevel%==0 (
    echo [INFO] Port 8000 is already listening - the service is running.
    echo Opening http://127.0.0.1:8000/docs
    start "" http://127.0.0.1:8000/docs
    echo.
    echo To restart: close the old black window first, then run this script again.
    pause
    exit /b 0
)

cd /d "%PROJECT_DIR%"
set PYTHONPATH=%PROJECT_DIR%;
set PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.8

echo Starting RVC service. First model load takes 20-40 seconds.
echo Docs: http://127.0.0.1:8000/docs
echo The browser opens automatically as soon as the service is ready.
echo KEEP THIS WINDOW OPEN - closing it stops the service.
echo.

if exist "%WATCHER%" start "RVC-ReadyWatcher" /min "%PYTHON_EXE%" "%WATCHER%"

"%PYTHON_EXE%" rvc_fastapi.py

echo.
echo Service exited. Check the messages above for errors.
pause
endlocal
