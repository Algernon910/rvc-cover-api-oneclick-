@echo off
title RVC Cover Backend - Installer
setlocal
cd /d "%~dp0"

echo ==============================================================
echo  RVC cover backend - one click installer
echo --------------------------------------------------------------
echo  Requires : Python 3.10-3.12, Git, NVIDIA GPU (CUDA 12.8 driver)
echo  Downloads: about 3-4 GB, takes 10-20 minutes
echo  Target   : .\runtime  (override with: install.bat --dir D:\path)
echo ==============================================================
echo.

where python >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found in PATH.
    echo         Install Python 3.12 from https://www.python.org/downloads/
    echo         and tick "Add python.exe to PATH" during setup, then run this again.
    echo.
    pause
    exit /b 1
)

for /f "delims=" %%v in ('python -c "import sys;print(sys.version.split()[0])"') do echo Python found: %%v
echo.
echo Starting installer - keep this window open.
echo.

python "%~dp0install.py" %*
set "RC=%errorlevel%"

echo.
if not "%RC%"=="0" (
    echo [FAILED] installer exited with code %RC% - read the messages above.
) else (
    echo [OK] installer finished successfully.
)
pause
endlocal
