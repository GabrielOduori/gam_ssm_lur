@echo off
setlocal

:: Prefer "python", fall back to "py" (Windows Python Launcher)
python --version >nul 2>&1
if errorlevel 1 (
    py --version >nul 2>&1
    if errorlevel 1 (
        echo Python not found. Install from https://www.python.org/downloads/
        exit /b 1
    )
    set PYTHON=py
) else (
    set PYTHON=python
)

%PYTHON% -m pip install -e .
