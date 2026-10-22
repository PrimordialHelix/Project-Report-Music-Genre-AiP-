@echo off
title MUSICVERSE — Genre Intelligence Studio

where py >nul 2>nul
if %ERRORLEVEL% equ 0 (
    py -m streamlit run app.py
    exit /b 0
)

where python >nul 2>nul
if %ERRORLEVEL% equ 0 (
    python -m streamlit run app.py
    exit /b 0
)

echo [ERROR] Neither 'py' nor 'python' was found on your PATH.
echo Please run req.bat first or ensure Python 3.10+ is installed.
pause
exit /b 1
