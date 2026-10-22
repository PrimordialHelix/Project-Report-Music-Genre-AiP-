@echo off
echo ========================================================
echo   MUSICVERSE - Installing Dependencies
echo ========================================================

where py >nul 2>nul
if %ERRORLEVEL% equ 0 (
    echo Using Windows py launcher to install requirements...
    py -m pip install -r requirements.txt
    goto done
)

where python >nul 2>nul
if %ERRORLEVEL% equ 0 (
    echo Using python to install requirements...
    python -m pip install -r requirements.txt
    goto done
)

echo [ERROR] Neither 'py' nor 'python' was found on your PATH.
echo Please ensure Python is installed and added to PATH.
pause
exit /b 1

:done
echo.
echo ========================================================
echo   Dependencies successfully installed/verified!
echo   You can now launch the app quickly with run.bat
echo ========================================================
pause
