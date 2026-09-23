@echo off
setlocal EnableExtensions
cd /d "%~dp0"

set "VENV_DIR=.venv"
set "PYTHON=%VENV_DIR%\Scripts\python.exe"
set "REQUIREMENTS=requirements.txt"

if not exist "%PYTHON%" (
    echo [.venv not found — setting up environment]
    echo.

    where py >nul 2>&1
    if %ERRORLEVEL%==0 (
        py -3 -m venv "%VENV_DIR%"
    ) else (
        python -m venv "%VENV_DIR%"
    )

    if not exist "%PYTHON%" (
        echo ERROR: Could not create virtual environment.
        echo Install Python 3 from https://www.python.org/downloads/
        echo and ensure "py" or "python" is available on PATH.
        pause
        exit /b 1
    )

    if not exist "%REQUIREMENTS%" (
        echo ERROR: Missing %REQUIREMENTS% in %CD%
        pause
        exit /b 1
    )

    echo Upgrading pip...
    "%PYTHON%" -m pip install --upgrade pip
    if errorlevel 1 goto :pip_failed

    echo Installing packages from %REQUIREMENTS%...
    "%PYTHON%" -m pip install -r "%REQUIREMENTS%"
    if errorlevel 1 goto :pip_failed

    echo.
    echo Environment ready.
    echo.
)

"%PYTHON%" "src\main.py"
set "APP_EXIT=%ERRORLEVEL%"
if not "%APP_EXIT%"=="0" (
    echo.
    echo Application exited with code %APP_EXIT%.
    pause
)
exit /b %APP_EXIT%

:pip_failed
echo.
echo ERROR: Failed to install dependencies from %REQUIREMENTS%.
pause
exit /b 1
