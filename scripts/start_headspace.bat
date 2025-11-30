@echo off
echo.
echo =====================================================
echo    HEADSPACE - COSMIC KNOWLEDGE SYSTEM
echo =====================================================
echo.

REM Check if virtual environment exists
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
    echo.
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Install requirements
echo Installing dependencies...
pip install -q -r requirements_headspace.txt 2>nul

echo.
echo Starting Headspace System...
echo.
echo Server will be available at: http://localhost:8000
echo Press Ctrl+C to stop the server
echo.
echo =====================================================
echo.

REM Start the server
python headspace_system.py