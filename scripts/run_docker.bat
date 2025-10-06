@echo off
echo.
echo =====================================================
echo    HEADSPACE - DOCKER MODE
echo =====================================================
echo.
echo Building Docker image...
docker-compose build

echo.
echo Starting Headspace container...
echo Server will be available at: http://localhost:8000
echo Press Ctrl+C to stop
echo.
echo =====================================================
echo.

docker-compose up

echo.
echo Container stopped.
pause