@echo off
REM DeepNexus Quick Deployment Script for Windows
REM This script automates the Docker-based deployment

setlocal enabledelayedexpansion

echo =========================================
echo DeepNexus Deployment Script (Windows)
echo =========================================

REM Configuration
set IMAGE_NAME=%1
if "!IMAGE_NAME!"=="" (
    set IMAGE_NAME=deepnexus
)

set REGISTRY=%2
set PORT=%3
if "!PORT!"=="" (
    set PORT=8000
)

REM Check Docker installation
docker --version >nul 2>&1
if errorlevel 1 (
    echo Docker is not installed. Please install Docker first.
    exit /b 1
)

REM Step 1: Build Docker image
echo.
echo Building Docker image...
docker build -t !IMAGE_NAME!:latest .
if errorlevel 1 (
    echo Failed to build Docker image
    exit /b 1
)
echo Docker image built successfully

REM Step 2: Test image locally
echo.
echo Testing image locally...
docker run --rm -p !PORT!:8000 --name deepnexus-test !IMAGE_NAME!:latest >nul 2>&1 &
set TEST_PID=!ERRORLEVEL!

timeout /t 5 /nobreak

REM Simple health check for Windows
curl http://localhost:!PORT!/api/health >nul 2>&1
if errorlevel 1 (
    echo Health check failed
    exit /b 1
)
echo Health check passed

REM Step 3: Display next steps
echo.
echo =========================================
echo Deployment preparation complete!
echo =========================================
echo.
echo Next steps:
echo 1. Choose your deployment platform:
echo    - Docker: docker run -p !PORT!:8000 !IMAGE_NAME!:latest
echo    - Docker Compose: docker-compose up
echo    - Vercel (Frontend) + Railway (Backend)
echo    - AWS EC2 with Docker
echo    - Render.com
echo.
echo 2. Configure environment variables:
echo    - Frontend: VITE_API_URL=^<backend-url^>
echo    - Backend: CORS_ORIGINS=^<frontend-url^>
echo.
echo 3. See DEPLOYMENT.md for detailed instructions
echo.
echo Environment Variable Examples:
echo VITE_API_URL=https://api.yourdomain.com
echo CORS_ORIGINS=https://yourdomain.com
