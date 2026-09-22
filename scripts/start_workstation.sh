#!/usr/bin/env bash
# ==============================================================================
# MDK Trading Oracle — Workstation Startup Script
# Automatically starts Docker TimescaleDB, FastAPI backend, and Vite frontend.
# ==============================================================================

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# Ensure Node environment is in PATH
if [ -d "$HOME/.local/node_env/bin" ]; then
    export PATH="$HOME/.local/node_env/bin:$PATH"
fi
if [ -d "/opt/homebrew/bin" ]; then
    export PATH="/opt/homebrew/bin:$PATH"
fi

echo "=========================================================="
echo " Starting MDK Trading Oracle Workstation..."
echo "=========================================================="

# 1. Start Docker Desktop if not running
if ! docker info >/dev/null 2>&1; then
    echo "[1/3] Docker is not running. Launching Docker Desktop..."
    open -a Docker
    echo "      Waiting for Docker daemon to become responsive..."
    until docker info >/dev/null 2>&1; do
        sleep 2
    done
fi

# Ensure TimescaleDB container is running
if [ "$(docker inspect -f '{{.State.Running}}' mdk-timescaledb 2>/dev/null)" != "true" ]; then
    echo "[1/3] Starting TimescaleDB container (mdk-timescaledb)..."
    docker start mdk-timescaledb
    sleep 2
else
    echo "[1/3] TimescaleDB container is running (Port 5432)."
fi

# 2. Check and start FastAPI Backend
mkdir -p "$PROJECT_ROOT/logs"
if lsof -i :8000 >/dev/null 2>&1; then
    echo "[2/3] Backend API is already running on http://127.0.0.1:8000"
else
    echo "[2/3] Launching FastAPI backend on http://127.0.0.1:8000..."
    nohup "$PROJECT_ROOT/.venv/bin/uvicorn" mdk_trading_oracle.api.app:app --host 127.0.0.1 --port 8000 > "$PROJECT_ROOT/logs/backend.log" 2>&1 &
    sleep 2
fi

# 3. Check and start Vite Frontend
if lsof -i :5173 >/dev/null 2>&1; then
    echo "[3/3] Frontend is already running on http://localhost:5173"
else
    echo "[3/3] Launching React/Vite frontend on http://localhost:5173..."
    nohup npm --prefix "$PROJECT_ROOT/frontend" run dev > "$PROJECT_ROOT/logs/frontend.log" 2>&1 &
    sleep 2
fi

echo "=========================================================="
echo " Workstation is fully operational!"
echo " 👉 Frontend: http://localhost:5173/"
echo " 👉 Backend API Docs: http://127.0.0.1:8000/docs"
echo "=========================================================="
