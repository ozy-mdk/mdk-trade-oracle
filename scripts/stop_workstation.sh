#!/usr/bin/env bash
# ==============================================================================
# MDK Trading Oracle — Workstation Shutdown Script
# Gracefully stops backend and frontend processes.
# ==============================================================================

echo "Stopping MDK Trading Oracle services..."

# Stop backend on port 8000
BACKEND_PID=$(lsof -ti :8000 2>/dev/null || true)
if [ -n "$BACKEND_PID" ]; then
    echo "Stopping FastAPI Backend (PID: $BACKEND_PID)..."
    kill -9 $BACKEND_PID 2>/dev/null || true
fi

# Stop frontend on port 5173
FRONTEND_PID=$(lsof -ti :5173 2>/dev/null || true)
if [ -n "$FRONTEND_PID" ]; then
    echo "Stopping Vite Frontend (PID: $FRONTEND_PID)..."
    kill -9 $FRONTEND_PID 2>/dev/null || true
fi

echo "Workstation services stopped."
