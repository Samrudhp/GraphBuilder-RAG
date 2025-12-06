#!/bin/bash

# Quick start script - runs all services in tmux sessions

set -e

# Check if tmux is installed
if ! command -v tmux &> /dev/null; then
    echo "❌ tmux is not installed"
    echo "Install with: brew install tmux"
    exit 1
fi

# Activate virtual environment
source venv/bin/activate

# Create tmux session
SESSION_NAME="graphbuilder"

# Kill existing session if it exists
tmux kill-session -t $SESSION_NAME 2>/dev/null || true

# Create new session with API
tmux new-session -d -s $SESSION_NAME -n api "source venv/bin/activate && python -m api.main"

# Create worker window
tmux new-window -t $SESSION_NAME -n worker "source venv/bin/activate && celery -A workers.tasks worker --loglevel=info --concurrency=4"

# Create beat window
tmux new-window -t $SESSION_NAME -n beat "source venv/bin/activate && celery -A workers.tasks beat --loglevel=info"

# Create agents window (optional)
tmux new-window -t $SESSION_NAME -n agents "source venv/bin/activate && python -m agents.agents"

# Create flower window (optional)
tmux new-window -t $SESSION_NAME -n flower "source venv/bin/activate && celery -A workers.tasks flower --port=5555"

# Attach to session
echo "✅ All services started in tmux session: $SESSION_NAME"
echo ""
echo "To attach: tmux attach -t $SESSION_NAME"
echo "To detach: Press Ctrl+B, then D"
echo "To switch windows: Press Ctrl+B, then window number (0-4)"
echo "To kill session: tmux kill-session -t $SESSION_NAME"
echo ""
echo "Windows:"
echo "  0: API Server (port 8000)"
echo "  1: Celery Worker"
echo "  2: Celery Beat"
echo "  3: Agents"
echo "  4: Flower (port 5555)"
echo ""

tmux attach -t $SESSION_NAME
