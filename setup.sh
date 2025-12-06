#!/bin/bash

# GraphBuilder-RAG Local Startup Script
# Run all services locally (no Docker)

set -e

echo "================================"
echo "GraphBuilder-RAG Local Setup"
echo "================================"

# Check Python version
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed"
    exit 1
fi

PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
echo "✅ Python $PYTHON_VERSION detected"

# Check if .env exists
if [ ! -f .env ]; then
    echo "Creating .env from .env.example..."
    cp .env.example .env
    echo "⚠️  Please update .env with your configuration!"
fi

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo ""
    echo "Creating virtual environment..."
    python3 -m venv venv
    echo "✅ Virtual environment created"
fi

# Activate virtual environment
source venv/bin/activate

# Install requirements
echo ""
echo "Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt
echo "✅ Dependencies installed"

# Create data directories
echo ""
echo "Creating data directories..."
mkdir -p data/faiss data/temp
echo "✅ Data directories created"

# Check services
echo ""
echo "Checking required services..."

# Check MongoDB
if ! curl -s http://localhost:27017 > /dev/null 2>&1; then
    echo "⚠️  MongoDB is not running on localhost:27017"
    echo "   Start with: brew services start mongodb-community"
    echo "   Or install: brew install mongodb-community"
fi

# Check Neo4j
if ! curl -s http://localhost:7474 > /dev/null 2>&1; then
    echo "⚠️  Neo4j is not running on localhost:7474"
    echo "   Start with: brew services start neo4j"
    echo "   Or install: brew install neo4j"
fi

# Check Redis
if ! redis-cli ping > /dev/null 2>&1; then
    echo "⚠️  Redis is not running on localhost:6379"
    echo "   Start with: brew services start redis"
    echo "   Or install: brew install redis"
fi

# Check Ollama
if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "⚠️  Ollama is not running on localhost:11434"
    echo "   Start with: ollama serve"
    echo "   Or install: brew install ollama"
else
    echo "✅ Ollama is running"
    
    # Check models
    if ! ollama list | grep -q "deepseek-r1:1.5b"; then
        echo "   Pulling deepseek-r1:1.5b..."
        ollama pull deepseek-r1:1.5b
    fi
    
    echo "✅ DeepSeek models ready"
fi

# Initialize databases
echo ""
echo "Initializing databases..."
python3 -c "
from shared.database.mongodb import get_mongodb
from shared.database.neo4j import get_neo4j

# MongoDB
try:
    mongo = get_mongodb()
    if mongo.ping():
        mongo.create_indexes()
        print('✅ MongoDB initialized')
    else:
        print('⚠️  MongoDB connection failed')
except Exception as e:
    print(f'⚠️  MongoDB error: {e}')

# Neo4j
try:
    neo4j = get_neo4j()
    if neo4j.ping():
        neo4j.create_constraints_and_indexes()
        print('✅ Neo4j initialized')
    else:
        print('⚠️  Neo4j connection failed')
except Exception as e:
    print(f'⚠️  Neo4j error: {e}')
" 2>/dev/null || echo "⚠️  Database initialization skipped (check connections)"

echo ""
echo "================================"
echo "Setup Complete!"
echo "================================"
echo ""
echo "To start the services, run in separate terminals:"
echo ""
echo "1. API Server:"
echo "   python -m api.main"
echo ""
echo "2. Celery Worker:"
echo "   celery -A workers.tasks worker --loglevel=info --concurrency=4"
echo ""
echo "3. Celery Beat (periodic tasks):"
echo "   celery -A workers.tasks beat --loglevel=info"
echo ""
echo "4. Agents (optional):"
echo "   python -m agents.agents"
echo ""
echo "5. Flower (task monitoring, optional):"
echo "   celery -A workers.tasks flower --port=5555"
echo ""
echo "Access points:"
echo "  - API:      http://localhost:8000"
echo "  - API Docs: http://localhost:8000/docs"
echo "  - Neo4j:    http://localhost:7474"
echo "  - Flower:   http://localhost:5555 (if running)"
echo ""
