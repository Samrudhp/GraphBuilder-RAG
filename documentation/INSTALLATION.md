# Installation Guide

## Quick Start (Recommended)

### Core Installation (All Platforms)
```bash
# Create virtual environment
python -m venv venv

# Activate (Mac/Linux)
source venv/bin/activate

# Activate (Windows)
venv\Scripts\activate

# Install core dependencies
pip install --upgrade pip
pip install -r requirements-core.txt
```

## Platform-Specific Notes

### macOS
```bash
# Install system dependencies
brew install tesseract           # For OCR (optional)
brew install mongodb-community   # Database
brew install redis              # Task queue
brew install neo4j              # Graph database
brew install ollama             # LLM

# Start services
brew services start mongodb-community
brew services start redis
brew services start neo4j
ollama serve &

# Pull LLM model (for extraction only)
ollama pull deepseek-r1:1.5b

# Get Groq API key for Q&A reasoning
# Visit: https://console.groq.com/keys (free tier available)
```

### Windows
```powershell
# Install system dependencies with Chocolatey
choco install tesseract          # For OCR (optional)
choco install mongodb           # Database
choco install redis-64          # Task queue  
choco install neo4j-community   # Graph database

# Install Ollama from: https://ollama.ai/download

# Start services (run as Administrator)
net start MongoDB
redis-server --service-start
neo4j start

# Pull LLM model (for extraction only)
ollama pull deepseek-r1:1.5b

# Get Groq API key from: https://console.groq.com/keys
```

### Linux (Ubuntu/Debian)
```bash
# Install system dependencies
sudo apt-get update
sudo apt-get install tesseract-ocr  # For OCR (optional)

# MongoDB
wget -qO - https://www.mongodb.org/static/pgp/server-7.0.asc | sudo apt-key add -
echo "deb [ arch=amd64,arm64 ] https://repo.mongodb.org/apt/ubuntu jammy/mongodb-org/7.0 multiverse" | sudo tee /etc/apt/sources.list.d/mongodb-org-7.0.list
sudo apt-get update
sudo apt-get install -y mongodb-org

# Redis
sudo apt-get install redis-server

# Neo4j
wget -O - https://debian.neo4j.com/neotechnology.gpg.key | sudo apt-key add -
echo 'deb https://debian.neo4j.com stable latest' | sudo tee /etc/apt/sources.list.d/neo4j.list
sudo apt-get update
sudo apt-get install neo4j

# Ollama
curl https://ollama.ai/install.sh | sh

# Start services
sudo systemctl start mongod
sudo systemctl start redis-server
sudo systemctl start neo4j
ollama serve &

# Pull LLM model (for extraction only)
ollama pull deepseek-r1:1.5b

# Get Groq API key from: https://console.groq.com/keys
```

## Optional Features

### Advanced Document Processing
```bash
pip install -r requirements-optional.txt
```

### Development Tools
```bash
pip install -r requirements-optional.txt
pre-commit install  # Git hooks for code quality
```

## Known Issues & Solutions

### 1. PyTorch Installation (Windows)
If torch fails to install, use CPU-only wheel:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### 2. FAISS on Windows
If faiss-cpu fails:
```bash
pip install faiss-cpu --no-cache-dir
```

### 3. lxml Compilation Errors
If lxml fails (rare on modern systems):
```bash
# Mac
brew install libxml2 libxslt
pip install lxml --no-binary lxml

# Ubuntu
sudo apt-get install libxml2-dev libxslt1-dev
```

### 4. Neo4j Connection Issues
Default password needs to be changed on first run:
```bash
# Access Neo4j browser: http://localhost:7474
# Default credentials: neo4j/neo4j
# Update .env with new password
```

## Verification

Test if all core services work:
```bash
python -c "import pymongo; import redis; import neo4j; import httpx; print('✅ All imports successful')"
```

## Minimal Test Run
```bash
# Set up configuration
cp .env.example .env

# Start Celery worker
celery -A workers worker --loglevel=info

# In another terminal, start API
uvicorn api.main:app --reload
```
