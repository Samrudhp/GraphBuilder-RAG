# Installation Checklist

## ✅ Pre-Installation

- [ ] macOS (or Linux/Windows with appropriate package manager)
- [ ] 16GB+ RAM
- [ ] ~10GB free disk space
- [ ] Internet connection for downloads

## ✅ Step 1: Install Required Services

### macOS (using Homebrew)

```bash
# Install Homebrew if needed
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install services
- [ ] brew install mongodb-community
- [ ] brew install neo4j
- [ ] brew install redis
- [ ] brew install ollama
- [ ] brew install tesseract
- [ ] brew install poppler
```

### Verify Installations

```bash
- [ ] mongod --version
- [ ] neo4j version
- [ ] redis-server --version
- [ ] ollama --version
- [ ] tesseract --version
- [ ] pdfinfo -v
```

## ✅ Step 2: Start Services

```bash
- [ ] brew services start mongodb-community
- [ ] brew services start neo4j
- [ ] brew services start redis
- [ ] ollama serve &
```

### Verify Services Running

```bash
- [ ] brew services list (all should show "started")
- [ ] curl http://localhost:27017 (MongoDB)
- [ ] curl http://localhost:7474 (Neo4j)
- [ ] redis-cli ping (should return PONG)
- [ ] curl http://localhost:11434/api/tags (Ollama)
```

## ✅ Step 3: Neo4j Setup

```bash
- [ ] Open http://localhost:7474 in browser
- [ ] Login with default: neo4j/neo4j
- [ ] Set new password (e.g., "password")
- [ ] Update .env with NEO4J_PASSWORD=<your-password>
```

## ✅ Step 4: Clone Project

```bash
- [ ] git clone <repository-url>
- [ ] cd graphbuilder-rag
```

## ✅ Step 5: Setup Project

```bash
- [ ] chmod +x setup.sh
- [ ] ./setup.sh
```

This will:
- [ ] Create virtual environment
- [ ] Install Python dependencies
- [ ] Create data directories
- [ ] Pull Ollama model for extraction (deepseek-r1:1.5b)
- [ ] Initialize database indexes

### Manual Setup (if script fails)

```bash
- [ ] python3 -m venv venv
- [ ] source venv/bin/activate
- [ ] pip install -r requirements.txt
- [ ] mkdir -p data/faiss data/temp
- [ ] ollama pull deepseek-r1:1.5b
- [ ] Get Groq API key from https://console.groq.com/keys
- [ ] cp .env.example .env
```

## ✅ Step 6: Configuration

```bash
- [ ] cp .env.example .env (if not done)
- [ ] Edit .env file:
  - [ ] NEO4J_PASSWORD=<your-password>
  - [ ] GROQ_API_KEY=<your-groq-api-key>
  - [ ] Check MONGODB_URI=mongodb://localhost:27017
  - [ ] Check NEO4J_URI=bolt://localhost:7687
  - [ ] Check REDIS_URI=redis://localhost:6379/0
  - [ ] Check OLLAMA_BASE_URL=http://localhost:11434
```

## ✅ Step 7: Run Application

### Option A: Tmux (Recommended)

```bash
- [ ] chmod +x run.sh
- [ ] ./run.sh
```

This starts all services in tmux session "graphbuilder"

### Option B: Manual (Separate Terminals)

Terminal 1 - API:
```bash
- [ ] source venv/bin/activate
- [ ] python -m api.main
```

Terminal 2 - Worker:
```bash
- [ ] source venv/bin/activate
- [ ] celery -A workers.tasks worker --loglevel=info --concurrency=4
```

Terminal 3 - Beat:
```bash
- [ ] source venv/bin/activate
- [ ] celery -A workers.tasks beat --loglevel=info
```

Terminal 4 - Agents (Optional):
```bash
- [ ] source venv/bin/activate
- [ ] python -m agents.agents
```

Terminal 5 - Flower (Optional):
```bash
- [ ] source venv/bin/activate
- [ ] celery -A workers.tasks flower --port=5555
```

## ✅ Step 8: Verify Installation

### Test API Health

```bash
- [ ] curl http://localhost:8000/health
```

Should return:
```json
{
  "status": "healthy",
  "services": {
    "mongodb": "up",
    "neo4j": "up"
  }
}
```

### Test API Docs

```bash
- [ ] open http://localhost:8000/docs
```

Should show Swagger UI with all endpoints

### Test Ingestion

```bash
- [ ] curl -X POST http://localhost:8000/api/v1/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "source": "https://en.wikipedia.org/wiki/Artificial_intelligence",
    "source_type": "HTML"
  }'
```

Should return document_id and status "processing"

### Test Query (after ingestion completes)

```bash
- [ ] curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is artificial intelligence?",
    "top_k_semantic": 5,
    "graph_depth": 2
  }'
```

Should return answer with verification status

## ✅ Step 9: Access Web Interfaces

- [ ] API: http://localhost:8000
- [ ] API Docs: http://localhost:8000/docs
- [ ] Neo4j Browser: http://localhost:7474
- [ ] Flower (if running): http://localhost:5555

## 🎉 Installation Complete!

If all checkboxes above are checked, your GraphBuilder-RAG system is fully operational.

## 📚 Next Steps

1. Read QUICKSTART.md for common commands
2. Read TESTING.md for example workflows
3. Customize .env for your use case
4. Add your own ontology rules in shared/config/settings.py
5. Start ingesting your documents!

## ⚠️ Troubleshooting

If something didn't work:

1. Check service status: `brew services list`
2. View logs in SETUP.md troubleshooting section
3. Restart services: `brew services restart <service>`
4. Check .env configuration
5. Verify ports are not in use: `lsof -i :<port>`

## 🆘 Getting Help

- Review SETUP.md for detailed instructions
- Check TESTING.md for examples
- Review error messages in terminal
- Ensure all prerequisites are met
- Verify .env configuration is correct
