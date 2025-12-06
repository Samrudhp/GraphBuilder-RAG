# GraphBuilder-RAG Quick Reference

## 🚀 Installation & Setup

```bash
# 1. Install services (macOS)
brew install mongodb-community neo4j redis ollama tesseract poppler

# 2. Start services
brew services start mongodb-community
brew services start neo4j
brew services start redis
ollama serve &

# 3. Pull models
ollama pull deepseek-r1:1.5b
ollama pull deepseek-r1:7b

# 4. Setup project
chmod +x setup.sh
./setup.sh

# 5. Run all services (tmux)
chmod +x run.sh
./run.sh
```

## 📝 Common Commands

### Start Services Manually

```bash
# Activate environment
source venv/bin/activate

# Terminal 1: API (http://localhost:8000)
python -m api.main

# Terminal 2: Worker
celery -A workers.tasks worker --loglevel=info --concurrency=4

# Terminal 3: Beat (periodic tasks)
celery -A workers.tasks beat --loglevel=info

# Terminal 4: Agents (optional)
python -m agents.agents

# Terminal 5: Flower (monitoring, optional)
celery -A workers.tasks flower --port=5555
```

### Stop Services

```bash
# Tmux session
tmux kill-session -t graphbuilder

# macOS services
brew services stop mongodb-community
brew services stop neo4j
brew services stop redis
pkill -f "ollama serve"
```

## 🔧 API Endpoints

### Ingest Document

```bash
# From URL
curl -X POST http://localhost:8000/api/v1/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "source": "https://example.com",
    "source_type": "HTML",
    "metadata": {"topic": "example"}
  }'

# Upload file
curl -X POST http://localhost:8000/api/v1/ingest/file \
  -F "file=@document.pdf" \
  -F "source_type=PDF"
```

### Query System

```bash
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is artificial intelligence?",
    "top_k_semantic": 5,
    "graph_depth": 2,
    "min_confidence": 0.7
  }'
```

### Check Document Status

```bash
curl http://localhost:8000/api/v1/documents/{document_id}
```

### System Stats

```bash
curl http://localhost:8000/api/v1/stats
```

### Health Check

```bash
curl http://localhost:8000/health
```

## 🔍 Database Access

### MongoDB

```bash
mongosh
use graphbuilder_rag

# View collections
show collections

# Query documents
db.raw_documents.find().limit(5)
db.validated_triples.find({"validation_result.confidence": {$gte: 0.9}}).limit(10)
```

### Neo4j

```bash
# Browser: http://localhost:7474
# Credentials: neo4j / password

# View entities
MATCH (n:Entity) RETURN n LIMIT 25

# View relationships
MATCH (n:Entity)-[r]->(m:Entity) 
RETURN n, r, m 
LIMIT 50

# Find most connected
MATCH (n:Entity)-[r]->()
RETURN n.canonical_name, count(r) as degree
ORDER BY degree DESC
LIMIT 10
```

### Redis

```bash
redis-cli

# View keys
KEYS *

# Monitor commands
MONITOR

# Clear database (careful!)
FLUSHDB
```

## 🐛 Troubleshooting

### Check Service Status

```bash
# macOS
brew services list

# Check ports
lsof -i :8000   # API
lsof -i :27017  # MongoDB
lsof -i :7687   # Neo4j
lsof -i :6379   # Redis
lsof -i :11434  # Ollama
```

### Restart Services

```bash
brew services restart mongodb-community
brew services restart neo4j
brew services restart redis

# Restart Ollama
pkill -f "ollama serve"
ollama serve &
```

### Clear Data

```bash
# Clear MongoDB
mongosh graphbuilder_rag --eval "db.dropDatabase()"

# Clear Neo4j
cypher-shell -u neo4j -p password "MATCH (n) DETACH DELETE n"

# Clear FAISS
rm -rf data/faiss/*

# Clear Redis
redis-cli FLUSHDB
```

### View Logs

```bash
# MongoDB
tail -f /opt/homebrew/var/log/mongodb/mongo.log

# Neo4j
tail -f /opt/homebrew/var/log/neo4j/neo4j.log

# API/Worker (in terminal where running)
# Logs display in real-time
```

## 📊 Monitoring

### Flower (Celery Tasks)

- URL: http://localhost:5555
- View tasks, workers, queues
- Retry failed tasks

### Prometheus Metrics

- URL: http://localhost:8000/metrics
- Export to Prometheus/Grafana

### Neo4j Browser

- URL: http://localhost:7474
- Visual graph exploration
- Run Cypher queries

## 📁 Project Structure

```
graphbuilder-rag/
├── api/main.py              # FastAPI app
├── agents/agents.py         # Autonomous agents
├── workers/tasks.py         # Celery tasks
├── services/                # Core services
│   ├── ingestion/
│   ├── normalization/
│   ├── extraction/
│   ├── embedding/
│   ├── validation/
│   ├── fusion/
│   ├── entity_resolution/
│   └── query/
├── shared/                  # Shared utilities
│   ├── config/
│   ├── models/
│   ├── database/
│   ├── prompts/
│   └── utils/
├── data/                    # Data storage
│   ├── faiss/
│   └── temp/
├── .env                     # Configuration
├── requirements.txt         # Dependencies
├── setup.sh                 # Setup script
└── run.sh                   # Run script (tmux)
```

## ⚙️ Configuration

Edit `.env` file:

```bash
# Key settings
MONGODB_URI=mongodb://localhost:27017
NEO4J_URI=bolt://localhost:7687
NEO4J_PASSWORD=password
OLLAMA_BASE_URL=http://localhost:11434
FAISS_INDEX_TYPE=IndexFlatIP
API_PORT=8000
API_WORKERS=4

# Agent intervals (seconds)
AGENT_REVERIFY_INTERVAL_SECONDS=3600
AGENT_CONFLICT_RESOLUTION_INTERVAL_SECONDS=7200
AGENT_SCHEMA_SUGGESTION_INTERVAL_SECONDS=86400
```

## 📚 Documentation

- `README.md` - Overview and quick start
- `SETUP.md` - Detailed installation guide
- `TESTING.md` - Testing workflows and examples
- `ARCHITECTURE.md` - Complete system documentation

## 🔗 URLs

- API: http://localhost:8000
- API Docs: http://localhost:8000/docs
- Neo4j: http://localhost:7474
- Flower: http://localhost:5555

## 💡 Tips

1. **Always activate venv**: `source venv/bin/activate`
2. **Use tmux for convenience**: `./run.sh` starts everything
3. **Monitor Flower**: Track task progress and failures
4. **Check logs**: If something fails, check service logs
5. **Clear data**: Use clear commands above for fresh start
6. **Adjust concurrency**: Reduce if system is slow
7. **Use health endpoint**: `/health` shows service status
