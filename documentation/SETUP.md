# Setup and Deployment Guide

## Prerequisites

- **macOS** (or Linux/Windows with appropriate package manager)
- **Python 3.11+**
- **Homebrew** (macOS) or appropriate package manager
- **16GB+ RAM** recommended
- **GPU optional** (for faster LLM inference with Ollama)

## Quick Start

### 1. Install System Dependencies

**macOS:**
```bash
# Install Homebrew if not already installed
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install required services
brew install mongodb-community neo4j redis ollama

# Install document processing tools
brew install tesseract poppler
```

**Ubuntu/Debian:**
```bash
# MongoDB
wget -qO - https://www.mongodb.org/static/pgp/server-7.0.asc | sudo apt-key add -
echo "deb [ arch=amd64,arm64 ] https://repo.mongodb.org/apt/ubuntu jammy/mongodb-org/7.0 multiverse" | sudo tee /etc/apt/sources.list.d/mongodb-org-7.0.list
sudo apt-get update
sudo apt-get install -y mongodb-org

# Neo4j
wget -O - https://debian.neo4j.com/neotechnology.gpg.key | sudo apt-key add -
echo 'deb https://debian.neo4j.com stable latest' | sudo tee /etc/apt/sources.list.d/neo4j.list
sudo apt-get update
sudo apt-get install neo4j

# Redis
sudo apt-get install redis-server

# Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Document processing
sudo apt-get install tesseract-ocr poppler-utils ghostscript
```

### 2. Start Required Services

**macOS:**
```bash
# Start MongoDB
brew services start mongodb-community

# Start Neo4j
brew services start neo4j

# Start Redis
brew services start redis

# Start Ollama (in a separate terminal, or run as service)
ollama serve
```

**Linux:**
```bash
sudo systemctl start mongod
sudo systemctl start neo4j
sudo systemctl start redis
ollama serve &
```

### 3. Setup the Project

```bash
# Clone repository
git clone <repository-url>
cd graphbuilder-rag

# Run setup script
chmod +x setup.sh
./setup.sh
```

This will:
- Create virtual environment
- Install Python dependencies
- Create data directories
- Pull Ollama model for extraction (deepseek-r1:1.5b)
- Initialize database indexes

**Note:** You'll need to get a Groq API key separately from https://console.groq.com/keys (free tier available)

### 4. Start the Application

**Option A: Manual (separate terminals)**

```bash
# Terminal 1: API Server
source venv/bin/activate
python -m api.main

# Terminal 2: Celery Worker
source venv/bin/activate
celery -A workers.tasks worker --loglevel=info --concurrency=4

# Terminal 3: Celery Beat
source venv/bin/activate
celery -A workers.tasks beat --loglevel=info

# Terminal 4 (optional): Agents
source venv/bin/activate
python -m agents.agents

# Terminal 5 (optional): Flower monitoring
source venv/bin/activate
celery -A workers.tasks flower --port=5555
```

**Option B: Tmux (all in one session)**

```bash
chmod +x run.sh
./run.sh
```

This starts all services in a tmux session. Use:
- `Ctrl+B, then 0-4` to switch between windows
- `Ctrl+B, then D` to detach
- `tmux attach -t graphbuilder` to reattach

### 5. Verify Installation

```bash
# Check API health
curl http://localhost:8000/health

# View API docs
open http://localhost:8000/docs  # macOS
# or visit http://localhost:8000/docs in browser
```

## Configuration

### Environment Variables

Key configuration options in `.env`:

- **MongoDB:**
  - `MONGODB_URI`: MongoDB connection string
  - `MONGODB_DATABASE`: Database name

- **Neo4j:**
  - `NEO4J_URI`: Neo4j Bolt connection string
  - `NEO4J_USER`: Username (default: neo4j)
  - `NEO4J_PASSWORD`: Password

- **Ollama (for extraction only):**
  - `OLLAMA_BASE_URL`: Ollama API endpoint
  - `OLLAMA_EXTRACTION_MODEL`: Model for triple extraction (deepseek-r1:1.5b)

- **Groq Cloud API (for Q&A/reasoning):**
  - `GROQ_API_KEY`: Your Groq API key (get from https://console.groq.com/keys)
  - `GROQ_MODEL`: Model for reasoning (llama-3.3-70b-versatile)
  - `GROQ_TIMEOUT`: Request timeout in seconds
  - `GROQ_MAX_TOKENS`: Maximum tokens per response
  - `GROQ_TEMPERATURE`: Sampling temperature (0.0-1.0)

- **FAISS:**
  - `FAISS_INDEX_TYPE`: Index type (IndexFlatIP, IndexIVFFlat, IndexHNSWFlat)
  - `STORAGE_FAISS_INDEX_PATH`: Path to save FAISS index

- **Agents:**
  - `AGENT_REVERIFY_INTERVAL_SECONDS`: How often to reverify triples
  - `AGENT_CONFLICT_RESOLUTION_INTERVAL_SECONDS`: Conflict check interval
  - `AGENT_SCHEMA_SUGGESTION_INTERVAL_SECONDS`: Schema suggestion interval

### Ontology Rules

Edit `shared/config/settings.py` to customize validation ontology:

```python
VALIDATION_ONTOLOGY_RULES = [
    {
        "predicate": "founded_by",
        "subject_type": "Organization",
        "object_type": "Person",
    },
    # Add your domain-specific rules here
]
```

## Testing the System

### 1. Ingest a Document

```bash
# From URL
curl -X POST http://localhost:8000/api/v1/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "source": "https://en.wikipedia.org/wiki/Artificial_intelligence",
    "source_type": "HTML",
    "metadata": {"topic": "AI"}
  }'
```

Response:
```json
{
  "document_id": "abc123...",
  "status": "processing",
  "message": "Document ingestion started..."
}
```

### 2. Check Document Status

```bash
curl http://localhost:8000/api/v1/documents/abc123...
```

Response:
```json
{
  "document_id": "abc123...",
  "status": {
    "ingested": true,
    "normalized": true,
    "extracted": true,
    "validated": true
  },
  "stats": {
    "candidate_triples": 156,
    "validated_triples": 142
  }
}
```

### 3. Query the Knowledge Graph

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

Response:
```json
{
  "question": "What is artificial intelligence?",
  "answer": "Artificial intelligence (AI) is the simulation of human intelligence processes by machines, especially computer systems...",
  "verification_status": "SUPPORTED",
  "confidence": 0.92,
  "evidence_edges": [
    {"edge_id": "e1", "relation": "defined_as", ...}
  ],
  "sources": [
    {"document_id": "abc123...", "section": "Introduction"}
  ]
}
```

### 4. Upload a File

```bash
curl -X POST http://localhost:8000/api/v1/ingest/file \
  -F "file=@/path/to/document.pdf" \
  -F "source_type=PDF"
```

### 5. Get System Statistics

```bash
curl http://localhost:8000/api/v1/stats
```

Response:
```json
{
  "documents": {
    "total": 42,
    "by_type": {
      "PDF": 20,
      "HTML": 15,
      "CSV": 7
    }
  },
  "triples": {
    "candidate": 5420,
    "validated": 4891
  },
  "graph": {
    "entities": 1234,
    "relationships": 3456
  },
  "embeddings": {
    "total_chunks": 8765,
    "index_size": "34.2 MB"
  }
}
```

## Monitoring

### Flower (Celery Task Monitor)

Access at http://localhost:5555

- View active/scheduled/completed tasks
- Monitor worker status
- Retry failed tasks
- View task details and tracebacks

### Neo4j Browser

Access at http://localhost:7474

```cypher
// View all entities
MATCH (n:Entity) RETURN n LIMIT 25

// View relationships
MATCH (n:Entity)-[r]->(m:Entity) 
RETURN n, r, m 
LIMIT 50

// Find most connected entities
MATCH (n:Entity)-[r]->()
RETURN n.canonical_name, count(r) as degree
ORDER BY degree DESC
LIMIT 10

// View recent changes
MATCH (n:Entity)-[r]->(m:Entity)
WHERE r.updated_at > datetime() - duration('P1D')
RETURN n, r, m
```

### Prometheus Metrics

Access at http://localhost:8000/metrics

Available metrics:
- `graphbuilder_requests_total`: Total API requests
- `graphbuilder_request_duration_seconds`: Request latency
- `graphbuilder_documents_ingested_total`: Documents processed
- `graphbuilder_queries_processed_total`: Queries answered

### Logs

View logs for each service:

```bash
# API logs
docker-compose logs -f api

# Worker logs
docker-compose logs -f worker

# Agent logs
docker-compose logs -f agents

# All logs
docker-compose logs -f
```

## Troubleshooting

### Services not starting

**Check service status (macOS):**
```bash
brew services list
```

**Restart a service:**
```bash
brew services restart mongodb-community
brew services restart neo4j
brew services restart redis
```

**Check logs:**
```bash
# MongoDB
tail -f /opt/homebrew/var/log/mongodb/mongo.log

# Neo4j
tail -f /opt/homebrew/var/log/neo4j/neo4j.log
```

### Ollama models not downloading

```bash
# Check Ollama service
curl http://localhost:11434/api/tags

# Manually pull models
ollama pull deepseek-r1:1.5b
ollama pull deepseek-r1:7b

# List installed models
ollama list
```

### MongoDB connection errors

```bash
# Check MongoDB is running
brew services list | grep mongodb

# Test connection
mongosh

# Check configuration
cat /opt/homebrew/etc/mongod.conf
```

### Neo4j authentication errors

Default credentials: `neo4j` / `password`

Change password:
```bash
# First time setup - access Neo4j Browser
open http://localhost:7474

# Or via command line
neo4j-admin set-initial-password your-password
```

Update `.env`:
```
NEO4J_PASSWORD=your-password
```

### Worker tasks not processing

```bash
# Check Redis connection
redis-cli ping

# Check worker logs (if running in terminal)
# Worker will show task processing in real-time

# Clear task queue
redis-cli FLUSHDB
```

### FAISS index errors

```bash
# Delete and rebuild index
rm -rf data/faiss/*

# Restart worker to rebuild
```

### Port already in use

```bash
# Find process using port
lsof -i :8000  # or :11434, :27017, etc.

# Kill process
kill -9 <PID>
```

### Out of memory errors

Reduce concurrency in `.env`:
```
API_WORKERS=2
```

Reduce Celery workers:
```bash
celery -A workers.tasks worker --loglevel=info --concurrency=2
```

## Production Deployment

### Security Hardening

1. **Change default passwords:**
   ```
   MONGODB_PASSWORD=<strong-password>
   NEO4J_PASSWORD=<strong-password>
   ```

2. **Enable authentication:**
   - Add API key middleware to api/main.py
   - Configure CORS properly in production

3. **Configure CORS:**
   ```python
   # api/main.py
   app.add_middleware(
       CORSMiddleware,
       allow_origins=["https://yourdomain.com"],  # Specific domains only
       allow_credentials=True,
       allow_methods=["GET", "POST"],
       allow_headers=["*"],
   )
   ```

4. **Enable HTTPS:**
   Use a reverse proxy (nginx, Caddy) with SSL/TLS certificates.

### Scaling

1. **Run multiple workers:**
   ```bash
   # Terminal 1
   celery -A workers.tasks worker --loglevel=info --concurrency=4 -n worker1@%h
   
   # Terminal 2
   celery -A workers.tasks worker --loglevel=info --concurrency=4 -n worker2@%h
   ```

2. **Run multiple API instances:**
   Use a process manager like systemd or supervisord to run multiple API processes, then load balance with nginx.

3. **Database replication:**
   - MongoDB: Enable replica sets
   - Neo4j: Configure read replicas

### Process Management with Systemd

Create service files in `/etc/systemd/system/`:

**graphbuilder-api.service:**
```ini
[Unit]
Description=GraphBuilder RAG API
After=network.target mongodb.service neo4j.service redis.service

[Service]
Type=simple
User=youruser
WorkingDirectory=/path/to/graphbuilder-rag
Environment="PATH=/path/to/graphbuilder-rag/venv/bin"
ExecStart=/path/to/graphbuilder-rag/venv/bin/python -m api.main
Restart=always

[Install]
WantedBy=multi-user.target
```

**graphbuilder-worker.service:**
```ini
[Unit]
Description=GraphBuilder RAG Celery Worker
After=network.target redis.service

[Service]
Type=simple
User=youruser
WorkingDirectory=/path/to/graphbuilder-rag
Environment="PATH=/path/to/graphbuilder-rag/venv/bin"
ExecStart=/path/to/graphbuilder-rag/venv/bin/celery -A workers.tasks worker --loglevel=info --concurrency=4
Restart=always

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
sudo systemctl enable graphbuilder-api graphbuilder-worker
sudo systemctl start graphbuilder-api graphbuilder-worker
```

### Backup and Recovery

1. **MongoDB backup:**
   ```bash
   mongodump --out /path/to/backup
   ```

   **Restore:**
   ```bash
   mongorestore /path/to/backup
   ```

2. **Neo4j backup:**
   ```bash
   neo4j-admin database dump neo4j --to=/path/to/backup/neo4j.dump
   ```

   **Restore:**
   ```bash
   neo4j-admin database load neo4j --from=/path/to/backup/neo4j.dump
   ```

3. **FAISS index backup:**
   ```bash
   tar -czf faiss-backup.tar.gz data/faiss/
   ```

   **Restore:**
   ```bash
   tar -xzf faiss-backup.tar.gz -C data/
   ```

## API Documentation

Interactive API documentation available at:
- **Swagger UI:** http://localhost:8000/docs
- **ReDoc:** http://localhost:8000/redoc

## Architecture

```
┌─────────────┐
│   Client    │
└──────┬──────┘
       │
┌──────▼──────────────────────────────────┐
│          FastAPI (api/main.py)         │
│  /ingest, /query, /health, /metrics    │
└──────┬──────────────────────────────────┘
       │
       ├─── MongoDB (raw docs, triples)
       ├─── Neo4j (knowledge graph)
       ├─── FAISS (embeddings)
       │
       ├─── Celery Workers (pipeline tasks)
       │    ├─ normalize_document
       │    ├─ extract_triples
       │    ├─ validate_triples
       │    ├─ fuse_triples
       │    └─ embed_document
       │
       ├─── Celery Beat (periodic tasks)
       │    ├─ rebuild_faiss_index
       │    └─ cleanup_old_audits
       │
       └─── Agents (autonomous maintenance)
            ├─ ReverifyAgent
            ├─ ConflictResolverAgent
            └─ SchemaSuggestorAgent
```

## Contributing

See CONTRIBUTING.md for development guidelines.

## License

See LICENSE file.
