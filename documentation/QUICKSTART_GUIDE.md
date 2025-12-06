# GraphBuilder-RAG Quick Start Guide

This guide walks you through running the GraphBuilder-RAG system from scratch.

---

## Prerequisites

1. **Services Running:**
   - MongoDB (localhost:27017)
   - Neo4j (localhost:7687)
   - Redis (localhost:6379)

2. **Environment Setup:**
   - Python 3.10+ with virtual environment activated
   - Groq API key in `.env` file
   - All dependencies installed: `pip install -r requirements.txt`

---

## Step 1: Start the Core Services

### Terminal 1: Start FastAPI Server
```bash
python api/main.py
```
- API will be available at: `http://localhost:8000`
- Swagger docs at: `http://localhost:8000/docs`

### Terminal 2: Start Celery Worker
```bash
celery -A workers.tasks worker --loglevel=info --pool=solo
```
- Processes async tasks (ingestion, extraction, embedding, etc.)

### Terminal 3: Start Celery Beat (Optional - for scheduled tasks)
```bash
celery -A workers.tasks beat --loglevel=info
```
- Handles periodic/scheduled tasks

---

## Step 2: Ingest and Process Documents

### Option A: Upload via API
```bash
# Upload a text document
python helpers/upload_test.py
```

### Option B: Manual Ingestion
```python
# In Python or helpers script
import requests

# Upload document
with open("your_document.txt", "r") as f:
    content = f.read()

response = requests.post(
    "http://localhost:8000/api/v1/ingest",
    json={"content": content, "metadata": {"title": "My Document"}}
)
print(response.json())
```

**What happens:**
1. Document is saved to MongoDB (`raw_docs`)
2. Celery extracts entities & relationships using LLM
3. Entities are normalized and deduplicated
4. Relationships are validated and inserted into Neo4j
5. Text is chunked and embedded into FAISS

---

## Step 3: Verify Data Pipeline

### Check MongoDB Collections
```bash
python helpers/check_neo4j.py  # Also checks MongoDB
```

### View Neo4j Graph Data
```bash
python helpers/view_triples.py
```
**Output:** All entities and relationships in the knowledge graph

### Check FAISS Embeddings
```bash
# Check if embeddings exist
ls data/faiss_index/
# Should see: index.faiss, chunk_map.pkl
```

### Trigger Manual Embedding (if needed)
```bash
python helpers/trigger_embedding.py
```

---

## Step 4: Query the System

### Via Test Script
```bash
python helpers/test_query.py
```

### Via API
```python
import requests

response = requests.post(
    "http://localhost:8000/api/v1/query",
    json={
        "question": "What did Albert Einstein discover?",
        "max_chunks": 5,
        "require_verification": True
    }
)

result = response.json()
print(result["answer"])
print(result["sources"])
```

**System performs:**
1. FAISS semantic search for relevant text chunks
2. Neo4j graph traversal for entity relationships
3. Hybrid retrieval combining both sources
4. LLM generates answer with citations
5. GraphVerify checks for hallucinations

---

## Step 5: Maintenance & Utilities

### Clear All Data (Reset System)
```bash
python helpers/clear_all.py
```
**WARNING:** Deletes all MongoDB data, Neo4j graph, and FAISS index

### Clear Specific Components

**Clear MongoDB only:**
```bash
python helpers/clear_db.py
```

**Clear Neo4j only:**
```bash
python helpers/clear_neo4j.py
```

**Clear FAISS only:**
```bash
python helpers/clear_faiss.py
```

---

## Step 6: Testing & Validation

### Test Database Connections
```bash
python tests/test_connections.py
```

### Test Individual Services
```bash
python tests/test_each_service.py
```

### Test Full Pipeline
```bash
python tests/test_end_to_end_pipeline.py
```

### Test Query & Hallucination Detection
```bash
python tests/test_query_and_hallucination.py
```

---

## Common Workflows

### Workflow 1: Fresh Start with Sample Data
```bash
# 1. Clear everything
python helpers/clear_all.py

# 2. Start services (3 terminals)
python api/main.py
celery -A workers.tasks worker --loglevel=info --pool=solo
celery -A workers.tasks beat --loglevel=info

# 3. Ingest sample data
python helpers/upload_test.py

# 4. Wait for processing (check Celery logs)

# 5. Verify data
python helpers/view_triples.py

# 6. Query
python helpers/test_query.py
```

### Workflow 2: Add New Document
```bash
# 1. Upload document
curl -X POST http://localhost:8000/api/v1/ingest \
  -H "Content-Type: application/json" \
  -d '{"content": "Your text here", "metadata": {"source": "paper.pdf"}}'

# 2. Check ingestion status in Celery logs

# 3. Trigger embedding if needed
python helpers/trigger_embedding.py

# 4. Query the new data
python helpers/test_query.py
```

### Workflow 3: Debugging Failed Pipeline
```bash
# 1. Check connections
python tests/test_connections.py

# 2. Check MongoDB collections
python helpers/check_neo4j.py

# 3. View graph data
python helpers/view_triples.py

# 4. Check Celery worker logs for errors

# 5. Manually trigger embedding
python helpers/trigger_embedding.py
```

---

## Helper Scripts Reference

| Script | Purpose |
|--------|---------|
| `upload_test.py` | Upload sample document via API |
| `test_query.py` | Test query with sample questions |
| `view_triples.py` | Display all entities & relationships in Neo4j |
| `check_neo4j.py` | Check Neo4j and MongoDB connections/data |
| `trigger_embedding.py` | Manually trigger FAISS embedding for documents |
| `clear_all.py` | **DANGER:** Delete all data (MongoDB + Neo4j + FAISS) |
| `clear_db.py` | Clear MongoDB only |
| `clear_neo4j.py` | Clear Neo4j graph only |
| `clear_faiss.py` | Clear FAISS index only |

---

## Troubleshooting

### "No graph context available"
- Check if entities exist: `python helpers/view_triples.py`
- Check entity extraction in Celery logs
- Verify Neo4j is running and accessible

### "No text chunks available"
- Check FAISS index exists: `ls data/faiss_index/`
- Trigger embedding: `python helpers/trigger_embedding.py`
- Check MongoDB `embeddings_meta` collection

### Celery tasks stuck
- Restart Celery worker
- Check Redis is running
- Look for errors in Celery logs

### API returning errors
- Check all services are running (MongoDB, Neo4j, Redis)
- Verify Groq API key in `.env`
- Check FastAPI logs for detailed errors

---

## Expected Data Flow

```
1. Document Upload (API)
   ↓
2. Raw Storage (MongoDB: raw_docs)
   ↓
3. Entity Extraction (Celery → LLM)
   ↓
4. Triple Extraction (Celery → LLM)
   ↓
5. Normalization (Celery → Entity Resolution)
   ↓
6. Graph Storage (Neo4j: Entity nodes + RELATED edges)
   ↓
7. Text Embedding (Celery → BGE-small → FAISS)
   ↓
8. Query Ready (Hybrid Retrieval: FAISS + Neo4j)
```

---

## Next Steps

- See `ARCHITECTURE.md` for system design details
- See `CELERY_AND_AGENTS_EXPLAINED.md` for task processing
- See `TESTING.md` for comprehensive test suite
- See `FRAMEWORK_GUIDE.md` for extending the system

---

**Your GraphBuilder-RAG system is ready! Start with Step 1 and follow the workflow.** 🚀
