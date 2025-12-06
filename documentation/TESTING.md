# Testing and Example Workflows

## Test Data

### Sample Documents

Create a test directory with sample files:

```bash
mkdir -p test_data
```

1. **test_data/ai_facts.txt:**
```
Artificial Intelligence (AI) was founded as an academic discipline in 1956.
John McCarthy coined the term "Artificial Intelligence".
Deep Learning is a subset of Machine Learning.
Geoffrey Hinton is known as the "Godfather of Deep Learning".
AlphaGo was developed by DeepMind.
```

2. **test_data/companies.csv:**
```csv
Company,Founded,Founder,Industry
OpenAI,2015,Sam Altman,AI
DeepMind,2010,Demis Hassabis,AI
Tesla,2003,Elon Musk,Automotive
SpaceX,2002,Elon Musk,Aerospace
```

3. **test_data/research.html:**
```html
<!DOCTYPE html>
<html>
<head><title>AI Research</title></head>
<body>
<h1>Recent AI Breakthroughs</h1>
<p>GPT-4 was released by OpenAI in March 2023.</p>
<p>It demonstrates significant improvements in reasoning capabilities.</p>
</body>
</html>
```

## End-to-End Workflow

### 1. Ingest Multiple Documents

```bash
# Ingest text file
curl -X POST http://localhost:8000/api/v1/ingest/file \
  -F "file=@test_data/ai_facts.txt" \
  -F "source_type=TEXT"

# Ingest CSV
curl -X POST http://localhost:8000/api/v1/ingest/file \
  -F "file=@test_data/companies.csv" \
  -F "source_type=CSV"

# Ingest HTML from URL
curl -X POST http://localhost:8000/api/v1/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "source": "https://en.wikipedia.org/wiki/Machine_learning",
    "source_type": "HTML",
    "metadata": {"topic": "machine_learning"}
  }'
```

### 2. Monitor Processing

Watch the worker logs to see the pipeline in action:

```bash
docker-compose logs -f worker
```

You should see:
1. Normalization task started
2. Extraction task started
3. Validation task started
4. Fusion task started
5. Embedding task started

### 3. Query the Knowledge Graph

Once processing completes, query the system:

```bash
# Question about founders
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Who founded OpenAI?",
    "top_k_semantic": 5,
    "graph_depth": 2,
    "min_confidence": 0.7
  }'

# Question about relationships
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What companies did Elon Musk found?",
    "top_k_semantic": 5,
    "graph_depth": 2
  }'

# Question about concepts
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is deep learning and who are the key researchers?",
    "top_k_semantic": 10,
    "graph_depth": 3
  }'
```

### 4. Verify Knowledge Graph in Neo4j

Open Neo4j Browser at http://localhost:7474 and run:

```cypher
// View all entities
MATCH (n:Entity) 
RETURN n.canonical_name, n.entity_type, n.aliases
LIMIT 25

// View founder relationships
MATCH (person:Entity {entity_type: 'Person'})-[r:founded_by]->(company:Entity)
RETURN person.canonical_name, company.canonical_name, r.confidence

// View the subgraph for a specific entity
MATCH path = (n:Entity {canonical_name: 'OpenAI'})-[*1..2]-(m)
RETURN path
```

### 5. Inspect MongoDB Collections

```bash
# Connect to MongoDB
docker exec -it graphbuilder-mongodb mongosh

# Switch to database
use graphbuilder

# Check raw documents
db.raw_documents.find().limit(5)

# Check validated triples
db.validated_triples.find({
  "validation_result.confidence": {$gte: 0.9}
}).limit(10)

# Check human review queue
db.human_review_queue.find()
```

## Testing Individual Services

### Ingestion Service

```python
import asyncio
from services.ingestion.service import IngestionService

async def test_ingestion():
    service = IngestionService()
    
    # Test URL ingestion
    doc = await service.ingest_from_url(
        url="https://example.com",
        source_type="HTML",
    )
    print(f"Ingested: {doc.document_id}")

asyncio.run(test_ingestion())
```

### Normalization Service

```python
import asyncio
from services.normalization.service import NormalizationService
from shared.models.schemas import DocumentType

async def test_normalization():
    service = NormalizationService()
    
    # Assume we have a raw document ID
    normalized = await service.normalize_document("doc_123")
    print(f"Sections: {len(normalized.sections)}")
    print(f"Tables: {len(normalized.tables)}")

asyncio.run(test_normalization())
```

### Extraction Service

```python
import asyncio
from services.extraction.service import ExtractionService

async def test_extraction():
    service = ExtractionService()
    
    text = "John McCarthy founded the field of AI in 1956."
    triples = await service.extract_from_text(
        text=text,
        document_id="test_doc",
        section="intro",
    )
    
    for triple in triples:
        print(f"{triple.subject} -> {triple.predicate} -> {triple.object}")

asyncio.run(test_extraction())
```

### Query Service

```python
import asyncio
from services.query.service import QueryService
from shared.models.schemas import QueryRequest

async def test_query():
    service = QueryService()
    
    request = QueryRequest(
        question="Who founded OpenAI?",
        top_k_semantic=5,
        graph_depth=2,
    )
    
    response = await service.answer_question(request)
    print(f"Answer: {response.answer}")
    print(f"Verification: {response.verification_status}")
    print(f"Confidence: {response.confidence}")

asyncio.run(test_query())
```

## Testing Agents

### ReverifyAgent

```python
import asyncio
from agents.agents import ReverifyAgent

async def test_reverify():
    agent = ReverifyAgent()
    summary = await agent.run_cycle()
    print(summary)

asyncio.run(test_reverify())
```

### ConflictResolverAgent

First, create a conflict in Neo4j:

```cypher
// Create entities
MERGE (openai:Entity {canonical_name: 'OpenAI', entity_type: 'Organization'})
MERGE (sam:Entity {canonical_name: 'Sam Altman', entity_type: 'Person'})
MERGE (elon:Entity {canonical_name: 'Elon Musk', entity_type: 'Person'})

// Create conflicting relationships
CREATE (openai)-[:founded_by {confidence: 0.8, version: 1}]->(sam)
CREATE (openai)-[:founded_by {confidence: 0.6, version: 1}]->(elon)
```

Then run the agent:

```python
import asyncio
from agents.agents import ConflictResolverAgent

async def test_conflict_resolution():
    agent = ConflictResolverAgent()
    summary = await agent.run_cycle()
    print(summary)

asyncio.run(test_conflict_resolution())
```

### SchemaSuggestorAgent

```python
import asyncio
from agents.agents import SchemaSuggestorAgent

async def test_schema_suggestion():
    agent = SchemaSuggestorAgent()
    summary = await agent.run_cycle()
    print(summary)

asyncio.run(test_schema_suggestion())
```

## Performance Testing

### Load Testing with Locust

Create `locustfile.py`:

```python
from locust import HttpUser, task, between

class GraphBuilderUser(HttpUser):
    wait_time = between(1, 3)
    
    @task(3)
    def query(self):
        self.client.post("/api/v1/query", json={
            "question": "What is AI?",
            "top_k_semantic": 5,
            "graph_depth": 2,
        })
    
    @task(1)
    def stats(self):
        self.client.get("/api/v1/stats")
    
    @task(1)
    def health(self):
        self.client.get("/health")
```

Run load test:

```bash
pip install locust
locust -f locustfile.py --host http://localhost:8000
```

Open http://localhost:8089 and start the test.

### Benchmark Extraction

```python
import asyncio
import time
from services.extraction.service import ExtractionService

async def benchmark_extraction():
    service = ExtractionService()
    
    text = """
    Artificial Intelligence was founded by John McCarthy in 1956.
    Deep Learning was pioneered by Geoffrey Hinton.
    OpenAI was founded by Sam Altman and Elon Musk.
    """ * 10  # Repeat to simulate larger document
    
    start = time.time()
    triples = await service.extract_from_text(text, "bench_doc", "test")
    duration = time.time() - start
    
    print(f"Extracted {len(triples)} triples in {duration:.2f}s")
    print(f"Throughput: {len(triples)/duration:.2f} triples/sec")

asyncio.run(benchmark_extraction())
```

## Integration Tests

Create `tests/test_integration.py`:

```python
import pytest
import asyncio
from services.ingestion.service import IngestionService
from services.normalization.service import NormalizationService
from services.extraction.service import ExtractionService
from services.validation.service import ValidationEngine
from services.fusion.service import FusionService
from services.query.service import QueryService
from shared.models.schemas import QueryRequest, DocumentType

@pytest.mark.asyncio
async def test_full_pipeline():
    """Test complete document processing pipeline."""
    
    # 1. Ingest
    ingestion = IngestionService()
    raw_doc = await ingestion.ingest_from_url(
        url="https://example.com",
        source_type=DocumentType.HTML,
    )
    
    # 2. Normalize
    normalization = NormalizationService()
    normalized = await normalization.normalize_document(raw_doc.document_id)
    
    # 3. Extract
    extraction = ExtractionService()
    candidates = await extraction.extract_triples(normalized.document_id)
    
    # 4. Validate
    validation = ValidationEngine()
    validated = []
    for candidate in candidates[:10]:  # Test with subset
        result = await validation.validate_triple(candidate)
        if result:
            validated.append(result)
    
    # 5. Fuse
    fusion = FusionService()
    for triple in validated:
        await fusion.fuse_triple(triple)
    
    # 6. Query
    query = QueryService()
    response = await query.answer_question(
        QueryRequest(question="Test query", top_k_semantic=5)
    )
    
    assert response.answer is not None
    assert len(response.evidence_edges) > 0

@pytest.mark.asyncio
async def test_graphverify():
    """Test GraphVerify hallucination detection."""
    
    query = QueryService()
    
    # This should be supported
    request = QueryRequest(
        question="What is a fact present in the graph?",
        top_k_semantic=5,
        graph_depth=2,
    )
    
    response = await query.answer_question(request)
    assert response.verification_status in ["SUPPORTED", "UNKNOWN"]
```

Run tests:

```bash
pytest tests/test_integration.py -v
```

## Debugging

### Enable Debug Logging

Update `.env`:

```
DEBUG=true
LOG_LEVEL=DEBUG
```

Restart services:

```bash
docker-compose restart
```

### Inspect Celery Tasks

```python
from workers.tasks import celery_app

# Get task info
task_id = "abc-123-def-456"
result = celery_app.AsyncResult(task_id)
print(result.state)
print(result.info)
```

### Test Ollama Directly

```bash
curl http://localhost:11434/api/generate -d '{
  "model": "deepseek-r1:1.5b",
  "prompt": "Extract entities: John founded OpenAI in 2015.",
  "stream": false
}'
```

### Monitor Resource Usage

```bash
# Container stats
docker stats

# MongoDB ops
docker exec -it graphbuilder-mongodb mongosh --eval "db.serverStatus().connections"

# Neo4j metrics
curl http://localhost:7474/db/system/tx/commit \
  -H "Content-Type: application/json" \
  -d '{"statements":[{"statement":"CALL dbms.listTransactions()"}]}'
```

## Cleanup

### Reset Database

```bash
# Drop all collections in MongoDB
docker exec -it graphbuilder-mongodb mongosh graphbuilder --eval "db.dropDatabase()"

# Clear Neo4j
docker exec -it graphbuilder-neo4j cypher-shell -u neo4j -p password "MATCH (n) DETACH DELETE n"

# Reset FAISS index
rm -rf data/faiss/*

# Restart services
docker-compose restart
```

### Clear Task Queue

```bash
# Purge all Celery tasks
docker exec -it graphbuilder-worker celery -A workers.tasks purge -f
```

### Remove All Data

```bash
# Stop and remove all containers and volumes
docker-compose down -v

# Remove data directory
rm -rf data/
```

## Common Issues and Solutions

### Issue: Tasks stuck in queue

**Solution:** Check worker is running and connected to Redis
```bash
docker-compose logs worker
docker exec -it graphbuilder-redis redis-cli ping
```

### Issue: Low extraction quality

**Solution:** Adjust temperature and prompts
```
EXTRACTION_TEMPERATURE=0.05  # More deterministic
EXTRACTION_MAX_TOKENS=4096   # Allow longer responses
```

### Issue: Memory errors with FAISS

**Solution:** Use IVF index for large datasets
```
FAISS_INDEX_TYPE=IndexIVFFlat
FAISS_NLIST=100
```

### Issue: GraphVerify false positives

**Solution:** Adjust thresholds
```
GRAPHVERIFY_SUPPORT_THRESHOLD=0.9
GRAPHVERIFY_CONTRADICTION_THRESHOLD=0.6
```

## Next Steps

1. **Customize for your domain:**
   - Update ontology rules in `shared/config/settings.py`
   - Add domain-specific prompts in `shared/prompts/templates.py`
   - Configure validation constraints

2. **Add authentication:**
   - Implement API key middleware
   - Add user management
   - Configure RBAC

3. **Scale for production:**
   - Set up database replication
   - Configure load balancing
   - Enable monitoring and alerting
   - Implement backup strategies

4. **Enhance agent capabilities:**
   - Connect to Wikidata/DBpedia APIs
   - Implement entity linking
   - Add feedback loops for model improvement
