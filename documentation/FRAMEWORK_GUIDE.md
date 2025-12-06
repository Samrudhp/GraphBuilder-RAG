# GraphBuilder-RAG Framework Guide

## 🎯 What is GraphBuilder-RAG?

GraphBuilder-RAG is a **production-ready framework** for building Graph-Enhanced Retrieval-Augmented Generation (RAG) systems. It combines:

- **Knowledge Graph Construction**: Automatically extracts entities, relationships, and facts from documents
- **Semantic Search**: Embeds and indexes text chunks for similarity-based retrieval
- **Hybrid Retrieval**: Combines graph traversal + vector search for comprehensive context
- **Fact Verification**: Validates LLM responses against the knowledge graph to reduce hallucinations
- **Intelligent Agents**: Automatically maintains graph quality, resolves conflicts, and suggests schema improvements

## 🏗️ Framework Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     API Layer (FastAPI)                      │
│  /ingest  /query  /graph  /triples  /validate  /agents      │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                    Task Queue (Celery)                       │
│  Async Pipeline: Ingest → Normalize → Extract → Embed       │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌──────────────┬──────────────┬──────────────┬───────────────┐
│  Ingestion   │ Normalization│  Extraction  │   Embedding   │
│   Service    │   Service    │   Service    │    Service    │
│              │              │              │               │
│ PDF/HTML/TXT │ Text/Tables  │ LLM Triples  │ BGE Vectors   │
│ to MongoDB   │ Chunking     │ via Ollama   │ to FAISS      │
└──────────────┴──────────────┴──────────────┴───────────────┘
                              ↓
┌──────────────┬──────────────┬──────────────┬───────────────┐
│  Validation  │    Fusion    │   Entity     │     Query     │
│   Service    │   Service    │  Resolution  │    Service    │
│              │              │              │               │
│ Fact Check   │ Merge into   │ Deduplicate  │ Hybrid Search │
│ Against KG   │ Neo4j Graph  │ Entities     │ + Graph Walk  │
└──────────────┴──────────────┴──────────────┴───────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                    Storage Layer                             │
│  MongoDB (docs)  Neo4j (graph)  FAISS (vectors)  Redis      │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                  Intelligent Agents                          │
│  ReverifyAgent  ConflictResolverAgent  SchemaSuggestorAgent │
└─────────────────────────────────────────────────────────────┘
```

## 📊 Core Components

### 1. **Ingestion Service** (`services/ingestion_service.py`)
- **Purpose**: Load documents from various sources
- **Inputs**: URLs, file paths, raw text
- **Outputs**: Stored in MongoDB with metadata
- **Supported Formats**: PDF, HTML, TXT, Markdown

### 2. **Normalization Service** (`services/normalization_service.py`)
- **Purpose**: Break documents into processable chunks
- **Inputs**: Raw document text
- **Outputs**: Text sections, tables, metadata
- **Features**: Intelligent chunking, table extraction, context preservation

### 3. **Extraction Service** (`services/extraction_service.py`)
- **Purpose**: Extract structured knowledge (subject-predicate-object triples)
- **Inputs**: Normalized text chunks
- **Outputs**: JSON triples with confidence scores
- **LLM Used**: DeepSeek-R1 1.5B via Ollama (local)

### 4. **Embedding Service** (`services/embedding_service.py`)
- **Purpose**: Create vector representations for semantic search
- **Inputs**: Text chunks
- **Outputs**: 384-dim vectors indexed in FAISS
- **Model Used**: BGE-small (BAAI/bge-small-en-v1.5)

### 5. **Validation Service** (`services/validation_service.py`)
- **Purpose**: Verify facts against existing knowledge graph
- **Inputs**: Extracted triples
- **Outputs**: Validated/flagged triples with conflict detection
- **Features**: Consistency checking, temporal validation, ontology validation

### 6. **Fusion Service** (`services/fusion_service.py`)
- **Purpose**: Merge validated triples into Neo4j knowledge graph
- **Inputs**: Validated triples
- **Outputs**: Updated graph with versioning
- **Features**: Conflict resolution, provenance tracking, versioning

### 7. **Entity Resolution Service** (`services/entity_resolution_service.py`)
- **Purpose**: Deduplicate and normalize entity names
- **Inputs**: Graph nodes
- **Outputs**: Merged entities with aliases
- **Features**: String similarity, semantic similarity, rule-based matching

### 8. **Query Service** (`services/query_service.py`)
- **Purpose**: Answer questions using hybrid retrieval
- **Inputs**: User questions
- **Outputs**: Answers with citations and verification status
- **LLM Used**: Llama-3.3-70B-Versatile via Groq Cloud API (fast inference)
- **Features**: Semantic search + graph traversal, fact verification, confidence scoring

## 🔧 Customization Guide

### For Domain-Specific Applications

#### 1. **Define Your Ontology**

Edit `shared/config/settings.py`:

```python
class Settings(BaseSettings):
    # ... existing settings ...
    
    # CUSTOMIZE: Define your domain entities
    EXTRACTION_ENTITY_TYPES: List[str] = [
        "Person", "Organization", "Location",  # Generic
        "Disease", "Symptom", "Treatment",     # Medical domain
        "Product", "Feature", "Price",         # E-commerce domain
        "Law", "Case", "Statute"               # Legal domain
    ]
    
    # CUSTOMIZE: Define your domain relationships
    EXTRACTION_RELATIONSHIP_TYPES: List[str] = [
        "treats", "causes", "diagnoses",       # Medical
        "manufactures", "competes_with",       # Business
        "regulates", "violates", "cites"       # Legal
    ]
```

#### 2. **Customize Extraction Prompts**

Edit `shared/prompts/extraction_prompts.py`:

```python
SYSTEM_PROMPT = """You are an expert knowledge graph builder specializing in {DOMAIN}.

Extract structured triples (subject, predicate, object) from text.

Focus on:
- {DOMAIN_SPECIFIC_ENTITIES}
- {DOMAIN_SPECIFIC_RELATIONSHIPS}
- {DOMAIN_SPECIFIC_RULES}

Example for Medical Domain:
Input: "Aspirin treats headaches but may cause stomach ulcers."
Output:
[
  {"subject": "Aspirin", "predicate": "treats", "object": "headaches", "confidence": 0.95},
  {"subject": "Aspirin", "predicate": "causes", "object": "stomach ulcers", "confidence": 0.85}
]
"""

# Add domain-specific examples
EXTRACTION_EXAMPLES = [
    {
        "text": "Metformin is prescribed for Type 2 Diabetes.",
        "triples": [
            {"subject": "Metformin", "predicate": "prescribed_for", "object": "Type 2 Diabetes", "confidence": 0.95}
        ]
    },
    # Add 5-10 domain examples
]
```

#### 3. **Customize Validation Rules**

Edit `services/validation_service.py`:

```python
async def validate_triple(self, triple: dict, context: dict) -> dict:
    """Validate a triple against domain rules."""
    
    # CUSTOMIZE: Add domain-specific validation
    
    # Example: Medical domain - drug interactions
    if triple["predicate"] == "interacts_with":
        contraindications = await self._check_contraindications(
            triple["subject"], 
            triple["object"]
        )
        if contraindications:
            return {
                **triple,
                "is_valid": False,
                "conflict": f"Known contraindication: {contraindications}"
            }
    
    # Example: Legal domain - temporal consistency
    if triple["predicate"] == "supersedes":
        if not self._check_temporal_order(triple["subject"], triple["object"]):
            return {
                **triple,
                "is_valid": False,
                "conflict": "Temporal violation: newer law cannot supersede older"
            }
    
    # Example: E-commerce - price validation
    if triple["predicate"] == "costs":
        if not self._is_valid_price(triple["object"]):
            return {
                **triple,
                "is_valid": False,
                "conflict": "Invalid price format"
            }
    
    return {**triple, "is_valid": True}
```

#### 4. **Customize Entity Resolution**

Edit `services/entity_resolution_service.py`:

```python
async def resolve_entities(self, entity_type: str = None):
    """Resolve entities with domain-specific rules."""
    
    # CUSTOMIZE: Add domain-specific normalization
    
    # Example: Medical domain - drug name normalization
    if entity_type == "Drug":
        normalized = await self._normalize_drug_names(entities)
        # Map "Acetaminophen" → "Paracetamol" (generic names)
    
    # Example: Legal domain - case citation normalization
    if entity_type == "Case":
        normalized = await self._normalize_case_citations(entities)
        # Map "Brown v. Board" → "Brown v. Board of Education, 347 U.S. 483"
    
    # Example: Business domain - company name normalization
    if entity_type == "Company":
        normalized = await self._normalize_company_names(entities)
        # Map "Apple Inc." → "Apple Inc." (canonical form)
```

#### 5. **Customize Query Logic**

Edit `services/query_service.py`:

```python
async def query(self, question: str, top_k: int = 5, graph_depth: int = 2):
    """Answer questions with domain-specific logic."""
    
    # CUSTOMIZE: Add domain-specific query patterns
    
    # Example: Medical domain - symptom checker
    if self._is_symptom_query(question):
        symptoms = self._extract_symptoms(question)
        candidates = await self._find_diseases_by_symptoms(symptoms)
        return self._generate_differential_diagnosis(candidates)
    
    # Example: Legal domain - case law search
    if self._is_legal_precedent_query(question):
        facts = self._extract_legal_facts(question)
        cases = await self._find_similar_cases(facts)
        return self._generate_legal_analysis(cases)
    
    # Example: E-commerce - product recommendation
    if self._is_product_query(question):
        requirements = self._extract_product_requirements(question)
        products = await self._find_matching_products(requirements)
        return self._generate_comparison(products)
```

### For Testing and Development

#### 1. **Use Test Data**

Create test fixtures in `tests/fixtures/`:

```python
# tests/fixtures/medical_documents.py
TEST_MEDICAL_DOC = """
Aspirin (acetylsalicylic acid) is a medication used to reduce pain, fever, 
or inflammation. It works by inhibiting cyclooxygenase enzymes. Common side 
effects include stomach ulcers and bleeding. It should not be given to 
children with fever due to risk of Reye's syndrome.
"""

# tests/fixtures/expected_triples.py
EXPECTED_MEDICAL_TRIPLES = [
    {"subject": "Aspirin", "predicate": "treats", "object": "pain"},
    {"subject": "Aspirin", "predicate": "treats", "object": "fever"},
    {"subject": "Aspirin", "predicate": "causes", "object": "stomach ulcers"},
    # ... more expected triples
]
```

#### 2. **Mock External Services**

Edit `tests/conftest.py`:

```python
import pytest
from unittest.mock import AsyncMock, MagicMock

@pytest.fixture
def mock_ollama():
    """Mock Ollama for testing without actual LLM calls."""
    mock = AsyncMock()
    mock.generate.return_value = {
        "response": json.dumps([
            {"subject": "Test Entity", "predicate": "test_rel", "object": "Test Object", "confidence": 0.9}
        ])
    }
    return mock

@pytest.fixture
def mock_embedding_model():
    """Mock embedding model for testing."""
    mock = MagicMock()
    mock.encode.return_value = np.random.rand(384)  # BGE-small dimensions
    return mock

@pytest.fixture
async def test_db():
    """Use in-memory databases for testing."""
    # Use MongoDB in-memory or Docker container
    # Use Neo4j test database
    # Use Redis mock
    pass
```

#### 3. **Unit Tests**

Create `tests/test_services.py`:

```python
import pytest
from services.extraction_service import ExtractionService

@pytest.mark.asyncio
async def test_extraction_service(mock_ollama):
    """Test extraction service with mocked LLM."""
    service = ExtractionService(llm_client=mock_ollama)
    
    text = "Aspirin treats headaches."
    triples = await service.extract(text)
    
    assert len(triples) > 0
    assert triples[0]["subject"] == "Aspirin"
    assert triples[0]["predicate"] == "treats"
    assert triples[0]["object"] == "headaches"

@pytest.mark.asyncio
async def test_validation_service(test_db):
    """Test validation against test knowledge graph."""
    service = ValidationService(neo4j_conn=test_db)
    
    # Insert known fact
    await test_db.add_triple("Aspirin", "treats", "headaches")
    
    # Test consistent triple
    result = await service.validate_triple({
        "subject": "Aspirin", "predicate": "treats", "object": "headaches"
    })
    assert result["is_valid"] is True
    
    # Test conflicting triple
    result = await service.validate_triple({
        "subject": "Aspirin", "predicate": "causes", "object": "headaches"
    })
    assert result["is_valid"] is False
    assert "conflict" in result
```

#### 4. **Integration Tests**

Create `tests/test_pipeline.py`:

```python
@pytest.mark.asyncio
async def test_end_to_end_pipeline(test_db):
    """Test complete pipeline from ingestion to query."""
    
    # 1. Ingest document
    doc_id = await ingest_document(TEST_MEDICAL_DOC, source_type="TEXT")
    
    # 2. Wait for processing
    await wait_for_task_completion(doc_id)
    
    # 3. Verify graph created
    graph = await neo4j_conn.query("MATCH (n) RETURN count(n)")
    assert graph[0]["count(n)"] > 0
    
    # 4. Query
    result = await query_service.query("What does Aspirin treat?")
    
    # 5. Verify answer
    assert "pain" in result["answer"].lower() or "fever" in result["answer"].lower()
    assert result["verification"]["status"] == "verified"
```

#### 5. **Load Testing**

Create `tests/load_test.py`:

```python
import asyncio
from locust import HttpUser, task, between

class RAGUser(HttpUser):
    wait_time = between(1, 3)
    
    @task(3)
    def query_endpoint(self):
        """Test query endpoint under load."""
        self.client.post("/api/v1/query", json={
            "question": "What is artificial intelligence?",
            "top_k_semantic": 5,
            "graph_depth": 2
        })
    
    @task(1)
    def ingest_endpoint(self):
        """Test ingestion endpoint under load."""
        self.client.post("/api/v1/ingest", json={
            "source": "test document text",
            "source_type": "TEXT"
        })

# Run: locust -f tests/load_test.py --host=http://localhost:8000
```

## 🎨 Configuration Reference

### Environment Variables (`.env`)

```bash
# Core Settings
API_HOST=0.0.0.0
API_PORT=8000
LOG_LEVEL=INFO

# MongoDB (Document Storage)
MONGODB_URI=mongodb://localhost:27017
MONGODB_DB=graphbuilder

# Neo4j (Knowledge Graph)
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password_here

# Redis (Task Queue)
REDIS_URI=redis://localhost:6379/0

# Ollama (LLM)
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL_EXTRACT=deepseek-r1:1.5b
OLLAMA_MODEL_QUERY=deepseek-r1:7b

# Embedding Model
EMBEDDING_MODEL_NAME=BAAI/bge-small-en-v1.5
EMBEDDING_DIMENSION=384

# FAISS (Vector Search)
STORAGE_FAISS_INDEX_PATH=./data/faiss/index.faiss
STORAGE_CHUNK_MAP_PATH=./data/faiss/chunk_map.json

# Pipeline Settings
EXTRACTION_ENTITY_TYPES=["Person","Organization","Location","Event"]
EXTRACTION_RELATIONSHIP_TYPES=["works_at","located_in","participates_in"]
NORMALIZATION_CHUNK_SIZE=500
NORMALIZATION_CHUNK_OVERLAP=50
QUERY_TOP_K_SEMANTIC=5
QUERY_GRAPH_DEPTH=2
QUERY_MIN_CONFIDENCE=0.7

# Celery Settings
CELERY_BROKER_URL=redis://localhost:6379/0
CELERY_RESULT_BACKEND=redis://localhost:6379/0
CELERY_WORKER_CONCURRENCY=4

# Agent Settings
AGENT_REVERIFY_SCHEDULE="0 2 * * *"  # Daily at 2 AM
AGENT_CONFLICT_THRESHOLD=0.5
```

## 🚀 Deployment Scenarios

### Development

```bash
# Local services via Homebrew
brew services start mongodb-community neo4j redis
ollama serve &

# Run with tmux
./run.sh
```

### Production (Single Server)

```bash
# Use systemd services
sudo systemctl enable mongodb neo4j redis-server ollama

# Run with supervisor or systemd
# See SETUP.md for systemd service files
```

### Production (Distributed)

```bash
# Separate servers for:
# - API (multiple instances behind load balancer)
# - Workers (autoscaling Celery workers)
# - MongoDB (replica set)
# - Neo4j (cluster)
# - Redis (sentinel/cluster)
# - Ollama (GPU server)

# Use Kubernetes/Docker Swarm for orchestration
```

## 📈 Monitoring and Observability

### Metrics Endpoint

```bash
curl http://localhost:8000/metrics
```

Provides:
- Request latency histograms
- Error rates
- Active task counts
- Database connection pool stats

### Celery Monitoring (Flower)

```bash
# Included in run.sh, or manually:
celery -A workers.tasks flower --port=5555

# Open: http://localhost:5555
```

### Logs

```bash
# View API logs
tail -f logs/api.log

# View worker logs
tail -f logs/celery_worker.log

# View agent logs
tail -f logs/agents.log
```

## 🔍 Common Use Cases

### 1. Medical Knowledge System

**Changes Needed**:
- Entity types: Disease, Symptom, Treatment, Drug, Gene
- Relationships: treats, causes, contraindicates, interacts_with
- Validation: Drug interaction checking, temporal disease progression
- Query: Symptom-based diagnosis, treatment recommendations

### 2. Legal Research Platform

**Changes Needed**:
- Entity types: Case, Statute, Court, Judge, Party
- Relationships: cites, overrules, distinguishes, applies
- Validation: Citation format, temporal consistency, jurisdiction
- Query: Precedent search, statute interpretation, case similarity

### 3. E-commerce Product Catalog

**Changes Needed**:
- Entity types: Product, Brand, Category, Feature, Review
- Relationships: manufactured_by, belongs_to, has_feature, competes_with
- Validation: Price ranges, specification formats, availability
- Query: Product comparison, recommendation, feature search

### 4. Scientific Literature Analysis

**Changes Needed**:
- Entity types: Paper, Author, Institution, Concept, Method
- Relationships: authored_by, cites, uses_method, affiliated_with
- Validation: Citation format, author disambiguation, concept consistency
- Query: Literature review, methodology search, author networks

## 🛠️ Extension Points

### Adding New Document Types

Edit `services/ingestion_service.py`:

```python
async def ingest(self, source: str, source_type: str):
    if source_type == "PDF":
        return await self._ingest_pdf(source)
    elif source_type == "HTML":
        return await self._ingest_html(source)
    # ADD NEW TYPE HERE
    elif source_type == "DOCX":
        return await self._ingest_docx(source)
    elif source_type == "CSV":
        return await self._ingest_csv(source)
```

### Adding New API Endpoints

Edit `api/main.py`:

```python
@router.post("/api/v1/custom-endpoint")
async def custom_endpoint(request: CustomRequest):
    """Add your custom endpoint logic."""
    result = await custom_service.process(request)
    return {"result": result}
```

### Adding New Agents

Create `agents/custom_agent.py`:

```python
class CustomAgent:
    """Your custom agent logic."""
    
    async def run_task(self):
        # Implement agent behavior
        pass

# Add to agents/agents.py:
custom_agent = CustomAgent()
```

## 📚 Additional Resources

- **QUICKSTART.md**: Quick reference for common commands
- **SETUP.md**: Detailed installation and configuration
- **TESTING.md**: Testing workflows and examples
- **ARCHITECTURE.md**: Deep dive into system design
- **INSTALL_CHECKLIST.md**: Step-by-step installation guide

## 🎯 Summary

GraphBuilder-RAG is a **framework**, not a fixed application. It provides:

✅ **Modular services** you can customize for your domain  
✅ **Configurable prompts** to guide LLM extraction  
✅ **Extensible validation** to enforce domain rules  
✅ **Flexible ontology** to define your knowledge structure  
✅ **Production-ready infrastructure** for deployment  

**To adapt it for your domain**: Update ontology, customize prompts, add validation rules, and extend query logic. The framework handles all the infrastructure complexity—you focus on your domain expertise.
