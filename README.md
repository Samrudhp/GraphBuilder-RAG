# GraphBuilder-RAG: Graph-Enhanced Retrieval Augmented Generation System

A production-grade, modular framework for building and querying knowledge graphs from heterogeneous documents with advanced RAG capabilities.

## 🎯 System Overview

GraphBuilder-RAG extracts structured knowledge from documents, validates facts, builds versioned knowledge graphs, and provides hybrid retrieval with hallucination detection.

### Key Features

- **Multi-format ingestion**: HTML, PDF, CSV, JSON APIs
- **Intelligent extraction**: Rule-based + LLM-based triple extraction
- **Fact validation**: Ontology rules + external verification
- **Versioned knowledge graph**: Neo4j with full provenance tracking
- **Hybrid retrieval**: FAISS semantic search + Neo4j graph traversal
- **Hallucination detection**: GraphVerify for claim validation
- **Self-healing agents**: Auto-verification, conflict resolution, schema evolution

## 🏗️ Architecture

```
┌─────────────────┐
│   Ingestion     │ → MongoDB GridFS (raw docs)
└────────┬────────┘
         ↓
┌─────────────────┐
│ Normalization   │ → MongoDB (normalized_docs)
└────────┬────────┘
         ↓
┌─────────────────┐
│   Extraction    │ → MongoDB (candidate_triples)
│  DeepSeek 1.5B  │
└────────┬────────┘
         ↓
┌─────────────────┐
│   Validation    │ → MongoDB (validated_triples)
└────────┬────────┘
         ↓
┌─────────────────┐
│     Fusion      │ → Neo4j (knowledge graph)
└────────┬────────┘
         ↓
┌─────────────────────────────────┐
│     Query Pipeline              │
│  ┌──────────┐  ┌─────────────┐ │
│  │  FAISS   │  │   Neo4j     │ │
│  │ Semantic │  │   Graph     │ │
│  └────┬─────┘  └──────┬──────┘ │
│       └────────┬───────┘        │
│                ↓                │
│         ┌────────────┐          │
│         │   Prompt   │          │
│         │  Builder   │          │
│         └─────┬──────┘          │
│               ↓                 │
│      ┌────────────────┐         │
│      │ Groq Llama 70B │         │
│      │   Reasoning    │         │
│      └────────┬───────┘         │
│               ↓                 │
│      ┌────────────────┐         │
│      │ GraphVerify    │         │
│      └────────────────┘         │
└─────────────────────────────────┘
```

## 🧠 Models Used

- **Extraction**: DeepSeek-R1-Distill-Qwen-1.5B (`deepseek-r1:1.5b`) via Ollama (local)
- **Reasoning/QA**: Llama-3.3-70B-Versatile via Groq Cloud API (fast inference)
- **Embeddings**: BGE-small (`BAAI/bge-small-en-v1.5`)

## 💾 Data Stores

- **MongoDB**: Document storage, triples, metadata, audit logs
- **Neo4j**: Canonical knowledge graph with versioning
- **FAISS**: Vector similarity search (CPU-based)

## 📁 Project Structure

```
graphbuilder-rag/
├── services/
│   ├── ingestion/          # Document ingestion
│   ├── normalization/      # Text extraction & cleaning
│   ├── extraction/         # Triple extraction (rules + LLM)
│   ├── embedding/          # BGE embeddings + FAISS
│   ├── entity_resolution/  # Entity linking & deduplication
│   ├── validation/         # Fact validation engine
│   ├── fusion/             # Neo4j graph fusion
│   ├── retrieval/          # Hybrid retrieval
│   ├── query/              # QA service with GraphVerify
│   └── agents/             # Self-healing agents
├── shared/
│   ├── config/             # Configuration management
│   ├── database/           # DB connectors
│   ├── models/             # Pydantic schemas
│   ├── prompts/            # LLM prompt templates
│   └── utils/              # Shared utilities
├── workers/                # Celery task workers
├── api/                    # FastAPI endpoints
├── tests/                  # Unit & integration tests
├── docker/                 # Docker configs
└── deployment/             # K8s/compose configs
```

## 🚀 Quick Start

### 1. Install Services

**macOS:**
```bash
brew install mongodb-community neo4j redis ollama tesseract poppler
```

**Linux:**
```bash
# See SETUP.md for detailed Linux installation
```

### 2. Start Services

```bash
# macOS
brew services start mongodb-community
brew services start neo4j
brew services start redis
ollama serve &

# Pull Ollama model (for extraction only)
ollama pull deepseek-r1:1.5b

# Get Groq API key for Q&A (free tier available)
# Visit: https://console.groq.com/keys
```

### 3. Setup Project

```bash
# Clone and setup
git clone <repository-url>
cd graphbuilder-rag
chmod +x setup.sh
./setup.sh
```

### 4. Run Application

**Option A: Separate terminals**
```bash
# Terminal 1: API
python -m api.main

# Terminal 2: Worker
celery -A workers.tasks worker --loglevel=info --concurrency=4

# Terminal 3: Beat
celery -A workers.tasks beat --loglevel=info

# Terminal 4: Agents (optional)
python -m agents.agents
```

**Option B: Tmux (all-in-one)**
```bash
chmod +x run.sh
./run.sh
```

### 5. Test the API

**Ingest a document:**
```bash
curl -X POST http://localhost:8000/api/v1/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "source": "https://en.wikipedia.org/wiki/Artificial_intelligence",
    "source_type": "HTML",
    "metadata": {"topic": "AI"}
  }'
```

**Query the system:**
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What are the side effects of aspirin?",
    "max_chunks": 5,
    "graph_depth": 2
  }'
```

## 🔧 Configuration

Edit `config/config.yaml`:

```yaml
mongodb:
  uri: mongodb://localhost:27017
  database: graphbuilder_rag

neo4j:
  uri: bolt://localhost:7687
  user: neo4j
  password: password

ollama:
  base_url: http://localhost:11434
  extraction_model: deepseek-r1:1.5b  # For entity/relationship extraction

groq:
  api_key: your-groq-api-key-here  # Get from https://console.groq.com/keys
  model: llama-3.3-70b-versatile  # For fast Q&A reasoning

faiss:
  index_type: IndexFlatIP
  embedding_dim: 384

agents:
  reverify_interval: 86400  # 24 hours
  conflict_check_interval: 3600  # 1 hour
```

## 📊 Monitoring

Access metrics at:
- API Health: `http://localhost:8000/health`
- Metrics: `http://localhost:8000/metrics`
- Neo4j Browser: `http://localhost:7474`
- MongoDB Compass: `mongodb://localhost:27017`

## 🧪 Testing

```bash
# Run all tests
pytest tests/

# Run specific service tests
pytest tests/services/extraction/

# Run integration tests
pytest tests/integration/
```

## 📖 Documentation

### Setup & Installation
- [Setup Guide](documentation/SETUP.md) - Complete installation and configuration
- [Installation Checklist](documentation/INSTALL_CHECKLIST.md) - Step-by-step setup verification
- [Quick Installation](documentation/INSTALLATION.md) - Fast setup for all platforms

### Architecture & Design
- [System Architecture](documentation/ARCHITECTURE.md) - Complete system overview
- [Framework Guide](documentation/FRAMEWORK_GUIDE.md) - Customization and extension guide
- [Celery & Agents](documentation/CELERY_AND_AGENTS_EXPLAINED.md) - Background tasks and autonomous agents

### Usage & Testing
- [Quick Start](documentation/QUICKSTART.md) - Get started in 5 minutes
- [Testing Guide](documentation/TESTING.md) - Test workflows and examples

### Advanced Topics
- [External Verification](documentation/EXTERNAL_VERIFICATION_SOLUTION.md) - Third-party fact checking

## 🤝 Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md)

## 📄 License

MIT License - see [LICENSE](LICENSE)
