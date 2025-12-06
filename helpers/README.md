# Helper Scripts

This directory contains utility scripts for testing and managing the GraphBuilder-RAG system.

## Test Scripts

### `upload_test.py`
Upload the test document to the API for ingestion.

```powershell
cd helpers
python upload_test.py
```

**What it does:**
- Uploads `scientists.txt` to the API
- Triggers the document processing pipeline
- Returns the document ID

---

### `test_query.py` ⭐ CONFERENCE FEATURE
Interactive natural language query interface with hybrid retrieval and NL2Cypher.

```powershell
cd helpers
python test_query.py
```

**What it demonstrates:**
- **Natural language → Cypher**: LLM converts questions to Cypher queries (NL2Cypher)
- **Graph-based retrieval**: Execute queries on Neo4j knowledge graph
- **Hybrid search**: Combines graph retrieval with FAISS semantic search
- **GraphVerify**: Detects hallucinations in generated answers
- **Verifiable reasoning**: Graph queries are logged for transparency

**Conference Paper Theme**: "Querying property graphs with natural language interfaces powered by LLMs"

**Example queries:**
- "Who was Isaac Newton?"
- "What was Marie Curie's occupation?"
- "Where was Einstein born?"

---

### `view_triples.py`
View all extracted and validated triples from MongoDB.

```powershell
cd helpers
python view_triples.py
```

**What it displays:**
- Total number of triples
- For each triple:
  - Subject, Predicate, Object
  - Confidence score
  - Validation status (accepted/rejected)
  - Wikipedia and Wikidata verification scores

---

### `check_neo4j.py`
Verify Neo4j graph database contents.

```powershell
cd helpers
python check_neo4j.py
```

**What it shows:**
- Entity nodes count
- Relationships count
- Sample entities with their properties
- Sample relationships (edges)

---

### `trigger_embedding.py`
Manually trigger embedding task for existing documents.

```powershell
cd helpers
python trigger_embedding.py
```

**What it does:**
- Finds normalized documents in MongoDB
- Triggers `embed_document` Celery task
- Creates FAISS vector embeddings for semantic search

---

### `clear_db.py`
Clear all test data from MongoDB collections.

```powershell
cd helpers
python clear_db.py
```

**What it clears:**
- raw_documents
- normalized_docs
- candidate_triples
- validated_triples
- embeddings_meta
- chunks

**Note:** Use this when you want a clean slate for testing.

---

## Test Data

### `scientists.txt`
Sample text document about famous scientists (Einstein, Curie, Newton).

Used for testing the complete document processing pipeline:
1. Ingestion
2. Normalization
3. Triple extraction
4. Validation
5. Fusion into Neo4j
6. Embedding for FAISS

---

## Conference Paper Testing Workflow

To demonstrate **NL2Cypher** for the paper:

```powershell
# 1. Ensure data is loaded
cd helpers
python check_neo4j.py  # Should show entities

# 2. Test interactive query with NL2Cypher + hybrid retrieval
python test_query.py
# Try: "Who was Isaac Newton?"
```

**Key Demonstration Points:**
- Natural language questions converted to Cypher by LLM
- Queries executed on property graph (Neo4j)
- Results are verifiable (queries logged in terminal output)
- Combines graph retrieval with semantic search (hybrid)
- GraphVerify detects hallucinations in generated answers

---

## Complete Test Workflow

```powershell
# 1. Clear previous test data
cd helpers
python clear_db.py

# 2. Upload test document
python upload_test.py

# 3. Wait for pipeline to complete (~2-3 minutes)
# Watch the Celery worker terminal for progress

# 4. Trigger embedding (if not automatic)
python trigger_embedding.py

# 5. Verify Neo4j data
python check_neo4j.py

# 6. View MongoDB triples
python view_triples.py

# 6. View MongoDB triples
python view_triples.py

# 7. Interactive querying with NL2Cypher
python test_query.py
```
## Running from Root Directory

If you're in the project root, use:

```powershell
python helpers/upload_test.py
python helpers/test_nl2cypher.py
```powershell
python helpers/upload_test.py
python helpers/test_query.py
python helpers/view_triples.py
python helpers/check_neo4j.py
python helpers/clear_db.py
```