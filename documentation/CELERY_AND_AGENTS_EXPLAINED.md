# Understanding Celery & Agents in GraphBuilder-RAG

## 🎯 The Problem We're Solving

Imagine you upload a 100-page PDF to process. If the API tried to process it immediately:
- The HTTP request would timeout (most browsers timeout after 30-60 seconds)
- The server couldn't handle multiple uploads at once
- Users would have to wait with their browser open until processing finishes

**Solution**: Celery — a distributed task queue system

---

## 📦 What is Celery?

Celery is like a **smart job dispatcher** that:
1. Takes long-running tasks
2. Puts them in a queue (Redis)
3. Lets "worker" processes pick them up and execute them
4. Allows the API to immediately return a response

### Real-World Analogy

Think of a restaurant:

```
┌─────────────┐         ┌─────────────┐         ┌─────────────┐
│  Customer   │         │   Waiter    │         │    Chef     │
│  (Client)   │────────>│   (API)     │────────>│  (Celery    │
│             │ "Order" │             │ "Ticket"│   Worker)   │
└─────────────┘         └─────────────┘         └─────────────┘
                              │                        │
                              │ "Here's your           │
                              │  order number"         │ (Cooks in
                              │                        │  background)
                              v                        v
                        "Come back                "Food ready!"
                         later"                   (Notification)
```

**Without Celery**: Waiter takes order, goes to kitchen, cooks food, brings it back (customer waits forever)  
**With Celery**: Waiter takes order, gives ticket to chef, tells customer "we'll call you" (customer can leave)

---

## 🏗️ How Celery Works in Our System

### Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                      FastAPI Server                          │
│  POST /api/v1/ingest → Creates task → Returns task_id       │
└────────────────────────┬─────────────────────────────────────┘
                         │
                         v
┌──────────────────────────────────────────────────────────────┐
│                    Redis (Message Broker)                    │
│  Queue: [Task1, Task2, Task3, Task4, ...]                   │
└────────────────────────┬─────────────────────────────────────┘
                         │
                         v
┌──────────────────────────────────────────────────────────────┐
│                    Celery Workers                            │
│                                                              │
│  Worker 1: Processing Task1                                 │
│  Worker 2: Processing Task2                                 │
│  Worker 3: Processing Task3                                 │
│  Worker 4: Idle (waiting for next task)                     │
└──────────────────────────────────────────────────────────────┘
                         │
                         v
┌──────────────────────────────────────────────────────────────┐
│                    Stores Results in Redis                   │
│  task_123: {"status": "completed", "result": {...}}         │
└──────────────────────────────────────────────────────────────┘
```

### Step-by-Step Flow

**1. User Sends Request**
```bash
curl -X POST http://localhost:8000/api/v1/ingest \
  -d '{"source": "document.pdf", "source_type": "PDF"}'
```

**2. API Creates Task** (`api/main.py`)
```python
@router.post("/api/v1/ingest")
async def ingest_document(request: IngestRequest):
    # Don't process now - send to Celery
    task = process_document_task.delay(request.source, request.source_type)
    
    # Immediately return task ID
    return {
        "document_id": str(task.id),
        "status": "processing",
        "message": "Task queued. Check status later."
    }
```

**3. Celery Picks Up Task** (`workers/tasks.py`)
```python
@celery_app.task(name="process_document")
def process_document_task(source: str, source_type: str):
    # This runs in the WORKER process, not the API
    
    # Step 1: Ingest
    doc_id = await ingestion_service.ingest(source, source_type)
    
    # Step 2: Normalize
    await normalization_service.normalize(doc_id)
    
    # Step 3: Extract triples
    await extraction_service.extract(doc_id)
    
    # Step 4: Embed chunks
    await embedding_service.embed(doc_id)
    
    return {"status": "completed", "document_id": doc_id}
```

**4. User Checks Status**
```bash
curl http://localhost:8000/api/v1/status/task_123
```

Returns:
```json
{
  "task_id": "task_123",
  "status": "processing",  // or "completed" or "failed"
  "progress": 60,
  "current_step": "extracting triples"
}
```

---

## 🔧 Celery Components in Our System

### 1. **Broker (Redis)**
- **Purpose**: Stores the task queue
- **Location**: `redis://localhost:6379/0`
- **What it stores**: Task messages (function name + arguments)

```python
# In Redis, a task looks like:
{
  "id": "abc123",
  "task": "process_document_task",
  "args": ["document.pdf", "PDF"],
  "kwargs": {},
  "eta": None
}
```

### 2. **Workers** (`celery -A workers.tasks worker`)
- **Purpose**: Execute tasks from the queue
- **Count**: Configurable (we use 4 concurrent workers)
- **What they do**: Poll Redis, pick up tasks, execute Python functions

```bash
# Start 4 workers
celery -A workers.tasks worker --loglevel=info --concurrency=4
```

Output:
```
[2025-12-06 10:00:00] celery@hostname ready.
[2025-12-06 10:00:05] Task process_document[abc123] received
[2025-12-06 10:00:10] Task process_document[abc123] succeeded in 5.2s
```

### 3. **Beat Scheduler** (`celery -A workers.tasks beat`)
- **Purpose**: Schedule periodic tasks (like cron)
- **What it does**: Sends tasks to queue at specified times

```python
# In workers/tasks.py
celery_app.conf.beat_schedule = {
    'reverify-old-triples-daily': {
        'task': 'reverify_old_triples',
        'schedule': crontab(hour=2, minute=0),  # Every day at 2 AM
    },
    'resolve-entities-hourly': {
        'task': 'resolve_entities_task',
        'schedule': crontab(minute=0),  # Every hour
    },
}
```

### 4. **Result Backend (Redis)**
- **Purpose**: Store task results
- **What it stores**: Task status, return values, errors

```python
# When task completes, Redis stores:
{
  "task_id": "abc123",
  "status": "SUCCESS",
  "result": {"document_id": "doc_456", "triples": 120},
  "traceback": null
}
```

### 5. **Flower (Monitoring UI)**
- **Purpose**: Web dashboard to monitor tasks
- **Access**: `http://localhost:5555`
- **Features**: See active workers, task history, success rates

---

## 🤖 What Are Agents?

Agents are **autonomous background processes** that continuously maintain and improve the knowledge graph.

Think of them as **robot janitors and quality inspectors** for your knowledge graph.

### Agent vs Regular Task

| Feature | Regular Celery Task | Agent |
|---------|-------------------|-------|
| **Triggered by** | User request (API call) | Runs continuously or on schedule |
| **Purpose** | Process specific document | Maintain entire system |
| **Duration** | Finishes after one document | Runs forever in background |
| **Example** | Process a PDF | Check all facts daily |

---

## 🧠 The 3 Agents in Our System

### 1. **ReverifyAgent** (`agents/agents.py`)

**Purpose**: Periodically re-check old facts against external sources to catch outdated information

**External Verification Sources**:
- **Wikidata** (0.9 weight): SPARQL queries to structured knowledge base
- **DBpedia** (0.8 weight): SPARQL queries to Wikipedia-extracted data  
- **Wikipedia** (0.7 weight): Text search in article content

**How it works**:
```
Every Hour (configurable):
1. Query MongoDB for triples not verified in last 7 days
2. For each triple:
   a. Query Wikidata SPARQL for structured verification
   b. Query DBpedia SPARQL for relationship confirmation
   c. Search Wikipedia API for co-occurrence evidence
3. Calculate weighted confidence score from all sources
4. If confidence drops > 0.2 → flag for human review
5. Update last_verified timestamp
```

**Code**:
```python
class ReverifyAgent:
    async def _verify_external(self, triple: ValidatedTriple) -> float:
        """Multi-source verification with weighted scoring."""
        results = []
        
        # 1. Wikidata SPARQL (highest confidence)
        wikidata_score = await self._verify_wikidata(triple)
        if wikidata_score: results.append((wikidata_score, 0.9))
        
        # 2. DBpedia SPARQL
        dbpedia_score = await self._verify_dbpedia(triple)
        if dbpedia_score: results.append((dbpedia_score, 0.8))
        
        # 3. Wikipedia text search
        wikipedia_score = await self._verify_wikipedia(triple)
        if wikipedia_score: results.append((wikipedia_score, 0.7))
        
        # Weighted average
        return sum(s*w for s,w in results) / sum(w for _,w in results)
    
    async def _verify_wikidata(self, triple) -> float:
        """Query Wikidata SPARQL endpoint."""
        query = f'''
        SELECT ?item ?value WHERE {{
          ?item rdfs:label "{triple.subject}"@en .
          ?item ?predicate ?value .
          ?value rdfs:label "{triple.object}"@en .
        }}
        '''
        # Returns 1.0 if found, 0.3 if not found, None if error
    
    async def _verify_dbpedia(self, triple) -> float:
        """Query DBpedia SPARQL endpoint."""
        # Similar SPARQL query to DBpedia
    
    async def _verify_wikipedia(self, triple) -> float:
        """Search Wikipedia API for co-occurrence."""
        # Checks if object is mentioned in subject's Wikipedia page
```

**Example**:
```
Day 1: Graph has "John works_at Google" (confidence 0.85)
Day 7: ReverifyAgent runs:
  - Wikidata: No result (None)
  - DBpedia: No result (None)
  - Wikipedia: "John" page mentions "Microsoft" not "Google" (0.4)
  - New confidence: 0.4 (dropped from 0.85)
  
Action: Flag for human review due to confidence drop > 0.2
MongoDB: Insert into human_review_queue collection
```

**Configuration** (`.env`):
```bash
AGENT_REVERIFY_INTERVAL_SECONDS=3600      # Run every hour
AGENT_REVERIFY_BATCH_SIZE=100             # Check 100 triples per run
VALIDATION_EXTERNAL_TIMEOUT=10            # 10s timeout per API call
```

---

### 2. **ConflictResolverAgent** (`agents/agents.py`)

**Purpose**: Detect and resolve contradictory facts in the knowledge graph using LLM reasoning

**How it works**:
```
Every 2 Hours (configurable):
1. Find triples that contradict each other
   Example: "Einstein died_in Germany" vs "Einstein died_in USA"
2. Query Neo4j for conflicting edges (same entity + relationship → different targets)
3. Use Ollama LLM to analyze evidence and determine winner
4. Deprecate losing edges, promote winner
```
   - Or flag for human review
```

**Code**:
```python
class ConflictResolverAgent:
    async def find_conflicts(self):
        # Find contradictory relationships
        conflicts = await self.neo4j.query("""
            MATCH (s)-[r1]->(o), (s)-[r2]->(o)
            WHERE r1.predicate = 'treats' AND r2.predicate = 'causes'
            RETURN s, r1, r2, o
        """)
        
        for conflict in conflicts:
            if conflict['r1'].confidence > conflict['r2'].confidence:
                await self.neo4j.delete_relationship(conflict['r2'])
            elif conflict['r1'].source_date > conflict['r2'].source_date:
                # Newer source wins
                await self.neo4j.update_relationship(conflict['r1'])
            else:
                # Flag for human review
                await self.flag_for_review(conflict)
```

**Example**:
```
Conflict Found:
  Triple 1: "Coffee causes insomnia" (confidence: 0.8, source: Study A)
  Triple 2: "Coffee treats fatigue" (confidence: 0.9, source: Study B)

Resolution:
  Both are valid in different contexts!
  Action: Add context nodes "short_term_effect" and "long_term_effect"
  
New Graph:
  Coffee -[causes, context=immediate]-> Insomnia
  Coffee -[treats, context=short_term]-> Fatigue
```

---

### 3. **SchemaSuggestorAgent** (`agents/agents.py`)

**Purpose**: Analyze the graph and suggest new entity types or relationships

**How it works**:
```
Every Week:
1. Analyze existing triples
2. Find common patterns not in current schema
3. Suggest new entity types or relationships
4. Log suggestions for admin review
```

**Code**:
```python
class SchemaSuggestorAgent:
    async def analyze_schema(self):
        # Find common entity patterns not in schema
        frequent_entities = await self.neo4j.query("""
            MATCH (n)
            WHERE NOT n:Person AND NOT n:Organization AND NOT n:Location
            RETURN labels(n), count(*) as freq
            ORDER BY freq DESC
            LIMIT 10
        """)
        
        for entity_type, frequency in frequent_entities:
            if frequency > 100:  # Threshold
                suggestion = f"Consider adding '{entity_type}' as entity type"
                await self.log_suggestion(suggestion)
        
        # Find common relationship patterns
        frequent_rels = await self.neo4j.query("""
            MATCH ()-[r]->()
            RETURN type(r), count(*) as freq
            ORDER BY freq DESC
        """)
        
        for rel_type, frequency in frequent_rels:
            if rel_type not in self.config.RELATIONSHIP_TYPES:
                suggestion = f"New relationship detected: '{rel_type}' ({frequency} occurrences)"
                await self.log_suggestion(suggestion)
```

**Example**:
```
Analysis Result:
  - Found 250 entities with label "Disease" (not in schema)
  - Found 180 relationships "prevents" (not in schema)
  
Suggestions:
  1. Add "Disease" to EXTRACTION_ENTITY_TYPES
  2. Add "prevents" to EXTRACTION_RELATIONSHIP_TYPES
  3. Update extraction prompts to recognize these patterns
```

---

## 🔄 How Agents and Celery Work Together

### Periodic Tasks with Celery Beat

```python
# In workers/tasks.py

@celery_app.task(name="reverify_old_triples")
def reverify_old_triples_task():
    """Celery task that triggers the ReverifyAgent."""
    agent = ReverifyAgent()
    asyncio.run(agent.run_task())

@celery_app.task(name="resolve_conflicts")
def resolve_conflicts_task():
    """Celery task that triggers the ConflictResolverAgent."""
    agent = ConflictResolverAgent()
    asyncio.run(agent.run_task())

# Schedule them with Celery Beat
celery_app.conf.beat_schedule = {
    'reverify-daily': {
        'task': 'reverify_old_triples',
        'schedule': crontab(hour=2, minute=0),  # 2 AM daily
    },
    'resolve-conflicts-hourly': {
        'task': 'resolve_conflicts',
        'schedule': crontab(minute=0),  # Every hour
    },
}
```

### Timeline Example

```
Time        | API Activity           | Celery Workers         | Agents
------------|------------------------|------------------------|------------------
10:00 AM    | User uploads PDF       | Task queued            |
10:00:05    | Returns task_id        | Worker 1 starts        |
10:05 AM    |                        | Worker 1 processing    |
10:10 AM    |                        | Task completed         |
11:00 AM    |                        |                        | ConflictResolver runs
11:05 AM    |                        |                        | Resolved 3 conflicts
02:00 AM    |                        |                        | ReverifyAgent runs
02:30 AM    |                        |                        | Checked 1000 triples
```

---

## 🎮 Controlling Celery & Agents

### Start All Services with `run.sh`

```bash
./run.sh
```

This creates a **tmux session** with 5 windows:

```
┌─────────────────────────────────────────┐
│ Tmux Session: graphbuilder             │
├─────────────────────────────────────────┤
│ Window 1: api       (FastAPI server)   │
│ Window 2: worker    (Celery workers)   │
│ Window 3: beat      (Task scheduler)   │
│ Window 4: agents    (Agent processes)  │
│ Window 5: flower    (Monitoring UI)    │
└─────────────────────────────────────────┘
```

### Navigate Tmux Windows

```bash
# Attach to session
tmux attach -t graphbuilder

# Switch windows
Ctrl+b, 1  # Go to API window
Ctrl+b, 2  # Go to Worker window
Ctrl+b, 3  # Go to Beat window
Ctrl+b, 4  # Go to Agents window
Ctrl+b, 5  # Go to Flower window

# Detach (leave running in background)
Ctrl+b, d

# Stop all
tmux kill-session -t graphbuilder
```

### Manual Control

```bash
# Start each component separately

# Terminal 1: API
source venv/bin/activate
python -m api.main

# Terminal 2: Celery Workers (4 concurrent)
source venv/bin/activate
celery -A workers.tasks worker --loglevel=info --concurrency=4

# Terminal 3: Celery Beat (scheduler)
source venv/bin/activate
celery -A workers.tasks beat --loglevel=info

# Terminal 4: Agents
source venv/bin/activate
python -m agents.agents

# Terminal 5: Flower (monitoring)
source venv/bin/activate
celery -A workers.tasks flower --port=5555
```

---

## 📊 Monitoring

### 1. Flower Dashboard

```bash
open http://localhost:5555
```

Shows:
- Active workers and their status
- Task history (success/failure rates)
- Task execution times
- Queue lengths

### 2. Celery CLI

```bash
# Check active workers
celery -A workers.tasks inspect active

# Check registered tasks
celery -A workers.tasks inspect registered

# Check scheduled tasks
celery -A workers.tasks inspect scheduled

# Purge all tasks in queue
celery -A workers.tasks purge
```

### 3. API Task Status

```bash
# Check task status via API
curl http://localhost:8000/api/v1/tasks/task_abc123

# Response
{
  "task_id": "task_abc123",
  "status": "PENDING",     # or SUCCESS, FAILURE, RETRY
  "result": null,          # or {"document_id": "..."}
  "error": null
}
```

---

## 🐛 Troubleshooting

### Workers Not Processing Tasks

```bash
# Check if workers are running
celery -A workers.tasks inspect active

# If no workers:
celery -A workers.tasks worker --loglevel=debug
```

### Tasks Getting Stuck

```bash
# Check Redis queue
redis-cli
> KEYS celery*
> LLEN celery  # Queue length

# Clear stuck tasks
celery -A workers.tasks purge
```

### Agents Not Running

```bash
# Check if agents process is running
ps aux | grep agents

# Check agent logs
tail -f logs/agents.log

# Restart agents
python -m agents.agents
```

### Beat Not Scheduling

```bash
# Check beat is running
ps aux | grep "celery beat"

# View scheduled tasks
celery -A workers.tasks inspect scheduled

# Restart beat
celery -A workers.tasks beat --loglevel=debug
```

---

## 🎯 Summary

### Celery
- **What**: Distributed task queue for async processing
- **Why**: Prevents API timeouts, enables parallel processing
- **How**: API queues tasks → Redis stores them → Workers execute them
- **Components**: Broker (Redis), Workers (Python processes), Beat (scheduler), Flower (UI)

### Agents
- **What**: Autonomous background processes for system maintenance
- **Why**: Keep knowledge graph accurate and up-to-date without manual intervention
- **How**: Run periodically via Celery Beat or continuously in background
- **Types**: 
  - ReverifyAgent (checks old facts)
  - ConflictResolverAgent (fixes contradictions)
  - SchemaSuggestorAgent (improves schema)

### Key Insight
```
User Request → API (instant response) → Celery (async processing) → Agents (maintenance)
     ↓              ↓                        ↓                           ↓
  "Upload PDF"   "Task queued"         "Processing..."           "Graph healthy"
```

The system is designed so **users get instant feedback**, **heavy processing happens in background**, and **agents keep everything clean automatically**. This is how modern production systems handle complex, long-running workflows! 🚀
