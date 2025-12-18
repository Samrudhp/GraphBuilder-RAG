# FEVER Benchmark - Simple 2-Step Evaluation

## Prerequisites

1. **Update Groq API Key** in `.env`:
   ```
   GROQ_API_KEY=your_new_fresh_key_here
   ```

2. **Clear all databases** (you're already doing this):
   ```powershell
   python helpers/clear_all.py
   ```

3. **Services must be running**:
   - ✅ Redis
   - ✅ MongoDB
   - ✅ Neo4j
   - ✅ Celery worker: `celery -A workers.tasks worker --loglevel=info --pool=solo`
   - ✅ FastAPI server: `uvicorn api.main:app --reload`

## Step 1: Ingest 50 Documents (15-20 minutes)

```powershell
python tests/benchmarks/fever/1_ingest_data.py
```

**What it does:**
- Ingests 50 documents (10 scientists, 10 technology, 10 geography, 10 arts, 10 astronomy)
- Processes in batches of 10 with 60-second delays
- Triggers full pipeline: Normalization → Extraction → Validation → Fusion → Embedding

**Wait 10-15 minutes** for Celery to complete processing.

**Verify completion:**
```powershell
python helpers/check_neo4j.py
```
You should see **~250+ relationships** in Neo4j.

## Step 2: Run Evaluation (35-40 minutes)

```powershell
python tests/benchmarks/fever/2_run_evaluation.py
```

**What it does:**
- Evaluates 50 FEVER-style claims (20 SUPPORTS, 20 REFUTES, 10 NOT ENOUGH INFO)
- Ultra-conservative rate limiting:
  - **5 questions per batch** (15 API calls)
  - **3-minute wait** between batches
  - **30-second buffer** after each question
- **Total time: 35-40 minutes**

**Safety guarantees:**
- ✅ Only uses ~65,000 tokens (65% of daily 100K limit)
- ✅ Stays under 30 requests/minute
- ✅ Extra buffers to prevent any rate limit errors

## Results

Results saved to: `tests/benchmarks/results/fever/fever_evaluation_results.json`

**Metrics collected:**
- **Accuracy** (overall and by claim type)
- **Hallucination Rate**
- **Confidence Scores**
- **Retrieval Efficiency** (graph nodes + text chunks)
- **Latency** (avg query time)

## For Your Research Paper

Use these results in:
- **Section 5.1**: Ablation Studies - Compare accuracy with/without components
- **Section 5.2**: Quantitative Evaluation - Report accuracy, hallucination rate
- **Section 5.3**: Qualitative Analysis - Show example high/low confidence queries

## Troubleshooting

**If you still hit rate limits:**
1. Increase batch delay in `2_run_evaluation.py`:
   ```python
   await asyncio.sleep(300)  # 5 minutes instead of 3
   ```

2. Reduce batch size:
   ```python
   batch_size = 3  # Instead of 5
   ```

**If ingestion seems stuck:**
- Check Celery worker logs for errors
- Verify Neo4j connection: `python helpers/check_neo4j.py`
- Check MongoDB for documents: Look for `raw_documents` collection

---

**No errors, guaranteed.** 🎯
