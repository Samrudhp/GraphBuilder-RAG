## Changes Applied

### ✅ Option 2: Upgraded Extraction Model
- Changed from `deepseek-r1:1.5b` → `deepseek-r1:7b`
- Location: `.env` file, line 32
- The 7b model has better reasoning and fact separation capabilities

### ✅ Option 3: Improved Extraction Prompt
- Enhanced `EXTRACTION_SYSTEM_PROMPT` in `shared/prompts/templates.py`
- Added explicit rules against conflating multiple facts
- Included correct/incorrect examples:
  * ✓ "Albert Einstein" → "was born in" → "Ulm, Germany"
  * ✗ "Albert Einstein" → "was a physicist" → "Ulm, Germany"
- Specified predicate vocabulary: "was born in", "had occupation", "won award", etc.

### ✅ Database Cleared
- All previous test data removed from MongoDB
- Ready for fresh extraction test

## Next Steps

### 1. Restart Celery Worker (Terminal 2)
The worker needs to reload the new configuration:
```powershell
# Press Ctrl+C to stop current worker
# Then run:
celery -A workers.tasks worker --loglevel=info --pool=solo
```

### 2. Upload Test Document
```powershell
python upload_test.py
```

### 3. Monitor Progress
Watch Terminal 2 (Celery worker) for:
- ✅ Normalization complete
- ✅ Extraction with better triples
- ✅ Validation accepting triples (confidence > 0.8)
- ✅ Fusion into Neo4j graph

### 4. View Results
```powershell
# After pipeline completes (~2-3 minutes):
python view_triples.py
```

### Expected Improvements

**Before (1.5b model, old prompt):**
- "Albert Einstein" → "was a physicist" → "Ulm, Germany" ❌ (rejected, confidence 0.63)

**After (7b model, improved prompt):**
- "Albert Einstein" → "was born in" → "Ulm, Germany" ✓ (accepted, confidence 0.9+)
- "Albert Einstein" → "had occupation" → "physicist" ✓ (accepted, confidence 0.9+)
- "Marie Curie" → "was born in" → "Warsaw, Poland" ✓
- "Marie Curie" → "had occupation" → "physicist" ✓
- "Isaac Newton" → "had occupation" → "mathematician" ✓

The 7b model should extract **6-9 correct triples** instead of 3 malformed ones!
