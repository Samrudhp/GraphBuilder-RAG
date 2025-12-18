# FEVER Benchmark - Token Usage Breakdown

## Ultra-Conservative Rate Limiting Strategy

### Per Query Token Usage
Each evaluation query makes **3 Groq API calls**:

1. **NL2Cypher** (Cypher query generation)
   - Tokens: ~750
   
2. **QA Generation** (Answer generation)
   - Tokens: ~400
   
3. **GraphVerify** (Claim verification)
   - Tokens: ~150

**TOTAL PER QUERY: ~1,300 tokens**

---

## 50-Question Evaluation Plan

### Total Token Budget
- **50 questions × 1,300 tokens = 65,000 tokens**
- **Groq daily limit: 100,000 tokens**
- **Usage: 65% of daily limit** ✅ SAFE

### Batching Strategy
- **Batch size**: 5 questions
- **Total batches**: 10
- **Tokens per batch**: 6,500 tokens
- **API calls per batch**: 15 requests (3 per question × 5)

### Rate Limit Compliance

**Groq Limits:**
- ✅ 30 requests/minute
- ✅ 6,000 tokens/minute  
- ✅ 100,000 tokens/day

**Our Strategy:**
- ✅ 15 requests per batch (50% of limit)
- ✅ 30-second buffer after EACH question
- ✅ 3-minute (180 second) wait between batches
- ✅ 65,000 tokens total (65% of daily limit)

### Timeline

**Per Batch:**
- 5 questions × 30 seconds buffer = 150 seconds (2.5 min)
- Actual query time: ~50 seconds
- **Total per batch: ~200 seconds (3.3 min)**

**Between Batches:**
- Inter-batch delay: 180 seconds (3 min)

**Total Time:**
- 10 batches × 3.3 min = 33 minutes (questions)
- 9 delays × 3 min = 27 minutes (waiting)
- **TOTAL: ~60 minutes** (with generous buffer)

---

## Safety Margins

### Multiple Layers of Protection

1. **Per-Question Buffer**: 30 seconds
   - Prevents rapid-fire requests
   - Allows API to reset between calls

2. **Batch Delay**: 180 seconds (3 minutes)
   - Ensures we never exceed 30 req/min
   - Gives API plenty of recovery time

3. **Conservative Batch Size**: 5 questions
   - Only 15 requests (50% of 30/min limit)
   - Only 6,500 tokens (108% of 6K/min limit, spread over 3+ minutes)

4. **Total Token Budget**: 65,000
   - Only 65% of 100K daily limit
   - 35K token safety margin

### What This Means

**You CANNOT hit rate limits with this setup because:**

✅ Per-minute requests: 15 requests per 3.3 min = **4.5 req/min** (limit: 30)  
✅ Per-minute tokens: 6,500 tokens per 3.3 min = **1,970 tokens/min** (limit: 6,000)  
✅ Daily tokens: 65,000 total (limit: 100,000)

**We're using:**
- 15% of requests/minute capacity
- 33% of tokens/minute capacity  
- 65% of daily token capacity

---

## Emergency Adjustments

If you somehow STILL get rate limit errors (extremely unlikely):

### Make it even safer:
```python
# In 2_run_evaluation.py

# Change this:
batch_size = 5
await asyncio.sleep(180)  # 3 minutes

# To this:
batch_size = 3           # Only 3 questions per batch
await asyncio.sleep(300)  # 5 minutes between batches
```

This would make it:
- **3 questions × 1,300 = 3,900 tokens per batch**
- **9 requests per batch** (instead of 15)
- **17 batches total** (instead of 10)
- **Total time: ~90 minutes** (instead of 60)

But with current settings, **you should have ZERO errors**. 🎯
