# FEVER Evaluation Metrics - Paper Alignment

## ✅ COMPLETE COVERAGE OF PAPER REQUIREMENTS

This document maps the evaluation script output to your paper's required metrics.

---

## 1️⃣ RETRIEVAL METRICS (Section 2.2 - Required)

### Paper Requirement → Script Output

| Paper Metric | JSON Key | Description |
|-------------|----------|-------------|
| **Precision@k** | `retrieval_metrics.precision_at_5` | Precision at k=5 (max_chunks=5) |
| **Recall@k** | `retrieval_metrics.recall_at_5` | Recall at k=5 |
| **MRR** | `retrieval_metrics.mrr` | Mean Reciprocal Rank |
| **Graph node coverage** | `retrieval_metrics.graph_node_coverage` | % queries using graph |
| **Path length distribution** | `graph_path_metrics.path_length_distribution` | Nodes per query |

### Additional Retrieval Metrics:
- `avg_graph_nodes_retrieved` - Average graph entities per query
- `avg_text_chunks_retrieved` - Average text chunks per query

---

## 2️⃣ GENERATION METRICS (Section 2.2 - Required)

### Paper Requirement → Script Output

| Paper Metric | JSON Key | Description |
|-------------|----------|-------------|
| **Factual accuracy** | `generation_metrics.factual_accuracy` | Overall accuracy % |
| **Hallucination rate** | `generation_metrics.hallucination_rate` | % hallucinated answers |
| **Verification status distribution** | `graphverify_metrics.verification_status_distribution` | VERIFIED/PARTIAL/UNSUPPORTED/CONTRADICTED |

### Breakdown by Claim Type:
- `accuracy_supports` - Accuracy on SUPPORTS claims (20 samples)
- `accuracy_refutes` - Accuracy on REFUTES claims (20 samples)
- `accuracy_nei` - Accuracy on NOT ENOUGH INFO (10 samples)

---

## 3️⃣ GRAPHVERIFY METRICS (Section 1.3 Ablation - Required)

### Paper Requirement → Script Output

| Paper Metric | JSON Key | Description |
|-------------|----------|-------------|
| **Hallucination Rate** | `graphverify_metrics.hallucination_rate` | % with contradicted/unsupported |
| **Unsupported claims per answer** | `graphverify_metrics.avg_unsupported_claims_per_answer` | Avg unsupported claims |
| **Verification Confidence** | `graphverify_metrics.verification_confidence` | Avg confidence score |

### Verification Status Counts:
- `verified_count` - Fully verified answers
- `partial_count` - Partially verified
- `unsupported_count` - Unsupported by graph
- `contradicted_count` - Contradicted by graph

---

## 4️⃣ GRAPH PATH METRICS (Multi-Hop Analysis)

### New Metrics for Multi-Hop Reasoning

| Metric | JSON Key | Description |
|--------|----------|-------------|
| **Avg path length** | `graph_path_metrics.avg_path_length` | Mean nodes traversed |
| **Max path length** | `graph_path_metrics.max_path_length` | Longest path |
| **Min path length** | `graph_path_metrics.min_path_length` | Shortest path |

### Path Length Distribution:
- `0_nodes` - Queries with no graph usage
- `1_2_nodes` - Simple 1-2 hop queries
- `3_5_nodes` - Medium complexity
- `5_plus_nodes` - Complex multi-hop

---

## 📊 PAPER SECTION MAPPING

### Table 2: FEVER Benchmark Results

```
| Metric              | Value    |
|---------------------|----------|
| Factual Accuracy    | XX.X%    | ← generation_metrics.factual_accuracy
| Precision@5         | 0.XXX    | ← retrieval_metrics.precision_at_5
| Recall@5            | 0.XXX    | ← retrieval_metrics.recall_at_5
| MRR                 | 0.XXX    | ← retrieval_metrics.mrr
| Hallucination Rate  | XX.X%    | ← graphverify_metrics.hallucination_rate
| Verification Conf.  | 0.XX     | ← graphverify_metrics.verification_confidence
| Avg Path Length     | X.X      | ← graph_path_metrics.avg_path_length
```

### Figure 3: Path Length Distribution (Bar Chart)

Use: `graph_path_metrics.path_length_distribution`

X-axis: [0 nodes, 1-2 nodes, 3-5 nodes, 5+ nodes]
Y-axis: Count of queries

### Figure 4: Verification Status (Pie Chart)

Use: `graphverify_metrics.verification_status_distribution`

Slices:
- VERIFIED: XX.X%
- PARTIAL: XX.X%
- UNSUPPORTED: XX.X%
- CONTRADICTED: XX.X%

### Figure 5: Accuracy by Claim Type (Grouped Bar Chart)

Use:
- `generation_metrics.accuracy_supports` (SUPPORTS)
- `generation_metrics.accuracy_refutes` (REFUTES)
- `generation_metrics.accuracy_nei` (NOT ENOUGH INFO)

---

## 🎯 ABLATION STUDY METRICS (Section 1)

### 1.3 GraphVerify Ablation

**Compare:**
- **With GraphVerify**: Current results
- **Without GraphVerify**: Run separate evaluation without verification

**Metrics to compare:**
- Hallucination Rate reduction
- Unsupported claims reduction
- Verification confidence improvement

**Expected Graph:**
```
Before GraphVerify:  Hallucination Rate = 45-60%
After GraphVerify:   Hallucination Rate = 10-20%
```

---

## 📈 QUANTITATIVE EVALUATION (Section 2)

### Section 2.1: FEVER Dataset

Use ALL metrics from above tables.

### Section 2.2: Key Findings

**Retrieval Performance:**
- Report Precision@5, Recall@5, MRR
- Highlight graph node coverage %

**Generation Quality:**
- Report factual accuracy by claim type
- Show verification status distribution

**GraphVerify Impact:**
- Report hallucination rate
- Show avg unsupported claims per answer

---

## 📝 SAMPLE JSON OUTPUT STRUCTURE

```json
{
  "timestamp": "2025-12-14T...",
  "total_samples": 50,
  
  "generation_metrics": {
    "factual_accuracy": 85.0,
    "accuracy_supports": 90.0,
    "accuracy_refutes": 85.0,
    "accuracy_nei": 70.0,
    "hallucination_rate": 15.0,
    "avg_confidence": 0.82,
    "avg_latency": 3.45
  },
  
  "retrieval_metrics": {
    "precision_at_5": 0.750,
    "recall_at_5": 0.850,
    "mrr": 0.850,
    "avg_graph_nodes_retrieved": 3.2,
    "avg_text_chunks_retrieved": 4.1,
    "graph_node_coverage": 85.0
  },
  
  "graph_path_metrics": {
    "avg_path_length": 3.2,
    "max_path_length": 7,
    "min_path_length": 0,
    "path_length_distribution": {
      "0_nodes": 5,
      "1_2_nodes": 15,
      "3_5_nodes": 25,
      "5_plus_nodes": 5
    }
  },
  
  "graphverify_metrics": {
    "hallucination_rate": 15.0,
    "avg_unsupported_claims_per_answer": 0.22,
    "verification_confidence": 0.82,
    "verification_status_distribution": {
      "VERIFIED": 70.0,
      "PARTIAL": 15.0,
      "UNSUPPORTED": 10.0,
      "CONTRADICTED": 5.0
    },
    "verified_count": 35,
    "partial_count": 8,
    "unsupported_count": 5,
    "contradicted_count": 2
  }
}
```

---

## ✅ COMPLIANCE CHECKLIST

### Paper Requirements vs Script Output

- [x] **Retrieval Metrics** (Section 2.2)
  - [x] Precision@k
  - [x] Recall@k
  - [x] MRR
  - [x] Graph node coverage
  - [x] Path length distribution

- [x] **Generation Metrics** (Section 2.2)
  - [x] Factual accuracy
  - [x] Hallucination rate
  - [x] Verification status distribution

- [x] **GraphVerify Metrics** (Section 1.3)
  - [x] Hallucination Rate
  - [x] Unsupported claims per answer
  - [x] Verification Confidence

- [x] **Additional KG Metrics** (Section 2.2)
  - [x] Graph node retrieval stats
  - [x] Multi-hop path analysis
  - [x] Latency measurements

---

## 🚀 RESULT

**THE EVALUATION SCRIPT NOW PROVIDES 100% OF REQUIRED METRICS FOR YOUR PAPER.**

All metrics align with:
- Section 1.3: Ablation Studies (GraphVerify)
- Section 2.1: FEVER Dataset Evaluation
- Section 2.2: Quantitative Metrics
- Figures 3-5: Visualization requirements
