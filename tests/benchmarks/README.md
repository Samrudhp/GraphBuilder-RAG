# GraphBuilder-RAG Benchmark Suite

Comprehensive benchmark testing framework for evaluating GraphBuilder-RAG system across multiple dimensions:
- **Factuality & Verification** (FEVER, SciFact)
- **Multi-Hop Reasoning** (HotpotQA, MetaQA)
- **Entity Linking** (Wikidata5M)
- **KG Construction** (DBpedia)
- **Trustworthiness** (TrustKG - custom synthetic)

## 🎯 Quick Start

### Install Dependencies

```bash
cd tests/benchmarks
pip install -r requirements.txt
```

### Run All Benchmarks

```bash
# Run all implemented benchmarks
python run_all_benchmarks.py --full

# Run specific benchmarks
python run_all_benchmarks.py --datasets fever scifact

# Include baseline comparisons
python run_all_benchmarks.py --full --baselines

# Generate report from cached results
python run_all_benchmarks.py --report-only
```

## 📊 Available Benchmarks

### 1. FEVER (Fact Extraction and VERification)
**Status:** ✅ Implemented  
**Dataset:** 185K claims, 3 classes (SUPPORTS, REFUTES, NOT_ENOUGH_INFO)  
**Metrics:** Accuracy, Precision, Recall, F1  
**Tests:** Factuality verification against evidence

```bash
python fever/test_fever.py
```

### 2. SciFact (Scientific Fact Verification)
**Status:** 🚧 In Progress  
**Dataset:** 1.4K scientific claims with evidence  
**Metrics:** Label accuracy, Evidence retrieval F1  
**Tests:** Domain-specific (scientific) fact checking

### 3. HotpotQA (Multi-Hop Reasoning)
**Status:** 🚧 In Progress  
**Dataset:** 113K multi-hop questions  
**Metrics:** Exact Match, F1, Answer accuracy  
**Tests:** 2+ hop reasoning over knowledge graph

### 4. MetaQA (Knowledge Graph QA)
**Status:** 🚧 In Progress  
**Dataset:** 400K questions over WikiMovies KG  
**Metrics:** Accuracy, Hits@1/5/10  
**Tests:** NL2Cypher query translation

### 5. Wikidata5M (Entity Linking)
**Status:** 🚧 In Progress  
**Dataset:** 5M entities, 21M triples  
**Metrics:** Precision@K, Recall@K, MRR  
**Tests:** Entity resolution and linking

### 6. DBpedia (KG Construction)
**Status:** 🚧 In Progress  
**Dataset:** Structured Wikipedia data  
**Metrics:** Triple precision/recall/F1  
**Tests:** Knowledge graph extraction quality

### 7. TrustKG (Trustworthiness - Custom)
**Status:** 🚧 In Progress  
**Dataset:** 400 synthetic test cases  
**Test Suites:**
- Controlled Hallucinations (100 cases)
- Temporal Contradictions (100 cases)
- Conflicting Evidence (100 cases)
- Missing Facts (100 cases)

**Metrics:** Detection rate, Precision, Recall, F1

## 📁 Directory Structure

```
tests/benchmarks/
├── __init__.py                 # Package initialization
├── config.py                   # Configuration settings
├── base_benchmark.py           # Abstract base class
├── metrics.py                  # Evaluation metrics
├── visualizations.py           # Plotting utilities
├── run_all_benchmarks.py       # Master test runner
├── requirements.txt            # Benchmark dependencies
│
├── data/                       # Downloaded datasets
│   ├── fever/
│   ├── scifact/
│   ├── hotpotqa/
│   ├── metaqa/
│   ├── wikidata5m/
│   ├── dbpedia/
│   └── trustkg/
│
├── reports/                    # Generated reports
│   ├── tables/                 # CSV/LaTeX tables
│   └── charts/                 # Visualizations
│
├── fever/                      # FEVER benchmark
│   └── test_fever.py
│
├── scifact/                    # SciFact benchmark
│   └── test_scifact.py
│
├── hotpotqa/                   # HotpotQA benchmark
│   └── test_hotpotqa.py
│
├── metaqa/                     # MetaQA benchmark
│   └── test_metaqa.py
│
├── wikidata5m/                 # Wikidata5M benchmark
│   └── test_wikidata.py
│
├── dbpedia/                    # DBpedia benchmark
│   └── test_dbpedia.py
│
├── trustkg/                    # TrustKG benchmark
│   ├── test_hallucinations.py
│   ├── test_temporal.py
│   ├── test_conflicts.py
│   ├── test_missing_facts.py
│   └── synthetic_data.json
│
└── baselines/                  # Baseline systems
    ├── pure_rag.py
    ├── pure_kg.py
    └── wikipedia_api.py
```

## 🧪 Running Individual Benchmarks

Each benchmark can be run independently:

```bash
# FEVER
cd fever && python test_fever.py

# SciFact
cd scifact && python test_scifact.py

# HotpotQA
cd hotpotqa && python test_hotpotqa.py
```

## 📈 Results & Reporting

### Output Files

After running benchmarks, you'll find:

**JSON Results:**
```
reports/benchmark_results_YYYYMMDD_HHMMSS.json
```

**Markdown Report:**
```
reports/report_YYYYMMDD_HHMMSS/report.md
```

**LaTeX Tables:**
```
reports/report_YYYYMMDD_HHMMSS/tables.tex
```

**Visualizations:**
```
reports/charts/
├── dataset_comparison.png
├── confusion_matrix.png
├── threshold_analysis.png
├── radar_comparison.png
└── system_heatmap.png
```

### Sample Output

```json
{
  "timestamp": "2025-12-07T12:00:00",
  "benchmarks": {
    "fever": {
      "num_samples": 1000,
      "num_errors": 5,
      "metrics": {
        "accuracy": 0.872,
        "precision": 0.865,
        "recall": 0.858,
        "f1": 0.861
      }
    }
  },
  "summary": {
    "num_benchmarks": 1,
    "total_samples": 1000,
    "total_errors": 5,
    "average_metrics": {
      "accuracy": 0.872,
      "f1": 0.861
    }
  }
}
```

## 🔬 Baseline Comparisons

Compare GraphBuilder against baseline systems:

- **Pure RAG:** FAISS semantic search only (no graph)
- **Pure KG:** Neo4j graph traversal only (no embeddings)
- **Wikipedia API:** Direct API queries (no KG construction)

```bash
python run_all_benchmarks.py --full --baselines
```

## 📊 Evaluation Metrics

### Classification Metrics
- **Accuracy:** Overall correctness
- **Precision:** True positives / (True positives + False positives)
- **Recall:** True positives / (True positives + False negatives)
- **F1 Score:** Harmonic mean of precision and recall

### QA Metrics
- **Exact Match (EM):** Answer exactly matches gold answer
- **F1:** Token-level overlap between prediction and gold answer

### Ranking Metrics
- **MRR (Mean Reciprocal Rank):** Average of 1/rank of first correct answer
- **Hits@K:** Proportion where correct answer is in top-K
- **Precision@K:** Relevant items in top-K / K
- **Recall@K:** Relevant items in top-K / total relevant

### Graph Metrics
- **Triple Accuracy:** Precision/recall/F1 of extracted triples
- **Entity Linking:** Precision/recall for entity resolution
- **Schema Alignment:** Predicate mapping accuracy

## 🎓 For Paper/Publication

The benchmark suite generates publication-ready outputs:

1. **LaTeX Tables:** Copy-paste into paper
2. **High-Resolution Charts:** 300 DPI PNG/PDF/SVG
3. **Statistical Analysis:** Significance tests, confidence intervals
4. **Error Analysis:** Detailed breakdown of failure modes

### Citation

If you use this benchmark suite, please cite:

```bibtex
@software{graphbuilder_benchmarks,
  title={GraphBuilder-RAG Benchmark Suite},
  author={Your Name},
  year={2025},
  url={https://github.com/your-repo}
}
```

## 🐛 Troubleshooting

### Dataset Download Issues
If datasets fail to download:
1. Check internet connection
2. Verify dataset URLs in `config.py`
3. Manually download and place in `data/{dataset}/`

### Memory Issues
For large datasets:
```bash
# Reduce sample size
python run_all_benchmarks.py --datasets fever --sample-size 100
```

### Timeout Errors
Increase timeout in `config.py`:
```python
EVAL_SETTINGS = {
    "timeout_seconds": 600  # 10 minutes
}
```

## 🤝 Contributing

To add a new benchmark:

1. Create directory: `tests/benchmarks/your_dataset/`
2. Implement `YourBenchmark(BaseBenchmark)` class
3. Add config to `config.py`
4. Register in `run_all_benchmarks.py`
5. Write tests and documentation

See `fever/test_fever.py` as a reference implementation.

## 📝 Development Status

- [x] Infrastructure (base classes, metrics, visualizations)
- [x] FEVER benchmark
- [ ] SciFact benchmark
- [ ] HotpotQA benchmark
- [ ] MetaQA benchmark
- [ ] Wikidata5M benchmark
- [ ] DBpedia benchmark
- [ ] TrustKG synthetic dataset
- [ ] Baseline implementations
- [ ] Automated report generation
- [ ] Statistical significance tests

## 📄 License

MIT License - See main project LICENSE file.
