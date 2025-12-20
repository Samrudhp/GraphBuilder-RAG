# Final Benchmark Evaluation Suite

**Purpose**: Complete evaluation for workshop paper submission (Dec 28, 2025)

## Overview

5-day evaluation plan with 25 samples per configuration:
- **Day 1**: FEVER full system
- **Day 2**: FEVER no GraphVerify (ablation)
- **Day 3**: HotpotQA vector-only RAG (baseline)
- **Day 4**: HotpotQA graph-only
- **Day 5**: HotpotQA hybrid (full system)

**Total**: 125 samples, ~125K tokens on Groq (well under daily limit)

## Scripts

### 1. `run_evaluations.py`
Main orchestrator - runs all 5 tests in sequence with proper delays.

```bash
python run_evaluations.py
```

### 2. `calculate_metrics.py`
Calculates all metrics from evaluation results:
- Accuracy, Precision@k, Recall@k, MRR
- Hallucination rate, confidence distributions
- Statistical significance tests

```bash
python calculate_metrics.py
```

### 3. `generate_plots.py`
Creates all 15 publication-quality visualizations.

```bash
python generate_plots.py
```

### 4. `ablation_configs.py`
Configuration definitions for ablation studies.

## Metrics Calculated

1. **Accuracy**: % correct answers
2. **Precision@k**: Precision at k=1,3,5,10
3. **Recall@k**: Recall at k=1,3,5,10
4. **MRR**: Mean Reciprocal Rank
5. **Hallucination Rate**: % incorrect triples
6. **Confidence Scores**: Distribution statistics
7. **Latency**: Retrieval + generation time
8. **Statistical Tests**: McNemar, paired t-test, Cohen's d

## Visualizations Generated

1. Accuracy comparison bar chart
2. Ablation study results
3. Precision@k curves
4. Hallucination rate comparison
5. Latency comparison
6. Confidence score distributions (hybrid)
7. Confidence score distributions (baseline)
8. Retrieval component usage
9. Query complexity vs performance
10. Precision-recall curves
11. Per-sample accuracy heatmap
12. Retrieval time breakdown
13. Error analysis by category
14. MRR comparison
15. Statistical significance markers

## Output Structure

```
tests/final_benchmark/
├── results/
│   ├── fever_full_system.json
│   ├── fever_no_graphverify.json
│   ├── hotpotqa_vector_only.json
│   ├── hotpotqa_graph_only.json
│   ├── hotpotqa_hybrid.json
│   └── metrics_summary.json
├── plots/
│   ├── accuracy_comparison.png
│   ├── ablation_study.png
│   ├── precision_at_k.png
│   └── ... (15 total)
└── paper_tables/
    ├── table1_accuracy.tex
    ├── table2_ablation.tex
    └── table3_metrics.tex
```

## Timeline

- **Dec 19**: Day 1 - FEVER full system
- **Dec 20**: Day 2 - FEVER ablation
- **Dec 21**: Day 3 - HotpotQA vector baseline
- **Dec 22**: Day 4 - HotpotQA graph-only
- **Dec 23**: Day 5 - HotpotQA hybrid
- **Dec 24**: Calculate metrics & generate plots
- **Dec 25-27**: Write paper
- **Dec 28**: Submit

## API Rate Limits

Each test uses ~25K tokens:
- Query input: ~500-800 tokens
- Answer generation: ~200-400 tokens
- GraphVerify: ~300-500 tokens
- **Total per query**: ~1K tokens
- **25 queries**: ~25K tokens ✅ (well under Groq's 100K/day limit)

## Usage

### Run all evaluations (5 days)
```bash
python run_evaluations.py --all
```

### Run single test
```bash
python run_evaluations.py --test fever_full
```

### Calculate metrics only
```bash
python calculate_metrics.py
```

### Generate plots only
```bash
python generate_plots.py
```

### Generate LaTeX tables for paper
```bash
python generate_plots.py --latex-only
```
