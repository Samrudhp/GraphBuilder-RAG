# Final Benchmark Evaluation - Quick Start Guide

## 🎯 Goal
Run honest 25-sample evaluation for workshop paper submission (Dec 28, 2025)

## 📊 What You'll Get
- **5 test configurations** (125 samples total)
- **15 publication-quality plots**
- **All metrics** (Accuracy, Precision@k, Recall@k, MRR, hallucination rate, etc.)
- **Statistical significance tests** (McNemar, paired t-test, Cohen's d)
- **Honest results** for workshop paper

---

## 🚀 Quick Start (3 Commands)

### Option 1: Automated (Recommended for First Run)
```bash
cd tests/final_benchmark
./setup.sh                    # Check dependencies
python run_evaluations.py --test fever_full_system  # Test one config first
```

### Option 2: Full Pipeline (All 5 Tests)
```bash
cd tests/final_benchmark
./run_full_pipeline.sh        # Runs everything automatically (~30-40 min)
```

### Option 3: Manual Step-by-Step (Recommended for Daily Runs)
```bash
# Day 1 (Dec 19)
python run_evaluations.py --test fever_full_system

# Day 2 (Dec 20)
python run_evaluations.py --test fever_no_graphverify

# Day 3 (Dec 21)
python run_evaluations.py --test hotpotqa_vector_only

# Day 4 (Dec 22)
python run_evaluations.py --test hotpotqa_graph_only

# Day 5 (Dec 23)
python run_evaluations.py --test hotpotqa_hybrid

# Day 6 (Dec 24) - Analysis
python calculate_metrics.py
python generate_plots.py
```

---

## 📁 File Structure

```
tests/final_benchmark/
├── README.md                      # Overview
├── QUICKSTART.md                  # This file
├── ablation_configs.py            # Test configurations + sample data
├── run_evaluations.py             # Main evaluation runner
├── calculate_metrics.py           # Metrics calculator
├── generate_plots.py              # Plot generator
├── setup.sh                       # Dependency checker
├── run_full_pipeline.sh          # Automated pipeline
├── results/                       # Generated results
│   ├── fever_full_system.json
│   ├── fever_no_graphverify.json
│   ├── hotpotqa_vector_only.json
│   ├── hotpotqa_graph_only.json
│   ├── hotpotqa_hybrid.json
│   └── metrics_summary.json
└── plots/                         # Generated visualizations
    ├── accuracy_comparison.png
    ├── ablation_study.png
    ├── precision_at_k.png
    └── ... (15 total)
```

---

## 🧪 Test Configurations

### 1. FEVER Full System
- **Dataset**: FEVER (fact verification)
- **Config**: All components enabled
- **Purpose**: Baseline for ablation comparison
- **Command**: `python run_evaluations.py --test fever_full_system`

### 2. FEVER No GraphVerify
- **Dataset**: FEVER
- **Config**: GraphVerify disabled
- **Purpose**: Show impact of verification on hallucination
- **Command**: `python run_evaluations.py --test fever_no_graphverify`

### 3. HotpotQA Vector-only
- **Dataset**: HotpotQA (multi-hop QA)
- **Config**: Only FAISS vector retrieval (standard RAG baseline)
- **Purpose**: Compare against what everyone uses
- **Command**: `python run_evaluations.py --test hotpotqa_vector_only`

### 4. HotpotQA Graph-only
- **Dataset**: HotpotQA
- **Config**: Only Neo4j graph retrieval
- **Purpose**: Show graph alone isn't enough
- **Command**: `python run_evaluations.py --test hotpotqa_graph_only`

### 5. HotpotQA Hybrid (YOUR CONTRIBUTION)
- **Dataset**: HotpotQA
- **Config**: Vector + Graph + GraphVerify (full system)
- **Purpose**: Demonstrate hybrid superiority
- **Command**: `python run_evaluations.py --test hotpotqa_hybrid`

---

## 📈 Metrics Calculated

### Accuracy Metrics
- **Accuracy**: % correct answers
- **Precision@k**: Precision at k=1,3,5,10
- **Recall@k**: Recall at k=1,3,5,10
- **MRR**: Mean Reciprocal Rank

### Quality Metrics
- **Hallucination Rate**: % of hallucinated responses
- **Confidence Scores**: Distribution statistics (mean, median, std)
- **Latency**: Retrieval + generation time

### Statistical Tests
- **McNemar Test**: Paired comparison for accuracy
- **Paired T-test**: Continuous metric comparison
- **Cohen's d**: Effect size measurement

---

## 🎨 Plots Generated (15 Total)

### Core Comparisons (5 plots)
1. **Accuracy Comparison** - Bar chart showing hybrid > graph > vector
2. **Ablation Study** - Impact of GraphVerify on accuracy + hallucination
3. **Precision@k Curves** - Retrieval quality across ranking positions
4. **Hallucination Rate** - With vs without GraphVerify
5. **Latency Comparison** - Speed tradeoffs

### Distributions (4 plots)
6. **Confidence Distribution (Hybrid)** - Histogram of model confidence
7. **Confidence Distribution (Baseline)** - Same for vector RAG
8. **Retrieval Component Usage** - Pie chart of graph vs vector usage
9. **Query Complexity** - Performance vs multi-hop difficulty

### Analysis (6 plots)
10. **Precision-Recall Curves** - Classic PR curves
11. **Per-Sample Heatmap** - Green/red correctness matrix
12. **Retrieval Time Breakdown** - Pipeline component latency
13. **Error Analysis** - Pie chart of error categories
14. **MRR Comparison** - Ranking quality
15. **Statistical Significance** - P-values and effect sizes

---

## 🔧 Troubleshooting

### Rate Limit Errors
If you get 429 errors:
```bash
# Wait 24 hours for Groq limit reset
# OR reduce samples in ablation_configs.py (not recommended)
# OR spread tests across multiple days (recommended)
```

### Import Errors
```bash
cd tests/final_benchmark
python3 -c "import sys; sys.path.insert(0, '../../'); from services.query.service import QueryService"
```
If this fails, check your project structure.

### Missing Dependencies
```bash
pip3 install matplotlib seaborn numpy
```

### No Results Generated
Check logs in terminal output. Common issues:
- Database not running (Redis, Neo4j, MongoDB)
- Groq API key not set
- Services not initialized

---

## 📊 Expected Results (Ballpark)

Based on 25 samples, you should see:

### HotpotQA Multi-hop QA
- **Hybrid**: 60-75% accuracy
- **Graph-only**: 50-65% accuracy
- **Vector RAG**: 40-55% accuracy

### FEVER Fact Verification
- **Full System**: 65-80% accuracy, 10-15% hallucination rate
- **No GraphVerify**: 50-70% accuracy, 20-30% hallucination rate

### Statistical Significance
- **Hybrid vs Vector**: p < 0.05 (significant)
- **Full vs No GraphVerify**: p < 0.05 (significant)
- **Effect sizes**: Medium to large (Cohen's d > 0.5)

**Note**: These are estimates. Your actual results may vary. Report whatever you actually get!

---

## 📝 Using Results in Paper

### Results Section Template

```latex
\section{Experimental Evaluation}

\subsection{Experimental Setup}
We evaluated GraphBuilder-RAG on 25 samples each from FEVER 
(fact verification) and HotpotQA (multi-hop QA) benchmarks. 
Samples were randomly selected to represent varying difficulty levels.

\subsection{Baselines}
\begin{itemize}
\item \textbf{Vector RAG}: FAISS + Llama-3.3-70B (standard baseline)
\item \textbf{Graph-only}: Pure Neo4j retrieval
\item \textbf{Hybrid (Ours)}: Vector + Graph + GraphVerify
\end{itemize}

\subsection{Results}
Table~\ref{tab:accuracy} presents accuracy across methods. 
Our hybrid approach achieves XX\% accuracy on HotpotQA, 
outperforming vector-only RAG (XX\%) with statistical 
significance (p<0.05, McNemar test).

\begin{table}[h]
\centering
\caption{Accuracy Results (n=25 samples)}
\label{tab:accuracy}
\begin{tabular}{lccc}
\hline
Method & Accuracy & P@5 & MRR \\
\hline
Vector RAG & XX\% & XX\% & X.XX \\
Graph-only & XX\% & XX\% & X.XX \\
Hybrid (Ours) & XX\% & XX\% & X.XX \\
\hline
\end{tabular}
\end{table}

\subsection{Ablation Study}
Figure~\ref{fig:ablation} shows removing GraphVerify reduced 
FEVER accuracy from XX\% to XX\% and increased hallucination 
rate from XX\% to XX\%.

\subsection{Limitations}
This preliminary evaluation uses 25 samples per configuration. 
While statistical tests confirm significance, larger-scale 
evaluation is planned for future work.
```

### Figures to Include

**Essential figures for paper (5 minimum):**
1. `accuracy_comparison.png` - Main result
2. `ablation_study.png` - Proves GraphVerify value
3. `precision_at_k.png` - Shows retrieval quality
4. `hallucination_rate.png` - Key contribution
5. `statistical_significance.png` - Proves validity

**Nice-to-have figures (if space allows):**
6. `sample_accuracy_heatmap.png` - Visual appeal
7. `error_analysis.png` - Shows understanding
8. `latency_comparison.png` - Practical concerns

---

## ⏰ Timeline (Dec 19-28)

```
Dec 19 (Thu): Run test 1 - fever_full_system
Dec 20 (Fri): Run test 2 - fever_no_graphverify
Dec 21 (Sat): Run test 3 - hotpotqa_vector_only
Dec 22 (Sun): Run test 4 - hotpotqa_graph_only
Dec 23 (Mon): Run test 5 - hotpotqa_hybrid
Dec 24 (Tue): Calculate metrics + generate plots
Dec 25 (Wed): Write results section
Dec 26 (Thu): Write introduction + related work
Dec 27 (Fri): Finalize paper, proofread
Dec 28 (Sat): SUBMIT PAPER! 🎉
```

---

## ✅ Pre-Submission Checklist

Before submitting paper, verify:

- [ ] All 5 tests completed successfully
- [ ] `metrics_summary.json` generated
- [ ] All 15 plots generated in `plots/` directory
- [ ] Reviewed actual numbers (not estimates)
- [ ] Figures embedded in paper with captions
- [ ] Tables formatted correctly
- [ ] Sample size (n=25) clearly stated everywhere
- [ ] Limitations section acknowledges small sample
- [ ] Statistical significance reported honestly
- [ ] No fabricated numbers (use actual results!)
- [ ] Code/data availability statement included

---

## 🆘 Getting Help

### Check Logs
```bash
# Review evaluation logs
cat tests/final_benchmark/results/*.json | grep "error"

# Check metrics calculation
python calculate_metrics.py 2>&1 | tee metrics.log
```

### Debug Single Sample
```python
# Test with 1 sample first
from ablation_configs import get_config, get_test_samples
from run_evaluations import BenchmarkEvaluator
import asyncio

config = get_config("fever_full_system")
samples = get_test_samples("fever")[:1]  # Just first sample

async def test():
    evaluator = BenchmarkEvaluator(config)
    await evaluator.run_evaluation(samples)

asyncio.run(test())
```

### Contact
If stuck, review:
1. Terminal output for error messages
2. `results/*.json` files for data issues
3. Project README for service setup
4. Groq API dashboard for rate limit status

---

## 🎓 Academic Integrity Reminder

**This framework generates HONEST results for legitimate research.**

✅ DO:
- Report actual numbers from your 25 samples
- State sample size clearly (n=25)
- Show statistical significance tests
- Acknowledge limitations
- Submit preliminary findings to workshop

❌ DON'T:
- Claim 1000 samples when you only ran 25
- Report expected results instead of actual
- Hide sample size in footnotes
- Fabricate any numbers
- Claim state-of-the-art without proper evaluation

**One honest workshop paper builds your career. One fraudulent paper ends it.**

---

## 🎉 Good Luck!

You have everything you need to:
1. Run legitimate 25-sample evaluation ✅
2. Get honest results in 5 days ✅
3. Generate publication-quality figures ✅
4. Submit workshop paper by Dec 28 ✅

**Your system is interesting regardless of exact accuracy numbers. Focus on methodology!**

Now go run those tests! 🚀
