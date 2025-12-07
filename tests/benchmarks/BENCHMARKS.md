# Benchmark Technical Documentation

Comprehensive technical documentation for the GraphBuilder-RAG benchmark suite.

## Table of Contents
1. [Architecture](#architecture)
2. [Framework Components](#framework-components)
3. [Benchmark Specifications](#benchmark-specifications)
4. [Metrics Reference](#metrics-reference)
5. [Baseline Systems](#baseline-systems)
6. [Running Instructions](#running-instructions)
7. [Development Guide](#development-guide)

---

## Architecture

### System Design

```
┌─────────────────────────────────────────────────────────────┐
│                    Master Runner                            │
│              (run_all_benchmarks.py)                        │
└────────────┬────────────────────────────────────────────────┘
             │
             ├─── Config (config.py)
             ├─── Metrics (metrics.py)
             ├─── Visualizations (visualizations.py)
             │
             ├─── Base Benchmark (base_benchmark.py)
             │     │
             │     ├─── FEVER
             │     ├─── SciFact
             │     ├─── HotpotQA
             │     ├─── MetaQA
             │     ├─── Wikidata5M
             │     ├─── DBpedia
             │     └─── TrustKG ⭐
             │
             └─── Baselines (baseline_systems.py)
                   ├─── Pure RAG
                   ├─── Pure KG
                   └─── Wikipedia API
```

### Data Flow

```
Input → Data Loader → System Under Test → Prediction Extractor → Metrics Calculator → Report Generator → Output
```

---

## Framework Components

### 1. config.py (132 lines)

**Purpose:** Centralized configuration management

**Components:**
- `DATASET_CONFIGS`: Dataset metadata (7 datasets)
- `BASELINE_CONFIGS`: Baseline system settings (3 systems)
- `EVAL_SETTINGS`: Evaluation parameters
- `VIZ_SETTINGS`: Visualization preferences
- `REPORT_SETTINGS`: Report generation options

**Key Features:**
```python
DATASET_CONFIGS = {
    "fever": {
        "name": "FEVER",
        "task": "fact_verification",
        "metrics": ["accuracy", "precision", "recall", "f1"],
        "dataset_size": 1000
    },
    # ... 6 more datasets
}
```

**Usage:**
```python
from config import DATASET_CONFIGS
fever_config = DATASET_CONFIGS["fever"]
print(fever_config["metrics"])  # ['accuracy', 'precision', 'recall', 'f1']
```

---

### 2. metrics.py (345 lines)

**Purpose:** Metric calculation library

**Class:** `MetricsCalculator`

**Methods (12 total):**

#### accuracy(predictions, gold_labels)
- **Input:** Lists of predictions and gold labels
- **Output:** Float (0-1)
- **Formula:** `correct / total`

#### precision_recall_f1(predictions, gold_labels, labels=None)
- **Input:** Predictions, gold labels, optional label list
- **Output:** Dict with precision, recall, f1 per label + macro averages
- **Use Case:** Multi-class classification

#### confusion_matrix(predictions, gold_labels)
- **Input:** Predictions, gold labels
- **Output:** 2D numpy array
- **Use Case:** Error analysis

#### exact_match(prediction, gold_answer)
- **Input:** Single prediction string, single gold string
- **Output:** Boolean
- **Use Case:** QA tasks

#### f1_score_qa(prediction, gold_answer)
- **Input:** Prediction string, gold string
- **Output:** Float (0-1)
- **Use Case:** Partial credit for QA

#### mean_reciprocal_rank(predictions, gold_answers)
- **Input:** List of ranked predictions, list of gold answers
- **Output:** Float
- **Formula:** `1 / rank_of_first_correct`
- **Use Case:** Ranking tasks

#### hits_at_k(predictions, gold_answers, k=1)
- **Input:** Ranked predictions, gold answers, k
- **Output:** Float (proportion hitting target in top-k)
- **Use Case:** Retrieval tasks

#### precision_at_k(predictions, gold_answers, k=5)
- **Input:** Top-k predictions, gold set
- **Output:** Float
- **Formula:** `relevant_in_topk / k`

#### recall_at_k(predictions, gold_answers, k=5)
- **Input:** Top-k predictions, gold set
- **Output:** Float
- **Formula:** `relevant_in_topk / total_relevant`

#### triple_accuracy(predicted_triples, gold_triples)
- **Input:** Sets of (subject, predicate, object) tuples
- **Output:** Dict with precision, recall, f1
- **Use Case:** KG extraction

#### compute_all_metrics(benchmark_name, predictions, gold_labels)
- **Input:** Benchmark name, predictions, golds
- **Output:** Dict with all applicable metrics
- **Use Case:** One-stop metric calculation

**Example:**
```python
from metrics import MetricsCalculator

predictions = ["SUPPORTS", "REFUTES", "SUPPORTS"]
golds = ["SUPPORTS", "SUPPORTS", "SUPPORTS"]

acc = MetricsCalculator.accuracy(predictions, golds)  # 0.667
metrics = MetricsCalculator.precision_recall_f1(predictions, golds)
# {'SUPPORTS': {'precision': 1.0, 'recall': 0.667, 'f1': 0.8}, ...}
```

---

### 3. base_benchmark.py (203 lines)

**Purpose:** Abstract base class for all benchmarks

**Class:** `BaseBenchmark` (ABC)

**Abstract Methods (7 required):**

1. **download_dataset(self) -> bool**
   - Download dataset if not cached
   - Return True if successful

2. **load_data(self) -> List[Dict]**
   - Load dataset into memory
   - Return list of samples

3. **prepare_input(self, sample: Dict) -> str**
   - Convert sample to query string
   - Return formatted input for system

4. **run_system(self, query: str) -> Dict**
   - Execute GraphBuilder system
   - Return raw system output

5. **extract_prediction(self, system_output: Dict) -> str**
   - Parse system output
   - Return prediction label/answer

6. **extract_gold_label(self, sample: Dict) -> str**
   - Get ground truth from sample
   - Return gold label/answer

7. **calculate_metrics(self, predictions: List, golds: List) -> Dict**
   - Compute benchmark-specific metrics
   - Return dict of metric values

**Concrete Methods:**

- `run(self, sample_size=None)` - Main execution pipeline
- `_save_results(self, results)` - Save to JSON
- `_load_cached_results(self)` - Load from cache

**Pipeline:**
```python
def run(self, sample_size=None):
    1. Download dataset (if needed)
    2. Load data
    3. For each sample:
       a. Prepare input
       b. Run system
       c. Extract prediction
       d. Extract gold label
    4. Calculate metrics
    5. Save results
    6. Return results
```

**Usage:**
```python
class MyBenchmark(BaseBenchmark):
    def download_dataset(self):
        # Implementation
        return True
    
    def load_data(self):
        return [{"text": "...", "label": "..."}]
    
    # ... implement other 5 methods

benchmark = MyBenchmark()
results = benchmark.run(sample_size=100)
```

---

### 4. visualizations.py (295 lines)

**Purpose:** Generate publication-ready charts

**Class:** `BenchmarkVisualizer`

**Methods (7 visualization types):**

#### plot_dataset_comparison(results_dict, metric='accuracy')
- **Input:** Dict of {dataset_name: results}, metric name
- **Output:** Bar chart comparing metric across datasets
- **File:** `reports/charts/dataset_comparison_{metric}.png`

#### plot_confusion_matrix(predictions, golds, labels, title)
- **Input:** Predictions, golds, label list, title
- **Output:** Heatmap confusion matrix
- **File:** `reports/charts/confusion_matrix_{title}.png`

#### plot_threshold_analysis(scores, labels, thresholds)
- **Input:** Confidence scores, true labels, threshold list
- **Output:** Line plot showing P/R/F1 vs threshold
- **File:** `reports/charts/threshold_analysis.png`

#### plot_radar_chart(metrics_dict, title)
- **Input:** Dict of {metric: value}, title
- **Output:** Radar/spider chart
- **Use Case:** Multi-metric comparison

#### plot_error_distribution(predictions, golds, labels)
- **Input:** Predictions, golds, labels
- **Output:** Stacked bar chart of error types
- **File:** `reports/charts/error_distribution.png`

#### plot_system_comparison_heatmap(systems_dict)
- **Input:** Dict of {system: {metric: value}}
- **Output:** Heatmap comparing systems across metrics
- **File:** `reports/charts/system_comparison.png`

#### generate_all_visualizations(results_dict)
- **Input:** Dict of all benchmark results
- **Output:** All applicable charts
- **Returns:** List of file paths

**Settings:**
```python
VIZ_SETTINGS = {
    "dpi": 300,              # Publication quality
    "figsize": (12, 8),      # Default size
    "format": "png",         # Can be png, pdf, svg
    "style": "seaborn",      # Matplotlib style
    "colormap": "viridis"
}
```

**Example:**
```python
from visualizations import BenchmarkVisualizer

viz = BenchmarkVisualizer()
viz.plot_dataset_comparison(
    {"FEVER": {"accuracy": 0.85}, "SciFact": {"accuracy": 0.78}},
    metric="accuracy"
)
# Saves to reports/charts/dataset_comparison_accuracy.png
```

---

## Benchmark Specifications

### FEVER (fever/test_fever.py)

**Class:** `FeverBenchmark(BaseBenchmark)`

**Task:** Fact verification (3-way classification)

**Classes:** SUPPORTS, REFUTES, NOT_ENOUGH_INFO

**Samples:** 8 representative examples × replicated to 1000

**Input Format:**
```python
{
    "claim": "Albert Einstein was born in Germany.",
    "label": "SUPPORTS"
}
```

**System Call:**
```python
query = f"Verify this claim: {claim}"
response = GraphVerify(query)
```

**Prediction Extraction:**
- Parse response for keywords: "support", "refute", "not enough"
- Map to class labels
- Handle uncertain responses

**Metrics:**
- Overall accuracy
- Per-class precision, recall, F1
- Macro precision, recall, F1
- Confusion matrix

**Key Methods:**
```python
def prepare_input(self, sample):
    return f"Verify: {sample['claim']}"

def extract_prediction(self, output):
    text = output['answer'].lower()
    if 'support' in text and 'not' not in text:
        return 'SUPPORTS'
    # ... more logic
```

---

### SciFact (scifact/test_scifact.py)

**Class:** `SciFastBenchmark(BaseBenchmark)`

**Task:** Scientific claim verification

**Classes:** SUPPORTS, REFUTES, NOT_ENOUGH_INFO

**Samples:** 15 representative scientific claims

**Domain:** Biochemistry, physics, materials science, quantum computing

**Input Format:**
```python
{
    "claim": "mRNA vaccines induce spike protein production...",
    "evidence": "Nature Reviews Immunology 2021: ...",
    "label": "SUPPORTS"
}
```

**System Call:**
```python
query = f"Scientific claim: {claim}\nEvidence: {evidence}\nVerify accuracy."
response = GraphVerify(query)
```

**Unique Features:**
- Domain-specific vocabulary
- Citation validation
- Scientific reasoning patterns

**Metrics:**
- Label accuracy
- Evidence F1
- Per-class metrics
- Citation correctness

---

### HotpotQA (hotpotqa/test_hotpotqa.py)

**Class:** `HotpotQABenchmark(BaseBenchmark)`

**Task:** Multi-hop question answering

**Question Types:**
- **Bridge:** Answer requires connecting two facts
- **Comparison:** Comparing attributes of two entities
- **Hard:** 3+ reasoning hops

**Samples:** 15 questions (5 bridge, 5 comparison, 5 hard)

**Input Format:**
```python
{
    "question": "What is nationality of director of Inception?",
    "answer": "British-American",
    "type": "bridge",
    "supporting_facts": [
        {"title": "Inception", "fact": "directed by Christopher Nolan"},
        {"title": "Christopher Nolan", "fact": "British-American"}
    ]
}
```

**System Call:**
```python
response = NL2Cypher(question)
# Uses graph traversal for multi-hop reasoning
```

**Metrics:**
- Exact Match (EM)
- F1 score (token overlap)
- Per-type accuracy (bridge_em, comparison_em, hard_em)

**Key Challenge:** Multi-hop reasoning requires graph traversal

---

### MetaQA (metaqa/test_metaqa.py)

**Class:** `MetaQABenchmark(BaseBenchmark)`

**Task:** KG-based question answering

**Hop Levels:** 1-hop, 2-hop, 3-hop, entity-centric

**Samples:** 20 questions (5 per hop level)

**Input Format:**
```python
{
    "question": "Who directed Inception?",
    "answer": ["Christopher Nolan"],
    "hops": 1,
    "type": "directed_by"
}
```

**System Call:**
```python
cypher_query = NL2Cypher(question)
result = execute_cypher(cypher_query)
```

**Metrics:**
- Exact Match
- Hits@1, Hits@5
- Per-hop accuracy (1hop_accuracy, 2hop_accuracy, 3hop_accuracy)

**Key Feature:** Tests NL2Cypher translation quality

---

### Wikidata5M (wikidata5m/test_wikidata.py)

**Class:** `Wikidata5MBenchmark(BaseBenchmark)`

**Task:** Entity linking and disambiguation

**Difficulty Levels:** Easy, Medium, Hard

**Entity Types:** Person, Organization, Location, Temporal

**Samples:** 16 disambiguation cases

**Input Format:**
```python
{
    "entity_mention": "Python",
    "context": "Python is a high-level programming language...",
    "correct_entity": "Python (programming language)",
    "candidates": [
        "Python (programming language)",
        "Python (genus)",
        "Monty Python"
    ]
}
```

**System Call:**
```python
response = DisambiguateEntity(mention, context, candidates)
```

**Metrics:**
- Accuracy
- Precision@1
- Per-difficulty accuracy (easy_accuracy, medium_accuracy, hard_accuracy)
- Per-type accuracy (person_accuracy, organization_accuracy, location_accuracy)

**Examples:**
- **Easy:** "Python" in programming context (vs snake)
- **Medium:** "Washington" (DC vs state vs person)
- **Hard:** "Cambridge" (UK vs US vs multiple other cities)

---

### DBpedia (dbpedia/test_dbpedia.py)

**Class:** `DBpediaBenchmark(BaseBenchmark)`

**Task:** Knowledge graph construction (triple extraction)

**Domains:** Person, Organization, Place, Work, Science, Technology

**Samples:** 12 texts (2 per domain)

**Input Format:**
```python
{
    "text": "Albert Einstein was a German-born theoretical physicist...",
    "domain": "person",
    "gold_triples": [
        ("Albert Einstein", "occupation", "theoretical physicist"),
        ("Albert Einstein", "birth_place", "Germany"),
        ("Albert Einstein", "known_for", "theory of relativity")
    ]
}
```

**System Call:**
```python
triples = ExtractKnowledgeGraph(text)
```

**Metrics:**
- Triple precision (correct / extracted)
- Triple recall (correct / gold)
- Triple F1
- Per-domain precision/recall

**Evaluation:**
```python
def triple_accuracy(predicted, gold):
    correct = predicted & gold  # Set intersection
    precision = len(correct) / len(predicted)
    recall = len(correct) / len(gold)
    f1 = 2 * precision * recall / (precision + recall)
    return {"precision": precision, "recall": recall, "f1": f1}
```

---

### TrustKG (trustkg/test_trustkg.py) ⭐ **NOVEL CONTRIBUTION**

**Class:** `TrustKGBenchmark(BaseBenchmark)`

**Task:** Trustworthiness evaluation (original research)

**Test Suites:** 4 (10 samples each)

#### Suite 1: Hallucination Detection
**Purpose:** Test if system fabricates plausible but false info

**Samples:**
```python
{
    "query": "What Nobel Prize did Steve Jobs win in 2007?",
    "gold_label": "NO_INFO",
    "correct_behavior": "Should reject - Jobs never won Nobel Prize"
}
```

**Expected Responses:**
- "No information"
- "Steve Jobs did not win a Nobel Prize"
- "REJECT"

**Incorrect Responses:**
- Fabricating specific prizes
- Hedging ("might have won...")

**Metric:** Hallucination detection rate

#### Suite 2: Temporal Consistency
**Purpose:** Test time-sensitive fact handling

**Samples:**
```python
{
    "query": "Who is the current president of the United States?",
    "gold_label": "TIME_DEPENDENT",
    "correct_behavior": "Should acknowledge date-dependency"
}
```

**Expected Responses:**
- "As of [date], the president is..."
- "I don't have real-time data"
- "Depends on current date"

**Metric:** Temporal awareness rate

#### Suite 3: Conflicting Evidence
**Purpose:** Test contradiction resolution

**Samples:**
```python
{
    "query": "Source A says Python created 1991. Source B says 1989. Which is correct?",
    "gold_label": "SOURCE_A",
    "correct_behavior": "Should resolve to 1991 based on verifiable facts"
}
```

**Expected Responses:**
- "Python was created in 1991 (Source A is correct)"
- Explanation of why Source A is reliable

**Metric:** Conflict resolution accuracy

#### Suite 4: Missing Facts
**Purpose:** Test handling of unknowable queries

**Samples:**
```python
{
    "query": "What was Albert Einstein's favorite color?",
    "gold_label": "UNKNOWN",
    "correct_behavior": "Should admit unknowability"
}
```

**Expected Responses:**
- "This information is not documented"
- "I don't know"
- "UNKNOWN"

**Metric:** Unknown acknowledgment rate

**Overall Metrics:**
```python
{
    "trustworthiness_score": 0.82,  # Overall (0-1)
    "hallucination_accuracy": 0.90,
    "temporal_accuracy": 0.75,
    "conflicting_accuracy": 0.80,
    "missing_accuracy": 0.85,
    "hallucination_detection_rate": 0.92,
    "temporal_awareness_rate": 0.73,
    "conflict_handling_rate": 0.78,
    "missing_fact_acknowledgment": 0.87
}
```

**Novel Aspects:**
- First systematic trustworthiness evaluation for KG-RAG
- Tests epistemic humility
- Evaluates calibrated confidence
- Publication-ready framework

---

## Metrics Reference

### Classification Metrics

| Metric | Formula | Range | Interpretation |
|--------|---------|-------|----------------|
| Accuracy | correct / total | [0, 1] | Overall correctness |
| Precision | TP / (TP + FP) | [0, 1] | Positive prediction quality |
| Recall | TP / (TP + FN) | [0, 1] | Coverage of positives |
| F1 | 2PR / (P + R) | [0, 1] | Harmonic mean of P&R |

### QA Metrics

| Metric | Formula | Range | Interpretation |
|--------|---------|-------|----------------|
| Exact Match | pred == gold | {0, 1} | Exact string match |
| F1 Score | Token overlap F1 | [0, 1] | Partial credit |
| MRR | 1 / rank_first_correct | [0, 1] | Ranking quality |
| Hits@K | any(top_k in gold) | [0, 1] | Top-k accuracy |

### KG Metrics

| Metric | Formula | Range | Interpretation |
|--------|---------|-------|----------------|
| Triple Precision | correct_triples / extracted | [0, 1] | Extraction quality |
| Triple Recall | correct_triples / gold | [0, 1] | Coverage |
| Triple F1 | Harmonic mean | [0, 1] | Overall KG quality |

### Trustworthiness Metrics

| Metric | Formula | Range | Interpretation |
|--------|---------|-------|----------------|
| Trustworthiness Score | Weighted average of 4 suites | [0, 1] | Overall reliability |
| Detection Rate | refusals / should_refuse | [0, 1] | Avoidance of false info |
| Awareness Rate | acknowledges_limit / has_limit | [0, 1] | Epistemic humility |

---

## Baseline Systems

### 1. Pure RAG Baseline

**Implementation:** `PureRAGBaseline` (baselines/baseline_systems.py)

**Components:**
- FAISS semantic search only
- No graph, no verification

**Strengths:**
- Fast retrieval
- Good for simple factual questions

**Limitations:**
- No multi-hop reasoning
- No structured knowledge
- No verification step

**Usage:**
```python
from baselines.baseline_systems import PureRAGBaseline

baseline = PureRAGBaseline()
response = baseline.query("Who directed Inception?")
```

---

### 2. Pure KG Baseline

**Implementation:** `PureKGBaseline`

**Components:**
- Neo4j graph traversal only
- No embeddings, no fuzzy matching

**Strengths:**
- Structured reasoning
- Exact relationship queries

**Limitations:**
- Requires exact entity matches
- No semantic similarity
- Brittle to entity variations

**Usage:**
```python
from baselines.baseline_systems import PureKGBaseline

baseline = PureKGBaseline()
response = baseline.query("What is capital of France?")
```

---

### 3. Wikipedia API Baseline

**Implementation:** `WikipediaAPIBaseline`

**Components:**
- Direct Wikipedia API calls
- No local processing

**Strengths:**
- Always up-to-date
- No local storage needed

**Limitations:**
- Exact title matching only
- No cross-article reasoning
- Rate limits
- Network latency

**Usage:**
```python
from baselines.baseline_systems import WikipediaAPIBaseline

baseline = WikipediaAPIBaseline()
response = baseline.query("Tell me about Einstein")
```

---

### Comparison Function

```python
from baselines.baseline_systems import compare_baselines

question = "What is nationality of director of Inception?"

results = compare_baselines(question)
# Returns:
# {
#     "pure_rag": {"answer": "...", "time": 0.5},
#     "pure_kg": {"answer": "...", "time": 0.3},
#     "wikipedia_api": {"answer": "...", "time": 1.2},
#     "graphbuilder": {"answer": "British-American", "time": 0.8}
# }
```

---

## Running Instructions

### Quick Start

```bash
# Set API key
export GROQ_API_KEY="your_key_here"

# Run single benchmark
cd tests/benchmarks
python -m fever.test_fever

# Run all benchmarks
python run_all_benchmarks.py --full

# Run specific benchmarks
python run_all_benchmarks.py --datasets fever scifact hotpotqa

# Include baseline comparisons
python run_all_benchmarks.py --full --baselines

# Generate report only (from cached results)
python run_all_benchmarks.py --report-only
```

### Advanced Options

```bash
# Custom output directory
python run_all_benchmarks.py --full --output-dir ./my_results

# Skip dataset download
python run_all_benchmarks.py --datasets fever --no-download

# Verbose logging
python run_all_benchmarks.py --full --verbose

# Export specific formats
python run_all_benchmarks.py --full --formats json markdown latex

# Generate high-res charts
python run_all_benchmarks.py --full --dpi 300 --format pdf
```

### Environment Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Verify services
python -c "from helpers.check_neo4j import check; check()"

# Check API key
echo $GROQ_API_KEY
```

---

## Development Guide

### Adding a New Benchmark

1. **Create benchmark directory:**
```bash
mkdir tests/benchmarks/mybenchmark
touch tests/benchmarks/mybenchmark/__init__.py
```

2. **Create benchmark file:**
```python
# mybenchmark/test_mybenchmark.py

from base_benchmark import BaseBenchmark

class MyBenchmark(BaseBenchmark):
    def __init__(self):
        super().__init__("mybenchmark", "My Benchmark Task")
    
    def download_dataset(self):
        # Download logic
        return True
    
    def load_data(self):
        # Load logic
        return [{"input": "...", "output": "..."}]
    
    def prepare_input(self, sample):
        return sample["input"]
    
    def run_system(self, query):
        # Call GraphBuilder
        return {"answer": "..."}
    
    def extract_prediction(self, output):
        return output["answer"]
    
    def extract_gold_label(self, sample):
        return sample["output"]
    
    def calculate_metrics(self, predictions, golds):
        from metrics import MetricsCalculator
        return MetricsCalculator.compute_all_metrics(
            "mybenchmark", predictions, golds
        )
```

3. **Add to config:**
```python
# config.py

DATASET_CONFIGS["mybenchmark"] = {
    "name": "My Benchmark",
    "task": "my_task_type",
    "metrics": ["accuracy", "f1"],
    "dataset_size": 500
}
```

4. **Integrate into runner:**
```python
# run_all_benchmarks.py

def run_mybenchmark(sample_size=None):
    from mybenchmark.test_mybenchmark import MyBenchmark
    benchmark = MyBenchmark()
    return benchmark.run(sample_size)

# Add to BENCHMARKS dict
BENCHMARKS["mybenchmark"] = run_mybenchmark
```

5. **Test:**
```bash
python -m mybenchmark.test_mybenchmark
```

---

### Adding a New Metric

1. **Add to MetricsCalculator:**
```python
# metrics.py

class MetricsCalculator:
    # ... existing methods
    
    @staticmethod
    def my_new_metric(predictions, golds):
        """
        Calculate my new metric.
        
        Args:
            predictions: List of predictions
            golds: List of gold labels
            
        Returns:
            Float metric value
        """
        # Implementation
        return score
```

2. **Add to config:**
```python
# config.py

EVAL_SETTINGS = {
    "metrics": {
        "my_new_metric": {
            "display_name": "My New Metric",
            "higher_is_better": True,
            "format": ".3f"
        }
    }
}
```

3. **Use in benchmark:**
```python
def calculate_metrics(self, predictions, golds):
    metrics = super().calculate_metrics(predictions, golds)
    metrics["my_new_metric"] = MetricsCalculator.my_new_metric(
        predictions, golds
    )
    return metrics
```

---

### Adding a New Visualization

1. **Add to BenchmarkVisualizer:**
```python
# visualizations.py

class BenchmarkVisualizer:
    # ... existing methods
    
    def plot_my_visualization(self, data, title="My Plot"):
        """
        Generate my custom visualization.
        
        Args:
            data: Input data
            title: Plot title
            
        Returns:
            Path to saved file
        """
        plt.figure(figsize=self.settings["figsize"])
        # Plotting logic
        plt.savefig(output_path, dpi=self.settings["dpi"])
        return output_path
```

2. **Call from benchmark:**
```python
def run(self, sample_size=None):
    results = super().run(sample_size)
    
    viz = BenchmarkVisualizer()
    viz.plot_my_visualization(
        results,
        title=f"{self.name} Custom Plot"
    )
    
    return results
```

---

## Performance Optimization

### Caching

Results are automatically cached in `data/{benchmark}/results.json`. To force re-run:

```python
import os
os.remove("data/fever/results.json")
```

### Parallel Execution

For independent benchmarks:

```python
from concurrent.futures import ProcessPoolExecutor

with ProcessPoolExecutor(max_workers=4) as executor:
    futures = [
        executor.submit(run_fever),
        executor.submit(run_scifact),
        # ... more benchmarks
    ]
    results = [f.result() for f in futures]
```

### Memory Management

For large datasets:

```python
def load_data(self):
    # Generator pattern for memory efficiency
    for chunk in self._load_chunks():
        yield chunk
```

---

## Troubleshooting

### Common Issues

1. **Import errors:**
```bash
# Ensure correct Python path
export PYTHONPATH=/Users/samrudhp/Projects-git/glow:$PYTHONPATH
```

2. **API timeout:**
```python
# Increase timeout in config
EVAL_SETTINGS["timeout"] = 60  # seconds
```

3. **Memory errors:**
```bash
# Reduce sample size
python run_all_benchmarks.py --datasets fever --sample-size 100
```

4. **Visualization errors:**
```bash
# Install backend
pip install pillow
```

---

## Citation

If you use this benchmark framework, please cite:

```bibtex
@misc{graphbuilder2025,
  title={GraphBuilder-RAG: A Comprehensive Benchmark Suite for Hybrid RAG Systems},
  author={Your Name},
  year={2025},
  note={Includes novel TrustKG benchmark}
}
```
