"""
Calculate Evaluation Metrics

Computes all metrics from evaluation results:
- Accuracy, Precision@k, Recall@k, MRR
- Hallucination rate, confidence distributions
- Statistical significance tests (McNemar, paired t-test, Cohen's d)
- Latency analysis
"""
import json
import sys
from pathlib import Path
from typing import List, Dict, Tuple
import statistics
from collections import defaultdict
import argparse

# Add project root for any shared utils
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


class MetricsCalculator:
    """Calculate comprehensive evaluation metrics."""
    
    def __init__(self, results_dir: Path):
        self.results_dir = results_dir
        self.results = {}
        self.load_all_results()
    
    def load_all_results(self):
        """Load all result files from directory."""
        print(f"Loading results from {self.results_dir}")
        
        for result_file in self.results_dir.glob("*.json"):
            if result_file.stem == "metrics_summary":
                continue
            
            with open(result_file) as f:
                data = json.load(f)
                self.results[result_file.stem] = data
        
        print(f"Loaded {len(self.results)} result files")
    
    def calculate_accuracy(self, results: List[Dict]) -> float:
        """Calculate classification accuracy."""
        if not results:
            return 0.0
        correct = sum(1 for r in results if r["is_correct"])
        return correct / len(results)
    
    def calculate_precision_at_k(
        self,
        results: List[Dict],
        k: int = 5
    ) -> float:
        """Calculate Precision@k for retrieval."""
        if not results:
            return 0.0
        
        precisions = []
        for r in results:
            retrieved = r["total_sources_count"]
            relevant = r["relevant_sources_count"]
            
            if retrieved == 0:
                continue
            
            # Precision@k = relevant in top-k / k
            precision = min(relevant, k) / k
            precisions.append(precision)
        
        return statistics.mean(precisions) if precisions else 0.0
    
    def calculate_recall_at_k(
        self,
        results: List[Dict],
        k: int = 5
    ) -> float:
        """Calculate Recall@k for retrieval."""
        if not results:
            return 0.0
        
        recalls = []
        for r in results:
            relevant = r["relevant_sources_count"]
            
            if relevant == 0:
                continue
            
            # Simplified: assume we want to retrieve all relevant items
            # Recall@k = min(retrieved_relevant, k) / total_relevant
            retrieved_relevant = min(relevant, k)
            recall = retrieved_relevant / relevant if relevant > 0 else 0.0
            recalls.append(recall)
        
        return statistics.mean(recalls) if recalls else 0.0
    
    def calculate_mrr(self, results: List[Dict]) -> float:
        """Calculate Mean Reciprocal Rank."""
        if not results:
            return 0.0
        
        reciprocal_ranks = []
        for r in results:
            if r["is_correct"]:
                # If correct, assume rank 1 for simplicity
                reciprocal_ranks.append(1.0)
            else:
                # If incorrect, no relevant item in top results
                reciprocal_ranks.append(0.0)
        
        return statistics.mean(reciprocal_ranks) if reciprocal_ranks else 0.0
    
    def calculate_hallucination_rate(self, results: List[Dict]) -> float:
        """Calculate hallucination detection rate."""
        if not results:
            return 0.0
        hallucinated = sum(1 for r in results if r["hallucination_detected"])
        return hallucinated / len(results)
    
    def calculate_confidence_stats(self, results: List[Dict]) -> Dict:
        """Calculate confidence score statistics."""
        if not results:
            return {}
        
        scores = [r["confidence_score"] for r in results if r["confidence_score"] > 0]
        
        if not scores:
            return {
                "mean": 0.0,
                "median": 0.0,
                "std": 0.0,
                "min": 0.0,
                "max": 0.0,
            }
        
        return {
            "mean": statistics.mean(scores),
            "median": statistics.median(scores),
            "std": statistics.stdev(scores) if len(scores) > 1 else 0.0,
            "min": min(scores),
            "max": max(scores),
        }
    
    def calculate_latency_stats(self, results: List[Dict]) -> Dict:
        """Calculate latency statistics."""
        if not results:
            return {}
        
        retrieval_times = [r["retrieval_time"] for r in results]
        generation_times = [r["generation_time"] for r in results]
        total_times = [r + g for r, g in zip(retrieval_times, generation_times)]
        
        return {
            "retrieval": {
                "mean": statistics.mean(retrieval_times),
                "median": statistics.median(retrieval_times),
                "std": statistics.stdev(retrieval_times) if len(retrieval_times) > 1 else 0.0,
            },
            "generation": {
                "mean": statistics.mean(generation_times),
                "median": statistics.median(generation_times),
                "std": statistics.stdev(generation_times) if len(generation_times) > 1 else 0.0,
            },
            "total": {
                "mean": statistics.mean(total_times),
                "median": statistics.median(total_times),
                "std": statistics.stdev(total_times) if len(total_times) > 1 else 0.0,
            },
        }
    
    def mcnemar_test(
        self,
        results1: List[Dict],
        results2: List[Dict]
    ) -> Dict:
        """
        McNemar's test for paired nominal data.
        Tests if two methods have significantly different error rates.
        """
        if len(results1) != len(results2):
            return {"error": "Sample sizes must match"}
        
        # Build contingency table
        # a = both correct
        # b = method1 correct, method2 incorrect
        # c = method1 incorrect, method2 correct
        # d = both incorrect
        
        a = sum(1 for r1, r2 in zip(results1, results2) 
                if r1["is_correct"] and r2["is_correct"])
        b = sum(1 for r1, r2 in zip(results1, results2) 
                if r1["is_correct"] and not r2["is_correct"])
        c = sum(1 for r1, r2 in zip(results1, results2) 
                if not r1["is_correct"] and r2["is_correct"])
        d = sum(1 for r1, r2 in zip(results1, results2) 
                if not r1["is_correct"] and not r2["is_correct"])
        
        # McNemar test statistic
        # chi2 = (|b - c| - 1)^2 / (b + c)
        # Continuity correction applied
        
        if b + c == 0:
            return {
                "contingency_table": {"a": a, "b": b, "c": c, "d": d},
                "chi2": 0.0,
                "p_value": 1.0,
                "significant": False,
            }
        
        chi2 = ((abs(b - c) - 1) ** 2) / (b + c)
        
        # Chi-square with 1 df
        # p < 0.05 if chi2 > 3.841
        # p < 0.01 if chi2 > 6.635
        
        if chi2 > 6.635:
            p_value = "p < 0.01"
            significant = True
        elif chi2 > 3.841:
            p_value = "p < 0.05"
            significant = True
        else:
            p_value = "p > 0.05"
            significant = False
        
        return {
            "contingency_table": {"a": a, "b": b, "c": c, "d": d},
            "chi2": chi2,
            "p_value": p_value,
            "significant": significant,
        }
    
    def paired_t_test(
        self,
        results1: List[Dict],
        results2: List[Dict],
        metric: str = "confidence_score"
    ) -> Dict:
        """
        Paired t-test for continuous metrics.
        Compares means of paired samples.
        """
        if len(results1) != len(results2):
            return {"error": "Sample sizes must match"}
        
        values1 = [r[metric] for r in results1]
        values2 = [r[metric] for r in results2]
        
        differences = [v1 - v2 for v1, v2 in zip(values1, values2)]
        
        if len(differences) < 2:
            return {"error": "Need at least 2 samples"}
        
        mean_diff = statistics.mean(differences)
        std_diff = statistics.stdev(differences)
        
        if std_diff == 0:
            return {
                "mean_diff": mean_diff,
                "t_statistic": 0.0,
                "p_value": "p = 1.0",
                "significant": False,
            }
        
        # t = mean_diff / (std_diff / sqrt(n))
        import math
        n = len(differences)
        t_stat = mean_diff / (std_diff / math.sqrt(n))
        
        # Simplified significance (df = n-1 ≈ 24)
        # t(24, 0.05) ≈ 2.064
        # t(24, 0.01) ≈ 2.797
        
        if abs(t_stat) > 2.797:
            p_value = "p < 0.01"
            significant = True
        elif abs(t_stat) > 2.064:
            p_value = "p < 0.05"
            significant = True
        else:
            p_value = "p > 0.05"
            significant = False
        
        return {
            "mean_diff": mean_diff,
            "std_diff": std_diff,
            "t_statistic": t_stat,
            "p_value": p_value,
            "significant": significant,
        }
    
    def cohens_d(
        self,
        results1: List[Dict],
        results2: List[Dict],
        metric: str = "confidence_score"
    ) -> float:
        """
        Cohen's d effect size.
        Measures standardized difference between two means.
        
        |d| < 0.2: small effect
        |d| < 0.5: medium effect
        |d| >= 0.8: large effect
        """
        values1 = [r[metric] for r in results1]
        values2 = [r[metric] for r in results2]
        
        mean1 = statistics.mean(values1)
        mean2 = statistics.mean(values2)
        
        if len(values1) < 2 or len(values2) < 2:
            return 0.0
        
        std1 = statistics.stdev(values1)
        std2 = statistics.stdev(values2)
        
        # Pooled standard deviation
        import math
        n1, n2 = len(values1), len(values2)
        pooled_std = math.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))
        
        if pooled_std == 0:
            return 0.0
        
        d = (mean1 - mean2) / pooled_std
        
        return d
    
    def calculate_all_metrics(self) -> Dict:
        """Calculate all metrics for all result files."""
        print("\n" + "="*80)
        print("CALCULATING METRICS")
        print("="*80 + "\n")
        
        all_metrics = {}
        
        for config_name, data in self.results.items():
            print(f"\nProcessing: {config_name}")
            results = data["results"]
            
            metrics = {
                "config_name": config_name,
                "total_samples": len(results),
                "accuracy": self.calculate_accuracy(results),
                "precision_at_1": self.calculate_precision_at_k(results, k=1),
                "precision_at_3": self.calculate_precision_at_k(results, k=3),
                "precision_at_5": self.calculate_precision_at_k(results, k=5),
                "precision_at_10": self.calculate_precision_at_k(results, k=10),
                "recall_at_1": self.calculate_recall_at_k(results, k=1),
                "recall_at_3": self.calculate_recall_at_k(results, k=3),
                "recall_at_5": self.calculate_recall_at_k(results, k=5),
                "recall_at_10": self.calculate_recall_at_k(results, k=10),
                "mrr": self.calculate_mrr(results),
                "hallucination_rate": self.calculate_hallucination_rate(results),
                "confidence_stats": self.calculate_confidence_stats(results),
                "latency_stats": self.calculate_latency_stats(results),
            }
            
            all_metrics[config_name] = metrics
            
            print(f"  Accuracy: {metrics['accuracy']:.2%}")
            print(f"  Precision@5: {metrics['precision_at_5']:.2%}")
            print(f"  MRR: {metrics['mrr']:.3f}")
            print(f"  Hallucination Rate: {metrics['hallucination_rate']:.2%}")
        
        # Statistical comparisons
        print("\n" + "="*80)
        print("STATISTICAL SIGNIFICANCE TESTS")
        print("="*80 + "\n")
        
        comparisons = self.calculate_statistical_comparisons()
        all_metrics["statistical_comparisons"] = comparisons
        
        return all_metrics
    
    def calculate_statistical_comparisons(self) -> Dict:
        """Calculate pairwise statistical comparisons."""
        comparisons = {}
        
        # Key comparisons for paper
        tests = [
            ("hotpotqa_hybrid", "hotpotqa_vector_only", "Hybrid vs Vector Baseline"),
            ("hotpotqa_hybrid", "hotpotqa_graph_only", "Hybrid vs Graph-only"),
            ("fever_full_system", "fever_no_graphverify", "Full vs No GraphVerify"),
        ]
        
        for config1, config2, description in tests:
            if config1 not in self.results or config2 not in self.results:
                continue
            
            results1 = self.results[config1]["results"]
            results2 = self.results[config2]["results"]
            
            print(f"\n{description}:")
            
            # McNemar test (for accuracy)
            mcnemar = self.mcnemar_test(results1, results2)
            print(f"  McNemar Test: {mcnemar['p_value']} "
                  f"({'significant' if mcnemar['significant'] else 'not significant'})")
            
            # Paired t-test (for confidence scores)
            ttest = self.paired_t_test(results1, results2, "confidence_score")
            if "error" not in ttest:
                print(f"  Paired T-test: {ttest['p_value']} "
                      f"({'significant' if ttest['significant'] else 'not significant'})")
            
            # Cohen's d (effect size)
            effect_size = self.cohens_d(results1, results2, "confidence_score")
            effect_label = "large" if abs(effect_size) >= 0.8 else "medium" if abs(effect_size) >= 0.5 else "small"
            print(f"  Cohen's d: {effect_size:.3f} ({effect_label} effect)")
            
            comparisons[f"{config1}_vs_{config2}"] = {
                "description": description,
                "mcnemar_test": mcnemar,
                "paired_ttest": ttest,
                "cohens_d": effect_size,
                "effect_size_label": effect_label,
            }
        
        return comparisons
    
    def save_metrics(self, output_path: Path):
        """Save metrics summary to JSON."""
        metrics = self.calculate_all_metrics()
        
        with open(output_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print(f"\n{'='*80}")
        print(f"Metrics saved to: {output_path}")
        print(f"{'='*80}\n")
    
    def print_summary_table(self):
        """Print formatted summary table."""
        metrics = self.calculate_all_metrics()
        
        print("\n" + "="*80)
        print("SUMMARY TABLE")
        print("="*80 + "\n")
        
        # Header
        print(f"{'Configuration':<30} {'Acc':<8} {'P@5':<8} {'MRR':<8} {'Hall%':<8}")
        print("-" * 80)
        
        # Rows
        for config_name in [
            "fever_full_system",
            "fever_no_graphverify",
            "hotpotqa_vector_only",
            "hotpotqa_graph_only",
            "hotpotqa_hybrid",
        ]:
            if config_name not in metrics:
                continue
            
            m = metrics[config_name]
            print(
                f"{config_name:<30} "
                f"{m['accuracy']:<8.2%} "
                f"{m['precision_at_5']:<8.2%} "
                f"{m['mrr']:<8.3f} "
                f"{m['hallucination_rate']:<8.2%}"
            )
        
        print("\n")


def main():
    parser = argparse.ArgumentParser(description="Calculate evaluation metrics")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path(__file__).parent / "results",
        help="Directory containing result JSON files"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent / "results" / "metrics_summary.json",
        help="Output path for metrics summary"
    )
    
    args = parser.parse_args()
    
    calculator = MetricsCalculator(args.results_dir)
    calculator.save_metrics(args.output)
    calculator.print_summary_table()


if __name__ == "__main__":
    main()
