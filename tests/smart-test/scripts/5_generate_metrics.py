#!/usr/bin/env python3
"""
Script 5: Generate comprehensive metrics and analysis
Accuracy, confusion matrices, confidence statistics, and paper-ready summaries
"""

import json
from pathlib import Path
from collections import defaultdict
import numpy as np

# Setup paths
script_dir = Path(__file__).parent
evaluation_dir = script_dir.parent / "evaluation_results-2"
metrics_dir = script_dir.parent / "metrics"

metrics_dir.mkdir(exist_ok=True)

print("📊 Generating comprehensive evaluation metrics...\n")

# Load all evaluation results
eval_files = sorted(evaluation_dir.glob("*.json"))
all_data = []

for f in eval_files:
    with open(f) as file:
        all_data.append(json.load(file))

print(f"✅ Loaded {len(all_data)} evaluation results\n")

# Initialize metrics containers
metrics = {
    "summary": {},
    "by_dataset": {},
    "by_verdict": {},
    "confusion_matrices": {},
    "confidence_stats": {}
}

# ===== OVERALL METRICS =====
print("=" * 80)
print("🎯 CALCULATING OVERALL METRICS")
print("=" * 80 + "\n")

overall_correct = {"hybrid": 0, "rag": 0, "kg": 0}
overall_total = len(all_data)

for data in all_data:
    if data["hybrid_evaluation"].get("correct"):
        overall_correct["hybrid"] += 1
    if data["rag_only_evaluation"].get("correct"):
        overall_correct["rag"] += 1
    if data["kg_only_evaluation"].get("correct"):
        overall_correct["kg"] += 1

# Calculate accuracy percentages
metrics["summary"] = {
    "total_samples": overall_total,
    "hybrid": {
        "correct": overall_correct["hybrid"],
        "total": overall_total,
        "accuracy": round(100 * overall_correct["hybrid"] / overall_total, 2)
    },
    "rag": {
        "correct": overall_correct["rag"],
        "total": overall_total,
        "accuracy": round(100 * overall_correct["rag"] / overall_total, 2)
    },
    "kg": {
        "correct": overall_correct["kg"],
        "total": overall_total,
        "accuracy": round(100 * overall_correct["kg"] / overall_total, 2)
    }
}

print(f"Hybrid: {overall_correct['hybrid']}/{overall_total} ({metrics['summary']['hybrid']['accuracy']}%)")
print(f"RAG:    {overall_correct['rag']}/{overall_total} ({metrics['summary']['rag']['accuracy']}%)")
print(f"KG:     {overall_correct['kg']}/{overall_total} ({metrics['summary']['kg']['accuracy']}%)\n")

# ===== DATASET BREAKDOWN =====
print("=" * 80)
print("🔀 CALCULATING METRICS BY DATASET")
print("=" * 80 + "\n")

datasets = ["FEVER", "HotpotQA"]

for dataset in datasets:
    dataset_data = [d for d in all_data if d.get("dataset") == dataset]
    if not dataset_data:
        continue
    
    n = len(dataset_data)
    metrics["by_dataset"][dataset] = {
        "total": n,
        "hybrid": {"correct": 0, "accuracy": 0},
        "rag": {"correct": 0, "accuracy": 0},
        "kg": {"correct": 0, "accuracy": 0}
    }
    
    for data in dataset_data:
        if data["hybrid_evaluation"].get("correct"):
            metrics["by_dataset"][dataset]["hybrid"]["correct"] += 1
        if data["rag_only_evaluation"].get("correct"):
            metrics["by_dataset"][dataset]["rag"]["correct"] += 1
        if data["kg_only_evaluation"].get("correct"):
            metrics["by_dataset"][dataset]["kg"]["correct"] += 1
    
    # Calculate percentages
    for method in ["hybrid", "rag", "kg"]:
        metrics["by_dataset"][dataset][method]["accuracy"] = round(
            100 * metrics["by_dataset"][dataset][method]["correct"] / n, 2
        )
    
    print(f"{dataset} ({n} samples):")
    print(f"   Hybrid: {metrics['by_dataset'][dataset]['hybrid']['correct']}/{n} ({metrics['by_dataset'][dataset]['hybrid']['accuracy']}%)")
    print(f"   RAG:    {metrics['by_dataset'][dataset]['rag']['correct']}/{n} ({metrics['by_dataset'][dataset]['rag']['accuracy']}%)")
    print(f"   KG:     {metrics['by_dataset'][dataset]['kg']['correct']}/{n} ({metrics['by_dataset'][dataset]['kg']['accuracy']}%)\n")

# ===== VERDICT BREAKDOWN =====
print("=" * 80)
print("📈 CALCULATING METRICS BY VERDICT")
print("=" * 80 + "\n")

verdicts = ["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"]
verdict_dist = {"hybrid": defaultdict(int), "rag": defaultdict(int), "kg": defaultdict(int)}
verdict_correct = {"hybrid": defaultdict(int), "rag": defaultdict(int), "kg": defaultdict(int)}

for data in all_data:
    # Hybrid
    h_verdict = data["hybrid_evaluation"]["verdict"]
    verdict_dist["hybrid"][h_verdict] += 1
    if data["hybrid_evaluation"].get("correct"):
        verdict_correct["hybrid"][h_verdict] += 1
    
    # RAG
    r_verdict = data["rag_only_evaluation"]["verdict"]
    verdict_dist["rag"][r_verdict] += 1
    if data["rag_only_evaluation"].get("correct"):
        verdict_correct["rag"][r_verdict] += 1
    
    # KG
    k_verdict = data["kg_only_evaluation"]["verdict"]
    verdict_dist["kg"][k_verdict] += 1
    if data["kg_only_evaluation"].get("correct"):
        verdict_correct["kg"][k_verdict] += 1

# Store verdict breakdown
for method in ["hybrid", "rag", "kg"]:
    metrics["by_verdict"][method] = {}
    print(f"{method.upper()}:")
    for verdict in verdicts:
        total = verdict_dist[method][verdict]
        correct = verdict_correct[method][verdict]
        accuracy = round(100 * correct / total, 2) if total > 0 else 0
        
        metrics["by_verdict"][method][verdict] = {
            "total": total,
            "correct": correct,
            "accuracy": accuracy
        }
        
        pct = round(100 * total / overall_total, 2)
        print(f"   {verdict:20} {correct:3}/{total:3} correct ({accuracy:5.1f}%) | {pct:5.1f}% of all")
    print()

# ===== CONFUSION MATRICES =====
print("=" * 80)
print("🔀 CALCULATING CONFUSION MATRICES")
print("=" * 80 + "\n")

for method_key, method_name in [("hybrid_evaluation", "Hybrid"), 
                                 ("rag_only_evaluation", "RAG"),
                                 ("kg_only_evaluation", "KG")]:
    # Build confusion matrix
    conf_matrix = defaultdict(lambda: defaultdict(int))
    
    for data in all_data:
        gt = data.get("ground_truth")
        pred = data[method_key]["verdict"]
        conf_matrix[gt][pred] += 1
    
    # Store as dictionary
    metrics["confusion_matrices"][method_name] = {}
    for gt_label in verdicts:
        metrics["confusion_matrices"][method_name][gt_label] = {}
        for pred_label in verdicts:
            metrics["confusion_matrices"][method_name][gt_label][pred_label] = conf_matrix[gt_label][pred_label]
    
    # Print as table
    print(f"{method_name} Confusion Matrix (GT → Prediction):")
    print(f"{'':25} {'SUPPORTS':>10} {'REFUTES':>10} {'NOT ENOUGH INFO':>20}")
    for gt in verdicts:
        row_data = [conf_matrix[gt][v] for v in verdicts]
        print(f"{gt:25} {row_data[0]:>10} {row_data[1]:>10} {row_data[2]:>20}")
    print()

# ===== CONFIDENCE STATISTICS =====
print("=" * 80)
print("📊 CALCULATING CONFIDENCE STATISTICS")
print("=" * 80 + "\n")

for method_key, method_name in [("hybrid_evaluation", "Hybrid"),
                                 ("rag_only_evaluation", "RAG"),
                                 ("kg_only_evaluation", "KG")]:
    confidences = []
    confidences_correct = []
    confidences_incorrect = []
    
    for data in all_data:
        conf = data[method_key].get("confidence", 0)
        if isinstance(conf, (int, float)):
            confidences.append(conf)
            
            if data[method_key].get("correct"):
                confidences_correct.append(conf)
            else:
                confidences_incorrect.append(conf)
    
    if confidences:
        metrics["confidence_stats"][method_name] = {
            "overall": {
                "mean": round(np.mean(confidences), 4),
                "std": round(np.std(confidences), 4),
                "min": round(np.min(confidences), 4),
                "max": round(np.max(confidences), 4),
                "count": len(confidences)
            }
        }
        
        if confidences_correct:
            metrics["confidence_stats"][method_name]["correct_predictions"] = {
                "mean": round(np.mean(confidences_correct), 4),
                "std": round(np.std(confidences_correct), 4),
                "count": len(confidences_correct)
            }
        
        if confidences_incorrect:
            metrics["confidence_stats"][method_name]["incorrect_predictions"] = {
                "mean": round(np.mean(confidences_incorrect), 4),
                "std": round(np.std(confidences_incorrect), 4),
                "count": len(confidences_incorrect)
            }
        
        print(f"{method_name}:")
        print(f"   Overall:  mean={metrics['confidence_stats'][method_name]['overall']['mean']:.4f}, " +
              f"std={metrics['confidence_stats'][method_name]['overall']['std']:.4f}, " +
              f"range=[{metrics['confidence_stats'][method_name]['overall']['min']:.4f}, {metrics['confidence_stats'][method_name]['overall']['max']:.4f}]")
        
        if confidences_correct:
            print(f"   Correct:  mean={metrics['confidence_stats'][method_name]['correct_predictions']['mean']:.4f} " +
                  f"({len(confidences_correct)} predictions)")
        
        if confidences_incorrect:
            print(f"   Incorrect: mean={metrics['confidence_stats'][method_name]['incorrect_predictions']['mean']:.4f} " +
                  f"({len(confidences_incorrect)} predictions)")
        print()

# ===== SAVE METRICS TO JSON =====
print("=" * 80)
print("💾 SAVING METRICS")
print("=" * 80 + "\n")

summary_file = metrics_dir / "summary.json"
with open(summary_file, "w") as f:
    json.dump(metrics, f, indent=2)
print(f"✅ Saved comprehensive metrics to: {summary_file}")

# ===== GENERATE PAPER-READY TABLES =====
print("\n" + "=" * 80)
print("📄 PAPER-READY SUMMARY")
print("=" * 80 + "\n")

# Table 1: Overall Results
print("TABLE 1: Overall Evaluation Results\n")
print(f"{'Method':<15} {'Accuracy':<15} {'#Correct':<15} {'#Total':<15}")
print("-" * 60)
for method, key in [("Hybrid", "hybrid"), ("RAG", "rag"), ("KG", "kg")]:
    acc = metrics["summary"][key]["accuracy"]
    correct = metrics["summary"][key]["correct"]
    total = metrics["summary"][key]["total"]
    print(f"{method:<15} {acc:>6.2f}%         {correct:>6}/{total:<8}")

# Table 2: Dataset Breakdown
print("\n\nTABLE 2: Performance by Dataset\n")
print(f"{'Dataset':<15} {'Method':<12} {'Accuracy':<15} {'#Correct/#Total':<20}")
print("-" * 62)
for dataset in ["FEVER", "HotpotQA"]:
    for i, (method, key) in enumerate([("Hybrid", "hybrid"), ("RAG", "rag"), ("KG", "kg")]):
        ds_data = metrics["by_dataset"].get(dataset, {})
        if ds_data:
            acc = ds_data[key]["accuracy"]
            correct = ds_data[key]["correct"]
            total = ds_data["total"]
            
            ds_label = dataset if i == 0 else ""
            print(f"{ds_label:<15} {method:<12} {acc:>6.2f}%         {correct:>3}/{total:<8}")

# Table 3: Verdict Distribution (Hybrid)
print("\n\nTABLE 3: Verdict Distribution (Hybrid Method)\n")
print(f"{'Verdict':<25} {'Count':<15} {'% of Total':<15} {'Accuracy':<15}")
print("-" * 70)
for verdict in ["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"]:
    v_data = metrics["by_verdict"]["hybrid"][verdict]
    count = v_data["total"]
    pct = round(100 * count / overall_total, 2)
    acc = v_data["accuracy"]
    print(f"{verdict:<25} {count:<15} {pct:>6.2f}%         {acc:>6.2f}%")

# Table 4: Confidence Statistics
print("\n\nTABLE 4: Confidence Score Statistics\n")
print(f"{'Method':<15} {'Mean Conf':<15} {'Std Dev':<15} {'Min':<12} {'Max':<12}")
print("-" * 69)
for method in ["Hybrid", "RAG", "KG"]:
    if method in metrics["confidence_stats"]:
        c_stats = metrics["confidence_stats"][method]["overall"]
        print(f"{method:<15} {c_stats['mean']:<15.4f} {c_stats['std']:<15.4f} {c_stats['min']:<12.4f} {c_stats['max']:<12.4f}")

print("\n" + "=" * 80)
print("✅ METRICS GENERATION COMPLETE!")
print("=" * 80)
print(f"\n📊 Results saved to: {metrics_dir}/")
print(f"   - summary.json (all numerical metrics)")
print(f"   - This output (paper-ready tables)")
