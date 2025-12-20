#!/usr/bin/env python3
"""
Script 7: Generate comprehensive visualization graphs
Creates publication-ready charts for evaluation results
"""

import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
import seaborn as sns

# Setup
script_dir = Path(__file__).parent
evaluation_dir = script_dir.parent / "evaluation_results-2"
metrics_dir = script_dir.parent / "metrics"
viz_dir = metrics_dir / "visualizations"
viz_dir.mkdir(exist_ok=True)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['legend.fontsize'] = 11

# Load metrics
with open(metrics_dir / "summary.json") as f:
    metrics = json.load(f)

# Load all evaluation data
eval_files = sorted(evaluation_dir.glob("*.json"))
all_data = []
for f in eval_files:
    with open(f) as file:
        all_data.append(json.load(file))

print("📊 Generating visualizations...\n")

# ===== GRAPH 1: OVERALL ACCURACY COMPARISON =====
print("📈 Creating: Overall Accuracy Comparison")
fig, ax = plt.subplots(figsize=(10, 6))

methods = ['Hybrid', 'RAG', 'KG']
accuracies = [
    metrics['summary']['hybrid']['accuracy'],
    metrics['summary']['rag']['accuracy'],
    metrics['summary']['kg']['accuracy']
]
colors = ['#2ecc71', '#3498db', '#e74c3c']

bars = ax.bar(methods, accuracies, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)

# Add value labels on bars
for bar, acc in zip(bars, accuracies):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{acc:.1f}%',
            ha='center', va='bottom', fontsize=13, fontweight='bold')

ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
ax.set_title('Overall Accuracy Comparison (1000 queries)', fontsize=14, fontweight='bold')
ax.set_ylim(0, 100)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(viz_dir / "01_overall_accuracy.png", dpi=300, bbox_inches='tight')
plt.savefig(viz_dir / "01_overall_accuracy.pdf", bbox_inches='tight')
print(f"   ✅ Saved: 01_overall_accuracy.png")
plt.close()

# ===== GRAPH 2: PERFORMANCE BY DATASET =====
print("📈 Creating: Performance by Dataset")
fig, ax = plt.subplots(figsize=(12, 6))

datasets = ['FEVER', 'HotpotQA']
x = np.arange(len(datasets))
width = 0.25

hybrid_acc = [
    metrics['by_dataset']['FEVER']['hybrid']['accuracy'],
    metrics['by_dataset']['HotpotQA']['hybrid']['accuracy']
]
rag_acc = [
    metrics['by_dataset']['FEVER']['rag']['accuracy'],
    metrics['by_dataset']['HotpotQA']['rag']['accuracy']
]
kg_acc = [
    metrics['by_dataset']['FEVER']['kg']['accuracy'],
    metrics['by_dataset']['HotpotQA']['kg']['accuracy']
]

bars1 = ax.bar(x - width, hybrid_acc, width, label='Hybrid', color='#2ecc71', alpha=0.8, edgecolor='black')
bars2 = ax.bar(x, rag_acc, width, label='RAG', color='#3498db', alpha=0.8, edgecolor='black')
bars3 = ax.bar(x + width, kg_acc, width, label='KG', color='#e74c3c', alpha=0.8, edgecolor='black')

# Add value labels
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom', fontsize=10)

ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
ax.set_title('Performance by Dataset', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(datasets, fontsize=11)
ax.legend(fontsize=11)
ax.set_ylim(0, 110)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(viz_dir / "02_performance_by_dataset.png", dpi=300, bbox_inches='tight')
plt.savefig(viz_dir / "02_performance_by_dataset.pdf", bbox_inches='tight')
print(f"   ✅ Saved: 02_performance_by_dataset.png")
plt.close()

# ===== GRAPH 3: CONFIDENCE SCORE DISTRIBUTION =====
print("📈 Creating: Confidence Score Distribution")
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

hybrid_confs = []
rag_confs = []
kg_confs = []

for data in all_data:
    hybrid_confs.append(data['hybrid_evaluation'].get('confidence', 0))
    rag_confs.append(data['rag_only_evaluation'].get('confidence', 0))
    kg_confs.append(data['kg_only_evaluation'].get('confidence', 0))

# Hybrid
axes[0].hist(hybrid_confs, bins=30, color='#2ecc71', alpha=0.7, edgecolor='black')
axes[0].axvline(np.mean(hybrid_confs), color='darkgreen', linestyle='--', linewidth=2, label=f'Mean: {np.mean(hybrid_confs):.3f}')
axes[0].set_title('Hybrid Confidence Distribution', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Confidence Score', fontsize=11)
axes[0].set_ylabel('Frequency', fontsize=11)
axes[0].legend()
axes[0].grid(axis='y', alpha=0.3)

# RAG
axes[1].hist(rag_confs, bins=30, color='#3498db', alpha=0.7, edgecolor='black')
axes[1].axvline(np.mean(rag_confs), color='darkblue', linestyle='--', linewidth=2, label=f'Mean: {np.mean(rag_confs):.3f}')
axes[1].set_title('RAG Confidence Distribution', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Confidence Score', fontsize=11)
axes[1].set_ylabel('Frequency', fontsize=11)
axes[1].legend()
axes[1].grid(axis='y', alpha=0.3)

# KG
axes[2].hist(kg_confs, bins=30, color='#e74c3c', alpha=0.7, edgecolor='black')
axes[2].axvline(np.mean(kg_confs), color='darkred', linestyle='--', linewidth=2, label=f'Mean: {np.mean(kg_confs):.3f}')
axes[2].set_title('KG Confidence Distribution', fontsize=12, fontweight='bold')
axes[2].set_xlabel('Confidence Score', fontsize=11)
axes[2].set_ylabel('Frequency', fontsize=11)
axes[2].legend()
axes[2].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(viz_dir / "03_confidence_distribution.png", dpi=300, bbox_inches='tight')
plt.savefig(viz_dir / "03_confidence_distribution.pdf", bbox_inches='tight')
print(f"   ✅ Saved: 03_confidence_distribution.png")
plt.close()

# ===== GRAPH 4: CONFIDENCE STATISTICS BOX PLOT =====
print("📈 Creating: Confidence Statistics Box Plot")
fig, ax = plt.subplots(figsize=(10, 6))

box_data = [hybrid_confs, rag_confs, kg_confs]
bp = ax.boxplot(box_data, labels=['Hybrid', 'RAG', 'KG'], patch_artist=True,
                 widths=0.6, showmeans=True)

colors = ['#2ecc71', '#3498db', '#e74c3c']
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

ax.set_ylabel('Confidence Score', fontsize=12, fontweight='bold')
ax.set_title('Confidence Score Statistics by Method', fontsize=14, fontweight='bold')
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(viz_dir / "04_confidence_boxplot.png", dpi=300, bbox_inches='tight')
plt.savefig(viz_dir / "04_confidence_boxplot.pdf", bbox_inches='tight')
print(f"   ✅ Saved: 04_confidence_boxplot.png")
plt.close()

# ===== GRAPH 5: VERDICT DISTRIBUTION (HYBRID) =====
print("📈 Creating: Verdict Distribution")
fig, ax = plt.subplots(figsize=(10, 6))

verdicts = ['SUPPORTS', 'REFUTES', 'NOT ENOUGH INFO']
verdict_counts = [
    metrics['by_verdict']['hybrid'][v]['total'] for v in verdicts
]
colors_pie = ['#2ecc71', '#e74c3c', '#f39c12']

wedges, texts, autotexts = ax.pie(verdict_counts, labels=verdicts, autopct='%1.1f%%',
                                    colors=colors_pie, startangle=90, textprops={'fontsize': 11})

for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontweight('bold')
    autotext.set_fontsize(12)

ax.set_title('Verdict Distribution (Hybrid Method)', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig(viz_dir / "05_verdict_distribution.png", dpi=300, bbox_inches='tight')
plt.savefig(viz_dir / "05_verdict_distribution.pdf", bbox_inches='tight')
print(f"   ✅ Saved: 05_verdict_distribution.png")
plt.close()

# ===== GRAPH 6: ACCURACY BY VERDICT (HYBRID) =====
print("📈 Creating: Accuracy by Verdict Type")
fig, ax = plt.subplots(figsize=(11, 6))

verdicts = ['SUPPORTS', 'REFUTES', 'NOT ENOUGH INFO']
verdict_accs = [
    metrics['by_verdict']['hybrid'][v]['accuracy'] for v in verdicts
]
verdict_counts = [
    metrics['by_verdict']['hybrid'][v]['total'] for v in verdicts
]

# Create bars with width proportional to count
max_count = max(verdict_counts)
bar_widths = [count / max_count * 0.8 for count in verdict_counts]

bars = ax.bar(range(len(verdicts)), verdict_accs, width=bar_widths, 
              color=['#2ecc71', '#e74c3c', '#f39c12'], alpha=0.8, edgecolor='black', linewidth=1.5)

# Add labels
for i, (bar, acc, count) in enumerate(zip(bars, verdict_accs, verdict_counts)):
    height = bar.get_height()
    ax.text(i, height, f'{acc:.1f}%\n(n={count})',
            ha='center', va='bottom', fontsize=11, fontweight='bold')

ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
ax.set_title('Accuracy by Verdict Type (Hybrid Method)', fontsize=14, fontweight='bold')
ax.set_xticks(range(len(verdicts)))
ax.set_xticklabels(verdicts, fontsize=11)
ax.set_ylim(0, 100)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(viz_dir / "06_accuracy_by_verdict.png", dpi=300, bbox_inches='tight')
plt.savefig(viz_dir / "06_accuracy_by_verdict.pdf", bbox_inches='tight')
print(f"   ✅ Saved: 06_accuracy_by_verdict.png")
plt.close()

# ===== GRAPH 7: METHOD COMPARISON (GROUPED) =====
print("📈 Creating: Method Comparison Grid")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Overall
ax = axes[0, 0]
methods = ['Hybrid', 'RAG', 'KG']
accs = [64.2, 43.8, 53.7]
bars = ax.bar(methods, accs, color=['#2ecc71', '#3498db', '#e74c3c'], alpha=0.8, edgecolor='black')
for bar, acc in zip(bars, accs):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
            f'{acc:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
ax.set_title('Overall', fontsize=12, fontweight='bold')
ax.set_ylabel('Accuracy (%)', fontsize=11)
ax.set_ylim(0, 110)
ax.grid(axis='y', alpha=0.3)

# FEVER
ax = axes[0, 1]
accs = [33.4, 32.6, 37.4]
bars = ax.bar(methods, accs, color=['#2ecc71', '#3498db', '#e74c3c'], alpha=0.8, edgecolor='black')
for bar, acc in zip(bars, accs):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
            f'{acc:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
ax.set_title('FEVER (Fact Verification)', fontsize=12, fontweight='bold')
ax.set_ylim(0, 110)
ax.grid(axis='y', alpha=0.3)

# HotpotQA
ax = axes[1, 0]
accs = [95.0, 55.0, 70.0]
bars = ax.bar(methods, accs, color=['#2ecc71', '#3498db', '#e74c3c'], alpha=0.8, edgecolor='black')
for bar, acc in zip(bars, accs):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
            f'{acc:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
ax.set_title('HotpotQA (Multi-hop QA)', fontsize=12, fontweight='bold')
ax.set_ylabel('Accuracy (%)', fontsize=11)
ax.set_ylim(0, 110)
ax.grid(axis='y', alpha=0.3)

# Performance margin
ax = axes[1, 1]
margins_vs_rag = [64.2-43.8, 0, 53.7-43.8]
margins_vs_kg = [64.2-53.7, 43.8-53.7, 0]
x = np.arange(len(methods))
width = 0.35
bars1 = ax.bar(x - width/2, margins_vs_rag, width, label='vs RAG', color='#3498db', alpha=0.7, edgecolor='black')
bars2 = ax.bar(x + width/2, margins_vs_kg, width, label='vs KG', color='#e74c3c', alpha=0.7, edgecolor='black')
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        if height != 0:
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%', ha='center', va='bottom', fontsize=9)
ax.set_title('Accuracy Margin vs Other Methods', fontsize=12, fontweight='bold')
ax.set_ylabel('Accuracy Difference (%)', fontsize=11)
ax.set_xticks(x)
ax.set_xticklabels(methods)
ax.legend()
ax.grid(axis='y', alpha=0.3)
ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)

plt.tight_layout()
plt.savefig(viz_dir / "07_method_comparison_grid.png", dpi=300, bbox_inches='tight')
plt.savefig(viz_dir / "07_method_comparison_grid.pdf", bbox_inches='tight')
print(f"   ✅ Saved: 07_method_comparison_grid.png")
plt.close()

# ===== GRAPH 8: SAMPLE COUNTS BY DATASET =====
print("📈 Creating: Sample Distribution")
fig, ax = plt.subplots(figsize=(10, 6))

datasets = ['FEVER', 'HotpotQA']
counts = [500, 500]
colors_data = ['#e67e22', '#9b59b6']

bars = ax.bar(datasets, counts, color=colors_data, alpha=0.8, edgecolor='black', linewidth=1.5)
for bar, count in zip(bars, counts):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
            f'{count}\n(50%)',
            ha='center', va='bottom', fontsize=12, fontweight='bold')

ax.set_ylabel('Number of Samples', fontsize=12, fontweight='bold')
ax.set_title('Evaluation Dataset Distribution', fontsize=14, fontweight='bold')
ax.set_ylim(0, 600)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(viz_dir / "08_sample_distribution.png", dpi=300, bbox_inches='tight')
plt.savefig(viz_dir / "08_sample_distribution.pdf", bbox_inches='tight')
print(f"   ✅ Saved: 08_sample_distribution.png")
plt.close()

# ===== GRAPH 9: CONFIDENCE VS ACCURACY SCATTER =====
print("📈 Creating: Confidence vs Correctness")
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

for idx, (method_key, method_name, ax_idx) in enumerate([
    ('hybrid_evaluation', 'Hybrid', 0),
    ('rag_only_evaluation', 'RAG', 1),
    ('kg_only_evaluation', 'KG', 2)
]):
    confidences_correct = []
    confidences_incorrect = []
    
    for data in all_data:
        conf = data[method_key].get('confidence', 0)
        is_correct = data[method_key].get('correct', False)
        
        if is_correct:
            confidences_correct.append(conf)
        else:
            confidences_incorrect.append(conf)
    
    ax = axes[ax_idx]
    
    # Create violin plot
    parts = ax.violinplot([confidences_incorrect, confidences_correct],
                          positions=[0, 1],
                          showmeans=True,
                          widths=0.7)
    
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Incorrect', 'Correct'], fontsize=11)
    ax.set_ylabel('Confidence Score', fontsize=11)
    ax.set_title(f'{method_name} Method', fontsize=12, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0.3, 1.0)
    
    # Add counts
    ax.text(0, 0.25, f'n={len(confidences_incorrect)}', ha='center', fontsize=10)
    ax.text(1, 0.25, f'n={len(confidences_correct)}', ha='center', fontsize=10)

plt.tight_layout()
plt.savefig(viz_dir / "09_confidence_vs_correctness.png", dpi=300, bbox_inches='tight')
plt.savefig(viz_dir / "09_confidence_vs_correctness.pdf", bbox_inches='tight')
print(f"   ✅ Saved: 09_confidence_vs_correctness.png")
plt.close()

# ===== GRAPH 10: PERFORMANCE SUMMARY TABLE (AS IMAGE) =====
print("📈 Creating: Performance Summary Table Visualization")
fig, ax = plt.subplots(figsize=(12, 6))
ax.axis('tight')
ax.axis('off')

# Create table data
table_data = [
    ['Method', 'Overall', 'FEVER', 'HotpotQA', '95% CI', 'Avg Conf'],
    ['Hybrid', '64.2%', '33.4%', '95.0%', '±2.97%', '0.879'],
    ['RAG', '43.8%', '32.6%', '55.0%', '±3.07%', '0.848'],
    ['KG', '53.7%', '37.4%', '70.0%', '±3.08%', '0.624']
]

table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                colWidths=[0.15, 0.15, 0.15, 0.15, 0.15, 0.15])

table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1, 2.5)

# Style header row
for i in range(6):
    table[(0, i)].set_facecolor('#34495e')
    table[(0, i)].set_text_props(weight='bold', color='white')

# Style data rows
colors_table = ['#2ecc71', '#3498db', '#e74c3c']
for i in range(1, 4):
    for j in range(6):
        table[(i, j)].set_facecolor(colors_table[i-1])
        table[(i, j)].set_alpha(0.3)

plt.title('Performance Summary Table', fontsize=14, fontweight='bold', pad=20)
plt.savefig(viz_dir / "10_performance_summary_table.png", dpi=300, bbox_inches='tight')
print(f"   ✅ Saved: 10_performance_summary_table.png")
plt.close()

# Print summary
print("\n" + "=" * 80)
print("✅ VISUALIZATION GENERATION COMPLETE!")
print("=" * 80)
print(f"\n📊 Generated {len(list(viz_dir.glob('*.png')))} visualization files:")
for i, f in enumerate(sorted(viz_dir.glob('*.png')), 1):
    size = f.stat().st_size / 1024
    print(f"   {i}. {f.name} ({size:.1f} KB)")

print(f"\n📁 All files saved to: {viz_dir}/")
print(f"\n📄 Available formats:")
print(f"   • PNG format (for web/presentations)")
print(f"   • PDF format (for papers/printing)")
