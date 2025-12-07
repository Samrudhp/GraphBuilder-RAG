"""
Visualization Utilities for Benchmark Results

Generates publication-ready charts and plots:
- Bar charts for dataset comparisons
- Confusion matrices for classification tasks
- Line plots for threshold analysis
- Radar charts for multi-metric comparison
- Heatmaps for error analysis
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd

from tests.benchmarks.config import CHARTS_DIR, VIZ_SETTINGS


class BenchmarkVisualizer:
    """Generate visualizations for benchmark results."""
    
    def __init__(self, output_dir: Optional[Path] = None):
        """
        Initialize visualizer.
        
        Args:
            output_dir: Directory to save charts (default: CHARTS_DIR)
        """
        self.output_dir = output_dir or CHARTS_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set style
        plt.style.use(VIZ_SETTINGS["style"])
        self.colors = VIZ_SETTINGS["colors"]
        self.figsize = VIZ_SETTINGS["figsize"]
        self.dpi = VIZ_SETTINGS["dpi"]
    
    def plot_dataset_comparison(
        self,
        results: Dict[str, Dict[str, float]],
        metric: str = "f1",
        title: Optional[str] = None,
        output_name: str = "dataset_comparison"
    ) -> Path:
        """
        Plot bar chart comparing performance across datasets.
        
        Args:
            results: Dict mapping dataset_name -> {system_name: metric_value}
            metric: Metric to plot
            title: Chart title
            output_name: Output filename (without extension)
        
        Returns:
            Path to saved chart
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        datasets = list(results.keys())
        systems = list(next(iter(results.values())).keys())
        
        x = np.arange(len(datasets))
        width = 0.8 / len(systems)
        
        for i, system in enumerate(systems):
            values = [results[ds].get(system, 0) for ds in datasets]
            offset = (i - len(systems)/2) * width + width/2
            ax.bar(
                x + offset,
                values,
                width,
                label=system,
                color=self.colors.get(system.lower().replace(" ", "_"), f"C{i}")
            )
        
        ax.set_xlabel("Dataset", fontsize=12, fontweight="bold")
        ax.set_ylabel(metric.upper(), fontsize=12, fontweight="bold")
        ax.set_title(title or f"{metric.upper()} Comparison Across Datasets", fontsize=14, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(datasets, rotation=45, ha="right")
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        # Save in multiple formats
        output_paths = []
        for fmt in VIZ_SETTINGS["export_formats"]:
            output_path = self.output_dir / f"{output_name}.{fmt}"
            plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
            output_paths.append(output_path)
        
        plt.close()
        return output_paths[0]  # Return PNG path
    
    def plot_confusion_matrix(
        self,
        confusion_matrix: np.ndarray,
        labels: List[str],
        title: str = "Confusion Matrix",
        output_name: str = "confusion_matrix",
        normalize: bool = True
    ) -> Path:
        """
        Plot confusion matrix heatmap.
        
        Args:
            confusion_matrix: Confusion matrix array
            labels: Class labels
            title: Chart title
            output_name: Output filename
            normalize: Normalize by row (True = show percentages)
        
        Returns:
            Path to saved chart
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        if normalize:
            cm = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]
            fmt = '.2%'
        else:
            cm = confusion_matrix
            fmt = 'd'
        
        sns.heatmap(
            cm,
            annot=True,
            fmt=fmt,
            cmap='Blues',
            xticklabels=labels,
            yticklabels=labels,
            ax=ax,
            cbar_kws={'label': 'Proportion' if normalize else 'Count'}
        )
        
        ax.set_xlabel("Predicted Label", fontsize=12, fontweight="bold")
        ax.set_ylabel("True Label", fontsize=12, fontweight="bold")
        ax.set_title(title, fontsize=14, fontweight="bold")
        
        plt.tight_layout()
        
        output_path = self.output_dir / f"{output_name}.png"
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        return output_path
    
    def plot_threshold_analysis(
        self,
        thresholds: List[float],
        metrics: Dict[str, List[float]],
        title: str = "Threshold Analysis",
        output_name: str = "threshold_analysis"
    ) -> Path:
        """
        Plot line chart showing metrics vs confidence thresholds.
        
        Args:
            thresholds: List of threshold values
            metrics: Dict mapping metric_name -> list of values
            title: Chart title
            output_name: Output filename
        
        Returns:
            Path to saved chart
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        for metric_name, values in metrics.items():
            ax.plot(thresholds, values, marker='o', label=metric_name, linewidth=2)
        
        ax.set_xlabel("Confidence Threshold", fontsize=12, fontweight="bold")
        ax.set_ylabel("Score", fontsize=12, fontweight="bold")
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1.05])
        
        plt.tight_layout()
        
        output_path = self.output_dir / f"{output_name}.png"
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        return output_path
    
    def plot_radar_chart(
        self,
        systems: Dict[str, Dict[str, float]],
        metrics: List[str],
        title: str = "Multi-Metric Comparison",
        output_name: str = "radar_comparison"
    ) -> Path:
        """
        Plot radar chart comparing systems across multiple metrics.
        
        Args:
            systems: Dict mapping system_name -> {metric: value}
            metrics: List of metrics to include
            title: Chart title
            output_name: Output filename
        
        Returns:
            Path to saved chart
        """
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle
        
        for i, (system_name, system_metrics) in enumerate(systems.items()):
            values = [system_metrics.get(m, 0) for m in metrics]
            values += values[:1]  # Complete the circle
            
            color = self.colors.get(system_name.lower().replace(" ", "_"), f"C{i}")
            ax.plot(angles, values, 'o-', linewidth=2, label=system_name, color=color)
            ax.fill(angles, values, alpha=0.15, color=color)
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics, size=10)
        ax.set_ylim([0, 1])
        ax.set_title(title, fontsize=14, fontweight="bold", pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        ax.grid(True)
        
        plt.tight_layout()
        
        output_path = self.output_dir / f"{output_name}.png"
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        return output_path
    
    def plot_error_distribution(
        self,
        error_categories: Dict[str, int],
        title: str = "Error Distribution",
        output_name: str = "error_distribution"
    ) -> Path:
        """
        Plot pie chart of error categories.
        
        Args:
            error_categories: Dict mapping error_type -> count
            title: Chart title
            output_name: Output filename
        
        Returns:
            Path to saved chart
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        categories = list(error_categories.keys())
        counts = list(error_categories.values())
        
        colors_list = plt.cm.Set3(np.linspace(0, 1, len(categories)))
        
        wedges, texts, autotexts = ax.pie(
            counts,
            labels=categories,
            autopct='%1.1f%%',
            startangle=90,
            colors=colors_list,
            textprops={'fontsize': 10}
        )
        
        ax.set_title(title, fontsize=14, fontweight="bold")
        
        plt.tight_layout()
        
        output_path = self.output_dir / f"{output_name}.png"
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        return output_path
    
    def plot_system_comparison_heatmap(
        self,
        results: Dict[str, Dict[str, float]],
        title: str = "System Performance Heatmap",
        output_name: str = "system_heatmap"
    ) -> Path:
        """
        Plot heatmap comparing systems across datasets/metrics.
        
        Args:
            results: Dict mapping dataset -> {system: metric_value}
            title: Chart title
            output_name: Output filename
        
        Returns:
            Path to saved chart
        """
        # Convert to DataFrame
        df = pd.DataFrame(results).T
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        sns.heatmap(
            df,
            annot=True,
            fmt='.3f',
            cmap='RdYlGn',
            center=0.7,
            vmin=0,
            vmax=1,
            ax=ax,
            cbar_kws={'label': 'Score'}
        )
        
        ax.set_xlabel("System", fontsize=12, fontweight="bold")
        ax.set_ylabel("Dataset", fontsize=12, fontweight="bold")
        ax.set_title(title, fontsize=14, fontweight="bold")
        
        plt.tight_layout()
        
        output_path = self.output_dir / f"{output_name}.png"
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        return output_path
    
    def generate_all_visualizations(
        self,
        benchmark_results: Dict[str, Any],
        baseline_results: Dict[str, Any]
    ) -> Dict[str, Path]:
        """
        Generate all visualizations for a complete benchmark run.
        
        Args:
            benchmark_results: Results from GraphBuilder system
            baseline_results: Results from baseline systems
        
        Returns:
            Dict mapping visualization_type -> output_path
        """
        outputs = {}
        
        # Dataset comparison
        comparison_data = {}
        for dataset, results in benchmark_results.items():
            comparison_data[dataset] = {
                "GraphBuilder": results["metrics"].get("f1", 0),
                **{
                    system: baseline_results[dataset][system]["metrics"].get("f1", 0)
                    for system in baseline_results.get(dataset, {})
                }
            }
        
        outputs["dataset_comparison"] = self.plot_dataset_comparison(
            comparison_data,
            metric="f1",
            title="F1 Score Comparison Across Datasets"
        )
        
        return outputs
