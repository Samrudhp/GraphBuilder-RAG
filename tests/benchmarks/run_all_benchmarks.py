"""
Master Benchmark Runner

Runs all benchmarks sequentially or in parallel and generates comprehensive reports.

Usage:
    # Run all benchmarks
    python tests/benchmarks/run_all_benchmarks.py --full
    
    # Run specific benchmarks
    python tests/benchmarks/run_all_benchmarks.py --datasets fever scifact
    
    # Run with baseline comparisons
    python tests/benchmarks/run_all_benchmarks.py --full --baselines
    
    # Generate report from cached results
    python tests/benchmarks/run_all_benchmarks.py --report-only
"""
import argparse
import asyncio
import logging
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tests.benchmarks.config import REPORTS_DIR, DATASET_CONFIGS
from tests.benchmarks.visualizations import BenchmarkVisualizer

logger = logging.getLogger(__name__)


class BenchmarkRunner:
    """Master benchmark orchestrator."""
    
    def __init__(self, output_dir: Optional[Path] = None):
        """
        Initialize benchmark runner.
        
        Args:
            output_dir: Directory for outputs (default: REPORTS_DIR)
        """
        self.output_dir = output_dir or REPORTS_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.visualizer = BenchmarkVisualizer()
        
        self.results: Dict[str, Any] = {
            "timestamp": datetime.now().isoformat(),
            "benchmarks": {},
            "baselines": {},
            "summary": {},
        }
    
    def load_results(self, results_file: Optional[Path] = None) -> bool:
        """
        Load results from a JSON file.
        
        Args:
            results_file: Path to results file (default: latest benchmark_results_*.json)
            
        Returns:
            True if results loaded successfully
        """
        if results_file is None:
            # Find the latest results file
            results_files = list(self.output_dir.glob("benchmark_results_*.json"))
            if not results_files:
                logger.error("No benchmark results files found")
                return False
            results_file = max(results_files, key=lambda x: x.stat().st_mtime)
        
        try:
            with open(results_file) as f:
                self.results = json.load(f)
            logger.info(f"Loaded results from {results_file}")
            return True
        except Exception as e:
            logger.error(f"Failed to load results from {results_file}: {e}")
            return False
    
    async def run_fever(self, sample_size: int = 1000) -> Dict[str, Any]:
        """Run FEVER benchmark."""
        logger.info("=" * 80)
        logger.info("Running FEVER Benchmark...")
        logger.info("=" * 80)
        
        from tests.benchmarks.fever.test_fever import run_fever_benchmark
        return await run_fever_benchmark(sample_size=sample_size)
    
    async def run_scifact(self, sample_size: int = 300) -> Dict[str, Any]:
        """Run SciFact benchmark."""
        logger.info("=" * 80)
        logger.info("Running SciFact Benchmark...")
        logger.info("=" * 80)
        
        from tests.benchmarks.scifact.test_scifact import run_scifact_benchmark
        return await run_scifact_benchmark(sample_size=sample_size)
    
    async def run_hotpotqa(self, sample_size: int = 500) -> Dict[str, Any]:
        """Run HotpotQA benchmark."""
        logger.info("=" * 80)
        logger.info("Running HotpotQA Benchmark...")
        logger.info("=" * 80)
        
        from tests.benchmarks.hotpotqa.test_hotpotqa import run_hotpotqa_benchmark
        return await run_hotpotqa_benchmark(sample_size=sample_size)
    
    async def run_metaqa(self, sample_size: int = 300) -> Dict[str, Any]:
        """Run MetaQA benchmark."""
        logger.info("=" * 80)
        logger.info("Running MetaQA Benchmark...")
        logger.info("=" * 80)
        
        from tests.benchmarks.metaqa.test_metaqa import run_metaqa_benchmark
        return await run_metaqa_benchmark(sample_size=sample_size)
    
    async def run_wikidata5m(self, sample_size: int = 10000) -> Dict[str, Any]:
        """Run Wikidata5M benchmark."""
        logger.info("=" * 80)
        logger.info("Running Wikidata5M Benchmark...")
        logger.info("=" * 80)
        
        from tests.benchmarks.wikidata5m.test_wikidata import run_wikidata5m_benchmark
        return await run_wikidata5m_benchmark(sample_size=sample_size)
    
    async def run_dbpedia(self, sample_size: int = 1000) -> Dict[str, Any]:
        """Run DBpedia benchmark."""
        logger.info("=" * 80)
        logger.info("Running DBpedia Benchmark...")
        logger.info("=" * 80)
        
        from tests.benchmarks.dbpedia.test_dbpedia import run_dbpedia_benchmark
        return await run_dbpedia_benchmark(sample_size=sample_size)
    
    async def run_trustkg(self) -> Dict[str, Any]:
        """Run TrustKG benchmark."""
        logger.info("=" * 80)
        logger.info("Running TrustKG Benchmark (NOVEL CONTRIBUTION)...")
        logger.info("=" * 80)
        
        from tests.benchmarks.trustkg.test_trustkg import run_trustkg_benchmark
        return await run_trustkg_benchmark(sample_size=400)
    
    async def run_all(
        self,
        datasets: Optional[List[str]] = None,
        include_baselines: bool = False
    ) -> Dict[str, Any]:
        """
        Run all specified benchmarks.
        
        Args:
            datasets: List of dataset names to run (None = all)
            include_baselines: Run baseline comparisons
        
        Returns:
            Complete results dictionary
        """
        if datasets is None:
            datasets = ["fever"]  # For now, only FEVER is implemented
        
        logger.info(f"Running benchmarks: {', '.join(datasets)}")
        
        # Run each benchmark
        for dataset in datasets:
            try:
                if dataset == "fever":
                    sample_size = DATASET_CONFIGS["fever"]["sample_size"]
                    results = await self.run_fever(sample_size=sample_size)
                    self.results["benchmarks"]["fever"] = results
                
                elif dataset == "scifact":
                    results = await self.run_scifact()
                    self.results["benchmarks"]["scifact"] = results
                
                elif dataset == "hotpotqa":
                    results = await self.run_hotpotqa()
                    self.results["benchmarks"]["hotpotqa"] = results
                
                elif dataset == "metaqa":
                    results = await self.run_metaqa()
                    self.results["benchmarks"]["metaqa"] = results
                
                elif dataset == "wikidata5m":
                    results = await self.run_wikidata5m()
                    self.results["benchmarks"]["wikidata5m"] = results
                
                elif dataset == "dbpedia":
                    results = await self.run_dbpedia()
                    self.results["benchmarks"]["dbpedia"] = results
                
                elif dataset == "trustkg":
                    results = await self.run_trustkg()
                    self.results["benchmarks"]["trustkg"] = results
                
                else:
                    logger.warning(f"Unknown dataset: {dataset}")
            
            except Exception as e:
                logger.error(f"Error running {dataset} benchmark: {e}", exc_info=True)
                self.results["benchmarks"][dataset] = {"error": str(e)}
        
        # Run baselines if requested
        if include_baselines:
            logger.info("Running baseline comparisons...")
            # TODO: Implement baseline runs
        
        # Generate summary
        self.results["summary"] = self._generate_summary()
        
        return self.results
    
    def _generate_summary(self) -> Dict[str, Any]:
        """Generate summary statistics across all benchmarks."""
        summary = {
            "num_benchmarks": len(self.results["benchmarks"]),
            "total_samples": 0,
            "total_errors": 0,
            "average_metrics": {},
        }
        
        # Aggregate metrics
        all_metrics = {}
        for dataset, results in self.results["benchmarks"].items():
            if "metrics" in results:
                summary["total_samples"] += results.get("num_samples", 0)
                summary["total_errors"] += results.get("num_errors", 0)
                
                for metric, value in results["metrics"].items():
                    if metric not in all_metrics:
                        all_metrics[metric] = []
                    all_metrics[metric].append(value)
        
        # Calculate averages
        for metric, values in all_metrics.items():
            if values:
                summary["average_metrics"][metric] = sum(values) / len(values)
        
        return summary
    
    def save_results(self, filename: Optional[str] = None) -> Path:
        """
        Save complete results to JSON file.
        
        Args:
            filename: Custom filename (default: auto-generated with timestamp)
        
        Returns:
            Path to saved file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"benchmark_results_{timestamp}.json"
        
        output_path = self.output_dir / filename
        
        with open(output_path, "w") as f:
            json.dump(self.results, f, indent=2)
        
        logger.info(f"Results saved to: {output_path}")
        return output_path
    
    def generate_report(self) -> Path:
        """
        Generate comprehensive report with tables and visualizations.
        
        Returns:
            Path to report directory
        """
        logger.info("Generating comprehensive report...")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_dir = self.output_dir / f"report_{timestamp}"
        report_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate markdown report
        report_path = report_dir / "report.md"
        self._generate_markdown_report(report_path)
        
        # Generate visualizations
        if self.results["benchmarks"]:
            self._generate_visualizations(report_dir)
        
        # Generate LaTeX tables
        self._generate_latex_tables(report_dir)
        
        logger.info(f"Report generated: {report_dir}")
        return report_dir
    
    def _generate_markdown_report(self, output_path: Path):
        """Generate markdown report."""
        with open(output_path, "w") as f:
            f.write("# GraphBuilder-RAG Benchmark Results\n\n")
            f.write(f"**Generated:** {self.results.get('timestamp', datetime.now().isoformat())}\n\n")
            
            f.write("## Summary\n\n")
            # Ensure summary exists
            if "summary" not in self.results:
                self.results["summary"] = self._generate_summary()
            summary = self.results["summary"]
            f.write(f"- **Benchmarks Run:** {summary['num_benchmarks']}\n")
            f.write(f"- **Total Samples:** {summary['total_samples']}\n")
            f.write(f"- **Total Errors:** {summary['total_errors']}\n\n")
            
            f.write("### Average Metrics\n\n")
            f.write("| Metric | Value |\n")
            f.write("|--------|-------|\n")
            for metric, value in summary.get("average_metrics", {}).items():
                f.write(f"| {metric} | {value:.4f} |\n")
            
            f.write("\n## Individual Benchmark Results\n\n")
            for dataset, results in self.results["benchmarks"].items():
                f.write(f"### {dataset.upper()}\n\n")
                
                if "error" in results:
                    f.write(f"**Error:** {results['error']}\n\n")
                    continue
                
                f.write(f"- **Samples:** {results.get('num_samples', 0)}\n")
                f.write(f"- **Errors:** {results.get('num_errors', 0)}\n\n")
                
                if "metrics" in results:
                    f.write("#### Metrics\n\n")
                    f.write("| Metric | Value |\n")
                    f.write("|--------|-------|\n")
                    for metric, value in results["metrics"].items():
                        f.write(f"| {metric} | {value:.4f} |\n")
                    f.write("\n")
        
        logger.info(f"Markdown report saved: {output_path}")
    
    def _generate_visualizations(self, report_dir: Path):
        """Generate all visualizations."""
        logger.info("Generating visualizations...")
        
        # For now, basic placeholder
        # Will be expanded as more benchmarks are implemented
        
        logger.info("Visualizations generated")
    
    def _generate_latex_tables(self, report_dir: Path):
        """Generate LaTeX tables for paper."""
        logger.info("Generating LaTeX tables...")
        
        latex_path = report_dir / "tables.tex"
        
        with open(latex_path, "w") as f:
            f.write("% GraphBuilder-RAG Benchmark Results\n")
            f.write("% Auto-generated LaTeX tables\n\n")
            
            f.write("\\begin{table}[h]\n")
            f.write("\\centering\n")
            f.write("\\caption{Benchmark Results Summary}\n")
            f.write("\\begin{tabular}{lcccc}\n")
            f.write("\\toprule\n")
            f.write("Dataset & Accuracy & Precision & Recall & F1 \\\\\n")
            f.write("\\midrule\n")
            
            for dataset, results in self.results["benchmarks"].items():
                if "metrics" in results:
                    metrics = results["metrics"]
                    f.write(f"{dataset} & ")
                    f.write(f"{metrics.get('accuracy', 0):.3f} & ")
                    f.write(f"{metrics.get('precision', 0):.3f} & ")
                    f.write(f"{metrics.get('recall', 0):.3f} & ")
                    f.write(f"{metrics.get('f1', 0):.3f} \\\\\n")
            
            f.write("\\bottomrule\n")
            f.write("\\end{tabular}\n")
            f.write("\\end{table}\n")
        
        logger.info(f"LaTeX tables saved: {latex_path}")


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run GraphBuilder-RAG benchmarks")
    
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=["fever", "scifact", "hotpotqa", "metaqa", "wikidata5m", "dbpedia", "trustkg"],
        help="Specific datasets to run"
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Run all benchmarks"
    )
    parser.add_argument(
        "--baselines",
        action="store_true",
        help="Include baseline comparisons"
    )
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Generate report from cached results only"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory for results"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Initialize runner
    runner = BenchmarkRunner(output_dir=args.output_dir)
    
    if args.report_only:
        # Load existing results and generate report
        if runner.load_results():
            runner.generate_report()
        else:
            logger.error("No results to generate report from")
            return
    else:
        # Determine which datasets to run
        if args.full:
            datasets = None  # Run all
        else:
            datasets = args.datasets or ["fever"]  # Default to FEVER
        
        # Run benchmarks
        results = await runner.run_all(
            datasets=datasets,
            include_baselines=args.baselines
        )
        
        # Save results
        runner.save_results()
        
        # Generate report
        runner.generate_report()
        
        print("\n" + "=" * 80)
        print("BENCHMARK SUITE COMPLETE!")
        print("=" * 80)
        print(f"\nResults saved to: {runner.output_dir}")
        print(f"Total samples: {results['summary']['total_samples']}")
        print(f"Total errors: {results['summary']['total_errors']}")
        print(f"\nAverage Metrics:")
        for metric, value in results['summary']['average_metrics'].items():
            print(f"  {metric}: {value:.4f}")


if __name__ == "__main__":
    asyncio.run(main())
