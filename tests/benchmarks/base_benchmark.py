"""
Base Benchmark Class

Abstract base class that all benchmark implementations inherit from.
Provides common interface for:
- Data loading
- System evaluation
- Metrics calculation
- Report generation
"""
import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

from tests.benchmarks.config import DATA_DIR, REPORTS_DIR
from tests.benchmarks.metrics import MetricsCalculator

logger = logging.getLogger(__name__)


class BaseBenchmark(ABC):
    """Abstract base class for all benchmarks."""
    
    def __init__(
        self,
        name: str,
        dataset_name: str,
        sample_size: Optional[int] = None
    ):
        """
        Initialize benchmark.
        
        Args:
            name: Benchmark name
            dataset_name: Dataset identifier
            sample_size: Number of samples to use (None = all)
        """
        self.name = name
        self.dataset_name = dataset_name
        self.sample_size = sample_size
        
        # Paths
        self.data_dir = DATA_DIR / dataset_name
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Results storage
        self.results: Dict[str, Any] = {
            "benchmark": name,
            "dataset": dataset_name,
            "timestamp": datetime.now().isoformat(),
            "sample_size": sample_size,
            "predictions": [],
            "gold_labels": [],
            "metrics": {},
            "errors": [],
        }
        
        # Metrics calculator
        self.metrics_calc = MetricsCalculator()
        
        logger.info(f"Initialized {name} benchmark")
    
    @abstractmethod
    def download_dataset(self) -> bool:
        """
        Download and prepare dataset.
        
        Returns:
            True if successful, False otherwise
        """
        pass
    
    @abstractmethod
    def load_data(self) -> List[Dict[str, Any]]:
        """
        Load dataset samples.
        
        Returns:
            List of data samples
        """
        pass
    
    @abstractmethod
    def prepare_input(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare a single sample for system input.
        
        Args:
            sample: Raw data sample
        
        Returns:
            Formatted input for the system
        """
        pass
    
    @abstractmethod
    async def run_system(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run the GraphBuilder system on input.
        
        Args:
            input_data: Prepared input
        
        Returns:
            System output/prediction
        """
        pass
    
    @abstractmethod
    def extract_prediction(self, system_output: Dict[str, Any]) -> Any:
        """
        Extract prediction from system output.
        
        Args:
            system_output: Raw system output
        
        Returns:
            Extracted prediction in comparable format
        """
        pass
    
    @abstractmethod
    def extract_gold_label(self, sample: Dict[str, Any]) -> Any:
        """
        Extract gold label from sample.
        
        Args:
            sample: Raw data sample
        
        Returns:
            Gold label in comparable format
        """
        pass
    
    @abstractmethod
    def calculate_metrics(self) -> Dict[str, float]:
        """
        Calculate benchmark-specific metrics.
        
        Returns:
            Dictionary of metric names and values
        """
        pass
    
    async def run(self, use_cached: bool = False) -> Dict[str, Any]:
        """
        Run complete benchmark pipeline.
        
        Args:
            use_cached: Use cached results if available
        
        Returns:
            Benchmark results dictionary
        """
        logger.info(f"Running {self.name} benchmark...")
        
        # Check for cached results
        cache_file = REPORTS_DIR / f"{self.dataset_name}_cache.json"
        if use_cached and cache_file.exists():
            logger.info(f"Loading cached results from {cache_file}")
            with open(cache_file) as f:
                return json.load(f)
        
        # Step 1: Download dataset
        logger.info("Step 1: Downloading dataset...")
        if not self.download_dataset():
            logger.error("Dataset download failed")
            return self.results
        
        # Step 2: Load data
        logger.info("Step 2: Loading data...")
        samples = self.load_data()
        logger.info(f"Loaded {len(samples)} samples")
        
        # Step 3: Run evaluation
        logger.info("Step 3: Running evaluation...")
        predictions = []
        gold_labels = []
        errors = []
        
        for i, sample in enumerate(samples):
            try:
                # Prepare input
                input_data = self.prepare_input(sample)
                
                # Run system
                system_output = await self.run_system(input_data)
                
                # Extract prediction and gold label
                pred = self.extract_prediction(system_output)
                gold = self.extract_gold_label(sample)
                
                predictions.append(pred)
                gold_labels.append(gold)
                
                if (i + 1) % 50 == 0:
                    logger.info(f"Processed {i + 1}/{len(samples)} samples")
                
            except Exception as e:
                logger.error(f"Error processing sample {i}: {e}")
                errors.append({
                    "sample_idx": i,
                    "error": str(e),
                })
        
        # Store results
        self.results["predictions"] = predictions
        self.results["gold_labels"] = gold_labels
        self.results["errors"] = errors
        self.results["num_samples"] = len(samples)
        self.results["num_errors"] = len(errors)
        
        # Step 4: Calculate metrics
        logger.info("Step 4: Calculating metrics...")
        metrics = self.calculate_metrics()
        self.results["metrics"] = metrics
        
        # Cache results
        with open(cache_file, "w") as f:
            json.dump(self.results, f, indent=2)
        
        logger.info(f"{self.name} benchmark complete!")
        logger.info(f"Metrics: {metrics}")
        
        return self.results
    
    def save_results(self, output_path: Optional[Path] = None) -> Path:
        """
        Save benchmark results to JSON file.
        
        Args:
            output_path: Custom output path (optional)
        
        Returns:
            Path to saved results file
        """
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = REPORTS_DIR / f"{self.dataset_name}_results_{timestamp}.json"
        
        with open(output_path, "w") as f:
            json.dump(self.results, f, indent=2)
        
        logger.info(f"Results saved to {output_path}")
        return output_path
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get benchmark summary.
        
        Returns:
            Summary dictionary with key metrics
        """
        return {
            "benchmark": self.name,
            "dataset": self.dataset_name,
            "num_samples": self.results.get("num_samples", 0),
            "num_errors": self.results.get("num_errors", 0),
            "metrics": self.results.get("metrics", {}),
        }
