"""
Benchmark Configuration Settings
"""
from pathlib import Path
from typing import Dict, Any

# Base paths
BENCHMARK_ROOT = Path(__file__).parent
DATA_DIR = BENCHMARK_ROOT / "data"
REPORTS_DIR = BENCHMARK_ROOT / "reports"
CHARTS_DIR = REPORTS_DIR / "charts"
TABLES_DIR = REPORTS_DIR / "tables"

# Dataset configurations
DATASET_CONFIGS: Dict[str, Dict[str, Any]] = {
    "fever": {
        "name": "FEVER",
        "full_name": "Fact Extraction and VERification",
        "url": "https://fever.ai/resources/download.html",
        "sample_size": 1000,  # Use 1000 samples from dev set
        "classes": ["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"],
        "metrics": ["accuracy", "precision", "recall", "f1"],
    },
    "scifact": {
        "name": "SciFact",
        "full_name": "Scientific Fact Verification",
        "url": "https://github.com/allenai/scifact",
        "sample_size": 300,  # Full test set
        "classes": ["SUPPORT", "CONTRADICT", "NOT_ENOUGH_INFO"],
        "metrics": ["accuracy", "precision", "recall", "f1"],
    },
    "hotpotqa": {
        "name": "HotpotQA",
        "full_name": "Multi-hop Question Answering",
        "url": "https://hotpotqa.github.io/",
        "sample_size": 500,  # Sample from dev set
        "question_types": ["bridge", "comparison"],
        "metrics": ["exact_match", "f1", "precision", "recall"],
    },
    "metaqa": {
        "name": "MetaQA",
        "full_name": "Knowledge Graph Question Answering",
        "url": "https://github.com/yuyuz/MetaQA",
        "sample_size": 300,  # 100 each: 1-hop, 2-hop, 3-hop
        "hop_types": [1, 2, 3],
        "metrics": ["accuracy", "hits@1", "hits@5", "hits@10"],
    },
    "wikidata5m": {
        "name": "Wikidata5M",
        "full_name": "Wikidata Entity Linking",
        "url": "https://deepgraphlearning.github.io/project/wikidata5m",
        "sample_size": 10000,  # Sample entities
        "metrics": ["precision@k", "recall@k", "mrr", "hits@1"],
    },
    "dbpedia": {
        "name": "DBpedia",
        "full_name": "DBpedia Knowledge Graph",
        "url": "https://www.dbpedia.org/",
        "sample_size": 1000,  # Sample entities
        "categories": ["Person", "Place", "Organization"],
        "metrics": ["triple_precision", "triple_recall", "triple_f1"],
    },
    "trustkg": {
        "name": "TrustKG",
        "full_name": "Trustworthiness Benchmark (Synthetic)",
        "url": "internal",
        "test_suites": {
            "hallucinations": 100,
            "temporal": 100,
            "conflicts": 100,
            "missing_facts": 100,
        },
        "metrics": ["detection_rate", "precision", "recall", "f1"],
    },
}

# Baseline configurations
BASELINE_CONFIGS = {
    "pure_rag": {
        "name": "Pure RAG",
        "description": "FAISS semantic search only (no graph)",
        "enabled": True,
    },
    "pure_kg": {
        "name": "Pure KG",
        "description": "Neo4j graph traversal only (no embeddings)",
        "enabled": True,
    },
    "wikipedia_api": {
        "name": "Wikipedia API",
        "description": "Direct Wikipedia API queries (no KG)",
        "enabled": True,
    },
}

# Evaluation settings
EVAL_SETTINGS = {
    "confidence_thresholds": [0.5, 0.6, 0.7, 0.8, 0.9],
    "k_values": [1, 5, 10, 20],  # For P@K, R@K metrics
    "parallel_workers": 4,
    "batch_size": 32,
    "timeout_seconds": 300,
}

# Visualization settings
VIZ_SETTINGS = {
    "figsize": (12, 8),
    "dpi": 300,
    "style": "seaborn-v0_8-darkgrid",
    "colors": {
        "graphbuilder": "#2E86AB",
        "pure_rag": "#A23B72",
        "pure_kg": "#F18F01",
        "wikipedia": "#C73E1D",
    },
    "export_formats": ["png", "pdf", "svg"],
}

# Report settings
REPORT_SETTINGS = {
    "generate_latex": True,
    "generate_markdown": True,
    "generate_json": True,
    "generate_csv": True,
    "include_error_analysis": True,
    "include_ablation_study": True,
}
