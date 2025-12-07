"""
Evaluation Metrics for Benchmarks

Implements standard metrics for:
- Classification: Accuracy, Precision, Recall, F1
- QA: Exact Match (EM), F1
- Ranking: MRR, Hits@K, P@K, R@K
- Graph: Triple accuracy, Edge coverage
"""
import numpy as np
from typing import List, Dict, Tuple, Any, Optional
from collections import Counter
import re


class MetricsCalculator:
    """Calculate evaluation metrics for benchmark results."""
    
    @staticmethod
    def accuracy(predictions: List[str], gold_labels: List[str]) -> float:
        """Calculate classification accuracy."""
        if len(predictions) != len(gold_labels):
            raise ValueError("Predictions and labels must have same length")
        
        correct = sum(p == g for p, g in zip(predictions, gold_labels))
        return correct / len(predictions) if predictions else 0.0
    
    @staticmethod
    def precision_recall_f1(
        predictions: List[str],
        gold_labels: List[str],
        label: Optional[str] = None,
        average: str = "macro"
    ) -> Dict[str, float]:
        """
        Calculate precision, recall, F1.
        
        Args:
            predictions: Predicted labels
            gold_labels: True labels
            label: Specific label to calculate metrics for (None = all labels)
            average: 'macro', 'micro', or 'weighted'
        """
        if len(predictions) != len(gold_labels):
            raise ValueError("Predictions and labels must have same length")
        
        # Get unique labels
        all_labels = sorted(set(gold_labels + predictions))
        
        if label:
            all_labels = [label]
        
        results = {}
        label_metrics = []
        label_counts = []
        
        for lbl in all_labels:
            tp = sum((p == lbl and g == lbl) for p, g in zip(predictions, gold_labels))
            fp = sum((p == lbl and g != lbl) for p, g in zip(predictions, gold_labels))
            fn = sum((p != lbl and g == lbl) for p, g in zip(predictions, gold_labels))
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            
            label_metrics.append({
                "precision": precision,
                "recall": recall,
                "f1": f1,
            })
            label_counts.append(sum(g == lbl for g in gold_labels))
        
        # Calculate average
        if average == "macro":
            results["precision"] = np.mean([m["precision"] for m in label_metrics])
            results["recall"] = np.mean([m["recall"] for m in label_metrics])
            results["f1"] = np.mean([m["f1"] for m in label_metrics])
        elif average == "weighted":
            total = sum(label_counts)
            results["precision"] = sum(m["precision"] * c for m, c in zip(label_metrics, label_counts)) / total
            results["recall"] = sum(m["recall"] * c for m, c in zip(label_metrics, label_counts)) / total
            results["f1"] = sum(m["f1"] * c for m, c in zip(label_metrics, label_counts)) / total
        
        return results
    
    @staticmethod
    def confusion_matrix(
        predictions: List[str],
        gold_labels: List[str]
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Generate confusion matrix.
        
        Returns:
            matrix: Confusion matrix as numpy array
            labels: Ordered list of label names
        """
        labels = sorted(set(gold_labels + predictions))
        n = len(labels)
        matrix = np.zeros((n, n), dtype=int)
        
        label_to_idx = {label: i for i, label in enumerate(labels)}
        
        for pred, gold in zip(predictions, gold_labels):
            i = label_to_idx[gold]
            j = label_to_idx[pred]
            matrix[i][j] += 1
        
        return matrix, labels
    
    @staticmethod
    def exact_match(predictions: List[str], gold_answers: List[str]) -> float:
        """
        Calculate Exact Match score for QA.
        
        Args:
            predictions: Predicted answers
            gold_answers: Gold standard answers
        """
        def normalize_answer(s: str) -> str:
            """Normalize answer for comparison."""
            s = s.lower()
            # Remove articles
            s = re.sub(r'\b(a|an|the)\b', ' ', s)
            # Remove punctuation
            s = re.sub(r'[^\w\s]', '', s)
            # Remove extra whitespace
            s = ' '.join(s.split())
            return s
        
        matches = [
            normalize_answer(pred) == normalize_answer(gold)
            for pred, gold in zip(predictions, gold_answers)
        ]
        return sum(matches) / len(matches) if matches else 0.0
    
    @staticmethod
    def f1_score_qa(prediction: str, gold_answer: str) -> float:
        """
        Calculate token-level F1 score for QA (single pair).
        """
        def tokenize(s: str) -> List[str]:
            return s.lower().split()
        
        pred_tokens = tokenize(prediction)
        gold_tokens = tokenize(gold_answer)
        
        if not pred_tokens or not gold_tokens:
            return 0.0
        
        common = Counter(pred_tokens) & Counter(gold_tokens)
        num_common = sum(common.values())
        
        if num_common == 0:
            return 0.0
        
        precision = num_common / len(pred_tokens)
        recall = num_common / len(gold_tokens)
        f1 = 2 * precision * recall / (precision + recall)
        
        return f1
    
    @staticmethod
    def mean_reciprocal_rank(
        predictions: List[List[str]],
        gold_answers: List[str]
    ) -> float:
        """
        Calculate Mean Reciprocal Rank (MRR).
        
        Args:
            predictions: List of ranked predictions for each query
            gold_answers: List of correct answers
        """
        reciprocal_ranks = []
        
        for preds, gold in zip(predictions, gold_answers):
            rank = None
            for i, pred in enumerate(preds, 1):
                if pred == gold:
                    rank = i
                    break
            
            if rank:
                reciprocal_ranks.append(1.0 / rank)
            else:
                reciprocal_ranks.append(0.0)
        
        return np.mean(reciprocal_ranks) if reciprocal_ranks else 0.0
    
    @staticmethod
    def hits_at_k(
        predictions: List[List[str]],
        gold_answers: List[str],
        k: int = 10
    ) -> float:
        """
        Calculate Hits@K metric.
        
        Args:
            predictions: List of ranked predictions (top-k) for each query
            gold_answers: List of correct answers
            k: Number of top predictions to consider
        """
        hits = [
            gold in preds[:k]
            for preds, gold in zip(predictions, gold_answers)
        ]
        return sum(hits) / len(hits) if hits else 0.0
    
    @staticmethod
    def precision_at_k(
        predictions: List[List[str]],
        gold_answers: List[List[str]],
        k: int = 10
    ) -> float:
        """
        Calculate Precision@K.
        
        Args:
            predictions: List of ranked predictions for each query
            gold_answers: List of relevant items for each query
            k: Number of top predictions to consider
        """
        precisions = []
        
        for preds, golds in zip(predictions, gold_answers):
            top_k = preds[:k]
            relevant_in_top_k = sum(pred in golds for pred in top_k)
            precisions.append(relevant_in_top_k / k if k > 0 else 0.0)
        
        return np.mean(precisions) if precisions else 0.0
    
    @staticmethod
    def recall_at_k(
        predictions: List[List[str]],
        gold_answers: List[List[str]],
        k: int = 10
    ) -> float:
        """
        Calculate Recall@K.
        
        Args:
            predictions: List of ranked predictions for each query
            gold_answers: List of relevant items for each query
            k: Number of top predictions to consider
        """
        recalls = []
        
        for preds, golds in zip(predictions, gold_answers):
            if not golds:
                continue
            
            top_k = preds[:k]
            relevant_in_top_k = sum(pred in golds for pred in top_k)
            recalls.append(relevant_in_top_k / len(golds))
        
        return np.mean(recalls) if recalls else 0.0
    
    @staticmethod
    def triple_accuracy(
        predicted_triples: List[Tuple[str, str, str]],
        gold_triples: List[Tuple[str, str, str]]
    ) -> Dict[str, float]:
        """
        Calculate triple extraction accuracy metrics.
        
        Returns:
            Dictionary with precision, recall, f1
        """
        pred_set = set(predicted_triples)
        gold_set = set(gold_triples)
        
        tp = len(pred_set & gold_set)
        fp = len(pred_set - gold_set)
        fn = len(gold_set - pred_set)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
        }
    
    @staticmethod
    def compute_all_metrics(
        predictions: List[Any],
        gold_labels: List[Any],
        metric_types: List[str]
    ) -> Dict[str, float]:
        """
        Compute all requested metrics.
        
        Args:
            predictions: List of predictions
            gold_labels: List of gold labels
            metric_types: List of metric names to compute
        
        Returns:
            Dictionary mapping metric names to values
        """
        results = {}
        calc = MetricsCalculator()
        
        for metric in metric_types:
            if metric == "accuracy":
                results["accuracy"] = calc.accuracy(predictions, gold_labels)
            
            elif metric in ["precision", "recall", "f1"]:
                prf = calc.precision_recall_f1(predictions, gold_labels)
                results.update(prf)
            
            elif metric == "exact_match":
                results["exact_match"] = calc.exact_match(predictions, gold_labels)
        
        return results
