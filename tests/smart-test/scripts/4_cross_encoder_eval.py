#!/usr/bin/env python3
"""
Script 4 (v2): Cross-encoder based evaluation - SOTA fact verification
Uses sentence-transformers cross-encoder for semantic textual entailment
"""

import json
from pathlib import Path
import numpy as np
from sentence_transformers import CrossEncoder

# Setup paths
script_dir = Path(__file__).parent
retrieved_contexts_dir = script_dir.parent / "retrieved_contexts"
evaluation_results_dir = script_dir.parent / "evaluation_results-2"

evaluation_results_dir.mkdir(exist_ok=True)

print("🚀 Loading cross-encoder model...")
# Model trained on entailment (does evidence entail the claim?)
try:
    model = CrossEncoder("cross-encoder/qnli-distilroberta-base", device="cpu")
except:
    print("⚠️  Using lighter model variant...")
    model = CrossEncoder("cross-encoder/mmarco-MiniLMv2-L12-H384-v1", device="cpu")
print("✅ Model loaded!\n")

def evaluate_with_cross_encoder(claim, chunks):
    """
    Score claim against chunks using cross-encoder
    Returns: max_score, best_chunk, all_scores
    """
    if not chunks:
        return 0.0, "", []
    
    # Prepare pairs: (claim, chunk)
    pairs = [[claim, chunk] for chunk in chunks]
    
    # Get scores (0-3 scale, we normalize to 0-1)
    scores = model.predict(pairs)
    
    # Normalize to 0-1 (cross-encoder outputs 0-3)
    normalized_scores = scores / 3.0
    
    max_idx = np.argmax(normalized_scores)
    max_score = float(normalized_scores[max_idx])
    best_chunk = chunks[max_idx]
    
    return max_score, best_chunk, [float(s) for s in normalized_scores]

def get_verdict_from_score(score):
    """
    Map cross-encoder score to SUPPORTS/REFUTES/NOT ENOUGH INFO
    """
    if score >= 0.7:
        return "SUPPORTS", score
    elif score <= 0.3:
        return "REFUTES", score
    else:
        return "NOT ENOUGH INFO", score

def evaluate_query(query_id):
    """Evaluate single query all 3 ways"""
    
    # Load context
    context_file = retrieved_contexts_dir / f"{query_id}.json"
    if not context_file.exists():
        return None
    
    with open(context_file) as f:
        context = json.load(f)
    
    claim = context["query_text"]
    ground_truth = context["ground_truth"]
    dataset = context.get("dataset", "unknown")
    
    # Extract contexts
    neo4j_context = context.get("neo4j_context", {})
    faiss_context = context.get("faiss_context", [])
    
    neo4j_edges = neo4j_context.get("edges", [])
    faiss_chunks = [chunk["text"] for chunk in faiss_context]
    
    # ===== HYBRID EVALUATION (Neo4j + FAISS) =====
    # 1. Check if Neo4j has supporting relationship
    kg_score = 0.0
    kg_match = False
    if neo4j_edges:
        kg_match = True
        kg_score = 0.9  # KG relationship found
    
    # 2. Use cross-encoder on FAISS chunks
    faiss_score, best_chunk, faiss_scores = evaluate_with_cross_encoder(claim, faiss_chunks)
    
    # 3. Combine: if both sources agree strongly, boost confidence
    if kg_match and faiss_score >= 0.7:
        # Both KG and text support
        hybrid_score = 0.95
        hybrid_verdict = "SUPPORTS"
        hybrid_reasoning = f"KG relationship + Text entailment: {best_chunk[:60]}..."
    elif kg_match or faiss_score >= 0.7:
        # One source supports strongly
        hybrid_score = max(kg_score if kg_match else 0, faiss_score)
        hybrid_verdict, _ = get_verdict_from_score(hybrid_score)
        if kg_match:
            hybrid_reasoning = f"KG confirms relationship"
        else:
            hybrid_reasoning = f"Text entailment: {best_chunk[:60]}..."
    else:
        # Neither source supports
        hybrid_score = min(kg_score if kg_match else 0.3, faiss_score)
        hybrid_verdict, _ = get_verdict_from_score(hybrid_score)
        hybrid_reasoning = "Insufficient evidence in both KG and text"
    
    # ===== RAG-ONLY EVALUATION (FAISS only) =====
    rag_score = faiss_score
    rag_verdict, _ = get_verdict_from_score(rag_score)
    rag_reasoning = f"Text entailment score: {rag_score:.2f} | Chunk: {best_chunk[:60]}..."
    
    # ===== KG-ONLY EVALUATION (Neo4j only) =====
    if neo4j_edges:
        kg_only_verdict = "SUPPORTS"
        kg_only_score = 0.85
        kg_only_reasoning = f"KG relationship found: {neo4j_edges[0].get('semantic_type', 'related')}"
    else:
        kg_only_verdict = "NOT ENOUGH INFO"
        kg_only_score = 0.3
        kg_only_reasoning = "No supporting relationship in knowledge graph"
    
    # Check correctness
    hybrid_correct = (hybrid_verdict == ground_truth)
    rag_correct = (rag_verdict == ground_truth)
    kg_correct = (kg_only_verdict == ground_truth)
    
    # Save evaluation
    evaluation = {
        "query_id": query_id,
        "dataset": dataset,
        "query_text": claim,
        "ground_truth": ground_truth,
        "hybrid_evaluation": {
            "verdict": hybrid_verdict,
            "confidence": min(hybrid_score, 1.0),
            "reasoning": hybrid_reasoning,
            "correct": hybrid_correct
        },
        "rag_only_evaluation": {
            "verdict": rag_verdict,
            "confidence": min(rag_score, 1.0),
            "reasoning": rag_reasoning,
            "correct": rag_correct,
            "cross_encoder_scores": faiss_scores[:5]
        },
        "kg_only_evaluation": {
            "verdict": kg_only_verdict,
            "confidence": kg_only_score,
            "reasoning": kg_only_reasoning,
            "correct": kg_correct
        }
    }
    
    # Save result
    output_file = evaluation_results_dir / f"{query_id}.json"
    with open(output_file, "w") as f:
        json.dump(evaluation, f, indent=2)
    
    return evaluation

# Main evaluation loop
context_files = sorted(retrieved_contexts_dir.glob("*.json"))
total = len(context_files)

print(f"📊 Evaluating {total} queries with cross-encoder...")
print(f"💾 Saving to: {evaluation_results_dir}\n")

successful = 0
for i, context_file in enumerate(context_files, 1):
    query_id = context_file.stem
    result = evaluate_query(query_id)
    
    if result:
        successful += 1
        # Progress every 100
        if i % 100 == 0 or i == total:
            h_correct = result["hybrid_evaluation"]["correct"]
            r_correct = result["rag_only_evaluation"]["correct"]
            k_correct = result["kg_only_evaluation"]["correct"]
            accuracy = sum([h_correct, r_correct, k_correct]) / 3
            print(f"✅ {i}/{total} | {query_id}: H={result['hybrid_evaluation']['verdict']} R={result['rag_only_evaluation']['verdict']} K={result['kg_only_evaluation']['verdict']} (acc={accuracy:.0%})")

print(f"\n{'='*70}")
print(f"✅ EVALUATION COMPLETE!")
print(f"   Evaluated: {successful}/{total} queries")
print(f"   Results saved to: {evaluation_results_dir}")
print(f"   Method: Cross-Encoder (QNLI) + Neo4j relationships")
print(f"{'='*70}")
