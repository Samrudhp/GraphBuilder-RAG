#!/usr/bin/env python3
"""
Script 4: Semantic similarity evaluation using BGE embeddings
Fast, accurate fact verification using cosine similarity
"""

import json
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer

# Setup paths
script_dir = Path(__file__).parent
retrieved_contexts_dir = script_dir.parent / "retrieved_contexts"
evaluation_results_dir = script_dir.parent / "evaluation_results-2"

evaluation_results_dir.mkdir(exist_ok=True)

print("🚀 Loading BGE embedding model...")
model = SentenceTransformer("BAAI/bge-small-en-v1.5")
print("✅ Model loaded!\n")

def cosine_similarity(emb1, emb2):
    """Compute cosine similarity between two embeddings"""
    return float(np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2)))

def get_verdict(score):
    """Map similarity score to SUPPORTS/REFUTES/NOT ENOUGH INFO"""
    if score >= 0.82:  # Stricter: need very high similarity
        return "SUPPORTS", score
    elif score <= 0.25:  # Very low similarity = refute
        return "REFUTES", score
    else:  # Medium range = insufficient info
        return "NOT ENOUGH INFO", score

def fever_verdict(claim, chunks, similarities):
    """
    FEVER-specific verdict logic.
    Since retrieved chunks don't contain natural contradictions (they're synthetic),
    use a conservative approach:
    - SUPPORTS: Very high similarity
    - REFUTES/NOT ENOUGH INFO: Lower similarity (can't distinguish without real contradictions)
    """
    if not chunks or not similarities:
        return "NOT ENOUGH INFO", 0.3, "No evidence retrieved"
    
    max_sim = max(similarities)
    
    # Conservative FEVER logic for synthetic data:
    # - Only SUPPORTS on very high similarity (natural claim-evidence match)
    # - Everything else = NOT ENOUGH INFO (safer than guessing REFUTES)
    
    if max_sim >= 0.85:
        # Very high similarity = likely real support
        verdict = "SUPPORTS"
        confidence = min(0.92, max_sim)
        reasoning = f"Very strong evidence: {max_sim:.2f}"
    elif max_sim >= 0.75:
        # Medium-high = could be support
        verdict = "SUPPORTS"
        confidence = 0.70
        reasoning = f"Strong evidence: {max_sim:.2f}"
    else:
        # Lower similarity = insufficient evidence
        # (Cannot reliably detect REFUTES without actual contradictory text)
        verdict = "NOT ENOUGH INFO"
        confidence = 0.4
        reasoning = f"Insufficient evidence: {max_sim:.2f}"
    
    return verdict, confidence, reasoning

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
    
    # Get claim embedding
    claim_emb = model.encode(claim, convert_to_numpy=True)
    
    # Compute similarity with FAISS chunks
    max_similarity = 0.0
    best_chunk = ""
    similarities = []
    
    for chunk in faiss_chunks:
        chunk_emb = model.encode(chunk, convert_to_numpy=True)
        sim = cosine_similarity(claim_emb, chunk_emb)
        similarities.append(sim)
        if sim > max_similarity:
            max_similarity = sim
            best_chunk = chunk
    
    # ===== DATASET-SPECIFIC EVALUATION =====
    if dataset == "FEVER":
        # FEVER: Use contradiction-aware logic
        hybrid_verdict, hybrid_score, hybrid_reasoning = fever_verdict(claim, faiss_chunks, similarities)
        rag_verdict, rag_score, rag_reasoning = fever_verdict(claim, faiss_chunks, similarities)
    else:
        # HotpotQA: Use similarity-based logic
        # ===== HYBRID EVALUATION (Neo4j + FAISS) =====
        kg_has_support = len(neo4j_edges) > 0
        
        if kg_has_support and max_similarity >= 0.75:
            # Both sources support (KG + reasonable text similarity)
            hybrid_verdict = "SUPPORTS"
            hybrid_score = 0.92
            hybrid_reasoning = f"KG + Text alignment (sim={max_similarity:.2f})"
        elif max_similarity >= 0.82:
            # Very strong text support alone
            hybrid_verdict = "SUPPORTS"
            hybrid_score = 0.85
            hybrid_reasoning = f"Strong text similarity: {max_similarity:.2f}"
        elif max_similarity <= 0.25:
            # Low similarity - refute or insufficient
            if max_similarity <= 0.15:
                hybrid_verdict = "REFUTES"
                hybrid_score = 0.4
                hybrid_reasoning = f"Low similarity: {max_similarity:.2f}"
            else:
                hybrid_verdict = "NOT ENOUGH INFO"
                hybrid_score = 0.5
                hybrid_reasoning = f"Weak text evidence (sim={max_similarity:.2f})"
        else:
            # Medium similarity - insufficient info
            hybrid_verdict = "NOT ENOUGH INFO"
            hybrid_score = 0.6
            hybrid_reasoning = f"Moderate similarity (sim={max_similarity:.2f}), needs more evidence"
        
        # ===== RAG-ONLY EVALUATION (FAISS only) =====
        rag_verdict, rag_score = get_verdict(max_similarity)
        rag_reasoning = f"Semantic similarity: {max_similarity:.3f} | '{best_chunk[:50]}...'"
    
    # ===== KG-ONLY EVALUATION (Neo4j only) =====
    if neo4j_edges:
        kg_only_verdict = "SUPPORTS"
        kg_only_score = 0.85
        kg_only_reasoning = f"KG: {neo4j_edges[0].get('semantic_type', 'related')}"
    else:
        kg_only_verdict = "NOT ENOUGH INFO"
        kg_only_score = 0.3
        kg_only_reasoning = "No KG relationships"
    
    # Check correctness
    # Different logic for FEVER (label-based) vs HotpotQA (answer-based)
    if dataset == "FEVER":
        # FEVER: Compare verdict to label (SUPPORTS/REFUTES/NOT ENOUGH INFO)
        hybrid_correct = (hybrid_verdict == ground_truth)
        rag_correct = (rag_verdict == ground_truth)
        kg_correct = (kg_only_verdict == ground_truth)
    else:
        # HotpotQA: Check if answer appears in retrieved chunks
        answer_in_chunks = any(
            ground_truth.lower() in chunk.lower() 
            for chunk in faiss_chunks
        )
        # For Q&A: SUPPORTS = answer found, NOT ENOUGH INFO = not found
        expected_verdict = "SUPPORTS" if answer_in_chunks else "NOT ENOUGH INFO"
        hybrid_correct = (hybrid_verdict == expected_verdict)
        rag_correct = (rag_verdict == expected_verdict)
        kg_correct = (kg_only_verdict == expected_verdict)
    
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
            "similarity_scores": similarities[:5]
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

print(f"📊 Evaluating {total} queries with semantic similarity...")
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
print(f"   Method: Semantic Similarity (BGE embeddings) + Neo4j")
print(f"{'='*70}")
