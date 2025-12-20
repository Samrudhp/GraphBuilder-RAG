#!/usr/bin/env python3
"""
Script 4 (v2): Semantic similarity evaluation using embeddings
Uses FAISS embeddings + MongoDB to compute claim-evidence similarity
"""

import json
from pathlib import Path
import numpy as np
import sys

# Setup paths
script_dir = Path(__file__).parent
project_root = script_dir.parent.parent.parent
retrieved_contexts_dir = script_dir.parent / "retrieved_contexts"
evaluation_results_dir = script_dir.parent / "evaluation_results-2"

# Add shared to path
sys.path.insert(0, str(project_root))

from shared.vector_database.faiss_client import FAISSClient
from shared.database.mongodb import MongoDB

evaluation_results_dir.mkdir(exist_ok=True)

print("🚀 Initializing semantic similarity evaluator...")
print("   Using FAISS embeddings for semantic matching\n")

# Initialize clients
faiss_client = FAISSClient(index_path=script_dir / "data/smart_test_faiss")
mongo = MongoDB(db_name="information_retrieval")

def get_embedding(text):
    """Get embedding for text using same model as ingestion"""
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("BAAI/bge-small-en-v1.5")
    return model.encode(text, convert_to_numpy=True)

def compute_similarity(emb1, emb2):
    """Cosine similarity"""
    return float(np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2)))

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
    claim_emb = get_embedding(claim)
    
    # Compute max similarity with FAISS chunks
    max_similarity = 0.0
    best_chunk = ""
    similarities = []
    
    for chunk in faiss_chunks:
        chunk_emb = get_embedding(chunk)
        sim = compute_similarity(claim_emb, chunk_emb)
        similarities.append(sim)
        if sim > max_similarity:
            max_similarity = sim
            best_chunk = chunk
    
    # Convert similarity to verdict
    # High similarity (>0.7) = SUPPORTS
    # Low similarity (<0.4) = REFUTES  
    # Mid range = NOT ENOUGH INFO
    
    def score_to_verdict(score):
        if score >= 0.7:
            return "SUPPORTS", score
        elif score <= 0.4:
            return "REFUTES", score
        else:
            return "NOT ENOUGH INFO", score
    
    # ===== HYBRID EVALUATION (Neo4j + FAISS) =====
    if neo4j_edges and max_similarity >= 0.65:
        # Both sources support
        hybrid_verdict = "SUPPORTS"
        hybrid_score = 0.95
        hybrid_reasoning = f"KG + Text alignment: {best_chunk[:50]}..."
    elif neo4j_edges or max_similarity >= 0.7:
        # One source supports strongly
        hybrid_score = max(0.85 if neo4j_edges else 0.3, max_similarity)
        hybrid_verdict, _ = score_to_verdict(hybrid_score)
        hybrid_reasoning = f"Text similarity: {hybrid_score:.2f} | {best_chunk[:50]}..."
    else:
        # Neither supports
        hybrid_score = min(0.3 if neo4j_edges else 0.2, max_similarity)
        hybrid_verdict, _ = score_to_verdict(hybrid_score)
        hybrid_reasoning = "Weak alignment in both KG and text"
    
    # ===== RAG-ONLY EVALUATION (FAISS only) =====
    rag_verdict, rag_score = score_to_verdict(max_similarity)
    rag_reasoning = f"Text similarity: {rag_score:.3f} | Chunk: {best_chunk[:50]}..."
    
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
    try:
        result = evaluate_query(query_id)
        
        if result:
            successful += 1
            # Progress every 50
            if i % 50 == 0 or i == total:
                h_correct = result["hybrid_evaluation"]["correct"]
                r_correct = result["rag_only_evaluation"]["correct"]
                k_correct = result["kg_only_evaluation"]["correct"]
                accuracy = sum([h_correct, r_correct, k_correct]) / 3
                print(f"✅ {i}/{total} | {query_id}: H={result['hybrid_evaluation']['verdict']} R={result['rag_only_evaluation']['verdict']} K={result['kg_only_evaluation']['verdict']} (acc={accuracy:.0%})")
    except Exception as e:
        print(f"⚠️  Error evaluating {query_id}: {str(e)[:50]}")
        continue

print(f"\n{'='*70}")
print(f"✅ EVALUATION COMPLETE!")
print(f"   Evaluated: {successful}/{total} queries")
print(f"   Results saved to: {evaluation_results_dir}")
print(f"   Method: Semantic Similarity (BGE embeddings) + Neo4j")
print(f"{'='*70}")
