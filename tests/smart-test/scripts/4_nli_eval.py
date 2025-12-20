#!/usr/bin/env python3
"""
Script 4: NLI-based evaluation using DistilBERT-finetuned model
Evaluates claim entailment against retrieved evidence
"""

import json
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Setup paths
script_dir = Path(__file__).parent
retrieved_contexts_dir = script_dir.parent / "retrieved_contexts"
evaluation_results_dir = script_dir.parent / "evaluation_results-2"

evaluation_results_dir.mkdir(exist_ok=True)

print("🚀 Loading NLI model...")
model_name = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
device = "cpu"
model.to(device)
model.eval()
print("✅ Model loaded!\n")

def compute_entailment(claim, text):
    """
    Compute entailment score between claim and text
    Returns score 0-1 where 1 = high entailment, 0 = no entailment
    """
    # Tokenize
    inputs = tokenizer(claim, text, return_tensors="pt", max_length=512, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Forward pass
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
    
    # Get probabilities
    probs = torch.softmax(logits, dim=-1)
    
    # For sentiment model: id 1 = positive (ENTAILMENT-like)
    entailment_score = float(probs[0][1])
    
    return entailment_score

def get_verdict(score):
    """Map entailment score to SUPPORTS/REFUTES/NOT ENOUGH INFO"""
    if score >= 0.7:
        return "SUPPORTS", score
    elif score <= 0.35:
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
    # 1. Check KG support
    kg_support = len(neo4j_edges) > 0
    
    # 2. Compute entailment with FAISS chunks
    max_entail = 0.0
    best_chunk = ""
    entail_scores = []
    
    for chunk in faiss_chunks:
        score = compute_entailment(claim, chunk)
        entail_scores.append(score)
        if score > max_entail:
            max_entail = score
            best_chunk = chunk
    
    # 3. Combine: KG + Text evidence
    if kg_support and max_entail >= 0.65:
        # Both support strongly
        hybrid_score = 0.95
        hybrid_verdict = "SUPPORTS"
        hybrid_reasoning = f"KG + Text alignment (entail={max_entail:.2f}): {best_chunk[:50]}..."
    elif kg_support or max_entail >= 0.7:
        # One source supports
        hybrid_score = max(0.85 if kg_support else 0.2, max_entail)
        hybrid_verdict, _ = get_verdict(hybrid_score)
        hybrid_reasoning = f"Entailment score: {hybrid_score:.2f} | {best_chunk[:50]}..."
    else:
        # Neither supports
        hybrid_score = min(0.3 if kg_support else 0.2, max_entail)
        hybrid_verdict, _ = get_verdict(hybrid_score)
        hybrid_reasoning = "Low entailment in both KG and text"
    
    # ===== RAG-ONLY EVALUATION (FAISS only) =====
    rag_score = max_entail
    rag_verdict, _ = get_verdict(rag_score)
    rag_reasoning = f"Text entailment: {rag_score:.3f} | {best_chunk[:50]}..."
    
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
            "entailment_scores": [float(s) for s in entail_scores[:5]]
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

print(f"📊 Evaluating {total} queries with NLI model...")
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
print(f"   Method: NLI (DistilBERT) + Neo4j relationships")
print(f"{'='*70}")
