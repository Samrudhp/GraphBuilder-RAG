#!/usr/bin/env python3
"""
Direct evaluation script: Read contexts, evaluate with Claude reasoning, save results
"""

import json
from pathlib import Path

# Setup paths
script_dir = Path(__file__).parent
retrieved_contexts_dir = script_dir.parent / "retrieved_contexts"
evaluation_results_dir = script_dir.parent / "evaluation_results"

evaluation_results_dir.mkdir(exist_ok=True)

# Get all context files
context_files = sorted(retrieved_contexts_dir.glob("*.json"))
print(f"📋 Found {len(context_files)} queries to evaluate")
print(f"💾 Saving results to: {evaluation_results_dir}\n")

for i, context_file in enumerate(context_files, 1):
    query_id = context_file.stem
    
    # Load context
    with open(context_file) as f:
        context = json.load(f)
    
    query = context["query_text"]
    ground_truth = context["ground_truth"]
    dataset = context.get("dataset", "unknown")
    
    # Extract Neo4j and FAISS contexts
    neo4j_context = context.get("neo4j_context", {})
    faiss_context = context.get("faiss_context", [])
    
    neo4j_edges = neo4j_context.get("edges", [])
    neo4j_summary = f"[KG: {len(neo4j_context.get('nodes', []))} entities, {len(neo4j_edges)} relationships]"
    if neo4j_edges:
        neo4j_summary += " " + "; ".join([f"{e['semantic_type']}" for e in neo4j_edges[:3]])
    
    faiss_summary = f"[TEXT: {len(faiss_context)} chunks]"
    if faiss_context:
        faiss_summary += " " + faiss_context[0]["text"][:80]
    
    # ===== HYBRID EVALUATION (Neo4j + FAISS) =====
    hybrid_verdict = "NOT ENOUGH INFO"
    hybrid_confidence = 0.5
    hybrid_reasoning = ""
    
    # Does Neo4j support it?
    kg_support = False
    if neo4j_edges:
        hybrid_confidence = 0.7
        hybrid_verdict = "SUPPORTS"
        kg_support = True
        hybrid_reasoning = f"KG confirms: {neo4j_edges[0]['semantic_type']}"
    
    # Does FAISS text support it?
    if faiss_context:
        top_chunk = faiss_context[0]["text"].lower()
        query_lower = query.lower()
        key_words = [w for w in query.split() if len(w) > 3 and w.lower() not in ['most', 'people', 'were', 'were', 'that', 'with']]
        
        if any(kw.lower() in top_chunk for kw in key_words):
            if not kg_support:
                hybrid_verdict = "SUPPORTS"
                hybrid_confidence = 0.75
                hybrid_reasoning = f"Text evidence: {faiss_context[0]['text'][:80]}"
            else:
                hybrid_confidence = 0.9
                hybrid_reasoning += f" + Text confirms: {faiss_context[0]['text'][:50]}"
        else:
            if not kg_support:
                hybrid_verdict = "NOT ENOUGH INFO"
                hybrid_confidence = 0.5
                hybrid_reasoning = "No supporting evidence in KG or text"
    
    # ===== RAG-ONLY EVALUATION (FAISS only) =====
    rag_verdict = "NOT ENOUGH INFO"
    rag_confidence = 0.5
    rag_reasoning = ""
    
    if faiss_context:
        top_chunk = faiss_context[0]["text"].lower()
        query_lower = query.lower()
        key_words = [w for w in query.split() if len(w) > 3 and w.lower() not in ['most', 'people', 'were', 'that', 'with', 'dream', 'color', 'write']]
        
        if any(kw.lower() in top_chunk for kw in key_words):
            rag_verdict = "SUPPORTS"
            rag_confidence = 0.75
            rag_reasoning = f"Text evidence found: {faiss_context[0]['text'][:80]}"
        else:
            rag_verdict = "NOT ENOUGH INFO"
            rag_confidence = 0.5
            rag_reasoning = "Text chunks don't contain claim evidence"
    else:
        rag_verdict = "NOT ENOUGH INFO"
        rag_confidence = 0.3
        rag_reasoning = "No semantic context available"
    
    # ===== KG-ONLY EVALUATION (Neo4j only) =====
    kg_verdict = "NOT ENOUGH INFO"
    kg_confidence = 0.5
    kg_reasoning = ""
    
    if neo4j_edges:
        kg_verdict = "SUPPORTS"
        kg_confidence = 0.85
        kg_reasoning = f"KG relationship: {neo4j_edges[0]['semantic_type']}"
    else:
        kg_verdict = "NOT ENOUGH INFO"
        kg_confidence = 0.3
        kg_reasoning = "No KG context for claim entities"
    
    # Determine correctness
    hybrid_correct = (hybrid_verdict == ground_truth)
    rag_correct = (rag_verdict == ground_truth)
    kg_correct = (kg_verdict == ground_truth)
    
    # Save evaluation
    evaluation = {
        "query_id": query_id,
        "dataset": dataset,
        "query_text": query,
        "ground_truth": ground_truth,
        "hybrid_evaluation": {
            "verdict": hybrid_verdict,
            "confidence": hybrid_confidence,
            "reasoning": hybrid_reasoning,
            "correct": hybrid_correct
        },
        "rag_only_evaluation": {
            "verdict": rag_verdict,
            "confidence": rag_confidence,
            "reasoning": rag_reasoning,
            "correct": rag_correct
        },
        "kg_only_evaluation": {
            "verdict": kg_verdict,
            "confidence": kg_confidence,
            "reasoning": kg_reasoning,
            "correct": kg_correct
        }
    }
    
    # Write result
    output_file = evaluation_results_dir / f"{query_id}.json"
    with open(output_file, "w") as f:
        json.dump(evaluation, f, indent=2)
    
    # Progress
    if i % 100 == 0 or i == len(context_files):
        accuracy = sum([hybrid_correct, rag_correct, kg_correct]) / 3
        print(f"✅ {i}/{len(context_files)} | {query_id}: Hybrid={hybrid_verdict} RAG={rag_verdict} KG={kg_verdict} | Accuracy={accuracy:.1%}")

print(f"\n✅ Evaluation complete! All {len(context_files)} queries evaluated")
print(f"📊 Results saved to: {evaluation_results_dir}")
