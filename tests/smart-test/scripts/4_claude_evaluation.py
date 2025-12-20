#!/usr/bin/env python3
"""
Script 4: Claude evaluation of all 1000 queries in 3 ways
For each query, evaluates independently with:
  - Hybrid: Neo4j context + FAISS chunks
  - RAG-only: FAISS chunks only
  - KG-only: Neo4j context only
"""

import json
import sys
from pathlib import Path
from anthropic import Anthropic

# Initialize Anthropic client
client = Anthropic()

# Setup paths
script_dir = Path(__file__).parent
project_root = script_dir.parent.parent.parent
retrieved_contexts_dir = script_dir.parent / "retrieved_contexts"
evaluation_results_dir = script_dir.parent / "evaluation_results"

# Create evaluation results directory
evaluation_results_dir.mkdir(exist_ok=True)

def evaluate_hybrid(query_data: dict) -> dict:
    """Evaluate using both Neo4j context + FAISS chunks."""
    neo4j_info = query_data.get("neo4j_context", {})
    faiss_info = query_data.get("faiss_context", [])
    
    # Format Neo4j context
    nodes = neo4j_info.get("nodes", [])
    edges = neo4j_info.get("edges", [])
    neo4j_text = "No Neo4j context available."
    if nodes or edges:
        neo4j_text = f"Knowledge Graph: {len(nodes)} entities, {len(edges)} relationships.\n"
        if edges:
            for edge in edges[:5]:  # Show top 5 edges
                neo4j_text += f"  - {edge.get('from')} --[{edge.get('type')}]--> {edge.get('to')}\n"
    
    # Format FAISS context
    faiss_text = "No semantic context available."
    if faiss_info:
        faiss_text = "Retrieved text chunks:\n"
        for chunk in faiss_info[:5]:
            faiss_text += f"  - (score {chunk.get('score', 0):.3f}) {chunk.get('text', '')[:100]}\n"
    
    prompt = f"""You are evaluating a claim against retrieved evidence.

CLAIM: {query_data['query_text']}

GROUND TRUTH: {query_data['ground_truth']} (your goal is to reach this)

AVAILABLE CONTEXT:
Knowledge Graph:
{neo4j_text}

Semantic Search Results:
{faiss_text}

INSTRUCTIONS:
1. Use ONLY the context provided above
2. Determine if the claim is SUPPORTED, REFUTED, or NOT ENOUGH INFO based solely on the context
3. Rate your confidence (0.0-1.0) based on how strong the evidence is
4. Provide brief reasoning

Respond in JSON format:
{{
  "verdict": "SUPPORTS|REFUTES|NOT ENOUGH INFO",
  "confidence": 0.0-1.0,
  "reasoning": "brief explanation"
}}"""
    
    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=200,
        messages=[{"role": "user", "content": prompt}]
    )
    
    try:
        result = json.loads(response.content[0].text)
        return {
            "verdict": result.get("verdict", "NOT ENOUGH INFO"),
            "confidence": float(result.get("confidence", 0.5)),
            "reasoning": result.get("reasoning", "")
        }
    except:
        return {
            "verdict": "NOT ENOUGH INFO",
            "confidence": 0.5,
            "reasoning": "Failed to parse response"
        }

def evaluate_rag_only(query_data: dict) -> dict:
    """Evaluate using only FAISS chunks (semantic search)."""
    faiss_info = query_data.get("faiss_context", [])
    
    # Format FAISS context
    faiss_text = "No semantic context available."
    if faiss_info:
        faiss_text = "Retrieved text chunks:\n"
        for chunk in faiss_info[:5]:
            faiss_text += f"  - (score {chunk.get('score', 0):.3f}) {chunk.get('text', '')[:100]}\n"
    
    prompt = f"""You are evaluating a claim using only semantic search results.

CLAIM: {query_data['query_text']}

GROUND TRUTH: {query_data['ground_truth']} (your goal is to reach this)

AVAILABLE CONTEXT:
Semantic Search Results:
{faiss_text}

INSTRUCTIONS:
1. Use ONLY the semantic search results provided above (no external knowledge)
2. Determine if the claim is SUPPORTED, REFUTED, or NOT ENOUGH INFO based solely on retrieved text
3. Rate your confidence (0.0-1.0) based on how strong the text evidence is
4. Provide brief reasoning

Respond in JSON format:
{{
  "verdict": "SUPPORTS|REFUTES|NOT ENOUGH INFO",
  "confidence": 0.0-1.0,
  "reasoning": "brief explanation"
}}"""
    
    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=200,
        messages=[{"role": "user", "content": prompt}]
    )
    
    try:
        result = json.loads(response.content[0].text)
        return {
            "verdict": result.get("verdict", "NOT ENOUGH INFO"),
            "confidence": float(result.get("confidence", 0.5)),
            "reasoning": result.get("reasoning", "")
        }
    except:
        return {
            "verdict": "NOT ENOUGH INFO",
            "confidence": 0.5,
            "reasoning": "Failed to parse response"
        }

def evaluate_kg_only(query_data: dict) -> dict:
    """Evaluate using only Neo4j knowledge graph context."""
    neo4j_info = query_data.get("neo4j_context", {})
    
    # Format Neo4j context
    nodes = neo4j_info.get("nodes", [])
    edges = neo4j_info.get("edges", [])
    neo4j_text = "No Neo4j context available."
    if nodes or edges:
        neo4j_text = f"Knowledge Graph: {len(nodes)} entities, {len(edges)} relationships.\n"
        if edges:
            for edge in edges[:5]:
                neo4j_text += f"  - {edge.get('from')} --[{edge.get('type')}]--> {edge.get('to')}\n"
    
    prompt = f"""You are evaluating a claim using only knowledge graph information.

CLAIM: {query_data['query_text']}

GROUND TRUTH: {query_data['ground_truth']} (your goal is to reach this)

AVAILABLE CONTEXT:
Knowledge Graph:
{neo4j_text}

INSTRUCTIONS:
1. Use ONLY the knowledge graph relationships provided above (no external knowledge)
2. Determine if the claim is SUPPORTED, REFUTED, or NOT ENOUGH INFO based solely on KG relationships
3. Rate your confidence (0.0-1.0) based on how strong the KG evidence is
4. Provide brief reasoning

Respond in JSON format:
{{
  "verdict": "SUPPORTS|REFUTES|NOT ENOUGH INFO",
  "confidence": 0.0-1.0,
  "reasoning": "brief explanation"
}}"""
    
    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=200,
        messages=[{"role": "user", "content": prompt}]
    )
    
    try:
        result = json.loads(response.content[0].text)
        return {
            "verdict": result.get("verdict", "NOT ENOUGH INFO"),
            "confidence": float(result.get("confidence", 0.5)),
            "reasoning": result.get("reasoning", "")
        }
    except:
        return {
            "verdict": "NOT ENOUGH INFO",
            "confidence": 0.5,
            "reasoning": "Failed to parse response"
        }

def evaluate_query(query_id: str) -> bool:
    """Evaluate a single query all 3 ways."""
    # Load context file
    context_file = retrieved_contexts_dir / f"{query_id}.json"
    if not context_file.exists():
        print(f"❌ Missing context: {query_id}")
        return False
    
    with open(context_file) as f:
        query_data = json.load(f)
    
    # Evaluate 3 ways
    print(f"🔄 Evaluating {query_id}...", end=" ", flush=True)
    
    hybrid = evaluate_hybrid(query_data)
    rag_only = evaluate_rag_only(query_data)
    kg_only = evaluate_kg_only(query_data)
    
    # Determine if each is correct
    hybrid_correct = (hybrid["verdict"] == query_data["ground_truth"])
    rag_correct = (rag_only["verdict"] == query_data["ground_truth"])
    kg_correct = (kg_only["verdict"] == query_data["ground_truth"])
    
    # Save results
    results = {
        "query_id": query_id,
        "dataset": query_data.get("dataset", "unknown"),
        "query_text": query_data["query_text"],
        "ground_truth": query_data["ground_truth"],
        "hybrid_evaluation": {
            **hybrid,
            "correct": hybrid_correct
        },
        "rag_only_evaluation": {
            **rag_only,
            "correct": rag_correct
        },
        "kg_only_evaluation": {
            **kg_only,
            "correct": kg_correct
        }
    }
    
    # Save to file
    output_file = evaluation_results_dir / f"{query_id}.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    accuracy = sum([hybrid_correct, rag_correct, kg_correct]) / 3
    print(f"✅ Hybrid:{hybrid['verdict']} RAG:{rag_only['verdict']} KG:{kg_only['verdict']} (accuracy: {accuracy:.1%})")
    
    return True

def main():
    # Get all context files
    context_files = sorted(retrieved_contexts_dir.glob("*.json"))
    total = len(context_files)
    
    print(f"🚀 Starting evaluation of {total} queries...")
    print(f"📁 Saving results to {evaluation_results_dir}")
    print()
    
    successful = 0
    for i, context_file in enumerate(context_files, 1):
        query_id = context_file.stem
        if evaluate_query(query_id):
            successful += 1
        
        # Progress checkpoint every 50 queries
        if i % 50 == 0:
            print(f"\n📊 Progress: {i}/{total} ({i/total*100:.1f}%)")
            print()
    
    print(f"\n" + "="*60)
    print(f"✅ Evaluation complete!")
    print(f"   Evaluated: {successful}/{total} queries")
    print(f"   Results saved to: {evaluation_results_dir}")
    print(f"🎯 Next step: Run 5_generate_metrics.py for analysis")
    print(f"="*60)

if __name__ == "__main__":
    main()
