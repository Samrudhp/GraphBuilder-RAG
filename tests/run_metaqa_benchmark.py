"""
Simple MetaQA Benchmark Test

Run this to test your GraphBuilder-RAG system on MetaQA-style questions.
"""
import asyncio
import json
import sys
from pathlib import Path
from datetime import datetime

# Add project root
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import requests

API_URL = "http://localhost:8000"


# MetaQA Test Samples
METAQA_SAMPLES = [
    # 1-hop questions
    {
        "id": "1hop_1",
        "question": "Who directed Inception?",
        "answer": ["Christopher Nolan"],
        "hops": 1,
        "type": "directed_by"
    },
    {
        "id": "1hop_2",
        "question": "Who wrote Harry Potter?",
        "answer": ["J.K. Rowling"],
        "hops": 1,
        "type": "written_by"
    },
    {
        "id": "1hop_3",
        "question": "When was Albert Einstein born?",
        "answer": ["March 14, 1879", "1879"],
        "hops": 1,
        "type": "birth_date"
    },
    {
        "id": "1hop_4",
        "question": "What did Isaac Newton publish?",
        "answer": ["Principia Mathematica"],
        "hops": 1,
        "type": "published_work"
    },
    {
        "id": "1hop_5",
        "question": "Who won the Nobel Prize in Physics in 1921?",
        "answer": ["Albert Einstein"],
        "hops": 1,
        "type": "award_winner"
    },
    
    # 2-hop questions
    {
        "id": "2hop_1",
        "question": "What did the person who developed the theory of relativity win?",
        "answer": ["Nobel Prize in Physics"],
        "hops": 2,
        "type": "physicist_award"
    },
    {
        "id": "2hop_2",
        "question": "What is the occupation of the person who published Principia Mathematica?",
        "answer": ["mathematician", "physicist"],
        "hops": 2,
        "type": "author_occupation"
    },
    {
        "id": "2hop_3",
        "question": "Who was the chemist that won the Nobel Prize?",
        "answer": ["Marie Curie"],
        "hops": 2,
        "type": "occupation_award"
    },
    
    # 3-hop questions
    {
        "id": "3hop_1",
        "question": "What field did the physicist who won the Nobel Prize in 1921 work in?",
        "answer": ["theoretical physics", "physics"],
        "hops": 3,
        "type": "winner_physicist_field"
    },
]


def query_system(question: str) -> dict:
    """Query the GraphBuilder-RAG API"""
    try:
        response = requests.post(
            f"{API_URL}/api/v1/query",
            json={
                "question": question,
                "max_chunks": 5,
                "require_verification": False
            },
            timeout=30
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"HTTP {response.status_code}"}
    except Exception as e:
        return {"error": str(e)}


def extract_answer(response: dict) -> str:
    """Extract answer text from API response"""
    if "error" in response:
        return ""
    
    answer_obj = response.get("answer", {})
    if isinstance(answer_obj, dict):
        return answer_obj.get("answer", "").lower().strip()
    elif isinstance(answer_obj, str):
        return answer_obj.lower().strip()
    return ""


def check_answer(predicted: str, gold_answers: list) -> bool:
    """Check if predicted answer matches any gold answer"""
    pred = predicted.lower().strip()
    for gold in gold_answers:
        if gold.lower().strip() in pred or pred in gold.lower().strip():
            return True
    return False


def run_metaqa_benchmark():
    """Run MetaQA benchmark test"""
    print("="*80)
    print("MetaQA Benchmark Test for GraphBuilder-RAG")
    print("="*80)
    print(f"Testing {len(METAQA_SAMPLES)} questions...")
    print()
    
    results = []
    correct = 0
    total = 0
    
    # Track by hop level
    hop_stats = {1: {"correct": 0, "total": 0}, 
                 2: {"correct": 0, "total": 0}, 
                 3: {"correct": 0, "total": 0}}
    
    for i, sample in enumerate(METAQA_SAMPLES, 1):
        print(f"\n[{i}/{len(METAQA_SAMPLES)}] Testing {sample['hops']}-hop question...")
        print(f"Q: {sample['question']}")
        print(f"Expected: {', '.join(sample['answer'])}")
        
        # Query the system
        response = query_system(sample['question'])
        predicted = extract_answer(response)
        
        # Check correctness
        is_correct = check_answer(predicted, sample['answer'])
        
        print(f"Predicted: {predicted}")
        print(f"Result: {'✓ CORRECT' if is_correct else '✗ INCORRECT'}")
        
        # Track stats
        total += 1
        hop_stats[sample['hops']]['total'] += 1
        
        if is_correct:
            correct += 1
            hop_stats[sample['hops']]['correct'] += 1
        
        results.append({
            "id": sample['id'],
            "question": sample['question'],
            "predicted": predicted,
            "gold": sample['answer'],
            "correct": is_correct,
            "hops": sample['hops'],
            "type": sample['type'],
            "sources": response.get("sources", [])
        })
    
    # Print summary
    print("\n" + "="*80)
    print("BENCHMARK RESULTS")
    print("="*80)
    
    accuracy = (correct / total * 100) if total > 0 else 0
    print(f"\nOverall Accuracy: {correct}/{total} ({accuracy:.1f}%)")
    
    print("\nPer-Hop Accuracy:")
    for hop, stats in hop_stats.items():
        if stats['total'] > 0:
            hop_acc = (stats['correct'] / stats['total'] * 100)
            print(f"  {hop}-hop: {stats['correct']}/{stats['total']} ({hop_acc:.1f}%)")
    
    print("\nPer-Question Results:")
    for result in results:
        status = "✓" if result['correct'] else "✗"
        print(f"  {status} [{result['hops']}-hop] {result['id']}: {result['question'][:50]}...")
    
    # Save results
    results_dir = Path(__file__).parent / "benchmarks" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    output_file = results_dir / "metaqa_results.json"
    with open(output_file, 'w') as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "total_questions": total,
            "correct": correct,
            "accuracy": accuracy,
            "hop_stats": hop_stats,
            "results": results
        }, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")
    print("="*80)
    
    return results


if __name__ == "__main__":
    # Check API health
    try:
        health = requests.get(f"{API_URL}/health", timeout=2)
        if health.status_code != 200:
            print("ERROR: API is not responding properly")
            sys.exit(1)
    except:
        print("ERROR: API is not running at", API_URL)
        print("Start the API with: python api/main.py")
        sys.exit(1)
    
    # Run benchmark
    run_metaqa_benchmark()
