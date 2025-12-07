"""
Simple HotpotQA Benchmark Test

Run this to test your GraphBuilder-RAG system on HotpotQA multi-hop questions.
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


# HotpotQA Test Samples
HOTPOTQA_SAMPLES = [
    # Bridge questions (2-hop reasoning)
    {
        "id": "bridge_1",
        "question": "What is the nationality of the director of the film Inception?",
        "answer": ["British-American", "British", "American"],
        "type": "bridge",
        "level": "easy"
    },
    {
        "id": "bridge_2",
        "question": "In what year was the university where Albert Einstein worked as a professor founded?",
        "answer": ["1834"],
        "type": "bridge",
        "level": "medium"
    },
    {
        "id": "bridge_3",
        "question": "Which programming language was developed by the creator of Python?",
        "answer": ["Python"],
        "type": "bridge",
        "level": "easy"
    },
    {
        "id": "bridge_4",
        "question": "What is the capital of the country where the Eiffel Tower is located?",
        "answer": ["Paris"],
        "type": "bridge",
        "level": "easy"
    },
    
    # Comparison questions (2-entity comparison)
    {
        "id": "comp_1",
        "question": "Which company was founded first, Apple or Microsoft?",
        "answer": ["Microsoft"],
        "type": "comparison",
        "level": "medium"
    },
    {
        "id": "comp_2",
        "question": "Who was born earlier, Isaac Newton or Galileo Galilei?",
        "answer": ["Galileo Galilei", "Galileo"],
        "type": "comparison",
        "level": "easy"
    },
    {
        "id": "comp_3",
        "question": "Which mountain is taller, Mount Everest or K2?",
        "answer": ["Mount Everest", "Everest"],
        "type": "comparison",
        "level": "easy"
    },
    {
        "id": "comp_4",
        "question": "Which planet is larger, Jupiter or Saturn?",
        "answer": ["Jupiter"],
        "type": "comparison",
        "level": "easy"
    },
    {
        "id": "comp_5",
        "question": "Which ocean is deeper, the Atlantic or the Pacific?",
        "answer": ["Pacific Ocean", "Pacific"],
        "type": "comparison",
        "level": "medium"
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


def run_hotpotqa_benchmark():
    """Run HotpotQA benchmark test"""
    print("="*80)
    print("HotpotQA Benchmark Test for GraphBuilder-RAG")
    print("="*80)
    print(f"Testing {len(HOTPOTQA_SAMPLES)} multi-hop questions...")
    print()
    
    results = []
    correct = 0
    total = 0
    
    # Track by question type
    type_stats = {"bridge": {"correct": 0, "total": 0}, 
                  "comparison": {"correct": 0, "total": 0}}
    
    for i, sample in enumerate(HOTPOTQA_SAMPLES, 1):
        print(f"\n[{i}/{len(HOTPOTQA_SAMPLES)}] Testing {sample['type']} question ({sample['level']})...")
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
        type_stats[sample['type']]['total'] += 1
        
        if is_correct:
            correct += 1
            type_stats[sample['type']]['correct'] += 1
        
        results.append({
            "id": sample['id'],
            "question": sample['question'],
            "predicted": predicted,
            "gold": sample['answer'],
            "correct": is_correct,
            "type": sample['type'],
            "level": sample['level'],
            "sources": response.get("sources", [])
        })
    
    # Print summary
    print("\n" + "="*80)
    print("BENCHMARK RESULTS")
    print("="*80)
    
    accuracy = (correct / total * 100) if total > 0 else 0
    print(f"\nOverall Accuracy: {correct}/{total} ({accuracy:.1f}%)")
    
    print("\nPer-Type Accuracy:")
    for qtype, stats in type_stats.items():
        if stats['total'] > 0:
            type_acc = (stats['correct'] / stats['total'] * 100)
            print(f"  {qtype}: {stats['correct']}/{stats['total']} ({type_acc:.1f}%)")
    
    print("\nPer-Question Results:")
    for result in results:
        status = "✓" if result['correct'] else "✗"
        print(f"  {status} [{result['type']}] {result['id']}: {result['question'][:60]}...")
    
    # Save results
    results_dir = Path(__file__).parent / "benchmarks" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    output_file = results_dir / "hotpotqa_results.json"
    with open(output_file, 'w') as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "total_questions": total,
            "correct": correct,
            "accuracy": accuracy,
            "type_stats": type_stats,
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
    run_hotpotqa_benchmark()
