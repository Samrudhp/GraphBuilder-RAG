"""
Test natural language querying with GraphBuilder-RAG

This demonstrates the complete RAG pipeline:
1. Natural language question
2. Hybrid retrieval (FAISS + Neo4j graph)
3. LLM answer generation
4. GraphVerify hallucination detection
"""
import requests
import json

API_URL = "http://localhost:8000/api/v1/query"

def ask_question(question: str, use_graphverify: bool = True):
    """Ask a question and get an answer with verification."""
    print(f"\n{'='*80}")
    print(f"QUESTION: {question}")
    print(f"{'='*80}\n")
    
    payload = {
        "question": question,
        "max_chunks": 5,
        "graph_depth": 2,
        "use_graphverify": use_graphverify
    }
    
    response = requests.post(API_URL, json=payload)
    
    if response.status_code == 200:
        result = response.json()
        
        # Print answer
        print("ANSWER:")
        print(result['answer'])
        print()
        
        # Print verification status
        print(f"Verification Status: {result['verification_status']}")
        print(f"Verification Score: {result.get('verification_score', 'N/A')}")
        print()
        
        # Print sources
        if result.get('sources'):
            print("SOURCES:")
            for source in result['sources']:
                print(f"  - {source}")
            print()
        
        # Print retrieval context
        if result.get('retrieval_context'):
            ctx = result['retrieval_context']
            print(f"Retrieved: {ctx.get('total_chunks', 0)} chunks, {ctx.get('graph_matches', 0)} graph matches")
            print()
        
        # Print verification details if available
        if result.get('verification_details'):
            print("VERIFICATION DETAILS:")
            print(json.dumps(result['verification_details'], indent=2))
        
    else:
        print(f"Error: {response.status_code}")
        print(response.text)
    
    print(f"\n{'='*80}\n")

if __name__ == "__main__":
    print("\n🤖 GraphBuilder-RAG - Natural Language Query\n")
    print("Ask questions about your documents. Type 'exit' to quit.\n")
    print("="*80)
    
    while True:
        # Get question from user
        question = input("\nYour question: ").strip()
        
        if not question:
            print("Please enter a question.")
            continue
        
        if question.lower() in ['exit', 'quit', 'q']:
            print("\nGoodbye! 👋\n")
            break
        
        # Ask the question
        ask_question(question, use_graphverify=True)
