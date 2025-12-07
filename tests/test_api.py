#!/usr/bin/env python3
"""Test script for GraphBuilder-RAG API"""
import requests
import json
import time

BASE_URL = "http://localhost:8000"

def test_health():
    """Test health endpoint"""
    print("\n🔍 Testing Health Endpoint...")
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status: {response.status_code}")
    print(json.dumps(response.json(), indent=2))
    return response.status_code == 200

def test_ingest():
    """Test document ingestion"""
    print("\n📄 Testing Document Ingestion...")
    data = {
        "source": "https://en.wikipedia.org/wiki/Artificial_intelligence",
        "source_type": "HTML",
        "metadata": {
            "topic": "AI",
            "test": "true"
        }
    }
    response = requests.post(f"{BASE_URL}/api/v1/ingest", json=data)
    print(f"Status: {response.status_code}")
    result = response.json()
    print(json.dumps(result, indent=2))
    return result.get("document_id")

def test_query(question: str):
    """Test query endpoint"""
    print(f"\n❓ Testing Query: {question}")
    data = {
        "question": question,
        "use_graph": True,
        "max_results": 5
    }
    response = requests.post(f"{BASE_URL}/api/v1/query", json=data)
    print(f"Status: {response.status_code}")
    result = response.json()
    print(json.dumps(result, indent=2))
    return result

def test_stats():
    """Test stats endpoint"""
    print("\n📊 Testing Stats Endpoint...")
    response = requests.get(f"{BASE_URL}/api/v1/stats")
    print(f"Status: {response.status_code}")
    print(json.dumps(response.json(), indent=2))

if __name__ == "__main__":
    print("=" * 60)
    print("🚀 GraphBuilder-RAG API Test Suite")
    print("=" * 60)
    
    # Test 1: Health Check
    if not test_health():
        print("❌ Health check failed! Is the API running?")
        exit(1)
    
    # Test 2: Stats
    test_stats()
    
    # Test 3: Ingest Document (optional - uncomment to test)
    # doc_id = test_ingest()
    # if doc_id:
    #     print(f"\n✅ Document ingested: {doc_id}")
    #     print("⏳ Waiting for processing (this may take a minute)...")
    #     time.sleep(60)
    
    # Test 4: Query (will work if you have data in the system)
    test_query("What is artificial intelligence?")
    
    print("\n" + "=" * 60)
    print("✅ Test suite completed!")
    print("=" * 60)
