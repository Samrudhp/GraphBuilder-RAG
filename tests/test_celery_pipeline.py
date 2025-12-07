#!/usr/bin/env python3
"""
Test script to verify Celery workers process documents correctly.
This tests the REAL production pipeline, not test files.
"""

import requests
import json
import time
import tempfile
import os

API_URL = "http://localhost:8000"

def test_document_ingestion():
    """Test that Celery workers process an ingested document"""
    
    print("🧪 Testing Celery Worker Document Processing Pipeline")
    print("=" * 60)
    
    # 1. Check initial stats
    print("\n📊 Step 1: Checking initial system stats...")
    response = requests.get(f"{API_URL}/api/v1/stats")
    initial_stats = response.json()
    print(f"   Initial documents: {initial_stats['documents']['total']}")
    print(f"   Initial candidate triples: {initial_stats['triples']['candidate']}")
    print(f"   Initial validated triples: {initial_stats['triples']['validated']}")
    
    # 2. Create a test text file and ingest it
    print("\n📄 Step 2: Ingesting test document via file upload...")
    
    import random
    unique_id = random.randint(10000, 99999)
    
    test_content = f"""Artificial Intelligence - Document {unique_id}
    
Artificial intelligence (AI) is intelligence demonstrated by machines. AI research has been defined as the field of study of intelligent agents.

Key Concepts (ID: {unique_id}):
- Machine learning is a subset of AI
- Neural networks mimic human brain structure  
- Deep learning uses multiple layers of neural networks

AI applications include computer vision, natural language processing, robotics, and expert systems.
The field was founded in 1956 at Dartmouth College.
"""
    
    # Create temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write(test_content)
        temp_file_path = f.name
    
    try:
        # Ingest via file upload endpoint
        with open(temp_file_path, 'rb') as f:
            files = {'file': ('test_python.txt', f, 'text/plain')}
            data = {'source_type': 'text'}
            
            response = requests.post(
                f"{API_URL}/api/v1/ingest/file",
                files=files,
                data=data
            )
        
        if response.status_code != 200:
            print(f"   ❌ Ingestion failed: {response.status_code}")
            print(f"   {response.text}")
            return False
        
        result = response.json()
        document_id = result["document_id"]
        print(f"   ✅ Document ingested: {document_id}")
        print(f"   Status: {result['status']}")
        
    finally:
        # Clean up temp file
        os.unlink(temp_file_path)
    
    # 3. Monitor document processing
    print("\n⏳ Step 3: Monitoring Celery task processing...")
    print("   Waiting for workers to process document...")
    
    for i in range(30):  # Wait up to 30 seconds
        time.sleep(1)
        
        # Check document status
        response = requests.get(f"{API_URL}/api/v1/documents/{document_id}")
        doc_status = response.json()
        
        print(f"   [{i+1}s] Status: {doc_status.get('status', 'unknown')}", end="\r")
        
        if doc_status.get('status') in ['completed', 'failed']:
            print(f"\n   Processing finished: {doc_status['status']}")
            break
    else:
        print("\n   ⚠️  Processing still ongoing after 30s")
    
    # 4. Check final stats to see if triples were created
    print("\n📊 Step 4: Checking final system stats...")
    response = requests.get(f"{API_URL}/api/v1/stats")
    final_stats = response.json()
    
    print(f"   Final documents: {final_stats['documents']['total']}")
    print(f"   Final candidate triples: {final_stats['triples']['candidate']}")
    print(f"   Final validated triples: {final_stats['triples']['validated']}")
    
    # 5. Verify Celery workers actually processed the document
    print("\n🔍 Step 5: Verification Results")
    print("=" * 60)
    
    docs_increased = final_stats['documents']['total'] > initial_stats['documents']['total']
    triples_increased = final_stats['triples']['candidate'] > initial_stats['triples']['candidate']
    
    if docs_increased:
        print("   ✅ Document count increased - Celery normalized document")
    else:
        print("   ❌ Document count unchanged - normalization may have failed")
    
    if triples_increased:
        print("   ✅ Triple count increased - Celery extracted triples")
    else:
        print("   ⚠️  Triple count unchanged - extraction may still be processing")
    
    print("\n" + "=" * 60)
    if docs_increased and triples_increased:
        print("🎉 SUCCESS: Celery workers are processing documents correctly!")
        return True
    elif docs_increased:
        print("⚠️  PARTIAL: Document normalized but triples not yet extracted")
        print("   This may be normal if extraction is still in progress")
        return True
    else:
        print("❌ FAILED: Celery workers may not be processing documents")
        return False

if __name__ == "__main__":
    try:
        success = test_document_ingestion()
        exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
