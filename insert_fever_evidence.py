#!/usr/bin/env python3
"""
Insert FEVER Evidence Documents into GraphBuilder-RAG

Extracts evidence from FEVER dataset and inserts as documents for testing.
"""
import json
import asyncio
import httpx
from pathlib import Path
import sys
import tempfile
import os

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

async def insert_fever_evidence():
    """Insert FEVER evidence as documents."""

    fever_file = Path("tests/benchmarks/data/fever/fever_expanded_5000.jsonl")

    if not fever_file.exists():
        print(f"FEVER file not found: {fever_file}")
        return

    print("Reading FEVER dataset...")

    # Collect unique evidence texts
    evidence_docs = {}
    with open(fever_file, 'r') as f:
        for line in f:
            try:
                entry = json.loads(line.strip())
                evidence = entry.get('evidence', '').strip()
                claim_id = entry['id']

                if evidence and len(evidence) > 10:  # Skip empty/short evidence
                    # Use evidence as key to avoid duplicates
                    if evidence not in evidence_docs:
                        evidence_docs[evidence] = {
                            'content': evidence,
                            'title': f'FEVER Evidence {claim_id}',
                            'source': 'fever_expanded_dataset'
                        }
            except json.JSONDecodeError:
                continue

    print(f"Found {len(evidence_docs)} unique evidence documents")

    # Insert documents via file upload API
    api_url = "http://localhost:8000/api/v1/ingest/file"

    async with httpx.AsyncClient(timeout=60.0) as client:
        for i, (evidence, doc_data) in enumerate(evidence_docs.items(), 1):
            try:
                # Create temporary file with evidence content
                with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as temp_file:
                    temp_file.write(doc_data['content'])
                    temp_file_path = temp_file.name

                try:
                    # Upload the file
                    with open(temp_file_path, 'rb') as f:
                        files = {'file': (f'fever_evidence_{i}.txt', f, 'text/plain')}
                        data = {'source_type': 'text'}

                        print(f"Inserting document {i}/{len(evidence_docs)}: {doc_data['title'][:50]}...")

                        response = await client.post(api_url, files=files, data=data)

                        if response.status_code == 200:
                            result = response.json()
                            print(f"✅ Success: {result.get('document_id', 'unknown')}")
                        else:
                            print(f"❌ Failed: {response.status_code} - {response.text}")

                finally:
                    # Clean up temporary file
                    os.unlink(temp_file_path)

                # Small delay to avoid overwhelming
                await asyncio.sleep(0.1)

            except Exception as e:
                print(f"❌ Error inserting document: {e}")
                continue

    print("\n🎉 FEVER evidence insertion complete!")
    print("📊 Check /api/v1/stats to see the updated document/triple counts")
    print("🧪 Then run: python tests/benchmarks/run_all_benchmarks.py --datasets fever")

if __name__ == "__main__":
    asyncio.run(insert_fever_evidence())