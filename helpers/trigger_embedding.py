"""Manually trigger document embedding (if Celery task didn't run)"""
from pymongo import MongoClient
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from workers.tasks import embed_document

# Get the normalized document ID
client = MongoClient('mongodb://localhost:27017')
db = client['graphbuilder_rag']

normalized_doc = db.normalized_docs.find_one({})
if not normalized_doc:
    print("No normalized documents found")
    exit(1)

doc_id = normalized_doc['document_id']
print(f"Triggering embedding for: {doc_id}")

# Trigger Celery task
result = embed_document.delay(doc_id)
print(f"Task submitted: {result.id}")
print("Check Celery worker logs for progress...")
