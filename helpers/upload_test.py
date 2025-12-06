"""Quick script to upload test document"""
import requests
import os

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, "scientists.txt")

url = "http://localhost:8000/api/v1/ingest/file"

with open(file_path, "rb") as f:
    files = {"file": ("scientists.txt", f, "text/plain")}
    data = {"source_type": "text"}
    
    response = requests.post(url, files=files, data=data)
    result = response.json()
    print(result)
    print(f"\n✅ Document uploaded! ID: {result['document_id']}")
