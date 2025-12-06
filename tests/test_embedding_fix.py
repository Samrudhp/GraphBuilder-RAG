#!/usr/bin/env python3
"""
Test embedding service directly to identify the issue
"""
import os
# Set environment BEFORE any imports
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import sys
from pathlib import Path

# Force CPU in torch before service import
import torch
torch.set_default_device('cpu')

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

print("Testing EmbeddingService directly...\n")

try:
    from services.embedding.service import EmbeddingService
    print("✅ Import successful")
    
    service = EmbeddingService()
    print("✅ Service initialized")
    
    # Test with a simple text
    test_text = "Hello world"
    print(f"\nTesting with: '{test_text}'")
    
    embedding = service.embed_text(test_text)
    print(f"✅ Embedding generated!")
    print(f"   Shape: {embedding.shape}")
    print(f"   First 5 values: {embedding[:5]}")
    
    # Test batch
    test_batch = ["Hello", "World", "Test"]
    print(f"\nTesting batch: {test_batch}")
    
    batch_embeddings = service.embed_batch(test_batch)
    print(f"✅ Batch embeddings generated!")
    print(f"   Shape: {batch_embeddings.shape}")
    
    print("\n🎉 All embedding tests passed!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
