#!/usr/bin/env python3
import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import torch
torch.set_default_device('cpu')

from transformers import AutoTokenizer, AutoModel

print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-small-en-v1.5")
model = AutoModel.from_pretrained("BAAI/bge-small-en-v1.5")
model = model.to('cpu')
model.eval()

print("✅ Model loaded")
print(f"Model device: {next(model.parameters()).device}")

# Test encoding
text = "Hello world"
encoded = tokenizer(text, return_tensors='pt')
print(f"Input device: {encoded['input_ids'].device}")

with torch.no_grad():
    output = model(**encoded)
    print(f"✅ Model inference working")
    print(f"Output shape: {output[0].shape}")
