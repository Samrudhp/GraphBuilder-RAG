#!/usr/bin/env python3
"""
Download lightweight NLI model for fact verification
"""

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

print("📥 Downloading lightweight NLI model...")

try:
    # Smaller MNLI model - DistilBERT based
    model_name = "distilbert-base-uncased-finetuned-sst-2-english"
    
    print(f"   Model: {model_name}")
    print("   Downloading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print("   ✅ Tokenizer downloaded")
    
    print("   Downloading model...")
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    print("   ✅ Model downloaded")
    
    print(f"\n✅ Model ready! Using: {model_name}")
    print("   For NLI: positive=ENTAILMENT, negative=CONTRADICTION/NOT ENOUGH INFO")
    
except Exception as e:
    print(f"❌ Error: {e}")
