#!/usr/bin/env python3
import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import torch
print(f"PyTorch version: {torch.__version__}")

# Force CPU
torch.set_default_device('cpu')

# Test basic tensor operation
x = torch.randn(3, 3)
print(f"Tensor device: {x.device}")
print("✅ PyTorch CPU mode working")
