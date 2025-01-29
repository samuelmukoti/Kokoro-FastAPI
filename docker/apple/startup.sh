#!/bin/bash

# Check PyTorch MPS availability
python << END
import torch
import sys

print("\n=== Apple Silicon Acceleration Status ===")
print(f"PyTorch version: {torch.__version__}")
print(f"MPS (Metal Performance Shaders) available: {torch.backends.mps.is_available()}")
print(f"MPS built: {torch.backends.mps.is_built()}")
print(f"Default device: {torch.device(torch._C._get_default_device())}")
try:
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        x = torch.zeros(1).to(device)
        print("✅ Successfully initialized MPS device")
    else:
        print("⚠️  MPS device not available, falling back to CPU")
except Exception as e:
    print(f"⚠️  Error testing MPS device: {e}")
print("=====================================\n")

if not torch.backends.mps.is_available():
    print("⚠️  Warning: MPS acceleration not available")
    print("   This might be because:")
    print("   1. You're not running on Apple Silicon")
    print("   2. PyTorch MPS support is not properly installed")
    print("   3. The container doesn't have access to the GPU\n")
END

# Start the FastAPI server
exec python -m uvicorn api.src.main:app --host 0.0.0.0 --port 8880 --log-level debug 