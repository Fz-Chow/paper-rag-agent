import os
import platform
import subprocess

print(f"OS: {platform.platform()}")
print(f"PAPER_RAG_DEVICE: {os.getenv('PAPER_RAG_DEVICE', 'auto')}")
try:
    import torch
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"HIP version: {getattr(torch.version, 'hip', None)}")
except ImportError:
    print("PyTorch: not installed; baseline TF-IDF mode remains available.")
try:
    result = subprocess.run(["rocminfo"], capture_output=True, text=True, timeout=15)
    print(f"rocminfo exit code: {result.returncode}")
    print(result.stdout[:1200])
except (FileNotFoundError, subprocess.TimeoutExpired):
    print("rocminfo: unavailable")

