import torch, os

print("=== PyTorch GPU Test ===")

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"GPU name: {torch.cuda.get_device_name(0)}")
    print(f"GPU count: {torch.cuda.device_count()}")
    print(f"Current device: {torch.cuda.current_device()}")

    print(f"Memory allocated: {torch.cuda.memory_allocated(0)}")
    print(f"Memory reserved: {torch.cuda.memory_reserved(0)}")

    print("CUDA version:", torch.version.cuda)
    print("cuDNN version:", torch.backends.cudnn.version())
    print("Compute capability:", torch.cuda.get_device_capability(0))
    print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))

    # Real computation test
    x = torch.rand(2000, 2000).cuda()
    y = torch.mm(x, x)
    print("Computation successful on:", x.device)
else:
    print("No GPU detected.")


