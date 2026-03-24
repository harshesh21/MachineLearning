import torch

print("--- PyTorch Environment Check ---")
print(f"PyTorch Version: {torch.__version__}")

# 1. Check for Hardware Acceleration (GPU)
if torch.cuda.is_available():
    device = "cuda"
    print("✅ NVIDIA GPU (CUDA) Detected. We are ready for high-speed training.")
elif torch.backends.mps.is_available():
    device = "mps"
    print("✅ Apple Silicon GPU (MPS) Detected.")
else:
    device = "cpu"
    print("⚠️ No GPU detected. Defaulting to CPU (Standard for beginners).")

# 2. Create your first Tensor and send it to the hardware
# We are creating a 2x3 matrix of random numbers
my_first_tensor = torch.rand(2, 3).to(device)

print("\nYour First Tensor:")
print(my_first_tensor)
print(f"Tensor Shape: {my_first_tensor.shape}")
print(f"Tensor Data Type: {my_first_tensor.dtype}")
print(f"Running on Device: {my_first_tensor.device}")