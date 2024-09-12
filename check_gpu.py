import torch


# Function to check and use available GPU acceleration
def check_and_use_gpu():
    # Check for CUDA (NVIDIA GPUs)
    if torch.cuda.is_available():
        print("CUDA is available. Running on GPU with CUDA.")
        device = torch.device("cuda")
    # Check for MPS (Apple Silicon GPUs)
    elif torch.backends.mps.is_available():
        print("MPS is available. Running on GPU with MPS.")
        device = torch.device("mps")
    else:
        print("GPU acceleration is not available. Running on CPU.")
        device = torch.device("cpu")

    # Create a tensor and perform operations
    a = torch.randn(10).to(device)
    print(a)
    print(a + 2)


# Run the function
check_and_use_gpu()
