import torch
import logging
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_gpu():
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"Device Count: {torch.cuda.device_count()}")
        print(f"Current Device: {torch.cuda.current_device()}")
        print(f"Device Name: {torch.cuda.get_device_name(0)}")
        
        # Test Tensor Operations on GPU
        print("\nRunning GPU Tensor Test...")
        try:
            device = torch.device("cuda")
            x = torch.randn(5000, 5000).to(device)
            y = torch.randn(5000, 5000).to(device)
            
            start_time = time.time()
            z = torch.matmul(x, y)
            torch.cuda.synchronize() # Wait for operation to finish
            end_time = time.time()
            
            print(f"Matrix Multiplication (5000x5000) Time: {end_time - start_time:.4f} seconds")
            print("GPU Test Passed!")
            return True
        except Exception as e:
            print(f"GPU Test Failed: {e}")
            return False
    else:
        print("No GPU found. Running on CPU.")
        return False

if __name__ == "__main__":
    test_gpu() 