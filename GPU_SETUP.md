# GPU Setup Guide for Alpaca-bot

This guide will help you set up proper GPU acceleration for TensorFlow in the Alpaca-bot project.

## Current Status

Based on the diagnostic results, your system has:
- Python 3.12.9
- NVIDIA GeForce RTX 4070 GPU
- CUDA 11.8 installed
- TensorFlow CPU version running

## Issue

TensorFlow isn't detecting your GPU because:
1. **Python version compatibility**: TensorFlow with GPU support on Windows requires Python 3.7-3.9, but you have Python 3.12.9
2. **TensorFlow version**: Native GPU support for Windows is only available in TensorFlow ≤ 2.10
3. **CUDA version mismatch**: Your installed CUDA packages are for CUDA 12.x but your system has CUDA 11.8

## Solution Options

### Option 1: Create a new Python 3.9 environment with TensorFlow 2.10 (Recommended)

1. **Install Miniconda** (lighter version of Anaconda):
   - Download from: https://docs.conda.io/en/latest/miniconda.html

2. **Create a new environment**:
   ```
   conda create -n tf_gpu python=3.9
   conda activate tf_gpu
   ```

3. **Install TensorFlow 2.10**:
   ```
   pip install tensorflow==2.10.0
   ```

4. **Install CUDA Toolkit 11.2**:
   - Download from: https://developer.nvidia.com/cuda-11.2.0-download-archive
   - Follow the installation instructions

5. **Install cuDNN 8.1**:
   - Download from: https://developer.nvidia.com/rdp/cudnn-archive
   - Choose cuDNN v8.1.1 for CUDA 11.2
   - Extract and copy files to your CUDA installation

6. **Set environment variables**:
   ```
   setx CUDA_PATH "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2"
   setx CUDA_HOME "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2"
   ```

7. **Install project dependencies**:
   ```
   pip install -r requirements.txt
   ```

8. **Verify GPU setup**:
   ```
   python -c "import tensorflow as tf; print('GPU available:', len(tf.config.list_physical_devices('GPU')))"
   ```

### Option 2: Use WSL2 (Windows Subsystem for Linux)

For newer TensorFlow versions, you can use WSL2:

1. **Install WSL2**:
   - Run in PowerShell as administrator: `wsl --install`

2. **Install Ubuntu**:
   - From Microsoft Store, install Ubuntu 22.04 LTS

3. **Set up CUDA in WSL2**:
   - Follow: https://docs.nvidia.com/cuda/wsl-user-guide/index.html

4. **Set up TensorFlow**:
   - In Ubuntu, install TensorFlow with GPU support

### Option 3: Downgrade Python on current system

If you prefer not to create a new environment:

1. **Uninstall Python 3.12**:
   - Use Windows Add/Remove Programs

2. **Install Python 3.9**:
   - Download from: https://www.python.org/downloads/release/python-3913/

3. **Follow steps 3-7 from Option 1**

## Troubleshooting

If you still face issues:

1. **Run the diagnostic script**:
   ```
   python src/check_gpu.py
   ```

2. **Verify CUDA installation**:
   ```
   nvcc --version
   nvidia-smi
   ```

3. **Check TensorFlow GPU detection**:
   ```
   python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
   ```

## References

- [TensorFlow GPU Support](https://www.tensorflow.org/install/gpu)
- [CUDA Toolkit Archive](https://developer.nvidia.com/cuda-toolkit-archive)
- [cuDNN Archive](https://developer.nvidia.com/rdp/cudnn-archive) 