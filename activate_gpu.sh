#!/bin/bash
# Activation script for JAX GPU support in homodyne package
# Source this before running homodyne with GPU acceleration
# 
# IMPORTANT: GPU acceleration only works on Linux with CUDA-enabled JAX
# Windows and macOS are not supported for GPU acceleration

# Check if running on Linux
if [[ "$(uname -s)" != "Linux" ]]; then
    echo "❌ GPU acceleration not supported on $(uname -s)"
    echo "   GPU acceleration requires Linux with CUDA-enabled JAX"
    echo "   Current platform: $(uname -s)"
    echo "   Homodyne will automatically use CPU-only mode"
    return 1 2>/dev/null || exit 1
fi

# Get Python site-packages directory
SITE_PACKAGES=$(python -c "import site; print(site.getsitepackages()[0])")

# Build LD_LIBRARY_PATH with all NVIDIA library paths
NVIDIA_LIB_PATH=""
for lib in cublas cudnn cufft curand cusolver cusparse nccl nvjitlink cuda_runtime cuda_cupti cuda_nvcc cuda_nvrtc; do
    LIB_DIR="$SITE_PACKAGES/nvidia/$lib/lib"
    if [ -d "$LIB_DIR" ]; then
        if [ -z "$NVIDIA_LIB_PATH" ]; then
            NVIDIA_LIB_PATH="$LIB_DIR"
        else
            NVIDIA_LIB_PATH="$NVIDIA_LIB_PATH:$LIB_DIR"
        fi
    fi
done

# Clean existing CUDA paths to avoid conflicts and export the library path
CLEAN_PATH=""
if [ -n "$LD_LIBRARY_PATH" ]; then
    CLEAN_PATH=$(echo "$LD_LIBRARY_PATH" | tr ':' '\n' | grep -v '/usr/local/cuda' | tr '\n' ':' | sed 's/:$//')
fi
export LD_LIBRARY_PATH="$NVIDIA_LIB_PATH${CLEAN_PATH:+:$CLEAN_PATH}"

# Set additional environment variables for JAX CUDA support
export XLA_FLAGS="--xla_gpu_cuda_data_dir=$SITE_PACKAGES/nvidia"
export JAX_PLATFORMS=""

echo "✓ JAX GPU support activated for homodyne on Linux"
echo "  LD_LIBRARY_PATH configured with pip-installed CUDA libraries"
echo "  XLA_FLAGS: $XLA_FLAGS"
echo "  Platform: $(uname -s) (GPU acceleration supported)"

# Test GPU detection
GPU_TEST=$(python -c "
import os
os.environ['JAX_PLATFORMS'] = ''
import jax
devices = jax.devices()
if len(devices) > 0 and any(d.platform == 'gpu' for d in devices):
    print('GPU detected')
else:
    print('CPU only')
" 2>/dev/null || echo 'Import error')

echo "  GPU: $GPU_TEST"

if [[ "$GPU_TEST" == "CPU only" ]]; then
    echo "  Note: If GPU is not detected, check:"
    echo "    1. NVIDIA driver installation (nvidia-smi)"
    echo "    2. CUDA libraries (pip list | grep nvidia)"
    echo "    3. JAX CUDA plugin (pip show jax-cuda12-plugin)"
fi
