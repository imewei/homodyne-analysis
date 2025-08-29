#!/bin/bash
# Activation script for JAX GPU support in homodyne package
# Source this before running homodyne with GPU acceleration

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

# Export the library path
export LD_LIBRARY_PATH="$NVIDIA_LIB_PATH${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

echo "✓ JAX GPU support activated for homodyne"
echo "  GPU: $(python -c 'import jax; print(jax.devices()[0])' 2>/dev/null || echo 'Not detected')"
