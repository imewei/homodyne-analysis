#!/bin/bash
#=============================================================================
# Homodyne GPU Activation Script - System CUDA Integration
#=============================================================================
# Purpose: Configure Python virtual environment to use system CUDA and cuDNN
# Requirements: Linux, CUDA 12.6, cuDNN 9.12, jax[cuda12-local]
# Usage: source activate_gpu.sh
#=============================================================================

# Detect if running in Python virtual environment
if [ -n "$VIRTUAL_ENV" ]; then
    echo "📦 Virtual environment: $(basename $VIRTUAL_ENV)"
elif [ -n "$CONDA_DEFAULT_ENV" ]; then
    echo "📦 Conda environment: $CONDA_DEFAULT_ENV"
else
    echo "⚠️  No Python virtual environment detected"
fi

# Platform check - GPU requires Linux
if [[ "$(uname -s)" != "Linux" ]]; then
    echo "❌ GPU acceleration requires Linux (detected: $(uname -s))"
    return 1 2>/dev/null || exit 1
fi

# Verify system CUDA installation
if [ ! -d "/usr/local/cuda" ]; then
    echo "❌ System CUDA not found at /usr/local/cuda"
    echo "   Please install CUDA Toolkit 12.x from NVIDIA"
    return 1 2>/dev/null || exit 1
fi

# Configure environment to use system CUDA and cuDNN
echo "🔧 Configuring system CUDA for JAX GPU acceleration..."

# System CUDA paths
export CUDA_ROOT="/usr/local/cuda"
export CUDA_HOME="$CUDA_ROOT"
export PATH="$CUDA_ROOT/bin:$PATH"

# Library paths: System CUDA + System cuDNN
export LD_LIBRARY_PATH="$CUDA_ROOT/lib64:/usr/lib/x86_64-linux-gnu:/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH}"

# JAX configuration for system CUDA
export XLA_FLAGS="--xla_gpu_cuda_data_dir=$CUDA_ROOT"
export JAX_PLATFORMS=""  # Allow GPU detection
export HOMODYNE_GPU_ACTIVATED="1"

# Display system configuration
echo "✅ System CUDA configured:"
echo "   • CUDA: $(nvcc --version 2>/dev/null | grep release | sed 's/.*release //' | cut -d',' -f1)"
echo "   • cuDNN: $(ls /usr/lib/x86_64-linux-gnu/libcudnn.so.9* 2>/dev/null | head -1 | sed 's/.*\.so\.//')"
echo "   • Driver: $(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null)"

# Test JAX GPU detection
echo "🧪 Testing JAX GPU detection..."
python3 -c "
import os
os.environ['LD_LIBRARY_PATH'] = '$LD_LIBRARY_PATH'
os.environ['JAX_PLATFORMS'] = ''
try:
    import jax
    devices = jax.devices()
    if any(d.platform == 'gpu' for d in devices):
        print('✅ JAX GPU ready: {} on {}'.format(jax.__version__, devices[0]))
    else:
        print('⚠️  JAX using CPU - install with: pip install jax[cuda12-local]')
except ImportError:
    print('❌ JAX not installed - run: pip install jax[cuda12-local]')
except Exception as e:
    print(f'⚠️  JAX initialization: {str(e)[:60]}')
"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✨ GPU environment ready. Usage:"
echo "   homodyne-gpu --config config.json --method mcmc"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
