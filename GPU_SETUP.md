# GPU Setup Guide for Homodyne ⚡

**Smart GPU activation with automatic CUDA detection and JAX optimization for homodyne analysis.**

*Updated: 2024-08-31 - Unified post-install system with smart GPU detection*

______________________________________________________________________

## 🚀 Quick GPU Setup (Unified System)

### System Requirements

**Required for GPU acceleration:**
- **Linux OS** (GPU acceleration only available on Linux)
- **NVIDIA GPU** with compatible driver
- **CUDA 12.x** (system or conda installation)
- **Virtual environment** (conda, mamba, venv, or virtualenv)

### One-Command Setup

```bash
# 1. Install homodyne with JAX support
pip install homodyne-analysis[jax]

# 2. Unified installation with GPU setup
homodyne-post-install --shell zsh --gpu --advanced

# 3. Restart shell or reactivate environment
conda deactivate && conda activate your-env

# 4. Test GPU setup (aliases now available)
gpu-status        # Check GPU activation status
hm config.json    # MCMC with smart GPU activation
```

### Smart Command Reference

**🖥️ Unified Commands (Smart GPU activation):**
```bash
# MCMC with automatic GPU detection
hm config.json           # Alias: homodyne --method mcmc (GPU if available)
homodyne --config config.json --method mcmc

# All methods with smart GPU for MCMC
ha config.json           # Alias: homodyne --method all
homodyne --config config.json --method all

# Classical/robust methods (CPU optimized)
hc config.json           # Alias: homodyne --method classical  
hr config.json           # Alias: homodyne --method robust
```

**🚀 Advanced GPU Tools (After setup):**
```bash
# GPU system status and optimization
gpu-status               # Alias: homodyne_gpu_status
gpu-bench                # Alias: homodyne_gpu_benchmark
homodyne-gpu-optimize    # Hardware-specific optimization
homodyne-validate --test gpu  # Validate GPU setup
```

### GPU Issues? Try These Solutions

**Smart troubleshooting with unified system:**

```bash
# 1. Check GPU system validation
homodyne-validate --test gpu --verbose

# 2. Check hardware detection
homodyne-gpu-optimize --report

# 3. Verify GPU activation status
gpu-status                        # Should show "✅ GPU activated"

# 4. Manual GPU optimization
homodyne-gpu-optimize --benchmark --apply

# 5. Reinstall GPU setup (unified system)
homodyne-cleanup --interactive    # Remove all GPU files
homodyne-post-install --gpu       # Reinstall with smart detection

# 6. Check CUDA installation automatically
nvidia-smi                        # Check GPU and driver
homodyne-validate --test gpu      # Comprehensive GPU testing

# 7. Fall back to reliable CPU mode
homodyne --config config.json --method mcmc
```

______________________________________________________________________

## 📦 GPU Installation Methods

### 1. Automatic GPU Detection (Recommended)

**Smart CUDA Detection:**
```bash
# The unified system automatically detects:
# - System CUDA installations (/usr/local/cuda)
# - Conda CUDA packages
# - Pip-installed CUDA libraries
# - GPU hardware capabilities

# Run comprehensive detection
homodyne-gpu-optimize --report
```

**Manual CUDA Installation (if needed):**
```bash
# Option 1: Conda CUDA (recommended)
conda install -c nvidia cuda-toolkit cuda-runtime

# Option 2: System CUDA
# Download CUDA Toolkit 12.x from NVIDIA
# https://developer.nvidia.com/cuda-downloads

# Verify installation
nvidia-smi                        # Check GPU and driver
homodyne-validate --test gpu      # Test homodyne GPU integration
```

### 2. JAX Installation with Smart GPU Support

**Unified JAX Installation:**
```bash
# Install homodyne with JAX GPU support
pip install homodyne-analysis[jax]

# This automatically includes:
# - JAX with CUDA 12 support
# - All required NVIDIA libraries
# - Optimized GPU backends

# Verify installation
homodyne-validate --test gpu
python -c "import jax; print('JAX devices:', jax.devices())"
```

### 3. Unified Virtual Environment Integration

**One-Command Setup:**
```bash
# Complete GPU setup with shell integration
homodyne-post-install --shell zsh --gpu --advanced

# This automatically:
# - Detects GPU hardware and CUDA installation
# - Configures optimal GPU settings
# - Sets up shell completion with GPU aliases
# - Creates conda activation scripts
# - Installs advanced GPU tools
```

### 4. Smart GPU Activation

**Automatic activation (unified system):**
```bash
# GPU activates automatically when environment loads
conda activate your-env

# Verify smart activation
gpu-status                        # Shows GPU activation status
homodyne --help                   # GPU-enabled if available
```

### 5. Shell Completion with GPU Integration

**Unified completion system:**
```bash
# After post-install, these aliases are available:
hm config.json        # homodyne --method mcmc (smart GPU)
hc config.json        # homodyne --method classical
hr config.json        # homodyne --method robust
ha config.json        # homodyne --method all (smart GPU)

# GPU-specific shortcuts:
gpu-status           # homodyne_gpu_status
gpu-bench            # homodyne_gpu_benchmark
gpu-on               # Manual GPU activation
gpu-off              # Manual GPU deactivation
```

______________________________________________________________________

## ⚙️ Smart GPU Configuration

### GPU Optimization Profiles

**Automatic profile selection:**
```bash
# Hardware-optimized settings applied automatically
homodyne-gpu-optimize --benchmark --apply

# Use specific optimization profile
homodyne-gpu-optimize --profile conservative
homodyne-gpu-optimize --profile aggressive
```

### Disable GPU (CPU-only mode)

**Via environment variables:**
```bash
# Disable GPU temporarily
export HOMODYNE_GPU_AUTO=0
homodyne --config config.json --method mcmc

# Force CPU-only mode
export JAX_PLATFORMS=cpu
homodyne --config config.json --method mcmc
```

**Via config file:**
```json
{
  "optimization_config": {
    "mcmc_sampling": {
      "use_jax_backend": false
    }
  }
}
```

### Advanced GPU Settings

**Environment variables for tuning:**
```bash
# GPU memory allocation
export HOMODYNE_GPU_MEMORY_FRACTION=0.8

# Optimal batch size
export HOMODYNE_BATCH_SIZE=2000

# Advanced XLA optimizations
export XLA_FLAGS="--xla_gpu_enable_triton_softmax_fusion=true"
```

______________________________________________________________________

## 🔧 Smart Troubleshooting

### GPU Detection Issues

**Comprehensive system validation:**
```bash
# Run full GPU validation
homodyne-validate --test gpu --verbose

# Check hardware and optimization
homodyne-gpu-optimize --report

# Manual system checks
nvidia-smi                        # GPU and driver status
gpu-status                        # Homodyne GPU activation
```

### Common Error Messages

| Error | Unified System Solution |
|-------|------------------------|
| `GPU not activated` | Run `homodyne-post-install --gpu` |
| `Unable to load cuSPARSE` | Run `homodyne-gpu-optimize --fix` |
| `JAX devices: [CpuDevice]` | Check `gpu-status` and reactivate environment |
| `Out of memory` | Use `homodyne-gpu-optimize --profile conservative` |

### Smart GPU Performance Testing

**Built-in GPU benchmarking:**
```bash
# Comprehensive GPU testing
homodyne-gpu-optimize --benchmark

# Quick GPU performance check
gpu-bench                         # Alias: homodyne_gpu_benchmark

# Validation with performance metrics
homodyne-validate --test gpu --verbose
```

**Manual performance verification:**
```python
import jax
print(f"Devices: {jax.devices()}")          # Should show [CudaDevice(id=0)]
print(f"Backend: {jax.default_backend()}")  # Should show 'gpu'

# Run built-in benchmark
from homodyne.gpu_benchmark import run_benchmark
run_benchmark()  # Comprehensive GPU performance test
```

______________________________________________________________________

## 📊 Smart Performance Optimization

### Automatic GPU Selection

**Smart GPU activation (unified system):**
- **Automatic detection:** GPU used when available and beneficial
- **Intelligent fallback:** CPU mode for unsupported platforms
- **Performance-based:** GPU only when it provides speedup
- **Memory-aware:** Optimal GPU memory allocation

### When GPU Acceleration Activates

**GPU automatically used for:**
- **MCMC sampling** with large datasets (>1000 data points)
- **Multiple chains** (4+ chains with GPU memory available)
- **Complex models** with many parameters
- **Linux systems** with CUDA-compatible GPUs

**CPU automatically used for:**
- **Classical optimization** (already CPU-optimized)
- **Robust methods** (CPU-optimized algorithms)
- **Small datasets** (GPU overhead not worth it)
- **Windows/macOS** (automatic fallback)

### Unified Command Usage

```bash
# Smart GPU/CPU selection (recommended)
hm config.json              # MCMC with automatic GPU detection
ha config.json              # All methods with smart selection

# Manual control (if needed)
HOMODYNE_GPU_AUTO=0 hm config.json    # Force CPU mode
HOMODYNE_GPU_AUTO=1 hm config.json    # Force GPU attempt
```

### Smart Performance Tips

**Automatic optimization:**
- **Hardware detection:** Optimal settings applied automatically
- **Benchmark-based:** Settings tuned to your specific GPU
- **Memory management:** Automatic GPU memory fraction optimization
- **Batch size tuning:** Hardware-specific batch sizes

**Performance monitoring:**
```bash
# Real-time GPU monitoring during analysis
nvidia-smi dmon -i 0 -s pucvmet &
hm config.json

# Built-in performance profiling
homodyne-gpu-optimize --benchmark --report
```

______________________________________________________________________

## 💻 Smart System Requirements

**Platform Support (Automatic Detection):**
- ✅ **Linux**: Full GPU acceleration with automatic CUDA detection
- ✅ **Windows**: CPU-only mode (automatic fallback)
- ✅ **macOS**: CPU-only mode (automatic fallback)

**GPU Requirements (Linux only):**
- **GPU:** NVIDIA GPU with CUDA capability
- **Python:** 3.9+
- **Driver:** NVIDIA driver compatible with installed CUDA
- **CUDA:** Any CUDA 12.x (system, conda, or pip-installed)

**Smart Detection Features:**
- **Hardware detection:** Automatic GPU memory and capability detection
- **CUDA detection:** Finds CUDA in system, conda, or pip locations
- **Performance tuning:** Hardware-specific optimization
- **Compatibility checking:** Validates driver and library compatibility

______________________________________________________________________

## 🔍 Advanced Troubleshooting

### Removing GPU Setup (Unified System)

**Clean removal of all GPU components:**
```bash
# Interactive cleanup (recommended)
homodyne-cleanup --interactive
# Choose: "Remove GPU setup? [y/N]: y"

# Complete cleanup
homodyne-cleanup

# Reinstall with different settings
homodyne-post-install --shell zsh --gpu
```

**This removes:**
- GPU activation scripts from virtual environment
- Smart GPU detection configuration
- Advanced GPU optimization tools
- GPU-related shell aliases and completion

### System Status Check

**Comprehensive system validation:**
```bash
# Get complete system status
homodyne-validate --verbose

# Hardware-specific report
homodyne-gpu-optimize --report

# Quick GPU status
gpu-status
```

**Example output for properly configured system:**
```
✅ **Platform**: Linux (GPU acceleration available)
✅ **Hardware**: NVIDIA GPU detected with CUDA support
✅ **Driver**: Compatible NVIDIA driver
✅ **CUDA**: CUDA 12.x found and configured
✅ **JAX**: JAX with GPU support installed
✅ **Homodyne**: GPU acceleration activated
```

### Smart CUDA Library Management

**Unified system handles common CUDA issues automatically:**

```bash
# Automatic CUDA library detection and configuration
homodyne-gpu-optimize --fix

# Manual CUDA troubleshooting
homodyne-validate --test gpu --verbose
```

**Common issues resolved automatically:**
- **Library path conflicts:** Smart LD_LIBRARY_PATH configuration
- **CUDA version mismatches:** Automatic version detection and compatibility
- **JAX plugin issues:** Optimized JAX configuration for your system
- **Memory conflicts:** Hardware-specific memory allocation

### Advanced GPU Configuration

**Custom GPU optimization (unified system):**

```bash
# Create custom GPU profile
homodyne-gpu-optimize --profile custom --memory-fraction 0.8 --batch-size 1000

# Benchmark and apply optimal settings
homodyne-gpu-optimize --benchmark --apply

# Advanced XLA flag optimization
export XLA_FLAGS="--xla_gpu_enable_triton_softmax_fusion=true --xla_gpu_triton_gemm_any=true"
homodyne-gpu-optimize --apply
```

**Environment variable control:**
```bash
# Force GPU activation
export HOMODYNE_GPU_AUTO=1
export HOMODYNE_GPU_MEMORY_FRACTION=0.9
export HOMODYNE_BATCH_SIZE=2000

# Run with custom settings
hm config.json
```

### JAX Installation Options (Unified System)

**Recommended installation (automatic):**
```bash
# Unified JAX installation with smart GPU support
pip install homodyne-analysis[jax]
homodyne-post-install --gpu
```

**Alternative installations:**
```bash
# CPU-only installation
pip install homodyne-analysis[mcmc]
# GPU support disabled automatically

# Manual JAX with system CUDA
pip install homodyne-analysis
pip install "jax[cuda12]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
homodyne-post-install --gpu
```

### Performance Comparison (Smart Selection)

**Automatic performance optimization:**
- **Small datasets:** CPU selected automatically (no GPU overhead)
- **Large MCMC:** GPU selected when beneficial (2-4x speedup)
- **Classical methods:** CPU-optimized algorithms used
- **Memory-aware:** GPU/CPU selection based on available memory

**Built-in performance monitoring:**
```bash
# Compare GPU vs CPU performance
homodyne-gpu-optimize --benchmark

# Profile specific analysis
HOMODYNE_PROFILE=1 hm config.json
```

### When GPU Issues Persist

**Smart fallback system:**
1. **Automatic CPU fallback** - GPU issues don't break analysis
2. **Diagnostic tools** - `homodyne-validate --test gpu` for troubleshooting
3. **Performance guidance** - Built-in recommendations for your hardware
4. **Support system** - Comprehensive validation and error reporting

______________________________________________________________________

## 📝 Smart GPU System Components

**`homodyne-post-install --gpu`:**
- Smart CUDA detection across system, conda, and pip locations
- Hardware-specific optimization profile generation
- Automatic GPU activation script creation
- Shell integration with GPU aliases

**`gpu_activation_smart.sh`:**
- Dynamic CUDA library path configuration
- JAX backend optimization
- Hardware-specific XLA flag setting
- Memory fraction optimization

**`homodyne-gpu-optimize`:**
- Hardware capability detection
- Performance benchmarking
- Automatic optimization profile application
- System compatibility validation

**`homodyne-validate --test gpu`:**
- Comprehensive GPU system testing
- Hardware and driver compatibility checks
- JAX backend validation
- Performance verification

**Smart activation system:**
- **Automatic detection:** GPU used when available and beneficial
- **Platform awareness:** CPU fallback on Windows/macOS
- **Performance-based:** GPU only when it provides speedup
- **Memory management:** Optimal GPU memory allocation

______________________________________________________________________

## 🎆 Advanced Features Integration

### GPU Monitoring and Benchmarking

**Real-time GPU monitoring:**
```bash
# Monitor GPU during analysis
watch -n 1 nvidia-smi

# Built-in GPU benchmark
gpu-bench                         # Comprehensive performance testing

# Performance profiling
homodyne-gpu-optimize --benchmark --report
```

### CI/CD Integration

**Automated GPU testing:**
```bash
# Validate GPU setup in CI/CD
homodyne-validate --json | jq '.test_results[] | select(.category == "GPU Setup")'

# Automated optimization
homodyne-gpu-optimize --benchmark --apply --quiet
```

### Docker GPU Support

**NVIDIA Container Toolkit integration:**
```dockerfile
FROM nvidia/cuda:12.2-devel-ubuntu22.04

# Install homodyne with GPU support
RUN pip install homodyne-analysis[jax]
RUN homodyne-post-install --gpu --non-interactive

# Validate installation
RUN homodyne-validate --test gpu
```

______________________________________________________________________
