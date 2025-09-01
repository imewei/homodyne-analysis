# Advanced Homodyne Features Guide

**Complete guide to unified shell completion, smart GPU optimization, and comprehensive system validation features.**

*Updated: 2024-08-31 - Reflects unified completion system and streamlined architecture*

---

## 🚀 Advanced Shell Completion (Phase 4)

### Features

- **Context-Aware Completion**: Suggests methods based on config file mode
- **Cached File Discovery**: Faster completion with intelligent caching  
- **Smart Suggestions**: Common parameter values and directory structures
- **Command Builder**: Interactive command construction

### Usage

```bash
# Auto-completes with recent config files
homodyne --config <TAB>

# Smart method suggestions based on config
homodyne --config laminar_flow.json --method <TAB>
# Shows: mcmc robust all (optimized for laminar flow)

# Common parameter values
homodyne --phi-angles <TAB>
# Shows: 0,45,90,135  0,36,72,108,144  0,30,60,90,120,150

# Interactive command builder
homodyne_build
```

### Configuration

```bash
# Cache location
~/.cache/homodyne/completion_cache

# Refresh cache manually
find . -name "*.json" > ~/.cache/homodyne/completion_cache
```

---

## ⚡ GPU Auto-Optimization (Phase 5)

### Smart GPU Detection

The system automatically:
- Detects NVIDIA GPU hardware and memory
- Finds CUDA installation in standard locations
- Configures optimal memory allocation
- Sets hardware-specific XLA flags

### Features

```bash
# Smart GPU activation (automatic)
conda activate your-env  # GPU activates automatically

# Detailed GPU status
homodyne_gpu_status
# Shows: Hardware info, JAX devices, memory usage, optimization flags

# GPU benchmark (via function)
homodyne_gpu_benchmark
# Runs matrix multiplication benchmarks across different sizes

# Hardware optimization (CLI tool)
homodyne-gpu-optimize
# Detects hardware and applies optimal settings

# Aliases available after activation
gpu-status    # Alias for homodyne_gpu_status
gpu-bench     # Alias for homodyne_gpu_benchmark
```

### Auto-Configuration

The system sets optimal configurations based on GPU memory:

| GPU Memory | Memory Fraction | Batch Size | XLA Flags |
|------------|-----------------|------------|-----------|
| < 4GB      | 70%            | 500        | Basic     |
| 4-8GB      | 80%            | 1000       | Standard  |
| > 8GB      | 90%            | 2000+      | Advanced  |

### Manual Tuning

```bash
# Run optimizer with benchmarking
homodyne-gpu-optimize --benchmark --apply

# Generate detailed report
homodyne-gpu-optimize --report

# Force hardware re-detection
homodyne-gpu-optimize --force
```

---

## 🧪 System Validation (Phase 6)

### Comprehensive Testing

```bash
# Full system validation
homodyne-validate

# Verbose output with details
homodyne-validate --verbose

# Test specific component
homodyne-validate --test gpu
homodyne-validate --test completion

# JSON output for automation
homodyne-validate --json
```

### Test Categories

1. **Environment Detection**
   - Platform identification
   - Python version
   - Virtual environment detection
   - Shell type

2. **Homodyne Installation**
   - Command availability
   - Help output validation
   - Module imports

3. **Shell Completion**
   - Completion file presence
   - Activation script testing
   - Alias functionality

4. **GPU Setup**
   - GPU hardware detection
   - JAX device availability
   - CUDA installation
   - Driver compatibility

5. **Integration**
   - Component interaction
   - Cross-module imports
   - Script execution

### Sample Validation Report

```
🔍 HOMODYNE SYSTEM VALIDATION REPORT
================================================================================

📊 Summary: 5/5 tests passed
🎉 All systems operational!

🖥️  Environment:
   platform: Linux
   python_version: 3.12.0
   conda_env: xpcs
   shell: zsh

📋 Test Results:
----------------------------------------

✅ PASS Environment Detection
   Message: Detected: Linux, Python 3.12.0, Shell: zsh
   Time: 0.003s

✅ PASS Homodyne Installation
   Message: Found 5/5 commands
   Time: 0.152s

✅ PASS Shell Completion
   Message: Found 2 completion files (aliases working)
   Time: 0.089s

✅ PASS GPU Setup
   Message: GPU ready: 1 GPU(s) with JAX support
   Time: 0.234s

✅ PASS Integration
   Message: Module imports: 4/4
   Time: 0.067s

💡 Recommendations:
   🚀 Your homodyne installation is ready!
   📖 Check documentation for usage examples
```

---

## 🛠️ Advanced Configuration

### Environment Variables

```bash
# GPU Optimization
export HOMODYNE_GPU_AUTO=1                    # Auto-activate GPU
export HOMODYNE_GPU_MEMORY_FRACTION=0.8       # Memory allocation
export HOMODYNE_BATCH_SIZE=2000               # Optimal batch size

# Shell Completion
export HOMODYNE_COMPLETION_CACHE_TTL=300      # Cache refresh time (5min)
export HOMODYNE_COMPLETION_MAX_FILES=20       # Max files in cache

# Advanced XLA Flags
export XLA_FLAGS="--xla_gpu_enable_triton_softmax_fusion=true \
                  --xla_gpu_triton_gemm_any=true \
                  --xla_gpu_enable_async_collectives=true"
```

### Custom Optimization Profiles

Create custom GPU profiles in `~/.config/homodyne/gpu_profiles.json`:

```json
{
  "profiles": {
    "conservative": {
      "memory_fraction": 0.6,
      "batch_size": 500,
      "enable_x64": false
    },
    "aggressive": {
      "memory_fraction": 0.95,
      "batch_size": 5000,
      "enable_x64": true,
      "xla_flags": [
        "--xla_gpu_enable_triton_gemm=true",
        "--xla_gpu_force_compilation_parallelism=4"
      ]
    }
  }
}
```

### Shell Completion Customization

Add custom completions in `~/.config/homodyne/completion_custom.sh`:

```bash
# Custom method completions
_homodyne_custom_methods() {
    echo "classical mcmc robust all custom_method"
}

# Custom config paths
_homodyne_custom_configs() {
    echo "$HOME/configs/*.json"
    echo "./analysis/*.json"
}
```

---

## 🐛 Troubleshooting

### Common Issues

**Shell completion not working:**
```bash
# Check activation scripts
ls -la $CONDA_PREFIX/etc/conda/activate.d/homodyne-*

# Manually reload
conda deactivate && conda activate your-env

# Test manually
source $CONDA_PREFIX/etc/conda/activate.d/homodyne-completion.sh
```

**GPU not activating:**
```bash
# Check GPU status
gpu-status

# Manual activation
gpu-on

# Check CUDA installation
ls -la /usr/local/cuda
nvidia-smi
```

**Performance issues:**
```bash
# Run optimization
homodyne-gpu-optimize --benchmark --apply

# Check memory usage
nvidia-smi -l 1

# Reduce batch size
export HOMODYNE_BATCH_SIZE=500
```

### Debug Mode

```bash
# Enable debug output
export HOMODYNE_DEBUG=1

# Verbose GPU activation
export HOMODYNE_GPU_VERBOSE=1

# Completion debug
export HOMODYNE_COMPLETION_DEBUG=1
```

---

## 📊 Performance Monitoring

### GPU Monitoring

```bash
# Real-time GPU usage
watch -n 1 nvidia-smi

# GPU utilization during analysis
nvidia-smi dmon -i 0 -s pucvmet -d 1 &
homodyne-gpu --method mcmc --config config.json

# Memory usage tracking
nvidia-smi --query-gpu=memory.used,memory.total --format=csv -l 1
```

### Profiling

```bash
# JAX profiling
export JAX_ENABLE_X64=1
export JAX_DEBUG_NANS=1

# PyTensor profiling  
export PYTENSOR_FLAGS="device=cpu,profile=True"

# System resource monitoring
htop
iotop
```

---

## 🔧 Integration with Development Tools

### IDE Integration

**VS Code settings.json:**
```json
{
  "python.terminal.activateEnvironment": true,
  "terminal.integrated.shellArgs.linux": [
    "-c", "source ~/.bashrc && exec bash"
  ],
  "homodyne.gpu.autoActivate": true,
  "homodyne.completion.enabled": true
}
```

### CI/CD Integration

```yaml
# .github/workflows/test.yml
name: Homodyne Tests
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.12'
      - name: Install homodyne
        run: |
          pip install -e .
          homodyne-post-install --interactive <<< $'y\nbash\nn'
      - name: Validate installation
        run: homodyne-validate --json
      - name: Run tests
        run: homodyne --config test_config.json --method all
```

### Docker Integration

```dockerfile
FROM nvidia/cuda:12.2-devel-ubuntu22.04

# Install homodyne with GPU support
RUN pip install homodyne-analysis[jax]
RUN homodyne-post-install --gpu --force

# Auto-activate GPU in container
ENV HOMODYNE_GPU_AUTO=1
ENV XLA_FLAGS="--xla_gpu_cuda_data_dir=/usr/local/cuda"

# Validate installation
RUN homodyne-validate
```

---

This advanced features system provides comprehensive automation, optimization, and validation for homodyne installations across different environments and use cases.