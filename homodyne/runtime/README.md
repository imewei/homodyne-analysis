# Homodyne Runtime System

**Advanced shell completion, smart GPU acceleration, and comprehensive system validation
for the homodyne analysis package.**

*This unified runtime system provides intelligent automation, cross-platform
compatibility, and performance optimization for homodyne workflows.*

______________________________________________________________________

## 📁 Directory Structure

```
runtime/
├── shell/              # Advanced shell completion system
│   └── completion.sh   # Unified completion with context awareness
├── gpu/                # Smart GPU acceleration system  
│   ├── activation.sh   # Intelligent CUDA detection and setup
│   └── optimizer.py    # Hardware-specific optimization
├── utils/              # System validation and utilities
│   └── system_validator.py  # Comprehensive system testing
└── README.md          # This file
```

______________________________________________________________________

## 🚀 Quick Setup (Unified System)

### One-Command Installation

```bash
# Complete setup with all features
homodyne-post-install --shell zsh --gpu --advanced

# Interactive setup (choose features)
homodyne-post-install --interactive

# Basic shell completion only
homodyne-post-install --shell zsh
```

### What Gets Installed

1. **🔧 Smart Shell Completion** - Context-aware completion with caching
1. **⚡ Smart GPU Acceleration** - Automatic CUDA detection and optimization
1. **📝 Unified Aliases** - Consistent shortcuts (`hm`, `hc`, `hr`, `ha`)
1. **🛠️ Advanced Tools** - GPU optimization, system validation, benchmarking
1. **📋 Environment Integration** - Auto-activation in conda, mamba, venv

______________________________________________________________________

## 🔧 Shell Completion System

### Features (`shell/completion.sh`)

- **Context-Aware Completion**: Suggests methods based on config file analysis
- **Intelligent Caching**: 5-minute TTL cache for faster file discovery
- **Smart Parameter Suggestions**: Common values for angles, contrast, output
  directories
- **Interactive Command Builder**: `homodyne_build` function for guided command creation
- **Cross-Shell Support**: Unified system works with bash and zsh

### Usage Examples

```bash
# Auto-completes with recent JSON config files
homodyne --config <TAB>

# Smart method suggestions based on detected config mode
homodyne --config laminar_flow.json --method <TAB>
# → Shows: mcmc robust all (optimized for laminar flow)

# Common parameter value completion
homodyne --phi-angles <TAB>
# → Shows: 0,45,90,135  0,36,72,108,144  0,30,60,90,120,150

# Interactive command builder
homodyne_build
# → Guided menu system for building commands
```

### Smart Completion Logic

The completion system analyzes your config files to provide intelligent suggestions:

| Config Mode | Suggested Methods | Reasoning |
|-------------|-------------------|-----------| | `static_isotropic` |
`classical robust` | Optimized for static analysis | | `static_anisotropic` |
`classical robust` | CPU-optimized methods |\
| `laminar_flow` | `mcmc robust all` | Benefits from GPU acceleration |

### Configuration

```bash
# Cache settings
export HOMODYNE_COMPLETION_CACHE_TTL=300      # 5 minutes
export HOMODYNE_COMPLETION_MAX_FILES=20       # Max cached files

# Cache location
~/.cache/homodyne/completion_cache

# Manual cache refresh
find . -name "*.json" > ~/.cache/homodyne/completion_cache
```

______________________________________________________________________

## ⚡ GPU Acceleration System

### Smart GPU Detection (`gpu/activation.sh`)

The GPU system provides intelligent CUDA detection and automatic optimization:

**Key Features:**

- **Platform Detection**: Linux-only with automatic CPU fallback on other platforms
- **CUDA Discovery**: Searches `/usr/local/cuda`, `/opt/cuda`, conda installations
- **Hardware Analysis**: Detects GPU memory and capabilities via `nvidia-smi`
- **Automatic Optimization**: Sets optimal memory fractions and XLA flags
- **Environment Integration**: Seamless conda/mamba/venv activation

### Auto-Configuration Logic

Memory-based optimization profiles applied automatically:

| GPU Memory | Memory Fraction | Batch Size | XLA Optimizations |
|------------|-----------------|------------|-------------------| | < 4GB | 70% |
Conservative | Basic flags | | 4-8GB | 80% | Standard | Standard optimizations | | > 8GB
| 90% | Aggressive | Advanced XLA features |

### GPU Commands Available After Setup

```bash
# Smart GPU activation (automatic in environments)
gpu-on                   # Manual activation: homodyne_gpu_activate_smart

# Detailed GPU status with hardware info
gpu-status               # Alias: homodyne_gpu_status_detailed  
# Shows: GPU hardware, CUDA version, JAX devices, memory usage

# Performance benchmarking
gpu-bench                # Alias: homodyne_gpu_benchmark
# Runs matrix multiplication benchmarks across sizes

# Hardware optimization
homodyne-gpu-optimize    # CLI tool from optimizer.py
# Detects hardware and applies optimal settings
```

### Automatic Smart Selection

The system intelligently chooses GPU vs CPU based on:

**GPU Automatically Used For:**

- MCMC sampling with large datasets (>1000 data points)
- Multiple chains (4+ chains with available GPU memory)
- Complex models with many parameters
- Linux systems with CUDA-compatible hardware

**CPU Automatically Used For:**

- Classical optimization (already CPU-optimized)
- Robust methods (CPU-optimized algorithms)
- Small datasets (GPU overhead not beneficial)
- Windows/macOS platforms (automatic fallback)

______________________________________________________________________

## 🚀 GPU Optimization System (`gpu/optimizer.py`)

### Advanced Hardware Detection

The GPU optimizer provides comprehensive hardware analysis and optimization:

```bash
# Generate detailed hardware report
homodyne-gpu-optimize --report
# → Shows: GPU specs, CUDA version, driver compatibility, optimal settings

# Run performance benchmark and apply optimal settings
homodyne-gpu-optimize --benchmark --apply
# → Tests performance across configurations, applies best settings

# Create custom optimization profile
homodyne-gpu-optimize --profile conservative
homodyne-gpu-optimize --profile aggressive
```

### Optimization Features

1. **Hardware Capability Detection**:

   - GPU memory and compute capability analysis
   - CUDA library version compatibility checking
   - Driver version validation
   - Thermal and power limit detection

1. **Performance Benchmarking**:

   - Matrix multiplication performance across sizes
   - Memory bandwidth testing
   - JAX compilation time measurement
   - Optimal batch size determination

1. **Automatic Configuration**:

   - XLA flag optimization based on hardware
   - Memory fraction tuning
   - Batch size recommendations
   - Environment variable management

### Custom GPU Configuration

Create custom profiles in `~/.config/homodyne/gpu_profiles.json`:

```json
{
  "profiles": {
    "conservative": {
      "memory_fraction": 0.6,
      "batch_size": 500,
      "enable_x64": false,
      "xla_flags": ["--xla_gpu_cuda_data_dir=/usr/local/cuda"]
    },
    "aggressive": {
      "memory_fraction": 0.95,
      "batch_size": 5000,
      "enable_x64": true,
      "xla_flags": [
        "--xla_gpu_enable_triton_gemm=true",
        "--xla_gpu_triton_gemm_any=true",
        "--xla_gpu_enable_latency_hiding_scheduler=true"
      ]
    }
  }
}
```

______________________________________________________________________

## 🧪 System Validation (`utils/system_validator.py`)

### Comprehensive System Testing

The system validator provides thorough testing of all homodyne components:

```bash
# Complete system validation
homodyne-validate
# → Tests all components with summary report

# Verbose validation with timing details
homodyne-validate --verbose  
# → Detailed output with performance metrics

# Component-specific testing
homodyne-validate --test gpu         # GPU system only
homodyne-validate --test completion  # Shell completion only
homodyne-validate --test integration # Component interaction

# JSON output for automation/CI
homodyne-validate --json
# → Machine-readable validation results
```

### Test Categories

1. **Environment Detection**:

   - Platform identification (Linux/Windows/macOS)
   - Python version compatibility (3.9+)
   - Virtual environment detection (conda/mamba/venv/virtualenv)
   - Shell type identification (bash/zsh/fish)

1. **Homodyne Installation Verification**:

   - Core command availability (`homodyne`, `homodyne-gpu`)
   - Help output validation
   - Module import testing
   - Version consistency checks

1. **Shell Completion Testing**:

   - Completion file presence and permissions
   - Activation script functionality
   - Alias availability (`hm`, `hc`, `hr`, `ha`)
   - Cache system operation

1. **GPU Setup Validation** (Linux only):

   - NVIDIA GPU hardware detection
   - CUDA installation verification
   - JAX device availability testing
   - Driver compatibility validation
   - Performance baseline establishment

1. **Integration Testing**:

   - Cross-component functionality
   - Module import chains
   - Script execution validation
   - Environment variable propagation

### Sample Validation Output

```
🔍 HOMODYNE SYSTEM VALIDATION REPORT
================================================================================

📊 Summary: 5/5 tests passed ✅
🎉 All systems operational!

🖥️  Environment:
   Platform: Linux x86_64
   Python: 3.12.0
   Environment: conda (homodyne)
   Shell: zsh

📋 Test Results:
----------------------------------------

✅ PASS Environment Detection (0.003s)
   Message: Detected: Linux, Python 3.12.0, conda env 'homodyne', Shell: zsh

✅ PASS Homodyne Installation (0.152s)
   Message: Found 5/5 commands, all help outputs valid
   Details: homodyne ✓, homodyne-gpu ✓, homodyne-config ✓, hm ✓, ha ✓

✅ PASS Shell Completion (0.089s)  
   Message: Found 2 completion files, aliases working
   Details: completion.sh active, cache functional (15 files)

✅ PASS GPU Setup (0.234s)
   Message: GPU ready - 1 GPU with JAX support
   Details: NVIDIA RTX 4090 (24GB), CUDA 12.3, JAX GPU backend active

✅ PASS Integration (0.067s)
   Message: All module imports successful (4/4)
   Details: Core modules, GPU optimizer, system validator all functional

💡 Recommendations:
   🚀 Your homodyne installation is optimized and ready!
   📖 Run 'hm --help' to see unified command options
   ⚡ GPU acceleration active - use 'gpu-bench' to test performance
```

______________________________________________________________________

## 🛠️ Advanced Configuration

### Environment Variables

```bash
# GPU Optimization Controls
export HOMODYNE_GPU_AUTO=1                    # Enable auto-GPU activation
export HOMODYNE_GPU_MEMORY_FRACTION=0.8       # GPU memory allocation
export HOMODYNE_BATCH_SIZE=2000               # Optimal batch size for GPU

# Shell Completion Tuning  
export HOMODYNE_COMPLETION_CACHE_TTL=300      # Cache refresh (5 min)
export HOMODYNE_COMPLETION_MAX_FILES=20       # Max files in cache
export HOMODYNE_COMPLETION_DEBUG=1            # Debug completion issues

# Advanced XLA GPU Optimization
export XLA_FLAGS="--xla_gpu_enable_triton_softmax_fusion=true \
                  --xla_gpu_triton_gemm_any=true \
                  --xla_gpu_enable_latency_hiding_scheduler=true \
                  --xla_gpu_force_compilation_parallelism=4"

# System Validation
export HOMODYNE_VALIDATE_TIMEOUT=30           # Test timeout in seconds
export HOMODYNE_VALIDATE_VERBOSE=1            # Always verbose output
```

______________________________________________________________________

## 📊 Performance Monitoring

### GPU Performance Monitoring

```bash
# Real-time GPU utilization during analysis
nvidia-smi dmon -i 0 -s pucvmet -d 1 &
hm config.json  # Run homodyne MCMC with GPU monitoring

# Memory usage tracking
nvidia-smi --query-gpu=memory.used,memory.total --format=csv -l 1

# Built-in performance benchmarking
gpu-bench                    # Quick performance test
homodyne-gpu-optimize --benchmark --report  # Comprehensive analysis
```

### System Resource Monitoring

```bash
# CPU and memory usage
htop                        # Interactive process viewer
iotop                       # I/O usage monitoring

# JAX debugging and profiling
export JAX_ENABLE_X64=1     # Enable 64-bit precision
export JAX_DEBUG_NANS=1     # Debug NaN values
export JAX_TRACE_LEVEL=2    # Detailed JAX tracing
```

______________________________________________________________________

## 🐛 Troubleshooting

### Shell Completion Issues

```bash
# Check completion system status
ls -la $CONDA_PREFIX/etc/conda/activate.d/homodyne-*
ls -la $CONDA_PREFIX/etc/zsh/homodyne-*

# Manual activation for testing
source $CONDA_PREFIX/etc/conda/activate.d/homodyne-completion.sh

# Cache debugging
export HOMODYNE_COMPLETION_DEBUG=1
homodyne --config <TAB>

# Reset completion cache
rm ~/.cache/homodyne/completion_cache
conda deactivate && conda activate your-env
```

### GPU Activation Problems

```bash
# Comprehensive GPU diagnostics
gpu-status                          # Check activation status
homodyne-validate --test gpu --verbose  # Detailed GPU testing
homodyne-gpu-optimize --report      # Hardware analysis

# Manual GPU activation
gpu-on                              # Force activation attempt
export HOMODYNE_GPU_AUTO=1          # Enable auto-activation

# CUDA troubleshooting
nvidia-smi                          # Check driver and GPU
ls -la /usr/local/cuda              # Verify CUDA installation
echo $LD_LIBRARY_PATH               # Check library paths
```

### Performance Issues

```bash
# GPU optimization and tuning
homodyne-gpu-optimize --benchmark --apply  # Find optimal settings
export HOMODYNE_BATCH_SIZE=500             # Reduce batch size
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.7  # Reduce memory usage

# CPU fallback for debugging
export JAX_PLATFORMS=cpu            # Force CPU-only mode
export HOMODYNE_GPU_AUTO=0          # Disable GPU auto-activation
```

### Environment Integration Issues

```bash
# Conda/Mamba environment problems
conda deactivate && conda activate your-env  # Reset environment
homodyne-post-install --shell zsh --force    # Reinstall integration

# venv/virtualenv manual setup
echo "source $(python -c 'import sys; print(sys.prefix)')/etc/zsh/homodyne-completion.zsh" >> ~/.zshrc

# Check environment detection
homodyne-validate --verbose        # Shows detected environment details
```

______________________________________________________________________

## 🔧 Development and Integration

### CI/CD Integration

```yaml
# .github/workflows/homodyne-test.yml
name: Homodyne System Tests
on: [push, pull_request]

jobs:
  test-homodyne:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.9", "3.10", "3.11", "3.12"]

    steps:
      - uses: actions/checkout@v4
      - name: Set up Python ${{ matrix.python-version }}
        uses: actions/setup-python@v4
        with:
          python-version: ${{ matrix.python-version }}

      - name: Install homodyne
        run: |
          pip install homodyne-analysis[jax]
          homodyne-post-install --shell bash --non-interactive

      - name: System validation
        run: homodyne-validate --json

      - name: Run analysis tests
        run: |
          homodyne --config test_config.json --method classical
          hm test_config.json  # Test unified aliases
```

### Docker Integration

```dockerfile
# GPU-enabled homodyne container
FROM nvidia/cuda:12.3-devel-ubuntu22.04

# Install Python and homodyne
RUN apt-get update && apt-get install -y python3 python3-pip
RUN pip install homodyne-analysis[jax]

# Install runtime system with GPU support
RUN homodyne-post-install --shell bash --gpu --non-interactive

# Configure for container usage  
ENV HOMODYNE_GPU_AUTO=1
ENV XLA_FLAGS="--xla_gpu_cuda_data_dir=/usr/local/cuda"
ENV HOMODYNE_VALIDATE_TIMEOUT=60

# Validate installation
RUN homodyne-validate
RUN gpu-bench  # Test GPU performance

# Set entrypoint
ENTRYPOINT ["homodyne"]
```

### IDE Integration (VS Code)

```json
{
  "python.terminal.activateEnvironment": true,
  "terminal.integrated.shellArgs.linux": [
    "-c", "source ~/.bashrc && exec bash"
  ],
  "terminal.integrated.env.linux": {
    "HOMODYNE_GPU_AUTO": "1",
    "HOMODYNE_COMPLETION_DEBUG": "0"
  },
  "files.associations": {
    "homodyne_config*.json": "jsonc"
  }
}
```

______________________________________________________________________

## 📝 Environment Support Matrix

| Environment | Shell Completion | GPU Setup | Auto-Activation | Advanced Tools |
|-------------|------------------|-----------|-----------------|----------------| |
**Conda** | ✅ Full support | ✅ Automatic | ✅ On activate | ✅ All features | | **Mamba**
| ✅ Full support | ✅ Automatic | ✅ On activate | ✅ All features |\
| **venv** | ✅ Manual setup | ✅ Manual setup | 🔶 Manual sourcing | ✅ All features | |
**virtualenv** | ✅ Manual setup | ✅ Manual setup | 🔶 Manual sourcing | ✅ All features |
| **System Python** | 🔶 User-wide | 🔶 User-wide | 🔴 Not recommended | 🔶 Limited |

**Legend:**

- ✅ Fully automated
- 🔶 Manual setup required
- 🔴 Not supported/recommended

______________________________________________________________________

## 📦 Complete System Architecture

### Installation Flow

1. **Package Installation**: `pip install homodyne-analysis[jax]`
1. **Runtime Setup**: `homodyne-post-install --shell zsh --gpu --advanced`
1. **System Integration**: Automatic activation scripts installed
1. **Validation**: `homodyne-validate` confirms all components
1. **Ready to Use**: Unified commands and GPU acceleration available

### Component Interaction

```
┌─────────────────────┐    ┌─────────────────────┐    ┌─────────────────────┐
│   Shell Completion  │    │  GPU Acceleration   │    │  System Validation  │
│   completion.sh     │    │  activation.sh      │    │  system_validator.py│
│                     │    │  optimizer.py       │    │                     │
│ • Context-aware     │◄──►│ • Smart detection   │◄──►│ • Component testing │
│ • Cached discovery  │    │ • Auto-optimization │    │ • Integration check │
│ • Command builder   │    │ • Performance tune  │    │ • Health monitoring │
└─────────────────────┘    └─────────────────────┘    └─────────────────────┘
           │                           │                           │
           └───────────────────────────┼───────────────────────────┘
                                       │
                           ┌─────────────────────┐
                           │   Unified Commands  │
                           │                     │
                           │ hm  hc  hr  ha      │
                           │ gpu-status          │
                           │ gpu-bench           │
                           │ homodyne-validate   │
                           │ homodyne-gpu-optimize│
                           └─────────────────────┘
```

______________________________________________________________________

## 🧹 Uninstallation and Cleanup

### Complete Uninstallation (Recommended)

**Always run cleanup first** to use the smart removal system:

```bash
# Step 1: Interactive cleanup (choose what to remove)
homodyne-cleanup --interactive

# Or complete unified cleanup
homodyne-cleanup

# Step 2: Uninstall the package
pip uninstall homodyne-analysis

# Step 3: Verify cleanup
homodyne-validate 2>/dev/null || echo "✅ Successfully uninstalled"
```

### Smart Interactive Cleanup

Interactive cleanup with unified system support:

```bash
homodyne-cleanup --interactive

# Choose unified system components to remove:
# ✅ Shell Completion - unified completion system & aliases
# ✅ GPU Acceleration - smart GPU detection & optimization  
# ✅ Advanced Features - GPU tools & system validation
# ✅ Legacy Files - old system files cleanup (recommended)

# Dry run to preview changes
homodyne-cleanup --dry-run
```

### Why Cleanup Order Matters

The unified cleanup system is **part of the homodyne package**. Running `pip uninstall`
first removes:

- `homodyne-cleanup` command
- `homodyne-validate` system validator
- Advanced cleanup intelligence

**Always run cleanup first** to use the smart removal system.

### Components Cleaned by Smart Cleanup

Smart cleanup removes all unified system components:

**Shell Completion (Unified)**

- `$CONDA_PREFIX/etc/zsh/homodyne-completion.zsh` (unified completion system)
- `$CONDA_PREFIX/etc/conda/activate.d/homodyne-completion.sh` (conda activation)
- `$CONDA_PREFIX/etc/bash_completion.d/homodyne-completion.bash` (bash completion)
- `$CONDA_PREFIX/share/fish/vendor_completions.d/homodyne.fish` (fish completion)

**GPU System (Smart)**

- `$CONDA_PREFIX/etc/homodyne/gpu/activation.sh` (smart GPU activation)
- `$CONDA_PREFIX/etc/conda/activate.d/homodyne-gpu.sh` (GPU environment setup)
- GPU optimization profiles and configuration

**Advanced Features**

- `$CONDA_PREFIX/bin/homodyne-gpu-optimize` (GPU optimization CLI)
- `$CONDA_PREFIX/bin/homodyne-validate` (system validation CLI)
- Performance benchmarking configuration
- System validation cache and reports

### Post-Uninstall Verification

**Verify complete removal:**

```bash
# Restart shell to complete cleanup
conda deactivate && conda activate <your-env>

# Verify unified system is removed
which hm 2>/dev/null || echo "✅ Aliases removed"
which homodyne-validate 2>/dev/null || echo "✅ Advanced features removed"
which gpu-status 2>/dev/null || echo "✅ GPU system removed"

# Check if any homodyne files remain
find "$CONDA_PREFIX" -name "*homodyne*" 2>/dev/null || echo "✅ All files cleaned"
```

**Complete verification:**

- ✅ Shell aliases (`hm`, `hc`, `hr`, `ha`) should not work
- ✅ Tab completion for `homodyne` should not work
- ✅ Advanced tools (`gpu-status`, `homodyne-validate`) should not be available
- ✅ No homodyne files should remain in environment directories

______________________________________________________________________

This runtime system provides a comprehensive, intelligent, and automated foundation for
homodyne analysis workflows across different platforms and environments.
