# GPU Setup Guide for Homodyne

Complete guide for setting up GPU acceleration with the homodyne package using
pip-installed NVIDIA packages.

______________________________________________________________________

## 🚀 Quick Solutions

### Choose the Right Command for Your Needs

**🖥️ CPU-Only Analysis (Recommended for most users):**
```bash
# MCMC on CPU (reliable, works on all platforms)
homodyne --config my_config.json --method mcmc

# All methods on CPU (classical + robust + MCMC)
homodyne --config my_config.json --method all

# Classical/robust methods (always CPU-only)
homodyne --config my_config.json --method classical
homodyne --config my_config.json --method robust
```

**🚀 GPU-Accelerated Analysis (Linux only):**
```bash
# MCMC with GPU acceleration (Linux with CUDA required)
homodyne-gpu --config my_config.json --method mcmc

# All methods with GPU for MCMC portion (Linux only)
homodyne-gpu --config my_config.json --method all

# ❌ These will show helpful error messages:
# homodyne-gpu --method classical  # Error: Use homodyne instead
# homodyne-gpu --method robust     # Error: Use homodyne instead
```

### MCMC Failed with GPU Errors?

If you're using `homodyne-gpu` on Linux and see errors like `RuntimeError: Unable to load cuSPARSE` or
`CUDA-enabled jaxlib is not installed`, try these fixes:

```bash
# Option 1: Activate GPU support manually first
source activate_gpu.sh
homodyne-gpu --config my_config.json --method mcmc

# Option 2: Fall back to CPU-only mode
homodyne --config my_config.json --method mcmc
```

**Why this happens:** pip installs NVIDIA libraries in Python's site-packages, but
they're not in the system library path by default.

______________________________________________________________________

## 📦 Installation & Setup

### 1. Install with GPU Support

```bash
pip install homodyne-analysis[mcmc]     # For MCMC (CPU + GPU capability)
pip install homodyne-analysis[jax]      # For JAX with GPU support
pip install homodyne-analysis[performance] # Full performance stack
```

**Note:** GPU acceleration only works on Linux. Windows/macOS installations will work but only provide CPU acceleration.

### 2. Activate GPU Support

**Method A: Use activation script**

```bash
source activate_gpu.sh
python -c "import jax; print(f'Backend: {jax.default_backend()}')"  # Should show 'gpu'
```

**Method B: Use wrapper command (automatic)**

```bash
homodyne-gpu --config my_config.json --method mcmc
```

### 3. Make It Permanent

**Option A: Source completion script (recommended)**

Add to your `~/.bashrc` or `~/.zshrc`:

```bash
# Load homodyne completion with GPU support
source /path/to/homodyne/homodyne_completion_bypass.zsh
```

This provides convenient aliases:
- `hgm` = `homodyne-gpu --method mcmc`
- `hga` = `homodyne-gpu --method all`
- `hm` = `homodyne --method mcmc` (CPU-only)
- `ha` = `homodyne --method all` (CPU-only)

**Option B: Manual aliases**

Add to your `~/.bashrc` or `~/.zshrc`:

```bash
# Homodyne GPU shortcuts
alias hgpu='source /path/to/homodyne/activate_gpu.sh && homodyne-gpu'
alias hgm='homodyne-gpu --method mcmc'
alias hga='homodyne-gpu --method all'
```

______________________________________________________________________

## ⚙️ Configuration Options

### Disable GPU (CPU-only)

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

**Via environment:**

```bash
export JAX_PLATFORMS=cpu
homodyne --config my_config.json --method mcmc
```

### Manual Library Path Setup

```bash
export LD_LIBRARY_PATH=$(python -c "
import site, os
sp = site.getsitepackages()[0]
libs = ['cublas', 'cudnn', 'cufft', 'cusolver', 'cusparse', 'nccl']
paths = [os.path.join(sp, 'nvidia', lib, 'lib') for lib in libs if os.path.exists(os.path.join(sp, 'nvidia', lib, 'lib'))]
print(':'.join(paths))
"):$LD_LIBRARY_PATH
```

______________________________________________________________________

## 🔧 Troubleshooting

### GPU Not Detected

**First verify platform:**
1. **Check operating system:** Must be Linux (Windows/macOS not supported)
2. **Check NVIDIA driver:** `nvidia-smi`
3. **Verify CUDA libraries:** `pip list | grep nvidia`
4. **Test JAX installation:** `pip show jax jaxlib`
5. **Activate GPU support:** `source activate_gpu.sh` (Linux only)

### Common Error Messages

| Error | Solution |
|-------|----------|
| `Unable to load cuSPARSE` | Run `source activate_gpu.sh` first |
| `CUDA-enabled jaxlib is not installed` | Use `homodyne-gpu` wrapper or activate GPU |
| `Out of memory` | Reduce MCMC chains or batch size |
| `JAX devices: [CpuDevice]` | GPU support not activated |

### GPU Performance Check

```python
import jax
print(f"Devices: {jax.devices()}")          # Should show [CudaDevice(id=0)]
print(f"Backend: {jax.default_backend()}")  # Should show 'gpu'

# Test GPU performance
import jax.numpy as jnp
x = jnp.ones((1000, 1000))
y = x @ x  # Should run on GPU
```

______________________________________________________________________

## 📊 Performance & Usage

### When to Use GPU vs CPU

**Use `homodyne-gpu` (GPU acceleration) when:**
- **Large datasets** (>1000 data points)
- **Multiple MCMC chains** (4-8 chains)  
- **Complex models** (many parameters)
- **Running on Linux** with CUDA-enabled JAX

**Use `homodyne` (CPU-only) when:**
- **Small to medium datasets** (reliable performance)
- **Running on Windows/macOS** (GPU not supported)
- **Prefer simplicity and reliability** over maximum speed
- **Using classical or robust methods only**

### Command Usage Examples

```bash
# CPU-only MCMC (reliable, all platforms)
homodyne --config config.json --method mcmc

# GPU-accelerated MCMC (Linux only, requires CUDA setup)
homodyne-gpu --config config.json --method mcmc

# All methods: classical + robust (CPU) + MCMC (GPU on Linux)
homodyne-gpu --config config.json --method all
```

### Performance Tips

- **Small problems** may be faster on CPU due to GPU initialization overhead
- **homodyne-gpu** automatically falls back to CPU if GPU setup fails
- **Monitor GPU memory** with `nvidia-smi` when using homodyne-gpu
- **Use homodyne for development/testing**, homodyne-gpu for production runs

______________________________________________________________________

## 💻 System Requirements

**⚠️ IMPORTANT: GPU acceleration only works on Linux**

- **OS:** Linux (Required - GPU acceleration not available on Windows/macOS)
- **GPU:** NVIDIA GPU with CUDA capability
- **Python:** 3.12+
- **Driver:** NVIDIA driver compatible with CUDA 12.x

**Platform Support:**
- ✅ **Linux**: Full GPU acceleration support with pip-installed CUDA
- ❌ **Windows**: GPU acceleration not supported (CPU-only mode automatic)
- ❌ **macOS**: GPU acceleration not supported (CPU-only mode automatic)

______________________________________________________________________

## 🔍 Detailed Troubleshooting

### Current Status Summary

If you have an **NVIDIA RTX 4090 GPU** with **CUDA 12.6 support** and all pip-installed CUDA libraries:

✅ **Hardware**: RTX 4090 GPU detected  
✅ **Driver**: NVIDIA driver 560.28.03+  
✅ **CUDA Libraries**: All pip-installed NVIDIA CUDA 12 libraries present  
✅ **JAX Installation**: JAX 0.7.1+ with CUDA 12 plugins installed  

### The cuSPARSE Loading Issue

The most common issue is JAX CUDA 12 plugin failing to initialize:

```
RuntimeError: Unable to load cuSPARSE. Is it installed?
```

Even though `libcusparse.so.12` exists in pip-installed locations, the JAX plugin cannot find it.

### Root Cause Analysis

The issue stems from JAX CUDA plugin's strict library loading requirements when:

1. Both system CUDA (`/usr/local/cuda`) and pip-installed CUDA libraries exist
2. Pip-installed cuSPARSE library links to system CUDA nvJitLink
3. JAX's cuSPARSE version check fails during plugin initialization
4. Library path conflicts between system and pip-installed CUDA versions

### Advanced Workarounds

**Force GPU Mode (Advanced Users):**

```bash
# Set comprehensive environment variables
export JAX_PLATFORMS=""
export LD_LIBRARY_PATH="/home/wei/miniforge3/envs/xpcs/lib/python3.13/site-packages/nvidia/cublas/lib:/home/wei/miniforge3/envs/xpcs/lib/python3.13/site-packages/nvidia/cudnn/lib:/home/wei/miniforge3/envs/xpcs/lib/python3.13/site-packages/nvidia/cufft/lib:/home/wei/miniforge3/envs/xpcs/lib/python3.13/site-packages/nvidia/cusolver/lib:/home/wei/miniforge3/envs/xpcs/lib/python3.13/site-packages/nvidia/cusparse/lib:/home/wei/miniforge3/envs/xpcs/lib/python3.13/site-packages/nvidia/nccl/lib:/home/wei/miniforge3/envs/xpcs/lib/python3.13/site-packages/nvidia/nvjitlink/lib:/home/wei/miniforge3/envs/xpcs/lib/python3.13/site-packages/nvidia/cuda_runtime/lib:/home/wei/miniforge3/envs/xpcs/lib/python3.13/site-packages/nvidia/cuda_cupti/lib:/home/wei/miniforge3/envs/xpcs/lib/python3.13/site-packages/nvidia/cuda_nvrtc/lib"

# Then run analysis
homodyne --config your_config.json
```

### Alternative JAX Installation Methods

**Install JAX with System CUDA:**
```bash
# Remove pip-installed CUDA JAX
pip uninstall jax jaxlib jax-cuda12-plugin

# Install with system CUDA (if available)
pip install --upgrade "jax[cuda12]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

**CPU-Only Mode (Recommended for Most Users):**
The homodyne package is highly optimized for CPU-only JAX. Unless running very large MCMC analyses, CPU performance is usually sufficient.

### Performance Comparison

**CPU vs GPU for Homodyne Analysis:**
- **Small to medium datasets**: CPU performance is excellent
- **MCMC sampling**: GPU can provide 2-3x speedup for large datasets  
- **Classical optimization**: CPU is typically sufficient

### When GPU Issues Persist

If you continue experiencing GPU setup issues:

1. **Use CPU-only mode** as the recommended default - it provides reliable performance
2. **Try the activation script** for experimental GPU support
3. **Monitor system compatibility** - JAX CUDA plugin compatibility may improve in future releases
4. **Consider GPU setup only for large-scale MCMC analyses** where the speedup justifies the setup complexity

______________________________________________________________________

## 📝 What the Scripts Do

**`activate_gpu.sh`:**

- Finds NVIDIA libraries in your Python environment
- Sets `LD_LIBRARY_PATH` environment variable
- Enables JAX to locate CUDA libraries

**`homodyne-gpu`:**

- Automatically checks platform (Linux required for GPU)
- Activates GPU support on compatible systems
- Falls back to CPU-only mode on Windows/macOS
- No manual activation required
