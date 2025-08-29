# GPU Setup Guide for Homodyne

Complete guide for setting up GPU acceleration with the homodyne package using pip-installed NVIDIA packages.

---

## 🚀 Quick Solutions

### MCMC Failed with GPU Errors?

If you see errors like `RuntimeError: Unable to load cuSPARSE` or `CUDA-enabled jaxlib is not installed`, try these immediate fixes:

```bash
# Option 1: Activate GPU support (recommended)
source activate_gpu.sh
homodyne --config my_config.json --method mcmc

# Option 2: Use the GPU wrapper (automatic)
homodyne-gpu --config my_config.json --method mcmc

# Option 3: Force CPU-only mode
JAX_PLATFORMS=cpu homodyne --config my_config.json --method mcmc
```

**Why this happens:** pip installs NVIDIA libraries in Python's site-packages, but they're not in the system library path by default.

---

## 📦 Installation & Setup

### 1. Install with GPU Support
```bash
pip install homodyne-analysis[mcmc]     # For MCMC with GPU
pip install homodyne-analysis[jax]      # For JAX with GPU  
pip install homodyne-analysis[performance] # Full performance stack
```

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

Add to your `~/.bashrc` or `~/.zshrc`:

```bash
# Homodyne GPU support
alias hgpu='source /path/to/homodyne/activate_gpu.sh && homodyne'
```

Then use: `hgpu --config my_config.json --method mcmc`

---

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

---

## 🔧 Troubleshooting

### GPU Not Detected

1. **Check NVIDIA driver:** `nvidia-smi`
2. **Verify CUDA libraries:** `pip list | grep nvidia`
3. **Test JAX installation:** `pip show jax jaxlib`
4. **Activate GPU support:** `source activate_gpu.sh`

### Common Error Messages

| Error | Solution |
|-------|----------|
| `Unable to load cuSPARSE` | Run `source activate_gpu.sh` first |
| `CUDA-enabled jaxlib is not installed` | Use `homodyne_gpu` wrapper or activate GPU |
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

---

## 📊 Performance & Usage

### When GPU Helps Most
- **Large datasets** (>1000 data points)
- **Multiple MCMC chains** (4-8 chains)
- **Complex models** (many parameters)

### MCMC Usage
```python
from homodyne.optimization.mcmc import MCMCSampler

# GPU acceleration is automatic when available
sampler = MCMCSampler(config)
# Logs: "Using JAX backend with NumPyro NUTS for GPU acceleration"
```

### Performance Tips
- Small problems may be faster on CPU due to GPU initialization overhead
- Use `homodyne-gpu` command for consistent GPU activation
- Monitor GPU memory with `nvidia-smi`

---

## 💻 System Requirements

- **OS:** Linux (GPU acceleration not available on Windows/macOS via pip)
- **GPU:** NVIDIA GPU with CUDA capability
- **Python:** 3.12+
- **Driver:** NVIDIA driver compatible with CUDA 12.x

---

## 📝 What the Scripts Do

**`activate_gpu.sh`:**
- Finds NVIDIA libraries in your Python environment
- Sets `LD_LIBRARY_PATH` environment variable
- Enables JAX to locate CUDA libraries

**`homodyne-gpu`:**
- Automatically activates GPU support
- Executes homodyne with GPU acceleration
- No manual activation required