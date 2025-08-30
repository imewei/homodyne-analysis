# Homodyne Installation & Uninstallation Guide

## Installation

### Standard Installation

```bash
pip install homodyne-analysis
```

**Automatic Setup (Linux + Virtual Environment)**: On Linux systems within conda or virtual environments, this automatically installs GPU auto-activation scripts, conda environment integration, and shell completion for system CUDA integration.

### Installation with Dependencies

```bash
# Full installation with all features
pip install homodyne-analysis[all]

# GPU support with JAX and system CUDA
pip install homodyne-analysis[jax]

# MCMC analysis capabilities
pip install homodyne-analysis[mcmc]
```

## Uninstallation

**⚠️ CRITICAL**: Follow this exact order to completely remove all components.

### Complete Uninstallation (Recommended)

```bash
# Step 1: Clean up environment scripts FIRST (while package is still installed)
homodyne-cleanup

# Step 2: Uninstall the package
pip uninstall homodyne-analysis
```

### If You Forgot Step 1 (Cleanup After Uninstall)

**Option A: Standalone Script**
```bash
# Download and run standalone cleanup
curl -sSL https://raw.githubusercontent.com/imewei/homodyne/main/standalone_cleanup.sh | bash

# Or from local source
bash /path/to/homodyne/standalone_cleanup.sh
```

**Option B: Manual Removal**
```bash
# Remove conda activation scripts
rm -f "$CONDA_PREFIX/etc/conda/activate.d/homodyne-gpu-activate.sh"
rm -f "$CONDA_PREFIX/etc/conda/deactivate.d/homodyne-gpu-deactivate.sh"

# Remove homodyne configuration directory
rm -rf "$CONDA_PREFIX/etc/homodyne"
```

## Why This Order Matters

The `homodyne-cleanup` command is **part of the homodyne package**. When you run `pip uninstall`, the entire package (including the cleanup command) is removed, making it impossible to clean up environment scripts afterwards.

## Files Cleaned Up

The cleanup process removes:
- `$CONDA_PREFIX/etc/conda/activate.d/homodyne-gpu-activate.sh` (conda activation hook)
- `$CONDA_PREFIX/etc/conda/deactivate.d/homodyne-gpu-deactivate.sh` (conda deactivation hook)
- `$CONDA_PREFIX/etc/homodyne/gpu_activation.sh` (GPU activation script)
- `$CONDA_PREFIX/etc/homodyne/homodyne_completion_bypass.zsh` (shell completion script)
- `$CONDA_PREFIX/etc/homodyne/homodyne_config.sh` (main configuration script)
- `$CONDA_PREFIX/etc/homodyne/` (empty directory after cleanup)

**Total**: 6 files/directories are automatically created during installation and removed during cleanup.

## Verification

After cleanup, restart your shell or reactivate your environment to ensure all scripts are properly unloaded.

```bash
# Reactivate environment to complete cleanup
conda deactivate && conda activate <your-env>
```