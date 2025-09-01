# Homodyne Installation & Uninstallation Guide 📦

**Unified installation system with smart shell completion, GPU acceleration, and advanced features.**

*Updated: 2024-08-31 - Streamlined post-install system with advanced features integration*

______________________________________________________________________

## 📦 Installation (Unified System)

### Quick Installation

```bash
# Standard installation
pip install homodyne-analysis

# With JAX GPU support
pip install homodyne-analysis[jax]

# Complete installation with all features
pip install homodyne-analysis[all]
```

### Unified Post-Installation Setup

**One-command setup (recommended):**
```bash
# Complete setup with all features
homodyne-post-install --shell zsh --gpu --advanced

# Interactive setup (choose features)
homodyne-post-install --interactive

# Basic shell completion only
homodyne-post-install --shell zsh
```

### Unified System Features

1. 🔧 **Smart Shell Completion** - Cross-shell compatibility with unified completion system
2. 🚀 **Smart GPU Acceleration** - Automatic CUDA detection and optimization  
3. 📝 **Unified Aliases** - Consistent shortcuts (`hm`, `hc`, `hr`, `ha`) across all shells
4. 🛠️ **Advanced Tools** - GPU optimization, system validation, and performance monitoring
5. 📋 **Environment Integration** - Automatic activation in conda, mamba, venv, virtualenv

### Smart Environment Support

Unified system works seamlessly with all virtual environments:

| Environment | Shell Completion | GPU Setup | Advanced Features | Auto-Activation |
|-------------|------------------|-----------|-------------------|------------------|
| **Conda** | ✅ Unified completion | ✅ Smart detection | ✅ Full integration | ✅ Automatic |
| **Mamba** | ✅ Unified completion | ✅ Smart detection | ✅ Full integration | ✅ Automatic |
| **venv** | ✅ Unified completion | ✅ Smart detection | ✅ Full integration | 🔴 Manual sourcing |
| **virtualenv** | ✅ Unified completion | ✅ Smart detection | ✅ Full integration | 🔴 Manual sourcing |

### Installation Examples (Unified System)

#### Complete Setup (Recommended)

```bash
# Create and activate environment
conda create -n homodyne python=3.12
conda activate homodyne

# Install homodyne with JAX support
pip install homodyne-analysis[jax]

# Complete unified setup
homodyne-post-install --shell zsh --gpu --advanced

# Restart shell or reactivate environment
conda deactivate && conda activate homodyne

# Test unified system
hm --help                         # homodyne --method mcmc
gpu-status                        # GPU activation status
homodyne-validate                 # System validation
```

#### Interactive Setup

```bash
# Interactive installation with choices
homodyne-post-install --interactive

# Choose features:
# ✅ Shell completion (zsh/bash/fish) - unified completion system
# ✅ GPU acceleration (Linux) - smart CUDA detection
# ✅ Advanced features - GPU optimization & system validation
```

#### venv Environment

```bash
# Create and activate venv
python -m venv homodyne-env
source homodyne-env/bin/activate

# Install homodyne
pip install homodyne-analysis

# Unified setup for venv
homodyne-post-install --shell bash --gpu

# Manual sourcing required for venv (one-time setup)
echo "source $(python -c 'import sys; print(sys.prefix)')/etc/zsh/homodyne-completion.zsh" >> ~/.zshrc

# Test unified system
hm --help                         # Works after shell restart
```

### Smart Dependency Installation

```bash
# Complete installation (recommended)
pip install homodyne-analysis[all]
homodyne-post-install --shell zsh --gpu --advanced

# GPU-optimized installation
pip install homodyne-analysis[jax]
homodyne-post-install --gpu --advanced

# CPU-only installation
pip install homodyne-analysis[mcmc]
homodyne-post-install --shell zsh

# Development installation
pip install homodyne-analysis[dev]
homodyne-post-install --shell zsh --advanced
```

### System Verification

**Comprehensive validation (recommended):**
```bash
# Complete system validation
homodyne-validate                 # Validates all installed components

# Verbose system report
homodyne-validate --verbose       # Detailed validation with timing

# Component-specific testing
homodyne-validate --test completion  # Test shell completion
homodyne-validate --test gpu         # Test GPU setup (Linux)
```

**Manual verification:**
```bash
# Test unified commands
hm --help                         # homodyne --method mcmc
ha --help                         # homodyne --method all
hc --help                         # homodyne --method classical

# Test GPU system (if installed)
gpu-status                        # GPU activation status
gpu-bench                         # GPU performance benchmark
homodyne-gpu-optimize --report    # Hardware report

# Test advanced features
homodyne-post-install --help      # Post-install help
homodyne-cleanup --help           # Cleanup options
```

______________________________________________________________________

## 🧹 Uninstallation (Unified System)

### Complete Uninstallation

**Recommended approach (unified system):** Clean up all components, then uninstall.

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

### Manual Cleanup (If Package Already Uninstalled)

**Unified system manual cleanup:**

```bash
# Remove unified completion system
rm -f "$CONDA_PREFIX/etc/zsh/homodyne-completion.zsh"
rm -f "$CONDA_PREFIX/etc/conda/activate.d/homodyne-"*
rm -f "$CONDA_PREFIX/etc/bash_completion.d/homodyne-"*
rm -f "$CONDA_PREFIX/share/fish/vendor_completions.d/homodyne.fish"

# Remove advanced features CLI tools
rm -f "$CONDA_PREFIX/bin/homodyne-gpu-optimize"
rm -f "$CONDA_PREFIX/bin/homodyne-validate"

# Clean up configuration directories
rm -rf "$CONDA_PREFIX/etc/homodyne"

# Remove any remaining GPU activation scripts
rm -f "$CONDA_PREFIX/etc/homodyne/gpu"/*
```

## 🔍 Why Cleanup Order Matters

The unified cleanup system is **part of the homodyne package**. Running `pip uninstall` first removes:
- `homodyne-cleanup` command
- `homodyne-validate` system validator  
- Advanced cleanup intelligence

**Always run cleanup first** to use the smart removal system.

## 📁 Unified System Components Cleaned

Smart cleanup removes all unified system components:

### Shell Completion (Unified)
- `$CONDA_PREFIX/etc/zsh/homodyne-completion.zsh` (unified completion system)
- `$CONDA_PREFIX/etc/conda/activate.d/homodyne-completion.sh` (conda activation)
- `$CONDA_PREFIX/etc/bash_completion.d/homodyne-completion.bash` (bash completion)
- `$CONDA_PREFIX/share/fish/vendor_completions.d/homodyne.fish` (fish completion)

### GPU System (Smart)
- `$CONDA_PREFIX/etc/homodyne/gpu/gpu_activation_smart.sh` (smart GPU activation)
- `$CONDA_PREFIX/etc/conda/activate.d/homodyne-gpu.sh` (GPU environment setup)
- GPU optimization profiles and configuration

### Advanced Features
- `$CONDA_PREFIX/bin/homodyne-gpu-optimize` (GPU optimization CLI)
- `$CONDA_PREFIX/bin/homodyne-validate` (system validation CLI)
- Performance benchmarking configuration
- System validation cache and reports

### Environment Integration
- Smart environment variable management
- Cross-shell alias consistency
- Automatic activation/deactivation scripts
- Configuration state tracking

**Unified System Total:** 15+ components across shell completion, GPU acceleration, advanced tools, and environment integration are automatically managed.

## ✅ Post-Uninstall Verification

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