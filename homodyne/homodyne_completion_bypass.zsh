#!/usr/bin/env zsh
#=============================================================================
# Homodyne Shell Completion & Aliases - Virtual Environment Integration
#=============================================================================
# Purpose: Provides convenient aliases and shortcuts for homodyne commands
# Features: Auto GPU activation, virtual environment awareness, system CUDA
# Usage: source homodyne_completion_bypass.zsh
#=============================================================================

# Display virtual environment info
_homodyne_show_env() {
    if [ -n "$VIRTUAL_ENV" ]; then
        echo "📦 Active: $(basename $VIRTUAL_ENV)"
    elif [ -n "$CONDA_DEFAULT_ENV" ]; then
        echo "📦 Active: $CONDA_DEFAULT_ENV (conda)"
    else
        echo "⚠️  No virtual environment detected"
    fi
}

# Smart system CUDA GPU activation function
homodyne_gpu_activate() {
    # Check if already activated
    if [[ "$HOMODYNE_GPU_ACTIVATED" == "1" ]]; then
        return 0
    fi
    
    # Find activation script in common locations
    local script_path=""
    if [ -f "./activate_gpu.sh" ]; then
        script_path="./activate_gpu.sh"
    elif [ -f "$(dirname $0)/activate_gpu.sh" ]; then
        script_path="$(dirname $0)/activate_gpu.sh"
    elif [ -f "$HOME/.local/share/homodyne/activate_gpu.sh" ]; then
        script_path="$HOME/.local/share/homodyne/activate_gpu.sh"
    fi
    
    if [ -n "$script_path" ]; then
        source "$script_path" 2>/dev/null || echo "⚠️ GPU activation failed - using CPU mode"
    else
        # Direct system CUDA setup as fallback
        if [[ "$(uname -s)" == "Linux" ]] && [ -d "/usr/local/cuda" ]; then
            export CUDA_ROOT="/usr/local/cuda"
            export CUDA_HOME="$CUDA_ROOT"
            export PATH="$CUDA_ROOT/bin:$PATH"
            export LD_LIBRARY_PATH="$CUDA_ROOT/lib64:/usr/lib/x86_64-linux-gnu:/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH"
            export XLA_FLAGS="--xla_gpu_cuda_data_dir=$CUDA_ROOT"
            export JAX_PLATFORMS=""
            export HOMODYNE_GPU_ACTIVATED="1"
        fi
    fi
}

# CPU-only aliases
alias hm='homodyne --method mcmc'
alias hc='homodyne --method classical'  
alias hr='homodyne --method robust'
alias ha='homodyne --method all'

# GPU-accelerated aliases (with auto-activation)
alias hgm='homodyne_gpu_activate && homodyne-gpu --method mcmc'
alias hga='homodyne_gpu_activate && homodyne-gpu --method all'

# Configuration shortcuts
alias hconfig='homodyne --config'
alias hgconfig='homodyne_gpu_activate && homodyne-gpu --config'

# Plotting shortcuts
alias hexp='homodyne --plot-experimental-data'
alias hsim='homodyne --plot-simulated-data'

# homodyne-config shortcuts
alias hc-iso='homodyne-config --mode static_isotropic'
alias hc-aniso='homodyne-config --mode static_anisotropic'
alias hc-flow='homodyne-config --mode laminar_flow'

# Enhanced help function
homodyne_help() {
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "🔬 Homodyne Analysis Shortcuts & GPU Integration"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    # Show current environment
    _homodyne_show_env
    echo ""
    
    echo "📊 CPU Analysis (All Platforms):"
    echo "  hm  = homodyne --method mcmc"
    echo "  hc  = homodyne --method classical"
    echo "  hr  = homodyne --method robust"
    echo "  ha  = homodyne --method all"
    echo ""
    
    echo "🚀 GPU Analysis (Linux + CUDA 12.6 + cuDNN 9.12):"
    echo "  hgm = Auto GPU + homodyne-gpu --method mcmc"
    echo "  hga = Auto GPU + homodyne-gpu --method all"
    echo "  Note: Automatically sources activate_gpu.sh"
    echo ""
    
    echo "⚙️  Configuration & Plotting:"
    echo "  hconfig     = homodyne --config"
    echo "  hgconfig    = Auto GPU + homodyne-gpu --config"
    echo "  hexp        = homodyne --plot-experimental-data"
    echo "  hsim        = homodyne --plot-simulated-data"
    echo "  hc-iso      = homodyne-config --mode static_isotropic"
    echo "  hc-aniso    = homodyne-config --mode static_anisotropic"
    echo "  hc-flow     = homodyne-config --mode laminar_flow"
    echo ""
    
    echo "💡 Usage Examples:"
    echo "  hm config.json             # CPU MCMC"
    echo "  hgm config.json            # GPU MCMC (auto-activated)"
    echo "  hc-iso -o my_config.json   # Create isotropic config"
    echo "  source activate_gpu.sh     # Manual GPU activation"
    echo ""
    
    echo "🔧 System Requirements for GPU:"
    echo "  • Linux OS"
    echo "  • System CUDA 12.6+ at /usr/local/cuda"
    echo "  • cuDNN 9.12+ in system libraries"
    echo "  • NVIDIA GPU with driver 560.28+"
    echo "  • jax[cuda12-local] installed"
    echo ""
    
    echo "📋 Available config files:"
    local configs=(*.json(N))
    if (( ${#configs} > 0 )); then
        printf "  %s\n" "${configs[@]}"
    else
        echo "  (no .json files in current directory)"
    fi
    
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
}

# GPU status check function
homodyne_gpu_status() {
    echo "🔧 System CUDA Status Check:"
    
    # Check Linux
    if [[ "$(uname -s)" != "Linux" ]]; then
        echo "❌ Platform: $(uname -s) (GPU requires Linux)"
        return 1
    fi
    echo "✅ Platform: Linux"
    
    # Check system CUDA
    if [ -d "/usr/local/cuda" ]; then
        echo "✅ System CUDA: $(nvcc --version 2>/dev/null | grep release | sed 's/.*release //' | cut -d',' -f1 || echo 'Found but nvcc not in PATH')"
    else
        echo "❌ System CUDA: Not found at /usr/local/cuda"
        return 1
    fi
    
    # Check cuDNN
    local cudnn_files=(/usr/lib/x86_64-linux-gnu/libcudnn.so.9*(N))
    if [[ ${#cudnn_files[@]} -gt 0 ]]; then
        echo "✅ cuDNN: $(basename ${cudnn_files[1]} | sed 's/libcudnn\.so\.//')"
    else
        echo "❌ cuDNN: Not found in system libraries"
        return 1
    fi
    
    # Check GPU driver
    local driver=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null)
    if [[ -n "$driver" ]]; then
        echo "✅ NVIDIA Driver: $driver"
    else
        echo "❌ NVIDIA Driver: Not detected (nvidia-smi failed)"
        return 1
    fi
    
    # Check JAX
    python3 -c "import jax; print(f'✅ JAX: {jax.__version__()} with {len(jax.devices())} device(s)')" 2>/dev/null || echo "❌ JAX: Not installed or error"
    
    echo ""
    echo "💡 Use 'source activate_gpu.sh' to configure GPU environment"
}

# Auto-display environment on load
echo "✨ Homodyne shortcuts loaded"
_homodyne_show_env
