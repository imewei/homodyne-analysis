#!/usr/bin/env python3
"""
Install GPU Auto-activation for Homodyne
========================================

This script installs GPU auto-activation support for homodyne by:
1. Creating a GPU activation script in ~/.config/homodyne/
2. Adding auto-sourcing to shell initialization files
3. Setting up environment persistence

Usage:
    python install_gpu_autoload.py
    python install_gpu_autoload.py --uninstall
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


def get_shell_type():
    """Detect the user's shell type."""
    shell = os.environ.get("SHELL", "").lower()
    if "zsh" in shell:
        return "zsh"
    elif "bash" in shell:
        return "bash"
    elif "fish" in shell:
        return "fish"
    else:
        # Try to detect from process
        try:
            result = subprocess.run(
                ["ps", "-p", str(os.getppid()), "-o", "comm="],
                capture_output=True,
                text=True,
            )
            shell = result.stdout.strip().lower()
            if "zsh" in shell:
                return "zsh"
            elif "bash" in shell:
                return "bash"
            elif "fish" in shell:
                return "fish"
        except:
            pass
    return "bash"  # Default to bash




def get_venv_config_dir():
    """Get the virtual environment config directory."""
    import sys
    venv_path = Path(sys.prefix)
    config_dir = venv_path / "etc" / "homodyne"
    return config_dir


def get_shell_rc_file(shell_type=None):
    """Get the appropriate RC file for the shell."""
    if shell_type is None:
        shell_type = get_shell_type()
    home = Path.home()
    
    if shell_type == "zsh":
        # Check for .zshrc
        zshrc = home / ".zshrc"
        if not zshrc.exists():
            # Try .zprofile as fallback
            zprofile = home / ".zprofile"
            if zprofile.exists():
                return zprofile
        return zshrc
    elif shell_type == "bash":
        # Check for .bashrc
        bashrc = home / ".bashrc"
        if not bashrc.exists():
            # Try .bash_profile as fallback
            bash_profile = home / ".bash_profile"
            if bash_profile.exists():
                return bash_profile
        return bashrc
    elif shell_type == "fish":
        return home / ".config" / "fish" / "config.fish"
    
    return home / ".bashrc"  # Default


def create_gpu_activation_script(config_dir):
    """Create the GPU activation script in config directory using system CUDA."""
    gpu_script = config_dir / "gpu_activation.sh"
    
    # Create a clean, working script from scratch
    script_content = '''#!/bin/bash
# Homodyne GPU Auto-activation Script
# Automatically sourced to enable GPU support

# Function to activate GPU for homodyne
homodyne_gpu_activate() {
    # Check if running on Linux
    if [[ "$(uname -s)" != "Linux" ]]; then
        return 0  # Silently skip on non-Linux
    fi
    
    # Check if we've already activated in this session
    if [[ "$HOMODYNE_GPU_ACTIVATED" == "1" ]]; then
        return 0
    fi
    
    # Verify system CUDA installation
    if [ ! -d "/usr/local/cuda" ]; then
        return 0  # Silently skip if system CUDA not available
    fi
    
    # Configure environment to use system CUDA and cuDNN
    export CUDA_ROOT="/usr/local/cuda"
    export CUDA_HOME="$CUDA_ROOT"
    export PATH="$CUDA_ROOT/bin:$PATH"
    
    # Library paths: System CUDA + System cuDNN
    export LD_LIBRARY_PATH="$CUDA_ROOT/lib64:/usr/lib/x86_64-linux-gnu:/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH}"
    
    # JAX configuration for system CUDA
    export XLA_FLAGS="--xla_gpu_cuda_data_dir=$CUDA_ROOT"
    export JAX_PLATFORMS=""  # Allow GPU detection
    export HOMODYNE_GPU_ACTIVATED="1"
}

# Auto-activate if homodyne-gpu command is detected or HOMODYNE_GPU_AUTO is set
if [[ "$0" == *"homodyne-gpu"* ]] || [[ "$HOMODYNE_GPU_AUTO" == "1" ]]; then
    homodyne_gpu_activate
fi

# Create alias for manual activation
alias homodyne-gpu-activate='homodyne_gpu_activate && echo "✓ Homodyne GPU support activated"'
'''
    
    gpu_script.write_text(script_content, encoding='utf-8')
    gpu_script.chmod(0o755)
    return gpu_script


def copy_completion_script(config_dir):
    """Create a compatible shell completion script in config directory."""
    completion_script_name = "homodyne_completion_bypass.zsh"
    dest_script = config_dir / completion_script_name
    
    # Create a compatible completion script that matches the GPU activation function
    script_content = '''#!/usr/bin/env zsh
# Homodyne Shell Completion - Compatible Version
# This uses zsh's programmable completion directly

# Initialize zsh completion system if not already done
if [[ -z "$_comps" ]] && (( $+functions[compinit] == 0 )); then
    autoload -U compinit && compinit -u
fi

# Create the completion function for homodyne
_homodyne_complete() {
    local cur="${BUFFER##* }"
    local line="$BUFFER"
    local words=("${(@)${(z)line}}")

    # Get the previous word - fix the parsing logic
    local prev=""
    if (( ${#words} > 1 )); then
        # If the line ends with a space, we're completing after the last word
        if [[ "$line" =~ ' $' ]]; then
            prev="${words[-1]}"
        else
            # Otherwise we're in the middle of a word, so previous is words[-2]
            prev="${words[-2]}"
        fi
    fi

    # Generate completions based on context
    local -a completions

    case "$prev" in
        --method)
            completions=(classical mcmc robust all)
            ;;
        --config)
            completions=(*.json(N) config/*.json(N) configs/*.json(N))
            ;;
        --output-dir)
            completions=(*/(N))
            ;;
        *)
            if [[ "$cur" == --* ]]; then
                completions=(
                    --method
                    --config
                    --output-dir
                    --verbose
                    --quiet
                    --static-isotropic
                    --static-anisotropic
                    --laminar-flow
                    --plot-experimental-data
                    --plot-simulated-data
                    --contrast
                    --offset
                    --phi-angles
                )
            else
                completions=(*)
            fi
            ;;
    esac

    # Filter completions based on current input
    if [[ -n "$cur" ]]; then
        completions=(${(M)completions:#${cur}*})
    fi

    # Set up completion
    if (( ${#completions} > 0 )); then
        compadd -a completions
    fi
}

# Create the completion function for homodyne-gpu
_homodyne_gpu_complete() {
    local cur="${BUFFER##* }"
    local line="$BUFFER"
    local words=("${(@)${(z)line}}")

    # Get the previous word - fix the parsing logic
    local prev=""
    if (( ${#words} > 1 )); then
        # If the line ends with a space, we're completing after the last word
        if [[ "$line" =~ ' $' ]]; then
            prev="${words[-1]}"
        else
            # Otherwise we're in the middle of a word, so previous is words[-2]
            prev="${words[-2]}"
        fi
    fi

    # homodyne-gpu only supports mcmc and all methods (classical/robust show error)
    local -a completions

    case "$prev" in
        --method)
            completions=(mcmc all)
            ;;
        --config)
            completions=(*.json(N) config/*.json(N) configs/*.json(N))
            ;;
        --output-dir)
            completions=(*/(N))
            ;;
        *)
            if [[ "$cur" == --* ]]; then
                completions=(
                    --method
                    --config
                    --output-dir
                    --verbose
                    --quiet
                    --static-isotropic
                    --static-anisotropic
                    --laminar-flow
                    --plot-experimental-data
                    --plot-simulated-data
                    --contrast
                    --offset
                    --phi-angles
                )
            else
                completions=(*)
            fi
            ;;
    esac

    # Filter completions based on current input
    if [[ -n "$cur" ]]; then
        completions=(${(M)completions:#${cur}*})
    fi

    # Set up completion
    if (( ${#completions} > 0 )); then
        compadd -a completions
    fi
}

# Create the completion function for homodyne-config
_homodyne_config_complete() {
    local cur="${BUFFER##* }"
    local line="$BUFFER"
    local words=("${(@)${(z)line}}")

    # Get the previous word - fix the parsing logic
    local prev=""
    if (( ${#words} > 1 )); then
        # If the line ends with a space, we're completing after the last word
        if [[ "$line" =~ ' $' ]]; then
            prev="${words[-1]}"
        else
            # Otherwise we're in the middle of a word, so previous is words[-2]
            prev="${words[-2]}"
        fi
    fi

    # Generate completions based on context
    local -a completions

    case "$prev" in
        --mode|-m)
            completions=(static_isotropic static_anisotropic laminar_flow)
            ;;
        --output|-o)
            completions=(*.json(N) config/*.json(N) configs/*.json(N))
            ;;
        --sample|-s|--experiment|-e|--author|-a)
            # These don't have specific completions, just return empty
            completions=()
            ;;
        *)
            if [[ "$cur" == --* ]]; then
                completions=(
                    --mode
                    --output
                    --sample
                    --experiment
                    --author
                    --help
                )
            elif [[ "$cur" == -* ]]; then
                completions=(
                    -m
                    -o
                    -s
                    -e
                    -a
                    -h
                )
            else
                # No positional arguments for homodyne-config
                completions=()
            fi
            ;;
    esac

    # Filter completions based on current input
    if [[ -n "$cur" ]]; then
        completions=(${(M)completions:#${cur}*})
    fi

    # Set up completion
    if (( ${#completions} > 0 )); then
        compadd -a completions
    fi
}

# CPU-only aliases  
alias hm='homodyne --method mcmc'
alias hc='homodyne --method classical'
alias hr='homodyne --method robust'
alias ha='homodyne --method all'

# GPU-accelerated aliases (with auto-activation using the correct function name)
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
alias hc-config='homodyne-config'

# Also create a simple completion helper function
homodyne_help() {
    echo "Homodyne command completions:"
    echo ""
    echo "Method shortcuts:"
    echo "  hc  = homodyne --method classical"
    echo "  hm  = homodyne --method mcmc"
    echo "  hr  = homodyne --method robust"
    echo "  ha  = homodyne --method all"
    echo ""
    echo "homodyne-gpu shortcuts (GPU acceleration, Linux only):"
    echo "  hgm = homodyne-gpu --method mcmc"
    echo "  hga = homodyne-gpu --method all"
    echo ""
    echo "Note: homodyne-gpu only supports mcmc/all methods"
    echo "      For classical/robust, use regular homodyne command"
    echo ""
    echo "Other shortcuts:"
    echo "  hconfig  = homodyne --config"
    echo "  hgconfig = homodyne-gpu --config"
    echo "  hexp     = homodyne --plot-experimental-data"
    echo "  hsim     = homodyne --plot-simulated-data"
    echo ""
    echo "homodyne-config shortcuts:"
    echo "  hc-iso    = homodyne-config --mode static_isotropic"
    echo "  hc-aniso  = homodyne-config --mode static_anisotropic"
    echo "  hc-flow   = homodyne-config --mode laminar_flow"
    echo "  hc-config = homodyne-config"
    echo ""
    echo "Available methods:"
    echo "  homodyne: classical mcmc robust all (all methods)"
    echo "  homodyne-gpu: mcmc all (GPU acceleration only)"
    echo ""
    echo "Config files in current dir:"
    local configs=(*.json(N))
    if (( ${#configs} > 0 )); then
        printf "  %s\\n" "${configs[@]}"
    else
        echo "  (no .json files found)"
    fi
    echo ""
    echo "Common flags: --verbose --quiet --static-isotropic --static-anisotropic --laminar-flow"
    echo ""
    echo "GPU requirements: Linux with CUDA-enabled JAX"
}

# Try compdef registration, but don't fail if it doesn't work
# (Silent registration - no startup messages)
compdef _homodyne_complete homodyne 2>/dev/null
compdef _homodyne_gpu_complete homodyne-gpu 2>/dev/null

# For homodyne-config, compdef has issues with the dash, so use compctl as fallback
if ! compdef _homodyne_config_complete homodyne-config 2>/dev/null; then
    # Use compctl as fallback for commands with dashes
    if compctl -K _homodyne_config_complete homodyne-config 2>/dev/null; then
        # Successfully registered with compctl
        true
    else
        # If both compdef and compctl fail, provide alternative shortcuts
        echo "Note: Automatic completion for homodyne-config may not work."
        echo "Use these shortcuts instead:"
        echo "  homodyne-config --mode static_isotropic"
        echo "  homodyne-config --mode static_anisotropic"
        echo "  homodyne-config --mode laminar_flow"
    fi
fi
'''
    
    try:
        dest_script.write_text(script_content, encoding='utf-8')
        dest_script.chmod(0o755)
        print(f"✓ Created compatible completion script: {dest_script}")
        return dest_script
    except Exception as e:
        print(f"⚠️  Failed to create completion script: {e}")
        return None


def create_config_script(config_dir):
    """Create the main homodyne configuration script for system CUDA integration."""
    config_script = config_dir / "homodyne_config.sh"
    
    script_content = '''#!/bin/bash
# Homodyne Configuration Script for Virtual Environment (homodyne_config.sh)
# Provides GPU auto-activation and shell completion

# Get the directory where this script is located
# Use fixed path since BASH_SOURCE can be unreliable in conda context
if [[ -n "$CONDA_PREFIX" ]]; then
    SCRIPT_DIR="$CONDA_PREFIX/etc/homodyne"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi

# Source GPU activation if available
if [[ -f "${SCRIPT_DIR}/gpu_activation.sh" ]]; then
    source "${SCRIPT_DIR}/gpu_activation.sh"
fi

# Source shell completion if available (for zsh)
if [[ -f "${SCRIPT_DIR}/homodyne_completion_bypass.zsh" ]]; then
    # For zsh completion (only load in zsh)
    if [[ "$SHELL" == *"zsh"* ]] || [[ -n "$ZSH_VERSION" ]]; then
        # Initialize zsh completion system if needed
        if [[ -z "$_comps" ]] && command -v compinit >/dev/null; then
            autoload -U compinit && compinit -u 2>/dev/null
        fi
        source "${SCRIPT_DIR}/homodyne_completion_bypass.zsh" 2>/dev/null
    fi
    # Note: The completion script contains zsh-specific syntax
    # For bash, users can install bash-completion separately
fi

# Utility function to check GPU status
homodyne_gpu_status() {
    if [[ "$(uname -s)" != "Linux" ]]; then
        echo "GPU Status: Not supported (non-Linux platform)"
        return 1
    fi
    
    python -c "
import os
os.environ['JAX_PLATFORMS'] = ''
try:
    import jax
    devices = jax.devices()
    gpu_devices = [d for d in devices if d.platform == 'gpu']
    if gpu_devices:
        print(f'GPU Status: Active ({len(gpu_devices)} GPU(s) detected)')
        for i, d in enumerate(gpu_devices):
            print(f'  GPU {i}: {d.device_kind}')
    else:
        print('GPU Status: Not detected (CPU mode)')
except Exception as e:
    print(f'GPU Status: Error ({e})')
" 2>/dev/null || echo "GPU Status: JAX not installed"
}

# Export functions for use
export -f homodyne_gpu_activate 2>/dev/null
export -f homodyne_gpu_status 2>/dev/null

# Info message (only show once per session)
if [[ -z "$HOMODYNE_CONFIG_LOADED" ]]; then
    export HOMODYNE_CONFIG_LOADED="1"
    # Uncomment the line below to see status on shell startup
    # echo "Homodyne configuration loaded from virtual environment. Use 'homodyne_gpu_status' to check GPU status."
fi
'''
    
    config_script.write_text(script_content, encoding='utf-8')
    config_script.chmod(0o755)
    return config_script


def create_conda_activation_scripts(config_dir):
    """Create conda environment activation/deactivation scripts."""
    # Get the conda environment directory
    import sys
    conda_env_dir = Path(sys.prefix)
    conda_activate_dir = conda_env_dir / "etc" / "conda" / "activate.d"
    conda_deactivate_dir = conda_env_dir / "etc" / "conda" / "deactivate.d"
    
    # Create activation/deactivation directories
    conda_activate_dir.mkdir(parents=True, exist_ok=True)
    conda_deactivate_dir.mkdir(parents=True, exist_ok=True)
    
    # Create activation script
    activate_script = conda_activate_dir / "homodyne-gpu-activate.sh"
    activate_content = f'''#!/bin/bash
# Homodyne GPU auto-activation script for conda environment
# This script is automatically sourced when the conda environment is activated

# Source homodyne configuration if available
if [[ -f "{config_dir}/homodyne_config.sh" ]]; then
    source "{config_dir}/homodyne_config.sh"
fi
'''
    
    activate_script.write_text(activate_content, encoding='utf-8')
    activate_script.chmod(0o755)
    
    # Create deactivation script
    deactivate_script = conda_deactivate_dir / "homodyne-gpu-deactivate.sh"
    deactivate_content = '''#!/bin/bash
# Homodyne GPU deactivation script for conda environment
# This script is automatically sourced when the conda environment is deactivated

# Clean up homodyne environment variables
unset HOMODYNE_GPU_ACTIVATED
unset HOMODYNE_GPU_INTENT
unset HOMODYNE_CONFIG_LOADED

# Remove homodyne aliases and functions
unalias hgm hga hm ha hc hr 2>/dev/null || true
unalias hgconfig hconfig hexp hsim 2>/dev/null || true
unalias hc-iso hc-aniso hc-flow hc-config 2>/dev/null || true
unalias homodyne-gpu-activate 2>/dev/null || true

# Remove completion functions
unfunction _homodyne_complete _homodyne_gpu_complete _homodyne_config_complete 2>/dev/null || true
unfunction homodyne_gpu_activate homodyne_gpu_status homodyne_help 2>/dev/null || true
'''
    
    deactivate_script.write_text(deactivate_content, encoding='utf-8')
    deactivate_script.chmod(0o755)
    
    return activate_script, deactivate_script






def install_shell_integration(shell_type, config_dir):
    """Install conda environment activation hooks (no shell RC modification needed)."""
    try:
        activate_script, deactivate_script = create_conda_activation_scripts(config_dir)
        print(f"✓ Created conda activation script: {activate_script}")
        print(f"✓ Created conda deactivation script: {deactivate_script}")
        print("✓ Homodyne will automatically activate when you activate this conda environment")
        return True
    except Exception as e:
        print(f"✗ Failed to create conda activation scripts: {e}")
        return False


def uninstall_shell_integration(shell_type):
    """Remove conda activation scripts and any legacy shell integration."""
    # Remove conda activation scripts
    try:
        import sys
        conda_env_dir = Path(sys.prefix)
        conda_activate_dir = conda_env_dir / "etc" / "conda" / "activate.d"
        conda_deactivate_dir = conda_env_dir / "etc" / "conda" / "deactivate.d"
        
        activate_script = conda_activate_dir / "homodyne-gpu-activate.sh"
        deactivate_script = conda_deactivate_dir / "homodyne-gpu-deactivate.sh"
        
        if activate_script.exists():
            activate_script.unlink()
            print(f"✓ Removed conda activation script: {activate_script}")
        
        if deactivate_script.exists():
            deactivate_script.unlink()
            print(f"✓ Removed conda deactivation script: {deactivate_script}")
            
    except Exception as e:
        print(f"⚠️  Failed to remove conda scripts: {e}")
    
    # Also clean up any legacy shell RC modifications
    rc_file = get_shell_rc_file(shell_type)
    
    if not rc_file.exists():
        return True
    
    try:
        lines = rc_file.read_text().split("\n")
        new_lines = []
        skip_next = False
        
        for line in lines:
            if "Homodyne GPU Auto-activation" in line or "Homodyne Configuration" in line:
                skip_next = True
                continue
            if skip_next and ("source" in line or "fi" in line):
                skip_next = False
                continue
            if not skip_next:
                new_lines.append(line)
        
        rc_file.write_text("\n".join(new_lines), encoding='utf-8')
        print(f"✓ Removed shell integration from {rc_file}")
        return True
    except Exception as e:
        print(f"✗ Failed to remove shell integration from {rc_file}: {e}")
        return False


def main():
    import platform
    
    parser = argparse.ArgumentParser(description="Install GPU auto-activation for homodyne")
    parser.add_argument(
        "--uninstall",
        action="store_true",
        help="Uninstall GPU auto-activation"
    )
    parser.add_argument(
        "--shell",
        choices=["bash", "zsh", "fish"],
        help="Specify shell type (auto-detected by default)"
    )
    args = parser.parse_args()
    
    # Check if running on Linux
    if platform.system() != "Linux":
        print("⚠️  GPU auto-activation requires Linux. This feature is not available on non-Linux platforms.")
        print(f"   Current platform: {platform.system()}")
        print("   For Linux systems: GPU acceleration requires CUDA-enabled JAX")
        return 0
    
    # Determine shell type
    shell_type = args.shell or get_shell_type()
    print(f"Detected shell: {shell_type}")
    
    # Use virtual environment config directory
    config_dir = get_venv_config_dir()
    print(f"Using config directory: {config_dir}")
    
    if args.uninstall:
        print("\nUninstalling Homodyne GPU auto-activation...")
        
        # Remove shell integration
        uninstall_shell_integration(shell_type)
        
        # Remove config directory only if it's in virtual environment
        if config_dir.exists():
            try:
                shutil.rmtree(config_dir)
                print(f"✓ Removed configuration directory: {config_dir}")
            except Exception as e:
                print(f"✗ Failed to remove {config_dir}: {e}")
        
        # Also try to remove old global config if it exists
        global_config_dir = Path.home() / ".config" / "homodyne"
        if global_config_dir.exists():
            try:
                shutil.rmtree(global_config_dir)
                print(f"✓ Removed old global configuration directory: {global_config_dir}")
            except Exception as e:
                print(f"✗ Failed to remove old global config {global_config_dir}: {e}")
        
        print("\n✓ Uninstallation complete")
        print("  Restart your shell or run: source ~/.bashrc (or ~/.zshrc)")
        return 0
        
    else:
        print("\nInstalling Homodyne GPU auto-activation...")
        
        # Create config directory
        config_dir.mkdir(parents=True, exist_ok=True)
        print(f"✓ Created configuration directory in virtual environment: {config_dir}")
        
        # Create GPU activation script
        gpu_script = create_gpu_activation_script(config_dir)
        print(f"✓ Created GPU activation script: {gpu_script}")
        
        # Copy shell completion script
        completion_script = copy_completion_script(config_dir)
        if completion_script:
            print(f"✓ Copied shell completion script: {completion_script}")
        
        # Create main config script
        config_script = create_config_script(config_dir)
        print(f"✓ Created configuration script: {config_script}")
        
        # Install conda environment activation hooks
        if install_shell_integration(shell_type, config_dir):
            print("\n✓ Installation complete!")
            print("\nTo activate GPU support:")
            print("  1. Deactivate and reactivate your conda environment:")
            print("     conda deactivate && conda activate xpcs")
            print("  2. Or manually source the configuration:")
            print(f"     source {config_dir}/homodyne_config.sh")
            print("\nUsage after environment activation:")
            print("  homodyne-gpu --config config.json  # Auto-activates GPU")
            print("  homodyne_gpu_status                # Check GPU status")
            print("  hgm --config config.json           # Shortcut for GPU MCMC")
            print("  hga --config config.json           # Shortcut for GPU all methods")
            print(f"\n✅ Conda integration: Scripts installed in {config_dir}")
            print("🔄 GPU support automatically activates when you activate this environment!")
            print("🧹 Clean deactivation when you switch to other environments.")
        else:
            print("\n⚠ Installation partially complete")
            print(f"  Manual sourcing required: source {config_dir}/homodyne_config.sh")
            return 1
        
        return 0


if __name__ == "__main__":
    main()