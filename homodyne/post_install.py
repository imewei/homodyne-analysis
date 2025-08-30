#!/usr/bin/env python3
"""
Post-installation hook for Homodyne with System CUDA Integration
================================================================

This script runs automatically after pip installation to:
1. Set up system CUDA GPU auto-activation on Linux systems
2. Install shell completion with virtual environment integration
3. Configure environment-specific settings for system CUDA

Provides seamless out-of-the-box experience with system CUDA 12.6+ and cuDNN 9.12+ support.
"""

import os
import platform
import subprocess
import sys
from pathlib import Path


def is_linux():
    """Check if running on Linux."""
    return platform.system() == "Linux"


def is_virtual_environment():
    """Check if running in a virtual environment."""
    return (
        hasattr(sys, "real_prefix")
        or (hasattr(sys, "base_prefix") and sys.base_prefix != sys.prefix)
        or os.environ.get("CONDA_DEFAULT_ENV") is not None
    )


def run_shell_completion_install():
    """Run shell completion installation for non-Linux platforms."""
    try:
        # Create shell completion installer for non-Linux
        if platform.system() == "Darwin":  # macOS
            install_macos_shell_completion()
        elif platform.system() == "Windows":
            install_windows_shell_completion()
        return True
    except Exception as e:
        print(f"ℹ️  Shell completion setup skipped: {e}")
        return False


def install_macos_shell_completion():
    """Install shell completion for macOS."""
    import sys
    from pathlib import Path

    venv_path = Path(sys.prefix)
    config_dir = venv_path / "etc" / "homodyne"
    config_dir.mkdir(parents=True, exist_ok=True)

    # Create aliases script that works on macOS
    aliases_script = config_dir / "homodyne_aliases.sh"
    script_content = """#!/bin/bash
# Homodyne Shell Aliases for macOS

# CPU-only aliases  
alias hm='homodyne --method mcmc'
alias hc='homodyne --method classical'
alias hr='homodyne --method robust'
alias ha='homodyne --method all'

# Configuration shortcuts
alias hconfig='homodyne --config'

# Plotting shortcuts
alias hexp='homodyne --plot-experimental-data'
alias hsim='homodyne --plot-simulated-data'

# homodyne-config shortcuts
alias hc-iso='homodyne-config --mode static_isotropic'
alias hc-aniso='homodyne-config --mode static_anisotropic'
alias hc-flow='homodyne-config --mode laminar_flow'

# Helper function
homodyne_help() {
    echo "Homodyne command shortcuts:"
    echo "  hc = homodyne --method classical"
    echo "  hm = homodyne --method mcmc"
    echo "  hr = homodyne --method robust"
    echo "  ha = homodyne --method all"
    echo ""
    echo "Config shortcuts:"
    echo "  hc-iso   = homodyne-config --mode static_isotropic"
    echo "  hc-aniso = homodyne-config --mode static_anisotropic"
    echo "  hc-flow  = homodyne-config --mode laminar_flow"
}
"""
    aliases_script.write_text(script_content)
    aliases_script.chmod(0o755)

    # Create activation script for conda
    activate_dir = venv_path / "etc" / "conda" / "activate.d"
    if activate_dir.parent.exists():
        activate_dir.mkdir(parents=True, exist_ok=True)
        activate_script = activate_dir / "homodyne-activate.sh"
        activate_content = f"""#!/bin/bash
# Source homodyne aliases
if [[ -f "{aliases_script}" ]]; then
    source "{aliases_script}"
fi
"""
        activate_script.write_text(activate_content)
        activate_script.chmod(0o755)
        print("✅ Shell aliases configured for macOS")
        print("   Restart your shell or reactivate your environment to use shortcuts")


def install_windows_shell_completion():
    """Install shell completion for Windows."""
    # Windows doesn't support bash aliases in the same way
    # We can create batch files or PowerShell scripts instead
    import sys
    from pathlib import Path

    venv_path = Path(sys.prefix)
    scripts_dir = venv_path / "Scripts"

    # Create batch file shortcuts for Windows
    shortcuts = {
        "hm.bat": "@homodyne --method mcmc %*",
        "hc.bat": "@homodyne --method classical %*",
        "hr.bat": "@homodyne --method robust %*",
        "ha.bat": "@homodyne --method all %*",
    }

    for name, command in shortcuts.items():
        batch_file = scripts_dir / name
        batch_file.write_text(command)

    print("✅ Shell shortcuts configured for Windows")
    print("   You can now use: hm, hc, hr, ha as shortcuts")


def run_gpu_autoload_install():
    """Run the GPU auto-activation installation."""
    try:
        # Find the install script in multiple locations
        script_locations = []

        # 1. Try package directory (development install)
        try:
            import homodyne

            if homodyne.__file__ is not None:
                package_dir = Path(homodyne.__file__).parent.parent
                script_locations.append(
                    package_dir / "scripts" / "install_gpu_autoload.py"
                )
            else:
                # For editable installs, try to find via __path__
                if hasattr(homodyne, "__path__") and homodyne.__path__:
                    homodyne_path = Path(homodyne.__path__[0])
                    package_dir = homodyne_path.parent
                    script_locations.append(
                        package_dir / "scripts" / "install_gpu_autoload.py"
                    )
        except (ImportError, AttributeError):
            pass

        # 2. Try site-packages (pip install)
        try:
            import site

            site_packages_list = site.getsitepackages()
            if site_packages_list:
                site_packages = Path(site_packages_list[0])
                script_locations.extend(
                    [
                        site_packages / "scripts" / "install_gpu_autoload.py",
                        site_packages
                        / "homodyne"
                        / "scripts"
                        / "install_gpu_autoload.py",
                    ]
                )
        except (ImportError, IndexError):
            pass

        # 3. Try current directory (fallback)
        script_locations.append(Path.cwd() / "scripts" / "install_gpu_autoload.py")

        install_script = None
        for script_path in script_locations:
            if script_path.exists():
                install_script = script_path
                break

        if not install_script:
            print("ℹ️  GPU auto-activation script not found, skipping automatic setup")
            return False

        # Run the installation silently
        result = subprocess.run(
            [sys.executable, str(install_script)],
            capture_output=True,
            text=True,
            timeout=30,
        )

        if result.returncode == 0:
            print("✅ GPU auto-activation installed successfully")
            print("   Restart your shell or run: source ~/.bashrc (or ~/.zshrc)")
            return True
        else:
            print("ℹ️  GPU auto-activation setup encountered issues (this is optional)")
            if result.stderr.strip():
                print(f"   Error: {result.stderr.strip()}")
            return False

    except Exception as e:
        print(f"ℹ️  GPU auto-activation setup skipped: {e}")
        return False


def show_installation_summary():
    """Show post-installation summary with system CUDA integration guidance."""
    print()
    print("🎉 Homodyne installation complete!")
    print()

    if is_linux():
        print("💡 Quick start on Linux:")
        print(
            "   homodyne-gpu --config config.json --method mcmc  # GPU with system CUDA"
        )
        print("   homodyne --config config.json --method mcmc      # CPU-only")
        print()
        print("🚀 System CUDA GPU setup:")
        print("   • GPU auto-activation configured for virtual environments")
        print("   • Use 'homodyne_gpu_status' to check system CUDA status")
        print("   • Use 'source activate_gpu.sh' to manually activate GPU support")
        print("   • Requires: Linux + CUDA 12.6+ + cuDNN 9.12+ + jax[cuda12-local]")
    else:
        print("💡 Quick start:")
        print("   homodyne --config config.json --method mcmc")
        print()
        print("ℹ️  GPU acceleration with system CUDA:")
        print(f"   • Not available on {platform.system()}")
        print("   • System CUDA GPU acceleration requires Linux")
        print("   • All methods work efficiently on CPU")

    print()
    print("📚 For detailed setup and usage information:")
    print("   • See GPU_SETUP.md for system CUDA configuration")
    print("   • See CLI_REFERENCE.md for command options")
    print("   • Run: homodyne --help")
    print()


def main():
    """Main post-installation routine with system CUDA integration."""
    print("🔧 Configuring Homodyne...")

    # Shell completion setup for all platforms in virtual environments
    if is_virtual_environment():
        if is_linux():
            print(
                "📦 Setting up shell completion and GPU auto-activation for Linux virtual environment..."
            )
            run_gpu_autoload_install()
        else:
            # For macOS and Windows, only install shell aliases (no GPU features)
            print(
                f"📦 Setting up shell completion for {platform.system()} virtual environment..."
            )
            run_shell_completion_install()
    elif is_linux():
        print(
            "ℹ️  Shell completion and GPU auto-activation require virtual environment, skipping"
        )
        print("ℹ️  Manual activation available with: source activate_gpu.sh")
    else:
        print(
            f"ℹ️  Shell completion requires virtual environment on {platform.system()}"
        )

    show_installation_summary()


if __name__ == "__main__":
    main()
