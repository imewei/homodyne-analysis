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


def run_gpu_autoload_install():
    """Run the GPU auto-activation installation."""
    try:
        # Find the install script in multiple locations
        script_locations = []

        # 1. Try package directory (development install)
        try:
            import homodyne

            package_dir = Path(homodyne.__file__).parent.parent
            script_locations.append(package_dir / "scripts" / "install_gpu_autoload.py")
        except ImportError:
            pass

        # 2. Try site-packages (pip install)
        try:
            import site

            site_packages = Path(site.getsitepackages()[0])
            script_locations.extend(
                [
                    site_packages / "scripts" / "install_gpu_autoload.py",
                    site_packages / "homodyne" / "scripts" / "install_gpu_autoload.py",
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
    print("🔧 Configuring Homodyne with system CUDA integration...")

    # Only attempt GPU auto-activation on Linux and in virtual environments
    if is_linux() and is_virtual_environment():
        print(
            "📦 Setting up system CUDA GPU auto-activation for Linux virtual environment..."
        )
        run_gpu_autoload_install()
    elif is_linux():
        print(
            "ℹ️  System CUDA GPU auto-activation requires virtual environment, skipping"
        )
        print("ℹ️  Manual activation available with: source activate_gpu.sh")
    else:
        print(
            f"ℹ️  System CUDA GPU auto-activation not available on {platform.system()}"
        )

    show_installation_summary()


if __name__ == "__main__":
    main()
