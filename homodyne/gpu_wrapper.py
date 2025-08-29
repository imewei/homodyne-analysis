#!/usr/bin/env python3
"""
GPU Wrapper for Homodyne Analysis
===============================

Command-line wrapper that automatically activates GPU support before running homodyne.
This provides a seamless way to use homodyne with GPU acceleration without manual
environment setup.

The wrapper handles the common cuSPARSE loading issues by:
1. Setting up NVIDIA library paths from pip-installed packages
2. Configuring JAX environment variables for GPU detection
3. Cleaning conflicting system CUDA paths
4. Falling back to activation script if direct setup fails

Usage:
    homodyne-gpu --config my_config.json --method mcmc
    homodyne-gpu --method all --output results/

For troubleshooting GPU setup issues, see GPU_SETUP.md.

Author: Claude Code Assistant
"""

import os
import subprocess
import sys
from pathlib import Path


def find_activation_script():
    """Find the GPU activation script in common locations."""
    # Check current directory
    current_dir = Path.cwd()
    script_path = current_dir / "activate_gpu.sh"
    if script_path.exists():
        return script_path

    # Check homodyne package directory
    try:
        import homodyne

        package_dir = Path(homodyne.__file__).parent.parent
        script_path = package_dir / "activate_gpu.sh"
        if script_path.exists():
            return script_path
    except ImportError:
        pass

    # Check site-packages
    try:
        import site

        site_packages = Path(site.getsitepackages()[0])
        script_path = site_packages.parent.parent / "bin" / "activate_gpu.sh"
        if script_path.exists():
            return script_path
    except (ImportError, IndexError):
        pass

    return None


def setup_gpu_environment():
    """Set up GPU environment variables directly."""
    try:
        import platform
        import site

        # Check if running on Linux (GPU acceleration requirement)
        if platform.system() != "Linux":
            print(f"GPU acceleration not available on {platform.system()}")
            print("GPU acceleration requires Linux with CUDA-enabled JAX")
            print("Using CPU-only mode")
            return False

        site_packages = site.getsitepackages()[0]

        # Build NVIDIA library paths
        nvidia_libs = []
        for lib in [
            "cublas",
            "cudnn",
            "cufft",
            "curand",
            "cusolver",
            "cusparse",
            "nccl",
            "nvjitlink",
            "cuda_runtime",
            "cuda_cupti",
            "cuda_nvcc",
            "cuda_nvrtc",
        ]:
            lib_dir = os.path.join(site_packages, "nvidia", lib, "lib")
            if os.path.exists(lib_dir):
                nvidia_libs.append(lib_dir)

        if nvidia_libs:
            # Clean existing CUDA paths to avoid conflicts
            current_path = os.environ.get("LD_LIBRARY_PATH", "")
            clean_path = (
                ":".join(
                    [
                        p
                        for p in current_path.split(":")
                        if "/usr/local/cuda" not in p and p
                    ]
                )
                if current_path
                else ""
            )
            new_path = ":".join(nvidia_libs + ([clean_path] if clean_path else []))

            # Set environment variables
            os.environ["LD_LIBRARY_PATH"] = new_path
            os.environ["XLA_FLAGS"] = f"--xla_gpu_cuda_data_dir={site_packages}/nvidia"
            os.environ["JAX_PLATFORMS"] = ""

            print(f"GPU environment configured for Linux")
            return True
        else:
            print("No NVIDIA CUDA libraries found in pip installation")
            return False
    except Exception:
        pass

    return False


def activate_gpu():
    """Activate GPU support by setting up environment directly or using activation script."""
    # Try direct environment setup first
    if setup_gpu_environment():
        return

    # Fall back to activation script method
    script_path = find_activation_script()
    if not script_path:
        return

    try:
        # Source the activation script and capture environment changes
        result = subprocess.run(
            f"source {script_path} && env",
            shell=True,
            capture_output=True,
            text=True,
            executable="/bin/bash",
        )

        if result.returncode == 0:
            # Parse environment variables and update current environment
            for line in result.stdout.strip().split("\n"):
                if "=" in line:
                    key, value = line.split("=", 1)
                    os.environ[key] = value
    except Exception:
        # Silently continue if GPU activation fails
        pass


def main():
    """Main entry point for homodyne-gpu wrapper."""
    import platform

    # Check if method is compatible with GPU wrapper
    if "--method" in sys.argv:
        method_idx = sys.argv.index("--method") + 1
        if method_idx < len(sys.argv):
            method = sys.argv[method_idx]
            if method in ["classical", "robust"]:
                print(
                    f"homodyne-gpu: GPU acceleration not needed for --method {method}"
                )
                print("Classical and robust methods run on CPU only")
                print(f"Use 'homodyne --method {method}' instead")
                sys.exit(1)

    # Check platform requirement
    if platform.system() != "Linux":
        print(f"homodyne-gpu: GPU acceleration not supported on {platform.system()}")
        print("GPU acceleration requires Linux with CUDA-enabled JAX")
        print("Automatically falling back to standard homodyne command (CPU-only)")
        print()
    else:
        print("homodyne-gpu: Attempting to activate GPU support on Linux...")

    # Set GPU intent flag to signal that user explicitly wants GPU acceleration
    os.environ["HOMODYNE_GPU_INTENT"] = "true"

    # Activate GPU support (will handle platform checks internally)
    activate_gpu()

    # Import and run homodyne main function
    try:
        from homodyne.run_homodyne import main as homodyne_main

        homodyne_main()
    except ImportError:
        print("Error: homodyne package not found", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
