#!/usr/bin/env python3
"""
GPU Wrapper for Homodyne Analysis
===============================

Command-line wrapper that automatically activates GPU support before running homodyne.
This provides a seamless way to use homodyne with GPU acceleration without manual
environment setup.

Usage:
    homodyne-gpu --config my_config.json --method mcmc
    homodyne-gpu --method all --output results/

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


def activate_gpu():
    """Activate GPU support by sourcing the activation script."""
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
    # Activate GPU support
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
