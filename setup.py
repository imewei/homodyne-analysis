#!/usr/bin/env python3
"""
Setup script for the Homodyne Scattering Analysis Package with JAX Backend
GPU Acceleration and Unified Shell Completion System.

Most configuration is in pyproject.toml following modern Python packaging standards.
This file adds post-installation functionality for:
- JAX backend GPU acceleration setup with smart CUDA detection
- Unified shell completion system with cross-shell compatibility
- Conda/mamba environment integration with activation scripts
- Advanced GPU optimization and system validation tools
- Environment cleanup utilities not supported by pyproject.toml

Updated: 2025-09-02 - Enhanced test fixes and pytest configuration
"""

import os

from setuptools import setup
from setuptools.command.develop import develop
from setuptools.command.install import install


class PostInstallCommand(install):
    """Post-installation command for pip install with unified shell completion
    and smart GPU acceleration setup."""

    def run(self):
        install.run(self)
        self._run_post_install()

    def _run_post_install(self):
        """Run post-installation setup with unified shell completion
        and smart GPU acceleration."""
        try:
            # Only run post-install in virtual environments
            if self._is_virtual_environment():
                print("\n" + "=" * 70)
                print(
                    "🚀 Setting up Homodyne with unified shell completion and GPU acceleration..."
                )
                # Import and run the post-install module
                from homodyne.post_install import main

                main()
                print("\n🎆 Installation complete! Available commands:")
                print("  homodyne                 - Main analysis command")
                print(
                    "  homodyne-gpu             - GPU-accelerated analysis (JAX backend)"
                )
                print("  homodyne-config          - Configuration generator")
                print("  homodyne-post-install    - Setup shell completion & GPU")
                print("  homodyne-cleanup         - Environment cleanup utility")
                print(
                    "\n📝 Shell completion: Unified system with aliases (hm, hc, hr, ha)"
                )
                print("🚀 GPU acceleration: Smart CUDA detection with optimization")
                print("🔧 Advanced tools: homodyne-gpu-optimize, homodyne-validate")
                print("=" * 70)
        except Exception as e:
            # Don't fail installation if post-install has issues
            print(f"Note: Post-installation setup encountered an issue: {e}")
            print("You can manually run: homodyne-post-install")
            print("For help: homodyne-post-install --help")
            print("For cleanup: homodyne-cleanup")

    def _is_virtual_environment(self):
        """Check if running in a virtual environment."""
        import sys

        return (
            hasattr(sys, "real_prefix")
            or (hasattr(sys, "base_prefix") and sys.base_prefix != sys.prefix)
            or os.environ.get("CONDA_DEFAULT_ENV") is not None
            or os.environ.get("MAMBA_ROOT_PREFIX") is not None
            or os.environ.get("VIRTUAL_ENV") is not None
        )


class PostDevelopCommand(develop):
    """Post-installation command for pip install -e (development mode) with
    unified shell completion and smart GPU acceleration setup."""

    def run(self):
        develop.run(self)
        self._run_post_install()

    def _run_post_install(self):
        """Run post-installation setup with unified shell completion
        and smart GPU acceleration for development mode."""
        try:
            # Only run post-install in virtual environments
            if self._is_virtual_environment():
                print("\n" + "=" * 70)
                print(
                    "🚀 Setting up Homodyne development mode with unified completion and GPU..."
                )
                # Import and run the post-install module
                from homodyne.post_install import main

                main()
                print("\n🎆 Development installation complete! Available commands:")
                print("  homodyne                 - Main analysis command")
                print(
                    "  homodyne-gpu             - GPU-accelerated analysis (JAX backend)"
                )
                print("  homodyne-config          - Configuration generator")
                print("  homodyne-post-install    - Setup shell completion & GPU")
                print("  homodyne-cleanup         - Environment cleanup utility")
                print(
                    "\n📝 Shell completion: Unified system with aliases (hm, hc, hr, ha)"
                )
                print("🚀 GPU acceleration: Smart CUDA detection with optimization")
                print("🔧 Advanced tools: homodyne-gpu-optimize, homodyne-validate")
                print("💻 Development: Use 'make setup-all' for full dev environment")
                print("=" * 70)
        except Exception as e:
            # Don't fail installation if post-install has issues
            print(
                f"Note: Development post-installation setup encountered an issue: {e}"
            )
            print("You can manually run: homodyne-post-install")
            print("Or use Makefile: make setup-all")
            print("For help: homodyne-post-install --help")

    def _is_virtual_environment(self):
        """Check if running in a virtual environment."""
        import sys

        return (
            hasattr(sys, "real_prefix")
            or (hasattr(sys, "base_prefix") and sys.base_prefix != sys.prefix)
            or os.environ.get("CONDA_DEFAULT_ENV") is not None
            or os.environ.get("MAMBA_ROOT_PREFIX") is not None
            or os.environ.get("VIRTUAL_ENV") is not None
        )


# Configuration is in pyproject.toml, but add custom install commands for
# unified shell completion and smart GPU acceleration setup
setup(
    cmdclass={
        "install": PostInstallCommand,
        "develop": PostDevelopCommand,
    },
)
