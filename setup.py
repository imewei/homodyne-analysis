#!/usr/bin/env python3
"""
Setup script for the Homodyne Scattering Analysis Package with System CUDA and Conda Integration.

Most configuration is in pyproject.toml following modern Python packaging standards.
This file adds post-installation functionality for system CUDA GPU acceleration setup,
conda environment integration, shell completion, and cleanup utilities not supported by pyproject.toml.
"""

import os
import platform
from setuptools import setup
from setuptools.command.develop import develop
from setuptools.command.install import install
from setuptools.command.easy_install import easy_install


class PostInstallCommand(install):
    """Post-installation command for pip install with system CUDA and conda integration."""
    
    def run(self):
        install.run(self)
        self._run_post_install()
    
    def _run_post_install(self):
        """Run post-installation setup with system CUDA and conda integration."""
        try:
            # Only run post-install on Linux systems in virtual environments
            if platform.system() == "Linux" and self._is_virtual_environment():
                print("\n" + "="*60)
                print("🚀 Setting up Homodyne with system CUDA and conda integration...")
                # Import and run the post-install module
                from homodyne.post_install import main
                main()
                print("\n🎆 Installation complete! Available commands:")
                print("  homodyne         - Main analysis command")
                print("  homodyne-gpu     - GPU-accelerated analysis (Linux)")
                print("  homodyne-config  - Configuration generator")
                print("  homodyne-cleanup - Environment cleanup utility")
                print("\n📝 For shell completion: homodyne --install-completion zsh")
                print("="*60)
        except Exception as e:
            # Don't fail installation if post-install has issues
            print(f"Note: System CUDA post-installation setup encountered an issue: {e}")
            print("You can manually run: homodyne-post-install")
            print("For system CUDA setup, see: GPU_SETUP.md")
            print("For cleanup: homodyne-cleanup")
    
    def _is_virtual_environment(self):
        """Check if running in a virtual environment."""
        import sys
        return (
            hasattr(sys, 'real_prefix') or
            (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix) or
            os.environ.get('CONDA_DEFAULT_ENV') is not None
        )


class PostDevelopCommand(develop):
    """Post-installation command for pip install -e (development mode) with system CUDA and conda integration."""
    
    def run(self):
        develop.run(self)
        self._run_post_install()
    
    def _run_post_install(self):
        """Run post-installation setup with system CUDA and conda integration."""
        try:
            # Only run post-install on Linux systems in virtual environments
            if platform.system() == "Linux" and self._is_virtual_environment():
                print("\n" + "="*60)
                print("🚀 Setting up Homodyne development mode with system CUDA and conda integration...")
                # Import and run the post-install module
                from homodyne.post_install import main
                main()
                print("\n🎆 Development installation complete! Available commands:")
                print("  homodyne         - Main analysis command")
                print("  homodyne-gpu     - GPU-accelerated analysis (Linux)")
                print("  homodyne-config  - Configuration generator")
                print("  homodyne-cleanup - Environment cleanup utility")
                print("\n📝 For shell completion: homodyne --install-completion zsh")
                print("="*60)
        except Exception as e:
            # Don't fail installation if post-install has issues
            print(f"Note: System CUDA post-installation setup encountered an issue: {e}")
            print("You can manually run: python homodyne/post_install.py")
            print("For system CUDA setup, see: GPU_SETUP.md")
            print("For cleanup: homodyne-cleanup")
    
    def _is_virtual_environment(self):
        """Check if running in a virtual environment."""
        import sys
        return (
            hasattr(sys, 'real_prefix') or
            (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix) or
            os.environ.get('CONDA_DEFAULT_ENV') is not None
        )



# Configuration is in pyproject.toml, but add custom install commands for system CUDA and conda integration
setup(
    cmdclass={
        'install': PostInstallCommand,
        'develop': PostDevelopCommand,
    },
)
