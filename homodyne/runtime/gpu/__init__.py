"""
GPU Runtime Components
=====================

This module contains GPU acceleration runtime components including:
- gpu_wrapper.py: Command-line wrapper for homodyne-gpu
- activation.sh: Environment setup script for GPU acceleration
- optimizer.py: GPU optimization and benchmarking utilities
"""

from .optimizer import GPUOptimizer

__all__ = ["GPUOptimizer"]
