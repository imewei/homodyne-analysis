"""
Core functionality for homodyne scattering analysis.

This subpackage contains the fundamental building blocks:
- Configuration management
- High-performance computational kernels
- Logging utilities
"""

from .config import ConfigManager, configure_logging
from .kernels import (
    create_time_integral_matrix_numba,
    calculate_diffusion_coefficient_numba,
    calculate_shear_rate_numba,
    compute_g1_correlation_numba,
    compute_sinc_squared_numba,
    memory_efficient_cache
)

__all__ = [
    'ConfigManager',
    'configure_logging',
    'create_time_integral_matrix_numba',
    'calculate_diffusion_coefficient_numba',
    'calculate_shear_rate_numba', 
    'compute_g1_correlation_numba',
    'compute_sinc_squared_numba',
    'memory_efficient_cache',
]