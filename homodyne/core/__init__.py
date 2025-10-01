"""
Core functionality for homodyne scattering analysis.

This subpackage contains the fundamental building blocks:
- Configuration management
- High-performance computational kernels
- Logging utilities

Optimized for fast startup with lazy loading of heavy computational modules.
"""

# Use lazy loading for heavy computational kernels to minimize startup time
from .lazy_imports import HeavyDependencyLoader

# Lazy-loaded kernel functions
_kernels_loader = HeavyDependencyLoader(".kernels", required=True)

# Provide lazy-loaded access to kernel functions
calculate_diffusion_coefficient_numba = (
    lambda *args, **kwargs: _kernels_loader.calculate_diffusion_coefficient_numba(
        *args, **kwargs
    )
)
calculate_shear_rate_numba = (
    lambda *args, **kwargs: _kernels_loader.calculate_shear_rate_numba(*args, **kwargs)
)
compute_g1_correlation_numba = (
    lambda *args, **kwargs: _kernels_loader.compute_g1_correlation_numba(
        *args, **kwargs
    )
)
compute_sinc_squared_numba = (
    lambda *args, **kwargs: _kernels_loader.compute_sinc_squared_numba(*args, **kwargs)
)
create_time_integral_matrix_numba = (
    lambda *args, **kwargs: _kernels_loader.create_time_integral_matrix_numba(
        *args, **kwargs
    )
)
memory_efficient_cache = lambda *args, **kwargs: _kernels_loader.memory_efficient_cache(
    *args, **kwargs
)

# Lazy-loaded config manager
ConfigManager = HeavyDependencyLoader(".config", "ConfigManager", required=True)

__all__ = [
    "ConfigManager",
    "calculate_diffusion_coefficient_numba",
    "calculate_shear_rate_numba",
    "compute_g1_correlation_numba",
    "compute_sinc_squared_numba",
    "configure_logging",
    "create_time_integral_matrix_numba",
    "memory_efficient_cache",
]
