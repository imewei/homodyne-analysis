"""
Scientific Computing Type Definitions for Homodyne Analysis
===========================================================

This module defines common types used throughout the homodyne package
for scientific computing applications, improving type safety and IDE support.
"""

from typing import Any, Protocol, TypedDict

import numpy as np
from numpy.typing import NDArray

# Scientific computing array types
FloatArray = NDArray[np.floating[Any]]
IntArray = NDArray[np.integer[Any]]
BoolArray = NDArray[np.bool_[Any]]

# MCMC-specific types
class TraceDict(TypedDict, total=False):
    """Type definition for MCMC trace results."""
    posterior_means: dict[str, float]
    posterior_stds: dict[str, float]
    convergence_stats: dict[str, Any]
    chains: int
    draws: int
    rhat_values: dict[str, float]
    effective_sample_sizes: dict[str, float]

# Optimization result types
class OptimizationResult(TypedDict, total=False):
    """Type definition for optimization results."""
    success: bool
    parameters: FloatArray
    chi_squared: float
    message: str
    method: str
    optimization_time: float
    function_evaluations: int

# Protocol definitions for scientific interfaces
class PyMCModelProtocol(Protocol):
    """Protocol for PyMC model objects."""
    def sample(self, **kwargs: Any) -> Any: ...
    def compile_fn(self, **kwargs: Any) -> Any: ...

class OptimizationProtocol(Protocol):
    """Protocol for optimization interfaces."""
    def optimize_parameters(
        self,
        initial: FloatArray,
        bounds: list[tuple[float | None, float | None]]
    ) -> OptimizationResult: ...

class AnalysisProtocol(Protocol):
    """Protocol for analysis core interfaces."""
    def calculate_chi_squared(
        self,
        experimental: FloatArray,
        theoretical: FloatArray
    ) -> float: ...
