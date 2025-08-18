"""
Homodyne Scattering Analysis Package
====================================

A comprehensive Python package for analyzing homodyne scattering in X-ray Photon
Correlation Spectroscopy (XPCS) under nonequilibrium conditions, specifically
designed for characterizing transport properties in flowing soft matter systems.

Physical Framework:
-------------------
This package implements the theoretical framework for analyzing time-dependent
intensity correlation functions g₂(φ,t₁,t₂) that capture the interplay between
Brownian diffusion and advective shear flow in complex fluids.

Reference:
H. He, H. Liang, M. Chu, Z. Jiang, J.J. de Pablo, M.V. Tirrell, S. Narayanan,
& W. Chen, "Transport coefficient approach for characterizing nonequilibrium
dynamics in soft matter", Proc. Natl. Acad. Sci. U.S.A. 121 (31) e2401162121 (2024).

THEORETICAL FOUNDATION:
-----------------------
The module implements three key equations that describe correlation functions in
nonequilibrium laminar flow systems:

    **Equation 13 - Full Nonequilibrium Laminar Flow Expression:**
    c₂(q⃗, t₁, t₂) = 1 + β[e^(-q²∫J(t)dt)] × sinc²[1/(2π) qh ∫γ̇(t)cos(φ(t))dt]

    This is the most general form describing intensity correlation functions in
    nonequilibrium systems where both diffusion and shear rate are time-dependent.
    - β: contrast parameter (depends on experimental setup and sample properties)
    - J(t): time-dependent diffusion integral accounting for non-stationary diffusion
    - ∫γ̇(t)cos(φ(t))dt: shear contribution integral with angular dependence

    **Equation S-75 - Equilibrium Condition Under Constant Shear:**
    c₂(q⃗, t₁, t₂) = 1 + β[e^(-6q²D(t₂-t₁))] sinc²[1/(2π) qh cos(φ)γ̇(t₂-t₁)]

    Simplified form for systems where diffusion coefficient D and shear rate γ̇
    are constant over the measurement time window. The factor of 6 comes from
    three-dimensional diffusion in the small-angle scattering limit.

    **Equation S-76 - One-time Correlation Function (Siegert Relation):**
    g₂(q⃗, τ) = 1 + β[e^(-6q²Dτ)] sinc²[1/(2π) qh cos(φ)γ̇τ]

    Single-time correlation function relating intensity correlations g₂ to the
    field correlation function g₁ through the Siegert relation: g₂ = 1 + β|g₁|².

PHYSICAL PARAMETERS:
====================
    - q⃗: scattering wavevector [units: Å⁻¹]
        Magnitude: q = 4π sin(θ/2)/λ, where θ is scattering angle, λ is wavelength
        Direction: determines the spatial scale and orientation of measured fluctuations

    - h: gap between stator and rotor [units: Å]
        Geometric parameter defining the shear cell dimensions
        Critical for determining shear flow velocity profile

    - φ(t): angle between shear/flow direction and q⃗ [units: degrees]
        Time-dependent angle that determines the relative orientation between
        the measured fluctuations and the applied shear flow
        cos(φ) = 0: perpendicular to flow (no shear contribution)
        cos(φ) = ±1: parallel/anti-parallel to flow (maximum shear effect)

    - γ̇(t): time-dependent shear rate [units: s⁻¹]
        Instantaneous shear rate that may vary during the measurement
        For steady shear: γ̇(t) = constant
        For creep/recovery: γ̇(t) = γ̇₀(t/t₀)^β + offset

    - D(t): time-dependent diffusion coefficient [units: Å²/s]
        Measures the rate of thermal motion and structural relaxation
        May depend on time due to aging, restructuring, or stress effects
        Typical form: D(t) = D₀(t/t₀)^α + D_offset

    - β: contrast parameter [dimensionless]
        Experimental parameter depending on:
        - Coherence properties of the X-ray beam
        - Sample thickness and scattering strength
        - Detector properties and geometry
        Theoretical maximum: β = 1 (fully coherent scattering)

    - J(t): diffusion integral function [units: Å²]
        Accumulated diffusion contribution: J(t) = ∫₀ᵗ D(t')dt'
        For constant D: J(t) = D·t
        For time-dependent D: requires numerical integration

Key Capabilities:
-----------------
- Dual Analysis Modes: Static (3 parameters) and Laminar Flow (7 parameters)
- Classical Optimization: Fast Nelder-Mead for point estimates
- Bayesian MCMC: Full posterior distributions with uncertainty quantification
- Performance Optimization: Numba JIT compilation and smart angle filtering
- Data Validation: Comprehensive quality control and visualization
- Result Management: JSON serialization with custom NumPy array handling

Core Modules:
-------------
- core.config: Configuration management with template system
- core.kernels: Optimized computational kernels for correlation functions
- core.io_utils: Data I/O with experimental data loading and result saving
- analysis.core: Main analysis engine and chi-squared fitting
- optimization.classical: Scipy-based optimization with angle filtering
- optimization.mcmc: PyMC-based Bayesian parameter estimation
- plotting: Comprehensive visualization for data validation and diagnostics

Authors: Wei Chen, Hongrui He
Institution: Argonne National Laboratory & University of Chicago
"""

from .core.config import ConfigManager, configure_logging, performance_monitor
from .core.kernels import (
    create_time_integral_matrix_numba,
    calculate_diffusion_coefficient_numba,
    calculate_shear_rate_numba,
    compute_g1_correlation_numba,
    compute_sinc_squared_numba,
    memory_efficient_cache,
    # New optimized kernels
    create_symmetric_matrix_optimized,
    matrix_vector_multiply_optimized,
    apply_scaling_vectorized,
    compute_chi_squared_fast,
    exp_negative_vectorized,
)
from .analysis.core import HomodyneAnalysisCore

# Import optimization modules with graceful degradation
# Classical optimization requires only scipy (typically available)
try:
    from .optimization.classical import ClassicalOptimizer
except ImportError as e:
    ClassicalOptimizer = None
    import logging

    logging.getLogger(__name__).warning(
        f"Classical optimization not available - missing scipy: {e}"
    )

# MCMC optimization requires PyMC ecosystem (optional advanced feature)
try:
    from .optimization.mcmc import MCMCSampler, create_mcmc_sampler
except ImportError as e:
    MCMCSampler = None
    create_mcmc_sampler = None
    import logging

    logging.getLogger(__name__).warning(
        f"MCMC Bayesian analysis not available - missing PyMC/ArviZ: {e}"
    )

# Core exports that should always be available
__all__ = [
    # Core functionality
    "ConfigManager",
    "configure_logging",
    "performance_monitor",
    "HomodyneAnalysisCore",
    # Computational kernels
    "create_time_integral_matrix_numba",
    "calculate_diffusion_coefficient_numba",
    "calculate_shear_rate_numba",
    "compute_g1_correlation_numba",
    "compute_sinc_squared_numba",
    "memory_efficient_cache",
    # Optimized kernels
    "create_symmetric_matrix_optimized",
    "matrix_vector_multiply_optimized",
    "apply_scaling_vectorized",
    "compute_chi_squared_fast",
    "exp_negative_vectorized",
    # Optimization (conditionally available)
    "ClassicalOptimizer",
    "MCMCSampler",
    "create_mcmc_sampler",
]

# Version information
__version__ = "6.0"
__author__ = "Wei Chen, Hongrui He"
__email__ = "wchen@anl.gov"
__institution__ = "Argonne National Laboratory & University of Chicago"
