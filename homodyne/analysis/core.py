"""
Core Analysis Engine for Homodyne Scattering Analysis
=====================================================

High-performance homodyne scattering analysis with configuration management.

This module implements the complete analysis pipeline for XPCS data in
nonequilibrium laminar flow systems, based on He et al. (2024).

Physical Theory
---------------
The theoretical framework describes the time-dependent intensity correlation function
g2(φ,t₁,t₂) for X-ray photon correlation spectroscopy (XPCS) measurements of fluids
under nonequilibrium laminar flow conditions. The model captures the interplay between
Brownian diffusion and advective shear flow in the two-time correlation dynamics.

The correlation function has the form:
    g2(φ,t₁,t₂) = [g1(φ,t₁,t₂)]²

where g1 is the field correlation function with separable contributions:
    g1(φ,t₁,t₂) = g1_diff(t₁,t₂) × g1_shear(φ,t₁,t₂)

Diffusion Contribution:
    g1_diff(t₁,t₂) = exp[-q²/2 ∫|t₂-t₁| D(t')dt']

Shear Contribution:
    g1_shear(φ,t₁,t₂) = [sinc(Φ(φ,t₁,t₂))]²
    Φ(φ,t₁,t₂) = (1/2π) q L cos(φ₀-φ) ∫|t₂-t₁| γ̇(t')dt'

Time-Dependent Transport Coefficients:
    D(t) = D₀ t^α + D_offset    (anomalous diffusion)
    γ̇(t) = γ̇₀ t^β + γ̇_offset   (time-dependent shear rate)

Parameter Models:
Static Mode (3 parameters):
- D₀: Reference diffusion coefficient [Å²/s]
- α: Diffusion time-dependence exponent [-]
- D_offset: Baseline diffusion [Å²/s]
(γ̇₀, β, γ̇_offset, φ₀ = 0 - automatically set and irrelevant)

Laminar Flow Mode (7 parameters):
- D₀: Reference diffusion coefficient [Å²/s]
- α: Diffusion time-dependence exponent [-]
- D_offset: Baseline diffusion [Å²/s]
- γ̇₀: Reference shear rate [s⁻¹]
- β: Shear rate time-dependence exponent [-]
- γ̇_offset: Baseline shear rate [s⁻¹]
- φ₀: Angular offset parameter [degrees]

Experimental Parameters:
- q: Scattering wavevector magnitude [Å⁻¹]
- L: Characteristic length scale (gap size) [Å]
- φ: Scattering angle [degrees]
- dt: Time step between frames [s/frame]

Features
--------
- JSON-based configuration management
- Experimental data loading with intelligent caching
- Parallel processing for multi-angle calculations
- Performance optimization with Numba JIT compilation
- Comprehensive parameter validation and bounds checking
- Memory-efficient matrix operations and caching

Usage
-----
>>> from homodyne.analysis.core import HomodyneAnalysisCore
>>> analyzer = HomodyneAnalysisCore('config.json')
>>> data = analyzer.load_experimental_data()
>>> chi2 = analyzer.calculate_chi_squared_optimized(parameters, phi_angles, data[0])

References
----------
He, H., Chen, W., et al. (2024). "Time-dependent dynamics in nonequilibrium
laminar flow systems via X-ray photon correlation spectroscopy."

Authors: Wei Chen, Hongrui He
Institution: Argonne National Laboratory & University of Chicago
"""

import logging
import multiprocessing as mp
import os
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np

# Import optional dependencies
try:
    from pyxpcsviewer import XpcsFile as xf

    PYXPCSVIEWER_AVAILABLE = True
except ImportError:
    PYXPCSVIEWER_AVAILABLE = False
    xf = None

# Import performance optimization dependencies
try:
    from numba import jit, njit, prange

    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

    def jit(*args, **kwargs):
        return args[0] if args and callable(args[0]) else lambda f: f

    def njit(*args, **kwargs):
        return args[0] if args and callable(args[0]) else lambda f: f

    prange = range

# Import core dependencies from the main module
from ..core.config import ConfigManager
from ..core.kernels import (
    create_time_integral_matrix_numba,
    calculate_diffusion_coefficient_numba,
    calculate_shear_rate_numba,
    compute_g1_correlation_numba,
    compute_sinc_squared_numba,
    memory_efficient_cache,
)

logger = logging.getLogger(__name__)

# Global optimization counter for performance tracking
OPTIMIZATION_COUNTER = 0

# Default thread count for parallelization
DEFAULT_NUM_THREADS = min(16, mp.cpu_count())

# Check for optional dependencies
try:
    import pymc  # noqa: F401

    PYMC_AVAILABLE = True
except ImportError:
    PYMC_AVAILABLE = False

try:
    import skopt  # noqa: F401

    SKOPT_AVAILABLE = True
except ImportError:
    SKOPT_AVAILABLE = False


class HomodyneAnalysisCore:
    """
    Core analysis engine for homodyne scattering data.

    This class provides the fundamental analysis capabilities including:
    - Configuration-driven parameter management
    - Experimental data loading with intelligent caching
    - Correlation function calculations with performance optimizations
    - Time-dependent diffusion and shear rate modeling
    """

    def __init__(
        self,
        config_file: str = "homodyne_config.json",
        config_override: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the core analysis system.

        Parameters
        ----------
        config_file : str
            Path to JSON configuration file
        config_override : dict, optional
            Runtime configuration overrides
        """
        # Load and validate configuration
        self.config_manager = ConfigManager(config_file)
        self.config = self.config_manager.config

        # Apply overrides if provided
        if config_override:
            self._apply_config_overrides(config_override)
            self.config_manager.setup_logging()

        # Extract core parameters
        self._initialize_parameters()

        # Setup performance optimizations
        self._setup_performance()

        # Initialize caching systems
        self._initialize_caching()

        # Warm up JIT functions
        if (
            NUMBA_AVAILABLE
            and self.config is not None
            and self.config.get("performance_settings", {}).get("warmup_numba", True)
        ):
            self._warmup_numba_functions()

        self._print_initialization_summary()

    def _initialize_parameters(self):
        """Initialize core analysis parameters from configuration."""
        if self.config is None:
            raise ValueError("Configuration not loaded: self.config is None.")
        params = self.config["analyzer_parameters"]

        # Time and frame parameters
        self.dt = params["temporal"]["dt"]
        self.start_frame = params["temporal"]["start_frame"]
        self.end_frame = params["temporal"]["end_frame"]
        self.time_length = self.end_frame - self.start_frame

        # Physical parameters
        self.wavevector_q = params["scattering"]["wavevector_q"]
        self.stator_rotor_gap = params["geometry"]["stator_rotor_gap"]

        # Parameter counts
        self.num_diffusion_params = 3
        self.num_shear_rate_params = 3

        # Pre-compute constants
        self.wavevector_q_squared = self.wavevector_q**2
        self.wavevector_q_squared_half_dt = 0.5 * self.wavevector_q_squared * self.dt
        self.sinc_prefactor = (
            0.5 / np.pi * self.wavevector_q * self.stator_rotor_gap * self.dt
        )

        # Time array
        self.time_array = np.linspace(
            self.dt,
            self.dt * self.time_length,
            self.time_length,
            dtype=np.float64,
        )

    def _setup_performance(self):
        """Configure performance settings."""
        if self.config is None:
            raise ValueError("Configuration not loaded: self.config is None.")
        params = self.config["analyzer_parameters"]
        comp_params = params.get("computational", {})

        # Thread configuration
        if comp_params.get("auto_detect_cores", False):
            detected = mp.cpu_count()
            max_threads = comp_params.get("max_threads_limit", 128)
            self.num_threads = min(detected, max_threads)
        else:
            self.num_threads = comp_params.get("num_threads", DEFAULT_NUM_THREADS)

    def _initialize_caching(self):
        """Initialize caching systems."""
        self._cache = {}
        self.cached_experimental_data = None
        self.cached_phi_angles = None

    def _warmup_numba_functions(self):
        """Pre-compile Numba functions to eliminate first-call overhead."""
        if not NUMBA_AVAILABLE:
            return

        logger.info("Warming up Numba JIT functions...")
        start_time = time.time()

        # Create small test arrays
        size = 10
        test_array = np.ones(size, dtype=np.float64)
        test_time = np.linspace(0.1, 1.0, size, dtype=np.float64)
        test_matrix = np.ones((size, size), dtype=np.float64)

        try:
            # Warm up each function
            create_time_integral_matrix_numba(test_array)
            calculate_diffusion_coefficient_numba(test_time, 1000.0, 0.0, 0.0)
            calculate_shear_rate_numba(test_time, 0.01, 0.0, 0.0)
            compute_g1_correlation_numba(test_matrix, 1.0)
            compute_sinc_squared_numba(test_matrix, 1.0)

            elapsed = time.time() - start_time
            logger.info(f"Numba warmup completed in {elapsed:.2f}s")

        except Exception as e:
            logger.warning(f"Numba warmup failed: {e}")
            logger.exception("Full traceback for Numba warmup failure:")

    def _print_initialization_summary(self):
        """Print initialization summary."""
        logger.info("HomodyneAnalysis Core initialized:")
        logger.info(
            f"  • Frames: {self.start_frame}-{self.end_frame} ({self.time_length} frames)"
        )
        logger.info(f"  • Time step: {self.dt} s/frame")
        logger.info(f"  • Wavevector: {self.wavevector_q:.6f} Å⁻¹")
        logger.info(f"  • Gap size: {self.stator_rotor_gap/1e4:.1f} μm")
        logger.info(f"  • Threads: {self.num_threads}")
        logger.info(
            f"  • Optimizations: {'Numba JIT' if NUMBA_AVAILABLE else 'Pure Python'}"
        )

    def is_static_mode(self) -> bool:
        """
        Check if the analysis is configured for static (no-flow) mode.

        In static mode:
        - Shear rate γ̇ = 0
        - Shear exponent β = 0
        - Shear offset γ̇_offset = 0
        - sinc² function = 1 (no shear decorrelation)
        - Only diffusion contribution g₁_diff remains

        Returns
        -------
        bool
            True if static mode is enabled in configuration
        """
        if self.config is None:
            return False

        # Check for static mode flag in configuration
        analysis_settings = self.config.get("analysis_settings", {})
        return analysis_settings.get("static_mode", False)

    def is_static_parameters(self, shear_params: np.ndarray) -> bool:
        """
        Check if shear parameters correspond to static conditions.

        In static conditions:
        - gamma_dot_t0 (shear_params[0]) ≈ 0
        - beta (shear_params[1]) = 0 (no time dependence)
        - gamma_dot_offset (shear_params[2]) ≈ 0

        Parameters
        ----------
        shear_params : np.ndarray
            Shear rate parameters [gamma_dot_t0, beta, gamma_dot_offset]

        Returns
        -------
        bool
            True if parameters indicate static conditions
        """
        if len(shear_params) < 3:
            return False

        gamma_dot_t0 = shear_params[0]
        beta = shear_params[1]
        gamma_dot_offset = shear_params[2]

        # Define small threshold for "effectively zero"
        threshold = 1e-10

        # Check if all shear parameters are effectively zero
        return bool(
            abs(gamma_dot_t0) < threshold
            and abs(beta) < threshold
            and abs(gamma_dot_offset) < threshold
        )

    def get_effective_parameter_count(self) -> int:
        """
        Get the effective number of parameters based on analysis mode.

        Returns
        -------
        int
            Number of parameters actually used in the analysis:
            - Static mode: 3 (only diffusion parameters: D₀, α, D_offset)
            - Laminar flow mode: 7 (all parameters including shear and φ₀)
        """
        if self.is_static_mode():
            # Static mode: only diffusion parameters are meaningful
            return self.num_diffusion_params  # 3 parameters
        else:
            # Laminar flow mode: all parameters are used
            return (
                self.num_diffusion_params + self.num_shear_rate_params + 1
            )  # 7 parameters

    def get_effective_parameters(self, parameters: np.ndarray) -> np.ndarray:
        """
        Extract only the effective parameters based on analysis mode.

        Parameters
        ----------
        parameters : np.ndarray
            Full parameter array [D0, alpha, D_offset, gamma_dot_t0, beta, gamma_dot_t_offset, phi0]

        Returns
        -------
        np.ndarray
            Effective parameters based on mode:
            - Static mode: [D0, alpha, D_offset] (shear params set to 0, phi0 ignored)
            - Laminar flow mode: all parameters as provided
        """
        if self.is_static_mode():
            # Return only diffusion parameters, set others to zero
            effective_params = np.zeros(7)  # Standard 7-parameter array
            effective_params[: self.num_diffusion_params] = parameters[
                : self.num_diffusion_params
            ]
            # Shear parameters (indices 3,4,5) remain zero
            # phi0 (index 6) remains zero - irrelevant in static mode
            return effective_params
        else:
            # Return all parameters as provided
            return parameters.copy()

    def _apply_config_overrides(self, overrides: Dict[str, Any]):
        """Apply configuration overrides with deep merging."""

        def deep_update(base, update):
            for key, value in update.items():
                if (
                    key in base
                    and isinstance(base[key], dict)
                    and isinstance(value, dict)
                ):
                    deep_update(base[key], value)
                else:
                    base[key] = value

        deep_update(self.config, overrides)
        logger.info(f"Applied {len(overrides)} configuration overrides")

    # ============================================================================
    # DATA LOADING AND PREPROCESSING
    # ============================================================================

    @memory_efficient_cache(maxsize=32)
    def load_experimental_data(
        self,
    ) -> Tuple[np.ndarray, int, np.ndarray, int]:
        """
        Load experimental correlation data with caching.

        Returns
        -------
        tuple
            (c2_experimental, time_length, phi_angles, num_angles)
        """
        logger.debug("Starting load_experimental_data method")

        # Return cached data if available
        if (
            self.cached_experimental_data is not None
            and self.cached_phi_angles is not None
        ):
            logger.debug("Cache hit: returning cached experimental data")
            return (
                self.cached_experimental_data,
                self.time_length,
                self.cached_phi_angles,
                len(self.cached_phi_angles),
            )

        # Ensure configuration is loaded
        if self.config is None:
            raise ValueError("Configuration not loaded: self.config is None.")

        # Load angle configuration
        phi_angles_path = self.config["experimental_data"].get("phi_angles_path", ".")
        phi_angles_file = self.config["experimental_data"]["phi_angles_file"]
        phi_file = os.path.join(phi_angles_path, phi_angles_file)
        logger.debug(f"Loading phi angles from: {phi_file}")
        phi_angles = np.loadtxt(phi_file, dtype=np.float64)
        num_angles = len(phi_angles)
        logger.debug(f"Loaded {num_angles} phi angles: {phi_angles}")

        # Check for cached processed data
        cache_template = self.config["experimental_data"]["cache_filename_template"]
        cache_file_path = self.config["experimental_data"].get("cache_file_path", ".")
        cache_filename = cache_template.format(
            start_frame=self.start_frame, end_frame=self.end_frame
        )
        cache_file = os.path.join(cache_file_path, cache_filename)
        logger.debug(f"Checking for cached data at: {cache_file}")

        if os.path.isfile(cache_file):
            logger.info(f"Cache hit: Loading cached data from {cache_file}")
            with np.load(cache_file) as data:
                c2_experimental = data["c2_exp"].astype(np.float64)
            logger.debug(f"Cached data shape: {c2_experimental.shape}")
        else:
            logger.info(
                f"Cache miss: Loading raw data (cache file {cache_file} not found)"
            )
            c2_experimental = self._load_raw_data(phi_angles, num_angles)
            logger.info(f"Raw data loaded with shape: {c2_experimental.shape}")

            # Save to cache
            compression_enabled = self.config["experimental_data"].get(
                "cache_compression", True
            )
            logger.debug(
                f"Saving data to cache with compression="
                f"{'enabled' if compression_enabled else 'disabled'}: "
                f"{cache_file}"
            )
            if compression_enabled:
                np.savez_compressed(cache_file, c2_exp=c2_experimental)
            else:
                np.savez(cache_file, c2_exp=c2_experimental)
            logger.debug(f"Data cached successfully to: {cache_file}")

        # Apply diagonal correction
        if self.config["advanced_settings"]["data_loading"].get(
            "use_diagonal_correction", True
        ):
            logger.debug("Applying diagonal correction to correlation matrices")
            c2_experimental = self._fix_diagonal_correction_vectorized(c2_experimental)
            logger.debug("Diagonal correction completed")

        # Cache in memory
        self.cached_experimental_data = c2_experimental
        self.cached_phi_angles = phi_angles
        logger.debug(f"Data cached in memory - final shape: {c2_experimental.shape}")

        logger.debug("load_experimental_data method completed successfully")
        return c2_experimental, self.time_length, phi_angles, num_angles

    def _load_raw_data(self, phi_angles: np.ndarray, num_angles: int) -> np.ndarray:
        """Load raw data from HDF5 files."""
        logger.debug("Starting _load_raw_data method")

        if self.config is None:
            raise ValueError("Configuration not loaded: self.config is None.")

        data_config = self.config["experimental_data"]
        folder = data_config["data_folder_path"]
        filename = data_config["data_file_name"]
        exchange_key = data_config.get("exchange_key", "exchange")

        full_path = os.path.join(folder, filename)
        logger.info(f"Opening HDF5 data file: {full_path}")
        logger.debug(f"Exchange key: {exchange_key}")
        logger.debug(
            f"Frame range: {self.start_frame}-{self.end_frame} (length: {self.time_length})"
        )

        # Open data file
        if not PYXPCSVIEWER_AVAILABLE or xf is None:
            raise ImportError(
                "pyxpcsviewer is required for loading raw experimental data. "
                "Install it with: pip install pyxpcsviewer"
            )

        try:
            data_file = xf(full_path)
            logger.debug(f"Successfully opened HDF5 file: {filename}")
        except Exception as e:
            logger.error(f"Failed to open HDF5 file {full_path}: {e}")
            raise

        # Pre-allocate output
        expected_shape = (num_angles, self.time_length, self.time_length)
        c2_experimental = np.zeros(expected_shape, dtype=np.float64)
        logger.debug(f"Pre-allocated output array with shape: {expected_shape}")

        # Load each angle
        logger.info(f"Loading data for {num_angles} angles...")
        for i in range(num_angles):
            angle_deg = phi_angles[i]
            logger.debug(f"Loading angle {i+1}/{num_angles} (φ={angle_deg:.2f}°)")

            try:
                # Fix: Pass correct_diag as bool, not int. If you want diagonal correction, set to True, else False.
                raw_data = data_file.get_twotime_c2(exchange_key, correct_diag=False)
                if raw_data is None:
                    raise ValueError(
                        f"get_twotime_c2 returned None for angle {i+1} (φ={angle_deg:.2f}°)"
                    )
                # Ensure raw_data is a NumPy array
                raw_data_np = np.array(raw_data)
                sliced_data = raw_data_np[
                    self.start_frame : self.end_frame,
                    self.start_frame : self.end_frame,
                ]
                c2_experimental[i] = sliced_data.astype(np.float64)
                logger.debug(
                    f"  Raw data shape: {raw_data_np.shape} -> sliced: {sliced_data.shape}"
                )
            except Exception as e:
                logger.error(
                    f"Failed to load data for angle {i+1} (φ={angle_deg:.2f}°): {e}"
                )
                raise

        logger.info(
            f"Successfully loaded raw data with final shape: {c2_experimental.shape}"
        )
        return c2_experimental

    def _fix_diagonal_correction_vectorized(self, c2_data: np.ndarray) -> np.ndarray:
        """Apply diagonal correction to correlation matrices."""
        if self.config is None or not (
            isinstance(self.config, dict)
            and self.config.get("advanced_settings", {})
            .get("data_loading", {})
            .get("vectorized_diagonal_fix", True)
        ):
            return c2_data

        num_angles, size, _ = c2_data.shape
        indices_i = np.arange(size - 1)
        indices_j = np.arange(1, size)

        for angle_idx in range(num_angles):
            matrix = c2_data[angle_idx]

            # Extract side-band values
            side_band = matrix[indices_i, indices_j]

            # Compute corrected diagonal
            diagonal = np.zeros(size, dtype=np.float64)
            diagonal[:-1] += side_band
            diagonal[1:] += side_band

            # Normalization
            norm = np.ones(size, dtype=np.float64)
            norm[1:-1] = 2.0

            # Apply correction
            np.fill_diagonal(matrix, diagonal / norm)

        return c2_data

    # ============================================================================
    # CORRELATION FUNCTION CALCULATIONS
    # ============================================================================

    def calculate_diffusion_coefficient_optimized(
        self, params: np.ndarray
    ) -> np.ndarray:
        """Calculate time-dependent diffusion coefficient."""
        D0, alpha, D_offset = params

        if NUMBA_AVAILABLE:
            return calculate_diffusion_coefficient_numba(
                self.time_array, D0, alpha, D_offset
            )
        else:
            return D0 * (self.time_array**alpha) + D_offset

    def calculate_shear_rate_optimized(self, params: np.ndarray) -> np.ndarray:
        """Calculate time-dependent shear rate."""
        gamma_dot_t0, beta, gamma_dot_t_offset = params

        if NUMBA_AVAILABLE:
            return calculate_shear_rate_numba(
                self.time_array, gamma_dot_t0, beta, gamma_dot_t_offset
            )
        else:
            return gamma_dot_t0 * (self.time_array**beta) + gamma_dot_t_offset

    def update_frame_range(self, start_frame: int, end_frame: int):
        """
        Update frame range with validation.

        Parameters
        ----------
        start_frame : int
            New start frame
        end_frame : int
            New end frame
        """
        if start_frame >= end_frame:
            raise ValueError(
                f"Invalid frame range: start_frame ({start_frame}) must be < end_frame ({end_frame})"
            )
        if start_frame < 0:
            raise ValueError(f"start_frame must be >= 0, got {start_frame}")

        if self.config is None:
            raise ValueError("Configuration not loaded: self.config is None.")
        min_frames = (
            self.config.get("validation_rules", {})
            .get("frame_range", {})
            .get("minimum_frames", 10)
        )
        frame_count = end_frame - start_frame
        if frame_count < min_frames:
            raise ValueError(f"Frame count ({frame_count}) must be >= {min_frames}")

        # Update configuration and recalculate derived parameters
        if self.config is None:
            raise ValueError("Configuration not loaded: self.config is None.")
        self.config["analyzer_parameters"]["temporal"]["start_frame"] = start_frame
        self.config["analyzer_parameters"]["temporal"]["end_frame"] = end_frame

        # Re-initialize parameters with new frame range
        self._initialize_parameters()
        logger.info(
            f"Frame range updated to {start_frame}-{end_frame} ({frame_count} frames)"
        )

    @memory_efficient_cache(maxsize=64)
    def create_time_integral_matrix_cached(
        self, param_hash: str, time_array: np.ndarray
    ) -> np.ndarray:
        """Create cached time integral matrix."""
        if self.config is None or not self.config.get("advanced_settings", {}).get(
            "integral_matrix", {}
        ).get("cache_matrices", True):
            # Direct computation when caching is disabled or config is None
            if NUMBA_AVAILABLE:
                return create_time_integral_matrix_numba(time_array)
            else:
                array_length = len(time_array)
                cumulative_sum = np.cumsum(time_array)
                cumsum_matrix = np.tile(cumulative_sum, (array_length, 1))
                return np.abs(cumsum_matrix - cumsum_matrix.T)

        if NUMBA_AVAILABLE:
            return create_time_integral_matrix_numba(time_array)
        else:
            n = len(time_array)
            cumsum = np.cumsum(time_array)
            cumsum_matrix = np.tile(cumsum, (n, 1))
            return np.abs(cumsum_matrix - cumsum_matrix.T)

    def calculate_c2_single_angle_optimized(
        self, parameters: np.ndarray, phi_angle: float
    ) -> np.ndarray:
        """
        Calculate correlation function for a single angle.

        Supports both laminar flow and static (no-flow) cases:
        - Laminar flow: Full 7-parameter model with diffusion and shear contributions
        - Static case: Only diffusion contribution (sinc² = 1), φ₀ irrelevant and set to 0

        Parameters
        ----------
        parameters : np.ndarray
            Model parameters [D0, alpha, D_offset, gamma_dot_t0, beta, gamma_dot_t_offset, phi0]
            In static mode: only first 3 diffusion parameters are used, others ignored/set to 0
        phi_angle : float
            Scattering angle in degrees

        Returns
        -------
        np.ndarray
            Correlation matrix c2(t1, t2)
        """
        # Check if we're in static mode
        static_mode = self.is_static_mode()

        # Extract parameters
        diffusion_params = parameters[: self.num_diffusion_params]
        shear_params = parameters[
            self.num_diffusion_params : self.num_diffusion_params
            + self.num_shear_rate_params
        ]
        phi_offset = parameters[-1]

        # Calculate time-dependent quantities
        param_hash = hash(tuple(parameters))
        D_t = self.calculate_diffusion_coefficient_optimized(diffusion_params)

        # Create diffusion integral matrix
        D_integral = self.create_time_integral_matrix_cached(f"D_{param_hash}", D_t)

        # Compute g1 correlation (diffusion contribution)
        if NUMBA_AVAILABLE:
            g1 = compute_g1_correlation_numba(
                D_integral, self.wavevector_q_squared_half_dt
            )
        else:
            g1 = np.exp(-self.wavevector_q_squared_half_dt * D_integral)

        # Handle shear contribution based on mode
        if static_mode or self.is_static_parameters(shear_params):
            # Static case: sinc² = 1 (no shear contribution)
            # g₁(t₁,t₂) = g₁_diff(t₁,t₂) = exp[-q²/2 ∫|t₂-t₁| D(t')dt']
            # g₂(t₁,t₂) = [g₁(t₁,t₂)]²
            # Note: φ₀ is irrelevant in static mode since shear term is not used
            sinc2 = np.ones_like(g1)
        else:
            # Laminar flow case: calculate full sinc² contribution
            gamma_dot_t = self.calculate_shear_rate_optimized(shear_params)
            gamma_integral = self.create_time_integral_matrix_cached(
                f"gamma_{param_hash}", gamma_dot_t
            )

            # Compute sinc² (shear contribution)
            angle_rad = np.deg2rad(phi_offset - phi_angle)
            cos_phi = np.cos(angle_rad)
            prefactor = self.sinc_prefactor * cos_phi

            if NUMBA_AVAILABLE:
                sinc2 = compute_sinc_squared_numba(gamma_integral, prefactor)
            else:
                arg = prefactor * gamma_integral
                sinc2 = np.sinc(arg) ** 2

        # Combine contributions: c2 = (g1 × sinc²)²
        return (sinc2 * g1) ** 2

    def calculate_c2_nonequilibrium_laminar_parallel(
        self, parameters: np.ndarray, phi_angles: np.ndarray
    ) -> np.ndarray:
        """
        Calculate correlation function for all angles with parallel processing.

        Parameters
        ----------
        parameters : np.ndarray
            Model parameters
        phi_angles : np.ndarray
            Array of scattering angles

        Returns
        -------
        np.ndarray
            3D array of correlation matrices [angles, time, time]
        """
        num_angles = len(phi_angles)
        use_parallel = True
        if self.config is not None:
            use_parallel = self.config.get("performance_settings", {}).get(
                "parallel_execution", True
            )

        # Avoid threading conflicts with Numba parallel operations
        if (
            self.num_threads == 1
            or num_angles < 4
            or not use_parallel
            or NUMBA_AVAILABLE
        ):
            # Sequential processing (Numba will handle internal parallelization)
            c2_calculated = np.zeros(
                (num_angles, self.time_length, self.time_length),
                dtype=np.float64,
            )

            for i in range(num_angles):
                c2_calculated[i] = self.calculate_c2_single_angle_optimized(
                    parameters, phi_angles[i]
                )

            return c2_calculated

        else:
            # Parallel processing (only when Numba not available)
            use_threading = True
            if self.config is not None:
                use_threading = self.config.get("performance_settings", {}).get(
                    "use_threading", True
                )
            Executor = ThreadPoolExecutor if use_threading else ProcessPoolExecutor

            with Executor(max_workers=self.num_threads) as executor:
                futures = [
                    executor.submit(
                        self.calculate_c2_single_angle_optimized,
                        parameters,
                        angle,
                    )
                    for angle in phi_angles
                ]

                c2_calculated = np.zeros(
                    (num_angles, self.time_length, self.time_length),
                    dtype=np.float64,
                )
                for i, future in enumerate(futures):
                    c2_calculated[i] = future.result()

                return c2_calculated

    def calculate_chi_squared_optimized(
        self,
        parameters: np.ndarray,
        phi_angles: np.ndarray,
        c2_experimental: np.ndarray,
        method_name: str = "",
        return_components: bool = False,
        filter_angles_for_optimization: bool = False,
    ) -> Union[float, Dict[str, Any]]:
        """
        Calculate chi-squared goodness of fit.

        Parameters
        ----------
        parameters : np.ndarray
            Model parameters
        phi_angles : np.ndarray
            Scattering angles
        c2_experimental : np.ndarray
            Experimental correlation data
        method_name : str
            Name of optimization method (for logging)
        return_components : bool
            If True, return detailed results dictionary
        filter_angles_for_optimization : bool
            If True, only include angles in ranges [-10°, 10°] and [170°, 190°] in chi-squared sum

        Returns
        -------
        float or dict
            Reduced chi-squared value or detailed results
        """
        global OPTIMIZATION_COUNTER

        # Parameter validation
        if self.config is None:
            raise ValueError("Configuration not loaded: self.config is None.")
        validation = self.config["advanced_settings"]["chi_squared_calculation"][
            "validity_check"
        ]

        diffusion_params = parameters[: self.num_diffusion_params]
        shear_params = parameters[
            self.num_diffusion_params : self.num_diffusion_params
            + self.num_shear_rate_params
        ]

        # Quick validity checks
        if validation.get("check_positive_D0", True) and diffusion_params[0] <= 0:
            return (
                np.inf
                if not return_components
                else {
                    "chi_squared": np.inf,
                    "valid": False,
                    "reason": "Negative D0",
                }
            )

        if validation.get("check_positive_gamma_dot_t0", True) and shear_params[0] <= 0:
            return (
                np.inf
                if not return_components
                else {
                    "chi_squared": np.inf,
                    "valid": False,
                    "reason": "Negative gamma_dot_t0",
                }
            )

        # Check parameter bounds
        if validation.get("check_parameter_bounds", True):
            bounds = self.config.get("parameter_space", {}).get("bounds", [])
            for i, bound in enumerate(bounds):
                if i < len(parameters):
                    param_val = parameters[i]
                    param_min = bound.get("min", -np.inf)
                    param_max = bound.get("max", np.inf)

                    if not (param_min <= param_val <= param_max):
                        reason = f'Parameter {bound.get("name", f"p{i}")
                                                        } out of bounds'
                        return (
                            np.inf
                            if not return_components
                            else {
                                "chi_squared": np.inf,
                                "valid": False,
                                "reason": reason,
                            }
                        )

        try:
            # Calculate theoretical correlation
            c2_theory = self.calculate_c2_nonequilibrium_laminar_parallel(
                parameters, phi_angles
            )

            # Chi-squared calculation
            chi_config = self.config["advanced_settings"]["chi_squared_calculation"]
            uncertainty_factor = chi_config.get("uncertainty_estimation_factor", 0.1)
            min_sigma = chi_config.get("minimum_sigma", 1e-10)

            # Calculate parameters for DOF calculation
            n_params = len(parameters)

            # Angle filtering for optimization
            if filter_angles_for_optimization:
                # Get target angle ranges from ConfigManager if available
                target_ranges = [
                    (-10.0, 10.0),
                    (170.0, 190.0),
                ]  # Default ranges
                if hasattr(self, "config_manager") and self.config_manager:
                    target_ranges = self.config_manager.get_target_angle_ranges()
                elif hasattr(self, "config") and self.config:
                    angle_config = self.config.get("optimization_config", {}).get(
                        "angle_filtering", {}
                    )
                    config_ranges = angle_config.get("target_ranges", [])
                    if config_ranges:
                        target_ranges = [
                            (
                                r.get("min_angle", -10.0),
                                r.get("max_angle", 10.0),
                            )
                            for r in config_ranges
                        ]

                # Find indices of angles in target ranges
                optimization_indices = []
                for i, angle in enumerate(phi_angles):
                    for min_angle, max_angle in target_ranges:
                        if min_angle <= angle <= max_angle:
                            optimization_indices.append(i)
                            break

                logger.debug(
                    f"Filtering angles for optimization: using {len(optimization_indices)}/{len(phi_angles)} angles"
                )
                if optimization_indices:
                    filtered_angles = phi_angles[optimization_indices]
                    logger.debug(f"Optimization angles: {filtered_angles.tolist()}")
                else:
                    # Check if fallback is enabled
                    should_fallback = True
                    if hasattr(self, "config_manager") and self.config_manager:
                        should_fallback = (
                            self.config_manager.should_fallback_to_all_angles()
                        )
                    elif hasattr(self, "config") and self.config:
                        angle_config = self.config.get("optimization_config", {}).get(
                            "angle_filtering", {}
                        )
                        should_fallback = angle_config.get(
                            "fallback_to_all_angles", True
                        )

                    if should_fallback:
                        logger.warning(
                            f"No angles found in target optimization ranges {target_ranges}"
                        )
                        logger.warning(
                            "Falling back to using all angles for optimization"
                        )
                        optimization_indices = list(
                            range(len(phi_angles))
                        )  # Fall back to all angles
                    else:
                        raise ValueError(
                            f"No angles found in target optimization ranges {target_ranges} and fallback disabled"
                        )
            else:
                optimization_indices = list(range(len(phi_angles)))

            total_chi2 = 0.0
            angle_chi2 = []
            angle_chi2_reduced = []
            angle_data_points = []
            scaling_solutions = []

            # Calculate chi-squared for all angles (for detailed results)
            for i in range(len(phi_angles)):
                theory = c2_theory[i].ravel()
                exp = c2_experimental[i].ravel()
                n_data_angle = len(exp)  # Number of data points for this angle

                # Optimal scaling
                if chi_config.get("scaling_optimization", True):
                    A = np.vstack([theory, np.ones(len(theory))]).T
                    scaling, residuals, _, _ = np.linalg.lstsq(A, exp, rcond=None)

                    if len(scaling) == 2:
                        contrast, offset = scaling
                        fitted = theory * contrast + offset
                        scaling_solutions.append([contrast, offset])
                    else:
                        fitted = theory
                        scaling_solutions.append([1.0, 0.0])
                else:
                    fitted = theory
                    scaling_solutions.append([1.0, 0.0])

                # Calculate chi-squared for this angle
                residuals = fitted - exp
                sigma = max(np.std(exp) * uncertainty_factor, min_sigma)
                chi2_angle = np.sum(residuals**2) / (sigma**2)

                # Calculate reduced chi-squared for this angle
                dof_angle = max(n_data_angle - n_params, 1)  # DOF for this angle
                chi2_reduced_angle = chi2_angle / dof_angle

                angle_chi2.append(chi2_angle)
                angle_chi2_reduced.append(chi2_reduced_angle)
                angle_data_points.append(n_data_angle)

                # Only include this angle in total chi2 if it's in the optimization set
                if not filter_angles_for_optimization or i in optimization_indices:
                    total_chi2 += chi2_angle

            # Reduced chi-squared calculation
            if filter_angles_for_optimization:
                # Use only data from optimization angles for DOF calculation
                n_data_optimization = sum(
                    angle_data_points[i] for i in optimization_indices
                )
                dof = max(n_data_optimization - n_params, 1)
                logger.debug(
                    f"Optimization mode: using {n_data_optimization} data points from {len(optimization_indices)} angles"
                )
            else:
                # Use all data for DOF calculation
                n_data = c2_experimental.size
                dof = max(n_data - n_params, 1)

            reduced_chi2 = total_chi2 / dof

            # Logging
            OPTIMIZATION_COUNTER += 1
            log_freq = self.config["performance_settings"].get(
                "optimization_counter_log_frequency", 50
            )
            if OPTIMIZATION_COUNTER % log_freq == 0:
                logger.info(
                    f"Iteration {OPTIMIZATION_COUNTER:06d} [{method_name}]: χ²_red = {reduced_chi2:.6e}"
                )

            if return_components:
                return {
                    "chi_squared": total_chi2,
                    "reduced_chi_squared": reduced_chi2,
                    "degrees_of_freedom": dof,
                    "angle_chi_squared": angle_chi2,
                    "angle_chi_squared_reduced": angle_chi2_reduced,
                    "angle_data_points": angle_data_points,
                    "phi_angles": phi_angles.tolist(),
                    "scaling_solutions": scaling_solutions,
                    "valid": True,
                }
            else:
                return reduced_chi2

        except Exception as e:
            logger.warning(f"Chi-squared calculation failed: {e}")
            logger.exception("Full traceback for chi-squared calculation failure:")
            if return_components:
                return {"chi_squared": np.inf, "valid": False, "error": str(e)}
            else:
                return np.inf

    def analyze_per_angle_chi_squared(
        self,
        parameters: np.ndarray,
        phi_angles: np.ndarray,
        c2_experimental: np.ndarray,
        method_name: str = "Final",
        save_to_file: bool = True,
        output_dir: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Analyze per-angle reduced chi-squared values and identify poorly fitting angles.

        Parameters
        ----------
        parameters : np.ndarray
            Optimized model parameters
        phi_angles : np.ndarray
            Scattering angles
        c2_experimental : np.ndarray
            Experimental correlation data
        method_name : str
            Name of the analysis method
        save_to_file : bool
            Whether to save results to file
        output_dir : str, optional
            Output directory for results

        Returns
        -------
        Dict[str, Any]
            Per-angle analysis results
        """
        # Get detailed chi-squared components
        chi_results = self.calculate_chi_squared_optimized(
            parameters,
            phi_angles,
            c2_experimental,
            method_name=method_name,
            return_components=True,
        )

        # Handle case where chi_results might be a float (when return_components=False fails)
        if not isinstance(chi_results, dict) or not chi_results.get("valid", False):
            logger.error("Chi-squared calculation failed for per-angle analysis")
            return {"valid": False, "error": "Chi-squared calculation failed"}

        # Extract per-angle data
        angle_chi2_reduced = chi_results["angle_chi_squared_reduced"]
        angles = chi_results["phi_angles"]

        # Analysis statistics
        mean_chi2_red = np.mean(angle_chi2_reduced)
        std_chi2_red = np.std(angle_chi2_reduced)
        min_chi2_red = np.min(angle_chi2_reduced)
        max_chi2_red = np.max(angle_chi2_reduced)

        # Get validation thresholds from configuration
        validation_config = (
            self.config.get("validation_rules", {}) if self.config else {}
        )
        fit_quality_config = validation_config.get("fit_quality", {})
        overall_config = fit_quality_config.get("overall_chi_squared", {})
        per_angle_config = fit_quality_config.get("per_angle_chi_squared", {})

        # Overall chi-squared quality assessment
        overall_chi2 = chi_results["reduced_chi_squared"]
        acceptable_overall = overall_config.get("acceptable_threshold", 10.0)
        warning_overall = overall_config.get("warning_threshold", 20.0)
        critical_overall = overall_config.get("critical_threshold", 50.0)

        overall_quality = "excellent"
        if overall_chi2 > acceptable_overall:
            overall_quality = (
                "acceptable" if overall_chi2 <= warning_overall else "poor"
            )
        if overall_chi2 > warning_overall:
            overall_quality = (
                "warning" if overall_chi2 <= critical_overall else "critical"
            )

        # Per-angle quality assessment
        acceptable_per_angle = per_angle_config.get("acceptable_threshold", 15.0)
        outlier_multiplier = per_angle_config.get("outlier_threshold_multiplier", 3.0)
        max_outlier_fraction = per_angle_config.get("max_outlier_fraction", 0.2)
        min_good_angles = per_angle_config.get("min_good_angles", 5)

        # Identify outlier angles using configurable threshold
        outlier_threshold = mean_chi2_red + outlier_multiplier * std_chi2_red
        outlier_indices = np.where(np.array(angle_chi2_reduced) > outlier_threshold)[0]
        outlier_angles = [angles[i] for i in outlier_indices]
        outlier_chi2 = [angle_chi2_reduced[i] for i in outlier_indices]

        # Identify unacceptable angles (above absolute threshold)
        unacceptable_indices = np.where(
            np.array(angle_chi2_reduced) > acceptable_per_angle
        )[0]
        unacceptable_angles = [angles[i] for i in unacceptable_indices]
        unacceptable_chi2 = [angle_chi2_reduced[i] for i in unacceptable_indices]

        # Good angles (below acceptable threshold)
        good_indices = np.where(np.array(angle_chi2_reduced) <= acceptable_per_angle)[0]
        good_angles = [angles[i] for i in good_indices]
        num_good_angles = len(good_angles)

        # Quality assessment
        outlier_fraction = len(outlier_angles) / len(angles)
        unacceptable_fraction = len(unacceptable_angles) / len(angles)

        per_angle_quality = "excellent"
        quality_issues = []

        if num_good_angles < min_good_angles:
            per_angle_quality = "critical"
            quality_issues.append(
                f"Only {num_good_angles} good angles (min required: {min_good_angles})"
            )

        if unacceptable_fraction > max_outlier_fraction:
            per_angle_quality = (
                "poor" if per_angle_quality != "critical" else per_angle_quality
            )
            quality_issues.append(
                f"{unacceptable_fraction:.1%} angles unacceptable (max allowed: {max_outlier_fraction:.1%})"
            )

        if outlier_fraction > max_outlier_fraction:
            per_angle_quality = (
                "warning" if per_angle_quality == "excellent" else per_angle_quality
            )
            quality_issues.append(
                f"{outlier_fraction:.1%} statistical outliers (max recommended: {max_outlier_fraction:.1%})"
            )

        # Combined assessment
        if overall_quality in ["critical", "poor"] or per_angle_quality in [
            "critical",
            "poor",
        ]:
            combined_quality = "poor"
        elif overall_quality == "warning" or per_angle_quality == "warning":
            combined_quality = "warning"
        elif overall_quality == "acceptable" or per_angle_quality == "acceptable":
            combined_quality = "acceptable"
        else:
            combined_quality = "excellent"

        # Create comprehensive results
        per_angle_results = {
            "method": method_name,
            "overall_reduced_chi_squared": chi_results["reduced_chi_squared"],
            "per_angle_analysis": {
                "phi_angles_deg": angles,
                "chi_squared_reduced": angle_chi2_reduced,
                "data_points_per_angle": chi_results["angle_data_points"],
                "scaling_solutions": chi_results["scaling_solutions"],
            },
            "statistics": {
                "mean_chi2_reduced": mean_chi2_red,
                "std_chi2_reduced": std_chi2_red,
                "min_chi2_reduced": min_chi2_red,
                "max_chi2_reduced": max_chi2_red,
                "range_chi2_reduced": max_chi2_red - min_chi2_red,
            },
            "quality_assessment": {
                "overall_quality": overall_quality,
                "per_angle_quality": per_angle_quality,
                "combined_quality": combined_quality,
                "quality_issues": quality_issues,
                "thresholds_used": {
                    "acceptable_overall": acceptable_overall,
                    "warning_overall": warning_overall,
                    "critical_overall": critical_overall,
                    "acceptable_per_angle": acceptable_per_angle,
                    "outlier_multiplier": outlier_multiplier,
                    "max_outlier_fraction": max_outlier_fraction,
                    "min_good_angles": min_good_angles,
                },
            },
            "angle_categorization": {
                "good_angles": {
                    "angles_deg": good_angles,
                    "count": num_good_angles,
                    "fraction": num_good_angles / len(angles),
                    "criteria": f"χ²_red ≤ {acceptable_per_angle}",
                },
                "unacceptable_angles": {
                    "angles_deg": unacceptable_angles,
                    "chi2_reduced": unacceptable_chi2,
                    "count": len(unacceptable_angles),
                    "fraction": unacceptable_fraction,
                    "criteria": f"χ²_red > {acceptable_per_angle}",
                },
                "statistical_outliers": {
                    "angles_deg": outlier_angles,
                    "chi2_reduced": outlier_chi2,
                    "count": len(outlier_angles),
                    "fraction": outlier_fraction,
                    "criteria": (
                        f"χ²_red > mean + {outlier_multiplier}×std ({outlier_threshold:.3f})"
                    ),
                },
            },
        }

        # Save to file if requested
        if save_to_file:
            if output_dir is None:
                output_dir = "./homodyne_results"
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            # Save detailed per-angle results
            angle_results_file = (
                output_path / f"per_angle_chi_squared_{method_name.lower()}.json"
            )
            try:
                import json  # noqa: F811

                with open(angle_results_file, "w") as f:
                    json.dump(per_angle_results, f, indent=2, default=str)
                logger.info(
                    f"Per-angle chi-squared analysis saved to: {angle_results_file}"
                )
            except Exception as e:
                logger.error(f"Failed to save per-angle results: {e}")

        # Log summary with quality assessment
        logger.info(f"Per-angle chi-squared analysis [{method_name}]:")
        logger.info(
            f"  Overall χ²_red: {chi_results['reduced_chi_squared']:.6e} ({overall_quality})"
        )
        logger.info(
            f"  Mean per-angle χ²_red: {mean_chi2_red:.6e} ± {std_chi2_red:.6e}"
        )
        logger.info(f"  Range: {min_chi2_red:.6e} - {max_chi2_red:.6e}")

        # Quality assessment logging
        logger.info(f"  Quality Assessment: {combined_quality.upper()}")
        logger.info(
            f"    Overall: {overall_quality} (threshold: {acceptable_overall:.1f})"
        )
        logger.info(f"    Per-angle: {per_angle_quality}")

        # Angle categorization
        logger.info("  Angle Categorization:")
        logger.info(
            f"    Good angles: {num_good_angles}/{len(angles)} ({100*num_good_angles/len(angles):.1f}%) [χ²_red ≤ {acceptable_per_angle}]"
        )
        logger.info(
            f"    Unacceptable angles: {len(unacceptable_angles)}/{len(angles)} ({100*unacceptable_fraction:.1f}%) [χ²_red > {acceptable_per_angle}]"
        )
        logger.info(
            f"    Statistical outliers: {len(outlier_angles)}/{len(angles)} ({100*outlier_fraction:.1f}%) [χ²_red > {outlier_threshold:.3f}]"
        )

        # Warnings and issues
        if quality_issues:
            for issue in quality_issues:
                logger.warning(f"  Quality Issue: {issue}")

        if unacceptable_angles:
            logger.warning(f"  Unacceptable angles: {unacceptable_angles}")

        if outlier_angles:
            logger.warning(f"  Statistical outlier angles: {outlier_angles}")

        # Overall quality verdict
        if combined_quality == "critical":
            logger.error(
                "  ❌ CRITICAL: Fit quality is unacceptable - consider parameter adjustment or data quality check"
            )
        elif combined_quality == "poor":
            logger.warning(
                "  ⚠ POOR: Fit quality is poor - optimization may need improvement"
            )
        elif combined_quality == "warning":
            logger.warning(
                "  ⚠ WARNING: Some angles show poor fit - consider investigation"
            )
        elif combined_quality == "acceptable":
            logger.info(
                "  ✓ ACCEPTABLE: Fit quality is acceptable with some limitations"
            )
        else:
            logger.info("  ✅ EXCELLENT: Fit quality is excellent across all angles")

        return per_angle_results

    # ============================================================================
    # PARAMETER MANAGEMENT AND RESULTS SAVING
    # ============================================================================

    def save_results_with_config(
        self, results: Dict[str, Any], output_dir: Optional[str] = None
    ):
        """
        Save comprehensive analysis results with configuration-based formatting.

        This method provides flexible, configuration-driven result saving that can
        accommodate different output formats, compression levels, and organizational
        structures. The saving process is designed to support:

        - Reproducible research through complete configuration archival
        - Multiple output formats for different analysis workflows
        - Efficient storage with configurable compression
        - Automatic organization and naming conventions
        - Integration with analysis pipelines and databases
        - Long-term archival and result retrieval

        Parameters
        ----------
        results : Dict[str, Any]
            Complete results dictionary from comprehensive analysis
        output_dir : Optional[str]
            Custom output directory path
        """
        import json  # noqa: F811
        import sys  # noqa: F811
        import time  # noqa: F811
        from pathlib import Path  # noqa: F811

        # Determine output directory from configuration or parameter
        config = self.config if self.config is not None else {}
        if output_dir is None:
            output_dir = config.get("output_settings", {}).get(
                "results_directory", "homodyne_analysis_results"
            )
        if output_dir is None:
            output_dir = "homodyne_analysis_results"

        # Ensure output directory exists with proper error handling
        output_path = Path(str(output_dir))
        try:
            output_path.mkdir(parents=True, exist_ok=True)
            print(f"📁 Saving results to: {output_path.absolute()}")
        except PermissionError:
            raise IOError(f"Permission denied: Cannot create directory {output_dir}")
        except OSError as e:
            raise IOError(f"Failed to create output directory {output_dir}: {e}")

        # Get output configuration settings
        output_config = (
            self.config.get("output_settings", {}) if self.config is not None else {}
        )

        # =======================
        # SAVE ANALYSIS CONFIGURATION
        # =======================

        print("  💾 Saving analysis configuration...")
        config_file = output_path / "run_configuration.json"

        try:
            # Enhanced configuration with execution metadata
            enhanced_config = self.config.copy() if self.config is not None else {}
            enhanced_config["execution_metadata"] = {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
                "total_execution_time_seconds": sum(
                    r.get("optimization_time", 0) + r.get("total_time", 0)
                    for r in results.values()
                    if isinstance(r, dict)
                ),
                "methods_executed": results.get("methods_used", []),
                "analysis_success": (
                    "best_overall" in results
                    and results["best_overall"]["parameters"] is not None
                ),
                "frame_range_analyzed": (self.start_frame, self.end_frame),
                "time_window_seconds": self.time_length * self.dt,
                "computational_environment": {
                    "numba_available": NUMBA_AVAILABLE,
                    "pymc_available": PYMC_AVAILABLE,
                    "skopt_available": SKOPT_AVAILABLE,
                    "threads_used": self.num_threads,
                    "python_version": sys.version.split()[0],
                },
            }

            # Add git information if available
            try:
                import subprocess

                git_commit = (
                    subprocess.check_output(
                        ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
                    )
                    .decode("utf-8")
                    .strip()
                )
                enhanced_config["execution_metadata"]["git_commit"] = git_commit
            except Exception:
                # Git not available or not in a git repository
                pass

            with open(config_file, "w") as f:
                json.dump(enhanced_config, f, indent=2, default=str, sort_keys=True)

            print(f"    ✓ Configuration saved: {config_file.name}")

        except Exception as e:
            print(f"    ⚠ Warning: Failed to save configuration: {e}")

        # =======================
        # SAVE EXPERIMENTAL DATA
        # =======================

        if output_config.get("save_experimental_data", True):
            print("  💾 Saving experimental data...")
            try:
                exp_data_file = output_path / "experimental_data.npz"
                # Get experimental data if not already loaded
                if (
                    self.cached_experimental_data is not None
                    and self.cached_phi_angles is not None
                ):
                    data_to_save = {
                        "c2_experimental": self.cached_experimental_data,
                        "phi_angles": self.cached_phi_angles,
                        "time_array": self.time_array,
                        "metadata": {
                            "start_frame": self.start_frame,
                            "end_frame": self.end_frame,
                            "dt": self.dt,
                            "wavevector_q": self.wavevector_q,
                            "stator_rotor_gap": self.stator_rotor_gap,
                        },
                    }

                    if output_config.get("compress_data", True):
                        np.savez_compressed(exp_data_file, **data_to_save)
                    else:
                        np.savez(exp_data_file, **data_to_save)

                    print(f"    ✓ Experimental data saved: {exp_data_file.name}")
                else:
                    print("    ⚠ Warning: No experimental data in cache to save")

            except Exception as e:
                print(f"    ⚠ Warning: Failed to save experimental data: {e}")

        # =======================
        # CREATE RESULTS SUMMARY
        # =======================

        try:
            summary_file = output_path / "analysis_summary.txt"
            with open(summary_file, "w") as f:
                f.write("HOMODYNE SCATTERING ANALYSIS SUMMARY\n")
                f.write("=" * 50 + "\n\n")

                f.write(
                    f"Analysis completed: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}\n"
                )
                f.write(
                    f"Core analysis configuration: {self.config_manager.config_file}\n"
                )
                f.write(
                    f"Frame range: {self.start_frame}-{self.end_frame} ({self.time_length} frames)\n"
                )
                f.write(f"Time step: {self.dt} s/frame\n")
                f.write(f"Wavevector: {self.wavevector_q:.6f} Å⁻¹\n")
                f.write(f"Gap size: {self.stator_rotor_gap/1e4:.1f} μm\n")
                f.write(f"Processing threads: {self.num_threads}\n\n")

                # Analysis results summary
                if (
                    "best_overall" in results
                    and results["best_overall"]["parameters"] is not None
                ):
                    f.write("ANALYSIS SUCCESS\n")
                    f.write("-" * 20 + "\n")
                    best_params = results["best_overall"]["parameters"]
                    param_names = [
                        "D₀",
                        "α",
                        "D_offset",
                        "γ̇₀",
                        "β",
                        "γ̇_offset",
                        "φ₀",
                    ]

                    for name, value in zip(param_names, best_params):
                        if abs(value) < 1e-3 or abs(value) > 1e3:
                            formatted_value = f"{value:.3e}"
                        else:
                            formatted_value = f"{value:.6f}"
                        f.write(f"{name:<15} {formatted_value}\n")

                    f.write(
                        f"\nBest χ²_red: {results['best_overall']['chi_squared']:.6e}\n"
                    )
                else:
                    f.write("ANALYSIS INCOMPLETE OR FAILED\n")
                    f.write("-" * 30 + "\n")
                    f.write("No successful optimization results available.\n")

                f.write(
                    f"\nSummary generated: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}\n"
                )

            print(f"    ✓ Analysis summary saved: {summary_file.name}")

        except Exception as e:
            print(f"    ⚠ Warning: Failed to create analysis summary: {e}")

        # =======================
        # SAVE OPTIMIZATION RESULTS
        # =======================

        print("  💾 Saving optimization results...")
        try:
            results_file = output_path / "optimization_results.json"

            # Create a clean results dictionary for JSON serialization
            clean_results = {}
            for key, value in results.items():
                if key == "methods_used" or key == "methods_attempted":
                    clean_results[key] = value
                elif isinstance(value, dict) and any(
                    opt_key in key for opt_key in ["optimization", "best_overall"]
                ):
                    clean_results[key] = {}
                    for sub_key, sub_value in value.items():
                        if isinstance(sub_value, np.ndarray):
                            clean_results[key][sub_key] = sub_value.tolist()
                        elif isinstance(sub_value, (np.integer, np.floating)):
                            clean_results[key][sub_key] = (
                                float(sub_value)
                                if isinstance(sub_value, np.floating)
                                else int(sub_value)
                            )
                        else:
                            clean_results[key][sub_key] = sub_value
                else:
                    clean_results[key] = value

            # Add metadata
            clean_results["metadata"] = {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
                "config_file": (
                    self.config_manager.config_file
                    if hasattr(self, "config_manager")
                    else "unknown"
                ),
                "frame_range": f"{self.start_frame}-{self.end_frame}",
                "analysis_parameters": {
                    "dt": self.dt,
                    "wavevector_q": self.wavevector_q,
                    "gap_microns": self.stator_rotor_gap / 1e4,
                },
            }

            with open(results_file, "w") as f:
                json.dump(clean_results, f, indent=2, default=str, sort_keys=True)

            print(f"    ✓ Optimization results saved: {results_file.name}")

            # Log summary of saved methods
            saved_methods = results.get("methods_used", [])
            print(f"    📊 Methods saved: {', '.join(saved_methods)}")
            for method in saved_methods:
                method_key = f"{method.lower()}_optimization"
                if method_key in results:
                    chi2 = results[method_key].get("chi_squared", "N/A")
                    params = results[method_key].get("parameters")
                    params_count = (
                        len(params)
                        if params is not None and len(params) > 0
                        else 0
                    )
                    print(
                        f"        {method}: χ²_red={chi2:.6e}, {params_count} parameters"
                        if isinstance(chi2, (int, float))
                        else f"        {method}: χ²_red={chi2}, {params_count} parameters"
                    )

        except Exception as e:
            print(f"    ⚠ Warning: Failed to save optimization results: {e}")
            import traceback

            logger.debug(f"Full traceback: {traceback.format_exc()}")

        # =======================
        # FINAL SUMMARY
        # =======================

        saved_files = list(output_path.glob("*"))
        total_size_mb = sum(f.stat().st_size for f in saved_files if f.is_file()) / (
            1024 * 1024
        )

        print(f"\n✅ Results saved successfully!")
        print(f"   📁 Output directory: {output_path.absolute()}")
        print(f"   📄 Files saved: {len(saved_files)}")
        print(f"   💾 Total size: {total_size_mb:.1f} MB")

        return output_path
