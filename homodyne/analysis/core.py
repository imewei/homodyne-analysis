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
    g2(φ,t₁,t₂) = 1 + contrast × [g1(φ,t₁,t₂)]²

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

        # Initialize plotting cache variables
        self._last_experimental_data = None
        self._last_phi_angles = None

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

        # Load angle configuration - skip for isotropic static mode
        if self.config_manager.is_static_isotropic_enabled():
            # In isotropic static mode, create a single dummy angle
            phi_angles = np.array([0.0], dtype=np.float64)
            num_angles = 1
            logger.info(
                "Isotropic static mode: Using single dummy angle (0.0°) instead of loading phi_angles_file"
            )
        else:
            # Normal mode: load phi angles from file
            phi_angles_path = self.config["experimental_data"].get(
                "phi_angles_path", "."
            )
            phi_angles_file = self.config["experimental_data"]["phi_angles_file"]
            phi_file = os.path.join(phi_angles_path, phi_angles_file)
            logger.debug(f"Loading phi angles from: {phi_file}")
            phi_angles = np.loadtxt(phi_file, dtype=np.float64)
            # Ensure phi_angles is always an array, even for single values
            phi_angles = np.atleast_1d(phi_angles)
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

        # Cache for plotting
        self._last_experimental_data = c2_experimental
        self._last_phi_angles = phi_angles
        logger.debug(f"Data cached in memory - final shape: {c2_experimental.shape}")

        # Plot experimental data for validation if enabled
        if (
            self.config.get("workflow_integration", {})
            .get("analysis_workflow", {})
            .get("plot_experimental_data_on_load", False)
        ):
            logger.info("Plotting experimental data for validation...")
            try:
                self._plot_experimental_data_validation(c2_experimental, phi_angles)
                logger.info("Experimental data validation plot created successfully")
            except Exception as e:
                logger.warning(
                    f"Failed to create experimental data validation plot: {e}"
                )

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

        # Handle data loading for isotropic static mode vs normal mode
        if self.config_manager.is_static_isotropic_enabled():
            # In isotropic static mode, load data only once and use for the single dummy angle
            logger.info(
                f"Isotropic static mode: Loading single correlation matrix for dummy angle"
            )

            try:
                # Load data once for isotropic case
                raw_data = data_file.get_twotime_c2(exchange_key, correct_diag=False)
                if raw_data is None:
                    raise ValueError(
                        "get_twotime_c2 returned None in isotropic static mode"
                    )

                # Ensure raw_data is a NumPy array
                raw_data_np = np.array(raw_data)
                sliced_data = raw_data_np[
                    self.start_frame : self.end_frame,
                    self.start_frame : self.end_frame,
                ]
                # Use the same data for the single dummy angle
                c2_experimental[0] = sliced_data.astype(np.float64)
                logger.debug(
                    f"  Isotropic mode - Raw data shape: {raw_data_np.shape} -> sliced: {sliced_data.shape}"
                )

            except Exception as e:
                logger.error(f"Failed to load data in isotropic static mode: {e}")
                raise
        else:
            # Normal mode: load data for each angle
            logger.info(f"Loading data for {num_angles} angles...")
            for i in range(num_angles):
                angle_deg = phi_angles[i]
                logger.debug(f"Loading angle {i+1}/{num_angles} (φ={angle_deg:.2f}°)")

                try:
                    # Fix: Pass correct_diag as bool, not int. If you want diagonal correction, set to True, else False.
                    raw_data = data_file.get_twotime_c2(
                        exchange_key, correct_diag=False
                    )
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
        Calculate chi-squared goodness of fit with per-angle analysis and uncertainty estimation.

        This method computes the reduced chi-squared statistic for model validation, with optional
        detailed per-angle analysis and uncertainty quantification. The uncertainty in reduced
        chi-squared provides insight into the consistency of fit quality across different angles.

        Parameters
        ----------
        parameters : np.ndarray
            Model parameters [D0, alpha, D_offset, gamma_dot_t0, beta, gamma_dot_t_offset, phi0]
        phi_angles : np.ndarray
            Scattering angles in degrees
        c2_experimental : np.ndarray
            Experimental correlation data with shape (n_angles, delay_frames, lag_frames)
        method_name : str, optional
            Name of optimization method for logging purposes
        return_components : bool, optional
            If True, return detailed results dictionary with per-angle analysis
        filter_angles_for_optimization : bool, optional
            If True, only include angles in optimization ranges [-10°, 10°] and [170°, 190°]
            for chi-squared calculation

        Returns
        -------
        float or dict
            If return_components=False: Reduced chi-squared value (float)
            If return_components=True: Dictionary containing:
                - chi_squared : float
                    Total chi-squared value
                - reduced_chi_squared : float
                    Averaged reduced chi-squared from optimization angles
                - reduced_chi_squared_uncertainty : float
                    Standard error of reduced chi-squared across angles (uncertainty estimate)
                - reduced_chi_squared_std : float
                    Standard deviation of reduced chi-squared across angles
                - n_optimization_angles : int
                    Number of angles used for optimization
                - degrees_of_freedom : int
                    Degrees of freedom for statistical testing (data_points - n_parameters)
                - angle_chi_squared : list
                    Chi-squared values for each angle
                - angle_chi_squared_reduced : list
                    Reduced chi-squared values for each angle
                - angle_data_points : list
                    Number of data points per angle
                - phi_angles : list
                    Scattering angles used
                - scaling_solutions : list
                    Contrast and offset parameters for each angle
                - valid : bool
                    Whether calculation was successful

        Notes
        -----
        The uncertainty calculation follows standard error of the mean:

        reduced_chi2_uncertainty = std(angle_chi2_reduced) / sqrt(n_angles)

        Interpretation of uncertainty:
        - Small uncertainty (< 0.1 * reduced_chi2): Consistent fit across angles
        - Large uncertainty (> 0.5 * reduced_chi2): High angle variability, potential
          systematic issues or model inadequacy

        The method uses averaged (not summed) chi-squared for better angle weighting:
        reduced_chi2 = mean(chi2_reduced_per_angle) for optimization angles only

        Quality assessment guidelines:
        - Excellent: reduced_chi2 ≤ 2.0
        - Acceptable: 2.0 < reduced_chi2 ≤ 5.0
        - Warning: 5.0 < reduced_chi2 ≤ 10.0
        - Poor/Critical: reduced_chi2 > 10.0
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
                        reason = f'Parameter {bound.get("name", f"p{i}")} out of bounds'
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

            optimization_chi2_angles = []
            angle_chi2 = []
            angle_chi2_reduced = []
            angle_data_points = []
            scaling_solutions = []

            # Calculate chi-squared for all angles (for detailed results)
            for i in range(len(phi_angles)):
                theory = c2_theory[i].ravel()
                exp = c2_experimental[i].ravel()
                n_data_angle = len(exp)  # Number of data points for this angle

                # SCALING OPTIMIZATION (ALWAYS ENABLED)
                # =====================================
                # This performs least squares fitting to determine the optimal scaling relationship:
                # g₂ = offset + contrast × g₁ where:
                # - g₁ is the theoretical correlation function
                # - g₂ is the experimental correlation function
                # - contrast and offset are fitted scaling parameters
                #
                # WHY THIS IS ESSENTIAL:
                # This scaling optimization is ALWAYS enabled because it is fundamental to proper
                # chi-squared calculation. Without it, we would compare raw theoretical values
                # directly to experimental data, which ignores systematic scaling factors and
                # offsets that are physically present due to:
                # - Instrumental response functions
                # - Background signals
                # - Detector gain variations
                # - Normalization differences
                #
                # The chi-squared is calculated as χ² = Σ(experimental - fitted)²/σ²
                # where fitted = theory × contrast + offset from the optimal scaling.
                # This provides meaningful fitting quality assessment regardless of the number
                # of scattering angles or analysis mode.
                #
                # Mathematical implementation: solve A·x = b where A = [theory, ones], x = [contrast, offset]
                A = np.vstack([theory, np.ones(len(theory))]).T
                scaling, residuals, _, _ = np.linalg.lstsq(A, exp, rcond=None)

                if len(scaling) == 2:
                    contrast, offset = scaling
                    fitted = theory * contrast + offset
                    scaling_solutions.append([contrast, offset])
                else:
                    fitted = theory
                    scaling_solutions.append([1.0, 0.0])

                # Calculate chi-squared for this angle
                residuals = exp - fitted
                sigma = max(np.std(exp) * uncertainty_factor, min_sigma)
                chi2_angle = np.sum(residuals**2) / (sigma**2)

                # Calculate reduced chi-squared for this angle
                dof_angle = max(n_data_angle - n_params, 1)  # DOF for this angle
                chi2_reduced_angle = chi2_angle / dof_angle

                angle_chi2.append(chi2_angle)
                angle_chi2_reduced.append(chi2_reduced_angle)
                angle_data_points.append(n_data_angle)

                # Collect chi2 values for optimization angles (for averaging)
                if not filter_angles_for_optimization or i in optimization_indices:
                    optimization_chi2_angles.append(chi2_reduced_angle)

            # Calculate average reduced chi-squared from optimization angles with uncertainty
            if optimization_chi2_angles:
                reduced_chi2 = np.mean(optimization_chi2_angles)
                n_optimization_angles = len(optimization_chi2_angles)

                # Calculate uncertainty in reduced chi-squared
                if n_optimization_angles > 1:
                    # Standard error of the mean
                    reduced_chi2_std = np.std(optimization_chi2_angles, ddof=1)
                    reduced_chi2_uncertainty = reduced_chi2_std / np.sqrt(
                        n_optimization_angles
                    )
                else:
                    # Single angle case
                    reduced_chi2_std = 0.0
                    reduced_chi2_uncertainty = 0.0

                logger.debug(
                    f"Using average of {n_optimization_angles} optimization angles: χ²_red = {reduced_chi2:.6e} ± {reduced_chi2_uncertainty:.6e}"
                )
            else:
                # Fallback if no optimization angles (shouldn't happen)
                reduced_chi2 = (
                    np.mean(angle_chi2_reduced) if angle_chi2_reduced else 1e6
                )
                reduced_chi2_std = (
                    np.std(angle_chi2_reduced, ddof=1)
                    if len(angle_chi2_reduced) > 1
                    else 0.0
                )
                reduced_chi2_uncertainty = (
                    reduced_chi2_std / np.sqrt(len(angle_chi2_reduced))
                    if len(angle_chi2_reduced) > 1
                    else 0.0
                )
                logger.warning(
                    "No optimization angles found, using average of all angles"
                )

            # Logging
            OPTIMIZATION_COUNTER += 1
            log_freq = self.config["performance_settings"].get(
                "optimization_counter_log_frequency", 50
            )
            if OPTIMIZATION_COUNTER % log_freq == 0:
                logger.info(
                    f"Iteration {OPTIMIZATION_COUNTER:06d} [{method_name}]: χ²_red = {reduced_chi2:.6e} ± {reduced_chi2_uncertainty:.6e}"
                )
                # Log reduced chi-square per angle
                for i, (phi, chi2_red_angle) in enumerate(
                    zip(phi_angles, angle_chi2_reduced)
                ):
                    logger.info(
                        f"  Angle {i+1} (φ={phi:.1f}°): χ²_red = {chi2_red_angle:.6e}"
                    )

            if return_components:
                # Calculate total chi2 for compatibility (sum of optimization angles)
                total_chi2_compat = (
                    sum(angle_chi2[i] for i in optimization_indices)
                    if filter_angles_for_optimization
                    else sum(angle_chi2)
                )

                # Calculate degrees of freedom
                total_data_points = (
                    sum(angle_data_points[i] for i in optimization_indices)
                    if filter_angles_for_optimization
                    else sum(angle_data_points)
                )
                num_parameters = len(parameters)
                degrees_of_freedom = max(1, total_data_points - num_parameters)

                return {
                    "chi_squared": total_chi2_compat,
                    "reduced_chi_squared": float(reduced_chi2),
                    "reduced_chi_squared_uncertainty": float(reduced_chi2_uncertainty),
                    "reduced_chi_squared_std": float(reduced_chi2_std),
                    "n_optimization_angles": len(optimization_chi2_angles),
                    "degrees_of_freedom": degrees_of_freedom,
                    "angle_chi_squared": angle_chi2,
                    "angle_chi_squared_reduced": angle_chi2_reduced,
                    "angle_data_points": angle_data_points,
                    "phi_angles": phi_angles.tolist(),
                    "scaling_solutions": scaling_solutions,
                    "valid": True,
                }
            else:
                return float(reduced_chi2)

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
        Comprehensive per-angle reduced chi-squared analysis with quality assessment.

        This method performs detailed analysis of chi-squared values across different
        scattering angles, providing quality metrics, uncertainty estimation, and
        angle categorization to identify systematic fitting issues.

        Parameters
        ----------
        parameters : np.ndarray
            Optimized model parameters [D0, alpha, D_offset, gamma_dot_t0, beta, gamma_dot_t_offset, phi0]
        phi_angles : np.ndarray
            Scattering angles in degrees
        c2_experimental : np.ndarray
            Experimental correlation data with shape (n_angles, delay_frames, lag_frames)
        method_name : str, optional
            Name of the analysis method for file naming and logging
        save_to_file : bool, optional
            Whether to save detailed results to JSON file
        output_dir : str, optional
            Output directory for saved results (defaults to current directory)

        Returns
        -------
        Dict[str, Any]
            Comprehensive analysis results containing:
                - method : str
                    Analysis method name
                - overall_reduced_chi_squared : float
                    Average reduced chi-squared across optimization angles
                - reduced_chi_squared_uncertainty : float
                    Standard error of reduced chi-squared (uncertainty measure)
                - quality_assessment : dict
                    Overall and per-angle quality evaluation with thresholds
                - angle_categorization : dict
                    Classification of angles into good, unacceptable, and outlier groups
                - per_angle_analysis : dict
                    Detailed per-angle chi-squared values and statistics
                - statistical_summary : dict
                    Summary statistics including means, medians, and percentiles
                - recommendations : list
                    Specific recommendations based on quality assessment

        Notes
        -----
        Quality Assessment Criteria:
        - Overall reduced chi-squared uncertainty indicates fit consistency:
          * Small uncertainty (< 10% of chi2): Consistent across angles
          * Large uncertainty (> 50% of chi2): High variability, investigate systematically

        Angle Classification:
        - Good angles: reduced_chi2 ≤ acceptable_threshold (default 5.0)
        - Unacceptable angles: reduced_chi2 > acceptable_threshold
        - Statistical outliers: reduced_chi2 > mean + 2.5*std

        The method uses configuration-driven thresholds from validation_rules.fit_quality
        for consistent quality assessment across the package.

        Files saved (if save_to_file=True):
        - per_angle_chi_squared_{method_name.lower()}.json: Detailed analysis results

        See Also
        --------
        calculate_chi_squared_optimized : Underlying chi-squared calculation
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

        # Overall reduced chi-squared quality assessment (updated thresholds for reduced chi2)
        overall_chi2 = chi_results["reduced_chi_squared"]
        excellent_threshold = overall_config.get("excellent_threshold", 2.0)
        acceptable_overall = overall_config.get("acceptable_threshold", 5.0)
        warning_overall = overall_config.get("warning_threshold", 10.0)
        critical_overall = overall_config.get("critical_threshold", 20.0)

        # Determine overall quality based on reduced chi-squared
        if overall_chi2 <= excellent_threshold:
            overall_quality = "excellent"
        elif overall_chi2 <= acceptable_overall:
            overall_quality = "acceptable"
        elif overall_chi2 <= warning_overall:
            overall_quality = "warning"
        elif overall_chi2 <= critical_overall:
            overall_quality = "poor"
        else:
            overall_quality = "critical"

        # Per-angle quality assessment (updated thresholds for reduced chi2)
        excellent_per_angle = per_angle_config.get("excellent_threshold", 2.0)
        acceptable_per_angle = per_angle_config.get("acceptable_threshold", 5.0)
        warning_per_angle = per_angle_config.get("warning_threshold", 10.0)
        outlier_multiplier = per_angle_config.get("outlier_threshold_multiplier", 2.5)
        max_outlier_fraction = per_angle_config.get("max_outlier_fraction", 0.25)
        min_good_angles = per_angle_config.get("min_good_angles", 3)

        # Identify outlier angles using configurable threshold
        outlier_threshold = mean_chi2_red + outlier_multiplier * std_chi2_red
        outlier_indices = np.where(np.array(angle_chi2_reduced) > outlier_threshold)[0]
        outlier_angles = [angles[i] for i in outlier_indices]
        outlier_chi2 = [angle_chi2_reduced[i] for i in outlier_indices]

        # Categorize angles by quality levels
        angle_chi2_array = np.array(angle_chi2_reduced)

        # Excellent angles (≤ 2.0)
        excellent_indices = np.where(angle_chi2_array <= excellent_per_angle)[0]
        excellent_angles = [angles[i] for i in excellent_indices]

        # Acceptable angles (≤ 5.0)
        acceptable_indices = np.where(angle_chi2_array <= acceptable_per_angle)[0]
        acceptable_angles = [angles[i] for i in acceptable_indices]

        # Warning angles (> 5.0, ≤ 10.0)
        warning_indices = np.where(
            (angle_chi2_array > acceptable_per_angle)
            & (angle_chi2_array <= warning_per_angle)
        )[0]
        warning_angles = [angles[i] for i in warning_indices]

        # Poor angles (> 10.0)
        poor_indices = np.where(angle_chi2_array > warning_per_angle)[0]
        poor_angles = [angles[i] for i in poor_indices]
        poor_chi2 = [angle_chi2_reduced[i] for i in poor_indices]

        # Legacy compatibility
        unacceptable_angles = poor_angles
        unacceptable_chi2 = poor_chi2
        good_angles = acceptable_angles
        num_good_angles = len(acceptable_angles)

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
            "overall_reduced_chi_squared_uncertainty": chi_results.get(
                "reduced_chi_squared_uncertainty", 0.0
            ),
            "overall_reduced_chi_squared_std": chi_results.get(
                "reduced_chi_squared_std", 0.0
            ),
            "n_optimization_angles": chi_results.get(
                "n_optimization_angles", len(angles)
            ),
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
                "uncertainty_from_angles": chi_results.get(
                    "reduced_chi_squared_uncertainty", 0.0
                ),
            },
            "quality_assessment": {
                "overall_quality": overall_quality,
                "per_angle_quality": per_angle_quality,
                "combined_quality": combined_quality,
                "quality_issues": quality_issues,
                "thresholds_used": {
                    "excellent_overall": excellent_threshold,
                    "acceptable_overall": acceptable_overall,
                    "warning_overall": warning_overall,
                    "critical_overall": critical_overall,
                    "excellent_per_angle": excellent_per_angle,
                    "acceptable_per_angle": acceptable_per_angle,
                    "warning_per_angle": warning_per_angle,
                    "outlier_multiplier": outlier_multiplier,
                    "max_outlier_fraction": max_outlier_fraction,
                    "min_good_angles": min_good_angles,
                },
                "interpretation": {
                    "overall_chi2_meaning": _get_chi2_interpretation(overall_chi2),
                    "quality_explanation": _get_quality_explanation(combined_quality),
                    "recommended_actions": _get_quality_recommendations(
                        combined_quality, quality_issues
                    ),
                },
            },
            "angle_categorization": {
                "excellent_angles": {
                    "angles_deg": excellent_angles,
                    "count": len(excellent_angles),
                    "fraction": len(excellent_angles) / len(angles),
                    "criteria": f"χ²_red ≤ {excellent_per_angle}",
                },
                "acceptable_angles": {
                    "angles_deg": acceptable_angles,
                    "count": len(acceptable_angles),
                    "fraction": len(acceptable_angles) / len(angles),
                    "criteria": f"χ²_red ≤ {acceptable_per_angle}",
                },
                "warning_angles": {
                    "angles_deg": warning_angles,
                    "count": len(warning_angles),
                    "fraction": len(warning_angles) / len(angles),
                    "criteria": f"{acceptable_per_angle} < χ²_red ≤ {warning_per_angle}",
                },
                "poor_angles": {
                    "angles_deg": poor_angles,
                    "chi2_reduced": poor_chi2,
                    "count": len(poor_angles),
                    "fraction": len(poor_angles) / len(angles),
                    "criteria": f"χ²_red > {warning_per_angle}",
                },
                # Legacy compatibility
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
        overall_uncertainty = chi_results.get("reduced_chi_squared_uncertainty", 0.0)
        if overall_uncertainty > 0:
            logger.info(
                f"  Overall χ²_red: {chi_results['reduced_chi_squared']:.6e} ± {overall_uncertainty:.6e} ({overall_quality})"
            )
        else:
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

    def save_results_with_config(self, results: Dict[str, Any], skip_plots: bool = False, output_dir: str = None) -> None:
        """
        Save optimization results along with configuration to JSON file.

        This method ensures all results including uncertainty fields are properly
        saved with the configuration for reproducibility.

        Parameters
        ----------
        results : Dict[str, Any]
            Results dictionary from optimization methods
        skip_plots : bool, optional
            If True, skip generating analysis plots (default: False)
        output_dir : str, optional
            Output directory for saving results file (default: current directory)
        """
        import json
        from datetime import datetime

        # Create comprehensive results with configuration
        from datetime import timezone

        timestamp = datetime.now(timezone.utc).isoformat()

        output_data = {
            "timestamp": timestamp,
            "config": self.config,
            "results": results,
        }

        # Add execution metadata
        if "execution_metadata" not in output_data:
            output_data["execution_metadata"] = {
                "analysis_success": True,
                "timestamp": timestamp,
            }

        # Determine output file name
        if self.config is not None:
            output_settings = self.config.get("output_settings", {})
            file_formats = output_settings.get("file_formats", {})
            results_format = file_formats.get("results_format", "json")
        else:
            results_format = "json"

        # Determine output file path
        if output_dir:
            from pathlib import Path
            output_dir_path = Path(output_dir)
            output_dir_path.mkdir(parents=True, exist_ok=True)
            if results_format == "json":
                output_file = output_dir_path / "homodyne_analysis_results.json"
            else:
                output_file = output_dir_path / f"homodyne_analysis_results.{results_format}"
        else:
            if results_format == "json":
                output_file = "homodyne_analysis_results.json"
            else:
                output_file = f"homodyne_analysis_results.{results_format}"

        try:
            # Save to JSON format regardless of specified format for compatibility
            with open(output_file, "w") as f:
                json.dump(output_data, f, indent=2, default=str)

            logger.info(f"Results and configuration saved to {output_file}")

            # Also save a copy to results directory if it exists
            results_dir = "homodyne_analysis_results"
            if os.path.exists(results_dir):
                results_file_path = os.path.join(results_dir, "run_configuration.json")
                with open(results_file_path, "w") as f:
                    json.dump(output_data, f, indent=2, default=str)
                logger.info(f"Results also saved to {results_file_path}")

        except Exception as e:
            logger.error(f"Failed to save results: {e}")
            raise

        # Generate plots if enabled in configuration and not skipped
        if not skip_plots:
            self._generate_analysis_plots(results, output_data)

    def _plot_experimental_data_validation(
        self, c2_experimental: np.ndarray, phi_angles: np.ndarray
    ) -> None:
        """
        Plot experimental C2 data immediately after loading for validation.

        This method creates a comprehensive validation plot of the loaded experimental
        data to verify data integrity and structure before analysis.

        Parameters
        ----------
        c2_experimental : np.ndarray
            Experimental correlation data with shape (n_angles, n_t2, n_t1)
        phi_angles : np.ndarray
            Array of scattering angles in degrees
        """
        try:
            # Import plotting dependencies
            import matplotlib.pyplot as plt
            import matplotlib.gridspec as gridspec
            from pathlib import Path

            logger.debug("Creating experimental data validation plot")

            # Set up plotting style
            plt.style.use("default")
            plt.rcParams.update(
                {
                    "font.size": 11,
                    "axes.labelsize": 12,
                    "axes.titlesize": 14,
                    "figure.dpi": 150,
                }
            )

            # Get temporal parameters
            dt = self.dt
            n_angles, n_t2, n_t1 = c2_experimental.shape
            time_t2 = np.arange(n_t2) * dt
            time_t1 = np.arange(n_t1) * dt

            logger.debug(f"Data shape for validation plot: {c2_experimental.shape}")
            logger.debug(
                f"Time parameters: dt={dt}, t2_max={time_t2[-1]:.1f}s, t1_max={time_t1[-1]:.1f}s"
            )

            # Create the validation plot
            n_plot_angles = min(3, n_angles)  # Show up to 3 angles
            fig = plt.figure(figsize=(16, 4 * n_plot_angles))
            gs = gridspec.GridSpec(n_plot_angles, 4, hspace=0.3, wspace=0.3)

            for i in range(n_plot_angles):
                angle_idx = i * (n_angles // n_plot_angles) if n_angles > 1 else 0
                if angle_idx >= n_angles:
                    angle_idx = n_angles - 1

                angle_data = c2_experimental[angle_idx, :, :]
                phi_deg = phi_angles[angle_idx] if len(phi_angles) > angle_idx else 0.0

                # 1. Full heatmap
                ax1 = fig.add_subplot(gs[i, 0])
                im1 = ax1.imshow(
                    angle_data,
                    aspect="equal",
                    origin="lower",
                    extent=[time_t1[0], time_t1[-1], time_t2[0], time_t2[-1]],  # type: ignore
                    cmap="viridis",
                )
                ax1.set_xlabel(r"Time $t_1$ (s)")
                ax1.set_ylabel(r"Time $t_2$ (s)")
                ax1.set_title(f"$g_2(t_1,t_2)$ at φ={phi_deg:.1f}°")
                plt.colorbar(im1, ax=ax1, shrink=0.8)

                # 2. Diagonal slice
                ax2 = fig.add_subplot(gs[i, 1])
                diagonal = np.diag(angle_data)
                time_diag = time_t1[: len(diagonal)]
                ax2.plot(time_diag, diagonal, "b-", linewidth=2)
                ax2.set_xlabel("Time (s)")
                ax2.set_ylabel("g₂(t,t)")
                ax2.set_title(f"Diagonal at φ={phi_deg:.1f}°")
                ax2.grid(True, alpha=0.3)

                # 3. Cross-sections
                ax3 = fig.add_subplot(gs[i, 2])
                n_sections = 3
                section_indices = np.linspace(5, n_t2 - 5, n_sections, dtype=int)
                colors = ["red", "blue", "green"]

                for idx, color in zip(section_indices, colors):
                    if idx < n_t2:
                        ax3.plot(
                            time_t1,
                            angle_data[idx, :],
                            color=color,
                            linewidth=1.5,
                            alpha=0.8,
                            label=f"$t_2$={time_t2[idx]:.1f}s",
                        )

                ax3.set_xlabel(r"Time $t_1$ (s)")
                ax3.set_ylabel(r"$g_2(t_1,t_2)$")
                ax3.set_title(f"Cross-sections at φ={phi_deg:.1f}°")
                ax3.legend(fontsize=9)
                ax3.grid(True, alpha=0.3)

                # 4. Statistics
                ax4 = fig.add_subplot(gs[i, 3])
                ax4.axis("off")

                # Calculate statistics
                mean_val = np.mean(angle_data)
                std_val = np.std(angle_data)
                min_val = np.min(angle_data)
                max_val = np.max(angle_data)
                diag_mean = np.mean(diagonal)
                contrast = (max_val - min_val) / min_val

                stats_text = f"""Data Statistics (φ={phi_deg:.1f}°):

Shape: {angle_data.shape[0]} × {angle_data.shape[1]}

g₂ Values:
Mean: {mean_val:.4f}
Std:  {std_val:.4f}
Min:  {min_val:.4f}
Max:  {max_val:.4f}

Diagonal mean: {diag_mean:.4f}
Contrast: {contrast:.3f}

Validation:
{'✓' if 0.9 < mean_val < 1.2 else '✗'} Mean around 1.0
{'✓' if diag_mean > mean_val else '✗'} Diagonal enhanced
{'✓' if contrast > 0.001 else '✗'} Sufficient contrast"""

                ax4.text(
                    0.05,
                    0.95,
                    stats_text,
                    transform=ax4.transAxes,
                    fontsize=9,
                    verticalalignment="top",
                    fontfamily="monospace",
                    bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.7),
                )

            # Overall title
            sample_desc = self.config.get("metadata", {}).get("sample_description", "Unknown Sample")  # type: ignore
            plt.suptitle(
                f"Experimental Data Validation: {sample_desc}",
                fontsize=16,
                fontweight="bold",
            )

            # Save the validation plot
            plots_base_dir = self.config.get("output_settings", {}).get("plotting", {}).get("output", {}).get("base_directory", "./plots")  # type: ignore
            plots_dir = Path(plots_base_dir)
            plots_dir.mkdir(parents=True, exist_ok=True)

            output_file = plots_dir / "experimental_data_validation.png"
            plt.savefig(output_file, dpi=300, bbox_inches="tight")
            logger.info(f"Experimental data validation plot saved to: {output_file}")

            # Optionally show the plot
            show_plots = self.config.get("output_settings", {}).get("plotting", {}).get("general", {}).get("show_plots", False)  # type: ignore
            if show_plots:
                plt.show()
            else:
                plt.close(fig)

        except Exception as e:
            logger.error(f"Failed to create experimental data validation plot: {e}")
            import traceback

            logger.debug(traceback.format_exc())

    def _generate_analysis_plots(
        self, results: Dict[str, Any], output_data: Dict[str, Any]
    ) -> None:
        """
        Generate analysis plots including C2 heatmaps with experimental vs theoretical comparison.

        Parameters
        ----------
        results : Dict[str, Any]
            Results dictionary from optimization methods
        output_data : Dict[str, Any]
            Complete output data including configuration
        """
        logger = logging.getLogger(__name__)

        # Check if plotting is enabled in configuration
        config = output_data.get("config") or {}
        output_settings = config.get("output_settings", {})
        reporting = output_settings.get("reporting", {})

        if not reporting.get("generate_plots", True):
            logger.info("Plotting disabled in configuration - skipping plot generation")
            return

        logger.info("Generating analysis plots...")

        try:
            # Import plotting module
            from homodyne.plotting import (
                plot_c2_heatmaps,
                plot_diagnostic_summary,
                plot_mcmc_corner,
                plot_mcmc_trace,
                plot_mcmc_convergence_diagnostics,
            )

            # Determine output directory
            results_dir = Path("homodyne_results")
            plots_dir = results_dir / "plots"
            plots_dir.mkdir(parents=True, exist_ok=True)

            # Prepare data for plotting
            plot_data = self._prepare_plot_data(results, config)

            if plot_data is None:
                logger.warning(
                    "Insufficient data for plotting - skipping plot generation"
                )
                return

            # Generate C2 heatmaps if experimental and theoretical data are available
            if all(
                key in plot_data
                for key in ["experimental_data", "theoretical_data", "phi_angles"]
            ):
                logger.info("Generating C2 correlation heatmaps...")
                try:
                    success = plot_c2_heatmaps(
                        plot_data["experimental_data"],
                        plot_data["theoretical_data"],
                        plot_data["phi_angles"],
                        plots_dir,
                        config,
                        t2=plot_data.get("t2"),
                        t1=plot_data.get("t1"),
                    )
                    if success:
                        logger.info("✓ C2 heatmaps generated successfully")
                    else:
                        logger.warning("⚠ Some C2 heatmaps failed to generate")
                except Exception as e:
                    logger.error(f"Failed to generate C2 heatmaps: {e}")

            # Parameter evolution plot - DISABLED (was non-functional)
            # This plot has been removed due to persistent issues

            # Generate MCMC plots if trace data is available
            if "mcmc_trace" in plot_data:
                logger.info("Generating MCMC plots...")

                # MCMC corner plot
                try:
                    success = plot_mcmc_corner(
                        plot_data["mcmc_trace"],
                        plots_dir,
                        config,
                        param_names=plot_data.get("parameter_names"),
                        param_units=plot_data.get("parameter_units"),
                    )
                    if success:
                        logger.info("✓ MCMC corner plot generated successfully")
                    else:
                        logger.warning("⚠ MCMC corner plot failed to generate")
                except Exception as e:
                    logger.error(f"Failed to generate MCMC corner plot: {e}")

                # MCMC trace plots
                try:
                    success = plot_mcmc_trace(
                        plot_data["mcmc_trace"],
                        plots_dir,
                        config,
                        param_names=plot_data.get("parameter_names"),
                        param_units=plot_data.get("parameter_units"),
                    )
                    if success:
                        logger.info("✓ MCMC trace plots generated successfully")
                    else:
                        logger.warning("⚠ MCMC trace plots failed to generate")
                except Exception as e:
                    logger.error(f"Failed to generate MCMC trace plots: {e}")

                # MCMC convergence diagnostics
                if "mcmc_diagnostics" in plot_data:
                    try:
                        success = plot_mcmc_convergence_diagnostics(
                            plot_data["mcmc_trace"],
                            plot_data["mcmc_diagnostics"],
                            plots_dir,
                            config,
                            param_names=plot_data.get("parameter_names"),
                        )
                        if success:
                            logger.info(
                                "✓ MCMC convergence diagnostics generated successfully"
                            )
                        else:
                            logger.warning(
                                "⚠ MCMC convergence diagnostics failed to generate"
                            )
                    except Exception as e:
                        logger.error(
                            f"Failed to generate MCMC convergence diagnostics: {e}"
                        )
            else:
                logger.debug("MCMC trace data not available - skipping MCMC plots")

            # Generate diagnostic summary plot
            logger.info("Generating diagnostic summary plot...")
            try:
                success = plot_diagnostic_summary(plot_data, plots_dir, config)
                if success:
                    logger.info("✓ Diagnostic summary plot generated successfully")
                else:
                    logger.warning("⚠ Diagnostic summary plot failed to generate")
            except Exception as e:
                logger.error(f"Failed to generate diagnostic summary plot: {e}")

            logger.info(f"Plots saved to: {plots_dir}")

        except ImportError as e:
            logger.warning(f"Plotting module not available: {e}")
            logger.info("Install matplotlib for plotting: pip install matplotlib")
        except Exception as e:
            logger.error(f"Unexpected error during plot generation: {e}")

    def _prepare_plot_data(
        self, results: Dict[str, Any], config: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Prepare data for plotting from analysis results.

        Parameters
        ----------
        results : Dict[str, Any]
            Results dictionary from optimization methods
        config : Dict[str, Any]
            Configuration dictionary

        Returns
        -------
        Optional[Dict[str, Any]]
            Plot data dictionary or None if insufficient data
        """
        logger = logging.getLogger(__name__)

        try:
            plot_data = {}

            # Find the best method results
            best_method = None
            best_chi2 = float("inf")

            # Check different method results
            for method_key in ["classical_optimization", "mcmc_optimization"]:
                if method_key in results:
                    method_results = results[method_key]
                    chi2 = method_results.get("chi_squared")
                    if chi2 is not None and chi2 < best_chi2:
                        best_chi2 = chi2
                        best_method = method_key

            if best_method is None:
                logger.warning("No valid optimization results found for plotting")
                return None

            # Extract best parameters
            best_params_list = results[best_method].get("parameters")
            if best_params_list is not None:
                # Convert parameter list to dictionary
                param_names = config.get("initial_parameters", {}).get(
                    "parameter_names", []
                )
                if len(param_names) == len(best_params_list):
                    plot_data["best_parameters"] = dict(
                        zip(param_names, best_params_list)
                    )
                else:
                    # Use generic names if parameter names don't match
                    plot_data["best_parameters"] = {
                        f"param_{i}": val for i, val in enumerate(best_params_list)
                    }

            # Extract parameter bounds
            parameter_space = config.get("parameter_space", {})
            if "bounds" in parameter_space:
                plot_data["parameter_bounds"] = parameter_space["bounds"]

            # Extract initial parameters
            initial_params = config.get("initial_parameters", {}).get("values")
            if initial_params is not None:
                param_names = config.get("initial_parameters", {}).get(
                    "parameter_names", []
                )
                if len(param_names) == len(initial_params):
                    plot_data["initial_parameters"] = dict(
                        zip(param_names, initial_params)
                    )

            # Try to reconstruct experimental and theoretical data for plotting
            if hasattr(self, "_last_experimental_data") and hasattr(
                self, "_last_phi_angles"
            ):
                plot_data["experimental_data"] = self._last_experimental_data
                plot_data["phi_angles"] = self._last_phi_angles

                # Generate theoretical data using best parameters
                if best_params_list is not None:
                    try:
                        theoretical_data = self._generate_theoretical_data(best_params_list, self._last_phi_angles)  # type: ignore
                        plot_data["theoretical_data"] = theoretical_data
                    except Exception as e:
                        logger.warning(
                            f"Failed to generate theoretical data for plotting: {e}"
                        )

            # Add time axes if available
            temporal = config.get("analyzer_parameters", {}).get("temporal", {})
            dt = temporal.get("dt", 0.1)
            start_frame = temporal.get("start_frame", 1)
            end_frame = temporal.get("end_frame", 1000)

            # Generate time arrays (these are approximate)
            n_frames = end_frame - start_frame + 1
            t_array = np.arange(n_frames) * dt
            plot_data["t1"] = t_array
            plot_data["t2"] = t_array

            # Add parameter names and units for MCMC plotting
            param_names = config.get("initial_parameters", {}).get(
                "parameter_names", []
            )
            if param_names:
                plot_data["parameter_names"] = param_names

            # Extract parameter units from bounds configuration
            parameter_space = config.get("parameter_space", {})
            bounds = parameter_space.get("bounds", [])
            if bounds:
                param_units = [bound.get("unit", "") for bound in bounds]
                plot_data["parameter_units"] = param_units

            # Add MCMC-specific data if available
            if "mcmc_optimization" in results:
                mcmc_results = results["mcmc_optimization"]

                # Add convergence diagnostics
                if "convergence_diagnostics" in mcmc_results:
                    plot_data["mcmc_diagnostics"] = mcmc_results[
                        "convergence_diagnostics"
                    ]

                # Add posterior means
                if "posterior_means" in mcmc_results:
                    plot_data["posterior_means"] = mcmc_results["posterior_means"]

                # Try to get MCMC trace data from live results first
                trace_data = None
                if "trace" in mcmc_results and mcmc_results["trace"] is not None:
                    trace_data = mcmc_results["trace"]
                    logger.debug("Using live MCMC trace data for plotting")
                elif "trace" in results and results["trace"] is not None:
                    # Check top-level trace data as fallback
                    trace_data = results["trace"]
                    logger.debug("Using top-level MCMC trace data for plotting")
                else:
                    # Final fallback: try to load from NetCDF file
                    try:
                        from pathlib import Path

                        mcmc_results_dir = Path("homodyne_results") / "mcmc_results"
                        trace_file = mcmc_results_dir / "mcmc_trace.nc"

                        if trace_file.exists():
                            try:
                                import arviz as az

                                trace_data = az.from_netcdf(str(trace_file))
                                logger.debug(
                                    f"Loaded MCMC trace data from {trace_file}"
                                )
                            except ImportError:
                                logger.warning(
                                    "ArviZ not available - cannot load MCMC trace for plotting"
                                )
                            except Exception as e:
                                logger.warning(f"Failed to load MCMC trace data: {e}")
                        else:
                            logger.debug("MCMC trace file not found")

                    except Exception as e:
                        logger.warning(f"Error checking for MCMC trace file: {e}")

                # Add trace data to plot_data if found
                if trace_data is not None:
                    plot_data["mcmc_trace"] = trace_data
                    logger.info("✓ MCMC trace data available for plotting")
                else:
                    logger.debug(
                        "MCMC trace data not available - trace plots will be skipped"
                    )

            # Add other plot data
            plot_data["chi_squared"] = best_chi2
            plot_data["method"] = best_method.replace("_optimization", "").title()

            return plot_data

        except Exception as e:
            logger.error(f"Error preparing plot data: {e}")
            return None

    def _generate_theoretical_data(
        self, parameters: list, phi_angles: np.ndarray
    ) -> np.ndarray:
        """
        Generate theoretical correlation data for plotting.

        Parameters
        ----------
        parameters : list
            Optimized parameters
        phi_angles : np.ndarray
            Array of phi angles

        Returns
        -------
        np.ndarray
            Theoretical correlation data
        """
        logger = logging.getLogger(__name__)

        try:
            # Use the existing physics model to generate theoretical data
            logger.debug(f"Generating theoretical data for {len(phi_angles)} angles")

            # Call the main correlation calculation method
            theoretical_data = self.calculate_c2_nonequilibrium_laminar_parallel(
                parameters, phi_angles  # type: ignore
            )

            logger.debug(
                f"Successfully generated theoretical data with shape: {theoretical_data.shape}"
            )
            return theoretical_data

        except Exception as e:
            logger.error(f"Error generating theoretical data: {e}")
            # Fallback: return experimental data shape filled with ones if available
            if (
                hasattr(self, "_last_experimental_data")
                and self._last_experimental_data is not None
            ):
                shape = self._last_experimental_data.shape
                logger.warning(f"Using fallback data with shape {shape}")
                return np.ones(shape)
            else:
                logger.warning("No fallback data available")
                return np.array([])


def _get_chi2_interpretation(chi2_value: float) -> str:
    """Provide interpretation of reduced chi-squared value with uncertainty context.

    The reduced chi-squared uncertainty quantifies the reliability of the average:
    - Small uncertainty (< 0.1 * χ²_red): Consistent fit quality across angles
    - Moderate uncertainty (0.1-0.5 * χ²_red): Some angle variation, generally acceptable
    - Large uncertainty (> 0.5 * χ²_red): High variability between angles, potential systematic issues

    Parameters
    ----------
    chi2_value : float
        Reduced chi-squared value

    Returns
    -------
    str
        Interpretation string with quality assessment and statistical meaning
    """
    if chi2_value <= 1.0:
        return f"Excellent fit (χ²_red = {chi2_value:.2f} ≤ 1.0): Model matches data within expected noise"
    elif chi2_value <= 2.0:
        return f"Very good fit (χ²_red = {chi2_value:.2f}): Model captures main features with minor deviations"
    elif chi2_value <= 5.0:
        return f"Acceptable fit (χ²_red = {chi2_value:.2f}): Model reasonable but some systematic deviations present"
    elif chi2_value <= 10.0:
        return f"Poor fit (χ²_red = {chi2_value:.2f}): Significant deviations suggest model inadequacy or underestimated uncertainties"
    else:
        return f"Very poor fit (χ²_red = {chi2_value:.2f}): Major systematic deviations, model likely inappropriate"


def _get_quality_explanation(quality: str) -> str:
    """Provide explanation of quality assessment."""
    explanations = {
        "excellent": "Model provides exceptional agreement with experimental data across all angles",
        "acceptable": "Model provides reasonable agreement with experimental data for most angles",
        "warning": "Model shows concerning deviations that may indicate systematic issues",
        "poor": "Model shows significant inadequacies in describing the experimental data",
        "critical": "Model is fundamentally inappropriate for this dataset",
    }
    return explanations.get(quality, "Unknown quality level")


def _get_quality_recommendations(quality: str, issues: list) -> list:
    """Provide actionable recommendations based on quality assessment."""
    recommendations = []

    if quality == "excellent":
        recommendations.append("Results are reliable for publication")
        recommendations.append("Consider this model for further analysis")
    elif quality == "acceptable":
        recommendations.append("Results may be suitable with appropriate caveats")
        recommendations.append(
            "Consider checking specific angles with higher chi-squared"
        )
    elif quality == "warning":
        recommendations.append("Investigate systematic deviations before publication")
        recommendations.append("Consider alternative models or parameter ranges")
        recommendations.append("Check experimental uncertainties and data quality")
    elif quality in ["poor", "critical"]:
        recommendations.append("Do not use results for quantitative conclusions")
        recommendations.append("Consider fundamental model revision")
        recommendations.append("Check experimental setup and data processing")
        recommendations.append("Investigate alternative theoretical approaches")

    # Add issue-specific recommendations
    for issue in issues:
        if "outliers" in issue.lower():
            recommendations.append(
                "Investigate outlier angles for experimental artifacts"
            )
        if "good angles" in issue.lower():
            recommendations.append(
                "Consider focusing analysis on subset of reliable angles"
            )

    return recommendations


# ============================================================================
# PARAMETER MANAGEMENT AND RESULTS SAVING
# ============================================================================

# Note: Additional methods would be defined here if needed
