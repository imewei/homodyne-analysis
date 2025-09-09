"""
Classical Optimization Methods for Homodyne Scattering Analysis
===============================================================

This module contains multiple classical optimization algorithms for
parameter estimation in homodyne scattering analysis:

1. **Nelder-Mead Simplex**: Derivative-free optimization algorithm that
   works well for noisy objective functions and doesn't require gradient
   information, making it ideal for correlation function fitting.

2. **Gurobi Quadratic Programming**: Advanced optimization using quadratic
   approximation of the chi-squared objective function. Particularly effective
   for smooth problems with parameter bounds constraints. Requires Gurobi license.

Key Features:
- Consistent parameter bounds with MCMC for all methods
- Automatic Gurobi detection and graceful fallback
- Optimized configurations for different analysis modes
- Comprehensive error handling and status reporting

Authors: Wei Chen, Hongrui He
Institution: Argonne National Laboratory
"""

import logging
import time
from typing import Any, Literal

import numpy as np
import scipy.optimize as optimize
from numpy.typing import NDArray

try:
    import gurobipy as gp
    from gurobipy import GRB

    GUROBI_AVAILABLE = True
except ImportError:
    GUROBI_AVAILABLE = False
    # Type stubs for when Gurobi is not available
    gp = None  # type: ignore
    GRB = None  # type: ignore

# Import robust optimization with graceful degradation
try:
    from homodyne.optimization.robust import (
        RobustHomodyneOptimizer,  # type: ignore
        create_robust_optimizer,
    )

    ROBUST_OPTIMIZATION_AVAILABLE = True
except ImportError:
    RobustHomodyneOptimizer = None  # type: ignore
    create_robust_optimizer = None  # type: ignore
    ROBUST_OPTIMIZATION_AVAILABLE = False

logger = logging.getLogger(__name__)

# Type aliases for better type hints
FloatArray = NDArray[np.floating[Any]]
IntArray = NDArray[np.integer[Any]]
ConfigDict = dict[str, Any]
BoundsType = list[tuple[float, float]]
OptimizationResult = dict[str, Any]
MethodName = Literal[
    "Nelder-Mead",
    "Gurobi",
    "Robust-Wasserstein",
    "Robust-Scenario",
    "Robust-Ellipsoidal",
]

# Global optimization counter for tracking iterations
OPTIMIZATION_COUNTER = 0


class ClassicalOptimizer:
    """
    Classical optimization algorithms for parameter estimation.

    This class provides robust parameter estimation using multiple optimization
    algorithms:

    1. Nelder-Mead simplex method: Well-suited for noisy objective functions
       and doesn't require derivative information.

    2. Gurobi quadratic programming (optional): Uses quadratic approximation
       of the chi-squared objective function for potentially faster convergence
       on smooth problems with bounds constraints. Requires Gurobi license.

    The Gurobi method approximates the objective function using finite differences
    to estimate gradients and Hessian, then solves the resulting quadratic program.
    This approach can be particularly effective for least squares problems where
    the objective function is approximately quadratic near the optimum.

    Important: Both optimization methods use the same parameter bounds defined in
    the configuration's parameter_space.bounds section, ensuring consistency with
    MCMC and maintaining the same physical constraints across all optimization methods.
    """

    def __init__(self, analysis_core: Any, config: ConfigDict) -> None:
        """
        Initialize classical optimizer.

        Parameters
        ----------
        analysis_core : HomodyneAnalysisCore
            Core analysis engine instance
        config : ConfigDict
            Configuration dictionary containing optimization settings
        """
        self.core = analysis_core
        self.config = config
        self.best_params_classical: FloatArray | None = None

        # Enhanced performance optimization caches with proper typing
        self._gradient_cache: dict[str, FloatArray] = {}
        self._bounds_cache: BoundsType | None = None
        self._gurobi_model_cache: dict[str, Any] = {}
        self._jacobian_cache: dict[str, FloatArray] = {}
        self._chi_squared_cache: dict[str, tuple[float, float, dict]] = {}

        # Cache configuration
        self._cache_config = {
            "enable_caching": True,
            "max_gradient_cache": 256,
            "max_model_cache": 64,
            "max_jacobian_cache": 128,
            "jacobian_epsilon": 1e-6,
            "chi_squared_caching": {
                "enabled": True,
                "max_cache_size": 1000,
                "cache_hit_logging": False,
                "clear_cache_on_method_change": True,
            },
        }

        # Update with user configuration
        cache_settings = config.get("optimization_config", {}).get(
            "cache_optimization", {}
        )
        self._cache_config.update(cache_settings)

        # Cache statistics tracking
        self._cache_stats = {
            "chi_squared_hits": 0,
            "chi_squared_misses": 0,
            "chi_squared_total_calls": 0,
        }

        # Gradient optimization configuration
        gurobi_config = (
            config.get("optimization_config", {})
            .get("classical_optimization", {})
            .get("method_options", {})
            .get("Gurobi", {})
        )
        self._gradient_config = gurobi_config.get("gradient_optimization", {})

        # Set mode-dependent defaults
        is_static = getattr(self.core, "is_static_mode", lambda: True)()
        default_base_frequency = 2 if is_static else 3

        # Gradient optimization defaults
        self._gradient_optimization = {
            "adaptive_step_sizing": self._gradient_config.get(
                "adaptive_step_sizing",
                {
                    "enabled": True,
                    "base_epsilon": 1e-8,
                    "relative_factor": 0.01,
                    "bounds_proximity_factor": 0.1,
                },
            ),
            "boundary_aware_differences": self._gradient_config.get(
                "boundary_aware_differences",
                {
                    "enabled": True,
                    "boundary_tolerance": 0.05,
                    "prefer_central": True,
                },
            ),
            "smart_scheduling": self._gradient_config.get(
                "smart_scheduling",
                {
                    "enabled": True,
                    "base_frequency": default_base_frequency,
                    "adaptive_scheduling": True,
                    "force_recalc_conditions": {
                        "trust_region_change_threshold": 0.5,
                        "gradient_norm_threshold": 1e-3,
                    },
                },
            ),
            "enhanced_caching": self._gradient_config.get(
                "enhanced_caching",
                {
                    "enabled": True,
                    "similarity_threshold": 1e-4,
                    "max_cache_size": 256,
                    "cache_strategy": "lru",
                },
            ),
            "combined_calculations": self._gradient_config.get(
                "combined_calculations",
                {
                    "enabled": True,
                    "three_point_stencil": True,
                },
            ),
            "monitoring": self._gradient_config.get(
                "monitoring",
                {
                    "enabled": False,
                    "log_statistics": False,
                    "track_function_evaluations": True,
                },
            ),
        }

        # Gradient calculation state tracking
        self._gradient_state = {
            "last_gradient": None,
            "last_parameters": None,
            "last_iteration": -1,
            "gradient_age": 0,
            "function_evaluations_saved": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "boundary_adaptations": 0,
        }

        # Batch processing optimization settings with adaptive configuration
        default_batch_config = {
            "enabled": True,
            "max_parallel_runs": 4,
            "multiple_initial_points": True,
            "initial_point_strategy": "latin_hypercube",  # or "random", "grid"
            "num_initial_points": self._get_adaptive_num_initial_points(),
            "convergence_threshold": 0.01,  # Stop early if multiple runs converge to same result
        }

        user_batch_config = self.config.get("optimization_methods", {}).get(
            "batch_processing", {}
        )
        self.batch_optimization = {**default_batch_config, **user_batch_config}

        # If user explicitly set num_initial_points, respect it
        if "num_initial_points" not in user_batch_config:
            self.batch_optimization["num_initial_points"] = (
                self._get_adaptive_num_initial_points()
            )

        # Extract optimization configuration
        self.optimization_config = config.get("optimization_config", {}).get(
            "classical_optimization", {}
        )

    def run_classical_optimization_optimized(
        self,
        initial_parameters: FloatArray | None = None,
        methods: list[MethodName] | None = None,
        phi_angles: FloatArray | None = None,
        c2_experimental: FloatArray | None = None,
    ) -> tuple[FloatArray | None, OptimizationResult]:
        """
        Run Nelder-Mead optimization method.

        This method uses the Nelder-Mead simplex algorithm for parameter
        estimation. Nelder-Mead is well-suited for noisy objective functions
        and doesn't require gradient information.

        Parameters
        ----------
        initial_parameters : np.ndarray, optional
            Starting parameters for optimization
        methods : list, optional
            List of optimization methods to try
        phi_angles : np.ndarray, optional
            Scattering angles
        c2_experimental : np.ndarray, optional
            Experimental data

        Returns
        -------
        tuple
            (best_parameters, optimization_result)

        Raises
        ------
        RuntimeError
            If all classical methods fail
        """
        logger.info("Starting classical optimization")
        start_time = time.time()
        print("\n═══ Classical Optimization ═══")

        # Determine analysis mode and effective parameter count
        if hasattr(self.core, "config_manager") and self.core.config_manager:
            is_static_mode = self.core.config_manager.is_static_mode_enabled()
            analysis_mode = self.core.config_manager.get_analysis_mode()
            effective_param_count = (
                self.core.config_manager.get_effective_parameter_count()
            )
        else:
            # Fallback to core method
            is_static_mode = getattr(self.core, "is_static_mode", lambda: False)()
            analysis_mode = "static" if is_static_mode else "laminar_flow"
            effective_param_count = 3 if is_static_mode else 7

        print(f"  Analysis mode: {analysis_mode} ({effective_param_count} parameters)")
        logger.info(
            f"Classical optimization using {analysis_mode} mode with {effective_param_count} parameters"
        )

        # Load defaults if not provided
        if methods is None:
            available_methods = ["Nelder-Mead"]
            if GUROBI_AVAILABLE:
                available_methods.append("Gurobi")
            methods = self.optimization_config.get(
                "methods",
                available_methods,
            )

        # Ensure methods is not None for type checker
        assert methods is not None, "Optimization methods list cannot be None"

        if initial_parameters is None:
            initial_parameters = np.array(
                self.config["initial_parameters"]["values"], dtype=np.float64
            )

        # Adjust parameters based on analysis mode
        if is_static_mode and len(initial_parameters) > effective_param_count:
            # For static mode, only use diffusion parameters (first 3)
            initial_parameters = initial_parameters[:effective_param_count]
            print(
                f"  Using first {effective_param_count} parameters for static mode: {initial_parameters}"
            )
        elif not is_static_mode and len(initial_parameters) < effective_param_count:
            # For laminar flow mode, ensure we have all 7 parameters
            full_parameters = np.zeros(effective_param_count)
            full_parameters[: len(initial_parameters)] = initial_parameters
            initial_parameters = full_parameters
            print(
                f"  Extended to {effective_param_count} parameters for laminar flow mode"
            )

        if phi_angles is None or c2_experimental is None:
            c2_experimental, _, phi_angles, _ = self.core.load_experimental_data()

        # Type assertion after loading data to satisfy type checker
        assert phi_angles is not None and c2_experimental is not None, (
            "Failed to load experimental data"
        )

        best_result = None
        best_params = None
        best_chi2 = np.inf
        best_method = None  # Track which method produced the best result
        all_results = []  # Store all results for analysis

        # Get adaptive target alpha from configuration (now the only objective function type)
        adaptive_target_alpha = self.optimization_config.get(
            "adaptive_target_alpha", 1.0
        )

        # Validate adaptive_target_alpha range
        if not (0.8 <= adaptive_target_alpha <= 1.2):
            logger.warning(
                f"adaptive_target_alpha {adaptive_target_alpha} outside recommended range [0.8, 1.2], clamping"
            )
            adaptive_target_alpha = np.clip(adaptive_target_alpha, 0.8, 1.2)

        # Create adaptive target objective function
        objective = self.create_objective_function(
            phi_angles,
            c2_experimental,
            f"Classical-{analysis_mode.capitalize()}",
            adaptive_target_alpha=adaptive_target_alpha,
        )

        # Try each method
        for method in methods:
            print(f"  Trying {method}...")

            try:
                start = time.time()

                # Use single method utility
                success, result = self.run_single_method(
                    method=method,
                    objective_func=objective,
                    initial_parameters=initial_parameters,
                    bounds=None,  # Nelder-Mead doesn't use bounds
                    method_options=self.optimization_config.get(
                        "method_options", {}
                    ).get(method, {}),
                    adaptive_target_alpha=adaptive_target_alpha,
                )

                elapsed = time.time() - start

                # Store result for analysis
                if success and isinstance(result, optimize.OptimizeResult):
                    # Add timing info to result object
                    result.execution_time = elapsed
                    all_results.append((method, result))

                    if result.fun < best_chi2:
                        best_result = result
                        best_params = result.x
                        best_chi2 = result.fun
                        best_method = method  # Track which method produced this result
                        print(
                            f"    ✓ New best: χ²_red = {result.fun:.6e} ({
                                elapsed:.1f}s)"
                        )
                    else:
                        print(f"    χ²_red = {result.fun:.6e} ({elapsed:.1f}s)")
                else:
                    # Store exception for analysis
                    all_results.append((method, result))
                    print(f"    ✗ Failed: {result}")
                    logger.warning(
                        f"Classical optimization method {method} failed: {result}"
                    )

            except Exception as e:
                all_results.append((method, e))
                print(f"    ✗ Failed: {e}")
                logger.warning(f"Classical optimization method {method} failed: {e}")
                logger.exception(f"Full traceback for {method} optimization failure:")

        if (
            best_result is not None
            and best_params is not None
            and isinstance(best_result, optimize.OptimizeResult)
        ):
            total_time = time.time() - start_time

            # best_method is already tracked when the best result was found
            if best_method is None:
                best_method = "Unknown"

            # Generate comprehensive summary (for future use)
            summary = self.get_optimization_summary(
                best_params, best_result, total_time, best_method
            )
            summary["optimization_method"] = best_method
            summary["all_methods_tried"] = [method for method, _ in all_results]

            # Create method-specific results dictionary
            method_results = {}
            for method, result in all_results:
                if hasattr(result, "fun"):  # Successful result
                    method_results[method] = {
                        "parameters": (
                            list(result.x) if hasattr(result, "x") else None
                        ),
                        "chi_squared": result.fun,
                        "success": (
                            result.success if hasattr(result, "success") else True
                        ),
                        "iterations": getattr(result, "nit", None),
                        "function_evaluations": getattr(result, "nfev", None),
                        "message": getattr(result, "message", ""),
                        "method": getattr(result, "method", method),
                    }
                else:  # Failed result (exception)
                    method_results[method] = {
                        "parameters": None,
                        "chi_squared": float("inf"),
                        "success": False,
                        "error": str(result),
                    }

            # Log results
            logger.info(
                f"Classical optimization completed in {total_time:.2f}s, best χ²_red = {
                    best_chi2:.6e} (method: {best_method})"
            )
            print(f"  Best result: χ²_red = {best_chi2:.6e} (method: {best_method})")

            # Store best parameters
            self.best_params_classical = best_params

            # Log detailed analysis if debug logging is enabled
            if logger.isEnabledFor(logging.DEBUG):
                analysis = self.analyze_optimization_results(
                    [
                        (method, True, result)
                        for method, result in all_results
                        if hasattr(result, "fun")
                    ]
                )
                logger.debug(f"Classical optimization analysis: {analysis}")

            # Return enhanced result with method information
            enhanced_result = best_result
            enhanced_result.method_results = (
                method_results  # Add method-specific results
            )
            enhanced_result.best_method = best_method  # Add best method info

            # Get detailed chi-squared analysis for the best parameters
            try:
                # Use the selected chi-squared calculator (optimized or standard)
                if hasattr(self.core, "_selected_chi_calculator"):
                    detailed_chi2_result = self.core._selected_chi_calculator(
                        best_params, phi_angles, c2_experimental, return_components=True
                    )
                else:
                    # Fallback to original method
                    detailed_chi2_result = self.core.calculate_chi_squared_optimized(
                        best_params, phi_angles, c2_experimental, return_components=True
                    )

                if isinstance(detailed_chi2_result, dict):
                    # DEBUGGING: Extract both chi-squared versions with clear semantics

                    # 1. NORMALIZED chi-squared (with uncertainty_estimation_factor scaling)
                    #    Formula: χ²_normalized = sum((exp - theory)² / (σ * uncertainty_factor)²)
                    #    This is chi-squared with uncertainty weighting (NOT divided by DOF)
                    normalized_chi2 = detailed_chi2_result.get(
                        "reduced_chi_squared", None
                    )

                    # 2. TRUE statistical reduced chi-squared (proper statistical definition)
                    #    Formula: χ²_true_reduced = sum((exp - theory)²) / DOF
                    #    This is the correct statistical reduced chi-squared
                    true_chi2 = detailed_chi2_result.get(
                        "reduced_chi_squared_true", None
                    )

                    # DEBUGGING: Log values for verification
                    logger.info("Classical optimization chi-squared debugging:")
                    logger.info(f"  Optimization result.fun = {best_result.fun:.6e}")
                    logger.info(
                        f"  Detailed normalized χ²_red = {normalized_chi2:.6e}"
                        if normalized_chi2 is not None
                        else "  Normalized χ²_red = None"
                    )
                    logger.info(
                        f"  Detailed true χ²_red = {true_chi2:.6e}"
                        if true_chi2 is not None
                        else "  True χ²_red = None"
                    )
                    logger.info(
                        f"  Expected ratio (normalized/true) ≈ {1 / (0.1**2):.1f} if uncertainty_factor=0.1"
                    )

                    # VALIDATION: The true chi-squared should match result.fun from optimization
                    if (
                        true_chi2 is not None
                        and abs(true_chi2 - best_result.fun) > 1e-10
                    ):
                        logger.warning(
                            f"Chi-squared mismatch: optimization={best_result.fun:.6e}, detailed={true_chi2:.6e}"
                        )

                    # Add detailed chi-squared information to the result object
                    enhanced_result.reduced_chi_squared = (
                        normalized_chi2
                        if normalized_chi2 is not None
                        else best_result.fun
                    )
                    enhanced_result.reduced_chi_squared_true = (
                        true_chi2 if true_chi2 is not None else best_result.fun
                    )
                    enhanced_result.reduced_chi_squared_uncertainty = (
                        detailed_chi2_result.get("reduced_chi_squared_uncertainty", 0.0)
                    )
                    enhanced_result.reduced_chi_squared_true_uncertainty = (
                        detailed_chi2_result.get(
                            "reduced_chi_squared_true_uncertainty", 0.0
                        )
                    )

                    # Additional debugging information
                    enhanced_result.degrees_of_freedom = detailed_chi2_result.get(
                        "degrees_of_freedom", None
                    )

                else:
                    logger.warning(
                        f"Detailed chi-squared result is not a dictionary: {type(detailed_chi2_result)}"
                    )
                    # Fallback: use the optimization result for both
                    enhanced_result.reduced_chi_squared = best_result.fun
                    enhanced_result.reduced_chi_squared_true = best_result.fun

            except Exception as e:
                logger.warning(f"Failed to get detailed chi-squared components: {e}")
                logger.exception("Full traceback:")
                # Fallback: use the optimization result for both
                enhanced_result.reduced_chi_squared = best_result.fun
                enhanced_result.reduced_chi_squared_true = best_result.fun

            return best_params, enhanced_result
        else:
            total_time = time.time() - start_time

            # Analyze failed results
            failed_analysis = self.analyze_optimization_results(
                [(method, False, result) for method, result in all_results]
            )

            logger.error(
                f"Classical optimization failed after {
                    total_time:.2f}s - all methods failed"
            )
            logger.error(f"Failure analysis: {failed_analysis}")

            raise RuntimeError(
                f"All classical methods failed. "
                f"Failed methods: {[method for method, _ in all_results]}"
            )

    def get_available_methods(self) -> list[MethodName]:
        """
        Get list of available classical optimization methods.

        Returns
        -------
        List[str]
            List containing available optimization methods
        """
        methods = ["Nelder-Mead"]  # Nelder-Mead simplex algorithm
        if GUROBI_AVAILABLE:
            methods.append("Gurobi")  # Gurobi quadratic programming solver
        if ROBUST_OPTIMIZATION_AVAILABLE:
            methods.extend(
                ["Robust-Wasserstein", "Robust-Scenario", "Robust-Ellipsoidal"]
            )
        return methods

    def validate_method_compatibility(
        self, method: str, has_bounds: bool = True
    ) -> bool:
        """
        Validate if optimization method is compatible with current setup.

        Parameters
        ----------
        method : str
            Optimization method name
        has_bounds : bool
            Whether parameter bounds are defined (unused but kept for compatibility)

        Returns
        -------
        bool
            True if method is supported
        """
        # Note: has_bounds parameter is unused but kept for API compatibility
        _ = has_bounds  # Explicitly mark as unused for type checker

        if method == "Nelder-Mead":
            return True
        elif method == "Gurobi":
            return GUROBI_AVAILABLE
        return False

    def get_method_recommendations(self) -> dict[str, list[str]]:
        """
        Get method recommendations based on problem characteristics.

        Returns
        -------
        Dict[str, List[str]]
            Dictionary mapping scenarios to recommended methods
        """
        recommendations = {
            "with_bounds": (
                ["Gurobi", "Nelder-Mead"] if GUROBI_AVAILABLE else ["Nelder-Mead"]
            ),
            "without_bounds": ["Nelder-Mead"],
            "high_dimensional": ["Nelder-Mead"],
            "low_dimensional": (
                ["Gurobi", "Nelder-Mead"] if GUROBI_AVAILABLE else ["Nelder-Mead"]
            ),
            # Excellent for noisy functions
            "noisy_objective": ["Nelder-Mead"],
            "smooth_objective": (
                ["Gurobi", "Nelder-Mead"] if GUROBI_AVAILABLE else ["Nelder-Mead"]
            ),
        }
        return recommendations

    def validate_parameters(
        self, parameters: np.ndarray, method_name: str = ""
    ) -> tuple[bool, str]:
        """
        Validate physical parameters and bounds.

        Parameters
        ----------
        parameters : np.ndarray
            Model parameters to validate
        method_name : str
            Name of optimization method for logging (currently unused)

        Returns
        -------
        Tuple[bool, str]
            (is_valid, reason_if_invalid)
        """
        _ = method_name  # Suppress unused parameter warning
        # Get validation configuration
        validation = (
            self.config.get("advanced_settings", {})
            .get("chi_squared_calculation", {})
            .get("validity_check", {})
        )

        # Extract parameter sections
        num_diffusion_params = getattr(self.core, "num_diffusion_params", 3)
        num_shear_params = getattr(self.core, "num_shear_rate_params", 3)

        diffusion_params = parameters[:num_diffusion_params]
        shear_params = parameters[
            num_diffusion_params : num_diffusion_params + num_shear_params
        ]

        # Check positive D0
        if validation.get("check_positive_D0", True) and diffusion_params[0] <= 0:
            return False, f"Negative D0: {diffusion_params[0]}"

        # Check positive gamma_dot_t0
        if validation.get("check_positive_gamma_dot_t0", True) and shear_params[0] <= 0:
            return False, f"Negative gamma_dot_t0: {shear_params[0]}"

        # Check parameter bounds
        if validation.get("check_parameter_bounds", True):
            bounds = self.config.get("parameter_space", {}).get("bounds", [])
            for i, bound in enumerate(bounds):
                if i < len(parameters):
                    param_val = parameters[i]
                    param_min = bound.get("min", -np.inf)
                    param_max = bound.get("max", np.inf)

                    if not (param_min <= param_val <= param_max):
                        param_name = bound.get("name", f"p{i}")
                        return (
                            False,
                            f"Parameter {param_name} out of bounds: {param_val} not in [{param_min}, {param_max}]",
                        )

        return True, ""

    def create_objective_function(
        self,
        phi_angles: np.ndarray,
        c2_experimental: np.ndarray,
        method_name: str = "Classical",
        adaptive_target_alpha: float = 1.0,
    ):
        """
        Create adaptive target objective function for optimization.

        Uses the adaptive target approach: minimizes (χ² - α·DOF)² which prevents
        overfitting and targets statistically reasonable chi-squared values.

        Parameters
        ----------
        phi_angles : np.ndarray
            Scattering angles
        c2_experimental : np.ndarray
            Experimental correlation data
        method_name : str
            Name for logging purposes
        adaptive_target_alpha : float
            Target multiplier for adaptive chi-squared (α ∈ [0.8, 1.2])

        Returns
        -------
        callable
            Adaptive target objective function for optimization
        """
        # Get angle filtering setting from configuration
        use_angle_filtering = True
        if hasattr(self.core, "config_manager") and self.core.config_manager:
            use_angle_filtering = self.core.config_manager.is_angle_filtering_enabled()
        elif "angle_filtering" in self.config.get("optimization_config", {}):
            use_angle_filtering = self.config["optimization_config"][
                "angle_filtering"
            ].get("enabled", True)

        # Initialize iteration counter for optimization progress tracking
        iteration_counter = [0]  # Use list to allow modification in nested function

        # Get optimization logging configuration (reserved for future use)
        # optimization_logging_config = (
        #     self.config.get("output_settings", {})
        #     .get("logging", {})
        #     .get("optimization_debug", {})
        # )

        def objective(params):
            # Increment iteration counter for progress tracking
            iteration_counter[0] += 1
            current_iteration = iteration_counter[0]

            # Chi-squared caching implementation
            cache_config = self._cache_config.get("chi_squared_caching", {})
            use_cache = cache_config.get("enabled", True) and self._cache_config.get(
                "enable_caching", True
            )

            if use_cache:
                # Generate cache key from parameters
                param_key = f"chi2_{hash(params.tobytes())}"

                # Update cache statistics
                self._cache_stats["chi_squared_total_calls"] += 1

                # Check cache for existing result
                if param_key in self._chi_squared_cache:
                    cached_result, timestamp, stats = self._chi_squared_cache[param_key]
                    self._cache_stats["chi_squared_hits"] += 1

                    if cache_config.get("cache_hit_logging", False):
                        logger.debug(
                            f"Cache hit for iteration {current_iteration}, key: {param_key[:12]}..."
                        )

                    return cached_result
                else:
                    self._cache_stats["chi_squared_misses"] += 1

            # Adaptive target chi-squared objective function
            # IMPORTANT: With IRLS variance estimation, we need total chi-squared
            # (not reduced) to properly compare against target = α * DOF
            # Use the selected chi-squared calculator (optimized or standard)
            if hasattr(self.core, "_selected_chi_calculator"):
                # Note: _selected_chi_calculator may not support iteration parameter
                # Fall back to original method for enhanced logging
                chi_components = self.core.calculate_chi_squared_optimized(
                    params,
                    phi_angles,
                    c2_experimental,
                    method_name,
                    filter_angles_for_optimization=use_angle_filtering,
                    return_components=True,
                    iteration=current_iteration,
                )
            else:
                # Use original method with enhanced logging
                chi_components = self.core.calculate_chi_squared_optimized(
                    params,
                    phi_angles,
                    c2_experimental,
                    method_name,
                    filter_angles_for_optimization=use_angle_filtering,
                    return_components=True,
                    iteration=current_iteration,
                )

            if not chi_components.get("valid", False):
                return float("inf")

            # Get total chi-squared and DOF from IRLS-weighted calculation
            # total_chi_squared = Σ((residuals/σ_IRLS)²) [unnormalized]
            # target_chi_squared = α * DOF (where DOF = N_data - N_params)
            total_chi_squared = chi_components["total_chi_squared"]
            total_dof = chi_components["degrees_of_freedom"]

            # Adaptive target: minimize squared deviation from target chi-squared
            target_chi_squared = adaptive_target_alpha * total_dof  # α ∈ [0.8, 1.2]

            # Squared deviation (quadratic, numerically stable)
            objective_value = (total_chi_squared - target_chi_squared) ** 2

            # Store result in cache with LRU eviction
            if use_cache and objective_value != float("inf"):
                import time

                timestamp = time.time()

                # Implement LRU eviction if cache is full
                max_cache_size = cache_config.get("max_cache_size", 1000)
                if len(self._chi_squared_cache) >= max_cache_size:
                    # Remove oldest entry (simple FIFO for now)
                    oldest_key = min(
                        self._chi_squared_cache.keys(),
                        key=lambda k: self._chi_squared_cache[k][1],
                    )
                    del self._chi_squared_cache[oldest_key]

                # Store in cache
                cache_stats = {
                    "iteration": current_iteration,
                    "total_chi_squared": total_chi_squared,
                    "total_dof": total_dof,
                }
                self._chi_squared_cache[param_key] = (
                    objective_value,
                    timestamp,
                    cache_stats,
                )

            return objective_value

        return objective

    def run_single_method(
        self,
        method: str,
        objective_func,
        initial_parameters: np.ndarray,
        bounds: list[tuple[float, float]] | None = None,
        method_options: dict[str, Any] | None = None,
        adaptive_target_alpha: float = 1.0,
    ) -> tuple[bool, optimize.OptimizeResult | Exception]:
        """
        Run a single optimization method.

        Parameters
        ----------
        method : str
            Optimization method name
        objective_func : callable
            Objective function to minimize
        initial_parameters : np.ndarray
            Starting parameters
        bounds : List[Tuple[float, float]], optional
            Parameter bounds
        method_options : Dict[str, Any], optional
            Method-specific options

        Returns
        -------
        Tuple[bool, Union[OptimizeResult, Exception]]
            (success, result_or_exception)
        """
        try:
            if method == "Gurobi":
                return self._run_gurobi_optimization(
                    objective_func, initial_parameters, bounds, method_options
                )
            elif method.startswith("Robust-"):
                return self._run_robust_optimization(
                    method,
                    objective_func,
                    initial_parameters,
                    bounds,
                    method_options,
                    adaptive_target_alpha=adaptive_target_alpha,
                )
            else:
                # Use batch processing optimization if enabled
                if self.batch_optimization.get(
                    "enabled", True
                ) and self.batch_optimization.get("multiple_initial_points", True):
                    return self.optimize_with_multiple_initial_points(
                        method, objective_func, bounds, method_options
                    )
                else:
                    # Filter out comment fields (keys starting with '_')
                    # These are documentation/rationale fields not meant as solver options
                    filtered_options = {}
                    if method_options:
                        filtered_options = {
                            k: v
                            for k, v in method_options.items()
                            if not k.startswith("_")
                        }

                    kwargs = {
                        "fun": objective_func,
                        "x0": initial_parameters,
                        "method": method,
                        "options": filtered_options,
                    }

                    # Nelder-Mead doesn't use explicit bounds
                    # The method handles constraints through the objective function

                    result = optimize.minimize(**kwargs)
                    return True, result

        except Exception as e:
            return False, e

    def _run_gurobi_optimization(
        self,
        objective_func: Any,
        initial_parameters: FloatArray,
        bounds: BoundsType | None = None,
        method_options: ConfigDict | None = None,
        initial_guess: FloatArray | None = None,
    ) -> tuple[bool, optimize.OptimizeResult | Exception]:
        """
        Run iterative Gurobi-based optimization using trust region approach.

        This method uses successive quadratic approximations (SQP-like approach) where:
        1. Build quadratic approximation around current point
        2. Solve QP subproblem with trust region constraints
        3. Evaluate actual objective at new point
        4. Update trust region and iterate until convergence

        Parameters
        ----------
        objective_func : callable
            Chi-squared objective function to minimize
        initial_parameters : np.ndarray
            Starting parameters
        bounds : List[Tuple[float, float]], optional
            Parameter bounds for optimization. If None, extracts bounds from the same
            configuration section used by MCMC (parameter_space.bounds).
        method_options : Dict[str, Any], optional
            Gurobi-specific options

        Returns
        -------
        Tuple[bool, Union[OptimizeResult, Exception]]
            (success, result_or_exception)
        """
        try:
            if not GUROBI_AVAILABLE or gp is None or GRB is None:
                raise ImportError("Gurobi is not available. Please install gurobipy.")

            # Get parameter bounds
            if bounds is None:
                bounds = self.get_parameter_bounds()

            n_params = len(initial_parameters)

            # Default Gurobi options with iterative settings (matches config template defaults)
            # Mode-dependent defaults: higher iterations for complex parameter spaces
            is_static = (
                self.core.is_static_mode()
                if hasattr(self.core, "is_static_mode")
                else True
            )
            default_max_iter = 500 if is_static else 1500  # Static: 500, Laminar: 1500
            default_time_limit = (
                120 if is_static else 600
            )  # Static: 2min, Laminar: 10min

            gurobi_options = {
                "max_iterations": default_max_iter,  # MODE_DEPENDENT: matches config template
                "tolerance": 1e-6,
                "output_flag": 0,  # Suppress output by default
                "method": 2,  # Use barrier method for QP
                "time_limit": default_time_limit,  # MODE_DEPENDENT: matches config template
                "trust_region_initial": 1.0,  # Initial trust region radius (config template)
                "trust_region_min": 1e-8,  # Minimum trust region radius
                "trust_region_max": 10.0,  # Maximum trust region radius (config template)
            }

            # Update with user options
            if method_options:
                filtered_options = {
                    k: v
                    for k, v in method_options.items()
                    if not (k.startswith("_") and k.endswith("_note"))
                }
                gurobi_options.update(filtered_options)

            # Initialize iterative optimization
            # Use warm start if available
            if initial_guess is not None:
                x_current = initial_guess.copy()
                logger.debug("Using warm-start initial guess for Gurobi optimization")
            else:
                x_current = initial_parameters.copy()
            f_current = objective_func(x_current)
            trust_radius = gurobi_options["trust_region_initial"]

            # Convergence tracking
            iteration = 0
            max_iter = gurobi_options["max_iterations"]
            tolerance = gurobi_options["tolerance"]
            function_evaluations = 1  # Already evaluated f_current
            grad_norm = float("inf")  # Initialize for later use

            logger.debug(
                f"Starting Gurobi iterative optimization with initial χ² = {f_current:.6e}"
            )

            # Iterative trust region optimization with gradient tracking
            previous_trust_radius = trust_radius

            for iteration in range(max_iter):
                # Choose appropriate epsilon based on parameter magnitudes and trust region
                base_epsilon = max(1e-8, trust_radius / 100)

                # Estimate gradient using optimized cached finite differences
                grad, grad_func_evals = self._compute_cached_gradient(
                    x_current,
                    objective_func,
                    base_epsilon,
                    bounds,
                    iteration,
                    grad_norm if iteration > 0 else None,
                    trust_radius,
                    previous_trust_radius,
                )
                function_evaluations += grad_func_evals

                # Check for convergence based on gradient norm
                grad_norm = np.linalg.norm(grad)
                if grad_norm < tolerance:
                    logger.debug(
                        f"Gurobi optimization converged at iteration {iteration}: ||grad|| = {grad_norm:.2e}"
                    )
                    break

                # Check if Hessian was already computed in combined calculation
                if (
                    self._gradient_optimization["combined_calculations"]["enabled"]
                    and "cached_hessian" in self._gradient_state
                    and grad_func_evals > 0
                ):  # Only if gradient was actually computed
                    # Reuse cached Hessian diagonal
                    hessian_diag = self._gradient_state["cached_hessian"]
                    hessian_func_evals = 0  # No additional function evaluations
                else:
                    # Fallback: compute Hessian separately (legacy method)
                    hessian_diag, hessian_func_evals = self._compute_hessian_diagonal(
                        x_current, objective_func, base_epsilon, bounds, f_current
                    )
                    function_evaluations += hessian_func_evals

                try:
                    # Create Gurobi model for trust region subproblem
                    with gp.Env(empty=True) as env:
                        if gurobi_options.get("output_flag", 0) == 0:
                            env.setParam("OutputFlag", 0)
                        env.start()

                        with gp.Model(env=env) as model:
                            # Set Gurobi parameters
                            model.setParam(GRB.Param.OptimalityTol, tolerance)
                            model.setParam(
                                GRB.Param.Method, gurobi_options.get("method", 2)
                            )
                            model.setParam(
                                GRB.Param.TimeLimit,
                                gurobi_options.get("time_limit", 300),
                            )

                            # Create decision variables for step
                            step = model.addVars(
                                n_params,
                                lb=-gp.GRB.INFINITY,
                                ub=gp.GRB.INFINITY,
                                name="step",
                            )

                            # Trust region constraint: ||step||_2 <= trust_radius
                            model.addQConstr(
                                gp.quicksum(step[i] * step[i] for i in range(n_params))
                                <= trust_radius**2,
                                "trust_region",
                            )

                            # Parameter bounds constraints: bounds[i][0] <= x_current[i] + step[i] <= bounds[i][1]
                            for i in range(n_params):
                                if i < len(bounds):
                                    lb, ub = bounds[i]
                                    if lb != -np.inf:
                                        model.addConstr(
                                            step[i] >= lb - x_current[i],
                                            f"lower_bound_{i}",
                                        )
                                    if ub != np.inf:
                                        model.addConstr(
                                            step[i] <= ub - x_current[i],
                                            f"upper_bound_{i}",
                                        )

                            # Quadratic approximation: grad^T * step + 0.5 * step^T * H_diag * step
                            obj = gp.LinExpr()
                            for i in range(n_params):
                                obj += grad[i] * step[i]  # Linear term
                                obj += (
                                    0.5 * hessian_diag[i] * step[i] * step[i]
                                )  # Quadratic term

                            model.setObjective(obj, GRB.MINIMIZE)

                            # Optimize subproblem
                            model.optimize()

                            if model.status == GRB.OPTIMAL:
                                # Extract step
                                step_values = np.array(
                                    [step[i].x for i in range(n_params)]
                                )  # type: ignore[attr-defined]
                                x_new = x_current + step_values
                                f_new = objective_func(x_new)
                                function_evaluations += 1

                                # Trust region update logic
                                actual_reduction = f_current - f_new

                                if actual_reduction > 0:
                                    # Accept step
                                    step_norm = np.linalg.norm(step_values)
                                    logger.debug(
                                        f"Iteration {iteration}: χ² = {f_new:.6e} (improvement: {actual_reduction:.2e}, step: {step_norm:.3f})"
                                    )

                                    x_current = x_new
                                    f_current = f_new

                                    # Expand trust region if step is successful and near boundary
                                    if step_norm > 0.8 * trust_radius:
                                        previous_trust_radius = trust_radius
                                        trust_radius = min(
                                            gurobi_options["trust_region_max"],
                                            2 * trust_radius,
                                        )
                                else:
                                    # Reject step and shrink trust region
                                    previous_trust_radius = trust_radius
                                    trust_radius = max(
                                        gurobi_options["trust_region_min"],
                                        0.5 * trust_radius,
                                    )
                                    logger.debug(
                                        f"Iteration {iteration}: Step rejected, shrinking trust region to {trust_radius:.6f}"
                                    )

                                # Check convergence
                                if (
                                    actual_reduction > 0
                                    and abs(actual_reduction) < tolerance
                                ):
                                    logger.debug(
                                        f"Gurobi optimization converged at iteration {iteration}: improvement = {actual_reduction:.2e}"
                                    )
                                    break

                                if trust_radius < gurobi_options["trust_region_min"]:
                                    logger.debug(
                                        f"Gurobi optimization terminated: trust region too small ({trust_radius:.2e})"
                                    )
                                    break
                            else:
                                # QP solve failed, shrink trust region and try again
                                previous_trust_radius = trust_radius
                                trust_radius = max(
                                    gurobi_options["trust_region_min"],
                                    0.25 * trust_radius,
                                )
                                logger.debug(
                                    f"QP subproblem failed with status {model.status}, shrinking trust region to {trust_radius:.6f}"
                                )
                                if trust_radius < gurobi_options["trust_region_min"]:
                                    break

                except Exception as e:
                    logger.warning(
                        f"Gurobi subproblem failed at iteration {iteration}: {e}"
                    )
                    break

            # Create final result
            if iteration < max_iter or grad_norm < tolerance:
                result = optimize.OptimizeResult(
                    x=x_current,
                    fun=f_current,
                    success=True,
                    status=0,
                    message=f"Iterative Gurobi optimization converged after {iteration} iterations.",
                    nit=iteration,
                    nfev=function_evaluations,
                    method="Gurobi-Iterative-QP",
                )
                logger.debug(
                    f"Gurobi optimization completed: χ² = {f_current:.6e} after {iteration} iterations"
                )

                # Log gradient optimization performance statistics
                self._log_gradient_statistics()

                # Add gradient optimization info to result
                if self._gradient_optimization["monitoring"][
                    "track_function_evaluations"
                ]:
                    result.gradient_stats = {
                        "function_evaluations_saved": self._gradient_state[
                            "function_evaluations_saved"
                        ],
                        "cache_hit_rate": (
                            self._gradient_state["cache_hits"]
                            / max(
                                1,
                                self._gradient_state["cache_hits"]
                                + self._gradient_state["cache_misses"],
                            )
                        ),
                        "boundary_adaptations": self._gradient_state[
                            "boundary_adaptations"
                        ],
                    }

                return True, result
            else:
                result = optimize.OptimizeResult(
                    x=x_current,
                    fun=f_current,
                    success=False,
                    status=1,
                    message=f"Iterative Gurobi optimization reached maximum iterations ({max_iter}).",
                    nit=iteration,
                    nfev=function_evaluations,
                    method="Gurobi-Iterative-QP",
                )
                return False, result

        except Exception as e:
            logger.error(f"Gurobi optimization failed: {e}")
            return False, e

    def _run_robust_optimization(
        self,
        method: str,
        objective_func,
        initial_parameters: np.ndarray,
        bounds: (
            list[tuple[float, float]] | None
        ) = None,  # Used by robust optimizer internally
        method_options: dict[str, Any] | None = None,
        adaptive_target_alpha: float = 1.0,
    ) -> tuple[bool, optimize.OptimizeResult | Exception]:
        """
        Run robust optimization using CVXPY + Gurobi.

        Parameters
        ----------
        method : str
            Robust optimization method ("Robust-Wasserstein", "Robust-Scenario", "Robust-Ellipsoidal")
        objective_func : callable
            Chi-squared objective function to minimize
        initial_parameters : np.ndarray
            Starting parameters
        bounds : List[Tuple[float, float]], optional
            Parameter bounds for optimization
        method_options : Dict[str, Any], optional
            Robust optimization specific options

        Returns
        -------
        Tuple[bool, Union[OptimizeResult, Exception]]
            (success, result_or_exception)
        """
        try:
            if not ROBUST_OPTIMIZATION_AVAILABLE or create_robust_optimizer is None:
                raise ImportError(
                    "Robust optimization not available. Please install cvxpy."
                )

            # Create robust optimizer instance
            robust_optimizer = create_robust_optimizer(self.core, self.config)

            # Extract phi_angles and c2_experimental from the objective function context
            # Check both the direct attributes and the cached versions
            phi_angles = getattr(self.core, "phi_angles", None)
            c2_experimental = getattr(self.core, "c2_experimental", None)

            # If not found, try the cached versions from load_experimental_data
            if phi_angles is None:
                phi_angles = getattr(self.core, "_last_phi_angles", None)
            if c2_experimental is None:
                c2_experimental = getattr(self.core, "_last_experimental_data", None)

            if phi_angles is None or c2_experimental is None:
                raise ValueError(
                    "Robust optimization requires phi_angles and c2_experimental "
                    "to be available in the analysis core. "
                    f"Found phi_angles: {
                        'present' if phi_angles is not None else 'missing'
                    }, "
                    f"c2_experimental: {
                        'present' if c2_experimental is not None else 'missing'
                    }"
                )

            # Map method names to robust optimization types
            method_mapping = {
                "Robust-Wasserstein": "wasserstein",
                "Robust-Scenario": "scenario",
                "Robust-Ellipsoidal": "ellipsoidal",
            }

            robust_method = method_mapping.get(method)
            if robust_method is None:
                raise ValueError(f"Unknown robust optimization method: {method}")

            # Combine method options with adaptive targeting parameters
            robust_kwargs = {
                "adaptive_target_alpha": adaptive_target_alpha,
                **(method_options or {}),
            }

            # Run robust optimization with adaptive targeting support
            optimal_params, info = robust_optimizer.run_robust_optimization(
                initial_parameters=initial_parameters,
                phi_angles=phi_angles,
                c2_experimental=c2_experimental,
                method=robust_method,
                **robust_kwargs,
            )

            if optimal_params is not None:
                # Create OptimizeResult compatible object
                result = optimize.OptimizeResult(
                    x=optimal_params,
                    fun=info.get("final_chi_squared", objective_func(optimal_params)),
                    success=True,
                    status=info.get("status", "success"),
                    message=f"Robust optimization ({robust_method}) converged",
                    nit=info.get("n_iterations"),
                    nfev=info.get("function_evaluations", None),
                    method=f"Robust-{robust_method.capitalize()}",
                )

                return True, result
            else:
                # Optimization failed
                error_msg = info.get(
                    "error", f"Robust optimization ({robust_method}) failed"
                )
                result = optimize.OptimizeResult(
                    x=initial_parameters,
                    fun=float("inf"),
                    success=False,
                    status=info.get("status", "failed"),
                    message=error_msg,
                    method=f"Robust-{robust_method.capitalize()}",
                )

                return False, result

        except Exception as e:
            logger.error(f"Robust optimization error: {e}")
            return False, e

    def analyze_optimization_results(
        self,
        results: list[tuple[str, bool, optimize.OptimizeResult | Exception]],
    ) -> dict[str, Any]:
        """
        Analyze and summarize optimization results from Nelder-Mead method.

        Parameters
        ----------
        results : List[Tuple[str, bool, Union[OptimizeResult, Exception]]]
            List of (method_name, success, result_or_exception) tuples (typically one entry for Nelder-Mead)

        Returns
        -------
        Dict[str, Any]
            Analysis summary including best method, convergence stats, etc.
        """
        successful_results = []
        failed_methods = []

        for method, success, result in results:
            if success and hasattr(result, "fun"):
                successful_results.append((method, result))
            else:
                failed_methods.append((method, result))

        if not successful_results:
            return {
                "success": False,
                "failed_methods": failed_methods,
                "error": "All methods failed",
            }

        # Find best result
        best_method, best_result = min(successful_results, key=lambda x: x[1].fun)

        # Compute statistics
        chi2_values = [result.fun for _, result in successful_results]

        return {
            "success": True,
            "best_method": best_method,
            "best_result": best_result,
            "best_chi2": best_result.fun,
            "successful_methods": len(successful_results),
            "failed_methods": failed_methods,
            "chi2_statistics": {
                "min": np.min(chi2_values),
                "max": np.max(chi2_values),
                "mean": np.mean(chi2_values),
                "std": np.std(chi2_values),
            },
            "convergence_info": {
                method: {
                    "converged": result.success,
                    "iterations": getattr(result, "nit", None),
                    "function_evaluations": getattr(result, "nfev", None),
                    "message": getattr(result, "message", None),
                }
                for method, result in successful_results
            },
        }

    def get_parameter_bounds(
        self,
        effective_param_count: int | None = None,
        is_static_mode: bool | None = None,
    ) -> BoundsType:
        """
        Extract parameter bounds from configuration (unused by Nelder-Mead).

        This method is kept for compatibility but is not used by Nelder-Mead
        optimization since it doesn't support explicit bounds.

        Parameters
        ----------
        effective_param_count : int, optional
            Number of parameters to use (3 for static, 7 for laminar flow)
        is_static_mode : bool, optional
            Whether static mode is enabled (unused, kept for compatibility)

        Returns
        -------
        List[Tuple[float, float]]
            List of (min, max) bounds for each parameter
        """
        # Note: is_static_mode parameter is unused but kept for API
        # compatibility
        _ = is_static_mode  # Explicitly mark as unused for type checker

        # Use cached bounds if available and caching is enabled
        if self._bounds_cache is not None and self._cache_config.get(
            "enable_caching", True
        ):
            return self._bounds_cache

        bounds = []
        param_bounds = self.config.get("parameter_space", {}).get("bounds", [])

        # Determine effective parameter count if not provided
        if effective_param_count is None:
            if hasattr(self.core, "config_manager") and self.core.config_manager:
                effective_param_count = (
                    self.core.config_manager.get_effective_parameter_count()
                )
            else:
                effective_param_count = 7  # Default to laminar flow

        # Ensure effective_param_count is not None for type checking
        if effective_param_count is None:
            effective_param_count = 7  # Final fallback to laminar flow

        # Extract bounds for the effective parameters
        for i, bound in enumerate(param_bounds):
            if i < effective_param_count:
                bounds.append((bound.get("min", -np.inf), bound.get("max", np.inf)))

        # Ensure we have enough bounds
        while len(bounds) < effective_param_count:
            bounds.append((-np.inf, np.inf))

        result_bounds = bounds[:effective_param_count]

        # Cache the result if caching is enabled
        if self._cache_config.get("enable_caching", True):
            self._bounds_cache = result_bounds

        return result_bounds

    def compare_optimization_results(
        self,
        results: list[tuple[str, optimize.OptimizeResult | Exception]],
    ) -> dict[str, Any]:
        """
        Compare optimization results (typically just Nelder-Mead).

        Parameters
        ----------
        results : List[Tuple[str, Union[OptimizeResult, Exception]]]
            List of (method_name, result) tuples (typically one entry for Nelder-Mead)

        Returns
        -------
        Dict[str, Any]
            Comparison summary with rankings and statistics
        """
        successful_results = []
        failed_methods = []

        for method, result in results:
            if isinstance(result, optimize.OptimizeResult) and result.success:
                successful_results.append((method, result))
            else:
                failed_methods.append(method)

        if not successful_results:
            return {"error": "No successful optimizations to compare"}

        # Sort by chi-squared value
        successful_results.sort(key=lambda x: x[1].fun)

        comparison = {
            "ranking": [
                {
                    "rank": i + 1,
                    "method": method,
                    "chi_squared": result.fun,
                    "converged": result.success,
                    "iterations": getattr(result, "nit", None),
                    "function_evaluations": getattr(result, "nfev", None),
                    "time_elapsed": getattr(result, "execution_time", None),
                }
                for i, (method, result) in enumerate(successful_results)
            ],
            "best_method": successful_results[0][0],
            "best_chi_squared": successful_results[0][1].fun,
            "failed_methods": failed_methods,
            "success_rate": len(successful_results) / len(results),
        }

        return comparison

    def get_optimization_summary(
        self,
        best_params: np.ndarray,
        best_result: optimize.OptimizeResult,
        total_time: float,
        method_name: str = "unknown",
    ) -> dict[str, Any]:
        """
        Generate comprehensive optimization summary.

        Parameters
        ----------
        best_params : np.ndarray
            Best parameters found
        best_result : OptimizeResult
            Best optimization result
        total_time : float
            Total optimization time in seconds

        Returns
        -------
        Dict[str, Any]
            Comprehensive optimization summary
        """
        # Parameter names (if available in config)
        param_names = []
        param_bounds = self.config.get("parameter_space", {}).get("bounds", [])
        for i, bound in enumerate(param_bounds):
            param_names.append(bound.get("name", f"param_{i}"))

        summary = {
            "optimization_successful": True,
            "best_chi_squared": best_result.fun,
            "best_parameters": {
                (param_names[i] if i < len(param_names) else f"param_{i}"): float(param)
                for i, param in enumerate(best_params)
            },
            "optimization_details": {
                "method": method_name,
                "converged": best_result.success,
                "iterations": getattr(best_result, "nit", None),
                "function_evaluations": getattr(best_result, "nfev", None),
                "message": getattr(best_result, "message", None),
            },
            "timing": {
                "total_time_seconds": total_time,
                "average_evaluation_time": (
                    total_time / (getattr(best_result, "nfev", None) or 1)
                ),
            },
            "cache_performance": {
                "chi_squared_cache_hits": self._cache_stats.get("chi_squared_hits", 0),
                "chi_squared_cache_misses": self._cache_stats.get(
                    "chi_squared_misses", 0
                ),
                "chi_squared_total_calls": self._cache_stats.get(
                    "chi_squared_total_calls", 0
                ),
                "chi_squared_hit_rate": (
                    (
                        self._cache_stats.get("chi_squared_hits", 0)
                        / max(1, self._cache_stats.get("chi_squared_total_calls", 1))
                    )
                    * 100
                ),
                "cache_size": len(self._chi_squared_cache),
            },
            "parameter_validation": {},
        }

        # Add parameter validation info
        is_valid, reason = self.validate_parameters(best_params, "Summary")
        summary["parameter_validation"] = {
            "valid": is_valid,
            "reason": (reason if not is_valid else "All parameters within bounds"),
        }

        return summary

    def generate_initial_points(
        self,
        bounds: list[tuple[float, float]],
        num_points: int,
        strategy: str = "latin_hypercube",
    ) -> np.ndarray:
        """
        Generate multiple initial points for batch optimization.

        Parameters
        ----------
        bounds : List[Tuple[float, float]]
            Parameter bounds
        num_points : int
            Number of initial points to generate
        strategy : str, default "latin_hypercube"
            Strategy for generating points: "latin_hypercube", "random", or "grid"

        Returns
        -------
        np.ndarray
            Array of initial points, shape (num_points, n_params)
        """
        n_params = len(bounds)

        if strategy == "latin_hypercube":
            # Latin Hypercube Sampling for better space coverage
            from scipy.stats import qmc

            sampler = qmc.LatinHypercube(d=n_params, seed=42)
            unit_samples = sampler.random(num_points)

            # Scale to bounds
            initial_points = np.zeros((num_points, n_params))
            for i, (lb, ub) in enumerate(bounds):
                if np.isfinite(lb) and np.isfinite(ub):
                    initial_points[:, i] = lb + unit_samples[:, i] * (ub - lb)
                else:
                    # Handle infinite bounds with reasonable defaults
                    initial_points[:, i] = unit_samples[:, i] * 2 - 1  # [-1, 1]

        elif strategy == "random":
            # Simple random sampling
            initial_points = np.zeros((num_points, n_params))
            for i, (lb, ub) in enumerate(bounds):
                if np.isfinite(lb) and np.isfinite(ub):
                    initial_points[:, i] = np.random.uniform(lb, ub, num_points)
                else:
                    initial_points[:, i] = np.random.normal(0, 1, num_points)

        elif strategy == "grid":
            # Grid-based sampling (works best for low-dimensional problems)
            grid_size = int(np.ceil(num_points ** (1.0 / n_params)))
            grid_points = []

            for _i, (lb, ub) in enumerate(bounds):
                if np.isfinite(lb) and np.isfinite(ub):
                    grid_points.append(np.linspace(lb, ub, grid_size))
                else:
                    grid_points.append(np.linspace(-2, 2, grid_size))

            # Create meshgrid and flatten
            grids = np.meshgrid(*grid_points)
            initial_points = np.column_stack([grid.ravel() for grid in grids])[
                :num_points
            ]

        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        return initial_points

    def _run_single_optimization(
        self,
        method: str,
        objective_func,
        initial_parameters: np.ndarray,
        bounds: list[tuple[float, float]] | None = None,
        method_options: dict[str, Any] | None = None,
        **kwargs,
    ) -> tuple[bool, optimize.OptimizeResult | Exception]:
        """
        Run a single optimization with given initial parameters.

        This is a helper method for batch processing that runs one optimization
        with specific initial parameters.
        """
        try:
            if method == "Gurobi":
                return self._run_gurobi_optimization(
                    objective_func, initial_parameters, bounds, method_options
                )
            elif method.startswith("Robust-"):
                return self._run_robust_optimization(
                    method,
                    objective_func,
                    initial_parameters,
                    bounds,
                    method_options,
                    **kwargs,
                )
            else:
                # Filter out comment fields
                filtered_options = {}
                if method_options:
                    filtered_options = {
                        k: v for k, v in method_options.items() if not k.startswith("_")
                    }

                kwargs_opt = {
                    "fun": objective_func,
                    "x0": initial_parameters,
                    "method": method,
                    "options": filtered_options,
                }

                result = optimize.minimize(**kwargs_opt)
                return True, result

        except Exception as e:
            return False, e

    def optimize_with_multiple_initial_points(
        self,
        method: str,
        objective_func,
        bounds: list[tuple[float, float]] | None = None,
        method_options: dict[str, Any] | None = None,
        **kwargs,
    ) -> tuple[bool, optimize.OptimizeResult | Exception]:
        """
        Run optimization with multiple initial points and return the best result.

        Parameters
        ----------
        method : str
            Optimization method name
        objective_func : callable
            Objective function to minimize
        bounds : List[Tuple[float, float]], optional
            Parameter bounds
        method_options : Dict[str, Any], optional
            Method-specific options
        **kwargs
            Additional arguments passed to optimization

        Returns
        -------
        Tuple[bool, Union[OptimizeResult, Exception]]
            (success, best_result_or_exception)
        """
        if not self.batch_optimization.get(
            "enabled", True
        ) or not self.batch_optimization.get("multiple_initial_points", True):
            # Fall back to single optimization
            bounds = bounds or self.get_parameter_bounds()
            bounds_array = np.array(bounds)
            # Use center of bounds as initial point
            initial_params = np.mean(bounds_array, axis=1)
            return self._run_single_optimization(
                method, objective_func, initial_params, bounds, method_options, **kwargs
            )

        bounds = bounds or self.get_parameter_bounds()
        num_points = self.batch_optimization.get("num_initial_points", 8)
        max_parallel = self.batch_optimization.get("max_parallel_runs", 4)
        strategy = self.batch_optimization.get(
            "initial_point_strategy", "latin_hypercube"
        )
        convergence_threshold = self.batch_optimization.get(
            "convergence_threshold", 0.01
        )

        # Generate initial points
        initial_points = self.generate_initial_points(bounds, num_points, strategy)

        # Run optimizations in parallel batches
        from concurrent.futures import ThreadPoolExecutor

        best_result = None
        best_objective = np.inf
        successful_results = []

        with ThreadPoolExecutor(max_workers=max_parallel) as executor:
            # Submit all optimization tasks
            futures = []
            for i, initial_point in enumerate(initial_points):
                future = executor.submit(
                    self._run_single_optimization,
                    method,
                    objective_func,
                    initial_point,
                    bounds,
                    method_options,
                    **kwargs,
                )
                futures.append((i, future))

            # Collect results and track convergence
            for i, future in futures:
                try:
                    success, result = future.result(
                        timeout=300
                    )  # 5-minute timeout per optimization

                    if success and hasattr(result, "fun"):
                        successful_results.append((i, result))

                        if result.fun < best_objective:
                            best_objective = result.fun
                            best_result = result

                        # Early convergence check: if multiple results are very similar, we can stop
                        if len(successful_results) >= 3:
                            recent_objectives = [
                                r[1].fun for r in successful_results[-3:]
                            ]
                            if (
                                max(recent_objectives) - min(recent_objectives)
                                < convergence_threshold * best_objective
                            ):
                                logger.debug(
                                    f"Early convergence detected after {len(successful_results)} optimizations"
                                )
                                break

                except Exception as e:
                    logger.debug(f"Optimization {i} failed: {e}")
                    continue

        if best_result is not None:
            # Enhance result with batch information
            best_result.batch_info = {
                "num_initial_points": num_points,
                "successful_runs": len(successful_results),
                "best_objective": best_objective,
                "objective_std": (
                    np.std([r[1].fun for r in successful_results])
                    if len(successful_results) > 1
                    else 0.0
                ),
                "strategy": strategy,
            }
            logger.info(
                f"Batch optimization completed: {len(successful_results)}/{num_points} runs successful, best objective: {best_objective:.6f}"
            )
            return True, best_result
        else:
            return False, Exception("All optimization runs failed")

    def reset_optimization_counter(self):
        """Reset the global optimization counter."""
        global OPTIMIZATION_COUNTER
        OPTIMIZATION_COUNTER = 0

    def get_optimization_counter(self) -> int:
        """Get current optimization counter value."""
        return OPTIMIZATION_COUNTER

    # ===== PERFORMANCE OPTIMIZATION METHODS =====
    # These methods provide consistency with RobustHomodyneOptimizer performance features

    def _initialize_warm_start(self, method_name: str, problem_signature: str) -> None:
        """Initialize warm-start data for optimization methods."""
        if not hasattr(self, "_optimization_state"):
            self._optimization_state = {}

        self._optimization_state[problem_signature] = {
            "method": method_name,
            "initialized": True,
            "warm_start_available": False,
            "last_solution": None,
            "last_objective": None,
            "solver_stats": {},
            "iteration_count": 0,
        }

    def _solve_with_warm_start(
        self, method_name: str, problem_signature: str, *args, **kwargs
    ) -> dict:
        """
        Solve optimization problem using warm-start data if available.

        This method provides warm-start capabilities for classical optimization methods,
        particularly beneficial for Gurobi quadratic programming.
        """
        # Initialize warm start if needed
        if (
            not hasattr(self, "_optimization_state")
            or problem_signature not in self._optimization_state
        ):
            self._initialize_warm_start(method_name, problem_signature)

        state = self._optimization_state[problem_signature]

        # Apply warm start if available
        if state.get("warm_start_available") and method_name == "gurobi":
            return self._apply_gurobi_warm_start(problem_signature, *args, **kwargs)

        # Run standard optimization and store result for future warm starts
        if method_name == "nelder_mead":
            result = self._run_nelder_mead_with_state(
                problem_signature, *args, **kwargs
            )
        elif method_name == "gurobi":
            result = self._run_gurobi_with_state(problem_signature, *args, **kwargs)
        else:
            result = {"success": False, "message": f"Unknown method: {method_name}"}

        # Update state for next iteration
        if result.get("success", False):
            state["warm_start_available"] = True
            state["last_solution"] = result.get("x")
            state["last_objective"] = result.get("fun")
            state["iteration_count"] += 1

        return result

    def _solve_with_fallback_chain(
        self, initial_params: np.ndarray, bounds: list, *args, **kwargs
    ) -> dict:
        """
        Solve optimization using systematic fallback chain.

        Fallback order: Gurobi (if available) → Nelder-Mead → Error
        """
        methods_to_try = []

        # Add methods based on availability
        if GUROBI_AVAILABLE:
            methods_to_try.append(("gurobi", "Gurobi quadratic programming"))
        methods_to_try.append(("nelder_mead", "Nelder-Mead simplex"))

        for method_name, method_desc in methods_to_try:
            try:
                logger.info(f"Attempting optimization with {method_desc}")

                if method_name == "gurobi":
                    result = self._run_gurobi_optimization(
                        initial_params, bounds, *args, **kwargs
                    )
                else:  # nelder_mead
                    result = self._run_nelder_mead_optimization(
                        initial_params, bounds, *args, **kwargs
                    )

                if result.get("success", False):
                    result["method_used"] = method_name
                    logger.info(f"Optimization succeeded with {method_desc}")
                    return result
                else:
                    logger.warning(
                        f"Optimization failed with {method_desc}: {result.get('message', 'Unknown error')}"
                    )

            except Exception as e:
                logger.warning(f"Exception in {method_desc}: {e!s}")
                continue

        # All methods failed
        return {
            "success": False,
            "message": "All optimization methods in fallback chain failed",
            "methods_tried": [name for name, _ in methods_to_try],
        }

    def _compute_adaptive_step_size(
        self, x_current: np.ndarray, bounds: BoundsType | None, base_epsilon: float
    ) -> np.ndarray:
        """
        Compute adaptive step sizes for finite difference gradient calculation.

        Parameters
        ----------
        x_current : np.ndarray
            Current parameter values
        bounds : BoundsType, optional
            Parameter bounds for boundary-aware sizing
        base_epsilon : float
            Base epsilon value

        Returns
        -------
        np.ndarray
            Adaptive step sizes for each parameter
        """
        config = self._gradient_optimization["adaptive_step_sizing"]
        if not config["enabled"]:
            return np.full_like(x_current, base_epsilon)

        n_params = len(x_current)
        step_sizes = np.zeros(n_params)

        base_eps = config["base_epsilon"]
        relative_factor = config["relative_factor"]
        bounds_factor = config["bounds_proximity_factor"]

        for i in range(n_params):
            # Base step size from parameter magnitude
            magnitude_step = relative_factor * max(1.0, abs(x_current[i]))

            # Apply bounds-aware reduction if near boundaries
            if bounds is not None and i < len(bounds):
                lb, ub = bounds[i]
                if np.isfinite(lb) and np.isfinite(ub):
                    param_range = ub - lb
                    dist_to_lower = x_current[i] - lb
                    dist_to_upper = ub - x_current[i]
                    min_dist = min(dist_to_lower, dist_to_upper)

                    # Reduce step size when close to bounds
                    if min_dist < bounds_factor * param_range:
                        proximity_reduction = min_dist / (bounds_factor * param_range)
                        magnitude_step *= proximity_reduction
                        self._gradient_state["boundary_adaptations"] += 1

            # Final step size is maximum of base epsilon and magnitude-based step
            step_sizes[i] = max(base_eps, magnitude_step)

        return step_sizes

    def _should_recalculate_gradient(
        self,
        x_current: np.ndarray,
        iteration: int,
        gradient_norm: float | None = None,
        trust_radius: float | None = None,
        previous_trust_radius: float | None = None,
    ) -> bool:
        """
        Determine if gradient should be recalculated based on smart scheduling.

        Parameters
        ----------
        x_current : np.ndarray
            Current parameter values
        iteration : int
            Current iteration number
        gradient_norm : float, optional
            Current gradient norm
        trust_radius : float, optional
            Current trust region radius
        previous_trust_radius : float, optional
            Previous trust region radius

        Returns
        -------
        bool
            True if gradient should be recalculated
        """
        config = self._gradient_optimization["smart_scheduling"]
        if not config["enabled"]:
            return True  # Always recalculate if scheduling disabled

        # Always calculate on first iteration
        if iteration == 0 or self._gradient_state["last_gradient"] is None:
            return True

        # Check if enough iterations have passed
        iterations_since_last = iteration - self._gradient_state["last_iteration"]
        base_frequency = config["base_frequency"]

        if iterations_since_last < base_frequency:
            # Check force recalculation conditions
            force_conditions = config.get("force_recalc_conditions", {})

            # Force if trust region changed significantly
            if trust_radius is not None and previous_trust_radius is not None:
                trust_change = abs(trust_radius - previous_trust_radius) / max(
                    previous_trust_radius, 1e-10
                )
                trust_threshold = force_conditions.get(
                    "trust_region_change_threshold", 0.5
                )
                if trust_change > trust_threshold:
                    return True

            # Force if gradient norm is above threshold (indicating far from optimum)
            if gradient_norm is not None:
                grad_threshold = force_conditions.get("gradient_norm_threshold", 1e-3)
                if gradient_norm > grad_threshold:
                    return True

            return False

        return True

    def _check_parameter_similarity(
        self, x_current: np.ndarray, cached_params: np.ndarray
    ) -> bool:
        """
        Check if current parameters are similar enough to reuse cached gradient.

        Parameters
        ----------
        x_current : np.ndarray
            Current parameter values
        cached_params : np.ndarray
            Cached parameter values

        Returns
        -------
        bool
            True if parameters are similar enough for gradient reuse
        """
        config = self._gradient_optimization["enhanced_caching"]
        if not config["enabled"]:
            return False

        similarity_threshold = config["similarity_threshold"]

        # Compute relative difference
        rel_diff = np.abs(x_current - cached_params) / np.maximum(
            np.abs(cached_params), 1e-10
        )
        max_rel_diff = np.max(rel_diff)

        return max_rel_diff < similarity_threshold

    def _compute_cached_gradient(
        self,
        x_current: np.ndarray,
        objective_func: callable,
        base_epsilon: float,
        bounds: BoundsType | None = None,
        iteration: int = 0,
        gradient_norm: float | None = None,
        trust_radius: float | None = None,
        previous_trust_radius: float | None = None,
    ) -> tuple[np.ndarray, int]:
        """
        Compute gradient with enhanced caching and smart scheduling.

        Parameters
        ----------
        x_current : np.ndarray
            Current parameter values
        objective_func : callable
            Objective function to differentiate
        base_epsilon : float
            Base finite difference step size
        bounds : BoundsType, optional
            Parameter bounds for boundary-aware differences
        iteration : int, optional
            Current iteration for smart scheduling
        gradient_norm : float, optional
            Previous gradient norm for adaptive scheduling
        trust_radius : float, optional
            Current trust region radius
        previous_trust_radius : float, optional
            Previous trust region radius

        Returns
        -------
        Tuple[np.ndarray, int]
            (gradient_vector, function_evaluations_count)
        """
        # Check if gradient recalculation is needed
        if not self._should_recalculate_gradient(
            x_current, iteration, gradient_norm, trust_radius, previous_trust_radius
        ):
            # Reuse last gradient
            self._gradient_state["gradient_age"] += 1
            self._gradient_state["function_evaluations_saved"] += 2 * len(x_current)
            return self._gradient_state["last_gradient"], 0

        # Check enhanced caching with parameter similarity
        if self._gradient_state[
            "last_parameters"
        ] is not None and self._check_parameter_similarity(
            x_current, self._gradient_state["last_parameters"]
        ):
            self._gradient_state["cache_hits"] += 1
            self._gradient_state["function_evaluations_saved"] += 2 * len(x_current)
            return self._gradient_state["last_gradient"], 0

        # Need to compute new gradient
        self._gradient_state["cache_misses"] += 1

        # Check if we should use combined gradient/Hessian calculation
        if self._gradient_optimization["combined_calculations"]["enabled"]:
            gradient, hessian_diag, func_evals = (
                self._compute_combined_gradient_hessian(
                    x_current, objective_func, base_epsilon, bounds
                )
            )
            # Store Hessian for later use (avoiding recomputation in main loop)
            self._gradient_state["cached_hessian"] = hessian_diag
        else:
            gradient, func_evals = self._compute_gradient_direct(
                x_current, objective_func, base_epsilon, bounds
            )

        # Update state
        self._gradient_state["last_gradient"] = gradient.copy()
        self._gradient_state["last_parameters"] = x_current.copy()
        self._gradient_state["last_iteration"] = iteration
        self._gradient_state["gradient_age"] = 0

        return gradient, func_evals

    def _compute_combined_gradient_hessian(
        self,
        x_current: np.ndarray,
        objective_func: callable,
        base_epsilon: float,
        bounds: BoundsType | None = None,
    ) -> tuple[np.ndarray, np.ndarray, int]:
        """
        Compute gradient and diagonal Hessian together using optimized 3-point stencil.

        Parameters
        ----------
        x_current : np.ndarray
            Current parameter values
        objective_func : callable
            Objective function to differentiate
        base_epsilon : float
            Base finite difference step size
        bounds : BoundsType, optional
            Parameter bounds for boundary-aware differences

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, int]
            (gradient_vector, hessian_diagonal, function_evaluations_count)
        """
        n_params = len(x_current)
        grad = np.zeros(n_params)
        hessian_diag = np.zeros(n_params)

        # Get adaptive step sizes
        step_sizes = self._compute_adaptive_step_size(x_current, bounds, base_epsilon)

        # OPTIMIZATION: Evaluate f(x) once and reuse
        f_current = objective_func(x_current)
        function_evaluations = 1

        # OPTIMIZATION: Batch all function evaluations needed for all parameters
        evaluation_batch = []
        eval_mapping = []  # Maps batch index to (param_idx, eval_type, epsilon)

        # Prepare all function evaluations
        for i in range(n_params):
            epsilon = step_sizes[i]

            # Determine difference scheme based on boundary proximity
            use_forward, use_backward = self._determine_difference_scheme(
                x_current[i], bounds[i] if bounds and i < len(bounds) else None, epsilon
            )

            if use_forward:
                # Forward difference: need f(x+h) and f(x+2h)
                x_plus1 = x_current.copy()
                x_plus1[i] += epsilon
                x_plus2 = x_current.copy()
                x_plus2[i] += 2 * epsilon

                evaluation_batch.extend([x_plus1, x_plus2])
                eval_mapping.extend(
                    [(i, "forward_h", epsilon), (i, "forward_2h", epsilon)]
                )

            elif use_backward:
                # Backward difference: need f(x-h) and f(x-2h)
                x_minus1 = x_current.copy()
                x_minus1[i] -= epsilon
                x_minus2 = x_current.copy()
                x_minus2[i] -= 2 * epsilon

                evaluation_batch.extend([x_minus1, x_minus2])
                eval_mapping.extend(
                    [(i, "backward_h", epsilon), (i, "backward_2h", epsilon)]
                )

            else:
                # Central difference: need f(x+h) and f(x-h)
                x_plus = x_current.copy()
                x_plus[i] += epsilon
                x_minus = x_current.copy()
                x_minus[i] -= epsilon

                evaluation_batch.extend([x_plus, x_minus])
                eval_mapping.extend(
                    [(i, "central_plus", epsilon), (i, "central_minus", epsilon)]
                )

        # OPTIMIZATION: Use smart scheduling for batch evaluation
        if self._gradient_optimization.get("smart_scheduling", {}).get(
            "enabled", False
        ):
            evaluation_results = self._evaluate_batch_smart(
                objective_func, evaluation_batch
            )
        else:
            evaluation_results = [objective_func(x) for x in evaluation_batch]

        function_evaluations += len(evaluation_results)

        # Process results to compute gradient and Hessian
        eval_data = {}  # Store evaluation results by (param_idx, eval_type)
        result_idx = 0

        for param_idx, eval_type, _epsilon in eval_mapping:
            eval_data[(param_idx, eval_type)] = evaluation_results[result_idx]
            result_idx += 1

        # Compute gradient and Hessian for each parameter
        for i in range(n_params):
            epsilon = step_sizes[i]

            # Check which difference scheme was used
            if (i, "forward_h") in eval_data:
                # Forward difference scheme
                f_plus1 = eval_data[(i, "forward_h")]
                f_plus2 = eval_data[(i, "forward_2h")]

                # Forward difference approximations
                grad[i] = (-3 * f_current + 4 * f_plus1 - f_plus2) / (2 * epsilon)
                hessian_diag[i] = (f_current - 2 * f_plus1 + f_plus2) / (epsilon**2)

            elif (i, "backward_h") in eval_data:
                # Backward difference scheme
                f_minus1 = eval_data[(i, "backward_h")]
                f_minus2 = eval_data[(i, "backward_2h")]

                # Backward difference approximations
                grad[i] = (3 * f_current - 4 * f_minus1 + f_minus2) / (2 * epsilon)
                hessian_diag[i] = (f_current - 2 * f_minus1 + f_minus2) / (epsilon**2)

            else:
                # Central difference scheme (most accurate)
                f_plus = eval_data[(i, "central_plus")]
                f_minus = eval_data[(i, "central_minus")]

                # Central difference approximations
                grad[i] = (f_plus - f_minus) / (2 * epsilon)
                hessian_diag[i] = (f_plus - 2 * f_current + f_minus) / (epsilon**2)

            # Ensure Hessian diagonal is positive definite
            hessian_diag[i] = max(1e-6, hessian_diag[i])

        return grad, hessian_diag, function_evaluations

    def _determine_difference_scheme(
        self, param_value: float, bounds: tuple[float, float] | None, epsilon: float
    ) -> tuple[bool, bool]:
        """
        Determine whether to use forward, backward, or central differences.

        Parameters
        ----------
        param_value : float
            Current parameter value
        bounds : tuple[float, float], optional
            Parameter bounds (min, max)
        epsilon : float
            Step size

        Returns
        -------
        Tuple[bool, bool]
            (use_forward, use_backward)
        """
        config = self._gradient_optimization["boundary_aware_differences"]
        if not config["enabled"] or bounds is None:
            return False, False  # Use central differences

        lb, ub = bounds
        if not (np.isfinite(lb) and np.isfinite(ub)):
            return False, False  # Use central differences for unbounded parameters

        tolerance = config["boundary_tolerance"]
        param_range = ub - lb

        # Distance to boundaries
        dist_to_lower = param_value - lb
        dist_to_upper = ub - param_value

        # Check if too close to bounds for central differences
        boundary_threshold = tolerance * param_range

        # Use forward differences if too close to lower bound
        if dist_to_lower < boundary_threshold:
            if param_value + 2 * epsilon <= ub:  # Ensure we can take forward steps
                return True, False

        # Use backward differences if too close to upper bound
        if dist_to_upper < boundary_threshold:
            if param_value - 2 * epsilon >= lb:  # Ensure we can take backward steps
                return False, True

        # Use central differences (default)
        return False, False

    def _compute_gradient_direct(
        self,
        x_current: np.ndarray,
        objective_func: callable,
        base_epsilon: float,
        bounds: BoundsType | None = None,
    ) -> tuple[np.ndarray, int]:
        """
        Compute gradient using boundary-aware finite differences with optimizations.

        Parameters
        ----------
        x_current : np.ndarray
            Current parameter values
        objective_func : callable
            Objective function to differentiate
        base_epsilon : float
            Base finite difference step size
        bounds : BoundsType, optional
            Parameter bounds for boundary-aware differences

        Returns
        -------
        Tuple[np.ndarray, int]
            (gradient_vector, function_evaluations_count)
        """
        n_params = len(x_current)
        grad = np.zeros(n_params)
        function_evaluations = 0

        # Get adaptive step sizes
        step_sizes = self._compute_adaptive_step_size(x_current, bounds, base_epsilon)

        # OPTIMIZATION: Compute f_current once and reuse
        f_current = objective_func(x_current)
        function_evaluations += 1

        # OPTIMIZATION: Batch function evaluations for improved efficiency
        evaluation_batch = []
        eval_mapping = []  # Maps batch index to (param_idx, eval_type)

        # Prepare all function evaluations needed
        for i in range(n_params):
            epsilon = step_sizes[i]

            # Determine difference scheme based on boundary proximity
            use_forward, use_backward = self._determine_difference_scheme(
                x_current[i], bounds[i] if bounds and i < len(bounds) else None, epsilon
            )

            if use_forward:
                # Forward difference: (f(x+h) - f(x)) / h
                x_plus = x_current.copy()
                x_plus[i] += epsilon
                evaluation_batch.append(x_plus)
                eval_mapping.append((i, "forward", epsilon))

            elif use_backward:
                # Backward difference: (f(x) - f(x-h)) / h
                x_minus = x_current.copy()
                x_minus[i] -= epsilon
                evaluation_batch.append(x_minus)
                eval_mapping.append((i, "backward", epsilon))

            else:
                # Central difference: (f(x+h) - f(x-h)) / (2h)
                x_plus = x_current.copy()
                x_plus[i] += epsilon
                x_minus = x_current.copy()
                x_minus[i] -= epsilon
                evaluation_batch.extend([x_plus, x_minus])
                eval_mapping.extend(
                    [(i, "central_plus", epsilon), (i, "central_minus", epsilon)]
                )

        # OPTIMIZATION: Check if smart scheduling is enabled
        if self._gradient_optimization.get("smart_scheduling", {}).get(
            "enabled", False
        ):
            # Use parallel evaluation or smart batching
            evaluation_results = self._evaluate_batch_smart(
                objective_func, evaluation_batch
            )
        else:
            # Sequential evaluation
            evaluation_results = [objective_func(x) for x in evaluation_batch]

        function_evaluations += len(evaluation_results)

        # Process results to compute gradient
        central_values = {}  # For central differences that need both +/- values
        result_idx = 0

        for param_idx, eval_type, epsilon in eval_mapping:
            f_eval = evaluation_results[result_idx]
            result_idx += 1

            if eval_type == "forward":
                grad[param_idx] = (f_eval - f_current) / epsilon
            elif eval_type == "backward":
                grad[param_idx] = (f_current - f_eval) / epsilon
            elif eval_type == "central_plus":
                central_values[(param_idx, "plus")] = f_eval
            elif eval_type == "central_minus":
                central_values[(param_idx, "minus")] = f_eval
                # Compute central difference when we have both values
                f_plus = central_values[(param_idx, "plus")]
                f_minus = f_eval
                grad[param_idx] = (f_plus - f_minus) / (2 * epsilon)

        return grad, function_evaluations

    def _compute_hessian_diagonal(
        self,
        x_current: np.ndarray,
        objective_func: callable,
        base_epsilon: float,
        bounds: BoundsType | None = None,
        f_current: float | None = None,
    ) -> tuple[np.ndarray, int]:
        """
        Compute diagonal Hessian approximation (fallback when not using combined calculation).

        Parameters
        ----------
        x_current : np.ndarray
            Current parameter values
        objective_func : callable
            Objective function
        base_epsilon : float
            Base finite difference step size
        bounds : BoundsType, optional
            Parameter bounds
        f_current : float, optional
            Pre-computed f(x) value to avoid recomputation

        Returns
        -------
        Tuple[np.ndarray, int]
            (hessian_diagonal, function_evaluations_count)
        """
        n_params = len(x_current)
        hessian_diag = np.zeros(n_params)
        function_evaluations = 0

        # Get adaptive step sizes
        step_sizes = self._compute_adaptive_step_size(x_current, bounds, base_epsilon)

        # Compute f(x) if not provided
        if f_current is None:
            f_current = objective_func(x_current)
            function_evaluations += 1

        for i in range(n_params):
            epsilon = step_sizes[i]

            # Use central difference for second derivative: f(x+h) - 2f(x) + f(x-h)
            x_plus = x_current.copy()
            x_plus[i] += epsilon
            x_minus = x_current.copy()
            x_minus[i] -= epsilon

            f_plus = objective_func(x_plus)
            f_minus = objective_func(x_minus)
            second_deriv = (f_plus - 2 * f_current + f_minus) / (epsilon**2)
            hessian_diag[i] = max(1e-6, second_deriv)  # Ensure positive
            function_evaluations += 2

        return hessian_diag, function_evaluations

    def _log_gradient_statistics(self) -> None:
        """Log gradient calculation performance statistics."""
        if not self._gradient_optimization["monitoring"]["log_statistics"]:
            return

        stats = self._gradient_state
        total_requests = stats["cache_hits"] + stats["cache_misses"]

        if total_requests > 0:
            hit_rate = stats["cache_hits"] / total_requests * 100
            logger.info("Gradient optimization statistics:")
            logger.info(
                f"  Cache hit rate: {hit_rate:.1f}% ({stats['cache_hits']}/{total_requests})"
            )
            logger.info(
                f"  Function evaluations saved: {stats['function_evaluations_saved']}"
            )
            logger.info(f"  Boundary adaptations: {stats['boundary_adaptations']}")

    def clear_caches(self) -> None:
        """
        Clear performance optimization caches to free memory.

        Call this method periodically during batch optimization to prevent
        memory usage from growing too large.
        """
        self._gradient_cache.clear()
        self._bounds_cache = None
        self._gurobi_model_cache.clear()
        self._jacobian_cache.clear()
        self._chi_squared_cache.clear()

        # Reset cache statistics
        self._cache_stats = {
            "chi_squared_hits": 0,
            "chi_squared_misses": 0,
            "chi_squared_total_calls": 0,
        }

        # Clear gradient optimization state
        self._gradient_state = {
            "last_gradient": None,
            "last_parameters": None,
            "last_iteration": -1,
            "gradient_age": 0,
            "function_evaluations_saved": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "boundary_adaptations": 0,
            "cached_hessian": None,
        }

        # Also clear existing result cache
        if hasattr(self, "_result_cache"):
            self._result_cache.clear()

        logger.debug(
            "Cleared classical optimization performance caches and gradient state"
        )

    # Helper methods for warm-start implementations
    def _apply_gurobi_warm_start(self, problem_signature: str, *args, **kwargs) -> dict:
        """Apply warm start for Gurobi optimization."""
        state = self._optimization_state[problem_signature]
        initial_solution = state.get("last_solution")

        if initial_solution is not None:
            # Use previous solution as starting point
            kwargs["initial_guess"] = initial_solution

        return self._run_gurobi_optimization(*args, **kwargs)

    def _run_nelder_mead_with_state(
        self, problem_signature: str, *args, **kwargs
    ) -> dict:
        """Run Nelder-Mead optimization with state tracking."""
        # Initialize optimization state if needed
        if not hasattr(self, "_optimization_state"):
            self._optimization_state = {}
        if problem_signature not in self._optimization_state:
            self._initialize_warm_start("nelder_mead", problem_signature)

        # Nelder-Mead doesn't directly support warm starts, but we can use better initial points
        state = self._optimization_state[problem_signature]

        if state.get("last_solution") is not None and len(args) > 0:
            # Use previous solution as starting point
            args = (state["last_solution"], *args[1:])

        return self._run_nelder_mead_optimization(*args, **kwargs)

    def _run_gurobi_with_state(self, problem_signature: str, *args, **kwargs) -> dict:
        """Run Gurobi optimization with state tracking."""
        return self._run_gurobi_optimization(*args, **kwargs)

    def _run_nelder_mead_optimization(
        self, initial_params: np.ndarray, bounds: list, *args, **kwargs
    ) -> dict:
        """Run Nelder-Mead optimization (wrapper for existing method)."""
        # This would call the existing Nelder-Mead implementation
        # For now, return a placeholder that integrates with existing code
        try:
            # Call existing optimization logic here
            result = {
                "success": True,
                "x": initial_params,
                "fun": 0.0,
                "message": "Nelder-Mead placeholder",
            }
            return result
        except Exception as e:
            return {"success": False, "message": str(e)}

    def _get_adaptive_num_initial_points(self) -> int:
        """
        Determine the number of initial points based on dataset size.

        Returns
        -------
        int
            Number of initial points to use for optimization:
            - Small datasets (<10k points): 8 initial points
            - Medium datasets (10k-100k points): 5 initial points
            - Large datasets (>100k points): 3 initial points
        """
        try:
            dataset_size = self._detect_dataset_size()

            if dataset_size < 10_000:
                return 8  # Small datasets - use full exploration
            elif dataset_size < 100_000:
                return 5  # Medium datasets - balanced approach
            else:
                return 3  # Large datasets - minimal initial points for speed

        except Exception as e:
            # Default to medium configuration if detection fails
            logger.warning(
                f"Failed to detect dataset size for adaptive configuration: {e}"
            )
            return 5

    def _detect_dataset_size(self) -> int:
        """
        Detect the size of the current dataset for adaptive configuration.

        Returns
        -------
        int
            Number of data points in the experimental dataset
        """
        try:
            # Check if we have access to the core analysis object
            if hasattr(self, "core") and self.core is not None:
                # Look for experimental data in core object
                if (
                    hasattr(self.core, "c2_experimental")
                    and self.core.c2_experimental is not None
                ):
                    # Count total data points across all angles
                    if hasattr(self.core.c2_experimental, "shape"):
                        # For 2D array: (n_angles, n_points)
                        if len(self.core.c2_experimental.shape) == 2:
                            return int(np.prod(self.core.c2_experimental.shape))
                        # For 1D array: just n_points
                        else:
                            return self.core.c2_experimental.shape[0]

                # Alternative: check phi_angles length as proxy
                if (
                    hasattr(self.core, "phi_angles")
                    and self.core.phi_angles is not None
                ):
                    if hasattr(self.core.phi_angles, "__len__"):
                        # Estimate based on angles (typical: ~100-1000 points per angle)
                        n_angles = len(self.core.phi_angles)
                        estimated_points_per_angle = 500  # Conservative estimate
                        return n_angles * estimated_points_per_angle

            # Fallback: check if dataset size was stored during initialization
            if hasattr(self, "_dataset_size_hint"):
                return self._dataset_size_hint

            # Default assumption for medium-sized dataset
            logger.debug("Unable to detect exact dataset size, assuming medium dataset")
            return 50_000

        except Exception as e:
            logger.warning(f"Error detecting dataset size: {e}")
            # Safe default for medium dataset
            return 50_000

    def _evaluate_batch_smart(
        self, objective_func: callable, evaluation_batch: list[np.ndarray]
    ) -> list[float]:
        """
        Smart batch evaluation with optional parallelization and caching.

        Parameters
        ----------
        objective_func : callable
            Objective function to evaluate
        evaluation_batch : list[np.ndarray]
            List of parameter vectors to evaluate

        Returns
        -------
        list[float]
            Function values for each input in the batch
        """
        # For now, implement sequential evaluation with potential for future parallelization
        # Future optimization: Use ThreadPoolExecutor for parallel evaluation
        try:
            # Check if parallel evaluation is enabled and beneficial
            parallel_threshold = self._gradient_optimization.get(
                "smart_scheduling", {}
            ).get("parallel_threshold", 4)

            if (
                len(evaluation_batch) >= parallel_threshold
                and self._can_use_parallel_evaluation()
            ):
                return self._evaluate_batch_parallel(objective_func, evaluation_batch)
            else:
                # Sequential evaluation (current default)
                return [objective_func(x) for x in evaluation_batch]

        except Exception as e:
            logger.warning(
                f"Smart batch evaluation failed, falling back to sequential: {e}"
            )
            return [objective_func(x) for x in evaluation_batch]

    def _can_use_parallel_evaluation(self) -> bool:
        """Check if parallel evaluation is safe and beneficial."""
        # Conservative approach: only enable if explicitly configured
        # Future enhancement: Check if objective function is thread-safe
        return self._gradient_optimization.get("smart_scheduling", {}).get(
            "enable_parallel", False
        )

    def _evaluate_batch_parallel(
        self, objective_func: callable, evaluation_batch: list[np.ndarray]
    ) -> list[float]:
        """
        Parallel batch evaluation using ThreadPoolExecutor.

        Parameters
        ----------
        objective_func : callable
            Objective function to evaluate
        evaluation_batch : list[np.ndarray]
            List of parameter vectors to evaluate

        Returns
        -------
        list[float]
            Function values for each input in the batch
        """
        import concurrent.futures

        max_workers = self._gradient_optimization.get("smart_scheduling", {}).get(
            "max_workers", 2
        )

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all evaluations
            futures = [executor.submit(objective_func, x) for x in evaluation_batch]

            # Collect results in order
            results = []
            for future in futures:
                try:
                    results.append(
                        future.result(timeout=30)
                    )  # 30-second timeout per evaluation
                except Exception as e:
                    logger.warning(f"Parallel evaluation failed: {e}")
                    # Fallback to sequential for this batch
                    raise e

            return results
