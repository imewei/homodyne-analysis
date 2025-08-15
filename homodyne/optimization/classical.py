"""
Classical Optimization Methods for Homodyne Scattering Analysis
==============================================================

This module contains classical optimization algorithms extracted from the
ConfigurableHomodyneAnalysis class, including:
- Scipy-based optimization methods (L-BFGS-B, TNC, SLSQP, etc.)
- Parameter bounds handling
- Multi-method optimization with fallback strategies
- Performance tracking and logging

The classical optimization serves as the first stage in the multi-stage
optimization pipeline, providing good initial parameter estimates for
subsequent Bayesian and MCMC methods.

Authors: Wei Chen, Hongrui He
Institution: Argonne National Laboratory & University of Chicago
"""

import time
import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import scipy.optimize as optimize

logger = logging.getLogger(__name__)

# Global optimization counter for tracking iterations across all methods
OPTIMIZATION_COUNTER = 0


class ClassicalOptimizer:
    """
    Classical optimization algorithms for parameter estimation.

    This class provides robust classical optimization using multiple scipy
    algorithms with intelligent fallback strategies and parameter bounds
    handling.
    """

    def __init__(self, analysis_core, config: Dict[str, Any]):
        """
        Initialize classical optimizer.

        Parameters
        ----------
        analysis_core : HomodyneAnalysisCore
            Core analysis engine instance
        config : Dict[str, Any]
            Configuration dictionary
        """
        self.core = analysis_core
        self.config = config
        self.best_params_classical = None

        # Extract optimization configuration
        self.optimization_config = config.get("optimization_config", {}).get(
            "classical_optimization", {}
        )

    def run_classical_optimization_optimized(
        self,
        initial_parameters: Optional[np.ndarray] = None,
        methods: Optional[List[str]] = None,
        phi_angles: Optional[np.ndarray] = None,
        c2_experimental: Optional[np.ndarray] = None,
    ) -> Tuple[Optional[np.ndarray], Any]:
        """
        Run classical optimization methods.

        This method systematically tries multiple classical optimization
        algorithms to find the best parameter estimates. It provides robust
        optimization with fallback strategies and comprehensive logging.

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
        if hasattr(self.core, 'config_manager') and self.core.config_manager:
            is_static_mode = self.core.config_manager.is_static_mode_enabled()
            analysis_mode = self.core.config_manager.get_analysis_mode()
            effective_param_count = self.core.config_manager.get_effective_parameter_count()
        else:
            # Fallback to core method
            is_static_mode = getattr(self.core, 'is_static_mode', lambda: False)()
            analysis_mode = "static" if is_static_mode else "laminar_flow"
            effective_param_count = 3 if is_static_mode else 7

        print(f"  Analysis mode: {analysis_mode} ({effective_param_count} parameters)")
        logger.info(f"Classical optimization using {analysis_mode} mode with {effective_param_count} parameters")

        # Load defaults if not provided
        if methods is None:
            methods = self.optimization_config.get(
                "methods",
                ["L-BFGS-B", "TNC", "SLSQP", "Powell", "Nelder-Mead"],
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
            print(f"  Using first {effective_param_count} parameters for static mode: {initial_parameters}")
        elif not is_static_mode and len(initial_parameters) < effective_param_count:
            # For laminar flow mode, ensure we have all 7 parameters
            full_parameters = np.zeros(effective_param_count)
            full_parameters[:len(initial_parameters)] = initial_parameters
            initial_parameters = full_parameters
            print(f"  Extended to {effective_param_count} parameters for laminar flow mode")

        if phi_angles is None or c2_experimental is None:
            c2_experimental, _, phi_angles, _ = self.core.load_experimental_data()

        # Type assertion after loading data to satisfy type checker
        assert (
            phi_angles is not None and c2_experimental is not None
        ), "Failed to load experimental data"

        best_result = None
        best_params = None
        best_chi2 = np.inf
        all_results = []  # Store all results for analysis

        # Get parameter bounds using utility method (adjusted for mode)
        bounds = self.get_parameter_bounds(effective_param_count, is_static_mode)

        # Create objective function using utility method
        objective = self.create_objective_function(
            phi_angles, c2_experimental, f"Classical-{analysis_mode.capitalize()}"
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
                    bounds=(bounds if method in ["L-BFGS-B", "TNC", "SLSQP"] else None),
                    method_options=self.optimization_config.get(
                        "method_options", {}
                    ).get(method, {}),
                )

                elapsed = time.time() - start

                # Store result for analysis
                if success and isinstance(result, optimize.OptimizeResult):
                    # Add timing info to result object
                    setattr(result, "execution_time", elapsed)
                    all_results.append((method, result))

                    if result.fun < best_chi2:
                        best_result = result
                        best_params = result.x
                        best_chi2 = result.fun
                        print(
                            f"    ✓ New best: χ²_red = {result.fun:.6e} ({elapsed:.1f}s)"
                        )
                    else:
                        print(f"    χ²_red = {result.fun:.6e} ({elapsed:.1f}s)")
                else:
                    all_results.append((method, result))  # Store exception for analysis
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

            # Generate comprehensive summary (for future use)
            _ = self.get_optimization_summary(best_params, best_result, total_time)

            # Log results
            logger.info(
                f"Classical optimization completed in {total_time:.2f}s, best χ²_red = {best_chi2:.6e}"
            )
            print(f"  Best result: χ²_red = {best_chi2:.6e}")

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

            return best_params, best_result
        else:
            total_time = time.time() - start_time

            # Analyze failed results
            failed_analysis = self.analyze_optimization_results(
                [(method, False, result) for method, result in all_results]
            )

            logger.error(
                f"Classical optimization failed after {total_time:.2f}s - all methods failed"
            )
            logger.error(f"Failure analysis: {failed_analysis}")

            raise RuntimeError(
                f"All classical methods failed. "
                f"Failed methods: {[method for method, _ in all_results]}"
            )

    def get_available_methods(self) -> List[str]:
        """
        Get list of available classical optimization methods.

        Returns
        -------
        List[str]
            List of available scipy optimization methods
        """
        return [
            "L-BFGS-B",  # Limited-memory BFGS with bounds
            "TNC",  # Truncated Newton with bounds
            "SLSQP",  # Sequential Least Squares Programming
            "Powell",  # Powell's method
            "Nelder-Mead",  # Nelder-Mead simplex
            "CG",  # Conjugate gradient
            "BFGS",  # BFGS
            "Newton-CG",  # Newton conjugate gradient
            "trust-ncg",  # Newton conjugate gradient trust-region
            "trust-exact",  # Exact trust-region
            "trust-krylov",  # Krylov trust-region
        ]

    def validate_method_compatibility(self, method: str, has_bounds: bool) -> bool:
        """
        Validate if optimization method is compatible with current setup.

        Parameters
        ----------
        method : str
            Optimization method name
        has_bounds : bool
            Whether parameter bounds are defined

        Returns
        -------
        bool
            True if method is compatible
        """
        bounds_required_methods = ["L-BFGS-B", "TNC", "SLSQP"]

        if has_bounds and method in bounds_required_methods:
            return True
        elif not has_bounds and method not in bounds_required_methods:
            return True
        else:
            return method not in bounds_required_methods

    def get_method_recommendations(self) -> Dict[str, List[str]]:
        """
        Get method recommendations based on problem characteristics.

        Returns
        -------
        Dict[str, List[str]]
            Dictionary mapping scenarios to lists of recommended methods
        """
        return {
            "with_bounds": ["L-BFGS-B", "TNC", "SLSQP"],
            "without_bounds": ["BFGS", "Powell", "Nelder-Mead"],
            "high_dimensional": ["L-BFGS-B", "BFGS", "CG"],
            "low_dimensional": ["Nelder-Mead", "Powell"],
            "noisy_objective": ["Powell", "Nelder-Mead"],
            "smooth_objective": ["L-BFGS-B", "BFGS", "Newton-CG"],
        }

    def validate_parameters(
        self, parameters: np.ndarray, method_name: str = ""
    ) -> Tuple[bool, str]:
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
    ):
        """
        Create objective function for optimization.

        Parameters
        ----------
        phi_angles : np.ndarray
            Scattering angles
        c2_experimental : np.ndarray
            Experimental correlation data
        method_name : str
            Name for logging purposes

        Returns
        -------
        callable
            Objective function for optimization
        """
        # Get angle filtering setting from configuration
        use_angle_filtering = True
        if hasattr(self.core, "config_manager") and self.core.config_manager:
            use_angle_filtering = self.core.config_manager.is_angle_filtering_enabled()
        elif "angle_filtering" in self.config.get("optimization_config", {}):
            use_angle_filtering = self.config["optimization_config"][
                "angle_filtering"
            ].get("enabled", True)

        def objective(params):
            return self.core.calculate_chi_squared_optimized(
                params,
                phi_angles,
                c2_experimental,
                method_name,
                filter_angles_for_optimization=use_angle_filtering,
            )

        return objective

    def run_single_method(
        self,
        method: str,
        objective_func,
        initial_parameters: np.ndarray,
        bounds: Optional[List[Tuple[float, float]]] = None,
        method_options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[bool, Union[optimize.OptimizeResult, Exception]]:
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
            # Filter out comment fields (keys starting with '_' and ending with '_note')
            filtered_options = {}
            if method_options:
                filtered_options = {
                    k: v
                    for k, v in method_options.items()
                    if not (k.startswith("_") and k.endswith("_note"))
                }

            kwargs = {
                "fun": objective_func,
                "x0": initial_parameters,
                "method": method,
                "options": filtered_options,
            }

            # Add bounds for methods that support them
            if method in ["L-BFGS-B", "TNC", "SLSQP"] and bounds:
                kwargs["bounds"] = bounds

            result = optimize.minimize(**kwargs)
            return True, result

        except Exception as e:
            return False, e

    def analyze_optimization_results(
        self,
        results: List[Tuple[str, bool, Union[optimize.OptimizeResult, Exception]]],
    ) -> Dict[str, Any]:
        """
        Analyze and summarize optimization results from multiple methods.

        Parameters
        ----------
        results : List[Tuple[str, bool, Union[OptimizeResult, Exception]]]
            List of (method_name, success, result_or_exception) tuples

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

    def get_parameter_bounds(self, effective_param_count: Optional[int] = None, is_static_mode: Optional[bool] = None) -> List[Tuple[float, float]]:
        """
        Extract parameter bounds from configuration, adjusted for analysis mode.

        Parameters
        ----------
        effective_param_count : int, optional
            Number of parameters to use (3 for static, 7 for laminar flow)
        is_static_mode : bool, optional
            Whether static mode is enabled

        Returns
        -------
        List[Tuple[float, float]]
            List of (min, max) bounds for each parameter
        """
        bounds = []
        param_bounds = self.config.get("parameter_space", {}).get("bounds", [])
        
        # Determine effective parameter count if not provided
        if effective_param_count is None:
            if hasattr(self.core, 'config_manager') and self.core.config_manager:
                effective_param_count = self.core.config_manager.get_effective_parameter_count()
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
            
        return bounds[:effective_param_count]

    def compare_optimization_results(
        self,
        results: List[Tuple[str, Union[optimize.OptimizeResult, Exception]]],
    ) -> Dict[str, Any]:
        """
        Compare optimization results from different methods.

        Parameters
        ----------
        results : List[Tuple[str, Union[OptimizeResult, Exception]]]
            List of (method_name, result) tuples

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
    ) -> Dict[str, Any]:
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
                "method": getattr(best_result, "method", "unknown"),
                "converged": best_result.success,
                "iterations": getattr(best_result, "nit", None),
                "function_evaluations": getattr(best_result, "nfev", None),
                "message": getattr(best_result, "message", None),
            },
            "timing": {
                "total_time_seconds": total_time,
                "average_evaluation_time": (
                    total_time / getattr(best_result, "nfev", 1)
                ),
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

    def reset_optimization_counter(self):
        """Reset the global optimization counter."""
        global OPTIMIZATION_COUNTER
        OPTIMIZATION_COUNTER = 0

    def get_optimization_counter(self) -> int:
        """Get current optimization counter value."""
        return OPTIMIZATION_COUNTER
