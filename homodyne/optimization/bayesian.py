"""
Bayesian Optimization Methods for Homodyne Scattering Analysis
=============================================================

This module contains Bayesian optimization algorithms extracted from the
ConfigurableHomodyneAnalysis class, including:
- Scikit-optimize (skopt) Gaussian Process optimization
- Intelligent search space construction
- Acquisition function strategies
- Warm start capabilities from classical optimization

Bayesian optimization provides efficient exploration of the parameter space
using Gaussian Process models to guide the search toward promising regions.

Authors: Wei Chen, Hongrui He
Institution: Argonne National Laboratory & University of Chicago
"""

import time
import logging
from typing import Any, Dict, List, Optional

import numpy as np

# Bayesian optimization dependencies
try:
    from skopt import gp_minimize
    from skopt.space import Real
    from skopt.utils import use_named_args

    SKOPT_AVAILABLE = True
except ImportError:
    # Mock functions and classes when skopt is not available
    gp_minimize = None
    Real = None
    use_named_args = None
    SKOPT_AVAILABLE = False

logger = logging.getLogger(__name__)


class BayesianOptimizer:
    """
    Bayesian optimization for efficient parameter space exploration.

    This class implements Gaussian Process-based optimization using
    scikit-optimize, providing intelligent parameter space exploration
    with acquisition function-guided sampling.
    """

    def __init__(self, analysis_core, config: Dict[str, Any]):
        """
        Initialize Bayesian optimizer.

        Parameters
        ----------
        analysis_core : HomodyneAnalysisCore
            Core analysis engine instance
        config : Dict[str, Any]
            Configuration dictionary
        """
        self.core = analysis_core
        self.config = config
        self.best_params_bo = None

        # Extract Bayesian optimization configuration
        self.bo_config = config.get("optimization_config", {}).get(
            "bayesian_optimization", {}
        )

        if not SKOPT_AVAILABLE:
            logger.warning(
                "scikit-optimize not available - Bayesian optimization disabled"
            )

    def run_bayesian_optimization_skopt_optimized(
        self,
        phi_angles: Optional[np.ndarray] = None,
        c2_experimental: Optional[np.ndarray] = None,
        bo_config: Optional[Dict[str, Any]] = None,
        x0: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Run Bayesian optimization using scikit-optimize.

        This method provides efficient parameter space exploration using
        Gaussian Process models to guide the optimization process. It
        supports warm starts from classical optimization results.

        Parameters
        ----------
        phi_angles : np.ndarray, optional
            Scattering angles
        c2_experimental : np.ndarray, optional
            Experimental data
        bo_config : dict, optional
            Bayesian optimization configuration
        x0 : np.ndarray, optional
            Initial point for warm start

        Returns
        -------
        dict
            Optimization results including best parameters and acquisition history

        Raises
        ------
        ImportError
            If scikit-optimize is not available
        """
        if not SKOPT_AVAILABLE:
            raise ImportError("scikit-optimize not available")

        # Type assertions for mypy/pylance - these are guaranteed to be available after the check
        assert use_named_args is not None
        assert gp_minimize is not None

        logger.info("Starting Bayesian optimization with scikit-optimize")
        start_time = time.time()
        print("\n═══ Bayesian Optimization ═══")

        # Load data if needed
        if phi_angles is None or c2_experimental is None:
            c2_experimental, _, phi_angles, _ = (
                self.core.load_experimental_data()
            )

        # Type assertion after loading data
        assert (
            phi_angles is not None and c2_experimental is not None
        ), "Failed to load experimental data"

        # Configuration
        if bo_config is None:
            bo_config = self.bo_config or {}

        # Ensure bo_config is not None for type checker
        assert bo_config is not None

        # Build search space
        parameter_space = self._build_search_space()

        @use_named_args(parameter_space)
        def objective(**params):
            param_array = np.array([params[s.name] for s in parameter_space])
            return self.core.calculate_chi_squared_optimized(
                param_array,
                phi_angles,
                c2_experimental,
                "BayesOpt",
                filter_angles_for_optimization=True,  # Use angle filtering during optimization
            )

        # Run optimization
        n_calls = bo_config.get("n_calls", 50)
        n_initial = bo_config.get("n_initial_points", 10)

        print(f"  Running {n_calls} evaluations ({n_initial} initial)...")
        start = time.time()

        kwargs = {
            "func": objective,
            "dimensions": parameter_space,
            "n_calls": n_calls,
            "n_initial_points": n_initial,
            "acq_func": bo_config.get("acquisition_func", "EI"),
            "random_state": bo_config.get("random_state", 42),
        }

        if x0 is not None:
            kwargs["x0"] = [list(x0)]
            print("  Using warm start from previous optimization")

        result = gp_minimize(**kwargs)

        # Ensure result is valid
        if result is None:
            raise RuntimeError(
                "Bayesian optimization failed - gp_minimize returned None"
            )

        elapsed = time.time() - start
        total_time = time.time() - start_time
        print(f"  ✓ Completed in {elapsed:.1f}s")
        print(f"  Best χ²_red: {result.fun:.6e}")

        best_params = np.array(result.x)
        self.best_params_bo = best_params

        logger.info(
            f"Bayesian optimization completed in {total_time:.2f}s, best χ²_red = {result.fun:.6e}"
        )

        return {
            "method": "scikit-optimize",
            "result": result,
            "best_params": best_params,
            "best_chi_squared": result.fun,
            "optimization_time": elapsed,
            "n_calls": n_calls,
            "parameter_space": parameter_space,
            "acquisition_function": bo_config.get("acquisition_func", "EI"),
        }

    def _build_search_space(self) -> List[Any]:
        """
        Build search space for Bayesian optimization.

        Returns
        -------
        List[Real]
            List of Real parameter spaces for skopt
        """
        if Real is None:
            raise ImportError("scikit-optimize Real class not available")

        parameter_space = []
        bounds = self.config.get("parameter_space", {}).get("bounds", [])

        for bound in bounds:
            if bound.get("type") == "log-uniform":
                space = Real(
                    bound["min"],
                    bound["max"],
                    prior="log-uniform",
                    name=bound["name"],
                )
            else:
                space = Real(bound["min"], bound["max"], name=bound["name"])
            parameter_space.append(space)

        return parameter_space

    def get_available_acquisition_functions(self) -> List[str]:
        """
        Get list of available acquisition functions.

        Returns
        -------
        List[str]
            List of available acquisition functions
        """
        return [
            "EI",  # Expected Improvement
            "PI",  # Probability of Improvement
            "LCB",  # Lower Confidence Bound
            "EIps",  # Expected Improvement per second
            "PIps",  # Probability of Improvement per second
        ]

    def analyze_convergence(self, result) -> Dict[str, Any]:
        """
        Analyze convergence behavior of Bayesian optimization.

        Parameters
        ----------
        result : skopt.OptimizeResult
            Result from skopt optimization

        Returns
        -------
        Dict[str, Any]
            Convergence analysis including improvement trends
        """
        if not hasattr(result, "func_vals"):
            return {"error": "No function values available for analysis"}

        func_vals = np.array(result.func_vals)

        # Calculate running minimum
        running_min = np.minimum.accumulate(func_vals)

        # Calculate improvement over iterations
        improvements = np.diff(running_min)
        significant_improvements = improvements < -0.01 * running_min[:-1]

        # Estimate convergence
        last_improvement_idx = np.where(significant_improvements)[0]
        if len(last_improvement_idx) > 0:
            convergence_iter = last_improvement_idx[-1] + 1
        else:
            convergence_iter = 0

        return {
            "total_evaluations": len(func_vals),
            "best_value": result.fun,
            "best_iteration": np.argmin(func_vals),
            "convergence_iteration": convergence_iter,
            "improvement_ratio": (
                np.sum(significant_improvements) / len(improvements)
            ),
            "final_improvement_rate": (
                np.mean(improvements[-5:]) if len(improvements) >= 5 else 0
            ),
            "running_minimum": running_min,
            "function_values": func_vals,
        }

    def suggest_acquisition_function(
        self, problem_characteristics: Dict[str, Any]
    ) -> str:
        """
        Suggest appropriate acquisition function based on problem characteristics.

        Parameters
        ----------
        problem_characteristics : Dict[str, Any]
            Problem characteristics (noise level, dimensionality, etc.)

        Returns
        -------
        str
            Recommended acquisition function
        """
        # High-dimensional problems
        if problem_characteristics.get("dimensionality", 7) > 10:
            return "LCB"  # More conservative exploration

        # Noisy objectives
        if problem_characteristics.get("noise_level", "low") == "high":
            return "EI"  # Robust to noise

        # Limited budget
        if problem_characteristics.get("budget", 50) < 30:
            return "PI"  # Focuses on improvement probability

        # Default: balanced exploration/exploitation
        return "EI"

    def create_parameter_importance_analysis(self, result) -> Dict[str, Any]:
        """
        Analyze parameter importance from Bayesian optimization results.

        Parameters
        ----------
        result : skopt.OptimizeResult
            Result from skopt optimization

        Returns
        -------
        Dict[str, Any]
            Parameter importance analysis
        """
        if not hasattr(result, "x_iters") or not hasattr(result, "func_vals"):
            return {"error": "Insufficient data for importance analysis"}

        try:
            # Get parameter samples and function values
            X = np.array(result.x_iters)
            y = np.array(result.func_vals)

            # Calculate correlation between parameters and objective
            correlations = []
            param_names = [dim.name for dim in result.space.dimensions]

            for i in range(X.shape[1]):
                correlation = np.corrcoef(X[:, i], y)[0, 1]
                correlations.append(
                    abs(correlation)
                )  # Use absolute correlation

            # Calculate parameter ranges explored
            param_ranges = []
            for i in range(X.shape[1]):
                param_range = np.max(X[:, i]) - np.min(X[:, i])
                param_ranges.append(param_range)

            # Normalize correlations
            max_correlation = max(correlations) if correlations else 1
            normalized_correlations = [
                c / max_correlation for c in correlations
            ]

            return {
                "parameter_names": param_names,
                "correlations": correlations,
                "normalized_correlations": normalized_correlations,
                "parameter_ranges": param_ranges,
                "most_important": (
                    param_names[np.argmax(correlations)]
                    if correlations
                    else None
                ),
                "least_important": (
                    param_names[np.argmin(correlations)]
                    if correlations
                    else None
                ),
            }

        except Exception as e:
            return {"error": f"Analysis failed: {str(e)}"}

    def optimize_acquisition_strategy(
        self, initial_results: Dict[str, Any], remaining_budget: int
    ) -> Dict[str, Any]:
        """
        Optimize acquisition strategy based on initial results.

        Parameters
        ----------
        initial_results : Dict[str, Any]
            Results from initial optimization phase
        remaining_budget : int
            Remaining evaluation budget

        Returns
        -------
        Dict[str, Any]
            Optimized acquisition strategy recommendations
        """
        convergence_analysis = self.analyze_convergence(
            initial_results.get("result")
        )

        # Check convergence status
        if convergence_analysis.get("improvement_ratio", 0) < 0.1:
            # Low improvement rate - focus on exploitation
            recommended_acq = "LCB"
            strategy = "exploitation"
        elif (
            convergence_analysis.get("best_iteration", 0)
            < remaining_budget * 0.3
        ):
            # Found good solution early - continue exploration
            recommended_acq = "EI"
            strategy = "balanced"
        else:
            # Still improving - maintain current strategy
            recommended_acq = initial_results.get("acquisition_function", "EI")
            strategy = "continuation"

        return {
            "recommended_acquisition": recommended_acq,
            "strategy": strategy,
            "reasoning": (
                f"Based on {convergence_analysis.get('improvement_ratio', 0):.2%} improvement rate"
            ),
            "suggested_n_calls": min(
                remaining_budget, max(10, int(remaining_budget * 0.5))
            ),
        }
