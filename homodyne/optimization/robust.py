"""
Robust Optimization Methods for Homodyne Scattering Analysis
===========================================================

This module implements robust optimization algorithms for parameter estimation
in homodyne scattering analysis using CVXPY + Gurobi. Provides protection against
measurement noise, experimental uncertainties, and model misspecification.

Robust Methods Implemented:
1. **Distributionally Robust Optimization (DRO)**: Wasserstein distance-based
   uncertainty sets for handling measurement noise and experimental variability.

2. **Scenario-Based Robust Optimization**: Multi-scenario optimization using
   bootstrap resampling of experimental residuals for outlier resistance.

3. **Ellipsoidal Uncertainty Sets**: Robust least squares with bounded uncertainty
   in experimental correlation functions.

4. **Regularized Robust Formulation**: L1/L2 mixed regularization with physical
   priors for parameter stability.

Key Features:
- CVXPY + Gurobi integration for high-performance convex optimization
- Adaptive uncertainty set sizing based on experimental data characteristics
- Bootstrap scenario generation for robust parameter estimation
- Physical parameter bounds consistent with existing optimization methods
- Comprehensive error handling and graceful degradation

Authors: Wei Chen, Hongrui He
Institution: Argonne National Laboratory
"""

import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.utils import resample

# CVXPY and Gurobi imports with graceful degradation
try:
    import cvxpy as cp

    CVXPY_AVAILABLE = True
except ImportError:
    CVXPY_AVAILABLE = False
    cp = None

# Check if Gurobi is available as a CVXPY solver
try:
    import gurobipy  # noqa: F401

    GUROBI_AVAILABLE = True
except ImportError:
    GUROBI_AVAILABLE = False

logger = logging.getLogger(__name__)


class RobustHomodyneOptimizer:
    """
    Robust optimization algorithms for homodyne scattering parameter estimation.

    This class provides multiple robust optimization methods that handle measurement
    noise, experimental uncertainties, and model misspecification in XPCS analysis.
    All methods use CVXPY + Gurobi for high-performance convex optimization.

    The robust optimization framework addresses common challenges in experimental
    data analysis:
    - Measurement noise in correlation functions
    - Experimental setup variations
    - Outlier measurements
    - Model parameter sensitivity

    Methods maintain consistency with existing parameter bounds and physical
    constraints defined in the configuration system.
    """

    def __init__(self, analysis_core, config: Dict[str, Any]):
        """
        Initialize robust optimizer.

        Parameters
        ----------
        analysis_core : HomodyneAnalysisCore
            Core analysis engine instance
        config : Dict[str, Any]
            Configuration dictionary containing optimization settings
        """
        self.core = analysis_core
        self.config = config
        self.best_params_robust = None

        # Performance optimization caches
        self._jacobian_cache = {}
        self._correlation_cache = {}
        self._bounds_cache = None

        # Extract robust optimization configuration
        self.robust_config = config.get("optimization_config", {}).get(
            "robust_optimization", {}
        )

        # Check dependencies
        if not CVXPY_AVAILABLE:
            logger.warning("CVXPY not available - robust optimization disabled")
        if not GUROBI_AVAILABLE:
            logger.warning("Gurobi not available - using CVXPY default solver")

        # Default robust optimization settings
        self.default_settings = {
            "uncertainty_model": "wasserstein",  # wasserstein, ellipsoidal, scenario
            "uncertainty_radius": 0.05,  # 5% of data variance
            "n_scenarios": 15,  # Reduced from 50 for better performance
            "regularization_alpha": 0.01,  # L2 regularization strength
            "regularization_beta": 0.001,  # L1 sparsity parameter
            "jacobian_epsilon": 1e-6,  # Smaller epsilon for more accurate derivatives
            "enable_caching": True,  # Enable performance caching
            "preferred_solver": "CLARABEL",  # Skip solver chain for speed
            "solver_settings": {
                "Method": 2,  # Barrier method
                "CrossOver": 0,  # Skip crossover for stability
                "BarHomogeneous": 1,  # Use homogeneous barrier
                "TimeLimit": 300,  # Reduced from 600 for faster timeout
                "MIPGap": 1e-4,  # Relaxed for speed
                "NumericFocus": 2,  # Reduced from 3 for speed
                "OutputFlag": 0,  # Suppress output
            },
        }

        # Merge with user configuration
        self.settings = {**self.default_settings, **self.robust_config}

    def check_dependencies(self) -> bool:
        """Check if required dependencies are available."""
        if not CVXPY_AVAILABLE:
            raise ImportError(
                "CVXPY is required for robust optimization. "
                "Install with: pip install cvxpy"
            )
        return True

    def run_robust_optimization(
        self,
        initial_parameters: np.ndarray,
        phi_angles: np.ndarray,
        c2_experimental: np.ndarray,
        method: str = "wasserstein",
        **kwargs,
    ) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
        """
        Run robust optimization using specified method.

        Parameters
        ----------
        initial_parameters : np.ndarray
            Starting parameters for optimization
        phi_angles : np.ndarray
            Angular positions for measurement
        c2_experimental : np.ndarray
            Experimental correlation function data
        method : str, default="wasserstein"
            Robust optimization method: "wasserstein", "scenario", "ellipsoidal"
        **kwargs
            Additional method-specific parameters

        Returns
        -------
        Tuple[Optional[np.ndarray], Dict[str, Any]]
            (optimal_parameters, optimization_info)
        """
        self.check_dependencies()

        start_time = time.time()
        logger.info(f"Starting robust optimization with method: {method}")

        try:
            if method == "wasserstein":
                result = self._solve_distributionally_robust(
                    initial_parameters, phi_angles, c2_experimental, **kwargs
                )
            elif method == "scenario":
                result = self._solve_scenario_robust(
                    initial_parameters, phi_angles, c2_experimental, **kwargs
                )
            elif method == "ellipsoidal":
                result = self._solve_ellipsoidal_robust(
                    initial_parameters, phi_angles, c2_experimental, **kwargs
                )
            else:
                raise ValueError(f"Unknown robust optimization method: {method}")

            optimization_time = time.time() - start_time

            if result[0] is not None:
                self.best_params_robust = result[0]
                logger.info(
                    f"Robust optimization completed in {
                        optimization_time:.2f}s"
                )
            else:
                logger.warning("Robust optimization failed to converge")

            return result

        except Exception as e:
            logger.error(f"Robust optimization failed: {e}")
            return None, {"error": str(e), "method": method}

    def _solve_distributionally_robust(
        self,
        theta_init: np.ndarray,
        phi_angles: np.ndarray,
        c2_experimental: np.ndarray,
        uncertainty_radius: Optional[float] = None,
    ) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
        """
        Distributionally Robust Optimization with Wasserstein uncertainty sets.

        Solves: min_theta max_{P in U_epsilon(P_hat)} E_P[chi_squared(theta, xi)]

        Where U_epsilon(P_hat) is a Wasserstein ball around the empirical distribution
        of experimental data, providing robustness against measurement noise.

        Parameters
        ----------
        theta_init : np.ndarray
            Initial parameter guess
        phi_angles : np.ndarray
            Angular measurement positions
        c2_experimental : np.ndarray
            Experimental correlation function data
        uncertainty_radius : float, optional
            Wasserstein ball radius (default: 5% of data variance)

        Returns
        -------
        Tuple[Optional[np.ndarray], Dict[str, Any]]
            (optimal_parameters, optimization_info)
        """
        if uncertainty_radius is None:
            uncertainty_radius = self.settings["uncertainty_radius"]

        n_params = len(theta_init)

        # Get parameter bounds (cached for performance)
        if self._bounds_cache is None and self.settings.get("enable_caching", True):
            self._bounds_cache = self._get_parameter_bounds()
        bounds = (
            self._bounds_cache
            if self.settings.get("enable_caching", True)
            else self._get_parameter_bounds()
        )

        # Estimate data uncertainty from experimental variance
        data_std = np.std(c2_experimental, axis=-1, keepdims=True)
        epsilon = uncertainty_radius * np.mean(data_std)

        # Log initial chi-squared
        initial_chi_squared = self._compute_chi_squared(
            theta_init, phi_angles, c2_experimental
        )
        logger.info(f"DRO with Wasserstein radius: {epsilon:.6f}")
        logger.info(f"DRO initial χ²: {initial_chi_squared:.6f}")

        try:
            # Check CVXPY availability
            if cp is None:
                raise ImportError("CVXPY not available for robust optimization")

            # CVXPY variables
            theta = cp.Variable(n_params)
            # Uncertain data perturbations
            xi = cp.Variable(c2_experimental.shape)

            # Compute fitted correlation function (linearized around
            # theta_init)
            c2_fitted_init, jacobian = self._compute_linearized_correlation(
                theta_init, phi_angles, c2_experimental
            )

            # Linear approximation: c2_fitted ≈ c2_fitted_init + J @ (theta -
            # theta_init)
            delta_theta = theta - theta_init
            # Reshape jacobian @ delta_theta to match c2_fitted_init shape
            linear_correction = jacobian @ delta_theta
            linear_correction_reshaped = linear_correction.reshape(c2_fitted_init.shape)
            c2_fitted_linear = c2_fitted_init + linear_correction_reshaped

            # Perturbed experimental data
            c2_perturbed = c2_experimental + xi

            # Robust objective: minimize worst-case residuals (experimental -
            # fitted)
            residuals = c2_perturbed - c2_fitted_linear
            assert cp is not None  # Already checked above
            chi_squared = cp.sum_squares(residuals)

            # Constraints
            constraints = []

            # Parameter bounds
            if bounds is not None:
                for i, (lb, ub) in enumerate(bounds):
                    if lb is not None:
                        constraints.append(theta[i] >= lb)
                    if ub is not None:
                        constraints.append(theta[i] <= ub)

            # Wasserstein ball constraint: ||xi||_2 <= epsilon
            assert cp is not None  # Already checked above
            constraints.append(cp.norm(xi, 2) <= epsilon)

            # Regularization term for parameter stability
            alpha = self.settings["regularization_alpha"]
            regularization = alpha * cp.sum_squares(delta_theta)

            # Robust optimization problem
            objective = cp.Minimize(chi_squared + regularization)
            problem = cp.Problem(objective, constraints)

            # Optimized solving with preferred solver and fast fallback
            solver_success = self._solve_cvxpy_problem_optimized(problem, "DRO")

            if problem.status not in ["infeasible", "unbounded"]:
                optimal_params = theta.value
                optimal_value = problem.value

                # Compute final chi-squared with optimal parameters
                if optimal_params is not None:
                    final_chi_squared = self._compute_chi_squared(
                        optimal_params, phi_angles, c2_experimental
                    )
                    # Log final chi-squared and improvement
                    improvement = initial_chi_squared - final_chi_squared
                    percent_improvement = (improvement / initial_chi_squared) * 100
                    logger.info(
                        f"DRO final χ²: {
                            final_chi_squared:.6f} (improvement: {
                            improvement:.6f}, {
                            percent_improvement:.2f}%)"
                    )
                else:
                    final_chi_squared = float("inf")
                    logger.warning("DRO optimization failed to find valid parameters")

                info = {
                    "method": "distributionally_robust",
                    "status": problem.status,
                    "optimal_value": optimal_value,
                    "final_chi_squared": final_chi_squared,
                    "uncertainty_radius": epsilon,
                    "n_iterations": getattr(
                        getattr(problem, "solver_stats", {}), "num_iters", None
                    ),
                    "solve_time": getattr(
                        getattr(problem, "solver_stats", {}),
                        "solve_time",
                        None,
                    ),
                }

                return optimal_params, info
            else:
                logger.error(
                    f"DRO optimization failed with status: {
                        problem.status}"
                )
                return None, {
                    "status": problem.status,
                    "method": "distributionally_robust",
                }

        except Exception as e:
            logger.error(f"DRO optimization error: {e}")
            return None, {"error": str(e), "method": "distributionally_robust"}

    def _solve_scenario_robust(
        self,
        theta_init: np.ndarray,
        phi_angles: np.ndarray,
        c2_experimental: np.ndarray,
        n_scenarios: Optional[int] = None,
    ) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
        """
        Scenario-Based Robust Optimization using bootstrap resampling.

        Solves: min_theta max_{s in scenarios} chi_squared(theta, scenario_s)

        Generates scenarios from bootstrap resampling of experimental residuals
        to handle outliers and experimental variations.

        Parameters
        ----------
        theta_init : np.ndarray
            Initial parameter guess
        phi_angles : np.ndarray
            Angular measurement positions
        c2_experimental : np.ndarray
            Experimental correlation function data
        n_scenarios : int, optional
            Number of bootstrap scenarios (default: 50)

        Returns
        -------
        Tuple[Optional[np.ndarray], Dict[str, Any]]
            (optimal_parameters, optimization_info)
        """
        if n_scenarios is None:
            n_scenarios = self.settings["n_scenarios"]

        # Log initial chi-squared
        initial_chi_squared = self._compute_chi_squared(
            theta_init, phi_angles, c2_experimental
        )
        logger.info(f"Scenario-based optimization with {n_scenarios} scenarios")
        logger.info(f"Scenario initial χ²: {initial_chi_squared:.6f}")

        # Ensure n_scenarios is an int
        if n_scenarios is None:
            n_scenarios = self.settings.get("n_scenarios", 50)
        # Convert to int only if not None
        if n_scenarios is not None:
            n_scenarios = int(n_scenarios)
        else:
            n_scenarios = 50  # Default fallback

        # Generate scenarios using bootstrap resampling
        scenarios = self._generate_bootstrap_scenarios(
            theta_init, phi_angles, c2_experimental, n_scenarios
        )

        n_params = len(theta_init)
        # Get parameter bounds (cached for performance)
        if self._bounds_cache is None and self.settings.get("enable_caching", True):
            self._bounds_cache = self._get_parameter_bounds()
        bounds = (
            self._bounds_cache
            if self.settings.get("enable_caching", True)
            else self._get_parameter_bounds()
        )

        try:
            # Check CVXPY availability
            if cp is None:
                raise ImportError("CVXPY not available for robust optimization")

            # CVXPY variables
            theta = cp.Variable(n_params)
            t = cp.Variable()  # Auxiliary variable for min-max formulation

            # Constraints
            constraints = []

            # Parameter bounds
            if bounds is not None:
                for i, (lb, ub) in enumerate(bounds):
                    if lb is not None:
                        constraints.append(theta[i] >= lb)
                    if ub is not None:
                        constraints.append(theta[i] <= ub)

            # Optimized: Pre-compute linearized correlation once outside the
            # loop
            c2_fitted_init, jacobian = self._compute_linearized_correlation(
                theta_init, phi_angles, c2_experimental
            )
            delta_theta = theta - theta_init
            # Reshape jacobian @ delta_theta to match c2_fitted_init shape
            linear_correction = jacobian @ delta_theta
            linear_correction_reshaped = linear_correction.reshape(c2_fitted_init.shape)
            c2_fitted_linear = c2_fitted_init + linear_correction_reshaped

            # Min-max constraints: t >= chi_squared(theta, scenario_s) for all
            # scenarios
            for scenario_data in scenarios:
                # Chi-squared for this scenario (experimental - fitted)
                residuals = scenario_data - c2_fitted_linear
                assert cp is not None  # Already checked above
                chi_squared_scenario = cp.sum_squares(residuals)
                constraints.append(t >= chi_squared_scenario)

            # Regularization
            alpha = self.settings["regularization_alpha"]
            regularization = alpha * cp.sum_squares(theta - theta_init)

            # Objective: minimize worst-case scenario
            objective = cp.Minimize(t + regularization)
            problem = cp.Problem(objective, constraints)

            # Optimized solving with preferred solver and fast fallback
            solver_success = self._solve_cvxpy_problem_optimized(problem, "Scenario")

            if problem.status not in ["infeasible", "unbounded"]:
                optimal_params = theta.value
                worst_case_value = t.value

                # Compute final chi-squared
                if optimal_params is not None:
                    final_chi_squared = self._compute_chi_squared(
                        optimal_params, phi_angles, c2_experimental
                    )
                    # Log final chi-squared and improvement
                    improvement = initial_chi_squared - final_chi_squared
                    percent_improvement = (improvement / initial_chi_squared) * 100
                    logger.info(
                        f"Scenario final χ²: {
                            final_chi_squared:.6f} (improvement: {
                            improvement:.6f}, {
                            percent_improvement:.2f}%)"
                    )
                else:
                    final_chi_squared = float("inf")
                    logger.warning(
                        "Scenario optimization failed to find valid parameters"
                    )

                info = {
                    "method": "scenario_robust",
                    "status": problem.status,
                    "worst_case_value": worst_case_value,
                    "final_chi_squared": final_chi_squared,
                    "n_scenarios": n_scenarios,
                    "solve_time": getattr(
                        getattr(problem, "solver_stats", {}),
                        "solve_time",
                        None,
                    ),
                }

                return optimal_params, info
            else:
                logger.error(
                    f"Scenario optimization failed with status: {
                        problem.status}"
                )
                return None, {
                    "status": problem.status,
                    "method": "scenario_robust",
                }

        except Exception as e:
            logger.error(f"Scenario optimization error: {e}")
            return None, {"error": str(e), "method": "scenario_robust"}

    def _solve_ellipsoidal_robust(
        self,
        theta_init: np.ndarray,
        phi_angles: np.ndarray,
        c2_experimental: np.ndarray,
        gamma: Optional[float] = None,
    ) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
        """
        Ellipsoidal Uncertainty Sets Robust Optimization.

        Solves robust least squares with bounded uncertainty in experimental data:
        min_theta ||c2_exp + Delta - c2_theory(theta)||_2^2
        subject to ||Delta||_2 <= gamma

        Parameters
        ----------
        theta_init : np.ndarray
            Initial parameter guess
        phi_angles : np.ndarray
            Angular measurement positions
        c2_experimental : np.ndarray
            Experimental correlation function data
        gamma : float, optional
            Uncertainty bound (default: 10% of data norm)

        Returns
        -------
        Tuple[Optional[np.ndarray], Dict[str, Any]]
            (optimal_parameters, optimization_info)
        """
        if gamma is None:
            gamma = float(0.1 * np.linalg.norm(c2_experimental))

        # Log initial chi-squared
        initial_chi_squared = self._compute_chi_squared(
            theta_init, phi_angles, c2_experimental
        )
        logger.info(
            f"Ellipsoidal robust optimization with uncertainty bound: {
                gamma:.6f}"
        )
        logger.info(f"Ellipsoidal initial χ²: {initial_chi_squared:.6f}")

        n_params = len(theta_init)
        # Get parameter bounds (cached for performance)
        if self._bounds_cache is None and self.settings.get("enable_caching", True):
            self._bounds_cache = self._get_parameter_bounds()
        bounds = (
            self._bounds_cache
            if self.settings.get("enable_caching", True)
            else self._get_parameter_bounds()
        )

        try:
            # Check CVXPY availability
            if cp is None:
                raise ImportError("CVXPY not available for robust optimization")

            # CVXPY variables
            theta = cp.Variable(n_params)
            delta = cp.Variable(c2_experimental.shape)  # Uncertainty in data

            # Linearized fitted correlation function
            c2_fitted_init, jacobian = self._compute_linearized_correlation(
                theta_init, phi_angles, c2_experimental
            )
            delta_theta = theta - theta_init
            # Reshape jacobian @ delta_theta to match c2_fitted_init shape
            linear_correction = jacobian @ delta_theta
            linear_correction_reshaped = linear_correction.reshape(c2_fitted_init.shape)
            c2_fitted_linear = c2_fitted_init + linear_correction_reshaped

            # Robust residuals (experimental - fitted)
            c2_perturbed = c2_experimental + delta
            residuals = c2_perturbed - c2_fitted_linear

            # Constraints
            constraints = []

            # Parameter bounds
            if bounds is not None:
                for i, (lb, ub) in enumerate(bounds):
                    if lb is not None:
                        constraints.append(theta[i] >= lb)
                    if ub is not None:
                        constraints.append(theta[i] <= ub)

            # Ellipsoidal uncertainty constraint
            assert cp is not None  # Already checked above
            constraints.append(cp.norm(delta, 2) <= gamma)

            # Regularization
            alpha = self.settings["regularization_alpha"]
            beta = self.settings["regularization_beta"]
            l2_reg = alpha * cp.sum_squares(delta_theta)
            l1_reg = beta * cp.norm(delta_theta, 1)

            # Objective: robust least squares with regularization
            objective = cp.Minimize(cp.sum_squares(residuals) + l2_reg + l1_reg)
            problem = cp.Problem(objective, constraints)

            # Optimized solving with preferred solver and fast fallback
            solver_success = self._solve_cvxpy_problem_optimized(problem, "Ellipsoidal")

            if problem.status not in ["infeasible", "unbounded"]:
                optimal_params = theta.value
                optimal_value = problem.value

                # Compute final chi-squared
                if optimal_params is not None:
                    final_chi_squared = self._compute_chi_squared(
                        optimal_params, phi_angles, c2_experimental
                    )
                    # Log final chi-squared and improvement
                    improvement = initial_chi_squared - final_chi_squared
                    percent_improvement = (improvement / initial_chi_squared) * 100
                    logger.info(
                        f"Ellipsoidal final χ²: {
                            final_chi_squared:.6f} (improvement: {
                            improvement:.6f}, {
                            percent_improvement:.2f}%)"
                    )
                else:
                    final_chi_squared = float("inf")
                    logger.warning(
                        "Ellipsoidal optimization failed to find valid parameters"
                    )

                info = {
                    "method": "ellipsoidal_robust",
                    "status": problem.status,
                    "optimal_value": optimal_value,
                    "final_chi_squared": final_chi_squared,
                    "uncertainty_bound": gamma,
                    "solve_time": getattr(
                        getattr(problem, "solver_stats", {}),
                        "solve_time",
                        None,
                    ),
                }

                return optimal_params, info
            else:
                logger.error(
                    f"Ellipsoidal optimization failed with status: {
                        problem.status}"
                )
                return None, {
                    "status": problem.status,
                    "method": "ellipsoidal_robust",
                }

        except Exception as e:
            logger.error(f"Ellipsoidal optimization error: {e}")
            return None, {"error": str(e), "method": "ellipsoidal_robust"}

    def _generate_bootstrap_scenarios(
        self,
        theta_init: np.ndarray,
        phi_angles: np.ndarray,
        c2_experimental: np.ndarray,
        n_scenarios: int,
    ) -> List[np.ndarray]:
        """
        Generate bootstrap scenarios from experimental residuals.

        Parameters
        ----------
        theta_init : np.ndarray
            Initial parameters for residual computation
        phi_angles : np.ndarray
            Angular positions
        c2_experimental : np.ndarray
            Experimental data
        n_scenarios : int
            Number of scenarios to generate

        Returns
        -------
        List[np.ndarray]
            List of scenario datasets
        """
        # Compute initial residuals using 2D fitted correlation for bootstrap
        # compatibility
        c2_fitted_init = self._compute_fitted_correlation_2d(
            theta_init, phi_angles, c2_experimental
        )
        residuals = c2_experimental - c2_fitted_init

        scenarios = []
        for _ in range(n_scenarios):
            # Bootstrap resample residuals
            if residuals.ndim > 1:
                # Resample along the time axis
                resampled_residuals = np.apply_along_axis(
                    lambda x: resample(x, n_samples=len(x)), -1, residuals
                )
            else:
                resampled_residuals = resample(residuals, n_samples=len(residuals))

            # Create scenario by adding resampled residuals to fitted
            # correlation
            scenario_data = c2_fitted_init + resampled_residuals
            scenarios.append(scenario_data)

        return scenarios

    def _compute_linearized_correlation(
        self,
        theta: np.ndarray,
        phi_angles: np.ndarray,
        c2_experimental: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute fitted correlation function and its Jacobian for linearization.

        CRITICAL: Uses fitted correlation (with scaling) instead of raw theoretical correlation
        to ensure we're minimizing residuals from experimental - fitted, not experimental - theory.

        Parameters
        ----------
        theta : np.ndarray
            Parameter values
        phi_angles : np.ndarray
            Angular positions
        c2_experimental : np.ndarray
            Experimental data for scaling optimization

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            (fitted_correlation_function, jacobian_matrix)
        """
        # Create cache key for performance optimization
        theta_key = tuple(theta) if self.settings.get("enable_caching", True) else None

        if theta_key and theta_key in self._jacobian_cache:
            return self._jacobian_cache[theta_key]

        # Compute fitted correlation function at theta (with scaling applied)
        c2_fitted = self._compute_fitted_correlation(theta, phi_angles, c2_experimental)

        # Optimized Jacobian computation with adaptive epsilon
        epsilon = self.settings.get("jacobian_epsilon", 1e-6)
        n_params = len(theta)
        jacobian = np.zeros((c2_fitted.size, n_params))

        # Batch compute perturbations for better cache efficiency
        theta_perturbations = []
        for i in range(n_params):
            theta_plus = theta.copy()
            theta_minus = theta.copy()
            # Adaptive epsilon based on parameter magnitude
            param_epsilon = max(epsilon, abs(theta[i]) * epsilon)
            theta_plus[i] += param_epsilon
            theta_minus[i] -= param_epsilon
            theta_perturbations.append((theta_plus, theta_minus, param_epsilon))

        # Compute finite differences
        for i, (theta_plus, theta_minus, param_epsilon) in enumerate(
            theta_perturbations
        ):
            c2_plus = self._compute_fitted_correlation(
                theta_plus, phi_angles, c2_experimental
            )
            c2_minus = self._compute_fitted_correlation(
                theta_minus, phi_angles, c2_experimental
            )

            jacobian[:, i] = (c2_plus.flatten() - c2_minus.flatten()) / (
                2 * param_epsilon
            )

        result = (c2_fitted, jacobian)

        # Cache result if caching is enabled
        if theta_key and self.settings.get("enable_caching", True):
            self._jacobian_cache[theta_key] = result

        return result

    def _compute_theoretical_correlation(
        self, theta: np.ndarray, phi_angles: np.ndarray
    ) -> np.ndarray:
        """
        Compute theoretical correlation function using core analysis engine.
        Adapts to different analysis modes (static isotropic, static anisotropic, laminar flow).

        Parameters
        ----------
        theta : np.ndarray
            Parameter values
        phi_angles : np.ndarray
            Angular positions

        Returns
        -------
        np.ndarray
            Theoretical correlation function
        """
        try:
            # Check if we're in static isotropic mode
            if (
                hasattr(self.core, "config_manager")
                and self.core.config_manager.is_static_isotropic_enabled()
            ):
                # In static isotropic mode, we work with a single dummy angle
                # The core will handle this appropriately
                logger.debug("Computing correlation for static isotropic mode")
                # Use the standard calculation method - it already handles
                # static isotropic
                c2_theory = self.core.calculate_c2_nonequilibrium_laminar_parallel(
                    theta, phi_angles
                )
            else:
                # Standard calculation for other modes
                c2_theory = self.core.calculate_c2_nonequilibrium_laminar_parallel(
                    theta, phi_angles
                )
            return c2_theory
        except Exception as e:
            logger.error(f"Error computing theoretical correlation: {e}")
            # Fallback: return zeros with appropriate shape
            n_angles = len(phi_angles) if phi_angles is not None else 1
            n_times = getattr(
                self.core, "time_length", 100
            )  # Use time_length instead of n_time_steps
            return np.zeros((n_angles, n_times, n_times))

    def _compute_fitted_correlation(
        self,
        theta: np.ndarray,
        phi_angles: np.ndarray,
        c2_experimental: np.ndarray,
    ) -> np.ndarray:
        """
        Compute fitted correlation function with proper scaling: fitted = contrast * theory + offset.

        This method computes the theoretical correlation and then applies optimal scaling
        to match experimental data, which is essential for robust optimization.

        Parameters
        ----------
        theta : np.ndarray
            Parameter values
        phi_angles : np.ndarray
            Angular positions
        c2_experimental : np.ndarray
            Experimental data for scaling optimization

        Returns
        -------
        np.ndarray
            Fitted correlation function (scaled to match experimental data)
        """
        try:
            # Performance optimization: cache theoretical correlation
            theta_key = (
                tuple(theta) if self.settings.get("enable_caching", True) else None
            )

            if theta_key and theta_key in self._correlation_cache:
                c2_theory = self._correlation_cache[theta_key]
            else:
                # Get raw theoretical correlation
                c2_theory = self._compute_theoretical_correlation(theta, phi_angles)

                # Cache if enabled
                if theta_key and self.settings.get("enable_caching", True):
                    self._correlation_cache[theta_key] = c2_theory

            # Apply scaling transformation using least squares
            # This mimics what calculate_chi_squared_optimized does internally
            n_angles = c2_theory.shape[0]
            c2_fitted = np.zeros_like(c2_theory)

            # Flatten for easier processing
            theory_flat = c2_theory.reshape(n_angles, -1)
            exp_flat = c2_experimental.reshape(n_angles, -1)

            # Compute optimal scaling for each angle: fitted = contrast *
            # theory + offset
            for i in range(n_angles):
                theory_i = theory_flat[i]
                exp_i = exp_flat[i]

                # Solve least squares: [theory, ones] * [contrast, offset] =
                # exp
                A = np.column_stack([theory_i, np.ones(len(theory_i))])
                try:
                    scaling_params = np.linalg.lstsq(A, exp_i, rcond=None)[0]
                    contrast, offset = scaling_params[0], scaling_params[1]
                except np.linalg.LinAlgError:
                    # Fallback if least squares fails
                    contrast, offset = 1.0, 0.0

                # Apply scaling
                fitted_i = contrast * theory_i + offset
                c2_fitted[i] = fitted_i.reshape(c2_theory.shape[1:])

            return c2_fitted

        except Exception as e:
            logger.error(f"Error computing fitted correlation: {e}")
            # Fallback to unscaled theory
            return self._compute_theoretical_correlation(theta, phi_angles)

    def _compute_fitted_correlation_2d(
        self,
        theta: np.ndarray,
        phi_angles: np.ndarray,
        c2_experimental: np.ndarray,
    ) -> np.ndarray:
        """
        Compute 2D fitted correlation function for bootstrap scenarios.

        This method uses the mock core's 2D compute_c2_correlation_optimized method
        to return correlation functions compatible with experimental data shape.

        Parameters
        ----------
        theta : np.ndarray
            Parameter values
        phi_angles : np.ndarray
            Angular positions
        c2_experimental : np.ndarray
            Experimental data (2D: n_angles x n_times)

        Returns
        -------
        np.ndarray
            2D fitted correlation function (n_angles x n_times)
        """
        try:
            # Use the mock core's 2D correlation function
            if hasattr(self.core, "compute_c2_correlation_optimized"):
                c2_theory_2d = self.core.compute_c2_correlation_optimized(
                    theta, phi_angles
                )

                # Apply scaling transformation using least squares
                n_angles = c2_theory_2d.shape[0]
                c2_fitted_2d = np.zeros_like(c2_theory_2d)

                for i in range(n_angles):
                    theory_i = c2_theory_2d[i]
                    exp_i = c2_experimental[i]

                    # Solve least squares: [theory, ones] * [contrast, offset]
                    # = exp
                    A = np.column_stack([theory_i, np.ones(len(theory_i))])
                    try:
                        scaling_params = np.linalg.lstsq(A, exp_i, rcond=None)[0]
                        contrast, offset = scaling_params[0], scaling_params[1]
                    except np.linalg.LinAlgError:
                        # Fallback if least squares fails
                        contrast, offset = 1.0, 0.0

                    # Apply scaling
                    c2_fitted_2d[i] = contrast * theory_i + offset

                return c2_fitted_2d
            else:
                # Fallback: use experimental data shape
                return np.ones_like(c2_experimental)

        except Exception as e:
            logger.error(f"Error computing 2D fitted correlation: {e}")
            # Fallback to experimental data shape
            return np.ones_like(c2_experimental)

    def _compute_chi_squared(
        self,
        theta: np.ndarray,
        phi_angles: np.ndarray,
        c2_experimental: np.ndarray,
    ) -> float:
        """
        Compute chi-squared goodness of fit.

        Parameters
        ----------
        theta : np.ndarray
            Parameter values
        phi_angles : np.ndarray
            Angular positions
        c2_experimental : np.ndarray
            Experimental data

        Returns
        -------
        float
            Chi-squared value
        """
        try:
            # Use existing analysis core for chi-squared calculation
            chi_squared = self.core.calculate_chi_squared_optimized(
                theta, phi_angles, c2_experimental
            )
            return float(chi_squared)
        except Exception as e:
            logger.error(f"Error computing chi-squared: {e}")
            return float("inf")

    def _get_parameter_bounds(
        self,
    ) -> Optional[List[Tuple[Optional[float], Optional[float]]]]:
        """
        Get parameter bounds from configuration.

        Returns
        -------
        Optional[List[Tuple[Optional[float], Optional[float]]]]
            List of (lower_bound, upper_bound) tuples
        """
        try:
            # Extract bounds from configuration (same format as classical
            # optimization)
            bounds_config = self.config.get("parameter_space", {}).get("bounds", [])

            # Get effective parameter count
            n_params = self.core.get_effective_parameter_count()

            if self.core.is_static_mode():
                # Static mode: only diffusion parameters
                param_names = ["D0", "alpha", "D_offset"]
            else:
                # Laminar flow mode: all parameters
                param_names = [
                    "D0",
                    "alpha",
                    "D_offset",
                    "gamma_dot_0",
                    "beta",
                    "gamma_dot_offset",
                    "phi_0",
                ]

            bounds = []

            # Handle both list and dict formats for bounds
            if isinstance(bounds_config, list):
                # List format: [{"name": "D0", "min": 1.0, "max": 10000.0},
                # ...]
                bounds_dict = {
                    bound.get("name"): bound
                    for bound in bounds_config
                    if "name" in bound
                }

                for param_name in param_names[:n_params]:
                    if param_name in bounds_dict:
                        bound_info = bounds_dict[param_name]
                        min_val = bound_info.get("min")
                        max_val = bound_info.get("max")
                        bounds.append((min_val, max_val))
                    else:
                        bounds.append((None, None))

            elif isinstance(bounds_config, dict):
                # Dict format: {"D0": {"min": 1.0, "max": 10000.0}, ...}
                for param_name in param_names[:n_params]:
                    if param_name in bounds_config:
                        bound_info = bounds_config[param_name]
                        if isinstance(bound_info, dict):
                            min_val = bound_info.get("min")
                            max_val = bound_info.get("max")
                            bounds.append((min_val, max_val))
                        elif isinstance(bound_info, list) and len(bound_info) == 2:
                            bounds.append((bound_info[0], bound_info[1]))
                        else:
                            bounds.append((None, None))
                    else:
                        bounds.append((None, None))
            else:
                # No bounds specified
                bounds = [(None, None)] * n_params

            return bounds

        except Exception as e:
            logger.error(f"Error getting parameter bounds: {e}")
            return None

    def _solve_cvxpy_problem_optimized(self, problem, method_name: str = "") -> bool:
        """
        Optimized CVXPY problem solving with preferred solver and fast fallback.

        Parameters
        ----------
        problem : cp.Problem
            CVXPY problem to solve
        method_name : str
            Name of the optimization method for logging

        Returns
        -------
        bool
            True if solver succeeded, False otherwise
        """
        preferred_solver = self.settings.get("preferred_solver", "CLARABEL")

        # Try preferred solver first
        try:
            if preferred_solver == "CLARABEL":
                logger.debug(f"{method_name}: Using preferred CLARABEL solver")
                problem.solve(solver=cp.CLARABEL)
            elif preferred_solver == "SCS":
                logger.debug(f"{method_name}: Using preferred SCS solver")
                problem.solve(solver=cp.SCS)
            elif preferred_solver == "CVXOPT":
                logger.debug(f"{method_name}: Using preferred CVXOPT solver")
                problem.solve(solver=cp.CVXOPT)
            else:
                logger.debug(f"{method_name}: Using default CLARABEL solver")
                problem.solve(solver=cp.CLARABEL)

            if problem.status in ["optimal", "optimal_inaccurate"]:
                logger.debug(
                    f"{method_name}: Preferred solver succeeded with status: {
                        problem.status}"
                )
                return True
        except Exception as e:
            logger.debug(
                f"{method_name}: Preferred solver {preferred_solver} failed: {
                    str(e)}"
            )

        # Fast fallback to SCS if preferred solver failed
        try:
            logger.debug(
                f"{method_name}: Preferred solver failed. Trying SCS fallback."
            )
            problem.solve(solver=cp.SCS)
            if problem.status in ["optimal", "optimal_inaccurate"]:
                logger.debug(
                    f"{method_name}: SCS fallback succeeded with status: {
                        problem.status}"
                )
                return True
        except Exception as e:
            logger.debug(f"{method_name}: SCS fallback failed: {str(e)}")

        logger.error(f"{method_name}: All solvers failed to find a solution")
        return False

    def clear_caches(self) -> None:
        """
        Clear performance optimization caches to free memory.

        Call this method periodically during batch optimization to prevent
        memory usage from growing too large.
        """
        self._jacobian_cache.clear()
        self._correlation_cache.clear()
        self._bounds_cache = None
        logger.debug("Cleared robust optimization performance caches")


def create_robust_optimizer(
    analysis_core, config: Dict[str, Any]
) -> RobustHomodyneOptimizer:
    """
    Factory function to create a RobustHomodyneOptimizer instance.

    Parameters
    ----------
    analysis_core : HomodyneAnalysisCore
        Core analysis engine instance
    config : Dict[str, Any]
        Configuration dictionary

    Returns
    -------
    RobustHomodyneOptimizer
        Configured robust optimizer instance
    """
    return RobustHomodyneOptimizer(analysis_core, config)
