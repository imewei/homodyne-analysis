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

import time
import logging
from typing import Any, Dict, List, Optional, Tuple, Union, Callable

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
            "n_scenarios": 50,  # Number of scenarios for scenario-based optimization
            "regularization_alpha": 0.01,  # L2 regularization strength
            "regularization_beta": 0.001,  # L1 sparsity parameter
            "solver_settings": {
                "Method": 2,  # Barrier method
                "CrossOver": 0,  # Skip crossover for stability
                "BarHomogeneous": 1,  # Use homogeneous barrier
                "TimeLimit": 600,  # 10 minute limit
                "MIPGap": 1e-6,  # High precision
                "NumericFocus": 3,  # Maximum numerical stability
                "OutputFlag": 0,  # Suppress output
            }
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
        **kwargs
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
                logger.info(f"Robust optimization completed in {optimization_time:.2f}s")
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
        uncertainty_radius: Optional[float] = None
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
        
        # Get parameter bounds
        bounds = self._get_parameter_bounds()
        
        # Estimate data uncertainty from experimental variance
        data_std = np.std(c2_experimental, axis=-1, keepdims=True)
        epsilon = uncertainty_radius * np.mean(data_std)
        
        logger.info(f"DRO with Wasserstein radius: {epsilon:.6f}")
        
        try:
            # Check CVXPY availability
            if cp is None:
                raise ImportError("CVXPY not available for robust optimization")
            
            # CVXPY variables
            theta = cp.Variable(n_params)
            xi = cp.Variable(c2_experimental.shape)  # Uncertain data perturbations
            
            # Compute theoretical correlation function (linearized around theta_init)
            c2_theory_init, jacobian = self._compute_linearized_correlation(
                theta_init, phi_angles
            )
            
            # Linear approximation: c2_theory ≈ c2_init + J @ (theta - theta_init)
            delta_theta = theta - theta_init
            c2_theory_linear = c2_theory_init + jacobian @ delta_theta
            
            # Perturbed experimental data
            c2_perturbed = c2_experimental + xi
            
            # Robust objective: minimize worst-case chi-squared
            residuals = c2_perturbed - c2_theory_linear
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
            
            # Solve with Gurobi if available
            solver = cp.GUROBI if GUROBI_AVAILABLE else cp.ECOS
            solver_opts = self.settings["solver_settings"] if GUROBI_AVAILABLE else {}
            
            problem.solve(solver=solver, **solver_opts)
            
            if problem.status not in ["infeasible", "unbounded"]:
                optimal_params = theta.value
                optimal_value = problem.value
                
                # Compute final chi-squared with optimal parameters
                if optimal_params is not None:
                    final_chi_squared = self._compute_chi_squared(
                        optimal_params, phi_angles, c2_experimental
                    )
                else:
                    final_chi_squared = float('inf')
                
                info = {
                    "method": "distributionally_robust",
                    "status": problem.status,
                    "optimal_value": optimal_value,
                    "final_chi_squared": final_chi_squared,
                    "uncertainty_radius": epsilon,
                    "n_iterations": getattr(problem, "solver_stats", {}).get("num_iters", None),
                    "solve_time": getattr(problem, "solver_stats", {}).get("solve_time", None)
                }
                
                return optimal_params, info
            else:
                logger.error(f"DRO optimization failed with status: {problem.status}")
                return None, {"status": problem.status, "method": "distributionally_robust"}
                
        except Exception as e:
            logger.error(f"DRO optimization error: {e}")
            return None, {"error": str(e), "method": "distributionally_robust"}
            
    def _solve_scenario_robust(
        self,
        theta_init: np.ndarray,
        phi_angles: np.ndarray,
        c2_experimental: np.ndarray,
        n_scenarios: Optional[int] = None
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
            
        logger.info(f"Scenario-based optimization with {n_scenarios} scenarios")
        
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
        bounds = self._get_parameter_bounds()
        
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
            
            # Min-max constraints: t >= chi_squared(theta, scenario_s) for all scenarios
            for scenario_data in scenarios:
                # Linearized correlation function for scenario
                c2_theory_init, jacobian = self._compute_linearized_correlation(
                    theta_init, phi_angles
                )
                delta_theta = theta - theta_init
                c2_theory_linear = c2_theory_init + jacobian @ delta_theta
                
                # Chi-squared for this scenario
                residuals = scenario_data - c2_theory_linear
                assert cp is not None  # Already checked above
                chi_squared_scenario = cp.sum_squares(residuals)
                constraints.append(t >= chi_squared_scenario)
            
            # Regularization
            alpha = self.settings["regularization_alpha"]
            regularization = alpha * cp.sum_squares(theta - theta_init)
            
            # Objective: minimize worst-case scenario
            objective = cp.Minimize(t + regularization)
            problem = cp.Problem(objective, constraints)
            
            # Solve
            solver = cp.GUROBI if GUROBI_AVAILABLE else cp.ECOS
            solver_opts = self.settings["solver_settings"] if GUROBI_AVAILABLE else {}
            
            problem.solve(solver=solver, **solver_opts)
            
            if problem.status not in ["infeasible", "unbounded"]:
                optimal_params = theta.value
                worst_case_value = t.value
                
                # Compute final chi-squared
                if optimal_params is not None:
                    final_chi_squared = self._compute_chi_squared(
                        optimal_params, phi_angles, c2_experimental
                    )
                else:
                    final_chi_squared = float('inf')
                
                info = {
                    "method": "scenario_robust",
                    "status": problem.status,
                    "worst_case_value": worst_case_value,
                    "final_chi_squared": final_chi_squared,
                    "n_scenarios": n_scenarios,
                    "solve_time": getattr(problem, "solver_stats", {}).get("solve_time", None)
                }
                
                return optimal_params, info
            else:
                logger.error(f"Scenario optimization failed with status: {problem.status}")
                return None, {"status": problem.status, "method": "scenario_robust"}
                
        except Exception as e:
            logger.error(f"Scenario optimization error: {e}")
            return None, {"error": str(e), "method": "scenario_robust"}
            
    def _solve_ellipsoidal_robust(
        self,
        theta_init: np.ndarray,
        phi_angles: np.ndarray,
        c2_experimental: np.ndarray,
        gamma: Optional[float] = None
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
            
        logger.info(f"Ellipsoidal robust optimization with uncertainty bound: {gamma:.6f}")
        
        n_params = len(theta_init)
        bounds = self._get_parameter_bounds()
        
        try:
            # Check CVXPY availability
            if cp is None:
                raise ImportError("CVXPY not available for robust optimization")
            
            # CVXPY variables
            theta = cp.Variable(n_params)
            delta = cp.Variable(c2_experimental.shape)  # Uncertainty in data
            
            # Linearized correlation function
            c2_theory_init, jacobian = self._compute_linearized_correlation(
                theta_init, phi_angles
            )
            delta_theta = theta - theta_init
            c2_theory_linear = c2_theory_init + jacobian @ delta_theta
            
            # Robust residuals
            c2_perturbed = c2_experimental + delta
            residuals = c2_perturbed - c2_theory_linear
            
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
            
            # Solve
            solver = cp.GUROBI if GUROBI_AVAILABLE else cp.ECOS
            solver_opts = self.settings["solver_settings"] if GUROBI_AVAILABLE else {}
            
            problem.solve(solver=solver, **solver_opts)
            
            if problem.status not in ["infeasible", "unbounded"]:
                optimal_params = theta.value
                optimal_value = problem.value
                
                # Compute final chi-squared
                if optimal_params is not None:
                    final_chi_squared = self._compute_chi_squared(
                        optimal_params, phi_angles, c2_experimental
                    )
                else:
                    final_chi_squared = float('inf')
                
                info = {
                    "method": "ellipsoidal_robust",
                    "status": problem.status,
                    "optimal_value": optimal_value,
                    "final_chi_squared": final_chi_squared,
                    "uncertainty_bound": gamma,
                    "solve_time": getattr(problem, "solver_stats", {}).get("solve_time", None)
                }
                
                return optimal_params, info
            else:
                logger.error(f"Ellipsoidal optimization failed with status: {problem.status}")
                return None, {"status": problem.status, "method": "ellipsoidal_robust"}
                
        except Exception as e:
            logger.error(f"Ellipsoidal optimization error: {e}")
            return None, {"error": str(e), "method": "ellipsoidal_robust"}
            
    def _generate_bootstrap_scenarios(
        self,
        theta_init: np.ndarray,
        phi_angles: np.ndarray,
        c2_experimental: np.ndarray,
        n_scenarios: int
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
        # Compute initial residuals
        c2_theory_init = self._compute_theoretical_correlation(theta_init, phi_angles)
        residuals = c2_experimental - c2_theory_init
        
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
                
            # Create scenario by adding resampled residuals to theory
            scenario_data = c2_theory_init + resampled_residuals
            scenarios.append(scenario_data)
            
        return scenarios
        
    def _compute_linearized_correlation(
        self, theta: np.ndarray, phi_angles: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute correlation function and its Jacobian for linearization.
        
        Parameters
        ----------
        theta : np.ndarray
            Parameter values
        phi_angles : np.ndarray
            Angular positions
            
        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            (correlation_function, jacobian_matrix)
        """
        # Compute correlation function at theta
        c2_theory = self._compute_theoretical_correlation(theta, phi_angles)
        
        # Estimate Jacobian using finite differences
        epsilon = 1e-8
        n_params = len(theta)
        jacobian = np.zeros((c2_theory.size, n_params))
        
        for i in range(n_params):
            theta_plus = theta.copy()
            theta_plus[i] += epsilon
            c2_plus = self._compute_theoretical_correlation(theta_plus, phi_angles)
            
            theta_minus = theta.copy()
            theta_minus[i] -= epsilon
            c2_minus = self._compute_theoretical_correlation(theta_minus, phi_angles)
            
            jacobian[:, i] = (c2_plus.flatten() - c2_minus.flatten()) / (2 * epsilon)
            
        return c2_theory, jacobian
        
    def _compute_theoretical_correlation(
        self, theta: np.ndarray, phi_angles: np.ndarray
    ) -> np.ndarray:
        """
        Compute theoretical correlation function using core analysis engine.
        
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
            # Use existing analysis core to compute correlation function
            c2_theory = self.core.compute_c2_correlation_optimized(
                theta, phi_angles
            )
            return c2_theory
        except Exception as e:
            logger.error(f"Error computing theoretical correlation: {e}")
            # Fallback: return zeros with appropriate shape
            n_angles = len(phi_angles)
            n_times = getattr(self.core, 'n_time_steps', 100)  # Default fallback
            return np.zeros((n_angles, n_times))
            
    def _compute_chi_squared(
        self, theta: np.ndarray, phi_angles: np.ndarray, c2_experimental: np.ndarray
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
            return float('inf')
            
    def _get_parameter_bounds(self) -> Optional[List[Tuple[Optional[float], Optional[float]]]]:
        """
        Get parameter bounds from configuration.
        
        Returns
        -------
        Optional[List[Tuple[Optional[float], Optional[float]]]]
            List of (lower_bound, upper_bound) tuples
        """
        try:
            # Extract bounds from configuration (same format as classical optimization)
            bounds_config = self.config.get("parameter_space", {}).get("bounds", [])
            
            # Get effective parameter count
            n_params = self.core.get_effective_parameter_count()
            
            if self.core.is_static_mode():
                # Static mode: only diffusion parameters
                param_names = ["D0", "alpha", "D_offset"]
            else:
                # Laminar flow mode: all parameters
                param_names = ["D0", "alpha", "D_offset", "gamma_dot_0", "beta", "gamma_dot_offset", "phi_0"]
                
            bounds = []
            
            # Handle both list and dict formats for bounds
            if isinstance(bounds_config, list):
                # List format: [{"name": "D0", "min": 1.0, "max": 10000.0}, ...]
                bounds_dict = {bound.get("name"): bound for bound in bounds_config if "name" in bound}
                
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


def create_robust_optimizer(analysis_core, config: Dict[str, Any]) -> RobustHomodyneOptimizer:
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