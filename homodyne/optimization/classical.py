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
Institution: Argonne National Laboratory & University of Chicago
"""

import time
import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import scipy.optimize as optimize

try:
    import gurobipy as gp
    from gurobipy import GRB
    GUROBI_AVAILABLE = True
except ImportError:
    GUROBI_AVAILABLE = False
    # Type stubs for when Gurobi is not available
    gp = None  # type: ignore
    GRB = None  # type: ignore

logger = logging.getLogger(__name__)

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
        assert (
            phi_angles is not None and c2_experimental is not None
        ), "Failed to load experimental data"

        best_result = None
        best_params = None
        best_chi2 = np.inf
        best_method = None  # Track which method produced the best result
        all_results = []  # Store all results for analysis

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
                    bounds=None,  # Nelder-Mead doesn't use bounds
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
                        best_method = method  # Track which method produced this result
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

            # best_method is already tracked when the best result was found
            if best_method is None:
                best_method = 'Unknown'
            
            # Generate comprehensive summary (for future use)
            summary = self.get_optimization_summary(best_params, best_result, total_time, best_method)
            summary['optimization_method'] = best_method
            summary['all_methods_tried'] = [method for method, _ in all_results]
            
            # Create method-specific results dictionary
            method_results = {}
            for method, result in all_results:
                if hasattr(result, 'fun'):  # Successful result
                    method_results[method] = {
                        'parameters': result.x.tolist() if hasattr(result, 'x') else None,
                        'chi_squared': result.fun,
                        'success': result.success if hasattr(result, 'success') else True,
                        'iterations': getattr(result, 'nit', None),
                        'function_evaluations': getattr(result, 'nfev', None),
                        'message': getattr(result, 'message', ''),
                        'method': getattr(result, 'method', method)
                    }
                else:  # Failed result (exception)
                    method_results[method] = {
                        'parameters': None,
                        'chi_squared': float('inf'),
                        'success': False,
                        'error': str(result)
                    }

            # Log results
            logger.info(
                f"Classical optimization completed in {total_time:.2f}s, best χ²_red = {best_chi2:.6e} (method: {best_method})"
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
            enhanced_result.method_results = method_results  # Add method-specific results
            enhanced_result.best_method = best_method       # Add best method info

            return best_params, enhanced_result
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
            List containing available optimization methods
        """
        methods = ["Nelder-Mead"]  # Nelder-Mead simplex algorithm
        if GUROBI_AVAILABLE:
            methods.append("Gurobi")  # Gurobi quadratic programming solver
        return methods

    def validate_method_compatibility(self, method: str, has_bounds: bool = True) -> bool:
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

    def get_method_recommendations(self) -> Dict[str, List[str]]:
        """
        Get method recommendations based on problem characteristics.

        Returns
        -------
        Dict[str, List[str]]
            Dictionary mapping scenarios to recommended methods
        """
        recommendations = {
            "with_bounds": ["Gurobi", "Nelder-Mead"] if GUROBI_AVAILABLE else ["Nelder-Mead"],
            "without_bounds": ["Nelder-Mead"],
            "high_dimensional": ["Nelder-Mead"],
            "low_dimensional": ["Gurobi", "Nelder-Mead"] if GUROBI_AVAILABLE else ["Nelder-Mead"],
            "noisy_objective": ["Nelder-Mead"],  # Excellent for noisy functions
            "smooth_objective": ["Gurobi", "Nelder-Mead"] if GUROBI_AVAILABLE else ["Nelder-Mead"],
        }
        return recommendations

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
            if method == "Gurobi":
                return self._run_gurobi_optimization(
                    objective_func, initial_parameters, bounds, method_options
                )
            else:
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

                # Nelder-Mead doesn't use explicit bounds
                # The method handles constraints through the objective function

                result = optimize.minimize(**kwargs)
                return True, result

        except Exception as e:
            return False, e

    def _run_gurobi_optimization(
        self,
        objective_func,
        initial_parameters: np.ndarray,
        bounds: Optional[List[Tuple[float, float]]] = None,
        method_options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[bool, Union[optimize.OptimizeResult, Exception]]:
        """
        Run Gurobi optimization for least squares fitting.
        
        This method approximates the chi-squared objective function using a quadratic model
        and solves it with Gurobi's quadratic programming solver. It uses the same parameter
        bounds as MCMC for consistency between optimization methods.
        
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
            
        Notes
        -----
        For consistency with MCMC, this method uses the same parameter bounds defined
        in the configuration's parameter_space.bounds section. This ensures that both
        classical and Bayesian optimization methods respect the same physical constraints.
        """
        try:
            if not GUROBI_AVAILABLE or gp is None or GRB is None:
                raise ImportError("Gurobi is not available. Please install gurobipy.")
                
            # Get parameter bounds
            if bounds is None:
                bounds = self.get_parameter_bounds()
                
            n_params = len(initial_parameters)
            
            # Default Gurobi options
            gurobi_options = {
                "max_iterations": 1000,
                "tolerance": 1e-6,
                "output_flag": 0,  # Suppress output by default
                "method": 2,  # Use barrier method for QP
                "time_limit": 300,  # 5 minute time limit
            }
            
            # Update with user options
            if method_options:
                filtered_options = {
                    k: v
                    for k, v in method_options.items()
                    if not (k.startswith("_") and k.endswith("_note"))
                }
                gurobi_options.update(filtered_options)
            
            # Build quadratic approximation around initial point
            x0 = initial_parameters.copy()
            f0 = objective_func(x0)
            
            # Estimate gradient using finite differences
            epsilon = 1e-8
            grad = np.zeros(n_params)
            for i in range(n_params):
                x_plus = x0.copy()
                x_plus[i] += epsilon
                x_minus = x0.copy()
                x_minus[i] -= epsilon
                grad[i] = (objective_func(x_plus) - objective_func(x_minus)) / (2 * epsilon)
            
            # Estimate Hessian using finite differences
            hessian = np.zeros((n_params, n_params))
            for i in range(n_params):
                for j in range(i, n_params):
                    if i == j:
                        # Diagonal elements
                        x_plus = x0.copy()
                        x_plus[i] += epsilon
                        x_minus = x0.copy()
                        x_minus[i] -= epsilon
                        hessian[i, j] = (objective_func(x_plus) - 2*f0 + objective_func(x_minus)) / (epsilon**2)
                    else:
                        # Off-diagonal elements
                        x_pp = x0.copy()
                        x_pp[i] += epsilon
                        x_pp[j] += epsilon
                        x_pm = x0.copy()
                        x_pm[i] += epsilon
                        x_pm[j] -= epsilon
                        x_mp = x0.copy()
                        x_mp[i] -= epsilon
                        x_mp[j] += epsilon
                        x_mm = x0.copy()
                        x_mm[i] -= epsilon
                        x_mm[j] -= epsilon
                        hessian[i, j] = (objective_func(x_pp) - objective_func(x_pm) - 
                                       objective_func(x_mp) + objective_func(x_mm)) / (4 * epsilon**2)
                        hessian[j, i] = hessian[i, j]  # Symmetric
            
            # Ensure Hessian is positive definite for QP
            eigenvals = np.linalg.eigvals(hessian)
            if np.min(eigenvals) <= 0:
                # Add regularization to make positive definite
                regularization = abs(np.min(eigenvals)) + 1e-6
                hessian += regularization * np.eye(n_params)
            
            # Create Gurobi model
            with gp.Env(empty=True) as env:
                if gurobi_options.get("output_flag", 0) == 0:
                    env.setParam('OutputFlag', 0)
                env.start()
                
                with gp.Model(env=env) as model:
                    # Set Gurobi parameters
                    for param, value in gurobi_options.items():
                        if param == "max_iterations":
                            model.setParam(GRB.Param.IterationLimit, int(value))
                        elif param == "tolerance":
                            model.setParam(GRB.Param.OptimalityTol, float(value))
                        elif param == "output_flag":
                            model.setParam(GRB.Param.OutputFlag, int(value))
                        elif param == "method":
                            model.setParam(GRB.Param.Method, int(value))
                        elif param == "time_limit":
                            model.setParam(GRB.Param.TimeLimit, float(value))
                    
                    # Create decision variables
                    x = model.addVars(n_params, lb=-gp.GRB.INFINITY, ub=gp.GRB.INFINITY, name="x")
                    
                    # Add bounds constraints
                    for i in range(n_params):
                        if i < len(bounds):
                            lb, ub = bounds[i]
                            if lb != -np.inf:
                                x[i].setAttr(GRB.Attr.LB, lb)
                            if ub != np.inf:
                                x[i].setAttr(GRB.Attr.UB, ub)
                    
                    # Build quadratic objective: f0 + grad^T(x-x0) + 0.5*(x-x0)^T*H*(x-x0)
                    obj = f0
                    
                    # Linear terms
                    for i in range(n_params):
                        obj += grad[i] * (x[i] - x0[i])
                    
                    # Quadratic terms
                    for i in range(n_params):
                        for j in range(n_params):
                            obj += 0.5 * hessian[i, j] * (x[i] - x0[i]) * (x[j] - x0[j])
                    
                    model.setObjective(obj, GRB.MINIMIZE)
                    
                    # Optimize
                    model.optimize()
                    
                    # Extract results
                    if model.status == GRB.OPTIMAL:
                        optimal_params = np.array([x[i].x for i in range(n_params)])  # type: ignore
                        optimal_value = objective_func(optimal_params)  # Evaluate true objective
                        
                        # Create OptimizeResult-like object
                        result = optimize.OptimizeResult(
                            x=optimal_params,
                            fun=optimal_value,
                            success=True,
                            status=0,
                            message="Optimization terminated successfully.",
                            nit=model.getAttr(GRB.Attr.IterCount),
                            nfev=model.getAttr(GRB.Attr.IterCount) + n_params * (n_params + 3) // 2 + 1,  # Approximate function evaluations
                            method="Gurobi-QP"
                        )
                        
                        return True, result
                    else:
                        status_messages = {
                            GRB.INFEASIBLE: "Model is infeasible",
                            GRB.INF_OR_UNBD: "Model is infeasible or unbounded", 
                            GRB.UNBOUNDED: "Model is unbounded",
                            GRB.CUTOFF: "Objective cutoff reached",
                            GRB.ITERATION_LIMIT: "Iteration limit reached",
                            GRB.NODE_LIMIT: "Node limit reached",
                            GRB.TIME_LIMIT: "Time limit reached",
                            GRB.SOLUTION_LIMIT: "Solution limit reached",
                            GRB.INTERRUPTED: "Optimization interrupted",
                            GRB.NUMERIC: "Numerical issues encountered",
                            GRB.SUBOPTIMAL: "Suboptimal solution found",
                        }
                        message = status_messages.get(model.status, f"Unknown status: {model.status}")
                        
                        result = optimize.OptimizeResult(
                            x=initial_parameters,
                            fun=f0,
                            success=False,
                            status=model.status,
                            message=message,
                            nit=model.getAttr(GRB.Attr.IterCount) if hasattr(model, 'IterCount') else 0,
                            nfev=n_params * (n_params + 3) // 2 + 1,
                            method="Gurobi-QP"
                        )
                        
                        return False, result
                        
        except Exception as e:
            return False, e

    def analyze_optimization_results(
        self,
        results: List[Tuple[str, bool, Union[optimize.OptimizeResult, Exception]]],
    ) -> Dict[str, Any]:
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
        effective_param_count: Optional[int] = None,
        is_static_mode: Optional[bool] = None,
    ) -> List[Tuple[float, float]]:
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
        # Note: is_static_mode parameter is unused but kept for API compatibility  
        _ = is_static_mode  # Explicitly mark as unused for type checker
        
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

        return bounds[:effective_param_count]

    def compare_optimization_results(
        self,
        results: List[Tuple[str, Union[optimize.OptimizeResult, Exception]]],
    ) -> Dict[str, Any]:
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
                "method": method_name,
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
