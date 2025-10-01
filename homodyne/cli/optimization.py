"""
CLI Optimization Module
=======================

Optimization execution functions for the homodyne CLI interface.

This module contains the optimization workflow functions that handle the execution
of classical and robust optimization methods, including result processing and
validation.
"""

import logging

import numpy as np

# Module-level logger
logger = logging.getLogger(__name__)


def run_classical_optimization(
    analyzer, initial_params, phi_angles, c2_exp, output_dir=None
):
    """
    Execute classical optimization using traditional methods only.

    This function is called by --method classical and runs ONLY:
    - Nelder-Mead (always available)
    - Gurobi (if available and licensed)

    It explicitly EXCLUDES robust methods (Robust-Wasserstein, Robust-Scenario,
    Robust-Ellipsoidal) which are run separately via --method robust.

    Provides fast parameter estimation with point estimates and goodness-of-fit
    statistics. Uses intelligent angle filtering for performance on large datasets.

    Parameters
    ----------
    analyzer : HomodyneAnalysisCore
        Main analysis engine with loaded configuration
    initial_params : list
        Starting parameter values for optimization
    phi_angles : ndarray
        Angular coordinates for the scattering data
    c2_exp : ndarray
        Experimental correlation function data
    output_dir : Path, optional
        Directory for saving classical results and fitted data

    Returns
    -------
    dict or None
        Results dictionary with optimized parameters and fit statistics,
        or None if optimization fails
    """
    logger.info("Running classical optimization... [CODE-VERSION: 2024-09-30-v2-empty-array-fix]")

    try:
        # Import here to avoid circular imports
        from ..optimization.classical import ClassicalOptimizer

        if ClassicalOptimizer is None:
            logger.error(
                "❌ ClassicalOptimizer is not available. Please ensure the "
                "homodyne.optimization.classical module is installed and accessible."
            )
            return None

        # Use enhanced optimizer if available, otherwise use standard optimizer
        if (
            hasattr(analyzer, "_enhanced_classical_optimizer")
            and analyzer._enhanced_classical_optimizer is not None
        ):
            logger.info("✓ Using enhanced classical optimizer")
            optimizer = analyzer._enhanced_classical_optimizer
        else:
            logger.info("✓ Creating new classical optimizer")
            optimizer = ClassicalOptimizer(analyzer, analyzer.config)

        # Validate data shapes before optimization
        if c2_exp is None or len(c2_exp) == 0:
            logger.error("❌ No experimental data provided for optimization")
            return None

        if phi_angles is None or len(phi_angles) == 0:
            logger.error("❌ No phi angles provided for optimization")
            return None

        # Check data consistency
        expected_shape = (len(phi_angles), c2_exp.shape[1], c2_exp.shape[2])
        if c2_exp.shape != expected_shape:
            logger.warning(
                f"⚠️  Data shape mismatch: expected {expected_shape}, got {c2_exp.shape}"
            )

        # Run the optimization (with return_tuple=True to get scipy result object)
        logger.debug("About to call run_optimization with return_tuple=True")
        params, result = optimizer.run_optimization(
            initial_params=initial_params,
            phi_angles=phi_angles,
            c2_experimental=c2_exp,
            return_tuple=True,
        )
        logger.debug(
            f"run_optimization returned: params type={type(params)}, "
            f"params={params if params is None else f'array[{len(params)}]'}, "
            f"result type={type(result)}"
        )

        if result is None or params is None:
            logger.error("❌ Classical optimization returned no result (None values)")
            logger.error(f"  params is None: {params is None}")
            logger.error(f"  result is None: {result is None}")
            return None

        # Validate optimization result - check multiple conditions
        if not hasattr(result, "x"):
            logger.error("❌ Optimization result has no 'x' attribute")
            return None

        if result.x is None:
            logger.error("❌ Optimization result.x is None")
            return None

        if not isinstance(result.x, np.ndarray):
            logger.error(
                f"❌ Optimization result.x is not ndarray: {type(result.x)}"
            )
            return None

        if result.x.size == 0 or len(result.x) == 0:
            logger.error(
                f"❌ Optimization result.x is empty: size={result.x.size}, len={len(result.x)}"
            )
            logger.error(f"  result.success={getattr(result, 'success', 'N/A')}")
            logger.error(f"  result.message={getattr(result, 'message', 'N/A')}")
            logger.error(f"  result.nit={getattr(result, 'nit', 'N/A')}")
            logger.error(f"  result.nfev={getattr(result, 'nfev', 'N/A')}")
            return None

        logger.info("✓ Classical optimization completed successfully")
        logger.info(f"✓ Final chi-squared: {result.fun:.6f}")
        logger.info(f"✓ Optimization success: {result.success}")

        # Log parameters
        param_names = analyzer.config.get(
            "parameter_names", [f"p{i}" for i in range(len(result.x))]
        )
        for i, (name, value) in enumerate(zip(param_names, result.x, strict=False)):
            logger.info(f"✓ {name}: {value:.6f}")

        return {
            "method": "classical",
            "parameters": result.x,
            "chi_squared": result.fun,
            "success": result.success,
            "result_object": result,
        }

    except Exception as e:
        logger.error(f"❌ Classical optimization failed: {e}")
        import traceback

        logger.debug(f"Full traceback: {traceback.format_exc()}")
        return None


def run_robust_optimization(
    analyzer, initial_params, phi_angles, c2_exp, output_dir=None
):
    """
    Execute robust optimization using uncertainty-aware methods only.

    This function is called by --method robust and runs ONLY:
    - Robust-Wasserstein (Distributionally Robust Optimization)
    - Robust-Scenario (Bootstrap-based)
    - Robust-Ellipsoidal (Bounded uncertainty)

    It explicitly EXCLUDES classical methods (Nelder-Mead, Gurobi) which are
    run separately via --method classical.

    Provides noise-resistant parameter estimation with uncertainty quantification
    and outlier robustness. Automatically handles data uncertainty and measurement
    noise for experimental robustness.

    Parameters
    ----------
    analyzer : HomodyneAnalysisCore
        Main analysis engine with loaded configuration
    initial_params : list
        Starting parameter values for optimization
    phi_angles : ndarray
        Angular coordinates for the scattering data
    c2_exp : ndarray
        Experimental correlation function data
    output_dir : Path, optional
        Directory for saving robust results and uncertainty plots

    Returns
    -------
    dict or None
        Results dictionary with robust parameters and uncertainty bounds,
        or None if optimization fails
    """
    logger.info("Running robust optimization...")

    try:
        # Import here to avoid circular imports
        from ..optimization.robust import create_robust_optimizer

        if create_robust_optimizer is None:
            logger.error(
                "❌ Robust optimization is not available. Please ensure the "
                "homodyne.optimization.robust module is installed and accessible."
            )
            return None

        # Use enhanced optimizer if available, otherwise create new one
        if (
            hasattr(analyzer, "_enhanced_robust_optimizer")
            and analyzer._enhanced_robust_optimizer is not None
        ):
            logger.info("✓ Using enhanced robust optimizer")
            optimizer = analyzer._enhanced_robust_optimizer
        else:
            logger.info("✓ Creating new robust optimizer with caching enabled")
            # Caching now uses safe_hash_object to handle unpicklable objects like RLock
            optimizer = create_robust_optimizer(
                analyzer, analyzer.config, enable_caching=True
            )

        # Validate data shapes before optimization
        if c2_exp is None or len(c2_exp) == 0:
            logger.error("❌ No experimental data provided for optimization")
            return None

        if phi_angles is None or len(phi_angles) == 0:
            logger.error("❌ No phi angles provided for optimization")
            return None

        # Check data consistency
        expected_shape = (len(phi_angles), c2_exp.shape[1], c2_exp.shape[2])
        if c2_exp.shape != expected_shape:
            logger.warning(
                f"⚠️  Data shape mismatch: expected {expected_shape}, got {c2_exp.shape}"
            )

        # Run the robust optimization
        # Returns tuple: (optimal_parameters, optimization_info_dict)
        parameters, result_info = optimizer.run_robust_optimization(
            initial_parameters=initial_params,
            phi_angles=phi_angles,
            c2_experimental=c2_exp,
        )

        if parameters is None or result_info is None:
            logger.error("❌ Robust optimization returned no result")
            logger.error(f"  parameters is None: {parameters is None}")
            logger.error(f"  result_info is None: {result_info is None}")
            return None

        # Validate parameters
        if not isinstance(parameters, np.ndarray) or len(parameters) == 0:
            logger.error(
                f"❌ Robust optimization returned invalid parameters: type={type(parameters)}, "
                f"len={len(parameters) if hasattr(parameters, '__len__') else 'N/A'}"
            )
            return None

        # Extract chi-squared from result_info
        # Check multiple possible keys: chi_squared, final_chi_squared, final_cost
        # Use 0.0 as default instead of None to avoid float() conversion errors
        chi_squared = result_info.get(
            "chi_squared",
            result_info.get("final_chi_squared", result_info.get("final_cost", 0.0)),
        )
        if chi_squared is None:
            chi_squared = 0.0
        method_used = result_info.get("method", "robust")
        success = result_info.get("success", True)

        logger.info(f"✓ Robust optimization completed with method: {method_used}")

        if chi_squared > 0:
            logger.info(f"✓ Final chi-squared: {chi_squared:.6f}")
        else:
            logger.info("✓ Optimization completed")

        # Log parameters
        param_names = analyzer.config.get(
            "parameter_names", [f"p{i}" for i in range(len(parameters))]
        )
        for i, (name, value) in enumerate(zip(param_names, parameters, strict=False)):
            logger.info(f"✓ {name}: {value:.6f}")

        return {
            "method": "robust",
            "parameters": parameters,
            "chi_squared": chi_squared,
            "success": success,
            "result_object": result_info,
        }

    except Exception as e:
        logger.error(f"❌ Robust optimization failed: {e}")
        import traceback

        logger.debug(f"Full traceback: {traceback.format_exc()}")
        return None


def run_all_methods(analyzer, initial_params, phi_angles, c2_exp, output_dir=None):
    """
    Execute both classical and robust optimization methods.

    This function is called by --method all and runs ALL available methods:

    Classical Methods:
    - Nelder-Mead (always available)
    - Gurobi (if available and licensed)

    Robust Methods:
    - Robust-Wasserstein (Distributionally Robust Optimization)
    - Robust-Scenario (Bootstrap-based)
    - Robust-Ellipsoidal (Bounded uncertainty)

    Provides comprehensive analysis with both traditional and robust approaches,
    allowing comparison of optimization strategies and assessment of parameter
    reliability across different methodologies.

    Parameters
    ----------
    analyzer : HomodyneAnalysisCore
        Main analysis engine with loaded configuration
    initial_params : list
        Starting parameter values for optimization
    phi_angles : ndarray
        Angular coordinates for the scattering data
    c2_exp : ndarray
        Experimental correlation function data
    output_dir : Path, optional
        Directory for saving all results and comparison plots

    Returns
    -------
    dict
        Combined results dictionary with both classical and robust results
    """
    logger.info("Running all optimization methods...")
    logger.info("=" * 50)

    results = {}

    # Run classical optimization
    logger.info("PHASE 1: Classical Optimization")
    logger.info("-" * 30)
    classical_result = run_classical_optimization(
        analyzer, initial_params, phi_angles, c2_exp, output_dir
    )

    if classical_result:
        results["classical"] = classical_result
        logger.info("✓ Classical optimization phase completed")
    else:
        logger.warning("⚠️  Classical optimization phase failed")

    logger.info("")

    # Run robust optimization
    logger.info("PHASE 2: Robust Optimization")
    logger.info("-" * 30)
    robust_result = run_robust_optimization(
        analyzer, initial_params, phi_angles, c2_exp, output_dir
    )

    if robust_result:
        results["robust"] = robust_result
        logger.info("✓ Robust optimization phase completed")
    else:
        logger.warning("⚠️  Robust optimization phase failed")

    logger.info("")
    logger.info("=" * 50)
    logger.info("OPTIMIZATION SUMMARY")
    logger.info("=" * 50)

    # Summary comparison
    if "classical" in results and "robust" in results:
        classical_chi2 = results["classical"]["chi_squared"]
        robust_chi2 = results["robust"]["chi_squared"]

        logger.info(f"Classical chi-squared: {classical_chi2:.6f}")
        logger.info(f"Robust chi-squared:    {robust_chi2:.6f}")

        if classical_chi2 < robust_chi2:
            logger.info("⭐ Classical optimization achieved better fit")
        elif robust_chi2 < classical_chi2:
            logger.info("⭐ Robust optimization achieved better fit")
        else:
            logger.info("📊 Both methods achieved similar fit quality")

    elif "classical" in results:
        logger.info("⚠️  Only classical optimization succeeded")

    elif "robust" in results:
        logger.info("⚠️  Only robust optimization succeeded")

    else:
        logger.error("❌ Both optimization phases failed")

    logger.info("=" * 50)

    return results
