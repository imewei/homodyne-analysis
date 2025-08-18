"""
Homodyne Analysis Runner
========================

Command-line interface for running homodyne scattering analysis in X-ray Photon
Correlation Spectroscopy (XPCS) under nonequilibrium conditions.

This script provides a unified interface for:
- Classical optimization (Nelder-Mead) for fast parameter estimation
- Bayesian MCMC sampling (NUTS) for full posterior distributions
- Dual analysis modes: Static (3 params) and Laminar Flow (7 params)
- Comprehensive data validation and quality control
- Automated result saving and visualization
"""

import argparse
import logging
import os
import sys
import time
import json
from pathlib import Path
import numpy as np

# Add the homodyne package to the path for local development
sys.path.insert(0, "./homodyne")

# Import core analysis components with graceful error handling
# This allows the script to provide informative error messages if dependencies are missing
try:
    from homodyne.analysis.core import HomodyneAnalysisCore
    from homodyne.optimization.classical import ClassicalOptimizer
except ImportError:
    # Will be handled with specific error messages during runtime
    HomodyneAnalysisCore = None
    ClassicalOptimizer = None

# Import MCMC components - these require additional dependencies (PyMC, ArviZ)
try:
    from homodyne.optimization.mcmc import create_mcmc_sampler

    MCMC_AVAILABLE = True
except ImportError:
    create_mcmc_sampler = None
    MCMC_AVAILABLE = False


def setup_logging(verbose: bool, output_dir: Path) -> None:
    """
    Configure comprehensive logging for the analysis session.

    Sets up both console and file logging with appropriate formatting.
    Debug level provides detailed execution information for troubleshooting.

    Parameters
    ----------
    verbose : bool
        Enable DEBUG level logging for detailed output
    output_dir : Path
        Directory where log file will be created
    """
    # Ensure output directory exists for log file
    os.makedirs(output_dir, exist_ok=True)

    # Set logging level based on verbosity preference
    level = logging.DEBUG if verbose else logging.INFO

    # Create formatter
    formatter = logging.Formatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Clear any existing handlers
    root_logger.handlers.clear()

    # Add console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # 3. Add file handler that writes to output_dir/run.log
    log_file_path = output_dir / "run.log"
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)


def print_banner(args: argparse.Namespace) -> None:
    """
    Display analysis configuration and session information.

    Provides a clear overview of the selected analysis parameters,
    methods, and output settings before starting the computation.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments containing analysis configuration
    """
    print("=" * 60)
    print("            HOMODYNE ANALYSIS RUNNER")
    print("=" * 60)
    print()
    print(f"Method:           {args.method}")
    print(f"Config file:      {args.config}")
    print(f"Output directory: {args.output_dir}")
    print(
        f"Verbose logging:  {'Enabled (DEBUG)' if args.verbose else 'Disabled (INFO)'}"
    )

    # Show analysis mode
    if args.static:
        print(
            f"Analysis mode:    Static anisotropic (3 parameters, with angle selection)"
        )
    elif args.static_isotropic:
        print(f"Analysis mode:    Static isotropic (3 parameters, no angle selection)")
    elif args.static_anisotropic:
        print(
            f"Analysis mode:    Static anisotropic (3 parameters, with angle selection)"
        )
    elif args.laminar_flow:
        print(f"Analysis mode:    Laminar flow (7 parameters)")
    else:
        print(f"Analysis mode:    From configuration file")

    print()
    print("Starting analysis...")
    print("-" * 60)


def run_analysis(args: argparse.Namespace) -> None:
    """
    Execute the complete homodyne scattering analysis workflow.

    This is the main analysis orchestrator that:
    1. Loads and validates configuration
    2. Initializes the analysis engine
    3. Loads experimental data with optional validation plots
    4. Runs selected optimization method(s)
    5. Saves results and generates diagnostic output

    Parameters
    ----------
    args : argparse.Namespace
        Command-line arguments specifying analysis configuration
    """
    logger = logging.getLogger(__name__)

    # Load configuration and initialize analysis engine

    # 1. Verify the config file exists; exit with clear error if not
    config_path = Path(args.config)
    if not config_path.exists():
        logger.error(f"❌ Configuration file not found: {config_path.absolute()}")
        logger.error(
            "Please check the file path and ensure the configuration file exists."
        )
        sys.exit(1)

    if not config_path.is_file():
        logger.error(f"❌ Configuration path is not a file: {config_path.absolute()}")
        sys.exit(1)

    logger.info(f"✓ Configuration file found: {config_path.absolute()}")

    # 2. Import HomodyneAnalysisCore and create instance
    try:
        from homodyne.analysis.core import HomodyneAnalysisCore

        logger.info("✓ Successfully imported HomodyneAnalysisCore")
    except ImportError as e:
        logger.error(f"❌ Failed to import HomodyneAnalysisCore: {e}")
        logger.error("Please ensure the homodyne package is properly installed.")
        sys.exit(1)

    # 3. Create analysis core instance with error handling
    try:
        logger.info(f"Initializing Homodyne Analysis with config: {config_path}")

        # Apply mode override if specified
        config_override = None
        if args.static:
            # Keep backward compatibility: --static maps to static anisotropic
            config_override = {
                "analysis_settings": {
                    "static_mode": True,
                    "static_submode": "anisotropic",
                }
            }
            logger.info(
                "Using command-line override: static anisotropic mode (3 parameters, with angle selection)"
            )
            logger.warning(
                "Note: --static is deprecated, use --static-anisotropic instead"
            )
        elif args.static_isotropic:
            config_override = {
                "analysis_settings": {
                    "static_mode": True,
                    "static_submode": "isotropic",
                }
            }
            logger.info(
                "Using command-line override: static isotropic mode (3 parameters, no angle selection)"
            )
        elif args.static_anisotropic:
            config_override = {
                "analysis_settings": {
                    "static_mode": True,
                    "static_submode": "anisotropic",
                }
            }
            logger.info(
                "Using command-line override: static anisotropic mode (3 parameters, with angle selection)"
            )
        elif args.laminar_flow:
            config_override = {"analysis_settings": {"static_mode": False}}
            logger.info("Using command-line override: laminar flow mode (7 parameters)")

        # Add experimental data plotting override if specified
        if args.plot_experimental_data:
            if config_override is None:
                config_override = {}
            if "workflow_integration" not in config_override:
                config_override["workflow_integration"] = {}
            if "analysis_workflow" not in config_override["workflow_integration"]:
                config_override["workflow_integration"]["analysis_workflow"] = {}  # type: ignore
            config_override["workflow_integration"]["analysis_workflow"]["plot_experimental_data_on_load"] = True  # type: ignore
            
            # Set the output directory for experimental data plots
            if "output_settings" not in config_override:
                config_override["output_settings"] = {}
            if "plotting" not in config_override["output_settings"]:
                config_override["output_settings"]["plotting"] = {}
            if "output" not in config_override["output_settings"]["plotting"]:
                config_override["output_settings"]["plotting"]["output"] = {}
            config_override["output_settings"]["plotting"]["output"]["base_directory"] = str(args.output_dir / "exp_data")
            
            logger.info(
                "Using command-line override: experimental data plotting enabled"
            )

        analyzer = HomodyneAnalysisCore(
            config_file=str(config_path), config_override=config_override
        )
        logger.info("✓ HomodyneAnalysisCore initialized successfully")

        # Log the actual analysis mode being used
        analysis_mode = analyzer.config_manager.get_analysis_mode()
        param_count = analyzer.config_manager.get_effective_parameter_count()
        logger.info(f"Analysis mode: {analysis_mode} ({param_count} parameters)")
    except (ImportError, ModuleNotFoundError) as e:
        logger.error(f"❌ Import error while creating HomodyneAnalysisCore: {e}")
        logger.error("Please ensure all required dependencies are installed.")
        sys.exit(1)
    except (ValueError, KeyError, FileNotFoundError) as e:
        logger.error(f"❌ JSON configuration error: {e}")
        logger.error("Please check your configuration file format and content.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Unexpected error initializing analysis core: {e}")
        logger.error("Please check your configuration and try again.")
        sys.exit(1)

    # Load experimental data
    logger.info("Loading experimental data...")
    c2_exp, time_length, phi_angles, num_angles = analyzer.load_experimental_data()

    # If only plotting experimental data, exit after loading and plotting
    if args.plot_experimental_data:
        logger.info("✓ Experimental data plotted successfully")
        logger.info("Analysis completed (plotting only mode - no fitting performed)")
        return

    # Get initial parameters from config
    if analyzer.config is None:
        logger.error(
            "❌ Analyzer configuration is None. Please check your configuration file and "
            "ensure it is loaded correctly."
        )
        sys.exit(1)
    initial_params = analyzer.config.get("initial_parameters", {}).get("values", None)
    if initial_params is None:
        logger.error(
            "❌ Initial parameters not found in configuration. Please check your configuration file format."
        )
        sys.exit(1)

    # Calculate chi-squared for initial parameters
    chi2_initial = analyzer.calculate_chi_squared_optimized(
        initial_params, phi_angles, c2_exp, method_name="Initial"
    )
    logger.info(f"Initial χ²_red: {chi2_initial:.6e}")

    # Run optimization based on selected method
    results = None
    methods_attempted = []

    try:
        if args.method == "classical":
            methods_attempted = ["Classical"]
            results = run_classical_optimization(
                analyzer, initial_params, phi_angles, c2_exp, args.output_dir
            )
        elif args.method == "mcmc":
            methods_attempted = ["MCMC"]
            results = run_mcmc_optimization(
                analyzer, initial_params, phi_angles, c2_exp, args.output_dir
            )
        elif args.method == "all":
            methods_attempted = ["Classical", "MCMC"]
            results = run_all_methods(
                analyzer, initial_params, phi_angles, c2_exp, args.output_dir
            )

        if results:
            # Save results - skip default plots for classical-only methods
            skip_plots = args.method == "classical"
            analyzer.save_results_with_config(results, skip_plots=skip_plots, output_dir=str(args.output_dir))

            # Perform per-angle chi-squared analysis for each successful method
            successful_methods = results.get("methods_used", [])
            logger.info(
                f"Running per-angle chi-squared analysis for methods: {', '.join(successful_methods)}"
            )

            for method in successful_methods:
                method_key = f"{method.lower()}_optimization"
                if method_key in results and "parameters" in results[method_key]:
                    method_params = results[method_key]["parameters"]
                    if method_params is not None:
                        if method.upper() == "MCMC":
                            # For MCMC, log convergence diagnostics instead of chi-squared
                            try:
                                mcmc_results = results[method_key]
                                if "diagnostics" in mcmc_results:
                                    diag = mcmc_results["diagnostics"]
                                    logger.info(
                                        f"MCMC convergence diagnostics [{method}]:"
                                    )
                                    logger.info(
                                        f"  Convergence status: {diag.get('assessment', 'Unknown')}"
                                    )
                                    logger.info(
                                        f"  Maximum R̂ (R-hat): {diag.get('max_rhat', 'N/A'):.4f}"
                                    )
                                    logger.info(
                                        f"  Minimum ESS: {diag.get('min_ess', 'N/A'):.0f}"
                                    )

                                    # Quality assessment based on convergence criteria from config
                                    max_rhat = diag.get("max_rhat", float("inf"))
                                    min_ess = diag.get("min_ess", 0)

                                    # Get thresholds from config or use defaults
                                    config = getattr(analyzer, "config", {})
                                    validation_config = config.get(
                                        "validation_rules", {}
                                    )
                                    mcmc_config = validation_config.get(
                                        "mcmc_convergence", {}
                                    )
                                    rhat_thresholds = mcmc_config.get(
                                        "rhat_thresholds", {}
                                    )
                                    ess_thresholds = mcmc_config.get(
                                        "ess_thresholds", {}
                                    )

                                    excellent_rhat = rhat_thresholds.get(
                                        "excellent_threshold", 1.01
                                    )
                                    good_rhat = rhat_thresholds.get(
                                        "good_threshold", 1.05
                                    )
                                    acceptable_rhat = rhat_thresholds.get(
                                        "acceptable_threshold", 1.1
                                    )

                                    excellent_ess = ess_thresholds.get(
                                        "excellent_threshold", 400
                                    )
                                    good_ess = ess_thresholds.get("good_threshold", 200)
                                    acceptable_ess = ess_thresholds.get(
                                        "acceptable_threshold", 100
                                    )

                                    if (
                                        max_rhat < excellent_rhat
                                        and min_ess > excellent_ess
                                    ):
                                        quality = "excellent"
                                    elif max_rhat < good_rhat and min_ess > good_ess:
                                        quality = "good"
                                    elif (
                                        max_rhat < acceptable_rhat
                                        and min_ess > acceptable_ess
                                    ):
                                        quality = "acceptable"
                                    else:
                                        quality = "poor"

                                    logger.info(f"  MCMC quality: {quality.upper()}")

                                    # Additional metrics if available
                                    if "trace" in mcmc_results:
                                        logger.info(
                                            f"  Sampling completed with posterior analysis available"
                                        )
                                else:
                                    logger.warning(
                                        f"No convergence diagnostics available for {method}"
                                    )
                            except Exception as e:
                                logger.warning(
                                    f"Failed to log MCMC diagnostics for {method}: {e}"
                                )
                        else:
                            # For classical optimization methods, use chi-squared analysis
                            try:
                                analyzer.analyze_per_angle_chi_squared(
                                    np.array(method_params),
                                    phi_angles,
                                    c2_exp,
                                    method_name=method,
                                    save_to_file=True,
                                    output_dir=args.output_dir,
                                )
                            except Exception as e:
                                logger.warning(
                                    f"Per-angle analysis failed for {method}: {e}"
                                )

            logger.info("✓ Analysis completed successfully!")
            logger.info(f"Successful methods: {', '.join(successful_methods)}")
        else:
            logger.error("❌ Analysis failed - no results generated")
            if len(methods_attempted) == 1:
                # Single method failed - this is a hard failure
                logger.error(
                    f"The only requested method ({args.method}) failed to complete"
                )
                sys.exit(1)
            else:
                # Multiple methods attempted - check if any succeeded
                logger.error("All attempted optimization methods failed")
                sys.exit(1)

    except Exception as e:
        logger.error(f"❌ Unexpected error during optimization: {e}")
        logger.error("Please check your configuration and data files")
        import traceback

        logger.debug(f"Full traceback: {traceback.format_exc()}")
        sys.exit(1)


def run_classical_optimization(analyzer, initial_params, phi_angles, c2_exp, output_dir=None):
    """
    Execute classical optimization using Nelder-Mead simplex method.

    Provides fast parameter estimation with point estimates and goodness-of-fit
    statistics. Uses scipy.optimize for robust convergence with intelligent
    angle filtering for performance on large datasets.

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
    logger = logging.getLogger(__name__)
    logger.info("Running classical optimization...")

    try:
        if ClassicalOptimizer is None:
            logger.error(
                "❌ ClassicalOptimizer is not available. Please ensure the "
                "homodyne.optimization.classical module is installed and accessible."
            )
            return None

        optimizer = ClassicalOptimizer(analyzer, analyzer.config)
        best_params, result = optimizer.run_classical_optimization_optimized(
            initial_parameters=initial_params,
            phi_angles=phi_angles,
            c2_experimental=c2_exp,
        )

        # Store best parameters on analyzer core for MCMC initialization
        if (
            hasattr(optimizer, "best_params_classical")
            and optimizer.best_params_classical is not None
        ):
            analyzer.best_params_classical = optimizer.best_params_classical
            logger.info("✓ Classical results stored for MCMC initialization")

        # Save experimental and fitted data to classical directory
        if output_dir is not None and best_params is not None:
            _save_classical_fitted_data(analyzer, best_params, phi_angles, c2_exp, output_dir)
            # Generate classical-specific plots
            _generate_classical_plots(analyzer, best_params, phi_angles, c2_exp, output_dir)

        return {
            "classical_optimization": {
                "parameters": best_params,
                "chi_squared": result.fun,
                "optimization_time": getattr(result, "execution_time", 0),
                "total_time": 0,
                "success": result.success,
                "method": getattr(result, "method", "unknown"),
                "iterations": getattr(result, "nit", None),
                "function_evaluations": getattr(result, "nfev", None),
            },
            "classical_summary": {
                "parameters": best_params,
                "chi_squared": result.fun,
                "method": "Classical",
                "evaluation_metric": "chi_squared",
                "_note": "Classical optimization uses chi-squared for quality assessment",
            },
            "methods_used": ["Classical"],
        }
    except ImportError as e:
        error_msg = f"Classical optimization failed - missing dependencies: {e}"
        logger.error(error_msg)
        if "scipy" in str(e).lower():
            logger.error("❌ Install scipy: pip install scipy")
        elif "numpy" in str(e).lower():
            logger.error("❌ Install numpy: pip install numpy")
        else:
            logger.error("❌ Install required dependencies: pip install scipy numpy")
        return None
    except (ValueError, KeyError) as e:
        error_msg = f"Classical optimization failed - configuration error: {e}"
        logger.error(error_msg)
        logger.error(
            "❌ Please check your configuration file format and parameter bounds"
        )
        return None
    except Exception as e:
        error_msg = f"Classical optimization failed - unexpected error: {e}"
        logger.error(error_msg)
        logger.error("❌ Please check your data files and configuration")
        return None


def run_mcmc_optimization(
    analyzer, initial_params, phi_angles, c2_exp, output_dir=None
):
    """
    Execute Bayesian MCMC sampling using NUTS (No-U-Turn Sampler).

    Provides full posterior distributions with uncertainty quantification.
    Uses PyMC for robust sampling with convergence diagnostics (R-hat, ESS).
    Results include parameter uncertainties and correlation analysis.

    Parameters
    ----------
    analyzer : HomodyneAnalysisCore
        Main analysis engine with loaded configuration
    initial_params : list
        Starting parameter values (used for prior initialization)
    phi_angles : ndarray
        Angular coordinates for the scattering data
    c2_exp : ndarray
        Experimental correlation function data
    output_dir : Path, optional
        Directory for saving MCMC traces and diagnostics

    Returns
    -------
    dict or None
        Results dictionary with posterior statistics and convergence info,
        or None if sampling fails
    """
    logger = logging.getLogger(__name__)
    logger.info("Running MCMC sampling...")

    # Step 1: Attempt to import create_mcmc_sampler
    try:
        from homodyne.optimization.mcmc import create_mcmc_sampler

        logger.info("✓ Successfully imported create_mcmc_sampler")
    except ImportError as e:
        logger.error(f"❌ Failed to import MCMC module: {e}")
        if (
            "pymc" in str(e).lower()
            or "pytensor" in str(e).lower()
            or "arviz" in str(e).lower()
        ):
            logger.error(
                "❌ MCMC sampling requires PyMC and ArviZ: pip install pymc arviz"
            )
        else:
            logger.error(
                "❌ Install required dependencies: pip install pymc arviz pytensor"
            )
        return None

    try:
        # Step 2.5: Set initial parameters for MCMC if not already set by classical optimization
        if (
            not hasattr(analyzer, "best_params_classical")
            or analyzer.best_params_classical is None
        ):
            analyzer.best_params_classical = initial_params
            logger.info("✓ Using provided initial parameters for MCMC initialization")
        else:
            logger.info("✓ Using stored classical results for MCMC initialization")

        # Step 3: Create MCMC sampler (this already validates)
        logger.info("Creating MCMC sampler...")
        sampler = create_mcmc_sampler(analyzer, analyzer.config)
        logger.info("✓ MCMC sampler created successfully")

        # Step 4: Run MCMC analysis and time execution
        logger.info("Starting MCMC sampling...")
        mcmc_start_time = time.time()

        # Run the MCMC analysis with angle filtering by default
        mcmc_results = sampler.run_mcmc_analysis(
            c2_experimental=c2_exp,
            phi_angles=phi_angles,
            filter_angles_for_optimization=True,  # Use angle filtering by default
        )

        mcmc_execution_time = time.time() - mcmc_start_time
        logger.info(f"✓ MCMC sampling completed in {mcmc_execution_time:.2f} seconds")

        # Step 5 & 6: Save inference data and write convergence diagnostics
        if output_dir is None:
            output_dir = Path("./homodyne_results")
        else:
            output_dir = Path(output_dir)

        # Create mcmc subdirectory
        mcmc_output_dir = output_dir / "mcmc"
        mcmc_output_dir.mkdir(parents=True, exist_ok=True)

        # Save inference data (NetCDF via arviz.to_netcdf) if trace is available
        if "trace" in mcmc_results and mcmc_results["trace"] is not None:
            try:
                import arviz as az

                netcdf_path = mcmc_output_dir / "mcmc_trace.nc"
                az.to_netcdf(mcmc_results["trace"], str(netcdf_path))
                logger.info(f"✓ MCMC trace saved to NetCDF: {netcdf_path}")
            except ImportError as import_err:
                logger.error(f"❌ ArviZ not available for saving trace: {import_err}")
                logger.error("❌ Install ArviZ: pip install arviz")
            except Exception as e:
                logger.error(f"❌ Failed to save NetCDF trace: {e}")

        # Prepare summary results for JSON
        summary_results = {
            "method": "MCMC_NUTS",
            "execution_time_seconds": mcmc_execution_time,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "posterior_means": mcmc_results.get("posterior_means", {}),
            "mcmc_config": mcmc_results.get("config", {}),
        }

        # Add convergence diagnostics to summary
        if "diagnostics" in mcmc_results:
            diagnostics = mcmc_results["diagnostics"]
            summary_results["convergence_diagnostics"] = {
                "max_rhat": diagnostics.get("max_rhat"),
                "min_ess": diagnostics.get("min_ess"),
                "converged": diagnostics.get("converged", False),
                "assessment": diagnostics.get("assessment", "Unknown"),
            }

            # Write convergence diagnostics to log (Step 6)
            logger.info("Convergence Diagnostics:")
            logger.info(f"  Max R-hat: {diagnostics.get('max_rhat', 'N/A')}")
            logger.info(f"  Min ESS: {diagnostics.get('min_ess', 'N/A')}")
            logger.info(f"  Converged: {diagnostics.get('converged', False)}")
            logger.info(f"  Assessment: {diagnostics.get('assessment', 'Unknown')}")

            if not diagnostics.get("converged", False):
                logger.warning(
                    "⚠ MCMC chains may not have converged - check diagnostics!"
                )

        # Add posterior statistics if available
        if hasattr(sampler, "extract_posterior_statistics"):
            try:
                posterior_stats = sampler.extract_posterior_statistics(
                    mcmc_results.get("trace")
                )
                if posterior_stats and "parameter_statistics" in posterior_stats:
                    summary_results["parameter_statistics"] = posterior_stats[
                        "parameter_statistics"
                    ]
            except Exception as e:
                logger.warning(f"Failed to extract posterior statistics: {e}")

        # Save summary JSON to output_dir/mcmc
        summary_json_path = mcmc_output_dir / "mcmc_summary.json"
        try:
            with open(summary_json_path, "w") as f:
                json.dump(summary_results, f, indent=2, default=str)
            logger.info(f"✓ MCMC summary saved to: {summary_json_path}")
        except Exception as e:
            logger.error(f"❌ Failed to save MCMC summary JSON: {e}")

        # Extract best parameters from posterior means for compatibility with other methods
        best_params = None
        if "posterior_means" in mcmc_results:
            param_names = analyzer.config.get("initial_parameters", {}).get(
                "parameter_names", []
            )
            posterior_means = mcmc_results["posterior_means"]
            best_params = [posterior_means.get(name, 0.0) for name in param_names]
            
        # Save experimental and fitted data to mcmc directory
        if output_dir is not None and best_params is not None:
            _save_mcmc_fitted_data(analyzer, best_params, phi_angles, c2_exp, output_dir)
            # Generate mcmc-specific plots
            _generate_mcmc_plots(analyzer, best_params, phi_angles, c2_exp, output_dir, mcmc_results)

        # Extract convergence quality for MCMC summary (no chi-squared calculation)
        convergence_quality = "unknown"
        if "diagnostics" in mcmc_results:
            diag = mcmc_results["diagnostics"]
            max_rhat = diag.get("max_rhat", float("inf"))
            min_ess = diag.get("min_ess", 0)

            # Use same thresholds as in per-angle analysis
            if max_rhat < 1.01 and min_ess > 400:
                convergence_quality = "excellent"
            elif max_rhat < 1.05 and min_ess > 200:
                convergence_quality = "good"
            elif max_rhat < 1.1 and min_ess > 100:
                convergence_quality = "acceptable"
            else:
                convergence_quality = "poor"

            logger.info(f"MCMC convergence quality: {convergence_quality.upper()}")
            logger.info(f"MCMC posterior mean parameters: {best_params}")
        else:
            logger.warning("No convergence diagnostics available for MCMC results")

        # Format results for compatibility with main analysis framework
        return {
            "mcmc_optimization": {
                "parameters": best_params,
                "convergence_quality": convergence_quality,
                "optimization_time": mcmc_execution_time,
                "total_time": mcmc_execution_time,
                "success": mcmc_results.get("diagnostics", {}).get("converged", True),
                "method": "MCMC_NUTS",
                "posterior_means": mcmc_results.get("posterior_means", {}),
                "convergence_diagnostics": mcmc_results.get("diagnostics", {}),
                # Include trace data for plotting
                "trace": mcmc_results.get("trace"),
                # Include chi_squared for plotting method selection
                "chi_squared": mcmc_results.get("chi_squared", np.inf),
            },
            "mcmc_summary": {
                "parameters": best_params,
                "convergence_quality": convergence_quality,
                "max_rhat": mcmc_results.get("diagnostics", {}).get("max_rhat", None),
                "min_ess": mcmc_results.get("diagnostics", {}).get("min_ess", None),
                "method": "MCMC",
                "evaluation_metric": "convergence_diagnostics",
                "_note": "MCMC uses convergence diagnostics instead of chi-squared for quality assessment",
            },
            "methods_used": ["MCMC"],
            # Include trace and diagnostics at top level for plotting functions
            "trace": mcmc_results.get("trace"),
            "diagnostics": mcmc_results.get("diagnostics"),
        }

    except ImportError as e:
        error_msg = f"MCMC optimization failed - missing dependencies: {e}"
        logger.error(error_msg)
        if "pymc" in str(e).lower():
            logger.error("❌ Install PyMC: pip install pymc")
        elif "arviz" in str(e).lower():
            logger.error("❌ Install ArviZ: pip install arviz")
        elif "pytensor" in str(e).lower():
            logger.error("❌ Install PyTensor: pip install pytensor")
        else:
            logger.error(
                "❌ Install required dependencies: pip install pymc arviz pytensor"
            )
        return None
    except (ValueError, KeyError) as e:
        error_msg = f"MCMC optimization failed - configuration error: {e}"
        logger.error(error_msg)
        logger.error("❌ Please check your MCMC configuration and parameter priors")
        return None
    except Exception as e:
        error_msg = f"MCMC optimization failed - unexpected error: {e}"
        logger.error(error_msg)
        logger.error("❌ Please check your data files and MCMC configuration")
        import traceback

        logger.debug(f"Full traceback: {traceback.format_exc()}")
        return None


def run_all_methods(analyzer, initial_params, phi_angles, c2_exp, output_dir=None):
    """
    Execute both classical and MCMC optimization methods sequentially.

    Recommended workflow that combines fast classical optimization for initial
    parameter estimates with comprehensive MCMC sampling for full uncertainty
    analysis. Gracefully handles failures in individual methods.

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
        Directory for saving results and diagnostics

    Returns
    -------
    dict or None
        Combined results from all successful methods,
        or None if all methods fail
    """
    logger = logging.getLogger(__name__)
    logger.info("Running all optimization methods...")

    all_results = {}
    methods_used = []
    methods_attempted = []

    # Run classical optimization
    methods_attempted.append("Classical")
    logger.info("Attempting Classical optimization...")
    classical_results = run_classical_optimization(
        analyzer, initial_params, phi_angles, c2_exp, output_dir
    )
    if classical_results:
        all_results.update(classical_results)
        methods_used.append("Classical")
        logger.info("✓ Classical optimization completed successfully")
    else:
        logger.warning("⚠ Classical optimization failed")

    # Run MCMC sampling
    methods_attempted.append("MCMC")
    logger.info("Attempting MCMC sampling...")

    # Use classical results for MCMC initialization if available
    mcmc_initial_params = initial_params
    if classical_results and "classical_summary" in classical_results:
        classical_best_params = classical_results["classical_summary"].get("parameters")
        if classical_best_params is not None:
            mcmc_initial_params = classical_best_params
            logger.info(
                "✓ Using classical optimization results for MCMC initialization"
            )
        else:
            logger.info(
                "⚠ Classical results available but no parameters found, using initial parameters for MCMC"
            )
    else:
        logger.info(
            "⚠ No classical results available, using initial parameters for MCMC"
        )

    mcmc_results = run_mcmc_optimization(
        analyzer, mcmc_initial_params, phi_angles, c2_exp, output_dir
    )
    if mcmc_results:
        all_results.update(mcmc_results)
        methods_used.append("MCMC")
        logger.info("✓ MCMC sampling completed successfully")
    else:
        logger.warning("⚠ MCMC sampling failed")

    # Summary of results
    logger.info(f"Methods attempted: {', '.join(methods_attempted)}")
    logger.info(f"Methods completed successfully: {', '.join(methods_used)}")

    if all_results:
        all_results["methods_used"] = methods_used
        all_results["methods_attempted"] = methods_attempted

        # Add method-appropriate summary information
        methods_summary = {}

        if "Classical" in methods_used and "classical_summary" in all_results:
            classical_summary = all_results["classical_summary"]
            methods_summary["Classical"] = {
                "evaluation_metric": "chi_squared",
                "chi_squared": classical_summary.get("chi_squared"),
                "parameters": classical_summary.get("parameters"),
                "quality_note": "Lower chi-squared indicates better fit to experimental data",
            }

        if "MCMC" in methods_used and "mcmc_summary" in all_results:
            mcmc_summary = all_results["mcmc_summary"]
            methods_summary["MCMC"] = {
                "evaluation_metric": "convergence_diagnostics",
                "convergence_quality": mcmc_summary.get("convergence_quality"),
                "max_rhat": mcmc_summary.get("max_rhat"),
                "min_ess": mcmc_summary.get("min_ess"),
                "parameters": mcmc_summary.get("parameters"),
                "quality_note": "Convergence quality based on R̂ and ESS criteria",
            }

        all_results["methods_comparison"] = {
            "_note": "Methods use different evaluation criteria - do not directly compare chi-squared to convergence diagnostics",
            "methods_summary": methods_summary,
            "recommendation": "Use Classical for fast parameter estimates; use MCMC for uncertainty quantification",
        }

        return all_results

    logger.error("❌ All optimization methods failed")
    return None


def _generate_classical_plots(analyzer, best_params, phi_angles, c2_exp, output_dir):
    """
    Generate plots specifically for classical optimization results.
    
    Parameters
    ----------
    analyzer : HomodyneAnalysisCore
        Main analysis engine with loaded configuration
    best_params : np.ndarray
        Optimized parameters from classical optimization
    phi_angles : np.ndarray
        Angular coordinates for the scattering data
    c2_exp : np.ndarray
        Experimental correlation function data
    output_dir : Path
        Output directory for saving classical results
    """
    logger = logging.getLogger(__name__)
    
    try:
        from pathlib import Path
        import numpy as np
        
        # Create classical subdirectory
        if output_dir is None:
            output_dir = Path("./homodyne_results")
        else:
            output_dir = Path(output_dir)
        
        classical_dir = output_dir / "classical"
        classical_dir.mkdir(parents=True, exist_ok=True)
        
        # Check if plotting is enabled
        config = analyzer.config
        output_settings = config.get("output_settings", {})
        reporting = output_settings.get("reporting", {})
        if not reporting.get("generate_plots", True):
            logger.info("Plotting disabled in configuration - skipping classical plot generation")
            return
            
        logger.info("Generating classical optimization plots...")
        
        # Calculate theoretical data with optimized parameters
        c2_theory = analyzer.calculate_c2_nonequilibrium_laminar_parallel(best_params, phi_angles)
        
        # Generate C2 heatmaps in classical directory
        try:
            from homodyne.plotting import plot_c2_heatmaps
            
            # Create time arrays for plotting
            dt = analyzer.dt
            n_angles, n_t2, n_t1 = c2_exp.shape
            t2 = np.arange(n_t2) * dt
            t1 = np.arange(n_t1) * dt
            
            logger.info("Generating C2 correlation heatmaps for classical results...")
            success = plot_c2_heatmaps(
                c2_exp,
                c2_theory,
                phi_angles,
                classical_dir,
                config,
                t2=t2,
                t1=t1,
            )
            
            if success:
                logger.info("✓ Classical C2 heatmaps generated successfully")
            else:
                logger.warning("⚠ Some classical C2 heatmaps failed to generate")
                
        except Exception as e:
            logger.error(f"Failed to generate classical C2 heatmaps: {e}")
            import traceback
            logger.debug(f"Full traceback: {traceback.format_exc()}")
        
    except Exception as e:
        logger.error(f"Failed to generate classical plots: {e}")
        import traceback
        logger.debug(f"Full traceback: {traceback.format_exc()}")


def _save_classical_fitted_data(analyzer, best_params, phi_angles, c2_exp, output_dir):
    """
    Calculate fitted data and save experimental and fitted data to classical directory.
    
    Parameters
    ----------
    analyzer : HomodyneAnalysisCore
        Main analysis engine with loaded configuration
    best_params : np.ndarray
        Optimized parameters from classical optimization
    phi_angles : np.ndarray
        Angular coordinates for the scattering data
    c2_exp : np.ndarray
        Experimental correlation function data
    output_dir : Path
        Output directory for saving classical results
    """
    logger = logging.getLogger(__name__)
    
    try:
        from pathlib import Path
        import numpy as np
        
        # Create classical subdirectory
        if output_dir is None:
            output_dir = Path("./homodyne_results")
        else:
            output_dir = Path(output_dir)
        
        classical_dir = output_dir / "classical"
        classical_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Calculating fitted data for classical optimization results...")
        
        # Calculate theoretical data with optimized parameters
        c2_theory = analyzer.calculate_c2_nonequilibrium_laminar_parallel(best_params, phi_angles)
        
        # Calculate fitted data for each angle using scaling optimization
        n_angles, n_t2, n_t1 = c2_exp.shape
        c2_fitted = np.zeros_like(c2_exp)
        
        uncertainty_factor = analyzer.config.get("advanced_settings", {}).get("chi_squared_calculation", {}).get("uncertainty_estimation_factor", 0.1)
        min_sigma = analyzer.config.get("advanced_settings", {}).get("chi_squared_calculation", {}).get("minimum_sigma", 1e-10)
        
        for i in range(n_angles):
            # Flatten the 2D correlation data for least squares fitting
            exp_flat = c2_exp[i].flatten()
            theory_flat = c2_theory[i].flatten()
            
            # Perform scaling optimization: fitted = theory * contrast + offset
            A = np.vstack([theory_flat, np.ones(len(theory_flat))]).T
            try:
                scaling, _, _, _ = np.linalg.lstsq(A, exp_flat, rcond=None)
                if len(scaling) == 2:
                    contrast, offset = scaling
                    c2_fitted[i] = c2_theory[i] * contrast + offset
                    logger.debug(f"Angle {i} (φ={phi_angles[i]:.1f}°): contrast={contrast:.3f}, offset={offset:.4f}")
                else:
                    c2_fitted[i] = c2_theory[i]
                    logger.debug(f"Angle {i} (φ={phi_angles[i]:.1f}°): using unscaled theory (no scaling solution)")
            except np.linalg.LinAlgError:
                c2_fitted[i] = c2_theory[i]
                logger.warning(f"Scaling optimization failed for angle {i}, using unscaled theory")
        
        # Calculate residuals: experimental - fitted
        c2_residuals = c2_exp - c2_fitted
        
        # Save data files
        exp_file = classical_dir / "experimental_data.npz"
        fitted_file = classical_dir / "fitted_data.npz"
        residuals_file = classical_dir / "residuals_data.npz"
        
        np.savez_compressed(exp_file, 
                           c2_experimental=c2_exp, 
                           phi_angles=phi_angles,
                           parameters=best_params,
                           parameter_names=analyzer.config.get("initial_parameters", {}).get("parameter_names", []))
        
        np.savez_compressed(fitted_file,
                           c2_fitted=c2_fitted,
                           phi_angles=phi_angles,
                           parameters=best_params,
                           parameter_names=analyzer.config.get("initial_parameters", {}).get("parameter_names", []))
        
        np.savez_compressed(residuals_file,
                           c2_residuals=c2_residuals,
                           phi_angles=phi_angles,
                           parameters=best_params,
                           parameter_names=analyzer.config.get("initial_parameters", {}).get("parameter_names", []))
        
        logger.info(f"✓ Classical fitting data saved to {classical_dir}/")
        logger.info(f"  - Experimental data: {exp_file}")
        logger.info(f"  - Fitted data: {fitted_file}")
        logger.info(f"  - Residuals data: {residuals_file}")
        
    except Exception as e:
        logger.error(f"Failed to save classical fitted data: {e}")
        import traceback
        logger.debug(f"Full traceback: {traceback.format_exc()}")


def _save_mcmc_fitted_data(analyzer, best_params, phi_angles, c2_exp, output_dir):
    """
    Calculate fitted data and save experimental and fitted data to mcmc directory.
    
    Parameters
    ----------
    analyzer : HomodyneAnalysisCore
        Analysis engine with loaded configuration
    best_params : np.ndarray
        Optimized parameters from MCMC posterior means
    phi_angles : np.ndarray
        Angular coordinates for the scattering data
    c2_exp : np.ndarray
        Experimental correlation function data
    output_dir : Path
        Output directory for saving MCMC results
    """
    logger = logging.getLogger(__name__)
    
    try:
        from pathlib import Path
        import numpy as np
        
        # Create mcmc subdirectory
        if output_dir is None:
            output_dir = Path("./homodyne_results")
        else:
            output_dir = Path(output_dir)
        
        mcmc_dir = output_dir / "mcmc"
        mcmc_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("Calculating fitted data for MCMC optimization results...")
        
        # Calculate theoretical data with optimized parameters
        c2_theory = analyzer.calculate_c2_nonequilibrium_laminar_parallel(best_params, phi_angles)
        
        # Calculate fitted data using least squares scaling for each angle
        # fitted = contrast * theory + offset
        c2_fitted = np.zeros_like(c2_exp)
        c2_residuals = np.zeros_like(c2_exp)
        
        for angle_idx in range(c2_exp.shape[0]):
            # Flatten data for least squares
            theory_flat = c2_theory[angle_idx].flatten()
            exp_flat = c2_exp[angle_idx].flatten()
            
            # Create design matrix for least squares: [theory, ones]
            A = np.vstack([theory_flat, np.ones(len(theory_flat))]).T
            
            try:
                # Solve for scaling parameters
                scaling_params, residuals, rank, s = np.linalg.lstsq(A, exp_flat, rcond=None)
                contrast, offset = scaling_params
                
                # Calculate fitted data
                fitted_flat = theory_flat * contrast + offset
                c2_fitted[angle_idx] = fitted_flat.reshape(c2_theory[angle_idx].shape)
                
                # Calculate residuals
                c2_residuals[angle_idx] = c2_exp[angle_idx] - c2_fitted[angle_idx]
                
            except np.linalg.LinAlgError as e:
                logger.warning(f"Least squares failed for angle {angle_idx}: {e}")
                # Fallback: use theory as fitted data
                c2_fitted[angle_idx] = c2_theory[angle_idx]
                c2_residuals[angle_idx] = c2_exp[angle_idx] - c2_theory[angle_idx]
        
        # Save data as compressed NPZ files
        experimental_file = mcmc_dir / "experimental_data.npz"
        fitted_file = mcmc_dir / "fitted_data.npz"
        residuals_file = mcmc_dir / "residuals_data.npz"
        
        np.savez_compressed(experimental_file, data=c2_exp)
        np.savez_compressed(fitted_file, data=c2_fitted)
        np.savez_compressed(residuals_file, data=c2_residuals)
        
        logger.info("✓ MCMC fitting data saved to homodyne_results/mcmc/")
        logger.info(f"  - Experimental data: {experimental_file}")
        logger.info(f"  - Fitted data: {fitted_file}")
        logger.info(f"  - Residuals data: {residuals_file}")
        
    except Exception as e:
        logger.error(f"Failed to save MCMC fitted data: {e}")
        import traceback
        logger.debug(f"Full traceback: {traceback.format_exc()}")


def _generate_mcmc_plots(analyzer, best_params, phi_angles, c2_exp, output_dir, mcmc_results):
    """
    Generate plots specifically for MCMC optimization results.
    
    Parameters
    ----------
    analyzer : HomodyneAnalysisCore
        Analysis engine with loaded configuration
    best_params : np.ndarray
        Optimized parameters from MCMC posterior means
    phi_angles : np.ndarray
        Angular coordinates for the scattering data
    c2_exp : np.ndarray
        Experimental correlation function data
    output_dir : Path
        Output directory for saving MCMC results
    mcmc_results : dict
        Complete MCMC results including trace data
    """
    logger = logging.getLogger(__name__)
    
    try:
        from pathlib import Path
        import numpy as np
        
        # Create mcmc subdirectory
        if output_dir is None:
            output_dir = Path("./homodyne_results")
        else:
            output_dir = Path(output_dir)
        
        mcmc_dir = output_dir / "mcmc"
        mcmc_dir.mkdir(parents=True, exist_ok=True)
        
        # Check if plotting is enabled
        config = analyzer.config
        output_settings = config.get("output_settings", {})
        reporting = output_settings.get("reporting", {})
        if not reporting.get("generate_plots", True):
            logger.info("Plotting disabled in configuration - skipping MCMC plot generation")
            return
            
        logger.info("Generating MCMC optimization plots...")
        
        # Calculate theoretical data with optimized parameters
        c2_theory = analyzer.calculate_c2_nonequilibrium_laminar_parallel(best_params, phi_angles)
        
        # Generate C2 heatmaps in mcmc directory
        try:
            from homodyne.plotting import plot_c2_heatmaps
            
            # Create time arrays for plotting
            dt = analyzer.dt
            n_angles, n_t2, n_t1 = c2_exp.shape
            t2 = np.arange(n_t2) * dt
            t1 = np.arange(n_t1) * dt
            
            logger.info("Generating C2 correlation heatmaps for MCMC results...")
            success = plot_c2_heatmaps(
                c2_exp,
                c2_theory,
                phi_angles,
                mcmc_dir,
                config,
                t2=t2,
                t1=t1,
            )
            
            if success:
                logger.info("✓ MCMC C2 heatmaps generated successfully")
            else:
                logger.warning("⚠ Some MCMC C2 heatmaps failed to generate")
                
        except Exception as e:
            logger.error(f"Failed to generate MCMC C2 heatmaps: {e}")
        
        # Generate 3D surface plots with confidence intervals
        try:
            from homodyne.plotting import plot_3d_surface
            
            # Extract posterior samples from trace for confidence intervals  
            trace = mcmc_results.get("trace")
            if trace is not None:
                logger.info("Generating 3D surface plots with MCMC confidence intervals...")
                
                # Get parameter samples from the trace
                try:
                    import arviz as az
                    
                    # Extract posterior samples - convert InferenceData to numpy array
                    if hasattr(trace, 'posterior'):
                        # Get parameter names from config
                        param_names = config.get("initial_parameters", {}).get("parameter_names", [])
                        
                        # Extract samples for each parameter and stack them
                        param_samples = []
                        for param_name in param_names:
                            if param_name in trace.posterior:
                                # Get all chains and draws for this parameter
                                param_data = trace.posterior[param_name].values
                                # Reshape from (chains, draws) to (chains*draws,)
                                param_samples.append(param_data.reshape(-1))
                        
                        if param_samples:
                            # Stack to get shape (n_samples, n_parameters) 
                            param_samples_array = np.column_stack(param_samples)
                            n_samples = min(500, param_samples_array.shape[0])  # Limit for performance
                            
                            # Subsample for performance
                            indices = np.linspace(0, param_samples_array.shape[0]-1, n_samples, dtype=int)
                            param_samples_subset = param_samples_array[indices]
                            
                            logger.info(f"Using {n_samples} posterior samples for 3D confidence intervals")
                            
                            # Generate C2 samples for each parameter sample
                            c2_posterior_samples = []
                            for i, params in enumerate(param_samples_subset):
                                if i % 50 == 0:  # Log progress every 50 samples
                                    logger.debug(f"Processing posterior sample {i+1}/{n_samples}")
                                
                                # Calculate theoretical C2 for this parameter sample
                                c2_sample = analyzer.calculate_c2_nonequilibrium_laminar_parallel(params, phi_angles)
                                
                                # Apply least squares scaling to match experimental data structure
                                for j in range(c2_exp.shape[0]):  # For each angle
                                    exp_data = c2_exp[j].flatten()
                                    theory_data = c2_sample[j].flatten()
                                    
                                    # Least squares scaling: fitted = contrast * theory + offset
                                    A = np.vstack([theory_data, np.ones(len(theory_data))]).T
                                    scaling, residuals, rank, s = np.linalg.lstsq(A, exp_data, rcond=None)
                                    contrast, offset = scaling
                                    
                                    # Apply scaling to this angle slice
                                    c2_sample[j] = theory_data.reshape(c2_exp[j].shape) * contrast + offset
                                
                                c2_posterior_samples.append(c2_sample)
                            
                            # Convert to numpy array: (n_samples, n_angles, n_t2, n_t1)
                            c2_posterior_samples = np.array(c2_posterior_samples)
                            
                            # Generate 3D plots for a subset of angles (to avoid too many plots)
                            n_angles = c2_exp.shape[0]
                            angle_indices = np.linspace(0, n_angles-1, min(5, n_angles), dtype=int)
                            
                            successful_3d_plots = 0
                            for angle_idx in angle_indices:
                                angle_deg = phi_angles[angle_idx] if angle_idx < len(phi_angles) else angle_idx
                                
                                # Extract data for this angle
                                c2_exp_angle = c2_exp[angle_idx]  # Shape: (n_t2, n_t1)
                                c2_fitted_angle = c2_theory[angle_idx]  # Shape: (n_t2, n_t1)  
                                c2_samples_angle = c2_posterior_samples[:, angle_idx, :, :]  # Shape: (n_samples, n_t2, n_t1)
                                
                                # Create 3D surface plot with confidence intervals
                                success = plot_3d_surface(
                                    c2_experimental=c2_exp_angle,
                                    c2_fitted=c2_fitted_angle,
                                    posterior_samples=c2_samples_angle,
                                    phi_angle=angle_deg,
                                    outdir=mcmc_dir,
                                    config=config,
                                    t2=t2,
                                    t1=t1,
                                    confidence_level=0.95
                                )
                                
                                if success:
                                    successful_3d_plots += 1
                            
                            if successful_3d_plots > 0:
                                logger.info(f"✓ Generated {successful_3d_plots} 3D surface plots with confidence intervals")
                            else:
                                logger.warning("⚠ No 3D surface plots were generated successfully")
                        
                        else:
                            logger.warning("No parameter samples found in MCMC trace")
                    
                    else:
                        logger.warning("MCMC trace does not contain posterior data")
                        
                except ImportError:
                    logger.warning("ArviZ not available - skipping 3D plots with confidence intervals")
                except Exception as e:
                    logger.warning(f"Failed to process MCMC samples for 3D plotting: {e}")
                    
            else:
                logger.info("No MCMC trace available - generating 3D plots without confidence intervals")
                # Generate basic 3D plots without confidence intervals
                n_angles = c2_exp.shape[0]
                angle_indices = np.linspace(0, n_angles-1, min(3, n_angles), dtype=int)
                
                successful_3d_plots = 0
                for angle_idx in angle_indices:
                    angle_deg = phi_angles[angle_idx] if angle_idx < len(phi_angles) else angle_idx
                    
                    success = plot_3d_surface(
                        c2_experimental=c2_exp[angle_idx],
                        c2_fitted=c2_theory[angle_idx],
                        posterior_samples=None,  # No confidence intervals
                        phi_angle=angle_deg,
                        outdir=mcmc_dir,
                        config=config,
                        t2=t2,
                        t1=t1
                    )
                    
                    if success:
                        successful_3d_plots += 1
                
                if successful_3d_plots > 0:
                    logger.info(f"✓ Generated {successful_3d_plots} basic 3D surface plots")
                        
        except Exception as e:
            logger.error(f"Failed to generate 3D surface plots: {e}")

        # Generate MCMC-specific plots (trace plots, corner plots, etc.)
        try:
            from homodyne.plotting import create_all_plots
            
            # Prepare results data for plotting
            plot_data = {
                "experimental_data": c2_exp,
                "theoretical_data": c2_theory,
                "phi_angles": phi_angles,
                "best_parameters": dict(zip(
                    config.get("initial_parameters", {}).get("parameter_names", []),
                    best_params
                )),
                "parameter_names": config.get("initial_parameters", {}).get("parameter_names", []),
                "parameter_units": config.get("initial_parameters", {}).get("units", []),
                "mcmc_trace": mcmc_results.get("trace"),
                "mcmc_diagnostics": mcmc_results.get("diagnostics", {}),
                "method": "MCMC",
            }
            
            logger.info("Generating MCMC-specific plots (trace, corner, etc.)...")
            plot_status = create_all_plots(plot_data, mcmc_dir, config)
            
            successful_plots = sum(1 for status in plot_status.values() if status)
            if successful_plots > 0:
                logger.info(f"✓ Generated {successful_plots} MCMC plots successfully")
            else:
                logger.warning("⚠ No MCMC plots were generated successfully")
                
        except Exception as e:
            logger.error(f"Failed to generate MCMC-specific plots: {e}")
        
    except Exception as e:
        logger.error(f"Failed to generate MCMC plots: {e}")
        import traceback
        logger.debug(f"Full traceback: {traceback.format_exc()}")


def main():
    """
    Command-line entry point for homodyne scattering analysis.

    Provides a complete interface for XPCS analysis under nonequilibrium
    conditions, supporting both static and laminar flow analysis modes
    with classical and Bayesian optimization approaches.
    """
    parser = argparse.ArgumentParser(
        description="Run homodyne scattering analysis for XPCS under nonequilibrium conditions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                                    # Run with default classical method
  %(prog)s --method all --verbose             # Run all methods with debug logging  
  %(prog)s --config my_config.json            # Use custom config file
  %(prog)s --output-dir ./results --verbose   # Custom output directory with verbose logging
  %(prog)s --static-isotropic                 # Force static mode (zero shear, 3 parameters)
  %(prog)s --laminar-flow --method mcmc       # Force laminar flow mode with MCMC
  %(prog)s --static-isotropic --method all    # Run all methods in static mode

Method Quality Assessment:
  Classical: Uses chi-squared goodness-of-fit (lower is better)
  MCMC:      Uses convergence diagnostics (R̂ < 1.1, ESS > 100 for acceptable quality)
  
  Note: When running --method all, results use different evaluation criteria.
        Do not directly compare chi-squared values to convergence diagnostics.
        """,
    )

    parser.add_argument(
        "--method",
        choices=["classical", "mcmc", "all"],
        default="classical",
        help="Analysis method to use (default: %(default)s)",
    )

    parser.add_argument(
        "--config",
        type=Path,
        default="./homodyne_config.json",
        help="Path to configuration file (default: %(default)s)",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default="./homodyne_results",
        help="Output directory for results (default: %(default)s)",
    )

    parser.add_argument(
        "--verbose", action="store_true", help="Enable verbose DEBUG logging"
    )

    # Add analysis mode selection
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--static",
        action="store_true",
        help="Force static anisotropic mode analysis (3 parameters, with angle selection) [deprecated: use --static-anisotropic]",
    )
    mode_group.add_argument(
        "--static-isotropic",
        action="store_true",
        help="Force static isotropic mode analysis (3 parameters, no angle selection)",
    )
    mode_group.add_argument(
        "--static-anisotropic",
        action="store_true",
        help="Force static anisotropic mode analysis (3 parameters, with angle selection)",
    )
    mode_group.add_argument(
        "--laminar-flow",
        action="store_true",
        help="Force laminar flow mode analysis (7 parameters: all diffusion and shear parameters)",
    )

    parser.add_argument(
        "--plot-experimental-data",
        action="store_true",
        help="Generate validation plots of experimental data after loading for quality checking",
    )

    args = parser.parse_args()

    # Setup logging and prepare output directory
    setup_logging(args.verbose, args.output_dir)

    # Create logger for this module
    logger = logging.getLogger(__name__)

    # Print informative banner
    print_banner(args)

    # Log the configuration
    logger.info(f"Homodyne analysis starting with method: {args.method}")
    logger.info(f"Configuration file: {args.config}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Log file: {args.output_dir / 'run.log'}")

    # Log analysis mode selection
    if args.static:
        logger.info(
            "Command-line mode: static anisotropic (3 parameters, with angle selection)"
        )
        logger.warning("Note: --static is deprecated, use --static-anisotropic instead")
    elif args.static_isotropic:
        logger.info(
            "Command-line mode: static isotropic (3 parameters, no angle selection)"
        )
    elif args.static_anisotropic:
        logger.info(
            "Command-line mode: static anisotropic (3 parameters, with angle selection)"
        )
    elif args.laminar_flow:
        logger.info("Command-line mode: laminar flow (7 parameters)")
    else:
        logger.info("Analysis mode: from configuration file")

    # Run the analysis
    try:
        run_analysis(args)
        print()
        print("✓ Analysis completed successfully!")
        print(f"Results saved to: {args.output_dir}")
        # Exit with code 0 - success
        sys.exit(0)
    except SystemExit:
        # Re-raise SystemExit to preserve exit code
        raise
    except Exception as e:
        logger.error(f"❌ Analysis failed: {e}")
        logger.error(
            "Please check your configuration and ensure all dependencies are installed"
        )
        # Exit with non-zero code - failure
        sys.exit(1)


if __name__ == "__main__":
    main()
