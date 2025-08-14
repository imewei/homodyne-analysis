"""
Homodyne Analysis Runner

A command-line interface for running homodyne analysis with various methods.
"""

import argparse
import logging
import os
import sys
import time
import json
from pathlib import Path
import numpy as np

# Add the homodyne package to the path if needed
sys.path.insert(0, './homodyne')

# Import homodyne modules (will be used when implementing actual analysis logic)
try:
    from homodyne.analysis.core import HomodyneAnalysisCore
    from homodyne.optimization.classical import ClassicalOptimizer
    from homodyne.optimization.bayesian import BayesianOptimizer
except ImportError:
    # Modules not available yet, will add TODO for implementation
    HomodyneAnalysisCore = None
    ClassicalOptimizer = None
    BayesianOptimizer = None

# Try to import MCMC sampler
try:
    from homodyne.optimization.mcmc import create_mcmc_sampler
    MCMC_AVAILABLE = True
except ImportError:
    create_mcmc_sampler = None
    MCMC_AVAILABLE = False


def setup_logging(verbose: bool, output_dir: Path) -> None:
    """Configure logging based on verbosity level and add file handler."""
    # 1. Prepare output directory & logging
    os.makedirs(output_dir, exist_ok=True)

    # 2. Configure logging with level INFO or DEBUG (if --verbose)
    level = logging.DEBUG if verbose else logging.INFO

    # Create formatter
    formatter = logging.Formatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
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
    log_file_path = output_dir / 'run.log'
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)


def print_banner(args: argparse.Namespace) -> None:
    """Print an informative banner with selected options."""
    print("=" * 60)
    print("            HOMODYNE ANALYSIS RUNNER")
    print("=" * 60)
    print()
    print(f"Method:           {args.method}")
    print(f"Config file:      {args.config}")
    print(f"Output directory: {args.output_dir}")
    print(
        f"Verbose logging:  {'Enabled (DEBUG)' if args.verbose else 'Disabled (INFO)'}")
    
    # Show analysis mode
    if args.static:
        print(f"Analysis mode:    Static (zero shear, 3 parameters)")
    elif args.laminar_flow:
        print(f"Analysis mode:    Laminar flow (7 parameters)")
    else:
        print(f"Analysis mode:    From configuration file")
    
    print()
    print("Starting analysis...")
    print("-" * 60)


def run_analysis(args: argparse.Namespace) -> None:
    """Run the homodyne analysis based on the selected method."""
    logger = logging.getLogger(__name__)

    # Step 2: Load configuration & create analysis core

    # 1. Verify the config file exists; exit with clear error if not
    config_path = Path(args.config)
    if not config_path.exists():
        logger.error(
            f"❌ Configuration file not found: {config_path.absolute()}")
        logger.error(
            "Please check the file path and ensure the configuration file exists.")
        sys.exit(1)

    if not config_path.is_file():
        logger.error(
            f"❌ Configuration path is not a file: {config_path.absolute()}")
        sys.exit(1)

    logger.info(f"✓ Configuration file found: {config_path.absolute()}")

    # 2. Import HomodyneAnalysisCore and create instance
    try:
        from homodyne.analysis.core import HomodyneAnalysisCore
        logger.info("✓ Successfully imported HomodyneAnalysisCore")
    except ImportError as e:
        logger.error(f"❌ Failed to import HomodyneAnalysisCore: {e}")
        logger.error(
            "Please ensure the homodyne package is properly installed.")
        sys.exit(1)

    # 3. Create analysis core instance with error handling
    try:
        logger.info(
            f"Initializing Homodyne Analysis with config: {config_path}")
        
        # Apply mode override if specified
        config_override = None
        if args.static:
            config_override = {"analysis_settings": {"static_mode": True}}
            logger.info("Using command-line override: static mode (3 parameters)")
        elif args.laminar_flow:
            config_override = {"analysis_settings": {"static_mode": False}}
            logger.info("Using command-line override: laminar flow mode (7 parameters)")
        
        analyzer = HomodyneAnalysisCore(config_file=str(config_path), config_override=config_override)
        logger.info("✓ HomodyneAnalysisCore initialized successfully")
        
        # Log the actual analysis mode being used
        analysis_mode = analyzer.config_manager.get_analysis_mode()
        param_count = analyzer.config_manager.get_effective_parameter_count()
        logger.info(f"Analysis mode: {analysis_mode} ({param_count} parameters)")
    except (ImportError, ModuleNotFoundError) as e:
        logger.error(
            f"❌ Import error while creating HomodyneAnalysisCore: {e}")
        logger.error("Please ensure all required dependencies are installed.")
        sys.exit(1)
    except (ValueError, KeyError, FileNotFoundError) as e:
        logger.error(f"❌ JSON configuration error: {e}")
        logger.error(
            "Please check your configuration file format and content.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Unexpected error initializing analysis core: {e}")
        logger.error("Please check your configuration and try again.")
        sys.exit(1)

    # Load experimental data
    logger.info("Loading experimental data...")
    c2_exp, time_length, phi_angles, num_angles = analyzer.load_experimental_data()

    # Get initial parameters from config
    if analyzer.config is None:
        logger.error("❌ Analyzer configuration is None. Please check your configuration file and "
                     "ensure it is loaded correctly.")
        sys.exit(1)
    initial_params = analyzer.config.get(
        "initial_parameters", {}).get("values", None)
    if initial_params is None:
        logger.error(
            "❌ Initial parameters not found in configuration. Please check your configuration file format.")
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
        if args.method == 'classical':
            methods_attempted = ['Classical']
            results = run_classical_optimization(
                analyzer, initial_params, phi_angles, c2_exp)
        elif args.method == 'bayesian':
            methods_attempted = ['Bayesian']
            results = run_bayesian_optimization(
                analyzer, initial_params, phi_angles, c2_exp)
        elif args.method == 'mcmc':
            methods_attempted = ['MCMC']
            results = run_mcmc_optimization(
                analyzer, initial_params, phi_angles, c2_exp, args.output_dir)
        elif args.method == 'all':
            methods_attempted = ['Classical', 'Bayesian', 'MCMC']
            results = run_all_methods(
                analyzer, initial_params, phi_angles, c2_exp, args.output_dir)

        if results:
            # Save results
            analyzer.save_results_with_config(results)

            # Perform per-angle chi-squared analysis for each successful method
            successful_methods = results.get("methods_used", [])
            logger.info(
                f"Running per-angle chi-squared analysis for methods: {', '.join(successful_methods)}")

            for method in successful_methods:
                method_key = f"{method.lower()}_optimization"
                if method_key in results and "parameters" in results[method_key]:
                    method_params = results[method_key]["parameters"]
                    if method_params is not None:
                        try:
                            analyzer.analyze_per_angle_chi_squared(
                                np.array(method_params),
                                phi_angles,
                                c2_exp,
                                method_name=method,
                                save_to_file=True,
                                output_dir=args.output_dir
                            )
                        except Exception as e:
                            logger.warning(
                                f"Per-angle analysis failed for {method}: {e}")

            logger.info("✓ Analysis completed successfully!")
            logger.info(f"Successful methods: {', '.join(successful_methods)}")
        else:
            logger.error("❌ Analysis failed - no results generated")
            if len(methods_attempted) == 1:
                # Single method failed - this is a hard failure
                logger.error(
                    f"The only requested method ({args.method}) failed to complete")
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


def run_classical_optimization(analyzer, initial_params, phi_angles, c2_exp):
    """Run classical optimization method."""
    logger = logging.getLogger(__name__)
    logger.info("Running classical optimization...")

    try:
        if ClassicalOptimizer is None:
            logger.error("❌ ClassicalOptimizer is not available. Please ensure the "
                         "homodyne.optimization.classical module is installed and accessible.")
            return None

        optimizer = ClassicalOptimizer(analyzer, analyzer.config)
        best_params, result = optimizer.run_classical_optimization_optimized(
            initial_parameters=initial_params,
            phi_angles=phi_angles,
            c2_experimental=c2_exp
        )

        return {
            "classical_optimization": {
                "parameters": best_params,
                "chi_squared": result.fun,
                "optimization_time": getattr(result, 'execution_time', 0),
                "total_time": 0,
                "success": result.success,
                "method": getattr(result, 'method', 'unknown'),
                "iterations": getattr(result, 'nit', None),
                "function_evaluations": getattr(result, 'nfev', None)
            },
            "best_overall": {
                "parameters": best_params,
                "chi_squared": result.fun,
                "method": "Classical"
            },
            "methods_used": ["Classical"]
        }
    except ImportError as e:
        error_msg = f"Classical optimization failed - missing dependencies: {e}"
        logger.error(error_msg)
        if "scipy" in str(e).lower():
            logger.error("❌ Install scipy: pip install scipy")
        elif "numpy" in str(e).lower():
            logger.error("❌ Install numpy: pip install numpy")
        else:
            logger.error(
                "❌ Install required dependencies: pip install scipy numpy")
        return None
    except (ValueError, KeyError) as e:
        error_msg = f"Classical optimization failed - configuration error: {e}"
        logger.error(error_msg)
        logger.error(
            "❌ Please check your configuration file format and parameter bounds")
        return None
    except Exception as e:
        error_msg = f"Classical optimization failed - unexpected error: {e}"
        logger.error(error_msg)
        logger.error("❌ Please check your data files and configuration")
        return None


def run_bayesian_optimization(analyzer, initial_params, phi_angles, c2_exp):
    """Run Bayesian optimization method."""
    logger = logging.getLogger(__name__)
    logger.info("Running Bayesian optimization...")

    try:
        # Check if scikit-optimize is available
        try:
            from homodyne.optimization.bayesian import BayesianOptimizer
        except ImportError as import_err:
            if "skopt" in str(import_err).lower() or "scikit-optimize" in str(import_err).lower():
                logger.error(
                    "❌ Bayesian optimization requires scikit-optimize: pip install scikit-optimize")
            else:
                logger.error(
                    f"❌ Failed to import Bayesian optimizer: {import_err}")
            return None

        # Create and run Bayesian optimizer
        optimizer = BayesianOptimizer(analyzer, analyzer.config)
        bo_results = optimizer.run_bayesian_optimization_skopt_optimized(
            phi_angles=phi_angles,
            c2_experimental=c2_exp,
            x0=initial_params
        )

        if bo_results and "best_params" in bo_results:
            best_params = bo_results["best_params"]
            best_chi2 = bo_results["best_chi_squared"]

            return {
                "bayesian_optimization": {
                    "parameters": best_params,
                    "chi_squared": best_chi2,
                    "optimization_time": bo_results.get("optimization_time", 0),
                    "total_time": bo_results.get("optimization_time", 0),
                    "success": True,
                    "method": "Bayesian_GP",
                    "n_calls": bo_results.get("n_calls", 0),
                    "acquisition_function": bo_results.get("acquisition_function", "EI")
                },
                "best_overall": {
                    "parameters": best_params,
                    "chi_squared": best_chi2,
                    "method": "Bayesian"
                },
                "methods_used": ["Bayesian"]
            }
        else:
            logger.error(
                "Bayesian optimization completed but returned no results")
            return None

    except ImportError as e:
        error_msg = f"Bayesian optimization failed - missing dependencies: {e}"
        logger.error(error_msg)
        if "skopt" in str(e).lower() or "scikit-optimize" in str(e).lower():
            logger.error(
                "❌ Install scikit-optimize: pip install scikit-optimize")
        elif "sklearn" in str(e).lower():
            logger.error("❌ Install scikit-learn: pip install scikit-learn")
        else:
            logger.error(
                "❌ Install required dependencies: pip install scikit-optimize scikit-learn")
        return None
    except (ValueError, KeyError) as e:
        error_msg = f"Bayesian optimization failed - configuration error: {e}"
        logger.error(error_msg)
        logger.error(
            "❌ Please check your configuration file and parameter bounds")
        return None
    except Exception as e:
        error_msg = f"Bayesian optimization failed - unexpected error: {e}"
        logger.error(error_msg)
        logger.error("❌ Please check your data files and configuration")
        return None


def run_mcmc_optimization(analyzer, initial_params, phi_angles, c2_exp, output_dir=None):
    """Run MCMC sampling with warm start."""
    logger = logging.getLogger(__name__)
    logger.info("Running MCMC sampling...")

    # Step 1: Attempt to import create_mcmc_sampler
    try:
        from homodyne.optimization.mcmc import create_mcmc_sampler
        logger.info("✓ Successfully imported create_mcmc_sampler")
    except ImportError as e:
        logger.error(f"❌ Failed to import MCMC module: {e}")
        if "pymc" in str(e).lower() or "pytensor" in str(e).lower() or "arviz" in str(e).lower():
            logger.error(
                "❌ MCMC sampling requires PyMC and ArviZ: pip install pymc arviz")
        else:
            logger.error(
                "❌ Install required dependencies: pip install pymc arviz pytensor")
        return None

    try:
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
            filter_angles_for_optimization=True  # Use angle filtering by default
        )

        mcmc_execution_time = time.time() - mcmc_start_time
        logger.info(
            f"✓ MCMC sampling completed in {mcmc_execution_time:.2f} seconds")

        # Step 5 & 6: Save inference data and write convergence diagnostics
        if output_dir is None:
            output_dir = Path('./homodyne_results')
        else:
            output_dir = Path(output_dir)

        # Create mcmc_results subdirectory
        mcmc_output_dir = output_dir / 'mcmc_results'
        mcmc_output_dir.mkdir(parents=True, exist_ok=True)

        # Save inference data (NetCDF via arviz.to_netcdf) if trace is available
        if 'trace' in mcmc_results and mcmc_results['trace'] is not None:
            try:
                import arviz as az
                netcdf_path = mcmc_output_dir / 'mcmc_trace.nc'
                az.to_netcdf(mcmc_results['trace'], str(netcdf_path))
                logger.info(f"✓ MCMC trace saved to NetCDF: {netcdf_path}")
            except ImportError as import_err:
                logger.error(
                    f"❌ ArviZ not available for saving trace: {import_err}")
                logger.error("❌ Install ArviZ: pip install arviz")
            except Exception as e:
                logger.error(f"❌ Failed to save NetCDF trace: {e}")

        # Prepare summary results for JSON
        summary_results = {
            "method": "MCMC_NUTS",
            "execution_time_seconds": mcmc_execution_time,
            "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
            "posterior_means": mcmc_results.get('posterior_means', {}),
            "mcmc_config": mcmc_results.get('config', {})
        }

        # Add convergence diagnostics to summary
        if 'diagnostics' in mcmc_results:
            diagnostics = mcmc_results['diagnostics']
            summary_results['convergence_diagnostics'] = {
                "max_rhat": diagnostics.get('max_rhat'),
                "min_ess": diagnostics.get('min_ess'),
                "converged": diagnostics.get('converged', False),
                "assessment": diagnostics.get('assessment', 'Unknown')
            }

            # Write convergence diagnostics to log (Step 6)
            logger.info("Convergence Diagnostics:")
            logger.info(f"  Max R-hat: {diagnostics.get('max_rhat', 'N/A')}")
            logger.info(f"  Min ESS: {diagnostics.get('min_ess', 'N/A')}")
            logger.info(f"  Converged: {diagnostics.get('converged', False)}")
            logger.info(
                f"  Assessment: {diagnostics.get('assessment', 'Unknown')}")

            if not diagnostics.get('converged', False):
                logger.warning(
                    "⚠ MCMC chains may not have converged - check diagnostics!")

        # Add posterior statistics if available
        if hasattr(sampler, 'extract_posterior_statistics'):
            try:
                posterior_stats = sampler.extract_posterior_statistics(
                    mcmc_results.get('trace'))
                if posterior_stats and 'parameter_statistics' in posterior_stats:
                    summary_results['parameter_statistics'] = posterior_stats['parameter_statistics']
            except Exception as e:
                logger.warning(f"Failed to extract posterior statistics: {e}")

        # Save summary JSON to output_dir/mcmc_results
        summary_json_path = mcmc_output_dir / 'mcmc_summary.json'
        try:
            with open(summary_json_path, 'w') as f:
                json.dump(summary_results, f, indent=2, default=str)
            logger.info(f"✓ MCMC summary saved to: {summary_json_path}")
        except Exception as e:
            logger.error(f"❌ Failed to save MCMC summary JSON: {e}")

        # Extract best parameters from posterior means for compatibility with other methods
        best_params = None
        if 'posterior_means' in mcmc_results:
            param_names = analyzer.config.get(
                'initial_parameters', {}).get('parameter_names', [])
            posterior_means = mcmc_results['posterior_means']
            best_params = [posterior_means.get(
                name, 0.0) for name in param_names]

        # Calculate chi-squared for best MCMC parameters
        chi_squared = None
        if best_params:
            try:
                chi_squared = analyzer.calculate_chi_squared_optimized(
                    best_params, phi_angles, c2_exp, method_name="MCMC"
                )
                logger.info(f"MCMC posterior mean χ²_red: {chi_squared:.6e}")
            except Exception as e:
                logger.warning(
                    f"Failed to calculate chi-squared for MCMC results: {e}")

        # Format results for compatibility with main analysis framework
        return {
            "mcmc_optimization": {
                "parameters": best_params,
                "chi_squared": chi_squared,
                "optimization_time": mcmc_execution_time,
                "total_time": mcmc_execution_time,
                "success": mcmc_results.get('diagnostics', {}).get('converged', True),
                "method": "MCMC_NUTS",
                "posterior_means": mcmc_results.get('posterior_means', {}),
                "convergence_diagnostics": mcmc_results.get('diagnostics', {})
            },
            "best_overall": {
                "parameters": best_params,
                "chi_squared": chi_squared,
                "method": "MCMC"
            },
            "methods_used": ["MCMC"]
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
                "❌ Install required dependencies: pip install pymc arviz pytensor")
        return None
    except (ValueError, KeyError) as e:
        error_msg = f"MCMC optimization failed - configuration error: {e}"
        logger.error(error_msg)
        logger.error(
            "❌ Please check your MCMC configuration and parameter priors")
        return None
    except Exception as e:
        error_msg = f"MCMC optimization failed - unexpected error: {e}"
        logger.error(error_msg)
        logger.error("❌ Please check your data files and MCMC configuration")
        import traceback
        logger.debug(f"Full traceback: {traceback.format_exc()}")
        return None


def run_all_methods(analyzer, initial_params, phi_angles, c2_exp, output_dir=None):
    """Run all available optimization methods."""
    logger = logging.getLogger(__name__)
    logger.info("Running all optimization methods...")

    all_results = {}
    methods_used = []
    methods_attempted = []

    # Run classical optimization
    methods_attempted.append("Classical")
    logger.info("Attempting Classical optimization...")
    classical_results = run_classical_optimization(
        analyzer, initial_params, phi_angles, c2_exp)
    if classical_results:
        all_results.update(classical_results)
        methods_used.append("Classical")
        logger.info("✓ Classical optimization completed successfully")
    else:
        logger.warning("⚠ Classical optimization failed")

    # Run Bayesian optimization
    methods_attempted.append("Bayesian")
    logger.info("Attempting Bayesian optimization...")
    bayesian_results = run_bayesian_optimization(
        analyzer, initial_params, phi_angles, c2_exp)
    if bayesian_results:
        all_results.update(bayesian_results)
        methods_used.append("Bayesian")
        logger.info("✓ Bayesian optimization completed successfully")
    else:
        logger.warning("⚠ Bayesian optimization failed")

    # Run MCMC sampling
    methods_attempted.append("MCMC")
    logger.info("Attempting MCMC sampling...")
    mcmc_results = run_mcmc_optimization(
        analyzer, initial_params, phi_angles, c2_exp, output_dir)
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
        return all_results

    logger.error("❌ All optimization methods failed")
    return None


def main():
    """Main entry point for the homodyne analysis runner."""
    parser = argparse.ArgumentParser(
        description="Run homodyne analysis with various methods",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                                    # Run with default classical method
  %(prog)s --method bayesian                  # Run Bayesian analysis
  %(prog)s --method all --verbose             # Run all methods with debug logging
  %(prog)s --config my_config.json            # Use custom config file
  %(prog)s --output-dir ./results --verbose   # Custom output directory with verbose logging
  %(prog)s --static                           # Force static mode (zero shear, 3 parameters)
  %(prog)s --laminar-flow --method mcmc       # Force laminar flow mode with MCMC
  %(prog)s --static --method all              # Run all methods in static mode
        """
    )

    parser.add_argument(
        '--method',
        choices=['classical', 'bayesian', 'mcmc', 'all'],
        default='classical',
        help='Analysis method to use (default: %(default)s)'
    )

    parser.add_argument(
        '--config',
        type=Path,
        default='./homodyne_config.json',
        help='Path to configuration file (default: %(default)s)'
    )

    parser.add_argument(
        '--output-dir',
        type=Path,
        default='./homodyne_results',
        help='Output directory for results (default: %(default)s)'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose DEBUG logging'
    )

    # Add analysis mode selection
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        '--static',
        action='store_true',
        help='Force static mode analysis (zero shear, 3 parameters: D₀, α, D_offset)'
    )
    mode_group.add_argument(
        '--laminar-flow',
        action='store_true',
        help='Force laminar flow mode analysis (7 parameters: all diffusion and shear parameters)'
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
        logger.info("Command-line mode: static (zero shear, 3 parameters)")
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
            "Please check your configuration and ensure all dependencies are installed")
        # Exit with non-zero code - failure
        sys.exit(1)


if __name__ == "__main__":
    main()
