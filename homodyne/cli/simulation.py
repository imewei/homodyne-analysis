"""
CLI Simulation Module
=====================

Data simulation and plotting functionality for the homodyne CLI interface.

This module handles the generation of theoretical data, simulation plotting,
and visualization of simulated correlation functions for analysis validation
and demonstration purposes.
"""

import argparse
import json
import logging
import sys
import tempfile
from pathlib import Path

import numpy as np

# Module-level logger
logger = logging.getLogger(__name__)


def get_default_simulation_config():
    """
    Get default configuration parameters for simulation fallback.

    Returns
    -------
    tuple
        (DEFAULT_INITIAL_PARAMS, DEFAULT_PARAM_NAMES, DEFAULT_ANALYZER_CONFIG)
    """
    DEFAULT_INITIAL_PARAMS = np.array([100.0, 0.0, 10.0, 1.0, 0.0, 0.0, 0.0])
    DEFAULT_PARAM_NAMES = [
        "D0",
        "alpha",
        "D_offset",
        "gamma_dot_t0",
        "beta",
        "gamma_dot_t_offset",
        "phi0",
    ]
    DEFAULT_TEMPORAL_CONFIG = {"dt": 0.1, "start_frame": 1, "end_frame": 100}
    DEFAULT_ANALYZER_CONFIG = {
        "temporal": DEFAULT_TEMPORAL_CONFIG,
        "scattering": {"wavevector_q": 0.01},
        "geometry": {"stator_rotor_gap": 2000000},  # 200 μm in Angstroms
    }
    return DEFAULT_INITIAL_PARAMS, DEFAULT_PARAM_NAMES, DEFAULT_ANALYZER_CONFIG


def create_simulation_config_override(args: argparse.Namespace) -> dict:
    """
    Create configuration overrides based on command-line arguments.

    Parameters
    ----------
    args : argparse.Namespace
        Command-line arguments

    Returns
    -------
    dict
        Configuration override dictionary
    """
    config_override = {}
    if args.static_isotropic:
        config_override["analysis_settings"] = {
            "static_mode": True,
            "isotropic_mode": True,
        }
    elif args.static_anisotropic:
        config_override["analysis_settings"] = {
            "static_mode": True,
            "isotropic_mode": False,
        }
    elif args.laminar_flow:
        config_override["analysis_settings"] = {"static_mode": False}
    return config_override


def initialize_analysis_core_for_simulation(args: argparse.Namespace):
    """
    Initialize analysis core with configuration or fallback to defaults.

    Parameters
    ----------
    args : argparse.Namespace
        Command-line arguments

    Returns
    -------
    tuple
        (core, config, initial_params) or raises exception on failure
    """
    DEFAULT_INITIAL_PARAMS, DEFAULT_PARAM_NAMES, DEFAULT_ANALYZER_CONFIG = (
        get_default_simulation_config()
    )

    # Import here to avoid circular imports
    from ..analysis.core import HomodyneAnalysisCore

    # Try to initialize analysis core with configuration
    try:
        # Check if config file exists
        if not Path(args.config).exists():
            logger.warning(f"Configuration file not found: {args.config}")
            logger.info("Using default parameters for simulation")
            return create_default_analysis_core(
                args,
                DEFAULT_INITIAL_PARAMS,
                DEFAULT_PARAM_NAMES,
                DEFAULT_ANALYZER_CONFIG,
            )

        # Apply command-line mode overrides
        config_override = create_simulation_config_override(args)

        if config_override:
            core = HomodyneAnalysisCore(str(args.config), config_override)
            logger.info(f"Applied command-line mode override: {config_override}")
        else:
            core = HomodyneAnalysisCore(str(args.config))

        # Get configuration and parameters from core
        config = core.config_manager.config
        if config is None or "initial_parameters" not in config:
            raise ValueError("Configuration does not contain initial_parameters")
        initial_params = np.array(config["initial_parameters"]["values"])
        logger.info(f"Using initial parameters from config: {initial_params}")

        return core, config, initial_params

    except Exception as e:
        logger.warning(f"Failed to initialize analysis core with config: {e}")
        logger.info("Falling back to default parameters for simulation")
        return create_default_analysis_core(
            args, DEFAULT_INITIAL_PARAMS, DEFAULT_PARAM_NAMES, DEFAULT_ANALYZER_CONFIG
        )


def create_default_analysis_core(
    args: argparse.Namespace,
    DEFAULT_INITIAL_PARAMS,
    DEFAULT_PARAM_NAMES,
    DEFAULT_ANALYZER_CONFIG,
):
    """
    Create analysis core with default configuration.

    Parameters
    ----------
    args : argparse.Namespace
        Command-line arguments
    DEFAULT_INITIAL_PARAMS : np.ndarray
        Default parameter values
    DEFAULT_PARAM_NAMES : list
        Default parameter names
    DEFAULT_ANALYZER_CONFIG : dict
        Default analyzer configuration

    Returns
    -------
    tuple
        (core, config, initial_params)
    """
    # Import here to avoid circular imports
    from ..analysis.core import HomodyneAnalysisCore

    # Create minimal configuration for simulation
    config = {
        "analyzer_parameters": DEFAULT_ANALYZER_CONFIG,
        "initial_parameters": {
            "values": DEFAULT_INITIAL_PARAMS.tolist(),
            "parameter_names": DEFAULT_PARAM_NAMES,
        },
    }
    initial_params = DEFAULT_INITIAL_PARAMS
    logger.info(f"Using default initial parameters: {initial_params}")

    # Create a minimal core for calculation
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        temp_config = {
            "metadata": {"config_version": "0.6.5.dev0"},
            "analyzer_parameters": DEFAULT_ANALYZER_CONFIG,
            "experimental_data": {
                "data_folder_path": "./data/",
                "data_file_name": "dummy.hdf",
                "phi_angles_path": "./data/",
                "phi_angles_file": "phi_list.txt",
            },
            "optimization_config": {
                "classical_optimization": {"methods": ["Nelder-Mead"]}
            },
            "initial_parameters": {
                "values": DEFAULT_INITIAL_PARAMS.tolist(),
                "parameter_names": DEFAULT_PARAM_NAMES,
            },
            "parameter_space": {
                "bounds": [
                    {"name": "D0", "min": 1e-3, "max": 1e6},
                    {"name": "alpha", "min": -2.0, "max": 2.0},
                    {"name": "D_offset", "min": -5000, "max": 5000},
                    {"name": "gamma_dot_t0", "min": 1e-6, "max": 1.0},
                    {"name": "beta", "min": -2.0, "max": 2.0},
                    {
                        "name": "gamma_dot_t_offset",
                        "min": -0.1,
                        "max": 0.1,
                    },
                    {"name": "phi0", "min": -15.0, "max": 15.0},
                ]
            },
        }

        # Apply command-line mode overrides
        config_override = create_simulation_config_override(args)
        if config_override:
            temp_config.update(config_override)

        json.dump(temp_config, f)
        temp_config_path = f.name

    try:
        core = HomodyneAnalysisCore(temp_config_path)
        logger.info("Created analysis core with default configuration")
        return core, config, initial_params
    except Exception as e:
        logger.error(f"Failed to create default analysis core: {e}")
        logger.error("Cannot proceed with simulation")
        sys.exit(1)
    finally:
        # Clean up temporary file
        Path(temp_config_path).unlink(missing_ok=True)


def create_phi_angles_for_simulation(args: argparse.Namespace) -> np.ndarray:
    """
    Create phi angles array for simulation based on command-line arguments.

    Parameters
    ----------
    args : argparse.Namespace
        Command-line arguments

    Returns
    -------
    np.ndarray
        Array of phi angles in degrees
    """
    if args.phi_angles is not None:
        # Parse custom phi angles from command line
        try:
            phi_angles = np.array(
                [float(x.strip()) for x in args.phi_angles.split(",")]
            )
            logger.info(f"Using custom phi angles: {phi_angles}")
        except ValueError as e:
            logger.error(f"❌ Invalid phi angles format: {args.phi_angles}")
            logger.error(f"Expected comma-separated numbers, got error: {e}")
            sys.exit(1)
    else:
        # Use default phi angles
        phi_angles = np.array([0, 30, 45, 60, 90, 120, 135, 150])
        logger.info(f"Using default phi angles: {phi_angles}")

    return phi_angles


def create_time_arrays_for_simulation(
    config: dict,
) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Create time arrays for simulation based on configuration.

    Parameters
    ----------
    config : dict
        Configuration dictionary

    Returns
    -------
    tuple
        (t1, t2, n_time) - time arrays and number of time points
    """
    # Extract temporal configuration
    temporal_config = config.get("analyzer_parameters", {}).get("temporal", {})
    dt = temporal_config.get("dt", 0.1)
    start_frame = temporal_config.get("start_frame", 1)
    end_frame = temporal_config.get("end_frame", 100)
    # Match core.py convention: time_length = end_frame - start_frame + 1 (inclusive counting)
    n_time = end_frame - start_frame + 1

    logger.info("Simulation temporal parameters:")
    logger.info(f"  dt: {dt}")
    logger.info(f"  start_frame: {start_frame}")
    logger.info(f"  end_frame: {end_frame}")
    logger.info(f"  n_time: {n_time}")

    # Create time arrays starting from 0 for plotting: np.linspace(0, dt*(n_time-1), n_time)
    t_values = np.linspace(0, dt * (n_time - 1), n_time)
    t1, t2 = np.meshgrid(t_values, t_values, indexing="ij")

    return t1, t2, n_time


def generate_theoretical_c2_data(
    core, initial_params: np.ndarray, phi_angles: np.ndarray, n_time: int
) -> np.ndarray:
    """
    Generate theoretical C2 data using the analysis core.

    Parameters
    ----------
    core : HomodyneAnalysisCore
        Analysis core for calculations
    initial_params : np.ndarray
        Initial parameter values
    phi_angles : np.ndarray
        Array of phi angles
    n_time : int
        Number of time points

    Returns
    -------
    np.ndarray
        Theoretical C2 data array with shape (n_phi, n_time, n_time)
    """
    try:
        logger.info(
            f"Generating theoretical C2 data for {len(phi_angles)} phi angles..."
        )

        # Use the core's method to calculate theoretical C2 directly
        # This is much cleaner than the previous workaround approach
        c2_theoretical = core.calculate_c2_nonequilibrium_laminar_parallel(
            initial_params, phi_angles
        )

        logger.info("✓ Theoretical C2 data generated successfully")
        logger.info(f"  Shape: {c2_theoretical.shape}")
        logger.info(
            f"  Data range: [{c2_theoretical.min():.6f}, {c2_theoretical.max():.6f}]"
        )

        return c2_theoretical

    except Exception as e:
        logger.error(f"❌ Failed to generate theoretical C2 data: {e}")
        logger.error("Using fallback dummy data")
        # Return dummy data as fallback
        return np.ones((len(phi_angles), n_time, n_time))


def apply_scaling_transformation(
    c2_theoretical: np.ndarray, args: argparse.Namespace
) -> tuple[np.ndarray, str]:
    """
    Apply scaling transformation to theoretical data.

    Parameters
    ----------
    c2_theoretical : np.ndarray
        Theoretical C2 data
    args : argparse.Namespace
        Command-line arguments with contrast and offset

    Returns
    -------
    tuple
        (c2_plot_data, data_type) - transformed data and type label
    """
    logger.info(
        f"Applying scaling transformation: fitted = {args.contrast} * theory + {
            args.offset
        }"
    )
    c2_fitted = args.contrast * c2_theoretical + args.offset
    c2_plot_data = c2_fitted

    # Determine data type and logging based on whether scaling is meaningful
    if args.contrast == 1.0 and args.offset == 0.0:
        data_type = "theoretical"
        logger.info(
            "✓ Default scaling applied (contrast=1.0, offset=0.0): equivalent to theoretical data"
        )
    else:
        data_type = "fitted"
        logger.info("✓ Custom scaling transformation applied successfully")

    return c2_plot_data, data_type


def plot_simulated_data(args: argparse.Namespace) -> None:
    """
    Generate and plot simulated homodyne correlation data.

    Creates theoretical C2 correlation functions using the analysis engine and
    generates visualization plots for different phi angles. Supports custom
    scaling transformations and phi angle selections.

    Parameters
    ----------
    args : argparse.Namespace
        Command-line arguments including config path, output directory,
        and plotting parameters (contrast, offset, phi_angles)
    """
    logger.info("=" * 60)
    logger.info("           SIMULATED DATA PLOTTING MODE")
    logger.info("=" * 60)
    logger.info("")

    # Initialize analysis core
    try:
        core, config, initial_params = initialize_analysis_core_for_simulation(args)
    except Exception as e:
        logger.error(f"❌ Failed to initialize analysis core: {e}")
        sys.exit(1)

    # Create phi angles array
    phi_angles = create_phi_angles_for_simulation(args)

    # Create time arrays
    t1, t2, n_time = create_time_arrays_for_simulation(config)

    # Generate theoretical C2 data
    c2_theoretical = generate_theoretical_c2_data(
        core, initial_params, phi_angles, n_time
    )

    # Apply scaling transformation
    c2_plot_data, data_type = apply_scaling_transformation(c2_theoretical, args)

    # Create output directory
    simulated_dir = args.output_dir / "simulated_data"
    simulated_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving simulated data to: {simulated_dir}")

    # Generate heatmap plots
    try:
        from .visualization import generate_c2_heatmap_plots

        plot_count = generate_c2_heatmap_plots(
            c2_plot_data, phi_angles, t1, t2, data_type, args, simulated_dir
        )
        logger.info(f"✓ Generated {plot_count} heatmap plots")
    except Exception as e:
        logger.error(f"❌ Failed to generate heatmap plots: {e}")

    # Save simulation data
    try:
        save_simulation_data(
            c2_plot_data,
            phi_angles,
            t1,
            t2,
            initial_params,
            config,
            data_type,
            args,
            simulated_dir,
        )
    except Exception as e:
        logger.error(f"❌ Failed to save simulation data: {e}")

    # Print summary
    print_simulation_summary(
        phi_angles, n_time, initial_params, config, data_type, args, simulated_dir
    )


def save_simulation_data(
    c2_plot_data: np.ndarray,
    phi_angles: np.ndarray,
    t1: np.ndarray,
    t2: np.ndarray,
    initial_params: np.ndarray,
    config: dict,
    data_type: str,
    args: argparse.Namespace,
    simulated_dir: Path,
) -> None:
    """
    Save simulation data to files.

    Parameters
    ----------
    c2_plot_data : np.ndarray
        C2 correlation data
    phi_angles : np.ndarray
        Array of phi angles
    t1, t2 : np.ndarray
        Time arrays
    initial_params : np.ndarray
        Parameter values used
    config : dict
        Configuration dictionary
    data_type : str
        Type of data ("theoretical" or "fitted")
    args : argparse.Namespace
        Command-line arguments
    simulated_dir : Path
        Output directory for simulation files
    """
    # Save C2 data in NumPy format
    data_file = simulated_dir / f"c2_{data_type}_data.npz"
    np.savez_compressed(
        data_file,
        c2_data=c2_plot_data,
        phi_angles=phi_angles,
        t1=t1,
        t2=t2,
        initial_params=initial_params,
        contrast=args.contrast,
        offset=args.offset,
    )
    logger.info(f"✓ Saved C2 data to: {data_file}")

    # Save configuration
    config_file = simulated_dir / f"simulation_config_{data_type}.json"
    simulation_metadata = {
        "command_line_args": {
            "contrast": args.contrast,
            "offset": args.offset,
            "phi_angles": args.phi_angles,
            "static_isotropic": args.static_isotropic,
            "static_anisotropic": args.static_anisotropic,
            "laminar_flow": args.laminar_flow,
        },
        "parameters": {
            "values": initial_params.tolist(),
            "names": config.get("initial_parameters", {}).get("parameter_names", []),
        },
        "data_type": data_type,
        "phi_angles": phi_angles.tolist(),
        "time_points": t1.shape[0],
        "scaling": {"contrast": args.contrast, "offset": args.offset},
    }

    with open(config_file, "w") as f:
        json.dump(simulation_metadata, f, indent=2)
    logger.info(f"✓ Saved configuration to: {config_file}")


def print_simulation_summary(
    phi_angles: np.ndarray,
    n_time: int,
    initial_params: np.ndarray,
    config: dict,
    data_type: str,
    args: argparse.Namespace,
    simulated_dir: Path,
) -> None:
    """
    Print simulation summary information.

    Parameters
    ----------
    phi_angles : np.ndarray
        Array of phi angles used
    n_time : int
        Number of time points
    initial_params : np.ndarray
        Parameter values used
    config : dict
        Configuration dictionary
    data_type : str
        Type of data generated
    args : argparse.Namespace
        Command-line arguments
    simulated_dir : Path
        Output directory
    """
    logger.info("=" * 60)
    logger.info("              SIMULATION SUMMARY")
    logger.info("=" * 60)
    logger.info("")
    logger.info(f"Data type:        {data_type}")
    logger.info(f"Phi angles:       {phi_angles} degrees")
    logger.info(f"Time points:      {n_time} x {n_time}")
    logger.info(f"Contrast:         {args.contrast}")
    logger.info(f"Offset:           {args.offset}")
    logger.info("")
    logger.info("Parameters used:")
    param_names = config.get("initial_parameters", {}).get("parameter_names", [])
    for i, param in enumerate(initial_params):
        name = param_names[i] if i < len(param_names) else f"p{i}"
        logger.info(f"  {name}: {param:.6f}")
    logger.info("")
    logger.info(f"Output directory: {simulated_dir}")
    logger.info("=" * 60)
