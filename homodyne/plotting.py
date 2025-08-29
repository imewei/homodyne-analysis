"""
Plotting Functions for Homodyne Scattering Analysis
===================================================

This module provides specialized plotting functions for visualizing results from
homodyne scattering analysis in XPCS (X-ray Photon Correlation Spectroscopy).

The plotting functions are designed to work with the configuration system and
provide publication-quality plots for:
- C2 correlation function heatmaps with experimental vs theoretical comparison
- Parameter evolution during optimization
- MCMC corner plots for uncertainty quantification

Created for: Rheo-SAXS-XPCS Homodyne Analysis
Authors: Wei Chen, Hongrui He
Institution: Argonne National Laboratory
"""

import logging
from pathlib import Path
from typing import Any

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np

from .core.io_utils import ensure_dir, save_fig

# Set up logging
logger = logging.getLogger(__name__)


# Lazy import functions for optional plotting dependencies
def _lazy_import_plotting_deps():
    """Lazy import plotting dependencies to improve module loading time."""
    imports = {}

    try:
        import arviz as arviz_module

        imports["arviz"] = arviz_module
        logger.debug("ArviZ imported for MCMC corner plots")
    except ImportError:
        imports["arviz"] = None
        logger.debug("ArviZ not available")

    try:
        import corner as corner_module

        imports["corner"] = corner_module
        logger.debug("corner package imported for enhanced corner plots")
    except ImportError:
        imports["corner"] = None
        logger.debug("corner package not available")

    return imports


# Check availability without importing - performance optimization
try:
    import importlib.util

    ARVIZ_AVAILABLE = importlib.util.find_spec("arviz") is not None
    CORNER_AVAILABLE = importlib.util.find_spec("corner") is not None
except ImportError:
    ARVIZ_AVAILABLE = False
    CORNER_AVAILABLE = False

# Global variables for lazy-loaded modules
_plotting_deps = None


def get_plot_config(config: dict | None = None) -> dict[str, Any]:
    """
    Extract plotting configuration from the main config dictionary.

    Args:
        config (Optional[Dict]): Main configuration dictionary

    Returns:
        Dict[str, Any]: Plotting configuration with defaults
    """
    # Default plotting configuration
    default_plot_config = {
        "plot_format": "png",
        "dpi": 300,
        "figure_size": [10, 8],
        "create_plots": True,
    }

    if (
        config
        and "output_settings" in config
        and "plotting" in config["output_settings"]
    ):
        plot_config = {
            **default_plot_config,
            **config["output_settings"]["plotting"],
        }
    else:
        plot_config = default_plot_config
        logger.warning("No plotting configuration found, using defaults")

    return plot_config


def setup_matplotlib_style(plot_config: dict[str, Any]) -> None:
    """
    Configure matplotlib with publication-quality settings.

    Args:
        plot_config (Dict[str, Any]): Plotting configuration
    """
    # Suppress matplotlib font debug messages to reduce log noise
    logging.getLogger("matplotlib.font_manager").setLevel(logging.WARNING)

    plt.rcParams.update(
        {
            "font.size": 12,
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
            "axes.labelsize": 14,
            "axes.titlesize": 16,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 12,
            "figure.titlesize": 18,
            "axes.linewidth": 1.2,
            "xtick.major.width": 1.2,
            "ytick.major.width": 1.2,
            "xtick.minor.width": 0.8,
            "ytick.minor.width": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": plot_config.get("dpi", 100),
            "savefig.dpi": plot_config.get("dpi", 300),
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.1,
        }
    )


def _setup_plot_environment(
    config: dict | None, outdir: str | Path
) -> tuple[dict[str, Any], Path]:
    """Set up common plotting environment and return config and output directory."""
    plot_config = get_plot_config(config)
    setup_matplotlib_style(plot_config)
    output_dir = ensure_dir(outdir)
    return plot_config, output_dir


def _setup_plot_style(config: dict | None) -> dict[str, Any]:
    """Set up plotting style without directory setup."""
    plot_config = get_plot_config(config)
    setup_matplotlib_style(plot_config)
    return plot_config


def _save_plot_with_logging(
    fig: plt.Figure, output_path: Path, plot_type: str, **save_kwargs: Any
) -> None:
    """Save plot with consistent logging and cleanup."""
    try:
        if save_kwargs:
            # Use custom save parameters
            plt.savefig(output_path, **save_kwargs)
        else:
            # Use standard save_fig utility
            save_fig(fig, output_path)
        logger.info(f"✓ {plot_type} plot saved to: {output_path}")
    except Exception as e:
        logger.error(f"❌ Failed to save {plot_type} plot: {e}")
    finally:
        plt.close(fig)


def plot_c2_heatmaps(
    exp: np.ndarray,
    theory: np.ndarray,
    phi_angles: np.ndarray,
    outdir: str | Path,
    config: dict | None = None,
    t2: np.ndarray | None = None,
    t1: np.ndarray | None = None,
    method_name: str | None = None,
) -> bool:
    """
    Create side-by-side heatmaps comparing experimental and theoretical C2 correlation functions,
    plus residuals for each phi angle.

    Args:
        exp (np.ndarray): Experimental correlation data [n_angles, n_t2, n_t1]
        theory (np.ndarray): Theoretical correlation data [n_angles, n_t2, n_t1]
        phi_angles (np.ndarray): Array of phi angles in degrees
        outdir (Union[str, Path]): Output directory for saved plots
        config (Optional[Dict]): Configuration dictionary
        t2 (Optional[np.ndarray]): Time lag values (t₂) for y-axis
        t1 (Optional[np.ndarray]): Delay time values (t₁) for x-axis
        method_name (Optional[str]): Optimization method name for filename prefix

    Returns:
        bool: True if plots were created successfully
    """
    # Validate inputs first
    try:
        phi_angles_len = len(phi_angles) if phi_angles is not None else 0
        logger.info(f"Creating C2 heatmaps for {phi_angles_len} phi angles")
    except TypeError:
        logger.error("Invalid phi_angles parameter - must be array-like")
        return False

    # Set up plotting environment
    plot_config, outdir = _setup_plot_environment(config, outdir)

    # Validate exp and theory inputs
    try:
        if exp is None or not hasattr(exp, "shape"):
            logger.error("Experimental data must be a numpy array with shape attribute")
            return False
        if theory is None or not hasattr(theory, "shape"):
            logger.error("Theoretical data must be a numpy array with shape attribute")
            return False
    except Exception as e:
        logger.error(f"Error validating input arrays: {e}")
        return False

    # Validate input dimensions
    if exp.shape != theory.shape:
        logger.error(f"Shape mismatch: exp {exp.shape} vs theory {theory.shape}")
        return False

    if len(phi_angles) != exp.shape[0]:
        logger.error(
            f"Number of angles ({len(phi_angles)}) doesn't match data shape ({
                exp.shape[0]
            })"
        )
        return False

    # Generate default axes if not provided
    if t2 is None:
        t2 = np.arange(exp.shape[1])
    if t1 is None:
        t1 = np.arange(exp.shape[2])

    # Type assertion to help Pylance understand these are no longer None
    assert t2 is not None and t1 is not None

    # SCALING OPTIMIZATION FOR PLOTTING (ALWAYS ENABLED)
    # ==================================================
    # Calculate fitted values and residuals with proper scaling optimization.
    # This determines the optimal scaling relationship g₂ = offset + contrast × g₁
    # for visualization purposes, ensuring plotted data is meaningful.
    fitted = np.zeros_like(theory)

    # SCALING OPTIMIZATION: ALWAYS PERFORMED
    # This scaling optimization is essential for meaningful plots because:
    # 1. Raw theoretical and experimental data may have different scales
    # 2. Systematic offsets need to be accounted for in visualization
    # 3. Residual plots (exp - fitted) are only meaningful with proper scaling
    # 4. Consistent with chi-squared calculation methodology used in analysis
    # The relationship g₂ = offset + contrast × g₁ is fitted for each angle
    # independently.

    for i in range(exp.shape[0]):  # For each phi angle
        exp_flat = exp[i].flatten()
        theory_flat = theory[i].flatten()

        # Optimal scaling: fitted = theory * contrast + offset
        A = np.vstack([theory_flat, np.ones(len(theory_flat))]).T
        try:
            scaling, _, _, _ = np.linalg.lstsq(A, exp_flat, rcond=None)
            if len(scaling) == 2:
                contrast, offset = scaling
                fitted[i] = theory[i] * contrast + offset
            else:
                fitted[i] = theory[i]
        except np.linalg.LinAlgError:
            fitted[i] = theory[i]

    # Calculate residuals: exp - fitted
    residuals = exp - fitted

    # Create plots for each phi angle
    success_count = 0

    for i, phi in enumerate(phi_angles):
        try:
            # Create figure with single row, 3 columns + 2 colorbars
            fig = plt.figure(
                figsize=(
                    plot_config["figure_size"][0] * 1.5,
                    plot_config["figure_size"][1] * 0.7,
                )
            )
            gs = gridspec.GridSpec(
                1,
                5,
                width_ratios=[1, 1, 1, 0.05, 0.05],
                hspace=0.2,
                wspace=0.3,
            )

            # Calculate appropriate vmin for this angle's data
            angle_data_min = min(np.min(exp[i]), np.min(fitted[i]))
            angle_vmin = 1.0 if angle_data_min >= 1.0 else angle_data_min

            # Experimental data heatmap
            ax1 = fig.add_subplot(gs[0, 0])
            im1 = ax1.imshow(
                exp[i],
                aspect="equal",  # Use square aspect ratio
                origin="lower",
                extent=(
                    float(t1[0]),
                    float(t1[-1]),
                    float(t2[0]),
                    float(t2[-1]),
                ),
                cmap="viridis",
                vmin=angle_vmin,
            )
            ax1.set_title(f"Experimental $C_2$\nφ = {phi:.1f}°")
            ax1.set_xlabel(r"$t_1$")
            ax1.set_ylabel(r"$t_2$")

            # Fitted data heatmap
            ax2 = fig.add_subplot(gs[0, 1])
            im2 = ax2.imshow(
                fitted[i],
                aspect="equal",  # Use square aspect ratio
                origin="lower",
                extent=(
                    float(t1[0]),
                    float(t1[-1]),
                    float(t2[0]),
                    float(t2[-1]),
                ),
                cmap="viridis",
                vmin=angle_vmin,
            )
            ax2.set_title(f"Theoretical $C_2$\nφ = {phi:.1f}°")
            ax2.set_xlabel(r"$t_1$")
            ax2.set_ylabel(r"$t_2$")

            # Residuals heatmap
            ax3 = fig.add_subplot(gs[0, 2])
            im3 = ax3.imshow(
                residuals[i],
                aspect="equal",  # Use square aspect ratio
                origin="lower",
                extent=(
                    float(t1[0]),
                    float(t1[-1]),
                    float(t2[0]),
                    float(t2[-1]),
                ),
                cmap="RdBu_r",
            )
            ax3.set_title(f"Residuals (Exp - Fit)\nφ = {phi:.1f}°")
            ax3.set_xlabel(r"$t_1$")
            ax3.set_ylabel(r"$t_2$")

            # Shared colorbar for exp and theory
            cbar_ax1 = fig.add_subplot(gs[0, 3])
            data_min = min(np.min(exp[i]), np.min(fitted[i]))
            data_max = max(np.max(exp[i]), np.max(fitted[i]))
            # Use the same vmin logic as the imshow calls
            colorbar_vmin = 1.0 if data_min >= 1.0 else data_min
            colorbar_vmax = data_max
            im1.set_clim(colorbar_vmin, colorbar_vmax)
            im2.set_clim(colorbar_vmin, colorbar_vmax)
            plt.colorbar(im1, cax=cbar_ax1, label=r"$C_2$")

            # Residuals colorbar
            cbar_ax2 = fig.add_subplot(gs[0, 4])
            plt.colorbar(im3, cax=cbar_ax2, label="Residual")

            # Add statistics text
            rmse = np.sqrt(np.mean(residuals[i] ** 2))
            mae = np.mean(np.abs(residuals[i]))
            stats_text = f"RMSE: {rmse:.6f}\nMAE: {mae:.6f}"
            ax3.text(
                0.02,
                0.98,
                stats_text,
                transform=ax3.transAxes,
                verticalalignment="top",
                bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8},
            )

            # Save the plot
            # Use simplified filename format when saving in method directories
            if (
                method_name
                and len(str(outdir).split("/")) > 1
                and str(outdir).split("/")[-1]
                in [
                    "nelder_mead",
                    "gurobi",
                    "wasserstein",
                    "scenario",
                    "ellipsoidal",
                ]
            ):
                # Simplified format for method directories:
                # c2_heatmaps_[method_name].png
                if len(phi_angles) == 1:
                    filename = f"c2_heatmaps_{method_name}.{plot_config['plot_format']}"
                else:
                    filename = f"c2_heatmaps_{method_name}_phi_{phi:.1f}deg.{
                        plot_config['plot_format']
                    }"
            else:
                # Original format for backward compatibility
                method_prefix = f"{method_name.lower()}_" if method_name else ""
                filename = f"{method_prefix}c2_heatmaps_phi_{phi:.1f}deg.{
                    plot_config['plot_format']
                }"
            filepath = outdir / filename

            if save_fig(
                fig,
                filepath,
                dpi=plot_config["dpi"],
                format=plot_config["plot_format"],
            ):
                success_count += 1
                logger.info(f"Saved C2 heatmap for φ = {phi:.1f}°")
            else:
                logger.error(f"Failed to save C2 heatmap for φ = {phi:.1f}°")

            plt.close(fig)  # Free memory

        except Exception as e:
            logger.error(f"Error creating C2 heatmap for φ = {phi:.1f}°: {e}")
            plt.close("all")  # Clean up any partial figures

    logger.info(
        f"Successfully created {success_count}/{len(phi_angles)} C2 heatmap plots"
    )
    return success_count == len(phi_angles)


def _get_active_parameters(config: dict | None) -> list[str] | None:
    """Extract active parameter names from config."""
    if (
        config
        and "initial_parameters" in config
        and "active_parameters" in config["initial_parameters"]
    ):
        active_params = config["initial_parameters"]["active_parameters"]
        logger.debug(f"Active parameters for corner plot: {active_params}")
        return active_params
    return None


def _convert_trace_to_samples(
    trace_data: Any, active_param_names: list[str] | None
) -> Any:
    """Convert various trace data formats to samples suitable for plotting."""
    # Validate trace data format first
    if callable(trace_data):
        logger.error(
            "Trace data is a function, not actual data - cannot create corner plot"
        )
        return None

    # Handle different trace data formats
    if hasattr(trace_data, "posterior"):
        # ArviZ InferenceData
        samples = trace_data.posterior

        # Filter to only active parameters if specified
        if active_param_names and hasattr(samples, "data_vars"):
            filtered_vars = {
                var: samples[var]
                for var in active_param_names
                if var in samples.data_vars
            }
            if filtered_vars:
                import xarray as xr

                samples = xr.Dataset(filtered_vars)
                logger.debug(
                    f"Filtered to active parameters: {list(samples.data_vars)}"
                )
        return samples

    elif isinstance(trace_data, dict):
        # Dictionary of samples
        samples = trace_data
        # Filter to only active parameters if specified
        if active_param_names:
            samples = {
                var: samples[var] for var in active_param_names if var in samples
            }
        return samples

    elif isinstance(trace_data, np.ndarray):
        # NumPy array - convert to ArviZ InferenceData
        try:
            if not ARVIZ_AVAILABLE:
                logger.error("ArviZ required for numpy array conversion")
                return None

            import arviz as az

            samples = az.convert_to_inference_data(trace_data).posterior
        except Exception as conversion_error:
            logger.error(
                f"Failed to convert numpy trace to ArviZ format: {conversion_error}"
            )
            return None

    else:
        logger.error(f"Unsupported trace data type: {type(trace_data)}")
        return None

    return samples


def plot_mcmc_corner(
    trace_data: Any,
    outdir: str | Path,
    config: dict | None = None,
    param_names: list[str] | None = None,
    param_units: list[str] | None = None,
    title_prefix: str = "MCMC",
) -> bool:
    """
    Create MCMC corner plot using ArviZ if trace exists.

    Args:
        trace_data: MCMC trace data (ArviZ InferenceData or similar)
        outdir (Union[str, Path]): Output directory for saved plots
        config (Optional[Dict]): Configuration dictionary
        param_names (Optional[List[str]]): Parameter names for labeling
        param_units (Optional[List[str]]): Parameter units for labeling
        title_prefix (str): Prefix for plot title

    Returns:
        bool: True if corner plot was created successfully
    """
    if not ARVIZ_AVAILABLE:
        logger.warning("ArviZ not available - cannot create MCMC corner plot")
        return False

    logger.info("Creating MCMC corner plot")

    # Set up plotting environment
    plot_config, outdir = _setup_plot_environment(config, outdir)

    try:
        # Get active parameters from config to filter out inactive ones
        active_param_names = None
        if (
            config
            and "initial_parameters" in config
            and "active_parameters" in config["initial_parameters"]
        ):
            active_param_names = config["initial_parameters"]["active_parameters"]
            logger.debug(f"Active parameters for corner plot: {active_param_names}")

        # Validate trace data format first
        if callable(trace_data):
            logger.error(
                "Trace data is a function, not actual data - cannot create corner plot"
            )
            return False

        # Handle different trace data formats
        if hasattr(trace_data, "posterior"):
            # ArviZ InferenceData
            samples = trace_data.posterior

            # Filter to only active parameters if specified
            if active_param_names and hasattr(samples, "data_vars"):
                filtered_vars = {
                    var: samples[var]
                    for var in active_param_names
                    if var in samples.data_vars
                }
                if filtered_vars:
                    import xarray as xr

                    samples = xr.Dataset(filtered_vars)
                    logger.debug(
                        f"Filtered to active parameters: {list(samples.data_vars)}"
                    )

        elif isinstance(trace_data, dict):
            # Dictionary of samples
            samples = trace_data

            # Filter to only active parameters if specified
            if active_param_names:
                samples = {
                    var: samples[var] for var in active_param_names if var in samples
                }
                logger.debug(
                    f"Filtered dict to active parameters: {list(samples.keys())}"
                )
        elif isinstance(trace_data, np.ndarray):
            # NumPy array - use directly
            samples = trace_data
        else:
            # Try to convert to DataFrame
            try:
                if not ARVIZ_AVAILABLE:
                    logger.error("Pandas not available for DataFrame conversion")
                    return False
                import pandas as pd  # type: ignore[import]

                samples = pd.DataFrame(trace_data)
            except Exception as conversion_error:
                logger.error(
                    f"Unsupported trace data format for corner plot: {
                        type(trace_data)
                    }, error: {conversion_error}"
                )
                return False

        # Create corner plot using ArviZ
        if hasattr(samples, "stack"):
            # ArviZ format - stack chains
            stacked_samples = samples.stack(sample=("chain", "draw"))  # type: ignore
            logger.debug(f"Stacked ArviZ samples: {type(stacked_samples)}")
            if hasattr(stacked_samples, "data_vars"):
                logger.debug(f"Available variables: {list(stacked_samples.data_vars)}")
            if hasattr(stacked_samples, "dims"):
                logger.debug(f"Dimensions: {stacked_samples.dims}")
        else:
            stacked_samples = samples

        # Check for parameters with no dynamic range and add explicit ranges
        ranges = None

        # Handle different data formats for corner plot range calculation
        # For ArviZ stacked samples, try to determine proper ranges
        if hasattr(stacked_samples, "to_numpy"):
            # xarray Dataset - use to_numpy() method
            try:
                sample_data = stacked_samples.to_numpy()  # type: ignore
                ranges = []
                for i in range(sample_data.shape[-1]):
                    param_data = sample_data[..., i].flatten()
                    param_range = np.max(param_data) - np.min(param_data)
                    if param_range == 0 or param_range < 1e-10:
                        # Constant parameter - add small range around the value
                        center = np.mean(param_data)
                        delta = max(abs(center) * 0.01, 1e-6)  # type: ignore
                        ranges.append((center - delta, center + delta))
                    else:
                        # Let corner determine automatically
                        ranges.append(None)
            except Exception as e:
                logger.debug(f"Could not extract ranges from stacked samples: {e}")
                # Fallback: try to use individual parameter ranges
                try:
                    if hasattr(stacked_samples, "data_vars"):
                        ranges = []
                        for var_name in list(stacked_samples.data_vars):  # type: ignore
                            # type: ignore
                            var_data = stacked_samples[var_name].values.flatten()
                            param_range = np.max(var_data) - np.min(var_data)
                            if param_range == 0 or param_range < 1e-10:
                                center = np.mean(var_data)
                                delta = max(abs(center) * 0.01, 1e-6)
                                ranges.append((center - delta, center + delta))
                            else:
                                ranges.append(None)
                    else:
                        ranges = None
                except Exception as e2:
                    logger.debug(f"Could not determine parameter ranges: {e2}")
                    ranges = None
        else:
            # For other data types, try basic conversion
            try:
                if isinstance(stacked_samples, np.ndarray):
                    sample_data = stacked_samples
                elif hasattr(stacked_samples, "values"):
                    sample_data = stacked_samples.values
                else:
                    sample_data = np.array(stacked_samples)

                ranges = []
                for i in range(sample_data.shape[-1]):  # type: ignore
                    param_data = sample_data[..., i].flatten()  # type: ignore
                    param_range = np.max(param_data) - np.min(param_data)
                    if param_range == 0 or param_range < 1e-10:
                        # Constant parameter - add small range around the value
                        center = np.mean(param_data)
                        delta = max(abs(center) * 0.01, 1e-6)  # type: ignore
                        ranges.append((center - delta, center + delta))
                    else:
                        # Let corner determine automatically
                        ranges.append(None)
            except Exception as e:
                logger.debug(f"Could not determine ranges for corner plot: {e}")
                ranges = None

        # Create the corner plot
        if CORNER_AVAILABLE:
            # Use corner package if available (better formatting)
            import corner

            # Debug: Check what we're passing to corner
            logger.debug(f"Stacked samples type: {type(stacked_samples)}")
            logger.debug(
                f"Stacked samples shape: {
                    getattr(stacked_samples, 'shape', 'No shape attr')
                }"
            )
            logger.debug(f"Ranges: {ranges}")

            # Try to convert ArviZ data to numpy for corner plot
            try:
                # Initialize corner_data variable
                corner_data: np.ndarray

                # Handle xarray Dataset conversion properly
                if hasattr(stacked_samples, "data_vars"):
                    # This is an xarray Dataset - need to extract data from
                    # each variable
                    var_names = list(stacked_samples.data_vars.keys())  # type: ignore
                    logger.debug(f"Extracting data from variables: {var_names}")

                    # Extract data arrays for each parameter and stack them
                    param_arrays = []
                    for var_name in var_names:
                        # type: ignore
                        var_data = stacked_samples[var_name].values.flatten()
                        param_arrays.append(var_data)
                        logger.debug(
                            f"Variable {var_name} shape after flatten: {var_data.shape}"
                        )

                    # Stack parameter arrays to create (n_samples, n_params)
                    # array
                    corner_data = np.column_stack(param_arrays)
                    logger.debug(f"Stacked corner data shape: {corner_data.shape}")

                elif hasattr(stacked_samples, "to_numpy"):
                    corner_data = stacked_samples.to_numpy()  # type: ignore
                    logger.debug(f"Converted to numpy shape: {corner_data.shape}")
                # type: ignore
                elif hasattr(stacked_samples, "values") and not callable(
                    stacked_samples.values
                ):
                    # .values is a property, not a method - access it correctly
                    corner_data = stacked_samples.values  # type: ignore
                    logger.debug(f"Using .values property shape: {corner_data.shape}")
                else:
                    corner_data = stacked_samples  # type: ignore

                # Ensure we have 2D data (samples x parameters)
                if hasattr(corner_data, "ndim") and corner_data.ndim > 2:
                    # Flatten extra dimensions
                    corner_data = corner_data.reshape(-1, corner_data.shape[-1])
                    logger.debug(f"Reshaped to: {corner_data.shape}")
                elif not hasattr(corner_data, "ndim"):
                    # For remaining objects without ndim, try to convert to
                    # numpy
                    try:
                        if hasattr(corner_data, "to_numpy"):
                            corner_data = corner_data.to_numpy()  # type: ignore
                            logger.debug(
                                f"Converted Dataset to numpy with shape: {
                                    corner_data.shape
                                }"
                            )
                        else:
                            # Convert using pandas if possible
                            corner_data = corner_data.to_pandas().values  # type: ignore
                            logger.debug(
                                f"Converted via pandas with shape: {corner_data.shape}"
                            )
                    except Exception as conversion_error:
                        logger.debug(
                            f"Failed to convert corner_data: {conversion_error}"
                        )
                        raise

                # Determine number of parameters
                n_params = (
                    corner_data.shape[1]
                    if hasattr(corner_data, "shape")
                    else (len(ranges) if ranges else 3)
                )

                # Filter parameter names and units to match active parameters
                filtered_param_names = param_names
                filtered_param_units = param_units

                if active_param_names and param_names:
                    # Create mapping from original param names to their indices
                    param_name_to_idx = {name: i for i, name in enumerate(param_names)}

                    # Filter param_names and param_units to only include active
                    # parameters
                    filtered_param_names = [
                        name for name in active_param_names if name in param_name_to_idx
                    ]
                    if param_units:
                        filtered_param_units = [
                            param_units[param_name_to_idx[name]]
                            for name in filtered_param_names
                            if name in param_name_to_idx
                        ]
                    else:
                        filtered_param_units = None

                    logger.debug(f"Filtered param names: {filtered_param_names}")
                    logger.debug(f"Filtered param units: {filtered_param_units}")

                # Create parameter labels with safe indexing
                labels = []
                for i in range(n_params):
                    if filtered_param_names and i < len(filtered_param_names):
                        if filtered_param_units and i < len(filtered_param_units):
                            labels.append(
                                f"{filtered_param_names[i]}\n[{
                                    filtered_param_units[i]
                                }]"
                            )
                        else:
                            labels.append(filtered_param_names[i])
                    else:
                        labels.append(f"Param {i}")

                fig = corner.corner(
                    corner_data,
                    labels=labels,
                    range=ranges,
                    show_titles=True,
                    title_kwargs={"fontsize": 12},
                    label_kwargs={"fontsize": 14},
                    hist_kwargs={"density": True, "alpha": 0.8},
                    contour_kwargs={"colors": ["C0", "C1", "C2"]},
                    fill_contours=True,
                    plot_contours=True,
                )
            except Exception as corner_error:
                logger.warning(
                    f"Corner plot failed with corner package: {corner_error}"
                )
                # Fall back to ArviZ built-in plot
                import arviz as az

                axes = az.plot_pair(
                    samples,  # Use original samples, not stacked
                    kind="kde",
                    marginals=True,
                    figsize=plot_config["figure_size"],
                )
                fig = axes.ravel()[0].figure
        else:
            # Use ArviZ built-in plot
            import arviz as az

            axes = az.plot_pair(
                stacked_samples,
                kind="kde",
                marginals=True,
                figsize=plot_config["figure_size"],
            )
            fig = axes.ravel()[0].figure

        # Add title
        fig.suptitle(
            f"{title_prefix} Posterior Distribution Corner Plot",
            fontsize=16,
            y=0.98,
        )

        # Save the plot
        filename = f"mcmc_corner_plot.{plot_config['plot_format']}"
        filepath = outdir / filename

        success = save_fig(
            fig,
            filepath,
            dpi=plot_config["dpi"],
            format=plot_config["plot_format"],
        )
        plt.close(fig)

        if success:
            logger.info("Successfully created MCMC corner plot")
        else:
            logger.error("Failed to save MCMC corner plot")

        return success

    except Exception as e:
        logger.error(f"Error creating MCMC corner plot: {e}")
        plt.close("all")
        return False


def plot_mcmc_trace(
    trace_data: Any,
    outdir: str | Path,
    config: dict | None = None,
    param_names: list[str] | None = None,
    param_units: list[str] | None = None,
) -> bool:
    """
    Create MCMC trace plots showing parameter evolution across chains.

    Args:
        trace_data: MCMC trace data (ArviZ InferenceData or similar)
        outdir (Union[str, Path]): Output directory for saved plots
        config (Optional[Dict]): Configuration dictionary
        param_names (Optional[List[str]]): Parameter names for labeling
        param_units (Optional[List[str]]): Parameter units for labeling

    Returns:
        bool: True if trace plots were created successfully
    """
    if not ARVIZ_AVAILABLE:
        logger.warning("ArviZ not available - cannot create MCMC trace plots")
        return False

    logger.info("Creating MCMC trace plots")

    # Get plotting configuration
    plot_config = get_plot_config(config)
    setup_matplotlib_style(plot_config)

    # Ensure output directory exists
    outdir = ensure_dir(outdir)

    try:
        import arviz as az
        import numpy as np  # Import numpy at the top to ensure it's available throughout

        # Handle different trace data formats
        if hasattr(trace_data, "posterior"):
            # ArviZ InferenceData
            trace_obj = trace_data
        else:
            logger.error("Unsupported trace data format for trace plots")
            return False

        # Create trace plot with proper variable name handling
        try:
            # Set up numpy error handling to prevent underflow from causing
            # failures
            old_err = np.seterr(all="ignore")

            try:
                # First check what variables are actually available
                if hasattr(trace_obj, "posterior") and hasattr(
                    trace_obj.posterior, "data_vars"
                ):
                    available_vars = list(trace_obj.posterior.data_vars.keys())
                    logger.debug(f"Available variables in trace: {available_vars}")

                    # Use only parameter names that exist in the trace
                    if param_names:
                        var_names_to_use = [
                            name for name in param_names if name in available_vars
                        ]
                        if not var_names_to_use:
                            logger.warning(
                                f"None of the requested parameter names {param_names} found in trace"
                            )
                            var_names_to_use = None  # Use all available
                    else:
                        var_names_to_use = None
                else:
                    var_names_to_use = None

                axes = az.plot_trace(
                    trace_obj,
                    var_names=var_names_to_use,
                    figsize=(
                        plot_config["figure_size"][0] * 1.2,
                        plot_config["figure_size"][1] * 1.5,
                    ),
                    compact=True,
                )
            finally:
                # Restore original numpy error handling
                np.seterr(**old_err)
        except Exception as e:
            logger.warning(f"Failed to create trace plot with requested variables: {e}")
            # Fallback: try without specifying variable names
            try:
                # Set up numpy error handling for fallback attempt too
                old_err = np.seterr(all="ignore")
                try:
                    axes = az.plot_trace(
                        trace_obj,
                        var_names=None,
                        figsize=(
                            plot_config["figure_size"][0] * 1.2,
                            plot_config["figure_size"][1] * 1.5,
                        ),
                        compact=True,
                    )
                finally:
                    np.seterr(**old_err)
            except Exception as e2:
                logger.error(
                    f"Failed to create trace plot even without variable names: {e2}"
                )
                return False

        fig = axes.ravel()[0].figure

        # Add parameter units to y-labels if available
        if param_names and param_units:
            for i, (name, unit) in enumerate(
                zip(param_names, param_units, strict=False)
            ):
                if i < len(axes):
                    # Find the KDE plot (right column)
                    if len(axes.shape) > 1 and axes.shape[1] > 1:
                        kde_ax = axes[i, 1]
                        kde_ax.set_ylabel(f"{name}\n[{unit}]")

        # Add title
        fig.suptitle("MCMC Trace Plots - Parameter Evolution", fontsize=16, y=0.98)

        # Save the plot
        filename = f"mcmc_trace_plots.{plot_config['plot_format']}"
        filepath = outdir / filename

        success = save_fig(
            fig,
            filepath,
            dpi=plot_config["dpi"],
            format=plot_config["plot_format"],
        )
        plt.close(fig)

        if success:
            logger.info("Successfully created MCMC trace plots")
        else:
            logger.error("Failed to save MCMC trace plots")

        return success

    except Exception as e:
        logger.error(f"Error creating MCMC trace plots: {e}")
        plt.close("all")
        return False


def plot_mcmc_convergence_diagnostics(
    trace_data: Any,
    diagnostics: dict[str, Any],
    outdir: str | Path,
    config: dict | None = None,
    param_names: list[str] | None = None,
) -> bool:
    """
    Create comprehensive MCMC convergence diagnostic plots.

    Args:
        trace_data: MCMC trace data (ArviZ InferenceData or similar)
        diagnostics: Convergence diagnostics dictionary
        outdir (Union[str, Path]): Output directory for saved plots
        config (Optional[Dict]): Configuration dictionary
        param_names (Optional[List[str]]): Parameter names for labeling

    Returns:
        bool: True if diagnostic plots were created successfully
    """
    if not ARVIZ_AVAILABLE:
        logger.warning("ArviZ not available - cannot create MCMC diagnostic plots")
        return False

    logger.info("Creating MCMC convergence diagnostic plots")

    # Get plotting configuration
    plot_config = get_plot_config(config)
    setup_matplotlib_style(plot_config)

    # Ensure output directory exists
    outdir = ensure_dir(outdir)

    try:
        import arviz as az

        # Validate trace data format
        if not hasattr(trace_data, "posterior"):
            logger.error("Unsupported trace data format for convergence diagnostics")
            return False

        # Get active parameters from config to filter out inactive ones
        active_param_names = None
        if (
            config
            and "initial_parameters" in config
            and "active_parameters" in config["initial_parameters"]
        ):
            active_param_names = config["initial_parameters"]["active_parameters"]
            logger.debug(f"Using active parameters from config: {active_param_names}")

        # Use active parameters if available, otherwise use param_names
        if active_param_names:
            param_names = active_param_names
            logger.debug(f"Filtered to active parameters: {param_names}")
        elif param_names:
            logger.debug(f"Using provided parameter names: {param_names}")
        else:
            logger.debug(
                "No parameter names provided, will use all available parameters"
            )

        # Create figure with multiple subplots
        fig = plt.figure(
            figsize=(
                plot_config["figure_size"][0] * 1.5,
                plot_config["figure_size"][1] * 1.2,
            )
        )
        gs = gridspec.GridSpec(2, 3, hspace=0.3, wspace=0.3)

        # Plot 1: R-hat values
        ax1 = fig.add_subplot(gs[0, 0])

        # Check for diagnostics with various key names (r_hat, rhat)
        r_hat_data = diagnostics.get("r_hat") or diagnostics.get("rhat")
        logger.debug(f"R-hat from diagnostics: {r_hat_data}")

        # Convert ArviZ Dataset to dict if needed
        r_hat_dict = {}
        if r_hat_data is not None:
            try:
                if hasattr(r_hat_data, "items"):
                    # ArviZ Dataset object
                    r_hat_dict = {str(k): float(v) for k, v in r_hat_data.items()}
                elif isinstance(r_hat_data, dict):
                    # Already a dictionary
                    r_hat_dict = r_hat_data
                logger.debug(f"Converted R-hat dict: {r_hat_dict}")
            except Exception as e:
                logger.warning(f"Could not convert R-hat data: {e}")

        # If r_hat dict is still missing or empty, compute from trace data
        if not r_hat_dict and hasattr(trace_data, "posterior"):
            try:
                r_hat_summary = az.rhat(trace_data)
                logger.debug(f"Computed R-hat summary: {r_hat_summary}")
                if hasattr(r_hat_summary, "to_dict"):
                    r_hat_dict = r_hat_summary.to_dict()  # type: ignore
                else:
                    # Convert DataArray to dict
                    r_hat_dict = {str(k): float(v) for k, v in r_hat_summary.items()}  # type: ignore
                logger.debug(f"Computed R-hat dict: {r_hat_dict}")
            except Exception as e:
                logger.warning(f"Could not compute R-hat from trace data: {e}")

        if r_hat_dict:
            logger.debug(f"Processing R-hat dict with {len(r_hat_dict)} entries")
            # Filter for active parameters if available in config
            if param_names is None:
                param_names_plot = list(r_hat_dict.keys())
                logger.debug(f"Using all R-hat parameters: {param_names_plot}")
            else:
                param_names_plot = param_names
                logger.debug(f"Using filtered parameter names: {param_names_plot}")

            # Further filter to only include parameters that actually exist in
            # r_hat_dict
            available_params = [name for name in param_names_plot if name in r_hat_dict]
            logger.debug(f"Parameters available in R-hat data: {available_params}")
            param_names_plot = available_params
            r_hat_values = [r_hat_dict.get(name, 1.0) for name in param_names_plot]
            logger.debug(
                f"R-hat values for plotting: {dict(zip(param_names_plot, r_hat_values, strict=False))}"
            )

            # Only plot if we have data
            if param_names_plot and r_hat_values:
                logger.debug(
                    f"Creating R-hat plot with {len(param_names_plot)} parameters"
                )
                colors = [
                    "green" if r < 1.1 else "orange" if r < 1.2 else "red"
                    for r in r_hat_values
                ]
                bars = ax1.barh(param_names_plot, r_hat_values, color=colors, alpha=0.7)

                # Set appropriate axis limits
                if max(r_hat_values) > 0:
                    ax1.set_xlim(0.9, max(max(r_hat_values) * 1.1, 1.3))

                # Add value labels
                for bar, value in zip(bars, r_hat_values, strict=False):
                    width = bar.get_width()
                    ax1.text(
                        width + 0.01,
                        bar.get_y() + bar.get_height() / 2,
                        f"{value:.3f}",
                        ha="left",
                        va="center",
                        fontsize=10,
                    )

                ax1.axvline(
                    x=1.1,
                    color="red",
                    linestyle="--",
                    alpha=0.7,
                    label="R̂ = 1.1 threshold",
                )
                ax1.set_xlabel("R̂ (Gelman-Rubin statistic)")
                ax1.set_title("Convergence: R̂ Values")
                ax1.legend()
                ax1.grid(True, alpha=0.3)

        # Plot 2: Effective Sample Size (ESS)
        ax2 = fig.add_subplot(gs[0, 1])

        # Check for diagnostics with various key names (ess_bulk, ess)
        ess_data = diagnostics.get("ess_bulk") or diagnostics.get("ess")
        logger.debug(f"ESS from diagnostics: {ess_data}")

        # Convert ArviZ Dataset to dict if needed
        ess_dict = {}
        if ess_data is not None:
            try:
                if hasattr(ess_data, "items"):
                    # ArviZ Dataset object
                    ess_dict = {str(k): float(v) for k, v in ess_data.items()}
                elif isinstance(ess_data, dict):
                    # Already a dictionary
                    ess_dict = ess_data
                logger.debug(f"Converted ESS dict: {ess_dict}")
            except Exception as e:
                logger.warning(f"Could not convert ESS data: {e}")

        # If ESS dict is still missing or empty, compute from trace data
        if not ess_dict and hasattr(trace_data, "posterior"):
            try:
                ess_summary = az.ess(trace_data)
                logger.debug(f"Computed ESS summary: {ess_summary}")
                if hasattr(ess_summary, "to_dict"):
                    ess_dict = ess_summary.to_dict()
                else:
                    # Convert DataArray to dict
                    ess_dict = {str(k): float(v) for k, v in ess_summary.items()}
                logger.debug(f"Computed ESS dict: {ess_dict}")
            except Exception as e:
                logger.warning(f"Could not compute ESS from trace data: {e}")

        if ess_dict:
            # Filter for active parameters if available in config
            if param_names is None:
                param_names_plot = list(ess_dict.keys())
            else:
                param_names_plot = param_names

            # Further filter to only include parameters that actually exist in
            # ess_dict
            param_names_plot = [name for name in param_names_plot if name in ess_dict]
            ess_values = [ess_dict.get(name, 0) for name in param_names_plot]

            # Only plot if we have data
            if param_names_plot and ess_values:
                # Color based on ESS quality (>400 good, >100 okay, <100 poor)
                colors = [
                    "green" if ess > 400 else "orange" if ess > 100 else "red"
                    for ess in ess_values
                ]
                bars = ax2.barh(param_names_plot, ess_values, color=colors, alpha=0.7)

                # Set appropriate axis limits
                if max(ess_values) > 0:
                    ax2.set_xlim(0, max(ess_values) * 1.1)

                # Add value labels
                for bar, value in zip(bars, ess_values, strict=False):
                    width = bar.get_width()
                    ax2.text(
                        width + max(ess_values) * 0.01,
                        bar.get_y() + bar.get_height() / 2,
                        f"{int(value)}",
                        ha="left",
                        va="center",
                        fontsize=10,
                    )

                ax2.axvline(
                    x=400,
                    color="green",
                    linestyle="--",
                    alpha=0.7,
                    label="ESS = 400 (good)",
                )
                ax2.axvline(
                    x=100,
                    color="orange",
                    linestyle="--",
                    alpha=0.7,
                    label="ESS = 100 (minimum)",
                )
                ax2.set_xlabel("Effective Sample Size")
                ax2.set_title("Sampling Efficiency: ESS")
                ax2.legend()
                ax2.grid(True, alpha=0.3)

        # Plot 3: Monte Carlo Standard Error
        ax3 = fig.add_subplot(gs[0, 2])

        # Check for diagnostics with various key names (mcse_mean, mcse)
        mcse_data = diagnostics.get("mcse_mean") or diagnostics.get("mcse")
        logger.debug(f"MCSE from diagnostics: {mcse_data}")

        # Convert ArviZ Dataset to dict if needed
        mcse_dict = {}
        if mcse_data is not None:
            try:
                if hasattr(mcse_data, "items"):
                    # ArviZ Dataset object
                    mcse_dict = {str(k): float(v) for k, v in mcse_data.items()}
                elif isinstance(mcse_data, dict):
                    # Already a dictionary
                    mcse_dict = mcse_data
                logger.debug(f"Converted MCSE dict: {mcse_dict}")
            except Exception as e:
                logger.warning(f"Could not convert MCSE data: {e}")

        # If MCSE dict is still missing or empty, compute from trace data
        if not mcse_dict and hasattr(trace_data, "posterior"):
            try:
                mcse_summary = az.mcse(trace_data)
                logger.debug(f"Computed MCSE summary: {mcse_summary}")
                if hasattr(mcse_summary, "to_dict"):
                    mcse_dict = mcse_summary.to_dict()
                else:
                    # Convert DataArray to dict
                    mcse_dict = {str(k): float(v) for k, v in mcse_summary.items()}
                logger.debug(f"Computed MCSE dict: {mcse_dict}")
            except Exception as e:
                logger.warning(f"Could not compute MCSE from trace data: {e}")

        if mcse_dict:
            # Filter for active parameters if available in config
            if param_names is None:
                param_names_plot = list(mcse_dict.keys())
            else:
                param_names_plot = param_names

            # Further filter to only include parameters that actually exist in
            # mcse_dict
            param_names_plot = [name for name in param_names_plot if name in mcse_dict]
            mcse_values = [mcse_dict.get(name, 0) for name in param_names_plot]

            # Only plot if we have data
            if param_names_plot and mcse_values:
                bars = ax3.barh(
                    param_names_plot, mcse_values, alpha=0.7, color="skyblue"
                )

                # Set appropriate axis limits
                if max(mcse_values) > 0:
                    ax3.set_xlim(0, max(mcse_values) * 1.1)

                # Add value labels
                for bar, value in zip(bars, mcse_values, strict=False):
                    width = bar.get_width()
                    ax3.text(
                        width + max(mcse_values) * 0.01,
                        bar.get_y() + bar.get_height() / 2,
                        f"{value:.2e}",
                        ha="left",
                        va="center",
                        fontsize=9,
                    )

                ax3.set_xlabel("Monte Carlo Standard Error")
                ax3.set_title("Uncertainty: MCSE")
                ax3.grid(True, alpha=0.3)

        # Plot 4: Energy plot (if available)
        ax4 = fig.add_subplot(gs[1, :2])
        try:
            if (
                hasattr(trace_data, "sample_stats")
                and "energy" in trace_data.sample_stats
            ):
                az.plot_energy(trace_data, ax=ax4)
                ax4.set_title("Energy Plot - Sampling Efficiency")
            else:
                ax4.text(
                    0.5,
                    0.5,
                    "Energy data not available",
                    ha="center",
                    va="center",
                    transform=ax4.transAxes,
                    fontsize=12,
                )
                ax4.set_title("Energy Plot - Not Available")
        except Exception as e:
            logger.warning(f"Could not create energy plot: {e}")
            ax4.text(
                0.5,
                0.5,
                f"Energy plot failed: {str(e)[:50]}...",
                ha="center",
                va="center",
                transform=ax4.transAxes,
                fontsize=10,
            )

        # Plot 5: Summary statistics
        ax5 = fig.add_subplot(gs[1, 2])
        ax5.axis("off")

        # Create summary text
        summary_text = "MCMC Summary:\n\n"
        if "max_rhat" in diagnostics:
            summary_text += f"Max R̂: {diagnostics['max_rhat']:.3f}\n"
        if "min_ess" in diagnostics:
            summary_text += f"Min ESS: {int(diagnostics['min_ess'])}\n"
        if "converged" in diagnostics:
            converged_status = "✓ Yes" if diagnostics["converged"] else "✗ No"
            summary_text += f"Converged: {converged_status}\n"
        if "assessment" in diagnostics:
            summary_text += f"Assessment: {diagnostics['assessment']}\n"

        # Add chain info if available
        if hasattr(trace_data, "posterior"):
            # Use sizes instead of dims to avoid FutureWarning
            posterior_sizes = getattr(
                trace_data.posterior, "sizes", trace_data.posterior.dims
            )
            n_chains = posterior_sizes.get("chain", "Unknown")
            n_draws = posterior_sizes.get("draw", "Unknown")
            summary_text += f"\nChains: {n_chains}\n"
            summary_text += f"Draws: {n_draws}"

        ax5.text(
            0.05,
            0.95,
            summary_text,
            transform=ax5.transAxes,
            fontsize=11,
            verticalalignment="top",
            fontfamily="monospace",
            bbox={"boxstyle": "round,pad=0.5", "facecolor": "lightgray", "alpha": 0.5},
        )

        # Add overall title
        fig.suptitle("MCMC Convergence Diagnostics", fontsize=16, y=0.98)

        # Save the plot
        filename = f"mcmc_convergence_diagnostics.{plot_config['plot_format']}"
        filepath = outdir / filename

        success = save_fig(
            fig,
            filepath,
            dpi=plot_config["dpi"],
            format=plot_config["plot_format"],
        )
        plt.close(fig)

        if success:
            logger.info("Successfully created MCMC convergence diagnostic plots")
        else:
            logger.error("Failed to save MCMC convergence diagnostic plots")

        return success

    except Exception as e:
        logger.error(f"Error creating MCMC convergence diagnostic plots: {e}")
        plt.close("all")
        return False


def plot_diagnostic_summary(
    results: dict[str, Any],
    outdir: str | Path,
    config: dict | None = None,
    method_name: str | None = None,
) -> bool:
    """
    Create a comprehensive diagnostic summary plot combining multiple visualizations.

    Generates a 2×3 grid layout containing:
    - Method comparison with chi-squared values
    - Parameter uncertainties visualization
    - MCMC convergence diagnostics (R-hat values)
    - Residuals distribution analysis with normal distribution overlay

    Features adaptive content with appropriate placeholders when data is unavailable,
    professional formatting with consistent styling, and cross-method comparison
    capabilities for quality assessment.

    Args:
        results (Dict[str, Any]): Complete analysis results dictionary
        outdir (Union[str, Path]): Output directory for saved plots
        config (Optional[Dict]): Configuration dictionary
        method_name (Optional[str]): Optimization method name for filename prefix

    Returns:
        bool: True if diagnostic plots were created successfully
    """
    logger.info("Creating diagnostic summary plots")

    # Get plotting configuration
    plot_config = get_plot_config(config)
    setup_matplotlib_style(plot_config)

    # Ensure output directory exists
    outdir = ensure_dir(outdir)

    try:
        # Create a summary figure with multiple subplots
        fig = plt.figure(
            figsize=(
                plot_config["figure_size"][0] * 1.5,
                plot_config["figure_size"][1] * 1.2,
            )
        )
        gs = gridspec.GridSpec(2, 3, hspace=0.3, wspace=0.3)

        # Plot 1: Chi-squared comparison (if multiple methods available)
        ax1 = fig.add_subplot(gs[0, 0])
        methods = []
        chi2_values = []

        for key, value in results.items():
            if "chi_squared" in key or "chi2" in key:
                chi2_method_name = key.replace("_chi_squared", "").replace("_chi2", "")
                methods.append(chi2_method_name.replace("_", " ").title())
                chi2_values.append(value)

        if chi2_values:
            bars = ax1.bar(
                methods,
                chi2_values,
                alpha=0.7,
                color=["C0", "C1", "C2", "C3"][: len(methods)],
            )
            ax1.set_ylabel("χ² Value")
            ax1.set_title("Method Comparison")
            ax1.set_yscale("log")

            # Add value labels
            for bar, value in zip(bars, chi2_values, strict=False):
                bar_width = bar.get_width()
                if bar_width > 0:  # Avoid division by zero
                    ax1.text(
                        bar.get_x() + bar_width / 2,
                        bar.get_height() * 1.1,
                        f"{value:.2e}",
                        ha="center",
                        va="bottom",
                        fontsize=10,
                    )

        # Plot 2: Parameter uncertainty (if available)
        ax2 = fig.add_subplot(gs[0, 1])
        uncertainties = results.get("parameter_uncertainties", {})

        # Try to compute uncertainties from MCMC trace if not available
        if not uncertainties and "mcmc_trace" in results and ARVIZ_AVAILABLE:
            try:
                import arviz as az

                trace_data = results["mcmc_trace"]
                if hasattr(trace_data, "posterior"):
                    # Get parameter names from config or trace data
                    param_names = None
                    if (
                        config
                        and "initial_parameters" in config
                        and "parameter_names" in config["initial_parameters"]
                    ):
                        param_names = config["initial_parameters"]["parameter_names"]
                    elif hasattr(trace_data.posterior, "data_vars"):
                        param_names = list(trace_data.posterior.data_vars.keys())

                    if param_names:
                        uncertainties = {}
                        for param in param_names:
                            if param in trace_data.posterior:
                                samples = trace_data.posterior[param].values.flatten()
                                uncertainties[param] = float(np.std(samples))
                        logger.debug(
                            f"Computed parameter uncertainties: {uncertainties}"
                        )
            except Exception as e:
                logger.warning(f"Could not compute parameter uncertainties: {e}")

        if uncertainties:
            param_names = list(uncertainties.keys())
            uncertainty_values = list(uncertainties.values())

            # Filter for active parameters if available
            if (
                config
                and "initial_parameters" in config
                and "active_parameters" in config["initial_parameters"]
            ):
                active_param_names = config["initial_parameters"]["active_parameters"]
                param_names = [
                    name for name in active_param_names if name in uncertainties
                ]
                uncertainty_values = [uncertainties[name] for name in param_names]

            if param_names and uncertainty_values:  # Check if we have data
                ax2.barh(param_names, uncertainty_values, alpha=0.7)
                # Set appropriate axis limits
                if max(uncertainty_values) > 0:
                    ax2.set_xlim(0, max(uncertainty_values) * 1.1)
                ax2.set_xlabel("Parameter Uncertainty (σ)")
                ax2.set_title("Parameter Uncertainties")
                ax2.grid(True, alpha=0.3)
        else:
            # Show placeholder message if no uncertainties available
            ax2.text(
                0.5,
                0.5,
                "No uncertainty data\navailable",
                ha="center",
                va="center",
                transform=ax2.transAxes,
                fontsize=12,
                color="gray",
            )
            ax2.set_title("Parameter Uncertainties")
            ax2.set_xticks([])
            ax2.set_yticks([])

        # Plot 3: Convergence diagnostics (if MCMC results available)
        ax3 = fig.add_subplot(gs[0, 2])
        if "mcmc_diagnostics" in results and ARVIZ_AVAILABLE:
            # Plot R-hat values
            diagnostics = results["mcmc_diagnostics"]

            # Check for diagnostics with various key names (r_hat, rhat)
            r_hat_data = diagnostics.get("r_hat") or diagnostics.get("rhat")
            logger.debug(f"R-hat data for summary plot: {r_hat_data}")

            # Convert ArviZ Dataset to dict if needed
            r_hat_dict = {}
            if r_hat_data is not None:
                try:
                    if hasattr(r_hat_data, "items"):
                        # ArviZ Dataset object
                        r_hat_dict = {str(k): float(v) for k, v in r_hat_data.items()}
                    elif isinstance(r_hat_data, dict):
                        # Already a dictionary
                        r_hat_dict = r_hat_data
                    logger.debug(f"Converted R-hat dict for summary: {r_hat_dict}")
                except Exception as e:
                    logger.warning(f"Could not convert R-hat data for summary: {e}")

            # Try to compute R-hat from trace data if missing
            if not r_hat_dict and "mcmc_trace" in results:
                try:
                    import arviz as az

                    trace_data = results["mcmc_trace"]
                    if hasattr(trace_data, "posterior"):
                        r_hat_summary = az.rhat(trace_data)
                        if hasattr(r_hat_summary, "to_dict"):
                            r_hat_dict = r_hat_summary.to_dict()  # type: ignore
                        else:
                            r_hat_dict = {
                                str(k): float(v) for k, v in r_hat_summary.items()
                            }  # type: ignore
                        logger.debug(f"Computed R-hat dict for summary: {r_hat_dict}")
                except Exception as e:
                    logger.warning(f"Could not compute R-hat for summary plot: {e}")

            if r_hat_dict:
                # Get active parameters from config to filter out inactive ones
                active_param_names = None
                if (
                    config
                    and "initial_parameters" in config
                    and "active_parameters" in config["initial_parameters"]
                ):
                    active_param_names = config["initial_parameters"][
                        "active_parameters"
                    ]
                    logger.debug(
                        f"Using active parameters for summary: {active_param_names}"
                    )

                # Filter for active parameters if available
                if active_param_names:
                    param_names = [
                        name for name in active_param_names if name in r_hat_dict
                    ]
                else:
                    param_names = list(r_hat_dict.keys())

                r_hat_values = [r_hat_dict.get(name, 1.0) for name in param_names]
                logger.debug(
                    f"Summary plot R-hat values: {dict(zip(param_names, r_hat_values, strict=False))}"
                )

                if param_names and r_hat_values:  # Check if we have data
                    colors = [
                        "green" if r < 1.1 else "orange" if r < 1.2 else "red"
                        for r in r_hat_values
                    ]
                    ax3.barh(param_names, r_hat_values, color=colors, alpha=0.7)
                    # Set appropriate axis limits
                    if max(r_hat_values) > 0:
                        ax3.set_xlim(0.9, max(max(r_hat_values) * 1.1, 1.3))
                    ax3.axvline(
                        x=1.1,
                        color="red",
                        linestyle="--",
                        alpha=0.7,
                        label="R̂ = 1.1",
                    )
                    ax3.set_xlabel("R̂ (Convergence)")
                    ax3.set_title("MCMC Convergence")
                    ax3.legend()
                    ax3.grid(True, alpha=0.3)
        else:
            # Show placeholder message if no MCMC diagnostics available
            ax3.text(
                0.5,
                0.5,
                "No MCMC convergence\ndiagnostics available",
                ha="center",
                va="center",
                transform=ax3.transAxes,
                fontsize=12,
                color="gray",
            )
            ax3.set_title("MCMC Convergence")
            ax3.set_xticks([])
            ax3.set_yticks([])

        # Plot 4: Residuals analysis (if available)
        ax4 = fig.add_subplot(gs[1, :])
        residuals = results.get("residuals")

        # Try to compute residuals from experimental and theoretical data if
        # not available
        if residuals is None:
            exp_data = results.get("experimental_data")
            theory_data = results.get("theoretical_data")

            if exp_data is not None and theory_data is not None:
                try:
                    if isinstance(exp_data, np.ndarray) and isinstance(
                        theory_data, np.ndarray
                    ):
                        if exp_data.shape == theory_data.shape:
                            residuals = exp_data - theory_data
                            logger.debug(
                                f"Computed residuals from exp - theory data, shape: {
                                    residuals.shape
                                }"
                            )
                        else:
                            logger.warning(
                                f"Shape mismatch: exp_data {
                                    exp_data.shape
                                } vs theory_data {theory_data.shape}"
                            )
                except Exception as e:
                    logger.warning(f"Could not compute residuals from data: {e}")

        if (
            residuals is not None
            and isinstance(residuals, np.ndarray)
            and residuals.size > 0
        ):
            # Flatten residuals for histogram
            flat_residuals = residuals.flatten()

            # Only plot if we have data
            if len(flat_residuals) > 0:
                # Create histogram
                ax4.hist(
                    flat_residuals,
                    bins=50,
                    alpha=0.7,
                    density=True,
                    color="skyblue",
                )

                # Overlay normal distribution for comparison
                mu, sigma = np.mean(flat_residuals), np.std(flat_residuals)

                # Avoid division by zero if sigma is too small
                if sigma > 1e-10:
                    x = np.linspace(flat_residuals.min(), flat_residuals.max(), 100)
                    ax4.plot(
                        x,
                        (1 / (sigma * np.sqrt(2 * np.pi)))
                        * np.exp(-0.5 * ((x - mu) / sigma) ** 2),
                        "r-",
                        linewidth=2,
                        label=f"Normal(μ={mu:.3e}, σ={sigma:.3e})",
                    )
                else:
                    # If sigma is effectively zero, just show the mean as a
                    # vertical line
                    ax4.axvline(
                        float(mu),
                        color="red",
                        linestyle="--",
                        linewidth=2,
                        label=f"Mean={mu:.3e} (σ≈0)",
                    )
                    logger.warning(
                        "Standard deviation is very small, showing mean line instead of normal distribution"
                    )

                ax4.set_xlabel("Residual Value")
                ax4.set_ylabel("Density")
                ax4.set_title("Residuals Distribution Analysis")
                ax4.legend()
                ax4.grid(True, alpha=0.3)
        else:
            # Show placeholder message if no residuals available
            ax4.text(
                0.5,
                0.5,
                "No residuals data available\n(requires experimental and theoretical data)",
                ha="center",
                va="center",
                transform=ax4.transAxes,
                fontsize=12,
                color="gray",
            )
            ax4.set_title("Residuals Distribution Analysis")
            ax4.set_xticks([])
            ax4.set_yticks([])

        # Add overall title
        fig.suptitle("Analysis Diagnostic Summary", fontsize=18, y=0.98)

        # Save the plot
        method_prefix = f"{method_name.lower()}_" if method_name else ""
        filename = f"{method_prefix}diagnostic_summary.{plot_config['plot_format']}"
        filepath = outdir / filename

        success = save_fig(
            fig,
            filepath,
            dpi=plot_config["dpi"],
            format=plot_config["plot_format"],
        )
        plt.close(fig)

        if success:
            logger.info("Successfully created diagnostic summary plot")
        else:
            logger.error("Failed to save diagnostic summary plot")

        return success

    except Exception as e:
        logger.error(f"Error creating diagnostic summary plot: {e}")
        plt.close("all")
        return False


# Utility function to create all plots at once
def create_all_plots(
    results: dict[str, Any],
    outdir: str | Path,
    config: dict | None = None,
) -> dict[str, bool]:
    """
    Create all available plots based on the results dictionary.
    For classical optimization results, creates method-specific plots.

    Args:
        results (Dict[str, Any]): Complete analysis results dictionary
        outdir (Union[str, Path]): Output directory for saved plots
        config (Optional[Dict]): Configuration dictionary

    Returns:
        Dict[str, bool]: Success status for each plot type
    """
    logger.info("Creating all available plots")

    plot_status = {}

    # Handle method-specific plotting for classical optimization
    method_results = results.get("method_results", {})

    # If we have method-specific results, create plots for each method
    if method_results:
        for method_name, method_data in method_results.items():
            method_outdir = Path(outdir) / f"plots_{method_name.lower()}"
            method_outdir.mkdir(parents=True, exist_ok=True)

            # Create method-specific results dict for plotting
            method_results_dict = results.copy()
            method_results_dict.update(method_data)

            # C2 heatmaps (if correlation data available)
            if all(
                key in method_results_dict
                for key in [
                    "experimental_data",
                    "theoretical_data",
                    "phi_angles",
                ]
            ):
                plot_key = f"c2_heatmaps_{method_name.lower()}"
                plot_status[plot_key] = plot_c2_heatmaps(
                    method_results_dict["experimental_data"],
                    method_results_dict["theoretical_data"],
                    method_results_dict["phi_angles"],
                    method_outdir,
                    config,
                    method_name=method_name,
                )

            # Note: Method-specific diagnostic summary plots removed - only main
            # diagnostic_summary.png for --method all is generated
    else:
        # Fallback to standard plotting without method specificity
        # C2 heatmaps (if correlation data available)
        if all(
            key in results
            for key in ["experimental_data", "theoretical_data", "phi_angles"]
        ):
            plot_status["c2_heatmaps"] = plot_c2_heatmaps(
                results["experimental_data"],
                results["theoretical_data"],
                results["phi_angles"],
                outdir,
                config,
            )

        # MCMC plots (if trace data available) - these are not method-specific
        if "mcmc_trace" in results:
            # MCMC corner plot
            plot_status["mcmc_corner"] = plot_mcmc_corner(
                results["mcmc_trace"],
                outdir,
                config,
                param_names=results.get("parameter_names"),
                param_units=results.get("parameter_units"),
            )

            # MCMC trace plots
            plot_status["mcmc_trace"] = plot_mcmc_trace(
                results["mcmc_trace"],
                outdir,
                config,
                param_names=results.get("parameter_names"),
                param_units=results.get("parameter_units"),
            )

            # MCMC convergence diagnostics (if diagnostics available)
            if "mcmc_diagnostics" in results:
                plot_status["mcmc_convergence"] = plot_mcmc_convergence_diagnostics(
                    results["mcmc_trace"],
                    results["mcmc_diagnostics"],
                    outdir,
                    config,
                    param_names=results.get("parameter_names"),
                )

        # Diagnostic summary (if not method-specific)
        if not method_results:
            plot_status["diagnostic_summary"] = plot_diagnostic_summary(
                results, outdir, config
            )

    # Log summary
    successful_plots = sum(plot_status.values())
    total_plots = len(plot_status)
    logger.info(f"Successfully created {successful_plots}/{total_plots} plots")

    return plot_status


if __name__ == "__main__":
    # Example usage and testing
    import tempfile

    # Set up logging for testing
    logging.basicConfig(level=logging.INFO)

    print("Testing plotting functions...")

    # Create test data
    n_angles, n_t2, n_t1 = 3, 50, 100
    phi_angles = np.array([0, 45, 90])

    # Generate synthetic correlation data
    np.random.seed(42)
    exp_data = 1 + 0.5 * np.random.exponential(1, (n_angles, n_t2, n_t1))
    theory_data = exp_data + 0.1 * np.random.normal(0, 1, exp_data.shape)

    # Test configuration
    test_config = {
        "output_settings": {
            "plotting": {
                "plot_format": "png",
                "dpi": 150,
                "figure_size": [8, 6],
            }
        }
    }

    with tempfile.TemporaryDirectory() as tmp_dir:
        print(f"Saving test plots to: {tmp_dir}")

        # Test C2 heatmaps
        success1 = plot_c2_heatmaps(
            exp_data, theory_data, phi_angles, tmp_dir, test_config
        )
        print(f"C2 heatmaps: {'Success' if success1 else 'Failed'}")

        # Parameter evolution test removed - function was non-functional

        print("Test completed!")


def plot_3d_surface(
    c2_experimental: np.ndarray,
    c2_fitted: np.ndarray,
    posterior_samples: np.ndarray | None = None,
    phi_angle: float = 0.0,
    outdir: str | Path = "./",
    config: dict | None = None,
    t2: np.ndarray | None = None,
    t1: np.ndarray | None = None,
    confidence_level: float = 0.95,
) -> bool:
    """
    Create 3D surface plots of C2 correlation data with confidence intervals.

    This function creates comprehensive 3D visualizations showing:
    - Experimental C2 data surface
    - Fitted C2 data surface (from MCMC posterior means)
    - Confidence interval bands (upper and lower bounds) as semi-transparent surfaces

    The plots provide intuitive 3D visualization of the correlation function
    structure and uncertainty quantification for publication-quality figures.

    Parameters
    ----------
    c2_experimental : np.ndarray
        Experimental correlation data [n_t2, n_t1]
    c2_fitted : np.ndarray
        Fitted correlation data [n_t2, n_t1] calculated from posterior means
    posterior_samples : np.ndarray, optional
        Posterior samples for uncertainty quantification [n_samples, n_t2, n_t1]
        If provided, confidence intervals will be calculated and plotted
    phi_angle : float, optional
        Angle in degrees for this data slice (default: 0.0)
    outdir : Union[str, Path], optional
        Output directory for saved plots (default: "./")
    config : Dict, optional
        Configuration dictionary for plot styling
    t2 : np.ndarray, optional
        Time lag values (t₂) for y-axis
    t1 : np.ndarray, optional
        Delay time values (t₁) for x-axis
    confidence_level : float, optional
        Confidence level for interval calculation (default: 0.95)

    Returns
    -------
    bool
        True if plots were created successfully, False otherwise

    Examples
    --------
    >>> # Basic usage with experimental and fitted data
    >>> success = plot_3d_surface(c2_exp, c2_fitted, phi_angle=0.0, outdir="./plots")

    >>> # With confidence intervals from MCMC samples
    >>> success = plot_3d_surface(
    ...     c2_exp, c2_fitted, posterior_samples=mcmc_samples,
    ...     phi_angle=45.0, outdir="./mcmc_results", confidence_level=0.95
    ... )
    """
    logger.info(f"Creating 3D surface plot for φ = {phi_angle:.1f}°")

    try:
        # Validate inputs
        if c2_experimental.shape != c2_fitted.shape:
            logger.error(
                f"Shape mismatch: experimental {c2_experimental.shape} vs fitted {
                    c2_fitted.shape
                }"
            )
            return False

        n_t2, n_t1 = c2_experimental.shape

        # Create time arrays if not provided
        if t2 is None:
            t2 = np.arange(n_t2)
        if t1 is None:
            t1 = np.arange(n_t1)

        # Ensure t1 and t2 are not None for type checking
        assert t1 is not None and t2 is not None

        # Create meshgrids for 3D plotting
        T1, T2 = np.meshgrid(t1, t2)

        # Validate meshgrid shapes
        if T1.shape != c2_experimental.shape or T2.shape != c2_experimental.shape:
            logger.error(
                f"Meshgrid shape {T1.shape} doesn't match data shape {
                    c2_experimental.shape
                }"
            )
            return False

        # Calculate confidence intervals if posterior samples provided
        upper_ci = None
        lower_ci = None
        if posterior_samples is not None:
            logger.info(
                f"Calculating {confidence_level * 100:.1f}% confidence intervals from {
                    posterior_samples.shape[0]
                } samples"
            )

            # Calculate percentiles for confidence intervals
            alpha = 1 - confidence_level
            lower_percentile = (alpha / 2) * 100
            upper_percentile = (1 - alpha / 2) * 100

            lower_ci = np.percentile(posterior_samples, lower_percentile, axis=0)
            upper_ci = np.percentile(posterior_samples, upper_percentile, axis=0)

            # Validate CI shapes
            if lower_ci.shape != c2_experimental.shape:
                logger.warning(
                    f"CI shape {lower_ci.shape} doesn't match data shape {
                        c2_experimental.shape
                    }"
                )
                lower_ci = upper_ci = None

        # Set up plotting style
        _setup_plot_style(config)

        # Create figure with two subplots: experimental and fitted
        fig = plt.figure(figsize=(16, 7))

        # =================================================================
        # Left subplot: Experimental data with confidence intervals
        # =================================================================
        ax1 = fig.add_subplot(121, projection="3d")

        # Plot experimental surface with enhanced aesthetics
        surf1 = ax1.plot_surface(
            T1,
            T2,
            c2_experimental,
            cmap="viridis",
            alpha=0.9,
            linewidth=0,
            antialiased=True,
            shade=True,
        )

        # Plot confidence interval surfaces if available
        if upper_ci is not None and lower_ci is not None:
            # Upper CI surface (semi-transparent red)
            ax1.plot_surface(
                T1,
                T2,
                upper_ci,
                color="red",
                alpha=0.3,
                linewidth=0,
                antialiased=True,
            )

            # Lower CI surface (semi-transparent red)
            ax1.plot_surface(
                T1,
                T2,
                lower_ci,
                color="red",
                alpha=0.3,
                linewidth=0,
                antialiased=True,
            )

            title1 = f"Experimental $C_2$ Data with {
                confidence_level * 100:.0f}% CI\nφ = {phi_angle:.1f}°"
        else:
            title1 = f"Experimental $C_2$ Data\nφ = {phi_angle:.1f}°"

        # Customize first subplot
        ax1.set_xlabel(r"$t_1$ (time units)", fontsize=12, labelpad=10)
        ax1.set_ylabel(r"$t_2$ (time units)", fontsize=12, labelpad=10)
        ax1.set_zlabel(r"$C_2$ (correlation)", fontsize=12, labelpad=8)
        ax1.set_title(title1, fontsize=14, pad=20)
        ax1.view_init(elev=20, azim=120)

        # Add colorbar for experimental data
        cbar1 = fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=20, pad=0.1)
        cbar1.set_label(r"$C_2$ (experimental)", fontsize=10)

        # =================================================================
        # Right subplot: Fitted data comparison
        # =================================================================
        ax2 = fig.add_subplot(122, projection="3d")

        # Plot fitted surface
        surf2 = ax2.plot_surface(
            T1,
            T2,
            c2_fitted,
            cmap="plasma",
            alpha=0.9,
            linewidth=0,
            antialiased=True,
            shade=True,
        )

        # Plot experimental as wireframe for comparison
        ax2.plot_wireframe(
            T1, T2, c2_experimental, color="gray", alpha=0.4, linewidth=0.5
        )

        # Customize second subplot
        ax2.set_xlabel(r"$t_1$ (time units)", fontsize=12, labelpad=10)
        ax2.set_ylabel(r"$t_2$ (time units)", fontsize=12, labelpad=10)
        ax2.set_zlabel(r"$C_2$ (correlation)", fontsize=12, labelpad=8)
        ax2.set_title(
            f"Fitted vs Experimental Data\nφ = {phi_angle:.1f}°",
            fontsize=14,
            pad=20,
        )
        ax2.view_init(elev=20, azim=120)

        # Add colorbar for fitted data
        cbar2 = fig.colorbar(surf2, ax=ax2, shrink=0.5, aspect=20, pad=0.1)
        cbar2.set_label(r"$C_2$ (fitted)", fontsize=10)

        # =================================================================
        # Additional plot enhancements
        # =================================================================

        # Set consistent z-axis limits for both subplots
        z_min = min(c2_experimental.min(), c2_fitted.min())
        z_max = max(c2_experimental.max(), c2_fitted.max())
        z_range = z_max - z_min
        z_margin = 0.1 * z_range

        ax1.set_zlim(z_min - z_margin, z_max + z_margin)
        ax2.set_zlim(z_min - z_margin, z_max + z_margin)

        # Add grid for better depth perception
        ax1.grid(True, alpha=0.3)
        ax2.grid(True, alpha=0.3)

        # Improve layout
        plt.tight_layout()

        # =================================================================
        # Save the plot
        # =================================================================
        outdir = Path(outdir)
        outdir.mkdir(parents=True, exist_ok=True)

        # Generate filename based on angle
        filename = f"3d_surface_c2_phi_{phi_angle:.1f}deg.png"
        output_path = outdir / filename

        # Save with high quality
        _save_plot_with_logging(
            fig,
            output_path,
            "3D surface",
            dpi=300,
            bbox_inches="tight",
            facecolor="white",
            edgecolor="none",
            format="png",
        )

        # =================================================================
        # Create additional residuals 3D plot if fitted data available
        # =================================================================
        try:
            residuals = c2_experimental - c2_fitted

            fig_res = plt.figure(figsize=(10, 7))
            ax_res = fig_res.add_subplot(111, projection="3d")

            # Plot residuals surface
            surf_res = ax_res.plot_surface(
                T1,
                T2,
                residuals,
                # Red-Blue colormap centered at zero
                cmap=plt.cm.get_cmap("RdBu_r"),
                alpha=0.8,
                linewidth=0,
                antialiased=True,
                vmin=-np.max(np.abs(residuals)),
                vmax=np.max(np.abs(residuals)),
            )

            # Add zero plane for reference
            zero_plane = np.zeros_like(residuals)
            ax_res.plot_surface(
                T1, T2, zero_plane, color="black", alpha=0.1, linewidth=0
            )

            # Customize residuals plot
            ax_res.set_xlabel(r"$t_1$ (time units)", fontsize=12, labelpad=10)
            ax_res.set_ylabel(r"$t_2$ (time units)", fontsize=12, labelpad=10)
            ax_res.set_zlabel("Residuals (exp - fitted)", fontsize=12, labelpad=8)
            ax_res.set_title(
                f"Residuals (Experimental - Fitted)\nφ = {phi_angle:.1f}°",
                fontsize=14,
                pad=20,
            )
            ax_res.view_init(elev=20, azim=120)
            ax_res.grid(True, alpha=0.3)

            # Add colorbar
            cbar_res = fig_res.colorbar(surf_res, shrink=0.5, aspect=20, pad=0.1)
            cbar_res.set_label("Residuals", fontsize=10)

            plt.tight_layout()

            # Save residuals plot
            residuals_filename = f"3d_residuals_phi_{phi_angle:.1f}deg.png"
            residuals_path = outdir / residuals_filename

            _save_plot_with_logging(
                fig_res,
                residuals_path,
                "3D residuals",
                dpi=300,
                bbox_inches="tight",
                facecolor="white",
                edgecolor="none",
                format="png",
            )

        except Exception as e:
            logger.warning(f"Failed to create residuals plot: {e}")

        return True

    except ImportError as e:
        logger.error(f"Missing required 3D plotting dependencies: {e}")
        logger.error("Install with: pip install matplotlib[3d]")
        return False

    except Exception as e:
        logger.error(f"Failed to create 3D surface plot: {e}")
        import traceback

        logger.debug(f"Full traceback: {traceback.format_exc()}")
        return False
