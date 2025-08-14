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
Institution: Argonne National Laboratory & University of Chicago
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
from homodyne.core.io_utils import ensure_dir, save_fig

# Set up logging
logger = logging.getLogger(__name__)

# Try to import optional dependencies for advanced plotting
try:
    import arviz as az
    # pandas is imported locally when needed
    ARVIZ_AVAILABLE = True
    logger.info("ArviZ imported - MCMC corner plots available")
except ImportError:
    ARVIZ_AVAILABLE = False
    logger.warning("ArviZ not available. Install with: pip install arviz")

try:
    # corner is imported locally when needed
    import corner
    CORNER_AVAILABLE = True
    logger.info("corner package imported - Enhanced corner plots available")
except ImportError:
    CORNER_AVAILABLE = False
    logger.warning("corner package not available. Install with: pip install corner")


def get_plot_config(config: Optional[Dict] = None) -> Dict[str, Any]:
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


def setup_matplotlib_style(plot_config: Dict[str, Any]) -> None:
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


def plot_c2_heatmaps(
    exp: np.ndarray,
    theory: np.ndarray,
    phi_angles: np.ndarray,
    outdir: Union[str, Path],
    config: Optional[Dict] = None,
    t2: Optional[np.ndarray] = None,
    t1: Optional[np.ndarray] = None,
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

    # Get plotting configuration
    plot_config = get_plot_config(config)
    setup_matplotlib_style(plot_config)

    # Ensure output directory exists
    outdir = ensure_dir(outdir)

    # Validate exp and theory inputs
    try:
        if exp is None or not hasattr(exp, 'shape'):
            logger.error("Experimental data must be a numpy array with shape attribute")
            return False
        if theory is None or not hasattr(theory, 'shape'):
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
            f"Number of angles ({len(phi_angles)}) doesn't match data shape ({exp.shape[0]})"
        )
        return False

    # Generate default axes if not provided
    if t2 is None:
        t2 = np.arange(exp.shape[1])
    if t1 is None:
        t1 = np.arange(exp.shape[2])
    
    # Type assertion to help Pylance understand these are no longer None
    assert t2 is not None and t1 is not None

    # Calculate residuals
    residuals = exp - theory

    # Create plots for each phi angle
    success_count = 0

    for i, phi in enumerate(phi_angles):
        try:
            # Create figure with subplots
            fig = plt.figure(figsize=plot_config["figure_size"])
            gs = gridspec.GridSpec(
                2,
                3,
                height_ratios=[1, 1],
                width_ratios=[1, 1, 0.05],
                hspace=0.3,
                wspace=0.3,
            )

            # Experimental data heatmap
            ax1 = fig.add_subplot(gs[0, 0])
            im1 = ax1.imshow(
                exp[i],
                aspect="auto",
                origin="lower",
                extent=(
                    float(t1[0]),
                    float(t1[-1]),
                    float(t2[0]),
                    float(t2[-1]),
                ),
                cmap="viridis",
            )
            ax1.set_title(f"Experimental C₂\nφ = {phi:.1f}°")
            ax1.set_xlabel("t₁")
            ax1.set_ylabel("t₂")

            # Theoretical data heatmap
            ax2 = fig.add_subplot(gs[0, 1])
            im2 = ax2.imshow(
                theory[i],
                aspect="auto",
                origin="lower",
                extent=(
                    float(t1[0]),
                    float(t1[-1]),
                    float(t2[0]),
                    float(t2[-1]),
                ),
                cmap="viridis",
            )
            ax2.set_title(f"Theoretical C₂\nφ = {phi:.1f}°")
            ax2.set_xlabel("t₁")
            ax2.set_ylabel("t₂")

            # Shared colorbar for exp and theory
            cbar_ax1 = fig.add_subplot(gs[0, 2])
            vmin = min(np.min(exp[i]), np.min(theory[i]))
            vmax = max(np.max(exp[i]), np.max(theory[i]))
            im1.set_clim(vmin, vmax)
            im2.set_clim(vmin, vmax)
            plt.colorbar(im1, cax=cbar_ax1, label="C₂")

            # Residuals heatmap (spans both columns)
            ax3 = fig.add_subplot(gs[1, :2])
            im3 = ax3.imshow(
                residuals[i],
                aspect="auto",
                origin="lower",
                extent=(
                    float(t1[0]),
                    float(t1[-1]),
                    float(t2[0]),
                    float(t2[-1]),
                ),
                cmap="RdBu_r",
            )
            ax3.set_title(f"Residuals (Exp - Theory)\nφ = {phi:.1f}°")
            ax3.set_xlabel("t₁")
            ax3.set_ylabel("t₂")

            # Residuals colorbar
            cbar_ax2 = fig.add_subplot(gs[1, 2])
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
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
            )

            # Save the plot
            filename = f"c2_heatmaps_phi_{phi:.1f}deg.{plot_config['plot_format']}"
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


def plot_parameter_evolution(
    best_params: Dict[str, float],
    bounds: List[Dict],
    outdir: Union[str, Path],
    config: Optional[Dict] = None,
    initial_params: Optional[Dict[str, float]] = None,
    optimization_history: Optional[List[Dict]] = None,
) -> bool:
    """
    Create bar chart or corner plot comparing initial parameters, best parameters, and bounds.

    Args:
        best_params (Dict[str, float]): Best-fit parameters from optimization
        bounds (List[Dict]): Parameter bounds from configuration
        outdir (Union[str, Path]): Output directory for saved plots
        config (Optional[Dict]): Configuration dictionary
        initial_params (Optional[Dict[str, float]]): Initial parameter values
        optimization_history (Optional[List[Dict]]): History of optimization iterations

    Returns:
        bool: True if plot was created successfully
    """
    logger.info("Creating parameter evolution plot")

    # Get plotting configuration
    plot_config = get_plot_config(config)
    setup_matplotlib_style(plot_config)

    # Ensure output directory exists
    outdir = ensure_dir(outdir)

    try:
        # Extract parameter information
        param_names = [bound["name"] for bound in bounds]
        param_units = [bound.get("unit", "") for bound in bounds]
        lower_bounds = [bound["min"] for bound in bounds]
        upper_bounds = [bound["max"] for bound in bounds]

        # Get parameter values
        best_values = [best_params.get(name, 0) for name in param_names]
        initial_values = [
            initial_params.get(name, 0) if initial_params else 0 for name in param_names
        ]

        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(
            2,
            1,
            figsize=(
                plot_config["figure_size"][0],
                plot_config["figure_size"][1] * 1.2,
            ),
        )

        # Plot 1: Parameter comparison bar chart
        x_pos = np.arange(len(param_names))
        width = 0.25

        # Normalize values for log-scale parameters
        normalized_best = []
        normalized_initial = []
        normalized_lower = []
        normalized_upper = []

        for i, bound in enumerate(bounds):
            if bound.get("type") == "log-uniform":
                # Use log scale for log-uniform parameters
                normalized_best.append(np.log10(max(abs(best_values[i]), 1e-10)))
                normalized_initial.append(np.log10(max(abs(initial_values[i]), 1e-10)))
                normalized_lower.append(np.log10(max(abs(lower_bounds[i]), 1e-10)))
                normalized_upper.append(np.log10(max(abs(upper_bounds[i]), 1e-10)))
            else:
                # Use linear scale
                normalized_best.append(best_values[i])
                normalized_initial.append(initial_values[i])
                normalized_lower.append(lower_bounds[i])
                normalized_upper.append(upper_bounds[i])

        # Create bars
        _bars1 = ax1.bar(
            x_pos - width,
            normalized_initial,
            width,
            label="Initial",
            alpha=0.7,
            color="lightblue",
        )
        bars2 = ax1.bar(
            x_pos,
            normalized_best,
            width,
            label="Best Fit",
            alpha=0.7,
            color="darkblue",
        )
        _bars3 = ax1.bar(
            x_pos + width,
            normalized_lower,
            width,
            label="Lower Bound",
            alpha=0.5,
            color="red",
        )
        _bars4 = ax1.bar(
            x_pos + 1.5 * width,
            normalized_upper,
            width,
            label="Upper Bound",
            alpha=0.5,
            color="green",
        )

        ax1.set_xlabel("Parameters")
        ax1.set_ylabel("Parameter Values")
        ax1.set_title("Parameter Evolution: Initial vs Best Fit vs Bounds")
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(
            [f"{name}\n[{unit}]" for name, unit in zip(param_names, param_units)],
            rotation=45,
            ha="right",
        )
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Add value labels on bars
        def add_value_labels(bars, values):
            for bar, value in zip(bars, values):
                height = bar.get_height()
                # Avoid division by zero if bar width is zero
                bar_width = bar.get_width()
                if bar_width > 0:
                    ax1.annotate(
                        (
                            f"{value:.2e}"
                            if abs(value) < 1e-3 or abs(value) > 1e3
                            else f"{value:.3f}"
                        ),
                        xy=(bar.get_x() + bar_width / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha="center",
                        va="bottom",
                        fontsize=8,
                        rotation=90,
                    )

        add_value_labels(
            bars2, best_values
        )  # Only label best fit values to avoid clutter

        # Plot 2: Optimization history (if available)
        if optimization_history:
            iterations = range(len(optimization_history))
            chi_squared = [
                hist.get("chi_squared", np.nan) for hist in optimization_history
            ]

            ax2.semilogy(
                iterations,
                chi_squared,
                "b-",
                marker="o",
                markersize=3,
                linewidth=1.5,
            )
            ax2.set_xlabel("Optimization Iteration")
            ax2.set_ylabel("χ² Value (log scale)")
            ax2.set_title("Optimization Convergence")
            ax2.grid(True, alpha=0.3)

            # Add final chi-squared value
            if chi_squared and not np.isnan(chi_squared[-1]):
                ax2.text(
                    0.98,
                    0.95,
                    f"Final χ² = {chi_squared[-1]:.6f}",
                    transform=ax2.transAxes,
                    ha="right",
                    va="top",
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
                )
        else:
            # Show parameter values as a bar chart instead of correlation matrix
            # (correlation requires multiple data points, which we don't have)
            bars = ax2.bar(param_names, best_values, alpha=0.7, color="darkgreen")
            ax2.set_title("Best Fit Parameter Values")
            ax2.tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar, value in zip(bars, best_values):
                height = bar.get_height()
                bar_width = bar.get_width()
                if bar_width > 0:  # Avoid division by zero
                    ax2.text(bar.get_x() + bar_width/2., height,
                            f'{value:.3g}',
                            ha='center', va='bottom', fontsize=8)

        plt.tight_layout()

        # Save the plot
        filename = f"parameter_evolution.{plot_config['plot_format']}"
        filepath = outdir / filename

        success = save_fig(
            fig,
            filepath,
            dpi=plot_config["dpi"],
            format=plot_config["plot_format"],
        )
        plt.close(fig)

        if success:
            logger.info("Successfully created parameter evolution plot")
        else:
            logger.error("Failed to save parameter evolution plot")

        return success

    except Exception as e:
        logger.error(f"Error creating parameter evolution plot: {e}")
        plt.close("all")
        return False


def plot_mcmc_corner(
    trace_data: Any,
    outdir: Union[str, Path],
    config: Optional[Dict] = None,
    param_names: Optional[List[str]] = None,
    param_units: Optional[List[str]] = None,
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

    # Get plotting configuration
    plot_config = get_plot_config(config)
    setup_matplotlib_style(plot_config)

    # Ensure output directory exists
    outdir = ensure_dir(outdir)

    try:
        # Handle different trace data formats
        if hasattr(trace_data, "posterior"):
            # ArviZ InferenceData
            samples = trace_data.posterior
        elif isinstance(trace_data, dict):
            # Dictionary of samples
            samples = trace_data
        else:
            # Try to convert to DataFrame
            try:
                if not ARVIZ_AVAILABLE:
                    logger.error("Pandas not available for DataFrame conversion")
                    return False
                import pandas as pd
                samples = pd.DataFrame(trace_data)
            except:
                logger.error("Unsupported trace data format for corner plot")
                return False

        # Create corner plot using ArviZ
        if hasattr(samples, "stack"):
            # ArviZ format - stack chains
            stacked_samples = samples.stack(sample=("chain", "draw"))  # type: ignore
        else:
            stacked_samples = samples

        # Create the corner plot
        if CORNER_AVAILABLE:
            # Use corner package if available (better formatting)
            import corner
            fig = corner.corner(
                stacked_samples,
                labels=[
                    (
                        f"{name}\n[{unit}]"
                        if param_names and param_units
                        else f"Param {i}"
                    )
                    for i, (name, unit) in enumerate(
                        zip(param_names or [], param_units or [])
                    )
                ],
                show_titles=True,
                title_kwargs={"fontsize": 12},
                label_kwargs={"fontsize": 14},
                hist_kwargs={"density": True, "alpha": 0.8},
                contour_kwargs={"colors": ["C0", "C1", "C2"]},
                fill_contours=True,
                plot_contours=True,
            )
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


def plot_diagnostic_summary(
    results: Dict[str, Any],
    outdir: Union[str, Path],
    config: Optional[Dict] = None,
) -> bool:
    """
    Create a comprehensive diagnostic summary plot combining multiple visualizations.

    Args:
        results (Dict[str, Any]): Complete analysis results dictionary
        outdir (Union[str, Path]): Output directory for saved plots
        config (Optional[Dict]): Configuration dictionary

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
                method_name = key.replace("_chi_squared", "").replace("_chi2", "")
                methods.append(method_name.replace("_", " ").title())
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
            for bar, value in zip(bars, chi2_values):
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
        if "parameter_uncertainties" in results:
            uncertainties = results["parameter_uncertainties"]
            param_names = list(uncertainties.keys())
            uncertainty_values = list(uncertainties.values())

            ax2.barh(param_names, uncertainty_values, alpha=0.7)
            ax2.set_xlabel("Parameter Uncertainty")
            ax2.set_title("Parameter Uncertainties")

        # Plot 3: Convergence diagnostics (if MCMC results available)
        ax3 = fig.add_subplot(gs[0, 2])
        if "mcmc_diagnostics" in results and ARVIZ_AVAILABLE:
            # Plot R-hat values
            diagnostics = results["mcmc_diagnostics"]
            if "r_hat" in diagnostics:
                param_names = list(diagnostics["r_hat"].keys())
                r_hat_values = list(diagnostics["r_hat"].values())

                colors = [
                    "green" if r < 1.1 else "orange" if r < 1.2 else "red"
                    for r in r_hat_values
                ]
                ax3.barh(param_names, r_hat_values, color=colors, alpha=0.7)
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

        # Plot 4: Residuals analysis (if available)
        ax4 = fig.add_subplot(gs[1, :])
        if "residuals" in results:
            residuals = results["residuals"]
            if isinstance(residuals, np.ndarray):
                # Flatten residuals for histogram
                flat_residuals = residuals.flatten()

                # Create histogram and Q-Q plot side by side
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
                    # If sigma is effectively zero, just show the mean as a vertical line
                    ax4.axvline(float(mu), color='red', linestyle='--', linewidth=2, 
                               label=f"Mean={mu:.3e} (σ≈0)")
                    logger.warning("Standard deviation is very small, showing mean line instead of normal distribution")

                ax4.set_xlabel("Residual Value")
                ax4.set_ylabel("Density")
                ax4.set_title("Residuals Distribution Analysis")
                ax4.legend()
                ax4.grid(True, alpha=0.3)

        # Add overall title
        fig.suptitle("Analysis Diagnostic Summary", fontsize=18, y=0.98)

        # Save the plot
        filename = f"diagnostic_summary.{plot_config['plot_format']}"
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
    results: Dict[str, Any],
    outdir: Union[str, Path],
    config: Optional[Dict] = None,
) -> Dict[str, bool]:
    """
    Create all available plots based on the results dictionary.

    Args:
        results (Dict[str, Any]): Complete analysis results dictionary
        outdir (Union[str, Path]): Output directory for saved plots
        config (Optional[Dict]): Configuration dictionary

    Returns:
        Dict[str, bool]: Success status for each plot type
    """
    logger.info("Creating all available plots")

    plot_status = {}

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

    # Parameter evolution plot
    if all(key in results for key in ["best_parameters", "parameter_bounds"]):
        plot_status["parameter_evolution"] = plot_parameter_evolution(
            results["best_parameters"],
            results["parameter_bounds"],
            outdir,
            config,
            initial_params=results.get("initial_parameters"),
            optimization_history=results.get("optimization_history"),
        )

    # MCMC corner plot (if trace data available)
    if "mcmc_trace" in results:
        plot_status["mcmc_corner"] = plot_mcmc_corner(
            results["mcmc_trace"],
            outdir,
            config,
            param_names=results.get("parameter_names"),
            param_units=results.get("parameter_units"),
        )

    # Diagnostic summary
    plot_status["diagnostic_summary"] = plot_diagnostic_summary(results, outdir, config)

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
    exp_data = 1 + 0.5 * np.random.exponential(
        1, (n_angles, n_t2, n_t1)
    )
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

        # Test parameter evolution
        best_params = {"D0": 1000, "alpha": -0.5, "beta": 0.3}
        bounds = [
            {"name": "D0", "min": 1, "max": 10000, "unit": "Å²/s"},
            {"name": "alpha", "min": -2, "max": 2, "unit": "dimensionless"},
            {"name": "beta", "min": -1, "max": 1, "unit": "dimensionless"},
        ]

        success2 = plot_parameter_evolution(best_params, bounds, tmp_dir, test_config)
        print(f"Parameter evolution: {'Success' if success2 else 'Failed'}")

        print("Test completed!")
