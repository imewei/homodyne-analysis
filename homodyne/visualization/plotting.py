"""
Plotting Functions for Homodyne Scattering Analysis
===================================================

This module provides specialized plotting functions for visualizing results from
homodyne scattering analysis in XPCS (X-ray Photon Correlation Spectroscopy).

The plotting functions are designed to work with the configuration system and
provide publication-quality plots for:
- C2 correlation function heatmaps with experimental vs theoretical comparison
- Parameter evolution during optimization

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

from homodyne.core.io_utils import ensure_dir
from homodyne.core.io_utils import save_fig

# Set up logging
logger = logging.getLogger(__name__)


# Removed MCMC plotting dependencies (arviz, corner) - no longer used


def get_plot_config(config: dict | None = None) -> dict[str, Any]:
    """
    Extract plotting configuration from the main config dictionary.

    Args:
        config (dict | None):Main configuration dictionary

    Returns:
        dict[str, Any]: Plotting configuration with defaults
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
        plot_config (dict[str, Any]): Plotting configuration
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
        outdir (str | Path): Output directory for saved plots
        config (dict | None):Configuration dictionary
        t2 (np.ndarray | None): Time lag values (t₂) for y-axis
        t1 (np.ndarray | None): Delay time values (t₁) for x-axis
        method_name (str | None): Optimization method name for filename prefix

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
    # This determines the optimal scaling relationship g₂ = offset + contrast * g₁
    # for visualization purposes, ensuring plotted data is meaningful.
    fitted = np.zeros_like(theory)

    # SCALING OPTIMIZATION: ALWAYS PERFORMED
    # This scaling optimization is essential for meaningful plots because:
    # 1. Raw theoretical and experimental data may have different scales
    # 2. Systematic offsets need to be accounted for in visualization
    # 3. Residual plots (exp - fitted) are only meaningful with proper scaling
    # 4. Consistent with chi-squared calculation methodology used in analysis
    # The relationship g₂ = offset + contrast * g₁ is fitted for each angle
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


def plot_diagnostic_summary(
    results: dict[str, Any],
    outdir: str | Path,
    config: dict | None = None,
    method_name: str | None = None,
) -> bool:
    """
    Create a comprehensive diagnostic summary plot combining multiple visualizations.

    Generates a 2x3 grid layout containing:
    - Method comparison with chi-squared values
    - Parameter uncertainties visualization
    - Residuals distribution analysis with normal distribution overlay

    Features adaptive content with appropriate placeholders when data is unavailable,
    professional formatting with consistent styling, and cross-method comparison
    capabilities for quality assessment.

    Args:
        results (dict[str, Any]): Complete analysis results dictionary
        outdir (str | Path): Output directory for saved plots
        config (dict | None):Configuration dictionary
        method_name (str | None): Optimization method name for filename prefix

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
                ax2.set_xlabel("Parameter Uncertainty (sigma)")
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

        # Plot 3: Convergence diagnostics placeholder
        ax3 = fig.add_subplot(gs[0, 2])
        # Show placeholder message for convergence diagnostics
        ax3.text(
            0.5,
            0.5,
            "Classical/Robust methods\ndo not require convergence\ndiagnostics",
            ha="center",
            va="center",
            transform=ax3.transAxes,
            fontsize=12,
            color="gray",
        )
        ax3.set_title("Convergence Diagnostics")
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
                        label=f"Normal(mu={mu:.3e}, sigma={sigma:.3e})",
                    )
                else:
                    # If sigma is effectively zero, just show the mean as a
                    # vertical line
                    ax4.axvline(
                        float(mu),
                        color="red",
                        linestyle="--",
                        linewidth=2,
                        label=f"Mean={mu:.3e} (sigma~0)",
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
        results (dict[str, Any]): Complete analysis results dictionary
        outdir (str | Path): Output directory for saved plots
        config (dict | None):Configuration dictionary

    Returns:
        dict[str, bool]: Success status for each plot type
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
