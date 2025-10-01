"""
CLI Visualization Module
========================

Visualization and plotting functions for the homodyne CLI interface.

This module handles the generation of plots, heatmaps, and visualizations
for both simulated and experimental data analysis results.
"""

import argparse
import logging
from pathlib import Path
from typing import Any

import numpy as np

# Module-level logger
logger = logging.getLogger(__name__)


def generate_c2_heatmap_plots(
    c2_plot_data: np.ndarray,
    phi_angles: np.ndarray,
    t1: np.ndarray,
    t2: np.ndarray,
    data_type: str,
    args: argparse.Namespace,
    simulated_dir: Path,
) -> int:
    """
    Generate C2 heatmap plots for all phi angles.

    Parameters
    ----------
    c2_plot_data : np.ndarray
        C2 data to plot
    phi_angles : np.ndarray
        Array of phi angles
    t1 : np.ndarray
        Time array 1
    t2 : np.ndarray
        Time array 2
    data_type : str
        Type of data ("theoretical" or "fitted")
    args : argparse.Namespace
        Command-line arguments
    simulated_dir : Path
        Output directory for plots

    Returns
    -------
    int
        Number of successfully generated plots
    """
    # Import matplotlib for custom plotting
    try:
        import matplotlib.colors  # noqa: F401
        import matplotlib.pyplot as plt
    except ImportError:
        logger.error("❌ Failed to import matplotlib")
        logger.error("Please ensure matplotlib is available")
        raise

    logger.info("Generating C2 theoretical heatmap plots...")
    success_count = 0

    try:
        for i, phi_angle in enumerate(phi_angles):
            # Get C2 data for this angle (theoretical or fitted)
            c2_data = c2_plot_data[i]

            # Calculate color scale: vmin=min, vmax=max value in this angle's data
            vmin = np.min(c2_data)
            vmax = np.max(c2_data)

            # Handle case where vmin == vmax (constant data)
            if np.abs(vmax - vmin) < 1e-10:
                # Add small epsilon to avoid singular transformation
                vmin = vmin - 0.01 if vmin != 0 else -0.01
                vmax = vmax + 0.01 if vmax != 0 else 0.01

            # Create figure for single heatmap
            fig, ax = plt.subplots(figsize=(8, 6))

            # Create heatmap with custom color scale
            # Note: With indexing='ij' in meshgrid:
            #   t1 varies along rows (axis 0), constant along columns
            #   t2 varies along columns (axis 1), constant along rows
            # So extent should be: (t1_min, t1_max, t2_min, t2_max)
            im = ax.imshow(
                c2_data,
                aspect="equal",
                origin="lower",
                extent=(t1[0, 0], t1[-1, 0], t2[0, 0], t2[0, -1]),
                vmin=vmin,
                vmax=vmax,
                cmap="viridis",
            )

            # Add colorbar with appropriate label
            cbar = plt.colorbar(im, ax=ax)
            if data_type == "fitted":
                cbar.set_label("C₂ Fitted (t₁, t₂)", fontsize=12)
            else:
                cbar.set_label("C₂(t₁, t₂)", fontsize=12)

            # Set labels and title
            ax.set_xlabel("t₁ (s)", fontsize=12)
            ax.set_ylabel("t₂ (s)", fontsize=12)

            if data_type == "fitted":
                ax.set_title(
                    f"Fitted C₂ Correlation Function (φ = {phi_angle:.1f}°)\nfitted = {
                        args.contrast
                    } * theory + {args.offset}",
                    fontsize=14,
                )
                filename = f"simulated_c2_fitted_phi_{phi_angle:.1f}deg.png"
            else:
                ax.set_title(
                    f"Theoretical C₂ Correlation Function (φ = {phi_angle:.1f}°)",
                    fontsize=14,
                )
                filename = f"simulated_c2_theoretical_phi_{phi_angle:.1f}deg.png"

            # Save the plot
            filepath = simulated_dir / filename

            plt.tight_layout()
            plt.savefig(filepath, dpi=300, bbox_inches="tight")
            plt.close(fig)

            logger.debug(f"✓ Saved plot: {filename}")
            success_count += 1

    except Exception as e:
        logger.error(f"❌ Error generating heatmap plots: {e}")
        import traceback

        logger.debug(f"Full traceback: {traceback.format_exc()}")

    logger.info(f"✓ Generated {success_count} out of {len(phi_angles)} heatmap plots")
    return success_count


def generate_classical_plots(
    analyzer,
    result_dict: dict[str, Any],
    phi_angles: np.ndarray,
    c2_exp: np.ndarray,
    output_dir: Path,
) -> None:
    """
    Generate plots for classical optimization results.

    Parameters
    ----------
    analyzer : HomodyneAnalysisCore
        Analysis engine
    result_dict : Dict[str, Any]
        Classical optimization results
    phi_angles : np.ndarray
        Array of phi angles
    c2_exp : np.ndarray
        Experimental correlation data
    output_dir : Path
        Directory for saving plots
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.error("❌ Failed to import matplotlib for plotting")
        return

    if "parameters" not in result_dict:
        logger.warning("⚠️  No parameters found in classical results for plotting")
        return

    parameters = result_dict["parameters"]

    try:
        # Create plots directory
        plots_dir = output_dir / "classical_plots"
        plots_dir.mkdir(parents=True, exist_ok=True)

        # Generate fitted vs experimental comparison
        logger.info("Generating classical optimization plots...")

        # Calculate theoretical C2 using optimized parameters
        c2_theoretical_raw = analyzer.calculate_c2_nonequilibrium_laminar_parallel(
            parameters, phi_angles
        )

        # Scale theoretical to match experimental intensities
        # Solve: y_exp = contrast * y_theory + offset (least squares)
        num_angles = len(phi_angles)
        c2_theoretical_scaled = np.zeros_like(c2_exp)
        scaling_params = []

        for i in range(num_angles):
            # Flatten arrays for least squares
            theory_flat = c2_theoretical_raw[i].flatten()
            exp_flat = c2_exp[i].flatten()

            # Solve: exp = contrast * theory + offset
            # Build design matrix A = [theory, ones]
            A = np.column_stack([theory_flat, np.ones_like(theory_flat)])
            # Solve: A @ [contrast, offset] = exp
            solution, _, _, _ = np.linalg.lstsq(A, exp_flat, rcond=None)
            contrast, offset = solution

            # Apply scaling
            c2_theoretical_scaled[i] = contrast * c2_theoretical_raw[i] + offset
            scaling_params.append((contrast, offset))

        # Create comparison plots for each phi angle
        for i, phi in enumerate(phi_angles):
            contrast, offset = scaling_params[i]
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

            # Plot experimental data
            im1 = ax1.imshow(c2_exp[i], cmap="viridis", aspect="equal", origin="lower")
            ax1.set_title(f"Experimental C₂ (φ={phi:.1f}°)")
            ax1.set_xlabel("t₁ (frames)")
            ax1.set_ylabel("t₂ (frames)")
            plt.colorbar(im1, ax=ax1, label="Intensity")

            # Plot scaled theoretical fit
            im2 = ax2.imshow(
                c2_theoretical_scaled[i], cmap="viridis", aspect="equal", origin="lower"
            )
            ax2.set_title(
                f"Classical Fit (φ={phi:.1f}°)\nC={contrast:.2e}, B={offset:.2e}"
            )
            ax2.set_xlabel("t₁ (frames)")
            ax2.set_ylabel("t₂ (frames)")
            plt.colorbar(im2, ax=ax2, label="Intensity")

            # Plot residuals (now correctly scaled)
            residuals = c2_exp[i] - c2_theoretical_scaled[i]
            vmax = np.max(np.abs(residuals))
            im3 = ax3.imshow(
                residuals, cmap="RdBu_r", aspect="equal", origin="lower", vmin=-vmax, vmax=vmax
            )
            ax3.set_title(f"Residuals (φ={phi:.1f}°)")
            ax3.set_xlabel("t₁ (frames)")
            ax3.set_ylabel("t₂ (frames)")
            plt.colorbar(im3, ax=ax3, label="Δ Intensity")

            plt.tight_layout()
            plot_file = plots_dir / f"classical_comparison_phi_{phi:.1f}deg.png"
            plt.savefig(plot_file, dpi=300, bbox_inches="tight")
            plt.close(fig)

        logger.info(f"✓ Classical plots saved to: {plots_dir}")

    except Exception as e:
        logger.error(f"❌ Error generating classical plots: {e}")


def generate_robust_plots(
    analyzer,
    result_dict: dict[str, Any],
    phi_angles: np.ndarray,
    c2_exp: np.ndarray,
    output_dir: Path,
) -> None:
    """
    Generate plots for robust optimization results.

    Parameters
    ----------
    analyzer : HomodyneAnalysisCore
        Analysis engine
    result_dict : Dict[str, Any]
        Robust optimization results
    phi_angles : np.ndarray
        Array of phi angles
    c2_exp : np.ndarray
        Experimental correlation data
    output_dir : Path
        Directory for saving plots
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.error("❌ Failed to import matplotlib for plotting")
        return

    if "parameters" not in result_dict:
        logger.warning("⚠️  No parameters found in robust results for plotting")
        return

    parameters = result_dict["parameters"]

    try:
        # Create plots directory
        plots_dir = output_dir / "robust_plots"
        plots_dir.mkdir(parents=True, exist_ok=True)

        # Generate fitted vs experimental comparison
        logger.info("Generating robust optimization plots...")

        # Calculate theoretical C2 using optimized parameters
        c2_theoretical_raw = analyzer.calculate_c2_nonequilibrium_laminar_parallel(
            parameters, phi_angles
        )

        # Scale theoretical to match experimental intensities
        # Solve: y_exp = contrast * y_theory + offset (least squares)
        num_angles = len(phi_angles)
        c2_theoretical_scaled = np.zeros_like(c2_exp)
        scaling_params = []

        for i in range(num_angles):
            # Flatten arrays for least squares
            theory_flat = c2_theoretical_raw[i].flatten()
            exp_flat = c2_exp[i].flatten()

            # Solve: exp = contrast * theory + offset
            # Build design matrix A = [theory, ones]
            A = np.column_stack([theory_flat, np.ones_like(theory_flat)])
            # Solve: A @ [contrast, offset] = exp
            solution, _, _, _ = np.linalg.lstsq(A, exp_flat, rcond=None)
            contrast, offset = solution

            # Apply scaling
            c2_theoretical_scaled[i] = contrast * c2_theoretical_raw[i] + offset
            scaling_params.append((contrast, offset))

        # Create comparison plots for each phi angle
        for i, phi in enumerate(phi_angles):
            contrast, offset = scaling_params[i]
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

            # Plot experimental data
            im1 = ax1.imshow(c2_exp[i], cmap="viridis", aspect="equal", origin="lower")
            ax1.set_title(f"Experimental C₂ (φ={phi:.1f}°)")
            ax1.set_xlabel("t₁ (frames)")
            ax1.set_ylabel("t₂ (frames)")
            plt.colorbar(im1, ax=ax1, label="Intensity")

            # Plot scaled theoretical fit
            im2 = ax2.imshow(
                c2_theoretical_scaled[i], cmap="viridis", aspect="equal", origin="lower"
            )
            ax2.set_title(
                f"Robust Fit (φ={phi:.1f}°)\nC={contrast:.2e}, B={offset:.2e}"
            )
            ax2.set_xlabel("t₁ (frames)")
            ax2.set_ylabel("t₂ (frames)")
            plt.colorbar(im2, ax=ax2, label="Intensity")

            # Plot residuals (now correctly scaled)
            residuals = c2_exp[i] - c2_theoretical_scaled[i]
            vmax = np.max(np.abs(residuals))
            im3 = ax3.imshow(
                residuals, cmap="RdBu_r", aspect="equal", origin="lower", vmin=-vmax, vmax=vmax
            )
            ax3.set_title(f"Residuals (φ={phi:.1f}°)")
            ax3.set_xlabel("t₁ (frames)")
            ax3.set_ylabel("t₂ (frames)")
            plt.colorbar(im3, ax=ax3, label="Δ Intensity")

            plt.tight_layout()
            plot_file = plots_dir / f"robust_comparison_phi_{phi:.1f}deg.png"
            plt.savefig(plot_file, dpi=300, bbox_inches="tight")
            plt.close(fig)

        logger.info(f"✓ Robust plots saved to: {plots_dir}")

    except Exception as e:
        logger.error(f"❌ Error generating robust plots: {e}")


def generate_comparison_plots(
    analyzer,
    classical_result: dict[str, Any] | None,
    robust_result: dict[str, Any] | None,
    phi_angles: np.ndarray,
    c2_exp: np.ndarray,
    output_dir: Path,
) -> None:
    """
    Generate comparison plots between classical and robust optimization results.

    Parameters
    ----------
    analyzer : HomodyneAnalysisCore
        Analysis engine
    classical_result : Optional[Dict[str, Any]]
        Classical optimization results
    robust_result : Optional[Dict[str, Any]]
        Robust optimization results
    phi_angles : np.ndarray
        Array of phi angles
    c2_exp : np.ndarray
        Experimental correlation data
    output_dir : Path
        Directory for saving plots
    """
    if not classical_result or not robust_result:
        logger.info(
            "⚠️  Skipping comparison plots - need both classical and robust results"
        )
        return

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.error("❌ Failed to import matplotlib for plotting")
        return

    try:
        # Create plots directory
        plots_dir = output_dir / "comparison_plots"
        plots_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Generating method comparison plots...")

        # Get parameters from both methods
        classical_params = classical_result["parameters"]
        robust_params = robust_result["parameters"]

        # Calculate theoretical C2 for both methods
        c2_classical = analyzer.calculate_correlation_function(
            classical_params, phi_angles
        )
        c2_robust = analyzer.calculate_correlation_function(robust_params, phi_angles)

        # Create comparison plots for each phi angle
        for i, phi in enumerate(phi_angles):
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))

            # Row 1: Classical results
            im1 = axes[0, 0].imshow(
                c2_exp[i], cmap="viridis", aspect="equal", origin="lower"
            )
            axes[0, 0].set_title(f"Experimental C₂ (φ={phi:.1f}°)")
            plt.colorbar(im1, ax=axes[0, 0])

            im2 = axes[0, 1].imshow(
                c2_classical[i], cmap="viridis", aspect="equal", origin="lower"
            )
            axes[0, 1].set_title(
                f"Classical Fit (χ²={classical_result['chi_squared']:.4f})"
            )
            plt.colorbar(im2, ax=axes[0, 1])

            residuals_classical = c2_exp[i] - c2_classical[i]
            im3 = axes[0, 2].imshow(
                residuals_classical, cmap="RdBu_r", aspect="equal", origin="lower"
            )
            axes[0, 2].set_title("Classical Residuals")
            plt.colorbar(im3, ax=axes[0, 2])

            # Row 2: Robust results
            axes[1, 0].imshow(c2_exp[i], cmap="viridis", aspect="equal", origin="lower")
            axes[1, 0].set_title(f"Experimental C₂ (φ={phi:.1f}°)")

            im5 = axes[1, 1].imshow(
                c2_robust[i], cmap="viridis", aspect="equal", origin="lower"
            )
            axes[1, 1].set_title(f"Robust Fit (χ²={robust_result['chi_squared']:.4f})")
            plt.colorbar(im5, ax=axes[1, 1])

            residuals_robust = c2_exp[i] - c2_robust[i]
            im6 = axes[1, 2].imshow(
                residuals_robust, cmap="RdBu_r", aspect="equal", origin="lower"
            )
            axes[1, 2].set_title("Robust Residuals")
            plt.colorbar(im6, ax=axes[1, 2])

            # Set common axis labels
            for ax in axes.flat:
                ax.set_xlabel("t₁")
                ax.set_ylabel("t₂")

            plt.tight_layout()
            plot_file = plots_dir / f"method_comparison_phi_{phi:.1f}deg.png"
            plt.savefig(plot_file, dpi=300, bbox_inches="tight")
            plt.close(fig)

        # Create parameter comparison plot
        fig, ax = plt.subplots(figsize=(10, 6))
        param_names = analyzer.config.get(
            "parameter_names", [f"p{i}" for i in range(len(classical_params))]
        )

        x = np.arange(len(param_names))
        width = 0.35

        ax.bar(x - width / 2, classical_params, width, label="Classical", alpha=0.8)
        ax.bar(x + width / 2, robust_params, width, label="Robust", alpha=0.8)

        ax.set_xlabel("Parameters")
        ax.set_ylabel("Parameter Values")
        ax.set_title("Parameter Comparison: Classical vs Robust")
        ax.set_xticks(x)
        ax.set_xticklabels(param_names, rotation=45)
        ax.legend()

        plt.tight_layout()
        param_plot_file = plots_dir / "parameter_comparison.png"
        plt.savefig(param_plot_file, dpi=300, bbox_inches="tight")
        plt.close(fig)

        logger.info(f"✓ Comparison plots saved to: {plots_dir}")

    except Exception as e:
        logger.error(f"❌ Error generating comparison plots: {e}")


def save_individual_method_results(
    results: dict[str, Any],
    method_name: str,
    analyzer,
    phi_angles: np.ndarray,
    c2_exp: np.ndarray,
    output_dir: Path,
) -> None:
    """
    Save individual method results with plots and data files.

    Parameters
    ----------
    results : Dict[str, Any]
        Optimization results
    method_name : str
        Name of the optimization method
    analyzer : HomodyneAnalysisCore
        Analysis engine
    phi_angles : np.ndarray
        Array of phi angles
    c2_exp : np.ndarray
        Experimental correlation data
    output_dir : Path
        Directory for saving results
    """
    try:
        # Create method-specific directory
        method_dir = output_dir / f"{method_name}_results"
        method_dir.mkdir(parents=True, exist_ok=True)

        # Calculate theoretical fit with scaling
        parameters = results["parameters"]
        c2_theoretical_raw = analyzer.calculate_c2_nonequilibrium_laminar_parallel(
            parameters, phi_angles
        )

        # Calculate scaling parameters and scaled theoretical
        num_angles = len(phi_angles)
        c2_theoretical_scaled = np.zeros_like(c2_exp)
        contrast_params = np.zeros(num_angles)
        offset_params = np.zeros(num_angles)

        for i in range(num_angles):
            # Solve: exp = contrast * theory + offset
            theory_flat = c2_theoretical_raw[i].flatten()
            exp_flat = c2_exp[i].flatten()
            A = np.column_stack([theory_flat, np.ones_like(theory_flat)])
            solution, _, _, _ = np.linalg.lstsq(A, exp_flat, rcond=None)
            contrast, offset = solution
            c2_theoretical_scaled[i] = contrast * c2_theoretical_raw[i] + offset
            contrast_params[i] = contrast
            offset_params[i] = offset

        # Save comprehensive results data
        results_file = method_dir / f"{method_name}_results.npz"
        np.savez_compressed(
            results_file,
            parameters=results["parameters"],
            chi_squared=results["chi_squared"],
            phi_angles=phi_angles,
            c2_experimental=c2_exp,
            c2_theoretical_raw=c2_theoretical_raw,
            c2_theoretical_scaled=c2_theoretical_scaled,
            contrast_params=contrast_params,
            offset_params=offset_params,
            residuals=c2_exp - c2_theoretical_scaled,
        )

        # Generate method-specific plots
        if method_name == "classical":
            generate_classical_plots(analyzer, results, phi_angles, c2_exp, output_dir)
        elif method_name == "robust":
            generate_robust_plots(analyzer, results, phi_angles, c2_exp, output_dir)

        logger.info(f"✓ {method_name.capitalize()} results saved to: {method_dir}")

    except Exception as e:
        logger.error(f"❌ Error saving {method_name} results: {e}")
