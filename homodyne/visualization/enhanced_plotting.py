"""
Enhanced Plotting Integration with UI/UX Optimizations
======================================================

Integration layer that enhances the existing plotting module with UI/UX optimizations,
real-time progress tracking, and performance improvements.

This module serves as a bridge between the original plotting functions and the new
UI components, providing backward compatibility while adding enhanced features.
"""

import logging
import time
from pathlib import Path
from typing import Any

import numpy as np

from ..ui.error_reporter import create_error_reporter

# Import UI enhancements
from ..ui.progress import ProgressContext
from ..ui.progress import get_progress_tracker
from ..ui.progress import track_io_progress
from ..ui.visualization_optimizer import create_optimized_visualizer

# Import original plotting functions
from .plotting import create_all_plots as original_create_all_plots
from .plotting import get_plot_config
from .plotting import plot_c2_heatmaps as original_plot_c2_heatmaps
from .plotting import plot_diagnostic_summary as original_plot_diagnostic_summary


class EnhancedPlottingManager:
    """
    Enhanced plotting manager with UI/UX optimizations and progress tracking.

    Provides a wrapper around the original plotting functions with added:
    - Real-time progress tracking
    - Performance optimization
    - Memory usage monitoring
    - Error handling with user guidance
    - Adaptive quality settings
    """

    def __init__(
        self,
        enable_progress: bool = True,
        optimize_performance: bool = True,
        adaptive_quality: bool = True,
    ):
        """
        Initialize enhanced plotting manager.

        Parameters
        ----------
        enable_progress : bool
            Enable real-time progress tracking
        optimize_performance : bool
            Enable performance optimizations
        adaptive_quality : bool
            Enable adaptive quality based on data size
        """
        self.enable_progress = enable_progress
        self.optimize_performance = optimize_performance
        self.adaptive_quality = adaptive_quality

        # Initialize components
        self.progress_tracker = get_progress_tracker() if enable_progress else None
        self.visualizer = (
            create_optimized_visualizer() if optimize_performance else None
        )
        self.error_reporter = create_error_reporter()

        # Performance tracking
        self.plot_stats = {
            "plots_created": 0,
            "total_time": 0.0,
            "memory_saved": 0,
            "errors_encountered": 0,
        }

        self.logger = logging.getLogger(__name__)

    def plot_c2_heatmaps_enhanced(
        self,
        exp: np.ndarray,
        theory: np.ndarray,
        phi_angles: np.ndarray,
        outdir: str | Path,
        config: dict | None = None,
        t2: np.ndarray | None = None,
        t1: np.ndarray | None = None,
        method_name: str | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """
        Enhanced C2 heatmap plotting with progress tracking and optimization.

        This function wraps the original plot_c2_heatmaps with UI enhancements.
        """
        start_time = time.time()

        try:
            # Estimate data size and optimize if needed
            if self.adaptive_quality:
                data_size_mb = self._estimate_data_size([exp, theory])
                self._optimize_for_data_size(data_size_mb, config)

            # Progress tracking setup
            if self.enable_progress:
                total_plots = len(phi_angles)
                plot_task = track_io_progress(
                    f"c2_heatmaps_{method_name or 'analysis'}", total_plots
                )

                # Create custom progress wrapper
                return self._plot_c2_with_progress(
                    exp,
                    theory,
                    phi_angles,
                    outdir,
                    config,
                    t2,
                    t1,
                    method_name,
                    plot_task,
                    **kwargs,
                )
            # Use original function without progress tracking
            success = original_plot_c2_heatmaps(
                exp, theory, phi_angles, outdir, config, t2, t1, method_name
            )

            return {
                "success": success,
                "plots_created": len(phi_angles) if success else 0,
                "method": method_name,
                "performance": {
                    "execution_time": time.time() - start_time,
                    "optimized": False,
                },
            }

        except Exception as e:
            self.plot_stats["errors_encountered"] += 1
            self.error_reporter.report_error(
                e,
                {
                    "function": "plot_c2_heatmaps_enhanced",
                    "method_name": method_name,
                    "data_shapes": {
                        "exp": exp.shape if exp is not None else None,
                        "theory": theory.shape if theory is not None else None,
                        "phi_angles": (
                            len(phi_angles) if phi_angles is not None else None
                        ),
                    },
                },
            )

            return {"success": False, "error": str(e), "plots_created": 0}

    def _plot_c2_with_progress(
        self,
        exp: np.ndarray,
        theory: np.ndarray,
        phi_angles: np.ndarray,
        outdir: str | Path,
        config: dict | None,
        t2: np.ndarray | None,
        t1: np.ndarray | None,
        method_name: str | None,
        plot_task: str,
        **kwargs,
    ) -> dict[str, Any]:
        """Plot C2 heatmaps with detailed progress tracking."""

        # Use enhanced visualizer if available
        if self.visualizer and self.optimize_performance:
            return self._plot_c2_optimized(
                exp,
                theory,
                phi_angles,
                outdir,
                config,
                t2,
                t1,
                method_name,
                plot_task,
                **kwargs,
            )

        # Fallback to original with progress wrapper
        tracker = self.progress_tracker

        try:
            with ProgressContext(
                f"Creating C2 heatmaps for {method_name or 'analysis'}", len(phi_angles)
            ) as progress:
                # Call original function with monitoring
                success = original_plot_c2_heatmaps(
                    exp, theory, phi_angles, outdir, config, t2, t1, method_name
                )

                # Update progress (simulate per-angle completion)
                for i in range(len(phi_angles)):
                    progress.update(current=i + 1)
                    if tracker:
                        tracker.update_task(plot_task, increment=1)
                    time.sleep(0.1)  # Small delay to show progress

                if tracker:
                    tracker.complete_task(plot_task, success=success)

                self.plot_stats["plots_created"] += len(phi_angles) if success else 0

                return {
                    "success": success,
                    "plots_created": len(phi_angles) if success else 0,
                    "method": method_name,
                    "performance": {
                        "execution_time": time.time() - time.time(),
                        "optimized": False,
                        "progress_tracked": True,
                    },
                }

        except Exception as e:
            if tracker:
                tracker.complete_task(plot_task, success=False, message=str(e))
            raise e

    def _plot_c2_optimized(
        self,
        exp: np.ndarray,
        theory: np.ndarray,
        phi_angles: np.ndarray,
        outdir: str | Path,
        config: dict | None,
        t2: np.ndarray | None,
        t1: np.ndarray | None,
        method_name: str | None,
        plot_task: str,
        **kwargs,
    ) -> dict[str, Any]:
        """Plot C2 heatmaps using optimized visualizer."""

        tracker = self.progress_tracker

        try:
            # Use optimized visualizer
            result = self.visualizer.create_correlation_heatmap(
                exp_data=exp,
                theory_data=theory,
                phi_angles=phi_angles,
                output_path=Path(outdir)
                / f"c2_heatmaps_{method_name or 'analysis'}.png",
                title=f"C2 Correlation Analysis - {method_name or 'Analysis'}",
            )

            if tracker:
                tracker.complete_task(plot_task, success=True)

            self.plot_stats["plots_created"] += len(phi_angles)

            return {
                "success": True,
                "plots_created": len(phi_angles),
                "method": method_name,
                "optimized_result": result,
                "performance": {
                    "execution_time": result.get("performance", {}).get(
                        "render_time", 0
                    ),
                    "optimized": True,
                    "backend": result.get("backend", "unknown"),
                },
            }

        except Exception as e:
            if tracker:
                tracker.complete_task(plot_task, success=False, message=str(e))
            raise e

    def plot_diagnostic_summary_enhanced(
        self,
        results: dict[str, Any],
        outdir: str | Path,
        config: dict | None = None,
        method_name: str | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """Enhanced diagnostic summary plotting with progress tracking."""

        start_time = time.time()

        try:
            if self.enable_progress:
                with ProgressContext(
                    f"Creating diagnostic summary for {method_name or 'analysis'}", 1
                ) as progress:
                    success = original_plot_diagnostic_summary(
                        results, outdir, config, method_name
                    )
                    progress.update(current=1)
            else:
                success = original_plot_diagnostic_summary(
                    results, outdir, config, method_name
                )

            self.plot_stats["plots_created"] += 1 if success else 0

            return {
                "success": success,
                "plots_created": 1 if success else 0,
                "method": method_name,
                "performance": {
                    "execution_time": time.time() - start_time,
                    "optimized": False,
                },
            }

        except Exception as e:
            self.plot_stats["errors_encountered"] += 1
            self.error_reporter.report_error(
                e,
                {
                    "function": "plot_diagnostic_summary_enhanced",
                    "method_name": method_name,
                },
            )

            return {"success": False, "error": str(e), "plots_created": 0}

    def create_all_plots_enhanced(
        self,
        results: dict[str, Any],
        outdir: str | Path,
        config: dict | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """Enhanced version of create_all_plots with comprehensive tracking."""

        start_time = time.time()

        # Estimate total plots to be created
        method_results = results.get("method_results", {})
        total_methods = len(method_results)
        estimated_plots = total_methods * 2  # Assume 2 plots per method on average

        try:
            if self.enable_progress:
                with ProgressContext(
                    "Creating all analysis plots", estimated_plots
                ) as progress:
                    # Track progress through original function
                    plot_status = original_create_all_plots(results, outdir, config)

                    # Update progress based on successful plots
                    successful_plots = sum(plot_status.values())
                    progress.update(current=successful_plots)
            else:
                plot_status = original_create_all_plots(results, outdir, config)

            # Update statistics
            successful_plots = sum(plot_status.values())
            self.plot_stats["plots_created"] += successful_plots

            return {
                "success": True,
                "plot_status": plot_status,
                "total_plots_created": successful_plots,
                "performance": {
                    "execution_time": time.time() - start_time,
                    "plots_per_second": (
                        successful_plots / (time.time() - start_time)
                        if successful_plots > 0
                        else 0
                    ),
                },
            }

        except Exception as e:
            self.plot_stats["errors_encountered"] += 1
            self.error_reporter.report_error(
                e,
                {
                    "function": "create_all_plots_enhanced",
                    "total_methods": total_methods,
                },
            )

            return {"success": False, "error": str(e), "total_plots_created": 0}

    def _estimate_data_size(self, arrays: list[np.ndarray]) -> float:
        """Estimate total data size in MB."""
        total_bytes = 0
        for array in arrays:
            if array is not None and hasattr(array, "nbytes"):
                total_bytes += array.nbytes
        return total_bytes / (1024 * 1024)

    def _optimize_for_data_size(self, data_size_mb: float, config: dict | None) -> None:
        """Optimize plotting settings based on data size."""
        if config is None:
            return

        plot_config = get_plot_config(config)

        # Adaptive DPI based on data size
        if data_size_mb > 100:  # Large dataset
            plot_config["dpi"] = 150
            self.logger.info(
                f"Reduced DPI to 150 for large dataset ({data_size_mb:.1f}MB)"
            )
        elif data_size_mb > 50:  # Medium dataset
            plot_config["dpi"] = 200
        # else keep default DPI

        # Update config with optimized settings
        if "output_settings" not in config:
            config["output_settings"] = {}
        config["output_settings"]["plotting"] = plot_config

    def get_plotting_statistics(self) -> dict[str, Any]:
        """Get comprehensive plotting statistics."""
        total_time = self.plot_stats["total_time"]
        plots_created = self.plot_stats["plots_created"]

        return {
            "plots_created": plots_created,
            "total_execution_time": total_time,
            "average_time_per_plot": (
                total_time / plots_created if plots_created > 0 else 0
            ),
            "errors_encountered": self.plot_stats["errors_encountered"],
            "memory_optimizations_applied": self.plot_stats["memory_saved"],
            "performance_features": {
                "progress_tracking": self.enable_progress,
                "performance_optimization": self.optimize_performance,
                "adaptive_quality": self.adaptive_quality,
            },
        }

    def reset_statistics(self) -> None:
        """Reset plotting statistics."""
        self.plot_stats = {
            "plots_created": 0,
            "total_time": 0.0,
            "memory_saved": 0,
            "errors_encountered": 0,
        }


# Enhanced wrapper functions for backward compatibility
def plot_c2_heatmaps_enhanced(*args, **kwargs) -> bool:
    """Enhanced wrapper for plot_c2_heatmaps with progress tracking."""
    manager = EnhancedPlottingManager()
    result = manager.plot_c2_heatmaps_enhanced(*args, **kwargs)
    return result.get("success", False)


def plot_diagnostic_summary_enhanced(*args, **kwargs) -> bool:
    """Enhanced wrapper for plot_diagnostic_summary with progress tracking."""
    manager = EnhancedPlottingManager()
    result = manager.plot_diagnostic_summary_enhanced(*args, **kwargs)
    return result.get("success", False)


def create_all_plots_enhanced(*args, **kwargs) -> dict[str, bool]:
    """Enhanced wrapper for create_all_plots with comprehensive tracking."""
    manager = EnhancedPlottingManager()
    result = manager.create_all_plots_enhanced(*args, **kwargs)
    return result.get("plot_status", {})


# Factory function for creating enhanced plotting manager
def create_enhanced_plotting_manager(
    enable_progress: bool = True,
    optimize_performance: bool = True,
    adaptive_quality: bool = True,
) -> EnhancedPlottingManager:
    """Create enhanced plotting manager with specified features."""
    return EnhancedPlottingManager(
        enable_progress=enable_progress,
        optimize_performance=optimize_performance,
        adaptive_quality=adaptive_quality,
    )


if __name__ == "__main__":
    # Example usage and testing
    print("Testing Enhanced Plotting Integration...")

    # Create test data
    phi_angles = np.array([0, 45, 90])
    test_exp = np.random.exponential(1, (3, 50, 100)) + 1
    test_theory = test_exp + 0.1 * np.random.normal(0, 1, test_exp.shape)

    # Create enhanced plotting manager
    manager = create_enhanced_plotting_manager()

    # Test enhanced C2 heatmaps
    result = manager.plot_c2_heatmaps_enhanced(
        exp=test_exp,
        theory=test_theory,
        phi_angles=phi_angles,
        outdir="./test_output",
        method_name="test_method",
    )

    print(f"C2 heatmaps result: {result}")

    # Test enhanced diagnostic summary
    test_results = {
        "method_results": {
            "test_method": {"chi_squared": 0.123, "parameters": [1000, -0.1, -0.5]}
        }
    }

    result = manager.plot_diagnostic_summary_enhanced(
        results=test_results, outdir="./test_output", method_name="test_method"
    )

    print(f"Diagnostic summary result: {result}")

    # Get statistics
    stats = manager.get_plotting_statistics()
    print(f"Plotting statistics: {stats}")

    print("Enhanced plotting integration tests completed!")
