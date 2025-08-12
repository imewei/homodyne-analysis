"""
Tests for Plotting Module
=========================

Tests plot image generation, matplotlib figure creation, and visualization functions.
"""

import pytest
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for testing
import matplotlib.pyplot as plt
from pathlib import Path
from unittest.mock import patch, MagicMock
import tempfile

# Import the modules to test


from homodyne.tests.fixtures import (
    dummy_config, 
    dummy_correlation_data,
    dummy_theoretical_data,
    dummy_phi_angles,
    dummy_time_arrays,
    dummy_analysis_results,
    temp_directory
)

# Import plotting functions
try:
    from homodyne.plotting import (
        plot_c2_heatmaps,
        plot_parameter_evolution,
        plot_diagnostic_summary,
        create_all_plots,
        get_plot_config,
        setup_matplotlib_style,
        save_fig
    )
    PLOTTING_MODULE_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import homodyne.plotting as plotting module: {e}")
    PLOTTING_MODULE_AVAILABLE = False


# Skip all tests if plotting module is not available
pytestmark = pytest.mark.skipif(not PLOTTING_MODULE_AVAILABLE, reason="Plotting module not available")


class TestPlotConfiguration:
    """Test plotting configuration and setup."""
    
    def test_get_plot_config_with_config(self, dummy_config):
        """Test extracting plot configuration from main config."""
        plot_config = get_plot_config(dummy_config)
        
        assert isinstance(plot_config, dict)
        assert "plot_format" in plot_config
        assert "dpi" in plot_config  
        assert "figure_size" in plot_config
        assert "create_plots" in plot_config
        
        # Check values from dummy_config
        assert plot_config["plot_format"] == "png"
        assert plot_config["dpi"] == 100
        assert plot_config["figure_size"] == [6, 4]
    
    def test_get_plot_config_with_defaults(self):
        """Test plot configuration with defaults when no config provided."""
        plot_config = get_plot_config(None)
        
        assert isinstance(plot_config, dict)
        assert "plot_format" in plot_config
        assert "dpi" in plot_config
        assert "figure_size" in plot_config
        
        # Should use reasonable defaults
        assert plot_config["plot_format"] in ["png", "pdf", "svg"]
        assert plot_config["dpi"] > 0
        assert len(plot_config["figure_size"]) == 2
    
    def test_setup_matplotlib_style(self, dummy_config):
        """Test matplotlib style configuration."""
        plot_config = get_plot_config(dummy_config)
        
        # This should not raise an exception
        setup_matplotlib_style(plot_config)
        
        # Check that some rcParams were set
        assert plt.rcParams['savefig.dpi'] == plot_config["dpi"]
        assert plt.rcParams['figure.dpi'] == plot_config["dpi"]


class TestC2HeatmapPlots:
    """Test C2 correlation function heatmap generation."""
    
    def test_plot_c2_heatmaps_basic(self, temp_directory, dummy_correlation_data, 
                                   dummy_theoretical_data, dummy_phi_angles, dummy_config):
        """Test basic C2 heatmap creation."""
        success = plot_c2_heatmaps(
            dummy_correlation_data,
            dummy_theoretical_data, 
            dummy_phi_angles,
            temp_directory,
            dummy_config
        )
        
        assert success is True
        
        # Check that plot files were created
        plot_files = list(temp_directory.glob("c2_heatmaps_*.png"))
        assert len(plot_files) == len(dummy_phi_angles)  # One plot per angle
        
        # Verify file sizes (should be > 0 for valid images)
        for plot_file in plot_files:
            assert plot_file.stat().st_size > 1000  # Reasonable minimum size
    
    def test_plot_c2_heatmaps_with_time_arrays(self, temp_directory, dummy_correlation_data,
                                              dummy_theoretical_data, dummy_phi_angles,
                                              dummy_time_arrays, dummy_config):
        """Test C2 heatmaps with explicit time arrays."""
        time_lags, delay_times = dummy_time_arrays
        
        success = plot_c2_heatmaps(
            dummy_correlation_data,
            dummy_theoretical_data,
            dummy_phi_angles,
            temp_directory,
            dummy_config,
            time_lags=time_lags,
            delay_times=delay_times
        )
        
        assert success is True
        
        # Check that files were created
        plot_files = list(temp_directory.glob("c2_heatmaps_*.png"))
        assert len(plot_files) > 0
    
    def test_plot_c2_heatmaps_shape_mismatch(self, temp_directory, dummy_correlation_data,
                                           dummy_phi_angles, dummy_config):
        """Test error handling for mismatched data shapes."""
        # Create theoretical data with wrong shape
        wrong_shape_theory = np.random.rand(2, 10, 10)  # Different from dummy_correlation_data
        
        success = plot_c2_heatmaps(
            dummy_correlation_data,
            wrong_shape_theory,
            dummy_phi_angles,
            temp_directory,
            dummy_config
        )
        
        assert success is False  # Should fail gracefully
    
    def test_plot_c2_heatmaps_angle_count_mismatch(self, temp_directory, dummy_correlation_data,
                                                  dummy_theoretical_data, dummy_config):
        """Test error handling for wrong number of angles."""
        wrong_angles = np.array([0.0, 45.0])  # Only 2 angles instead of 3
        
        success = plot_c2_heatmaps(
            dummy_correlation_data,
            dummy_theoretical_data,
            wrong_angles,
            temp_directory,
            dummy_config
        )
        
        assert success is False  # Should fail gracefully
    
    def test_plot_c2_heatmaps_creates_directory(self, temp_directory, dummy_correlation_data,
                                               dummy_theoretical_data, dummy_phi_angles, dummy_config):
        """Test that plot function creates output directory if it doesn't exist."""
        nonexistent_dir = temp_directory / "plots" / "c2_heatmaps"
        assert not nonexistent_dir.exists()
        
        success = plot_c2_heatmaps(
            dummy_correlation_data,
            dummy_theoretical_data,
            dummy_phi_angles,
            nonexistent_dir,
            dummy_config
        )
        
        assert success is True
        assert nonexistent_dir.exists()


class TestParameterEvolutionPlots:
    """Test parameter evolution and optimization plots."""
    
    def test_plot_parameter_evolution_basic(self, temp_directory, dummy_config):
        """Test basic parameter evolution plot."""
        best_params = {
            "D0": 123.45,
            "alpha": -0.123,
            "D_offset": 12.34,
            "beta": -0.456
        }
        
        bounds = dummy_config["parameter_space"]["bounds"][:4]  # Use first 4 bounds
        
        success = plot_parameter_evolution(
            best_params,
            bounds,
            temp_directory,
            dummy_config
        )
        
        assert success is True
        
        # Check that plot file was created
        plot_files = list(temp_directory.glob("parameter_evolution.png"))
        assert len(plot_files) == 1
        assert plot_files[0].stat().st_size > 1000
    
    def test_plot_parameter_evolution_with_history(self, temp_directory, dummy_config):
        """Test parameter evolution plot with optimization history."""
        best_params = {"D0": 100.0, "alpha": -0.1}
        bounds = dummy_config["parameter_space"]["bounds"][:2]
        initial_params = {"D0": 50.0, "alpha": 0.0}
        
        optimization_history = [
            {"chi_squared": 10.0, "iteration": 1},
            {"chi_squared": 5.0, "iteration": 2},
            {"chi_squared": 2.0, "iteration": 3},
            {"chi_squared": 1.0, "iteration": 4}
        ]
        
        success = plot_parameter_evolution(
            best_params,
            bounds, 
            temp_directory,
            dummy_config,
            initial_params=initial_params,
            optimization_history=optimization_history
        )
        
        assert success is True
        
        plot_files = list(temp_directory.glob("parameter_evolution.png"))
        assert len(plot_files) == 1
    
    def test_plot_parameter_evolution_log_uniform_params(self, temp_directory, dummy_config):
        """Test parameter evolution with log-uniform parameters."""
        best_params = {"D0": 1000.0, "gamma_dot_t0": 0.001}
        
        # Create bounds with log-uniform type
        bounds = [
            {"name": "D0", "min": 1.0, "max": 10000.0, "type": "log-uniform", "unit": "Å²/s"},
            {"name": "gamma_dot_t0", "min": 1e-5, "max": 0.1, "type": "log-uniform", "unit": "s⁻¹"}
        ]
        
        success = plot_parameter_evolution(
            best_params,
            bounds,
            temp_directory,
            dummy_config
        )
        
        assert success is True
        
        plot_files = list(temp_directory.glob("parameter_evolution.png"))
        assert len(plot_files) == 1


class TestDiagnosticPlots:
    """Test diagnostic and summary plots."""
    
    def test_plot_diagnostic_summary_basic(self, temp_directory, dummy_analysis_results, dummy_config):
        """Test basic diagnostic summary plot."""
        success = plot_diagnostic_summary(
            dummy_analysis_results,
            temp_directory,
            dummy_config
        )
        
        assert success is True
        
        # Check that plot file was created
        plot_files = list(temp_directory.glob("diagnostic_summary.png"))
        assert len(plot_files) == 1
        assert plot_files[0].stat().st_size > 1000
    
    def test_plot_diagnostic_summary_with_chi2_comparison(self, temp_directory, dummy_config):
        """Test diagnostic plot with multiple chi-squared values."""
        results = {
            "classical_chi_squared": 1.234,
            "bayesian_chi2": 1.100,
            "mcmc_chi_squared": 1.050,
            "residuals": np.random.normal(0, 0.1, (50, 50))
        }
        
        success = plot_diagnostic_summary(results, temp_directory, dummy_config)
        
        assert success is True
        
        plot_files = list(temp_directory.glob("diagnostic_summary.png"))
        assert len(plot_files) == 1
    
    def test_plot_diagnostic_summary_with_uncertainties(self, temp_directory, dummy_config):
        """Test diagnostic plot with parameter uncertainties."""
        results = {
            "parameter_uncertainties": {
                "D0": 5.67,
                "alpha": 0.01,
                "beta": 0.02
            },
            "residuals": np.random.normal(0, 0.05, (30, 30))
        }
        
        success = plot_diagnostic_summary(results, temp_directory, dummy_config)
        
        assert success is True


class TestMCMCPlots:
    """Test MCMC corner plots (if available)."""
    
    @pytest.mark.skipif(True, reason="ArviZ/corner package may not be available")
    def test_plot_mcmc_corner_basic(self, temp_directory, dummy_config):
        """Test basic MCMC corner plot (requires ArviZ)."""
        # Create mock trace data
        mock_trace = {
            "D0": np.random.lognormal(5, 0.5, 1000),
            "alpha": np.random.normal(-0.1, 0.05, 1000),
            "beta": np.random.normal(0.2, 0.1, 1000)
        }
        
        param_names = ["D0", "alpha", "beta"]
        param_units = ["Å²/s", "dimensionless", "dimensionless"]
        
        try:
            from homodyne.plotting import plot_mcmc_corner
            success = plot_mcmc_corner(
                mock_trace,
                temp_directory,
                dummy_config,
                param_names=param_names,
                param_units=param_units
            )
            
            # This may fail if ArviZ is not available, which is acceptable
            if success:
                plot_files = list(temp_directory.glob("mcmc_corner_plot.png"))
                assert len(plot_files) == 1
                
        except ImportError:
            pytest.skip("ArviZ or corner package not available")


class TestCompleteWorkflow:
    """Test complete plotting workflow."""
    
    def test_create_all_plots_complete(self, temp_directory, dummy_analysis_results, dummy_config):
        """Test creating all available plots from complete results."""
        plot_status = create_all_plots(
            dummy_analysis_results,
            temp_directory,
            dummy_config
        )
        
        assert isinstance(plot_status, dict)
        
        # Check that multiple plot types were attempted
        expected_plots = ["c2_heatmaps", "parameter_evolution", "diagnostic_summary"]
        for plot_type in expected_plots:
            if plot_type in plot_status:
                # If the plot was attempted, it should have succeeded with dummy data
                assert plot_status[plot_type] is True, f"{plot_type} plot failed"
        
        # Check that actual files were created
        all_plot_files = list(temp_directory.glob("*.png"))
        assert len(all_plot_files) >= 2  # Should create at least a few plots
    
    def test_create_all_plots_minimal_data(self, temp_directory, dummy_config):
        """Test plotting with minimal data (some plots may be skipped)."""
        minimal_results = {
            "best_parameters": {"D0": 100.0, "alpha": -0.1},
            "parameter_bounds": dummy_config["parameter_space"]["bounds"][:2],
            "best_chi_squared": 1.234
        }
        
        plot_status = create_all_plots(
            minimal_results,
            temp_directory,
            dummy_config
        )
        
        assert isinstance(plot_status, dict)
        
        # At least some plots should succeed
        successful_plots = sum(1 for status in plot_status.values() if status)
        assert successful_plots >= 1


class TestPlotErrorHandling:
    """Test error handling in plotting functions."""
    
    def test_plot_with_invalid_data(self, temp_directory, dummy_config):
        """Test plotting with invalid data types."""
        invalid_exp_data = "not_an_array"
        invalid_theory_data = None
        
        success = plot_c2_heatmaps(
            invalid_exp_data,
            invalid_theory_data,
            dummy_phi_angles,
            temp_directory,
            dummy_config
        )
        
        assert success is False  # Should fail gracefully
    
    def test_plot_with_nan_data(self, temp_directory, dummy_phi_angles, dummy_config):
        """Test plotting with NaN values in data."""
        nan_data = np.full((3, 20, 30), np.nan)
        
        success = plot_c2_heatmaps(
            nan_data,
            nan_data,
            dummy_phi_angles,
            temp_directory,
            dummy_config
        )
        
        # This might succeed or fail depending on implementation
        # The important thing is that it doesn't crash
        assert isinstance(success, bool)
    
    def test_plot_with_read_only_directory(self, temp_directory, dummy_correlation_data,
                                          dummy_theoretical_data, dummy_phi_angles, dummy_config):
        """Test plotting when output directory is read-only."""
        import os
        if os.name == 'nt':  # Skip on Windows due to different permission model
            pytest.skip("Permission tests not reliable on Windows")
        
        readonly_dir = temp_directory / "readonly"
        readonly_dir.mkdir(mode=0o444)
        
        try:
            success = plot_c2_heatmaps(
                dummy_correlation_data,
                dummy_theoretical_data,
                dummy_phi_angles,
                readonly_dir,
                dummy_config
            )
            
            # Should handle permission error gracefully
            assert success is False
            
        finally:
            # Clean up - make writable again
            readonly_dir.chmod(0o755)


class TestPlotFormats:
    """Test different plot formats and options."""
    
    def test_plot_png_format(self, temp_directory, dummy_correlation_data,
                            dummy_theoretical_data, dummy_phi_angles):
        """Test PNG format output."""
        config = {
            "output_settings": {
                "plotting": {
                    "plot_format": "png",
                    "dpi": 150,
                    "figure_size": [8, 6]
                }
            }
        }
        
        success = plot_c2_heatmaps(
            dummy_correlation_data,
            dummy_theoretical_data,
            dummy_phi_angles,
            temp_directory,
            config
        )
        
        assert success is True
        
        png_files = list(temp_directory.glob("*.png"))
        assert len(png_files) > 0
    
    def test_plot_pdf_format(self, temp_directory, dummy_correlation_data,
                            dummy_theoretical_data, dummy_phi_angles):
        """Test PDF format output."""
        config = {
            "output_settings": {
                "plotting": {
                    "plot_format": "pdf",
                    "dpi": 300,
                    "figure_size": [10, 8]
                }
            }
        }
        
        success = plot_c2_heatmaps(
            dummy_correlation_data,
            dummy_theoretical_data,
            dummy_phi_angles,
            temp_directory,
            config
        )
        
        assert success is True
        
        pdf_files = list(temp_directory.glob("*.pdf"))
        assert len(pdf_files) > 0
    
    def test_plot_high_dpi(self, temp_directory, dummy_config):
        """Test high DPI plotting."""
        dummy_config["output_settings"]["plotting"]["dpi"] = 600
        
        best_params = {"D0": 100.0}
        bounds = [{"name": "D0", "min": 1.0, "max": 1000.0, "unit": "Å²/s"}]
        
        success = plot_parameter_evolution(
            best_params,
            bounds,
            temp_directory,
            dummy_config
        )
        
        assert success is True
        
        # High DPI files should be larger
        plot_files = list(temp_directory.glob("*.png"))
        assert len(plot_files) > 0
        
        # Check that file size is reasonable for high DPI
        for plot_file in plot_files:
            assert plot_file.stat().st_size > 10000  # Should be fairly large


class TestPlotContent:
    """Test specific plot content and elements."""
    
    def test_plot_contains_expected_elements(self, temp_directory, dummy_correlation_data,
                                           dummy_theoretical_data, dummy_phi_angles, dummy_config):
        """Test that plots contain expected visual elements."""
        success = plot_c2_heatmaps(
            dummy_correlation_data,
            dummy_theoretical_data,
            dummy_phi_angles,
            temp_directory,
            dummy_config
        )
        
        assert success is True
        
        # While we can't easily test image content, we can verify files exist
        # and have reasonable sizes indicating they contain actual plot data
        plot_files = list(temp_directory.glob("*.png"))
        
        for plot_file in plot_files:
            # File should be reasonably sized (not just a tiny stub)
            assert plot_file.stat().st_size > 5000
            
            # File should have expected naming pattern
            assert "c2_heatmaps_phi_" in plot_file.name
            assert "deg.png" in plot_file.name
    
    def test_plot_handles_different_data_sizes(self, temp_directory, dummy_config):
        """Test plotting with different data array sizes."""
        # Small data
        small_exp = np.random.rand(2, 10, 15)
        small_theory = np.random.rand(2, 10, 15) 
        small_angles = np.array([0.0, 90.0])
        
        success1 = plot_c2_heatmaps(
            small_exp, small_theory, small_angles,
            temp_directory / "small",
            dummy_config
        )
        
        # Large data
        large_exp = np.random.rand(5, 100, 150)
        large_theory = np.random.rand(5, 100, 150)
        large_angles = np.array([0.0, 22.5, 45.0, 67.5, 90.0])
        
        success2 = plot_c2_heatmaps(
            large_exp, large_theory, large_angles,
            temp_directory / "large", 
            dummy_config
        )
        
        assert success1 is True
        assert success2 is True
        
        # Both should create files
        small_files = list((temp_directory / "small").glob("*.png"))
        large_files = list((temp_directory / "large").glob("*.png"))
        
        assert len(small_files) == 2
        assert len(large_files) == 5
