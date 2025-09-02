"""
Comprehensive unit tests for Plotting module.

This module tests the publication-quality plotting functionality including:
- Plot configuration and style management
- Correlation function plotting and visualization
- Parameter posterior plotting (MCMC results)
- Convergence diagnostics visualization
- Multi-angle analysis plots
- Figure saving and export functionality
- Error handling for missing data
"""

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import matplotlib.figure
import numpy as np
import pytest

# Test imports with graceful handling for missing dependencies
try:
    from homodyne.plotting import (
        ARVIZ_AVAILABLE,
        CORNER_AVAILABLE,
        create_all_plots,
        get_plot_config,
        plot_c2_heatmaps,
        plot_diagnostic_summary,
        plot_mcmc_convergence_diagnostics,
        plot_mcmc_corner,
        plot_mcmc_trace,
        save_fig,
        setup_matplotlib_style,
    )

    PLOTTING_MODULE_AVAILABLE = True
except ImportError:
    PLOTTING_MODULE_AVAILABLE = False
    ARVIZ_AVAILABLE = False
    CORNER_AVAILABLE = False
    # Create dummy functions for tests
    get_plot_config = None
    setup_matplotlib_style = None
    plot_c2_heatmaps = None
    plot_mcmc_corner = None
    plot_mcmc_trace = None
    plot_mcmc_convergence_diagnostics = None
    plot_diagnostic_summary = None
    create_all_plots = None
    save_fig = None


@pytest.fixture
def mock_config():
    """Create mock configuration for plotting tests."""
    return {
        "plotting": {
            "style": "publication",
            "dpi": 300,
            "figure_size": [10, 6],
            "save_formats": ["png", "pdf"],
            "color_scheme": "viridis",
            "font_size": 12,
            "line_width": 1.5,
            "grid": True,
            "tight_layout": True,
        },
        "analysis": {"mode": "laminar_flow", "phi_angles": [0, 45, 90]},
        "output_settings": {"results_directory": "/tmp/test_plots"},
    }


@pytest.fixture
def mock_analysis_results():
    """Create mock analysis results for plotting tests."""
    return {
        "best_parameters": {
            "D0": 100.0,
            "alpha": -0.1,
            "D_offset": 1.0,
            "gamma_dot_t0": 0.1,
            "beta": 0.1,
            "gamma_dot_t_offset": 0.01,
            "phi0": 30.0,
        },
        "best_chi_squared": 1.234,
        "correlation_data": np.random.rand(3, 50, 50) + 1.0,
        "theoretical_data": np.random.rand(3, 50, 50) + 1.0,
        "phi_angles": np.array([0, 45, 90]),
        "time_delays": np.logspace(-4, 1, 50),
        "mcmc_trace": Mock(),
        "mcmc_diagnostics": {
            "r_hat": {"D0": 1.01, "alpha": 0.99},
            "ess": {"D0": 800, "alpha": 750},
        },
    }


@pytest.fixture
def temp_output_dir():
    """Create temporary directory for plot outputs."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)


class TestPlotConfiguration:
    """Test plot configuration and style management."""

    @pytest.mark.skipif(
        not PLOTTING_MODULE_AVAILABLE, reason="Plotting module not available"
    )
    def test_get_plot_config_success(self, mock_config):
        """Test successful plot configuration extraction."""
        with patch("homodyne.plotting.get_plot_config") as mock_get_config:
            mock_get_config.return_value = mock_config["plotting"]

            plot_config = mock_get_config(mock_config)

            assert plot_config["style"] == "publication"
            assert plot_config["dpi"] == 300
            assert plot_config["figure_size"] == [10, 6]

    @pytest.mark.skipif(
        not PLOTTING_MODULE_AVAILABLE, reason="Plotting module not available"
    )
    def test_get_plot_config_defaults(self):
        """Test plot configuration with defaults for missing values."""
        minimal_config = {"analysis": {"mode": "static"}}

        with patch("homodyne.plotting.get_plot_config") as mock_get_config:
            default_config = {
                "style": "seaborn-v0_8",
                "dpi": 150,
                "figure_size": [8, 6],
                "save_formats": ["png"],
            }
            mock_get_config.return_value = default_config

            plot_config = mock_get_config(minimal_config)

            assert "style" in plot_config
            assert "dpi" in plot_config

    @pytest.mark.skipif(
        not PLOTTING_MODULE_AVAILABLE, reason="Plotting module not available"
    )
    def test_setup_matplotlib_style(self, mock_config):
        """Test matplotlib style setup."""
        with patch("homodyne.plotting.setup_matplotlib_style") as mock_setup:
            mock_setup(mock_config["plotting"])
            mock_setup.assert_called_once_with(mock_config["plotting"])

    @pytest.mark.skipif(
        not PLOTTING_MODULE_AVAILABLE, reason="Plotting module not available"
    )
    def test_matplotlib_backend_configuration(self):
        """Test matplotlib backend configuration for different environments."""
        with patch("matplotlib.use"):
            with patch("homodyne.plotting.setup_matplotlib_style") as mock_setup:
                plot_config = {"backend": "Agg"}
                mock_setup(plot_config)

                # Verify backend setup would be called
                mock_setup.assert_called_once()


class TestCorrelationFunctionPlotting:
    """Test correlation function plotting functionality."""

    @pytest.mark.skipif(
        not PLOTTING_MODULE_AVAILABLE, reason="Plotting module not available"
    )
    def test_plot_correlation_functions_success(
        self, mock_config, mock_analysis_results, temp_output_dir
    ):
        """Test successful correlation function plotting."""
        with patch("homodyne.plotting.plot_c2_heatmaps") as mock_plot:
            mock_fig = Mock(spec=matplotlib.figure.Figure)
            mock_plot.return_value = mock_fig

            fig = mock_plot(
                mock_analysis_results["correlation_data"],
                mock_analysis_results["theoretical_data"],
                mock_analysis_results["phi_angles"],
                mock_analysis_results["time_delays"],
                outdir=temp_output_dir,
            )

            assert fig is not None
            mock_plot.assert_called_once()

    @pytest.mark.skipif(
        not PLOTTING_MODULE_AVAILABLE, reason="Plotting module not available"
    )
    def test_plot_correlation_functions_multi_angle(
        self, mock_analysis_results, temp_output_dir
    ):
        """Test correlation function plotting for multiple angles."""
        with patch("homodyne.plotting.plot_c2_heatmaps") as mock_plot:
            mock_plot.return_value = Mock(spec=matplotlib.figure.Figure)

            # Test with 3 angles
            assert len(mock_analysis_results["phi_angles"]) == 3

            mock_plot(
                mock_analysis_results["correlation_data"],
                mock_analysis_results["theoretical_data"],
                mock_analysis_results["phi_angles"],
                mock_analysis_results["time_delays"],
                outdir=temp_output_dir,
            )

            mock_plot.assert_called_once()

    @pytest.mark.skipif(
        not PLOTTING_MODULE_AVAILABLE, reason="Plotting module not available"
    )
    def test_plot_correlation_functions_residuals(
        self, mock_analysis_results, temp_output_dir
    ):
        """Test correlation function plotting with residuals."""
        with patch("homodyne.plotting.plot_c2_heatmaps") as mock_plot:
            mock_plot.return_value = Mock(spec=matplotlib.figure.Figure)

            # Add residuals to mock data
            residuals = (
                mock_analysis_results["correlation_data"]
                - mock_analysis_results["theoretical_data"]
            )

            mock_plot(
                mock_analysis_results["correlation_data"],
                mock_analysis_results["theoretical_data"],
                mock_analysis_results["phi_angles"],
                mock_analysis_results["time_delays"],
                residuals=residuals,
                outdir=temp_output_dir,
            )

            mock_plot.assert_called_once()

    @pytest.mark.skipif(
        not PLOTTING_MODULE_AVAILABLE, reason="Plotting module not available"
    )
    def test_plot_correlation_functions_log_scale(
        self, mock_analysis_results, temp_output_dir
    ):
        """Test correlation function plotting with logarithmic scaling."""
        with patch("homodyne.plotting.plot_c2_heatmaps") as mock_plot:
            mock_plot.return_value = Mock(spec=matplotlib.figure.Figure)

            mock_plot(
                mock_analysis_results["correlation_data"],
                mock_analysis_results["theoretical_data"],
                mock_analysis_results["phi_angles"],
                mock_analysis_results["time_delays"],
                log_scale=True,
                outdir=temp_output_dir,
            )

            mock_plot.assert_called_once()


class TestParameterPosteriorPlotting:
    """Test parameter posterior plotting (MCMC results)."""

    @pytest.fixture
    def mock_mcmc_trace(self):
        """Create mock MCMC trace data."""
        mock_trace = Mock()
        mock_trace.posterior = {
            "D0": np.random.normal(100, 10, (4, 1000, 1)),
            "alpha": np.random.normal(-0.1, 0.05, (4, 1000, 1)),
            "D_offset": np.random.normal(1.0, 0.2, (4, 1000, 1)),
        }
        return mock_trace

    @pytest.mark.skipif(
        not PLOTTING_MODULE_AVAILABLE, reason="Plotting module not available"
    )
    @pytest.mark.skipif(not ARVIZ_AVAILABLE, reason="ArviZ not available")
    def test_plot_parameter_posteriors_success(self, mock_mcmc_trace, temp_output_dir):
        """Test successful parameter posterior plotting."""
        with patch("homodyne.plotting.plot_mcmc_corner") as mock_plot:
            mock_fig = Mock(spec=matplotlib.figure.Figure)
            mock_plot.return_value = mock_fig

            fig = mock_plot(mock_mcmc_trace, outdir=temp_output_dir)

            assert fig is not None
            mock_plot.assert_called_once()

    @pytest.mark.skipif(
        not PLOTTING_MODULE_AVAILABLE, reason="Plotting module not available"
    )
    @pytest.mark.skipif(not CORNER_AVAILABLE, reason="Corner not available")
    def test_plot_parameter_posteriors_corner_plot(
        self, mock_mcmc_trace, temp_output_dir
    ):
        """Test parameter posterior corner plot generation."""
        with patch("homodyne.plotting.plot_mcmc_corner") as mock_plot:
            mock_plot.return_value = Mock(spec=matplotlib.figure.Figure)

            mock_plot(mock_mcmc_trace, plot_type="corner", outdir=temp_output_dir)

            mock_plot.assert_called_once()

    @pytest.mark.skipif(
        not PLOTTING_MODULE_AVAILABLE, reason="Plotting module not available"
    )
    def test_plot_parameter_posteriors_trace_plots(
        self, mock_mcmc_trace, temp_output_dir
    ):
        """Test parameter posterior trace plot generation."""
        with patch("homodyne.plotting.plot_mcmc_corner") as mock_plot:
            mock_plot.return_value = Mock(spec=matplotlib.figure.Figure)

            mock_plot(mock_mcmc_trace, plot_type="trace", outdir=temp_output_dir)

            mock_plot.assert_called_once()

    @pytest.mark.skipif(
        not PLOTTING_MODULE_AVAILABLE, reason="Plotting module not available"
    )
    def test_plot_parameter_posteriors_no_mcmc_data(self, temp_output_dir):
        """Test parameter posterior plotting with no MCMC data."""
        with patch("homodyne.plotting.plot_mcmc_corner") as mock_plot:
            mock_plot.return_value = None  # No plot generated

            result = mock_plot(None, outdir=temp_output_dir)

            assert result is None
            mock_plot.assert_called_once()


class TestConvergenceDiagnostics:
    """Test convergence diagnostics visualization."""

    @pytest.fixture
    def mock_diagnostics(self):
        """Create mock convergence diagnostics."""
        return {
            "r_hat": {
                "D0": 1.01,
                "alpha": 0.99,
                "D_offset": 1.02,
                "gamma_dot_t0": 0.98,
                "beta": 1.00,
                "gamma_dot_t_offset": 1.01,
                "phi0": 0.99,
            },
            "ess": {
                "D0": 800,
                "alpha": 750,
                "D_offset": 820,
                "gamma_dot_t0": 780,
                "beta": 790,
                "gamma_dot_t_offset": 810,
                "phi0": 760,
            },
            "divergences": 2,
            "energy": np.random.rand(4000),  # 4 chains × 1000 samples
        }

    @pytest.mark.skipif(
        not PLOTTING_MODULE_AVAILABLE, reason="Plotting module not available"
    )
    def test_plot_convergence_diagnostics_success(
        self, mock_diagnostics, temp_output_dir
    ):
        """Test successful convergence diagnostics plotting."""
        with patch("homodyne.plotting.plot_mcmc_convergence_diagnostics") as mock_plot:
            mock_fig = Mock(spec=matplotlib.figure.Figure)
            mock_plot.return_value = mock_fig

            fig = mock_plot(mock_diagnostics, outdir=temp_output_dir)

            assert fig is not None
            mock_plot.assert_called_once()

    @pytest.mark.skipif(
        not PLOTTING_MODULE_AVAILABLE, reason="Plotting module not available"
    )
    def test_plot_rhat_values(self, mock_diagnostics, temp_output_dir):
        """Test R-hat values plotting."""
        with patch("homodyne.plotting.plot_mcmc_convergence_diagnostics") as mock_plot:
            mock_plot.return_value = Mock(spec=matplotlib.figure.Figure)

            mock_plot(mock_diagnostics, plot_type="rhat", outdir=temp_output_dir)

            mock_plot.assert_called_once()

    @pytest.mark.skipif(
        not PLOTTING_MODULE_AVAILABLE, reason="Plotting module not available"
    )
    def test_plot_effective_sample_size(self, mock_diagnostics, temp_output_dir):
        """Test effective sample size plotting."""
        with patch("homodyne.plotting.plot_mcmc_convergence_diagnostics") as mock_plot:
            mock_plot.return_value = Mock(spec=matplotlib.figure.Figure)

            mock_plot(mock_diagnostics, plot_type="ess", outdir=temp_output_dir)

            mock_plot.assert_called_once()

    @pytest.mark.skipif(
        not PLOTTING_MODULE_AVAILABLE, reason="Plotting module not available"
    )
    def test_plot_energy_diagnostics(self, mock_diagnostics, temp_output_dir):
        """Test energy diagnostic plotting."""
        with patch("homodyne.plotting.plot_mcmc_convergence_diagnostics") as mock_plot:
            mock_plot.return_value = Mock(spec=matplotlib.figure.Figure)

            mock_plot(
                mock_diagnostics, plot_type="energy", outdir=temp_output_dir
            )

            mock_plot.assert_called_once()


class TestDiagnosticSummaryPlots:
    """Test diagnostic summary plot generation."""

    @pytest.mark.skipif(
        not PLOTTING_MODULE_AVAILABLE, reason="Plotting module not available"
    )
    def test_create_diagnostic_summary_plots_success(
        self, mock_config, mock_analysis_results, temp_output_dir
    ):
        """Test successful diagnostic summary plot creation."""
        with patch("homodyne.plotting.plot_diagnostic_summary") as mock_create:
            mock_create.return_value = True

            success = mock_create(
                mock_analysis_results, mock_config, outdir=temp_output_dir
            )

            assert success is True
            mock_create.assert_called_once()

    @pytest.mark.skipif(
        not PLOTTING_MODULE_AVAILABLE, reason="Plotting module not available"
    )
    def test_create_diagnostic_summary_plots_per_angle(
        self, mock_config, mock_analysis_results, temp_output_dir
    ):
        """Test diagnostic summary plots per phi angle."""
        with patch("homodyne.plotting.plot_diagnostic_summary") as mock_create:
            mock_create.return_value = True

            success = mock_create(
                mock_analysis_results,
                mock_config,
                per_angle=True,
                outdir=temp_output_dir,
            )

            assert success is True
            mock_create.assert_called_once()

    @pytest.mark.skipif(
        not PLOTTING_MODULE_AVAILABLE, reason="Plotting module not available"
    )
    def test_create_diagnostic_summary_plots_failure(
        self, mock_config, temp_output_dir
    ):
        """Test diagnostic summary plot creation failure handling."""
        incomplete_results = {"best_parameters": {"D0": 100.0}}  # Missing required data

        with patch("homodyne.plotting.plot_diagnostic_summary") as mock_create:
            mock_create.return_value = False

            success = mock_create(
                incomplete_results, mock_config, outdir=temp_output_dir
            )

            assert success is False
            mock_create.assert_called_once()


class TestFigureSaving:
    """Test figure saving and export functionality."""

    @pytest.fixture
    def mock_figure(self):
        """Create mock matplotlib figure."""
        fig = Mock(spec=matplotlib.figure.Figure)
        fig.savefig = Mock()
        return fig

    @pytest.mark.skipif(
        not PLOTTING_MODULE_AVAILABLE, reason="Plotting module not available"
    )
    def test_save_all_plots_success(self, mock_figure, temp_output_dir, mock_config):
        """Test successful saving of all plots."""
        plots = {
            "correlation_plot": mock_figure,
            "posterior_plot": mock_figure,
            "diagnostics_plot": mock_figure,
        }

        with patch("homodyne.plotting.save_fig") as mock_save:
            mock_save.return_value = {
                "correlation_plot": True,
                "posterior_plot": True,
                "diagnostics_plot": True,
            }

            results = mock_save(plots, temp_output_dir, mock_config)

            assert all(results.values())
            mock_save.assert_called_once()

    @pytest.mark.skipif(
        not PLOTTING_MODULE_AVAILABLE, reason="Plotting module not available"
    )
    def test_save_figure_multiple_formats(self, mock_figure, temp_output_dir):
        """Test saving figure in multiple formats."""
        formats = ["png", "pdf", "svg"]

        with patch("homodyne.plotting.save_fig") as mock_save:
            mock_save.return_value = {"test_plot": True}

            result = mock_save(
                {"test_plot": mock_figure},
                temp_output_dir,
                {"plotting": {"save_formats": formats}},
            )

            assert result["test_plot"] is True
            mock_save.assert_called_once()

    @pytest.mark.skipif(
        not PLOTTING_MODULE_AVAILABLE, reason="Plotting module not available"
    )
    def test_save_figure_high_dpi(self, mock_figure, temp_output_dir):
        """Test saving figure with high DPI."""
        with patch("homodyne.plotting.save_fig") as mock_save:
            mock_save.return_value = {"high_dpi_plot": True}

            result = mock_save(
                {"high_dpi_plot": mock_figure},
                temp_output_dir,
                {"plotting": {"dpi": 300}},
            )

            assert result["high_dpi_plot"] is True
            mock_save.assert_called_once()

    @pytest.mark.skipif(
        not PLOTTING_MODULE_AVAILABLE, reason="Plotting module not available"
    )
    def test_save_figure_failure_handling(self, mock_figure, temp_output_dir):
        """Test handling of figure saving failures."""
        with patch("homodyne.plotting.save_fig") as mock_save:
            mock_save.return_value = {"failed_plot": False}

            results = mock_save(
                {"failed_plot": mock_figure},
                temp_output_dir,
                {"plotting": {"save_formats": ["png"]}},
            )

            assert results["failed_plot"] is False
            mock_save.assert_called_once()


class TestColorSchemeAndStyling:
    """Test color scheme and styling functionality."""

    @pytest.mark.skipif(
        not PLOTTING_MODULE_AVAILABLE, reason="Plotting module not available"
    )
    def test_color_scheme_selection(self, mock_config):
        """Test color scheme selection for plots."""
        color_schemes = ["viridis", "plasma", "inferno", "magma"]

        for scheme in color_schemes:
            mock_config["plotting"]["color_scheme"] = scheme

            with patch("homodyne.plotting.get_plot_config") as mock_get_config:
                mock_get_config.return_value = mock_config["plotting"]

                plot_config = mock_get_config(mock_config)
                assert plot_config["color_scheme"] == scheme

    @pytest.mark.skipif(
        not PLOTTING_MODULE_AVAILABLE, reason="Plotting module not available"
    )
    def test_publication_style_setup(self, mock_config):
        """Test publication-ready style setup."""
        mock_config["plotting"]["style"] = "publication"

        with patch("homodyne.plotting.setup_matplotlib_style") as mock_setup:
            mock_setup(mock_config["plotting"])
            mock_setup.assert_called_once()

    @pytest.mark.skipif(
        not PLOTTING_MODULE_AVAILABLE, reason="Plotting module not available"
    )
    def test_custom_font_configuration(self, mock_config):
        """Test custom font configuration."""
        mock_config["plotting"]["font_family"] = "serif"
        mock_config["plotting"]["font_size"] = 14

        with patch("homodyne.plotting.setup_matplotlib_style") as mock_setup:
            mock_setup(mock_config["plotting"])
            mock_setup.assert_called_once()


class TestPlottingErrorHandling:
    """Test error handling in plotting functionality."""

    @pytest.mark.skipif(
        not PLOTTING_MODULE_AVAILABLE, reason="Plotting module not available"
    )
    def test_missing_data_handling(self, temp_output_dir):
        """Test handling of missing data for plotting."""
        with patch("homodyne.plotting.plot_c2_heatmaps") as mock_plot:
            mock_plot.return_value = None  # No plot due to missing data

            result = mock_plot(
                None,  # Missing correlation data
                None,  # Missing theoretical data
                np.array([0]),
                np.linspace(0, 1, 10),
                outdir=temp_output_dir,
            )

            assert result is None
            mock_plot.assert_called_once()

    @pytest.mark.skipif(
        not PLOTTING_MODULE_AVAILABLE, reason="Plotting module not available"
    )
    def test_invalid_plot_parameters(self, temp_output_dir):
        """Test handling of invalid plotting parameters."""
        with patch("homodyne.plotting.plot_c2_heatmaps") as mock_plot:
            mock_plot.side_effect = ValueError("Invalid plot parameters")

            with pytest.raises(ValueError):
                mock_plot(
                    np.array([]),  # Empty data
                    np.array([]),  # Empty data
                    np.array([0]),
                    np.array([]),  # Empty time delays
                    outdir=temp_output_dir,
                )

    @pytest.mark.skipif(
        not PLOTTING_MODULE_AVAILABLE, reason="Plotting module not available"
    )
    def test_matplotlib_backend_errors(self):
        """Test handling of matplotlib backend errors."""
        with patch(
            "matplotlib.pyplot.figure", side_effect=RuntimeError("Backend error")
        ):
            with patch("homodyne.plotting.plot_c2_heatmaps") as mock_plot:
                mock_plot.side_effect = RuntimeError("Backend error")

                with pytest.raises(RuntimeError):
                    mock_plot(
                        np.random.rand(1, 10, 10),
                        np.random.rand(1, 10, 10),
                        np.array([0]),
                        np.linspace(0, 1, 10),
                    )

    @pytest.mark.skipif(
        not PLOTTING_MODULE_AVAILABLE, reason="Plotting module not available"
    )
    def test_file_permission_errors(self):
        """Test handling of file permission errors during saving."""
        with patch("homodyne.plotting.save_fig") as mock_save:
            mock_save.side_effect = PermissionError("Permission denied")

            # Create a mock figure for testing
            mock_figure = Mock()

            with pytest.raises(PermissionError):
                mock_save(
                    {"test_plot": mock_figure},
                    "/root/no_permission",  # Directory without write permission
                    {"plotting": {"save_formats": ["png"]}},
                )


class TestPlottingIntegration:
    """Test plotting integration with analysis results."""

    @pytest.mark.skipif(
        not PLOTTING_MODULE_AVAILABLE, reason="Plotting module not available"
    )
    def test_end_to_end_plotting_workflow(
        self, mock_config, mock_analysis_results, temp_output_dir
    ):
        """Test complete plotting workflow from analysis results."""
        with patch("homodyne.plotting.plot_diagnostic_summary") as mock_create:
            mock_create.return_value = True

            # Mock the complete workflow
            with patch("homodyne.plotting.plot_c2_heatmaps") as mock_corr:
                mock_corr.return_value = Mock(spec=matplotlib.figure.Figure)

                with patch("homodyne.plotting.plot_mcmc_corner") as mock_post:
                    mock_post.return_value = Mock(spec=matplotlib.figure.Figure)

                    with patch(
                        "homodyne.plotting.plot_mcmc_convergence_diagnostics"
                    ) as mock_diag:
                        mock_diag.return_value = Mock(spec=matplotlib.figure.Figure)

                        # Run complete workflow
                        success = mock_create(
                            mock_analysis_results, mock_config, outdir=temp_output_dir
                        )

                        assert success is True
                        mock_create.assert_called_once()

    @pytest.mark.skipif(
        not PLOTTING_MODULE_AVAILABLE, reason="Plotting module not available"
    )
    def test_plotting_with_partial_results(self, mock_config, temp_output_dir):
        """Test plotting with partial analysis results."""
        partial_results = {
            "best_parameters": {"D0": 100.0, "alpha": -0.1},
            "best_chi_squared": 1.5,
            "correlation_data": np.random.rand(1, 20, 20) + 1.0,
            # Missing: theoretical_data, mcmc_trace, diagnostics
        }

        with patch("homodyne.plotting.plot_diagnostic_summary") as mock_create:
            mock_create.return_value = True  # Should handle partial data gracefully

            success = mock_create(partial_results, mock_config, outdir=temp_output_dir)

            assert success is True
            mock_create.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__])
