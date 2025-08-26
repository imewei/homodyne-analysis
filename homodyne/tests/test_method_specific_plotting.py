"""
Test suite for method-specific plotting functionality.

This module tests the enhanced plotting capabilities that generate separate
plots for each optimization method (e.g., Nelder-Mead, Gurobi).
"""

import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest

# Check plotting dependencies
try:
    import matplotlib
    import matplotlib.pyplot as plt

    from homodyne.run_homodyne import _generate_classical_plots

    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    _generate_classical_plots = None

# Add project root to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


pytestmark = pytest.mark.skipif(
    not PLOTTING_AVAILABLE, reason="Plotting dependencies not available"
)


class TestMethodSpecificPlotting:
    """Test method-specific plotting functionality."""

    @pytest.fixture
    def mock_analyzer(self):
        """Create a mock analyzer with minimal required attributes."""
        analyzer = Mock()
        analyzer.config = {
            "output_settings": {
                "reporting": {"generate_plots": True},
                "plotting": {
                    "general": {
                        "create_plots": True,
                        "plot_format": "png",
                        "dpi": 150,
                        "figure_size": [8, 6],
                        "style": "publication",
                        "save_plots": True,
                        "show_plots": False,
                    },
                    "c2_heatmaps": {
                        "enabled": True,
                        "layout": "single_row",
                        "include_experimental": True,
                        "include_theoretical": True,
                        "include_residuals": True,
                        "colormap": "viridis",
                    },
                },
            }
        }
        analyzer.dt = 0.5

        # Mock the calculate_c2_nonequilibrium_laminar_parallel method
        def mock_calculate_c2(params, phi_angles):
            n_angles = len(phi_angles)
            return np.random.rand(n_angles, 10, 10) + 1.0

        analyzer.calculate_c2_nonequilibrium_laminar_parallel = (
            mock_calculate_c2
        )
        return analyzer

    @pytest.fixture
    def mock_result_with_methods(self):
        """Create a mock optimization result with method_results."""
        result = Mock()
        result.method_results = {
            "Nelder-Mead": {
                "success": True,
                "parameters": np.array([1000.0, -1.2, 2.5]),
                "chi_squared": 6.544454,
            },
            "Gurobi": {
                "success": True,
                "parameters": np.array([1100.0, -1.3, 2.8]),
                "chi_squared": 2.431911,
            },
        }
        result.best_method = "Gurobi"
        return result

    @pytest.fixture
    def mock_result_without_methods(self):
        """Create a mock optimization result without method_results."""
        result = Mock()
        # Set method_results to None to simulate missing attribute
        result.method_results = None
        return result

    @pytest.fixture
    def test_data(self):
        """Create test data for plotting."""
        n_angles, n_t2, n_t1 = 2, 10, 10
        best_params = np.array([1000.0, -1.2, 2.5])
        phi_angles = np.array([0.0, 45.0])
        c2_exp = np.random.rand(n_angles, n_t2, n_t1) + 1.0
        return best_params, phi_angles, c2_exp

    @patch("homodyne.plotting.plot_c2_heatmaps")
    def test_generate_classical_plots_with_method_results(
        self, mock_plot_c2, mock_analyzer, mock_result_with_methods, test_data
    ):
        """Test that method-specific plots are generated when method_results available."""
        mock_plot_c2.return_value = True
        best_params, phi_angles, c2_exp = test_data

        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)

            # Call the function
            _generate_classical_plots(
                mock_analyzer,
                best_params,
                mock_result_with_methods,
                phi_angles,
                c2_exp,
                output_dir,
            )

            # Should have been called once for each successful method
            assert mock_plot_c2.call_count == 2

            # Check that method names were passed (standardized format)
            calls = mock_plot_c2.call_args_list
            method_names = [call.kwargs.get("method_name") for call in calls]

            assert "nelder_mead" in method_names
            assert "gurobi" in method_names

    @patch("homodyne.plotting.plot_c2_heatmaps")
    def test_generate_classical_plots_without_method_results(
        self,
        mock_plot_c2,
        mock_analyzer,
        mock_result_without_methods,
        test_data,
    ):
        """Test fallback behavior when method_results not available."""
        mock_plot_c2.return_value = True
        best_params, phi_angles, c2_exp = test_data

        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)

            # Call the function
            _generate_classical_plots(
                mock_analyzer,
                best_params,
                mock_result_without_methods,
                phi_angles,
                c2_exp,
                output_dir,
            )

            # Should have been called once with fallback
            assert mock_plot_c2.call_count == 1

            # Check that method name was "best"
            call = mock_plot_c2.call_args_list[0]
            assert call.kwargs.get("method_name") == "best"

    @patch("homodyne.plotting.plot_c2_heatmaps")
    def test_generate_classical_plots_skips_unsuccessful_methods(
        self, mock_plot_c2, mock_analyzer, test_data
    ):
        """Test that unsuccessful methods are skipped."""
        mock_plot_c2.return_value = True
        best_params, phi_angles, c2_exp = test_data

        # Create result with one successful and one unsuccessful method
        result = Mock()
        result.method_results = {
            "Nelder-Mead": {
                "success": True,
                "parameters": np.array([1000.0, -1.2, 2.5]),
                "chi_squared": 6.544454,
            },
            "Gurobi": {
                "success": False,  # This one failed
                "parameters": None,
                "chi_squared": np.inf,
            },
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)

            # Call the function
            _generate_classical_plots(
                mock_analyzer,
                best_params,
                result,
                phi_angles,
                c2_exp,
                output_dir,
            )

            # Should have been called only once (for successful method)
            assert mock_plot_c2.call_count == 1

            # Check that only Nelder-Mead was processed (standardized format)
            call = mock_plot_c2.call_args_list[0]
            assert call.kwargs.get("method_name") == "nelder_mead"

    @patch("homodyne.plotting.plot_c2_heatmaps")
    def test_generate_classical_plots_handles_missing_parameters(
        self, mock_plot_c2, mock_analyzer, test_data
    ):
        """Test that methods with missing parameters are skipped."""
        mock_plot_c2.return_value = True
        best_params, phi_angles, c2_exp = test_data

        # Create result with method that has no parameters
        result = Mock()
        result.method_results = {
            "Nelder-Mead": {
                "success": True,
                "parameters": np.array([1000.0, -1.2, 2.5]),
                "chi_squared": 6.544454,
            },
            "Gurobi": {
                "success": True,
                "parameters": None,  # Missing parameters
                "chi_squared": 2.431911,
            },
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)

            # Call the function
            _generate_classical_plots(
                mock_analyzer,
                best_params,
                result,
                phi_angles,
                c2_exp,
                output_dir,
            )

            # Should have been called only once (for method with parameters)
            assert mock_plot_c2.call_count == 1

            # Check that only Nelder-Mead was processed (standardized format)
            call = mock_plot_c2.call_args_list[0]
            assert call.kwargs.get("method_name") == "nelder_mead"

    def test_generate_classical_plots_creates_classical_directory(
        self, mock_analyzer, mock_result_with_methods, test_data
    ):
        """Test that classical subdirectory is created."""
        best_params, phi_angles, c2_exp = test_data

        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            classical_dir = output_dir / "classical"

            # Ensure directory doesn't exist initially
            assert not classical_dir.exists()

            with patch("homodyne.plotting.plot_c2_heatmaps") as mock_plot:
                mock_plot.return_value = True

                # Call the function
                _generate_classical_plots(
                    mock_analyzer,
                    best_params,
                    mock_result_with_methods,
                    phi_angles,
                    c2_exp,
                    output_dir,
                )

                # Directory should now exist
                assert classical_dir.exists()
                assert classical_dir.is_dir()

    def test_generate_classical_plots_respects_plotting_disabled(
        self, mock_analyzer, mock_result_with_methods, test_data
    ):
        """Test that function returns early when plotting is disabled."""
        # Disable plotting in config
        mock_analyzer.config["output_settings"]["reporting"][
            "generate_plots"
        ] = False

        best_params, phi_angles, c2_exp = test_data

        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)

            with patch("homodyne.plotting.plot_c2_heatmaps") as mock_plot:
                # Call the function
                _generate_classical_plots(
                    mock_analyzer,
                    best_params,
                    mock_result_with_methods,
                    phi_angles,
                    c2_exp,
                    output_dir,
                )

                # Plot function should not have been called
                assert mock_plot.call_count == 0


if __name__ == "__main__":
    pytest.main([__file__])
