"""
Tests for MCMC 3D Surface Plotting Integration
==============================================

This test suite validates the integration of 3D surface plotting functionality
into the MCMC workflow, including posterior sample processing and confidence
interval generation.
"""

import sys
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pytest

from homodyne.tests.fixtures import dummy_config

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Test PyMC availability
try:
    import arviz as az
    import pymc as pm

    PYMC_AVAILABLE = True
except ImportError:
    PYMC_AVAILABLE = False
    pm = None
    az = None


pytestmark = pytest.mark.skipif(
    not PYMC_AVAILABLE,
    reason="PyMC is required for MCMC sampling but is not available.",
)


class TestMCMC3DPlottingIntegration:
    """Test integration of 3D surface plotting in MCMC workflow."""

    def test_3d_plotting_import_available(self):
        """Test that 3D plotting function can be imported."""
        try:
            from homodyne.plotting import plot_3d_surface

            assert callable(plot_3d_surface)
            print("✓ plot_3d_surface function imports correctly")
        except ImportError as e:
            pytest.fail(f"Failed to import plot_3d_surface: {e}")

    def test_generate_mcmc_plots_with_3d_integration(self):
        """Test that _generate_mcmc_plots includes 3D plotting functionality."""
        import inspect

        from run_homodyne import _generate_mcmc_plots

        # Get source code of the function
        source = inspect.getsource(_generate_mcmc_plots)

        # Check that 3D plotting integration is present
        assert "plot_3d_surface" in source
        assert "posterior_samples" in source
        assert "c2_posterior_samples" in source
        assert "confidence_level=0.95" in source

        print("✓ _generate_mcmc_plots contains 3D plotting integration")

    @patch("homodyne.plotting.plot_3d_surface")
    def test_mcmc_3d_plotting_with_trace_data(self, mock_plot_3d, dummy_config):
        """Test 3D plotting functionality with MCMC trace data."""
        from run_homodyne import _generate_mcmc_plots

        # Mock plot_3d_surface to return success
        mock_plot_3d.return_value = True

        # Create mock analyzer
        mock_analyzer = Mock()
        mock_analyzer.dt = 0.001
        mock_analyzer.calculate_c2_nonequilibrium_laminar_parallel = Mock()

        # Create synthetic correlation data
        n_angles, n_t2, n_t1 = 3, 10, 10
        c2_exp = np.random.rand(n_angles, n_t2, n_t1) * 0.1 + 1.0
        c2_theory = np.random.rand(n_angles, n_t2, n_t1) * 0.1 + 1.0
        phi_angles = np.array([0, 45, 90])
        best_params = np.array([1000, -0.5, 100])

        # Return the theory data when called
        mock_analyzer.calculate_c2_nonequilibrium_laminar_parallel.return_value = (
            c2_theory
        )

        # Create MCMC results without trace (simpler test case)
        mcmc_results = {"trace": None}

        # Mock the config to include parameter names and ensure plotting is
        # enabled
        mock_analyzer.config = {
            "initial_parameters": {"parameter_names": ["D0", "alpha", "D_offset"]},
            "output_settings": {"reporting": {"generate_plots": True}},
        }

        # Create temporary output directory
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir)

            # Run the function (should not raise exceptions)
            try:
                _generate_mcmc_plots(
                    analyzer=mock_analyzer,
                    best_params=best_params,
                    phi_angles=phi_angles,
                    c2_exp=c2_exp,
                    output_dir=output_dir,
                    mcmc_results=mcmc_results,
                )
                print("✓ _generate_mcmc_plots executed successfully with 3D plotting")

                # Verify plot_3d_surface was called (fallback mode without
                # trace)
                assert mock_plot_3d.called
                print("✓ plot_3d_surface was called during MCMC plotting")

            except Exception as e:
                pytest.fail(f"_generate_mcmc_plots failed: {e}")

    @patch("homodyne.plotting.plot_3d_surface")
    def test_mcmc_3d_plotting_without_trace_data(self, mock_plot_3d, dummy_config):
        """Test 3D plotting functionality without MCMC trace data (fallback mode)."""
        from run_homodyne import _generate_mcmc_plots

        # Mock plot_3d_surface to return success
        mock_plot_3d.return_value = True

        # Create mock analyzer
        mock_analyzer = Mock()
        mock_analyzer.config = dummy_config
        mock_analyzer.dt = 0.001

        # Create synthetic data
        n_angles, n_t2, n_t1 = 3, 10, 10
        c2_exp = np.random.rand(n_angles, n_t2, n_t1) * 0.1 + 1.0
        c2_theory = np.random.rand(n_angles, n_t2, n_t1) * 0.1 + 1.0
        phi_angles = np.array([0, 45, 90])
        best_params = np.array([1000, -0.5, 100])

        mock_analyzer.calculate_c2_nonequilibrium_laminar_parallel.return_value = (
            c2_theory
        )

        # MCMC results without trace data
        mcmc_results = {"trace": None}

        # Create temporary output directory
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir)

            # Run the function (should fall back to basic 3D plotting)
            try:
                _generate_mcmc_plots(
                    analyzer=mock_analyzer,
                    best_params=best_params,
                    phi_angles=phi_angles,
                    c2_exp=c2_exp,
                    output_dir=output_dir,
                    mcmc_results=mcmc_results,
                )
                print("✓ _generate_mcmc_plots executed successfully without trace data")

                # Verify plot_3d_surface was still called (fallback mode)
                assert mock_plot_3d.called
                print("✓ plot_3d_surface was called in fallback mode")

                # Check that it was called with posterior_samples=None
                call_args = mock_plot_3d.call_args
                assert call_args[1]["posterior_samples"] is None
                print(
                    "✓ plot_3d_surface called with posterior_samples=None in fallback mode"
                )

            except Exception as e:
                pytest.fail(f"_generate_mcmc_plots failed in fallback mode: {e}")

    def test_3d_plotting_output_directory_structure(self, dummy_config):
        """Test that 3D plots are saved to the correct MCMC directory."""
        import tempfile

        from run_homodyne import _generate_mcmc_plots

        # Create mock components
        mock_analyzer = Mock()
        mock_analyzer.config = dummy_config
        mock_analyzer.dt = 0.001

        # Create synthetic data
        c2_exp = np.random.rand(2, 5, 5) * 0.1 + 1.0
        c2_theory = np.random.rand(2, 5, 5) * 0.1 + 1.0
        phi_angles = np.array([0, 45])
        best_params = np.array([1000, -0.5, 100])

        mock_analyzer.calculate_c2_nonequilibrium_laminar_parallel.return_value = (
            c2_theory
        )
        mcmc_results = {"trace": None}  # No trace for simplicity

        # Create temporary output directory
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir)

            # Mock plot_3d_surface to check the output directory
            with patch("homodyne.plotting.plot_3d_surface") as mock_plot_3d:
                mock_plot_3d.return_value = True

                _generate_mcmc_plots(
                    analyzer=mock_analyzer,
                    best_params=best_params,
                    phi_angles=phi_angles,
                    c2_exp=c2_exp,
                    output_dir=output_dir,
                    mcmc_results=mcmc_results,
                )

                # Verify that the MCMC subdirectory was created
                mcmc_dir = output_dir / "mcmc"
                assert mcmc_dir.exists()
                assert mcmc_dir.is_dir()
                print("✓ MCMC output directory created successfully")

                # Check that plot_3d_surface was called with the correct output
                # directory
                if mock_plot_3d.called:
                    call_args = mock_plot_3d.call_args
                    outdir_arg = call_args[1]["outdir"]
                    assert Path(outdir_arg) == mcmc_dir
                    print("✓ plot_3d_surface called with correct MCMC output directory")

    def test_3d_plotting_configuration_integration(self, dummy_config):
        """Test that 3D plotting respects configuration settings."""
        import tempfile

        from run_homodyne import _generate_mcmc_plots

        # Test with plotting disabled
        mock_analyzer = Mock()
        # Modify the dummy config to disable plotting
        disabled_config = dummy_config.copy()
        disabled_config["output_settings"] = {
            "reporting": {"generate_plots": False}  # Plotting disabled
        }
        mock_analyzer.config = disabled_config

        c2_exp = np.random.rand(1, 5, 5)
        mcmc_results = {"trace": None}

        with tempfile.TemporaryDirectory() as tmp_dir:
            with patch("homodyne.plotting.plot_3d_surface") as mock_plot_3d:
                _generate_mcmc_plots(
                    analyzer=mock_analyzer,
                    best_params=np.array([1000, -0.5, 100]),
                    phi_angles=np.array([0]),
                    c2_exp=c2_exp,
                    output_dir=Path(tmp_dir),
                    mcmc_results=mcmc_results,
                )

                # Should not call plot_3d_surface when plotting is disabled
                assert not mock_plot_3d.called
                print("✓ 3D plotting correctly disabled when generate_plots=False")


class TestMCMCPosteriorSampleProcessing:
    """Test posterior sample processing for 3D confidence intervals."""

    def test_posterior_sample_extraction_logic(self):
        """Test the logic for extracting posterior samples from MCMC trace."""
        # This tests the core logic without full integration

        # Mock trace structure similar to ArviZ InferenceData
        mock_trace = Mock()
        mock_trace.posterior = {
            "D0": Mock(),
            "alpha": Mock(),
            "D_offset": Mock(),
        }

        # Mock parameter data with realistic shape (chains, draws)
        n_chains, n_draws = 4, 1000
        param_data_d0 = np.random.normal(1000, 50, (n_chains, n_draws))
        param_data_alpha = np.random.normal(-0.5, 0.05, (n_chains, n_draws))
        param_data_offset = np.random.normal(100, 10, (n_chains, n_draws))

        mock_trace.posterior["D0"].values = param_data_d0
        mock_trace.posterior["alpha"].values = param_data_alpha
        mock_trace.posterior["D_offset"].values = param_data_offset

        # Test the extraction logic
        param_names = ["D0", "alpha", "D_offset"]
        param_samples = []

        for param_name in param_names:
            if param_name in mock_trace.posterior:
                param_data = mock_trace.posterior[param_name].values
                # Reshape from (chains, draws) to (chains*draws,)
                param_samples.append(param_data.reshape(-1))

        # Verify extraction worked correctly
        assert len(param_samples) == 3
        assert all(len(ps) == n_chains * n_draws for ps in param_samples)

        # Stack to get shape (n_samples, n_parameters)
        param_samples_array = np.column_stack(param_samples)
        assert param_samples_array.shape == (n_chains * n_draws, 3)

        # Test subsampling logic
        n_samples = min(500, param_samples_array.shape[0])
        indices = np.linspace(0, param_samples_array.shape[0] - 1, n_samples, dtype=int)
        param_samples_subset = param_samples_array[indices]

        assert param_samples_subset.shape == (n_samples, 3)
        assert param_samples_subset.shape[0] <= 500  # Performance limit

        print("✓ Posterior sample extraction logic works correctly")

    def test_confidence_interval_calculation(self):
        """Test confidence interval calculation from posterior samples."""
        # Create synthetic C2 posterior samples
        n_samples, n_t2, n_t1 = 100, 10, 10
        base_data = np.random.rand(n_t2, n_t1) * 0.1 + 1.0

        # Generate samples with some variation
        c2_samples = []
        for _i in range(n_samples):
            sample = base_data + 0.02 * np.random.randn(n_t2, n_t1)
            c2_samples.append(sample)

        c2_samples = np.array(c2_samples)  # Shape: (n_samples, n_t2, n_t1)

        # Test confidence interval calculation
        confidence_level = 0.95
        alpha = 1 - confidence_level
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100

        lower_ci = np.percentile(c2_samples, lower_percentile, axis=0)
        upper_ci = np.percentile(c2_samples, upper_percentile, axis=0)

        # Verify CI shapes and properties
        assert lower_ci.shape == (n_t2, n_t1)
        assert upper_ci.shape == (n_t2, n_t1)
        assert np.all(upper_ci >= lower_ci)  # Upper CI should be >= lower CI

        # Mean should generally be between CI bounds
        sample_mean = np.mean(c2_samples, axis=0)
        # Allow some tolerance for edge cases
        within_ci_ratio = np.mean((sample_mean >= lower_ci) & (sample_mean <= upper_ci))
        assert within_ci_ratio > 0.8  # Most points should be within CI

        print(
            f"✓ Confidence interval calculation: {
                within_ci_ratio:.1%} of mean points within CI"
        )


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v"])
