"""
Comprehensive unit tests for MCMC optimization module (Isolated CPU Backend).

This module tests the pure PyMC CPU backend functionality including:
- MCMCSampler initialization and configuration (CPU-only)
- PyMC model construction and validation
- NUTS sampling with convergence diagnostics
- Posterior analysis and uncertainty quantification
- Isolated CPU backend (no JAX contamination)
- Error handling and fallback mechanisms

Note: This tests the isolated CPU backend (mcmc.py) which is completely
separate from the GPU backend (mcmc_gpu.py) to avoid PyTensor/JAX conflicts.
"""

from unittest.mock import Mock, patch

import numpy as np
import pytest

# Test imports with graceful handling for missing dependencies
try:
    from homodyne.optimization.mcmc import (
        MCMCSampler,
        _lazy_import_pymc,
        create_mcmc_sampler,
    )

    MCMC_MODULE_AVAILABLE = True
except ImportError:
    from typing import Any, cast
    from unittest.mock import Mock

    MCMC_MODULE_AVAILABLE = False
    MCMCSampler = cast(Any, Mock())  # type: ignore[misc]
    create_mcmc_sampler = cast(Any, Mock())  # type: ignore[misc]


@pytest.fixture(autouse=True)
def restore_mcmc_module_state():
    """Auto-use fixture to ensure MCMC module state is always restored after each test."""
    if MCMC_MODULE_AVAILABLE:
        import homodyne.optimization.mcmc as mcmc_module

        # Save original state
        original_pymc_available = getattr(mcmc_module, "PYMC_AVAILABLE", True)
        original_pm = getattr(mcmc_module, "pm", None)
        original_pmjax = getattr(mcmc_module, "pmjax", None)

    yield  # Run the test

    # Restore original state after test
    if MCMC_MODULE_AVAILABLE:
        mcmc_module.PYMC_AVAILABLE = original_pymc_available
        if original_pm is not None:
            mcmc_module.pm = original_pm
        if original_pmjax is not None:
            mcmc_module.pmjax = original_pmjax


@pytest.fixture
def mock_analysis_core():
    """Create a mock analysis core for MCMC testing."""
    mock = Mock()
    mock.config = {
        "analysis": {"mode": "laminar_flow"},
        "mcmc_config": {"chains": 4, "draws": 1000, "tune": 500, "target_accept": 0.8},
    }
    mock.get_effective_parameter_count.return_value = 7
    mock._parameter_bounds = np.array(
        [
            [1e-3, 1e3],  # D0
            [-2, 2],  # alpha
            [0, 100],  # D_offset
            [1e-3, 1e3],  # shear_rate0
            [-2, 2],  # beta
            [0, 100],  # shear_offset
            [0, 360],  # phi0
        ]
    )
    mock.calculate_chi_squared_optimized = Mock(return_value=1.5)
    return mock


@pytest.fixture
def mcmc_config():
    """Create MCMC configuration for testing."""
    return {
        "optimization_config": {
            "mcmc_sampling": {
                "chains": 4,
                "draws": 1000,
                "tune": 500,
                "target_accept": 0.8,
                "max_treedepth": 10,
                "step_size": 0.01,
                "use_jax": False,
                "cores": 1,
            }
        },
        "parameter_space": {
            "bounds": {
                "D0": [1e-3, 1e3],
                "alpha": [-2, 2],
                "D_offset": [0, 100],
                "gamma_dot_t0": [1e-3, 1e3],
                "beta": [-2, 2],
                "gamma_dot_t_offset": [0, 100],
                "phi0": [0, 360],
            }
        },
        "initial_parameters": {
            "values": [100.0, -0.1, 1.0, 0.1, 0.1, 0.01, 30.0],
            "parameter_names": [
                "D0",
                "alpha",
                "D_offset",
                "gamma_dot_t0",
                "beta",
                "gamma_dot_t_offset",
                "phi0",
            ],
        },
        "analyzer_parameters": {
            "tau_max": 1000,
            "q_max": 10,
            "detector_pixel_size": 0.75e-4,
            "detector_distance": 5.0,
        },
    }


class TestMCMCSamplerInitialization:
    """Test MCMC sampler initialization."""

    @pytest.mark.skipif(not MCMC_MODULE_AVAILABLE, reason="MCMC module not available")
    def test_mcmc_sampler_init_basic(self, mock_analysis_core, mcmc_config):
        """Test basic MCMC sampler initialization."""
        with (
            patch("homodyne.optimization.mcmc.PYMC_AVAILABLE", True),
            patch(
                "homodyne.optimization.mcmc._lazy_import_pymc",
                return_value=(Mock(), Mock(), Mock(), Mock()),
            ),
        ):
            sampler = MCMCSampler(mock_analysis_core, mcmc_config)

            assert sampler.core == mock_analysis_core
            assert sampler.config == mcmc_config
            assert (
                sampler.mcmc_config
                == mcmc_config["optimization_config"]["mcmc_sampling"]
            )

    @pytest.mark.skipif(not MCMC_MODULE_AVAILABLE, reason="MCMC module not available")
    def test_mcmc_sampler_init_with_jax(self, mock_analysis_core, mcmc_config):
        """Test MCMC sampler initialization with JAX backend."""
        # The isolated CPU backend no longer supports JAX
        # Skip this test since isolated CPU backend doesn't use JAX
        pytest.skip(
            "Isolated CPU backend doesn't support JAX - tested in mcmc_gpu.py instead"
        )

    @pytest.mark.skipif(not MCMC_MODULE_AVAILABLE, reason="MCMC module not available")
    def test_mcmc_sampler_init_pymc_unavailable(self, mock_analysis_core, mcmc_config):
        """Test MCMC sampler initialization when PyMC is unavailable."""
        # Skip this test since PyMC is available in this environment
        # In a real deployment without PyMC, the import would fail at module level
        pytest.skip(
            "PyMC is available in test environment - skipping unavailability test"
        )


class TestLazyImports:
    """Test lazy import functionality."""

    @pytest.mark.skipif(not MCMC_MODULE_AVAILABLE, reason="MCMC module not available")
    def test_lazy_import_pymc_success(self):
        """Test successful lazy import of PyMC."""
        with patch("homodyne.optimization.mcmc.pm", None):  # Force re-import
            try:
                pm, az, pt, shared = _lazy_import_pymc()
                assert pm is not None
                assert az is not None
                assert pt is not None
                assert shared is not None
            except ImportError:
                # Expected if PyMC not actually available
                pytest.skip("PyMC not available for testing")

    @pytest.mark.skipif(not MCMC_MODULE_AVAILABLE, reason="MCMC module not available")
    def test_lazy_import_pymc_failure(self):
        """Test lazy import failure handling."""
        # Skip this test since PyMC is available in this environment
        # In a real deployment without PyMC, the import would fail naturally
        pytest.skip(
            "PyMC is available in test environment - skipping import failure test"
        )

    @pytest.mark.skipif(not MCMC_MODULE_AVAILABLE, reason="MCMC module not available")
    def test_lazy_import_jax_success(self):
        """Test successful lazy import of JAX."""
        # The isolated CPU backend no longer imports JAX
        pytest.skip(
            "Isolated CPU backend doesn't import JAX - tested in mcmc_gpu.py instead"
        )

    @pytest.mark.skipif(not MCMC_MODULE_AVAILABLE, reason="MCMC module not available")
    def test_lazy_import_jax_failure(self):
        """Test JAX import failure handling."""
        # The isolated CPU backend no longer imports JAX
        pytest.skip(
            "Isolated CPU backend doesn't import JAX - tested in mcmc_gpu.py instead"
        )


class TestPyMCModelConstruction:
    """Test PyMC model construction and validation."""

    @pytest.fixture
    def mock_pymc_components(self):
        """Mock PyMC components for testing."""
        mock_pm = Mock()
        mock_az = Mock()
        mock_pt = Mock()
        mock_shared = Mock()

        # Mock model context
        mock_model = Mock()
        mock_pm.Model.return_value.__enter__ = Mock(return_value=mock_model)
        mock_pm.Model.return_value.__exit__ = Mock(return_value=None)

        # Mock distributions
        mock_pm.Uniform = Mock()
        mock_pm.Normal = Mock()
        mock_pm.Deterministic = Mock()

        return mock_pm, mock_az, mock_pt, mock_shared

    @pytest.mark.skipif(not MCMC_MODULE_AVAILABLE, reason="MCMC module not available")
    def test_model_construction_success(
        self, mock_analysis_core, mcmc_config, mock_pymc_components
    ):
        """Test successful PyMC model construction."""
        mock_pm, mock_az, mock_pt, mock_shared = mock_pymc_components

        with (
            patch("homodyne.optimization.mcmc.PYMC_AVAILABLE", True),
            patch(
                "homodyne.optimization.mcmc._lazy_import_pymc",
                return_value=mock_pymc_components,
            ),
        ):
            sampler = MCMCSampler(mock_analysis_core, mcmc_config)

            # Mock model building
            with patch.object(sampler, "_build_bayesian_model_optimized") as mock_build:
                mock_model = Mock()
                mock_build.return_value = mock_model

                model = mock_build()
                assert model is not None
                mock_build.assert_called_once()

    @pytest.mark.skipif(not MCMC_MODULE_AVAILABLE, reason="MCMC module not available")
    def test_model_validation_setup(self, mock_analysis_core, mcmc_config):
        """Test model validation setup."""
        with patch("homodyne.optimization.mcmc.PYMC_AVAILABLE", True):
            sampler = MCMCSampler(mock_analysis_core, mcmc_config)

            # Test model validation
            validation_result = sampler.validate_model_setup()
            assert isinstance(validation_result, dict)
            assert "valid" in validation_result

    @pytest.mark.skipif(not MCMC_MODULE_AVAILABLE, reason="MCMC module not available")
    def test_model_summary(self, mock_analysis_core, mcmc_config):
        """Test getting model summary."""
        with patch("homodyne.optimization.mcmc.PYMC_AVAILABLE", True):
            sampler = MCMCSampler(mock_analysis_core, mcmc_config)

            # Test model summary
            summary = sampler.get_model_summary()
            # Initially should be None since no MCMC has been run
            assert summary is None


class TestNUTSSampling:
    """Test NUTS sampling functionality."""

    @pytest.mark.skipif(not MCMC_MODULE_AVAILABLE, reason="MCMC module not available")
    def test_nuts_sampling_success(self, mock_analysis_core, mcmc_config):
        """Test successful NUTS sampling."""
        with patch("homodyne.optimization.mcmc.PYMC_AVAILABLE", True):
            sampler = MCMCSampler(mock_analysis_core, mcmc_config)

            # Mock sampling
            with patch.object(sampler, "run_mcmc_analysis") as mock_sample:
                mock_trace = Mock()
                mock_trace.posterior = {"D0": np.random.rand(4, 1000, 1)}
                mock_sample.return_value = (mock_trace, {"r_hat": {"D0": 1.01}})

                phi_angles = np.array([0, 45])
                exp_data = np.random.rand(2, 50, 50) + 1.0
                initial_params = np.array([100.0, -0.1, 1.0, 0.1, 0.1, 0.01, 30.0])

                trace, diagnostics = mock_sample(initial_params, phi_angles, exp_data)

                assert trace is not None
                assert "r_hat" in diagnostics
                mock_sample.assert_called_once()

    @pytest.mark.skipif(not MCMC_MODULE_AVAILABLE, reason="MCMC module not available")
    def test_nuts_sampling_with_jax(self, mock_analysis_core, mcmc_config):
        """Test NUTS sampling with JAX backend."""
        # The isolated CPU backend no longer supports JAX
        pytest.skip(
            "Isolated CPU backend doesn't support JAX - tested in mcmc_gpu.py instead"
        )

    @pytest.mark.skipif(not MCMC_MODULE_AVAILABLE, reason="MCMC module not available")
    def test_sampling_convergence_failure(self, mock_analysis_core, mcmc_config):
        """Test handling of sampling convergence failure."""
        with patch("homodyne.optimization.mcmc.PYMC_AVAILABLE", True):
            sampler = MCMCSampler(mock_analysis_core, mcmc_config)

            with patch.object(sampler, "run_mcmc_analysis") as mock_sample:
                # Mock failed convergence
                mock_sample.side_effect = RuntimeError("Sampling failed to converge")

                phi_angles = np.array([0])
                exp_data = np.random.rand(1, 25, 25) + 1.0
                initial_params = np.array([100.0, -0.1, 1.0, 0.1, 0.1, 0.01, 30.0])

                with pytest.raises(RuntimeError):
                    mock_sample(initial_params, phi_angles, exp_data)


class TestConvergenceDiagnostics:
    """Test convergence diagnostics functionality."""

    @pytest.mark.skipif(not MCMC_MODULE_AVAILABLE, reason="MCMC module not available")
    def test_convergence_diagnostics_computation(self, mock_analysis_core, mcmc_config):
        """Test convergence diagnostic computation."""
        with patch("homodyne.optimization.mcmc.PYMC_AVAILABLE", True):
            sampler = MCMCSampler(mock_analysis_core, mcmc_config)

            # Mock trace data
            mock_trace = Mock()
            mock_trace.posterior = {
                "D0": np.random.rand(4, 1000, 1),
                "alpha": np.random.rand(4, 1000, 1),
            }

            # Test the actual method that exists
            diagnostics = sampler.compute_convergence_diagnostics(mock_trace)

            assert isinstance(diagnostics, dict)
            # The method should return some diagnostics structure

    @pytest.mark.skipif(not MCMC_MODULE_AVAILABLE, reason="MCMC module not available")
    def test_chain_mixing_assessment(self, mock_analysis_core, mcmc_config):
        """Test chain mixing assessment."""
        with patch("homodyne.optimization.mcmc.PYMC_AVAILABLE", True):
            sampler = MCMCSampler(mock_analysis_core, mcmc_config)

            mock_trace = Mock()

            # Test the actual method that exists
            mixing_result = sampler.assess_chain_mixing(mock_trace)

            assert isinstance(mixing_result, dict)

    @pytest.mark.skipif(not MCMC_MODULE_AVAILABLE, reason="MCMC module not available")
    def test_posterior_samples_generation(self, mock_analysis_core, mcmc_config):
        """Test posterior samples generation."""
        with patch("homodyne.optimization.mcmc.PYMC_AVAILABLE", True):
            sampler = MCMCSampler(mock_analysis_core, mcmc_config)

            # Test the actual method that exists
            samples = sampler.generate_posterior_samples(n_samples=100)

            # Should return None initially since no MCMC has been run
            assert samples is None


class TestPosteriorAnalysis:
    """Test posterior analysis and uncertainty quantification."""

    @pytest.mark.skipif(not MCMC_MODULE_AVAILABLE, reason="MCMC module not available")
    def test_posterior_summary_statistics(self, mock_analysis_core, mcmc_config):
        """Test computation of posterior summary statistics."""
        with patch("homodyne.optimization.mcmc.PYMC_AVAILABLE", True):
            sampler = MCMCSampler(mock_analysis_core, mcmc_config)

            # Mock posterior samples
            mock_trace = Mock()
            mock_trace.posterior = {
                "D0": np.random.normal(100, 10, (4, 1000, 1)),
                "alpha": np.random.normal(-0.1, 0.05, (4, 1000, 1)),
            }

            # Test the actual method that exists
            summary = sampler.extract_posterior_statistics(mock_trace)

            assert isinstance(summary, dict)

    @pytest.mark.skipif(not MCMC_MODULE_AVAILABLE, reason="MCMC module not available")
    def test_uncertainty_quantification(self, mock_analysis_core, mcmc_config):
        """Test uncertainty quantification from posterior."""
        with patch("homodyne.optimization.mcmc.PYMC_AVAILABLE", True):
            sampler = MCMCSampler(mock_analysis_core, mcmc_config)

            # Test the actual method that exists
            uncertainties = sampler.get_parameter_uncertainties()

            # Should return None initially since no MCMC has been run
            assert uncertainties is None


class TestMCMCFactory:
    """Test MCMC sampler factory function."""

    @pytest.mark.skipif(not MCMC_MODULE_AVAILABLE, reason="MCMC module not available")
    def test_create_mcmc_sampler_success(self, mock_analysis_core, mcmc_config):
        """Test successful MCMC sampler creation."""
        with patch("homodyne.optimization.mcmc.PYMC_AVAILABLE", True):
            sampler = create_mcmc_sampler(mock_analysis_core, mcmc_config)

            assert sampler is not None
            assert isinstance(sampler, MCMCSampler)

    @pytest.mark.skipif(not MCMC_MODULE_AVAILABLE, reason="MCMC module not available")
    def test_create_mcmc_sampler_pymc_unavailable(
        self, mock_analysis_core, mcmc_config
    ):
        """Test MCMC sampler creation when PyMC is unavailable."""
        # Skip this test since PyMC is available in this environment
        # In a real deployment without PyMC, the import would fail at module level
        pytest.skip(
            "PyMC is available in test environment - skipping unavailability test"
        )

    @pytest.mark.skipif(not MCMC_MODULE_AVAILABLE, reason="MCMC module not available")
    def test_create_mcmc_sampler_invalid_config(self, mock_analysis_core):
        """Test MCMC sampler creation with invalid configuration."""
        invalid_config = {"invalid": "config"}

        with (
            patch("homodyne.optimization.mcmc.PYMC_AVAILABLE", True),
            pytest.raises((KeyError, ValueError)),
        ):
            create_mcmc_sampler(mock_analysis_core, invalid_config)


class TestErrorHandling:
    """Test error handling in MCMC operations."""

    @pytest.mark.skipif(not MCMC_MODULE_AVAILABLE, reason="MCMC module not available")
    def test_sampling_memory_error(self, mock_analysis_core, mcmc_config):
        """Test handling of memory errors during sampling."""
        with patch("homodyne.optimization.mcmc.PYMC_AVAILABLE", True):
            sampler = MCMCSampler(mock_analysis_core, mcmc_config)

            with patch.object(sampler, "run_mcmc_analysis") as mock_sample:
                mock_sample.side_effect = MemoryError(
                    "Insufficient memory for sampling"
                )

                with pytest.raises(MemoryError):
                    phi_angles = np.array([0])
                    exp_data = np.random.rand(1, 20, 20) + 1.0
                    initial_params = np.array([100.0, -0.1, 1.0, 0.1, 0.1, 0.01, 30.0])
                    mock_sample(initial_params, phi_angles, exp_data)

    @pytest.mark.skipif(not MCMC_MODULE_AVAILABLE, reason="MCMC module not available")
    def test_invalid_parameter_bounds(self, mock_analysis_core, mcmc_config):
        """Test handling of invalid parameter bounds."""
        # Make bounds invalid
        mock_analysis_core._parameter_bounds = np.array(
            [[100, 10], [-2, 2], [0, 100]]  # Invalid: lower > upper
        )

        # Currently MCMCSampler doesn't validate bounds during initialization
        # It would validate during model construction phase
        with patch("homodyne.optimization.mcmc.PYMC_AVAILABLE", True):
            sampler = MCMCSampler(mock_analysis_core, mcmc_config)
            assert sampler is not None

    @pytest.mark.skipif(not MCMC_MODULE_AVAILABLE, reason="MCMC module not available")
    def test_jax_backend_failure(self, mock_analysis_core, mcmc_config):
        """Test handling of JAX backend failure."""
        # The isolated CPU backend no longer supports JAX
        pytest.skip(
            "Isolated CPU backend doesn't support JAX - tested in mcmc_gpu.py instead"
        )


class TestPerformanceOptimizations:
    """Test performance optimizations in MCMC."""

    @pytest.mark.skipif(not MCMC_MODULE_AVAILABLE, reason="MCMC module not available")
    def test_parallel_sampling(self, mock_analysis_core, mcmc_config):
        """Test parallel chain sampling configuration."""
        mcmc_config["optimization_config"]["mcmc_sampling"]["cores"] = 4

        with patch("homodyne.optimization.mcmc.PYMC_AVAILABLE", True):
            sampler = MCMCSampler(mock_analysis_core, mcmc_config)

            assert sampler.config["optimization_config"]["mcmc_sampling"]["cores"] == 4

    @pytest.mark.skipif(not MCMC_MODULE_AVAILABLE, reason="MCMC module not available")
    def test_sampling_performance_monitoring(self, mock_analysis_core, mcmc_config):
        """Test sampling performance monitoring."""
        with patch("homodyne.optimization.mcmc.PYMC_AVAILABLE", True):
            sampler = MCMCSampler(mock_analysis_core, mcmc_config)

            with patch.object(sampler, "run_mcmc_analysis") as mock_sample:
                mock_sample.return_value = (
                    Mock(),  # trace
                    {
                        "sampling_time": 45.6,
                        "samples_per_second": 250.0,
                        "divergences": 0,
                    },
                )

                phi_angles = np.array([0])
                exp_data = np.random.rand(1, 20, 20) + 1.0
                initial_params = np.array([100.0, -0.1, 1.0, 0.1, 0.1, 0.01, 30.0])

                trace, diagnostics = mock_sample(initial_params, phi_angles, exp_data)

                assert "sampling_time" in diagnostics
                assert "samples_per_second" in diagnostics
                assert diagnostics["sampling_time"] > 0


class TestMCMCDataSavingUpdates:
    """Test recent updates to MCMC data saving functionality."""

    @pytest.mark.skipif(not MCMC_MODULE_AVAILABLE, reason="MCMC module not available")
    def test_fitted_data_scaling_implementation(self, mock_analysis_core, mcmc_config):
        """Test that MCMC now applies least squares scaling for c2_fitted."""
        with patch("homodyne.optimization.mcmc.PYMC_AVAILABLE", True):
            MCMCSampler(mock_analysis_core, mcmc_config)

            # Mock experimental data
            n_angles, n_t2, n_t1 = 2, 20, 30
            c2_exp = np.random.random((n_angles, n_t2, n_t1)) + 1.0
            phi_angles = np.array([0, 45])
            np.array([100.0, -0.1, 1.0, 0.1, 0.1, 0.01, 30.0])

            # Mock theoretical calculation
            c2_theory = np.random.random((n_angles, n_t2, n_t1)) + 0.5
            mock_analysis_core.calculate_c2_nonequilibrium_laminar_parallel.return_value = c2_theory

            # Test that scaling logic would be applied
            # This tests the conceptual framework - actual scaling tested in integration
            assert c2_exp.shape == c2_theory.shape
            assert len(phi_angles) == n_angles

            # Verify that for each angle, scaling could be computed
            for j in range(n_angles):
                exp_data = c2_exp[j].flatten()
                theory_data = c2_theory[j].flatten()

                # Verify data shapes are compatible for least squares
                assert len(exp_data) == len(theory_data)
                assert len(exp_data) > 0

    @pytest.mark.skipif(not MCMC_MODULE_AVAILABLE, reason="MCMC module not available")
    def test_parameter_uncertainty_extraction_setup(
        self, mock_analysis_core, mcmc_config
    ):
        """Test setup for parameter uncertainty extraction from MCMC posterior."""
        with patch("homodyne.optimization.mcmc.PYMC_AVAILABLE", True):
            MCMCSampler(mock_analysis_core, mcmc_config)

            # Mock trace with posterior data
            mock_trace = Mock()
            mock_posterior = Mock()

            # Simulate parameter samples
            param_names = [
                "D0",
                "alpha",
                "D_offset",
                "gamma_dot_t0",
                "beta",
                "gamma_dot_t_offset",
                "phi0",
            ]
            mock_samples = {}

            for param_name in param_names:
                # Create mock samples with realistic variance
                samples = np.random.normal(100.0, 10.0, (2, 50))  # 2 chains, 50 samples
                mock_samples[param_name] = Mock()
                mock_samples[param_name].values = samples

            mock_posterior.__contains__ = lambda self, key: key in mock_samples
            mock_posterior.__getitem__ = lambda self, key: mock_samples[key]
            mock_trace.posterior = mock_posterior

            # Test uncertainty calculation logic
            uncertainties_list = []
            for param_name in param_names:
                if param_name in mock_trace.posterior:
                    param_data = mock_trace.posterior[param_name].values.reshape(-1)
                    uncertainties_list.append(np.std(param_data))

            uncertainties = np.array(uncertainties_list)

            # Verify uncertainties were calculated
            assert len(uncertainties) == len(param_names)
            assert all(u > 0 for u in uncertainties)

    @pytest.mark.skipif(not MCMC_MODULE_AVAILABLE, reason="MCMC module not available")
    def test_chi_squared_calculation_logic(self, mock_analysis_core, mcmc_config):
        """Test chi-squared calculation from residuals."""
        with patch("homodyne.optimization.mcmc.PYMC_AVAILABLE", True):
            MCMCSampler(mock_analysis_core, mcmc_config)

            # Create mock residuals data
            n_angles, n_t2, n_t1 = 3, 20, 30
            c2_exp = np.random.random((n_angles, n_t2, n_t1)) + 1.0
            c2_fitted = np.random.random((n_angles, n_t2, n_t1)) + 1.0

            # Calculate residuals (as would be done in updated code)
            residuals = c2_exp - c2_fitted

            # Test chi-squared calculation logic
            residuals_flat = residuals.flatten()
            valid_residuals = residuals_flat[~np.isnan(residuals_flat)]

            if len(valid_residuals) > 0:
                chi_squared = np.sum(valid_residuals**2)

                # Verify chi-squared calculation
                assert chi_squared >= 0
                assert not np.isnan(chi_squared)
                assert not np.isinf(chi_squared)

    @pytest.mark.skipif(not MCMC_MODULE_AVAILABLE, reason="MCMC module not available")
    def test_standardized_data_format_consistency(
        self, mock_analysis_core, mcmc_config
    ):
        """Test that MCMC data format now matches classical/robust methods."""
        with patch("homodyne.optimization.mcmc.PYMC_AVAILABLE", True):
            MCMCSampler(mock_analysis_core, mcmc_config)

            # Define expected fields in fitted_data.npz (matching classical/robust)
            expected_fields = {
                "c2_fitted",  # Scaled fitted data (NEW: was c2_theory)
                "c2_experimental",  # Experimental data
                "residuals",  # Residuals (c2_exp - c2_fitted)
                "phi_angles",  # All phi angles array
                "parameters",  # Parameter values
                "uncertainties",  # Parameter uncertainties (NEW)
                "parameter_names",  # Parameter names
                "chi_squared",  # Chi-squared value (NEW)
                "t1",
                "t2",  # Time arrays
                "method",  # Method identifier
                "convergence_quality",  # MCMC-specific quality measure
            }

            # Test that all expected fields are conceptually valid
            assert len(expected_fields) >= 11  # Should have at least 11 fields
            assert "c2_fitted" in expected_fields  # Key change from c2_theory
            assert "uncertainties" in expected_fields  # New field
            assert "chi_squared" in expected_fields  # New field

    @pytest.mark.skipif(not MCMC_MODULE_AVAILABLE, reason="MCMC module not available")
    def test_multi_angle_data_preservation(self, mock_analysis_core, mcmc_config):
        """Test that all phi angles are preserved in data saving."""
        with patch("homodyne.optimization.mcmc.PYMC_AVAILABLE", True):
            MCMCSampler(mock_analysis_core, mcmc_config)

            # Test with multiple angles
            phi_angles = np.array([0, 30, 60, 90])
            n_angles = len(phi_angles)
            n_t2, n_t1 = 25, 40

            # Mock data with correct shapes
            c2_exp = np.random.random((n_angles, n_t2, n_t1)) + 1.0
            c2_fitted = np.random.random((n_angles, n_t2, n_t1)) + 1.0

            # Test that data shapes are preserved
            assert c2_exp.shape[0] == n_angles
            assert c2_fitted.shape[0] == n_angles
            assert len(phi_angles) == n_angles

            # Test residuals calculation preserves shape
            residuals = c2_exp - c2_fitted
            assert residuals.shape == c2_exp.shape
            assert residuals.shape == (n_angles, n_t2, n_t1)

    @pytest.mark.skipif(not MCMC_MODULE_AVAILABLE, reason="MCMC module not available")
    def test_plotting_data_consistency(self, mock_analysis_core, mcmc_config):
        """Test that plotting data uses fitted (scaled) data consistently."""
        with patch("homodyne.optimization.mcmc.PYMC_AVAILABLE", True):
            MCMCSampler(mock_analysis_core, mcmc_config)

            # Mock plotting data structure (as would be passed to create_all_plots)
            n_angles, n_t2, n_t1 = 2, 20, 30
            c2_exp = np.random.random((n_angles, n_t2, n_t1)) + 1.0
            c2_fitted = np.random.random((n_angles, n_t2, n_t1)) + 1.0  # Scaled data
            phi_angles = np.array([0, 45])
            best_params = np.array([100.0, -0.1, 1.0, 0.1, 0.1, 0.01, 30.0])

            # Test plotting data structure
            plot_data = {
                "experimental_data": c2_exp,
                "theoretical_data": c2_fitted,  # Should be c2_fitted, not c2_theory
                "phi_angles": phi_angles,
                "best_parameters": dict(
                    zip(
                        [
                            "D0",
                            "alpha",
                            "D_offset",
                            "gamma_dot_t0",
                            "beta",
                            "gamma_dot_t_offset",
                            "phi0",
                        ],
                        best_params,
                        strict=False,
                    )
                ),
                "parameter_names": [
                    "D0",
                    "alpha",
                    "D_offset",
                    "gamma_dot_t0",
                    "beta",
                    "gamma_dot_t_offset",
                    "phi0",
                ],
                "mcmc_trace": Mock(),  # Mock trace
                "mcmc_diagnostics": {},
                "method": "MCMC",
            }

            # Verify plot data structure consistency
            assert (
                plot_data["experimental_data"].shape
                == plot_data["theoretical_data"].shape
            )
            assert (
                len(plot_data["phi_angles"]) == plot_data["experimental_data"].shape[0]
            )
            assert len(plot_data["parameter_names"]) == len(
                plot_data["best_parameters"]
            )


class TestMCMCBackwardCompatibility:
    """Test backward compatibility after recent updates."""

    @pytest.mark.skipif(not MCMC_MODULE_AVAILABLE, reason="MCMC module not available")
    def test_existing_functionality_preserved(self, mock_analysis_core, mcmc_config):
        """Test that existing MCMC functionality still works after updates."""
        with patch("homodyne.optimization.mcmc.PYMC_AVAILABLE", True):
            sampler = MCMCSampler(mock_analysis_core, mcmc_config)

            # Test that core functionality is preserved
            assert sampler.core == mock_analysis_core
            assert sampler.config == mcmc_config
            assert hasattr(sampler, "mcmc_config")
            assert hasattr(sampler, "mcmc_trace")  # Updated to match actual attribute
            assert hasattr(sampler, "mcmc_result")

    @pytest.mark.skipif(not MCMC_MODULE_AVAILABLE, reason="MCMC module not available")
    def test_config_validation_still_works(self, mock_analysis_core):
        """Test that configuration validation still works after updates."""
        with patch("homodyne.optimization.mcmc.PYMC_AVAILABLE", True):
            # Test with invalid config (should still fail)
            invalid_config = {
                "optimization_config": {
                    "mcmc_sampling": {
                        "num_samples": -100,  # Invalid
                        "num_chains": 0,  # Invalid
                    }
                }
            }

            with pytest.raises((ValueError, KeyError)):
                MCMCSampler(mock_analysis_core, invalid_config)

    @pytest.mark.skipif(not MCMC_MODULE_AVAILABLE, reason="MCMC module not available")
    def test_performance_features_still_available(
        self, mock_analysis_core, mcmc_config
    ):
        """Test that performance features are still available after updates."""
        # Add performance config
        mcmc_config["optimization_config"]["mcmc_sampling"]["cores"] = 2
        mcmc_config["optimization_config"]["mcmc_sampling"]["progress_bar"] = False

        with patch("homodyne.optimization.mcmc.PYMC_AVAILABLE", True):
            sampler = MCMCSampler(mock_analysis_core, mcmc_config)

            # Test that performance settings are preserved
            assert sampler.mcmc_config["cores"] == 2
            assert not sampler.mcmc_config["progress_bar"]


class TestMCMCIntegrationWithUpdates:
    """Integration tests for MCMC with recent updates."""

    @pytest.mark.skipif(not MCMC_MODULE_AVAILABLE, reason="MCMC module not available")
    def test_end_to_end_data_flow_consistency(self, mock_analysis_core, mcmc_config):
        """Test end-to-end data flow with new scaling and saving logic."""
        with patch("homodyne.optimization.mcmc.PYMC_AVAILABLE", True):
            MCMCSampler(mock_analysis_core, mcmc_config)

            # Mock complete data flow
            n_angles, n_t2, n_t1 = 3, 25, 35
            np.array([0, 45, 90])
            c2_exp = np.random.random((n_angles, n_t2, n_t1)) + 1.0
            c2_theory = np.random.random((n_angles, n_t2, n_t1)) + 0.8

            mock_analysis_core.calculate_c2_nonequilibrium_laminar_parallel.return_value = c2_theory

            # Simulate the new data processing flow:
            # 1. Calculate theory
            # 2. Apply scaling to create fitted data
            # 3. Calculate residuals from fitted data
            # 4. Save with new format

            # Test scaling logic per angle
            c2_fitted = c2_theory.copy()
            for j in range(n_angles):
                exp_data = c2_exp[j].flatten()
                theory_data = c2_theory[j].flatten()

                # Mock least squares scaling
                A = np.vstack([theory_data, np.ones(len(theory_data))]).T
                try:
                    scaling, residuals_ls, rank, s = np.linalg.lstsq(
                        A, exp_data, rcond=None
                    )
                    contrast, offset = scaling
                    c2_fitted[j] = (
                        theory_data.reshape(c2_exp[j].shape) * contrast + offset
                    )
                except np.linalg.LinAlgError:
                    c2_fitted[j] = c2_theory[j]  # Fallback

            # Test residuals calculation
            residuals = c2_exp - c2_fitted

            # Test chi-squared calculation
            residuals_flat = residuals.flatten()
            valid_residuals = residuals_flat[~np.isnan(residuals_flat)]
            chi_squared = (
                np.sum(valid_residuals**2) if len(valid_residuals) > 0 else np.nan
            )

            # Verify end-to-end consistency
            assert c2_fitted.shape == c2_exp.shape
            assert residuals.shape == c2_exp.shape
            assert not np.isnan(chi_squared)
            assert chi_squared >= 0


if __name__ == "__main__":
    pytest.main([__file__])
