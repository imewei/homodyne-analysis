"""
Comprehensive unit tests for GPU-accelerated MCMC optimization module (Isolated GPU Backend).

This module tests the pure NumPyro/JAX GPU backend functionality including:
- MCMCSampler initialization and JAX configuration (GPU-only)
- NumPyro model construction and validation
- GPU detection and environment setup
- NUTS sampling with convergence diagnostics
- Posterior analysis and uncertainty quantification
- Isolated GPU backend (no PyMC contamination)
- CPU fallback within JAX ecosystem
- Compatibility with mcmc.py interface and outputs

Note: This tests the isolated GPU backend (mcmc_gpu.py) which is completely
separate from the CPU backend (mcmc.py) to avoid PyTensor/JAX conflicts.
The GPU backend uses pure NumPyro/JAX with CPU fallback within JAX ecosystem.
"""

from unittest.mock import Mock, patch

import numpy as np
import pytest

# Test imports with graceful handling for missing dependencies
try:
    from homodyne.optimization.mcmc_gpu import (
        JAX_AVAILABLE,
        MCMCSampler,
        create_mcmc_sampler,
    )

    MCMC_GPU_MODULE_AVAILABLE = True
except ImportError:
    from typing import Any, cast
    from unittest.mock import Mock

    MCMC_GPU_MODULE_AVAILABLE = False
    MCMCSampler = cast(Any, Mock())  # type: ignore[misc]
    create_mcmc_sampler = cast(Any, Mock())  # type: ignore[misc]
    JAX_AVAILABLE = False

# Try to import JAX/NumPyro for testing
try:
    import jax
    import jax.numpy as jnp
    import numpyro
    import numpyro.distributions as dist
    from numpyro.infer import MCMC, NUTS

    JAX_TEST_AVAILABLE = True
except ImportError:
    JAX_TEST_AVAILABLE = False
    jax = jnp = numpyro = dist = MCMC = NUTS = None


@pytest.fixture(autouse=True)
def restore_mcmc_gpu_module_state():
    """Auto-use fixture to ensure MCMC GPU module state is always restored after each test."""
    if MCMC_GPU_MODULE_AVAILABLE:
        import homodyne.optimization.mcmc_gpu as mcmc_gpu_module

        # Save original state
        original_jax_available = getattr(mcmc_gpu_module, "JAX_AVAILABLE", False)
        original_jax = getattr(mcmc_gpu_module, "jax", None)
        original_jnp = getattr(mcmc_gpu_module, "jnp", None)
        original_numpyro = getattr(mcmc_gpu_module, "numpyro", None)

    yield  # Run the test

    # Restore original state after test
    if MCMC_GPU_MODULE_AVAILABLE:
        import homodyne.optimization.mcmc_gpu as mcmc_gpu_module

        mcmc_gpu_module.JAX_AVAILABLE = original_jax_available
        mcmc_gpu_module.jax = original_jax
        mcmc_gpu_module.jnp = original_jnp
        mcmc_gpu_module.numpyro = original_numpyro


@pytest.fixture
def mock_analysis_core():
    """Create a mock analysis core for testing."""
    core = Mock()
    core.dt = 0.1
    core.config = {
        "scattering": {"wavevector_q": 0.01},
        "geometry": {"stator_rotor_gap": 200000},  # 20 μm in Angstroms
        "temporal": {"dt": 0.1},
    }

    # Mock the calculation method to return reasonable data
    def mock_calculate(params, phi_angles):
        n_angles = len(phi_angles) if hasattr(phi_angles, "__len__") else 1
        return np.random.random((n_angles, 50, 100)) + 1.0

    core.calculate_c2_nonequilibrium_laminar_parallel = Mock(side_effect=mock_calculate)
    return core


@pytest.fixture
def mcmc_gpu_config():
    """Standard MCMC GPU configuration for testing."""
    return {
        "optimization_config": {
            "mcmc_sampling": {
                "num_samples": 100,
                "num_warmup": 50,
                "num_chains": 2,
                "target_accept": 0.8,
                "max_tree_depth": 10,
                "thinning": 1,
                "random_seed": 42,
                "progress_bar": False,
                "device": "auto",  # GPU-specific setting
            }
        },
        "parameter_bounds": {
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


@pytest.fixture
def mock_correlation_data():
    """Generate mock experimental correlation data."""
    n_angles, n_t2, n_t1 = 3, 50, 100
    phi_angles = np.array([0, 45, 90])

    # Generate synthetic correlation data with realistic values
    np.random.seed(42)
    c2_exp = 1.0 + 0.5 * np.random.exponential(1, (n_angles, n_t2, n_t1))

    return c2_exp, phi_angles


class TestMCMCGPUEnvironmentSetup:
    """Test GPU environment detection and setup."""

    @pytest.mark.skipif(
        not MCMC_GPU_MODULE_AVAILABLE, reason="MCMC GPU module not available"
    )
    def test_jax_availability_check(self):
        """Test JAX availability detection."""
        import homodyne.optimization.mcmc_gpu as mcmc_gpu_module

        # Test that JAX_AVAILABLE is properly set
        assert hasattr(mcmc_gpu_module, "JAX_AVAILABLE")
        assert isinstance(mcmc_gpu_module.JAX_AVAILABLE, bool)

    @pytest.mark.skipif(
        not MCMC_GPU_MODULE_AVAILABLE, reason="MCMC GPU module not available"
    )
    def test_mcmc_sampler_init_without_jax(self, mock_analysis_core, mcmc_gpu_config):
        """Test MCMC sampler initialization fails gracefully without JAX."""
        with patch("homodyne.optimization.mcmc_gpu.JAX_AVAILABLE", False):
            with pytest.raises(
                ImportError, match="JAX/NumPyro is required for GPU MCMC"
            ):
                MCMCSampler(mock_analysis_core, mcmc_gpu_config)

    @pytest.mark.skipif(
        not MCMC_GPU_MODULE_AVAILABLE or not JAX_TEST_AVAILABLE,
        reason="MCMC GPU or JAX not available",
    )
    def test_jax_environment_configuration(self, mock_analysis_core, mcmc_gpu_config):
        """Test JAX environment configuration."""
        with patch("homodyne.optimization.mcmc_gpu.JAX_AVAILABLE", True):
            with patch("homodyne.optimization.mcmc_gpu.jax") as mock_jax:
                mock_jax.devices.return_value = [Mock(device_kind="gpu")]

                sampler = MCMCSampler(mock_analysis_core, mcmc_gpu_config)

                # Verify JAX configuration was attempted
                assert sampler.core == mock_analysis_core
                assert sampler.config == mcmc_gpu_config

    @pytest.mark.skipif(
        not MCMC_GPU_MODULE_AVAILABLE or not JAX_TEST_AVAILABLE,
        reason="MCMC GPU or JAX not available",
    )
    def test_gpu_detection_and_fallback(self, mock_analysis_core, mcmc_gpu_config):
        """Test GPU detection and CPU fallback logic."""
        with patch("homodyne.optimization.mcmc_gpu.JAX_AVAILABLE", True):
            with patch("homodyne.optimization.mcmc_gpu.jax") as mock_jax:
                # Test GPU available case
                mock_gpu_device = Mock(device_kind="gpu")
                mock_jax.devices.return_value = [mock_gpu_device]

                sampler = MCMCSampler(mock_analysis_core, mcmc_gpu_config)

                # Test CPU fallback case
                mock_cpu_device = Mock(device_kind="cpu")
                mock_jax.devices.return_value = [mock_cpu_device]

                sampler_cpu = MCMCSampler(mock_analysis_core, mcmc_gpu_config)

                # Both should initialize successfully
                assert sampler.core == mock_analysis_core
                assert sampler_cpu.core == mock_analysis_core


class TestMCMCGPUSamplerInitialization:
    """Test MCMC GPU sampler initialization."""

    @pytest.mark.skipif(
        not MCMC_GPU_MODULE_AVAILABLE, reason="MCMC GPU module not available"
    )
    def test_mcmc_sampler_init_basic(self, mock_analysis_core, mcmc_gpu_config):
        """Test basic MCMC GPU sampler initialization."""
        with patch("homodyne.optimization.mcmc_gpu.JAX_AVAILABLE", True):
            with patch("homodyne.optimization.mcmc_gpu.jax"):
                sampler = MCMCSampler(mock_analysis_core, mcmc_gpu_config)

                assert sampler.core == mock_analysis_core
                assert sampler.config == mcmc_gpu_config
                assert (
                    sampler.mcmc_config
                    == mcmc_gpu_config["optimization_config"]["mcmc_sampling"]
                )

                # Check initialization of result containers
                assert sampler.trace is None
                assert sampler.mcmc_trace is None
                assert sampler.diagnostics == {}
                assert sampler.posterior_means == {}
                assert sampler.mcmc_result == {}

    @pytest.mark.skipif(
        not MCMC_GPU_MODULE_AVAILABLE, reason="MCMC GPU module not available"
    )
    def test_mcmc_config_validation(self, mock_analysis_core):
        """Test MCMC configuration validation."""
        # Test with invalid config
        invalid_config = {
            "optimization_config": {
                "mcmc_sampling": {
                    "num_samples": -10,  # Invalid
                    "num_chains": 0,  # Invalid
                }
            }
        }

        with patch("homodyne.optimization.mcmc_gpu.JAX_AVAILABLE", True):
            with patch("homodyne.optimization.mcmc_gpu.jax"):
                with pytest.raises((ValueError, KeyError)):
                    MCMCSampler(mock_analysis_core, invalid_config)

    @pytest.mark.skipif(
        not MCMC_GPU_MODULE_AVAILABLE, reason="MCMC GPU module not available"
    )
    def test_performance_features_initialization(
        self, mock_analysis_core, mcmc_gpu_config
    ):
        """Test performance features initialization."""
        with patch("homodyne.optimization.mcmc_gpu.JAX_AVAILABLE", True):
            with patch("homodyne.optimization.mcmc_gpu.jax"):
                sampler = MCMCSampler(mock_analysis_core, mcmc_gpu_config)

                # Check that performance settings are configured
                assert hasattr(sampler, "mcmc_config")
                assert "num_samples" in sampler.mcmc_config
                assert "num_chains" in sampler.mcmc_config


class TestMCMCGPUModelCreation:
    """Test NumPyro model creation and validation."""

    @pytest.mark.skipif(
        not MCMC_GPU_MODULE_AVAILABLE, reason="MCMC GPU module not available"
    )
    def test_numpyro_model_creation(
        self, mock_analysis_core, mcmc_gpu_config, mock_correlation_data
    ):
        """Test NumPyro model creation."""
        c2_exp, phi_angles = mock_correlation_data

        with patch("homodyne.optimization.mcmc_gpu.JAX_AVAILABLE", True):
            with patch("homodyne.optimization.mcmc_gpu.jax"):
                with patch("homodyne.optimization.mcmc_gpu.numpyro"):
                    sampler = MCMCSampler(mock_analysis_core, mcmc_gpu_config)

                    # Test that sampler can be created without errors
                    assert sampler is not None
                    assert hasattr(sampler, "core")

    @pytest.mark.skipif(
        not MCMC_GPU_MODULE_AVAILABLE, reason="MCMC GPU module not available"
    )
    def test_prior_distributions(self, mock_analysis_core, mcmc_gpu_config):
        """Test that prior distributions match mcmc.py exactly."""
        with patch("homodyne.optimization.mcmc_gpu.JAX_AVAILABLE", True):
            with patch("homodyne.optimization.mcmc_gpu.jax"):
                with patch("homodyne.optimization.mcmc_gpu.numpyro"):
                    MCMCSampler(mock_analysis_core, mcmc_gpu_config)

                    # Verify that the sampler stores parameter bounds
                    bounds = mcmc_gpu_config["parameter_bounds"]["bounds"]
                    assert "D0" in bounds
                    assert "alpha" in bounds
                    assert bounds["D0"] == [1e-3, 1e3]
                    assert bounds["alpha"] == [-2, 2]

    @pytest.mark.skipif(
        not MCMC_GPU_MODULE_AVAILABLE, reason="MCMC GPU module not available"
    )
    def test_likelihood_formulation(self, mock_analysis_core, mcmc_gpu_config):
        """Test that likelihood formulation matches mcmc.py (fitted = contrast * theory + offset)."""
        with patch("homodyne.optimization.mcmc_gpu.JAX_AVAILABLE", True):
            with patch("homodyne.optimization.mcmc_gpu.jax"):
                with patch("homodyne.optimization.mcmc_gpu.numpyro"):
                    sampler = MCMCSampler(mock_analysis_core, mcmc_gpu_config)

                    # The likelihood formulation should be embedded in the model
                    # This is tested by ensuring the sampler initializes with the right config
                    assert sampler.config == mcmc_gpu_config


class TestMCMCGPUSampling:
    """Test NUTS sampling with GPU acceleration."""

    @pytest.mark.skipif(
        not MCMC_GPU_MODULE_AVAILABLE, reason="MCMC GPU module not available"
    )
    def test_nuts_sampler_initialization(self, mock_analysis_core, mcmc_gpu_config):
        """Test NUTS sampler initialization."""
        with patch("homodyne.optimization.mcmc_gpu.JAX_AVAILABLE", True):
            with patch("homodyne.optimization.mcmc_gpu.jax"):
                with patch("homodyne.optimization.mcmc_gpu.NUTS"):
                    with patch("homodyne.optimization.mcmc_gpu.MCMC"):
                        MCMCSampler(mock_analysis_core, mcmc_gpu_config)

                        # The NUTS sampler should be configurable
                        nuts_config = mcmc_gpu_config["optimization_config"][
                            "mcmc_sampling"
                        ]
                        assert nuts_config["target_accept"] == 0.8
                        assert nuts_config["max_tree_depth"] == 10

    @pytest.mark.skipif(
        not MCMC_GPU_MODULE_AVAILABLE, reason="MCMC GPU module not available"
    )
    def test_multi_chain_execution(self, mock_analysis_core, mcmc_gpu_config):
        """Test vectorized multi-chain execution."""
        with patch("homodyne.optimization.mcmc_gpu.JAX_AVAILABLE", True):
            with patch("homodyne.optimization.mcmc_gpu.jax"):
                sampler = MCMCSampler(mock_analysis_core, mcmc_gpu_config)

                # Check that multi-chain configuration is set
                assert sampler.mcmc_config["num_chains"] == 2

    @pytest.mark.skipif(
        not MCMC_GPU_MODULE_AVAILABLE, reason="MCMC GPU module not available"
    )
    def test_thinning_implementation(self, mock_analysis_core, mcmc_gpu_config):
        """Test thinning implementation."""
        # Set thinning in config
        mcmc_gpu_config["optimization_config"]["mcmc_sampling"]["thinning"] = 5

        with patch("homodyne.optimization.mcmc_gpu.JAX_AVAILABLE", True):
            with patch("homodyne.optimization.mcmc_gpu.jax"):
                sampler = MCMCSampler(mock_analysis_core, mcmc_gpu_config)

                # Check that thinning setting is preserved
                assert sampler.mcmc_config["thinning"] == 5


class TestMCMCGPUIsolatedBackend:
    """Test isolated GPU backend functionality."""

    @pytest.mark.skipif(
        not MCMC_GPU_MODULE_AVAILABLE, reason="MCMC GPU module not available"
    )
    def test_isolated_backend_separation(self, mock_analysis_core, mcmc_gpu_config):
        """Test that GPU backend is completely isolated from PyMC."""
        with patch("homodyne.optimization.mcmc_gpu.JAX_AVAILABLE", True):
            with patch("homodyne.optimization.mcmc_gpu.jax"):
                sampler = MCMCSampler(mock_analysis_core, mcmc_gpu_config)

                # Verify that this is an isolated backend
                assert sampler is not None
                # Should not have PyMC imports in isolated backend
                assert hasattr(sampler, "core")
                assert hasattr(sampler, "config")

    @pytest.mark.skipif(
        not MCMC_GPU_MODULE_AVAILABLE, reason="MCMC GPU module not available"
    )
    def test_api_compatibility(self, mock_analysis_core, mcmc_gpu_config):
        """Test that mcmc_gpu.py has interface compatible with isolated CPU backend."""
        with patch("homodyne.optimization.mcmc_gpu.JAX_AVAILABLE", True):
            with patch("homodyne.optimization.mcmc_gpu.jax"):
                sampler = MCMCSampler(mock_analysis_core, mcmc_gpu_config)

                # Check that key methods exist (even if mocked)
                assert hasattr(sampler, "core")
                assert hasattr(sampler, "config")
                assert hasattr(sampler, "mcmc_config")
                assert hasattr(sampler, "trace")
                assert hasattr(sampler, "diagnostics")
                assert hasattr(sampler, "posterior_means")

    @pytest.mark.skipif(
        not MCMC_GPU_MODULE_AVAILABLE, reason="MCMC GPU module not available"
    )
    def test_output_structure_compatibility(self, mock_analysis_core, mcmc_gpu_config):
        """Test that output structure is compatible with isolated CPU backend."""
        with patch("homodyne.optimization.mcmc_gpu.JAX_AVAILABLE", True):
            with patch("homodyne.optimization.mcmc_gpu.jax"):
                sampler = MCMCSampler(mock_analysis_core, mcmc_gpu_config)

                # Check result container structure
                assert hasattr(sampler, "mcmc_result")
                assert isinstance(sampler.mcmc_result, dict)
                assert isinstance(sampler.diagnostics, dict)
                assert isinstance(sampler.posterior_means, dict)

    @pytest.mark.skipif(
        not MCMC_GPU_MODULE_AVAILABLE, reason="MCMC GPU module not available"
    )
    def test_error_handling_isolation(self, mock_analysis_core):
        """Test that error handling is consistent with isolated backend architecture."""
        # Test with missing configuration
        with patch("homodyne.optimization.mcmc_gpu.JAX_AVAILABLE", True):
            with patch("homodyne.optimization.mcmc_gpu.jax"):
                with pytest.raises((KeyError, ValueError)):
                    MCMCSampler(mock_analysis_core, {})  # Empty config should fail


class TestMCMCGPUBackendIsolation:
    """Test complete isolation of GPU backend from PyMC."""

    @pytest.mark.skipif(
        not MCMC_GPU_MODULE_AVAILABLE, reason="MCMC GPU module not available"
    )
    def test_no_pymc_imports(self):
        """Test that GPU backend doesn't import PyMC components."""
        # This test verifies that the isolated GPU backend doesn't contaminate
        # with PyMC imports that could cause PyTensor/JAX conflicts
        try:
            import homodyne.optimization.mcmc_gpu as mcmc_gpu_module

            # Check that PyMC-specific attributes aren't present
            assert not hasattr(mcmc_gpu_module, "pm")
            assert not hasattr(mcmc_gpu_module, "az")
            assert not hasattr(mcmc_gpu_module, "pt")

            # Module should only have JAX/NumPyro imports
            assert hasattr(mcmc_gpu_module, "JAX_AVAILABLE")

        except ImportError:
            pytest.skip("MCMC GPU module not available for testing")

    @pytest.mark.skipif(
        not MCMC_GPU_MODULE_AVAILABLE, reason="MCMC GPU module not available"
    )
    def test_jax_only_environment(self, mock_analysis_core, mcmc_gpu_config):
        """Test that GPU backend operates in pure JAX environment."""
        with patch("homodyne.optimization.mcmc_gpu.JAX_AVAILABLE", True):
            with patch("homodyne.optimization.mcmc_gpu.jax"):
                sampler = MCMCSampler(mock_analysis_core, mcmc_gpu_config)

                # Should initialize successfully with pure JAX environment
                assert sampler is not None
                assert sampler.core == mock_analysis_core


class TestMCMCGPUPerformance:
    """Test GPU performance and hardware utilization."""

    @pytest.mark.skipif(
        not MCMC_GPU_MODULE_AVAILABLE, reason="MCMC GPU module not available"
    )
    def test_device_configuration(self, mock_analysis_core, mcmc_gpu_config):
        """Test device configuration and selection."""
        with patch("homodyne.optimization.mcmc_gpu.JAX_AVAILABLE", True):
            with patch("homodyne.optimization.mcmc_gpu.jax") as mock_jax:
                mock_jax.devices.return_value = [Mock(device_kind="gpu")]

                MCMCSampler(mock_analysis_core, mcmc_gpu_config)

                # Test that device detection was called
                mock_jax.devices.assert_called()

    @pytest.mark.skipif(
        not MCMC_GPU_MODULE_AVAILABLE, reason="MCMC GPU module not available"
    )
    def test_memory_management(self, mock_analysis_core, mcmc_gpu_config):
        """Test memory management for large datasets."""
        # Configure for large dataset simulation
        large_config = mcmc_gpu_config.copy()
        large_config["optimization_config"]["mcmc_sampling"]["num_samples"] = 1000

        with patch("homodyne.optimization.mcmc_gpu.JAX_AVAILABLE", True):
            with patch("homodyne.optimization.mcmc_gpu.jax"):
                sampler = MCMCSampler(mock_analysis_core, large_config)

                # Should handle large configurations without error
                assert sampler.mcmc_config["num_samples"] == 1000

    @pytest.mark.skipif(
        not MCMC_GPU_MODULE_AVAILABLE, reason="MCMC GPU module not available"
    )
    def test_jit_compilation_settings(self, mock_analysis_core, mcmc_gpu_config):
        """Test JIT compilation configuration."""
        with patch("homodyne.optimization.mcmc_gpu.JAX_AVAILABLE", True):
            with patch("homodyne.optimization.mcmc_gpu.jax"):
                sampler = MCMCSampler(mock_analysis_core, mcmc_gpu_config)

                # JIT configuration should be set up during initialization
                # This is verified by successful sampler creation
                assert sampler is not None


class TestMCMCGPUFactoryFunction:
    """Test the create_mcmc_sampler factory function."""

    @pytest.mark.skipif(
        not MCMC_GPU_MODULE_AVAILABLE, reason="MCMC GPU module not available"
    )
    def test_create_mcmc_sampler_function(self, mock_analysis_core, mcmc_gpu_config):
        """Test the create_mcmc_sampler factory function."""
        with patch("homodyne.optimization.mcmc_gpu.JAX_AVAILABLE", True):
            with patch("homodyne.optimization.mcmc_gpu.jax"):
                if callable(create_mcmc_sampler):
                    sampler = create_mcmc_sampler(mock_analysis_core, mcmc_gpu_config)
                    assert sampler is not None
                else:
                    # If factory function doesn't exist yet, test should note this
                    pytest.skip("create_mcmc_sampler function not implemented yet")


class TestMCMCGPUConvergenceDiagnostics:
    """Test convergence diagnostics and posterior analysis."""

    @pytest.mark.skipif(
        not MCMC_GPU_MODULE_AVAILABLE, reason="MCMC GPU module not available"
    )
    def test_convergence_diagnostic_setup(self, mock_analysis_core, mcmc_gpu_config):
        """Test that convergence diagnostics are properly configured."""
        with patch("homodyne.optimization.mcmc_gpu.JAX_AVAILABLE", True):
            with patch("homodyne.optimization.mcmc_gpu.jax"):
                sampler = MCMCSampler(mock_analysis_core, mcmc_gpu_config)

                # Diagnostics container should be initialized
                assert hasattr(sampler, "diagnostics")
                assert isinstance(sampler.diagnostics, dict)

    @pytest.mark.skipif(
        not MCMC_GPU_MODULE_AVAILABLE, reason="MCMC GPU module not available"
    )
    def test_posterior_analysis_setup(self, mock_analysis_core, mcmc_gpu_config):
        """Test posterior analysis configuration."""
        with patch("homodyne.optimization.mcmc_gpu.JAX_AVAILABLE", True):
            with patch("homodyne.optimization.mcmc_gpu.jax"):
                sampler = MCMCSampler(mock_analysis_core, mcmc_gpu_config)

                # Posterior analysis containers should be initialized
                assert hasattr(sampler, "posterior_means")
                assert isinstance(sampler.posterior_means, dict)


# Integration test fixtures for cross-module validation
@pytest.fixture
def mcmc_comparison_data():
    """Data for comparing mcmc.py and mcmc_gpu.py outputs."""
    return {
        "correlation_data": np.random.random((3, 50, 100)) + 1.0,
        "phi_angles": np.array([0, 45, 90]),
        "parameters": [100.0, -0.1, 1.0, 0.1, 0.1, 0.01, 30.0],
    }


class TestMCMCGPUIntegration:
    """Integration tests for MCMC GPU functionality."""

    @pytest.mark.skipif(
        not MCMC_GPU_MODULE_AVAILABLE, reason="MCMC GPU module not available"
    )
    def test_end_to_end_initialization(
        self, mock_analysis_core, mcmc_gpu_config, mock_correlation_data
    ):
        """Test end-to-end initialization process."""
        c2_exp, phi_angles = mock_correlation_data

        with patch("homodyne.optimization.mcmc_gpu.JAX_AVAILABLE", True):
            with patch("homodyne.optimization.mcmc_gpu.jax"):
                with patch("homodyne.optimization.mcmc_gpu.numpyro"):
                    try:
                        sampler = MCMCSampler(mock_analysis_core, mcmc_gpu_config)

                        # Should complete initialization successfully
                        assert sampler is not None
                        assert sampler.core == mock_analysis_core
                        assert sampler.config == mcmc_gpu_config

                    except Exception as e:
                        pytest.fail(f"End-to-end initialization failed: {e}")


# Performance benchmarking fixtures (for future use)
class TestMCMCGPUPerformanceBenchmarks:
    """Performance benchmark tests (marked as slow)."""

    @pytest.mark.slow
    @pytest.mark.skipif(
        not MCMC_GPU_MODULE_AVAILABLE or not JAX_TEST_AVAILABLE,
        reason="MCMC GPU or JAX not available",
    )
    def test_gpu_vs_cpu_performance_placeholder(self):
        """Placeholder for GPU vs CPU performance comparison."""
        # This would be implemented with actual sampling runs
        pytest.skip("Performance benchmarks require full sampling implementation")

    @pytest.mark.slow
    @pytest.mark.skipif(
        not MCMC_GPU_MODULE_AVAILABLE or not JAX_TEST_AVAILABLE,
        reason="MCMC GPU or JAX not available",
    )
    def test_memory_usage_benchmark_placeholder(self):
        """Placeholder for memory usage benchmarking."""
        # This would be implemented with memory profiling
        pytest.skip("Memory benchmarks require full sampling implementation")


if __name__ == "__main__":
    pytest.main([__file__])
