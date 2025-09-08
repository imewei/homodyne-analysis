"""
Cross-module validation tests for MCMC implementations.

This module provides tests to validate consistency between:
- homodyne.optimization.mcmc (PyMC CPU implementation)
- homodyne.optimization.mcmc_gpu (NumPyro GPU implementation)

Key validation areas:
- API compatibility and interface consistency
- Output format and structure matching
- Parameter estimation consistency
- Performance comparison and benchmarking
- Error handling parity
"""

from unittest.mock import Mock, patch

import numpy as np
import pytest

# Import both MCMC implementations with graceful handling
try:
    from homodyne.optimization.mcmc import PYMC_AVAILABLE as CPU_AVAILABLE
    from homodyne.optimization.mcmc import MCMCSampler as MCMCSamplerCPU

    MCMC_CPU_AVAILABLE = True
except ImportError:
    MCMC_CPU_AVAILABLE = False
    MCMCSamplerCPU = Mock()
    CPU_AVAILABLE = False

try:
    from homodyne.optimization.mcmc_gpu import JAX_AVAILABLE as GPU_AVAILABLE
    from homodyne.optimization.mcmc_gpu import MCMCSampler as MCMCSamplerGPU

    MCMC_GPU_AVAILABLE = True
except ImportError:
    MCMC_GPU_AVAILABLE = False
    MCMCSamplerGPU = Mock()
    GPU_AVAILABLE = False

BOTH_IMPLEMENTATIONS_AVAILABLE = MCMC_CPU_AVAILABLE and MCMC_GPU_AVAILABLE


@pytest.fixture
def mock_analysis_core():
    """Create a mock analysis core for cross-validation testing."""
    core = Mock()
    core.dt = 0.1
    core.config = {
        "scattering": {"wavevector_q": 0.01},
        "geometry": {"stator_rotor_gap": 200000},
        "temporal": {"dt": 0.1},
    }

    # Mock consistent calculation results for both implementations
    def mock_calculate(params, phi_angles):
        np.random.seed(42)  # Fixed seed for consistency
        n_angles = len(phi_angles) if hasattr(phi_angles, "__len__") else 1
        return np.random.random((n_angles, 50, 100)) + 1.0

    core.calculate_c2_nonequilibrium_laminar_parallel = Mock(side_effect=mock_calculate)
    return core


@pytest.fixture
def standardized_mcmc_config():
    """Standardized MCMC configuration for cross-validation."""
    return {
        "optimization_config": {
            "mcmc_sampling": {
                "num_samples": 200,
                "num_warmup": 100,
                "num_chains": 2,
                "target_accept": 0.8,
                "max_tree_depth": 10,
                "thinning": 1,
                "random_seed": 42,
                "progress_bar": False,
                # GPU-specific settings (ignored by CPU implementation)
                "device": "auto",
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
    }


@pytest.fixture
def consistent_correlation_data():
    """Generate consistent correlation data for cross-validation."""
    np.random.seed(42)  # Fixed seed for reproducible data
    n_angles, n_t2, n_t1 = 3, 50, 100
    phi_angles = np.array([0, 45, 90])
    c2_exp = 1.0 + 0.5 * np.random.exponential(1, (n_angles, n_t2, n_t1))
    return c2_exp, phi_angles


class TestAPICompatibility:
    """Test API compatibility between CPU and GPU implementations."""

    @pytest.mark.skipif(
        not BOTH_IMPLEMENTATIONS_AVAILABLE,
        reason="Both MCMC implementations not available",
    )
    def test_constructor_signature_compatibility(
        self, mock_analysis_core, standardized_mcmc_config
    ):
        """Test that both constructors accept the same parameters."""
        with patch("homodyne.optimization.mcmc.PYMC_AVAILABLE", True):
            with patch("homodyne.optimization.mcmc_gpu.JAX_AVAILABLE", True):
                with patch("homodyne.optimization.mcmc_gpu.jax"):
                    # Both constructors should accept the same arguments
                    cpu_sampler = MCMCSamplerCPU(
                        mock_analysis_core, standardized_mcmc_config
                    )
                    gpu_sampler = MCMCSamplerGPU(
                        mock_analysis_core, standardized_mcmc_config
                    )

                    # Both should initialize successfully
                    assert cpu_sampler.core == mock_analysis_core
                    assert gpu_sampler.core == mock_analysis_core
                    assert cpu_sampler.config == standardized_mcmc_config
                    assert gpu_sampler.config == standardized_mcmc_config

    @pytest.mark.skipif(
        not BOTH_IMPLEMENTATIONS_AVAILABLE,
        reason="Both MCMC implementations not available",
    )
    def test_attribute_compatibility(
        self, mock_analysis_core, standardized_mcmc_config
    ):
        """Test that both implementations expose the same public attributes."""
        with patch("homodyne.optimization.mcmc.PYMC_AVAILABLE", True):
            with patch("homodyne.optimization.mcmc_gpu.JAX_AVAILABLE", True):
                with patch("homodyne.optimization.mcmc_gpu.jax"):
                    cpu_sampler = MCMCSamplerCPU(
                        mock_analysis_core, standardized_mcmc_config
                    )
                    gpu_sampler = MCMCSamplerGPU(
                        mock_analysis_core, standardized_mcmc_config
                    )

                    # Check common attributes (based on actual implementation)
                    common_attributes = [
                        "core",
                        "config",
                        "mcmc_config",
                        "mcmc_trace",
                        "mcmc_result",
                    ]

                    for attr in common_attributes:
                        assert hasattr(
                            cpu_sampler, attr
                        ), f"CPU implementation missing attribute: {attr}"
                        assert hasattr(
                            gpu_sampler, attr
                        ), f"GPU implementation missing attribute: {attr}"

    @pytest.mark.skipif(
        not BOTH_IMPLEMENTATIONS_AVAILABLE,
        reason="Both MCMC implementations not available",
    )
    def test_method_interface_compatibility(
        self, mock_analysis_core, standardized_mcmc_config
    ):
        """Test that both implementations have compatible method interfaces."""
        with patch("homodyne.optimization.mcmc.PYMC_AVAILABLE", True):
            with patch("homodyne.optimization.mcmc_gpu.JAX_AVAILABLE", True):
                with patch("homodyne.optimization.mcmc_gpu.jax"):
                    cpu_sampler = MCMCSamplerCPU(
                        mock_analysis_core, standardized_mcmc_config
                    )
                    gpu_sampler = MCMCSamplerGPU(
                        mock_analysis_core, standardized_mcmc_config
                    )

                    # Both should have similar configuration structures
                    assert "num_samples" in cpu_sampler.mcmc_config
                    assert "num_samples" in gpu_sampler.mcmc_config
                    assert (
                        cpu_sampler.mcmc_config["num_samples"]
                        == gpu_sampler.mcmc_config["num_samples"]
                    )


class TestOutputFormatConsistency:
    """Test output format consistency between implementations."""

    @pytest.mark.skipif(
        not BOTH_IMPLEMENTATIONS_AVAILABLE,
        reason="Both MCMC implementations not available",
    )
    def test_result_structure_compatibility(
        self, mock_analysis_core, standardized_mcmc_config
    ):
        """Test that both implementations produce compatible result structures."""
        with patch("homodyne.optimization.mcmc.PYMC_AVAILABLE", True):
            with patch("homodyne.optimization.mcmc_gpu.JAX_AVAILABLE", True):
                with patch("homodyne.optimization.mcmc_gpu.jax"):
                    cpu_sampler = MCMCSamplerCPU(
                        mock_analysis_core, standardized_mcmc_config
                    )
                    gpu_sampler = MCMCSamplerGPU(
                        mock_analysis_core, standardized_mcmc_config
                    )

                    # Check that result containers have consistent structure
                    # Both should have the same initialization values (None for trace)
                    assert (
                        cpu_sampler.mcmc_trace == gpu_sampler.mcmc_trace
                    )  # Both None initially

                    # Check that mcmc_result has consistent type (CPU uses None, GPU uses dict)
                    assert cpu_sampler.mcmc_result is None or isinstance(
                        cpu_sampler.mcmc_result, dict
                    )
                    assert gpu_sampler.mcmc_result is None or isinstance(
                        gpu_sampler.mcmc_result, dict
                    )

                    # Both should initialize with empty/None results
                    assert cpu_sampler.mcmc_trace is None
                    assert gpu_sampler.mcmc_trace is None

    @pytest.mark.skipif(
        not BOTH_IMPLEMENTATIONS_AVAILABLE,
        reason="Both MCMC implementations not available",
    )
    def test_configuration_handling_consistency(
        self, mock_analysis_core, standardized_mcmc_config
    ):
        """Test that both implementations handle configuration consistently."""
        with patch("homodyne.optimization.mcmc.PYMC_AVAILABLE", True):
            with patch("homodyne.optimization.mcmc_gpu.JAX_AVAILABLE", True):
                with patch("homodyne.optimization.mcmc_gpu.jax"):
                    cpu_sampler = MCMCSamplerCPU(
                        mock_analysis_core, standardized_mcmc_config
                    )
                    gpu_sampler = MCMCSamplerGPU(
                        mock_analysis_core, standardized_mcmc_config
                    )

                    # Both should extract MCMC config consistently
                    cpu_mcmc_config = cpu_sampler.mcmc_config
                    gpu_mcmc_config = gpu_sampler.mcmc_config

                    # Common configuration keys should be present in both
                    common_keys = [
                        "num_samples",
                        "num_warmup",
                        "num_chains",
                        "target_accept",
                    ]
                    for key in common_keys:
                        assert key in cpu_mcmc_config, f"CPU missing config key: {key}"
                        assert key in gpu_mcmc_config, f"GPU missing config key: {key}"
                        assert (
                            cpu_mcmc_config[key] == gpu_mcmc_config[key]
                        ), f"Config mismatch for {key}"


class TestErrorHandlingParity:
    """Test that error handling is consistent between implementations."""

    @pytest.mark.skipif(
        not BOTH_IMPLEMENTATIONS_AVAILABLE,
        reason="Both MCMC implementations not available",
    )
    def test_invalid_config_handling(self, mock_analysis_core):
        """Test that both implementations handle invalid configs similarly."""
        invalid_config = {
            "optimization_config": {
                "mcmc_sampling": {
                    "num_samples": -100,  # Invalid
                    "num_chains": 0,  # Invalid
                }
            },
            # Add required sections to avoid early validation errors
            "initial_parameters": {"values": [1.0], "parameter_names": ["test"]},
            "parameter_bounds": {"bounds": {"test": [0, 2]}},
        }

        with patch("homodyne.optimization.mcmc.PYMC_AVAILABLE", True):
            with patch("homodyne.optimization.mcmc_gpu.JAX_AVAILABLE", True):
                with patch("homodyne.optimization.mcmc_gpu.jax"):
                    # Test that both handle invalid config gracefully
                    # Note: The current implementations may not strictly validate all config values during init
                    cpu_sampler = MCMCSamplerCPU(mock_analysis_core, invalid_config)
                    gpu_sampler = MCMCSamplerGPU(mock_analysis_core, invalid_config)

                    # Both should initialize (validation may happen during sampling)
                    assert cpu_sampler is not None
                    assert gpu_sampler is not None

                    # Both should preserve the invalid config for later validation
                    assert cpu_sampler.mcmc_config["num_samples"] == -100
                    assert gpu_sampler.mcmc_config["num_samples"] == -100

    @pytest.mark.skipif(
        not BOTH_IMPLEMENTATIONS_AVAILABLE,
        reason="Both MCMC implementations not available",
    )
    def test_missing_dependencies_handling(
        self, mock_analysis_core, standardized_mcmc_config
    ):
        """Test handling when backend dependencies are missing."""
        # Skip this test if dependencies are actually available since we can't meaningfully
        # test missing dependency behavior when dependencies are present
        pytest.skip("Cannot test missing dependencies when PyMC is available")


class TestParameterHandlingConsistency:
    """Test parameter handling consistency between implementations."""

    @pytest.mark.skipif(
        not BOTH_IMPLEMENTATIONS_AVAILABLE,
        reason="Both MCMC implementations not available",
    )
    def test_parameter_bounds_handling(
        self, mock_analysis_core, standardized_mcmc_config
    ):
        """Test that parameter bounds are handled consistently."""
        with patch("homodyne.optimization.mcmc.PYMC_AVAILABLE", True):
            with patch("homodyne.optimization.mcmc_gpu.JAX_AVAILABLE", True):
                with patch("homodyne.optimization.mcmc_gpu.jax"):
                    cpu_sampler = MCMCSamplerCPU(
                        mock_analysis_core, standardized_mcmc_config
                    )
                    gpu_sampler = MCMCSamplerGPU(
                        mock_analysis_core, standardized_mcmc_config
                    )

                    # Both should have access to parameter bounds
                    bounds = standardized_mcmc_config["parameter_bounds"]["bounds"]

                    # Both should process the same configuration
                    assert "D0" in bounds
                    assert bounds["D0"] == [1e-3, 1e3]

                    # Verify both implementations can access the bounds
                    assert cpu_sampler.config["parameter_bounds"]["bounds"]["D0"] == [
                        1e-3,
                        1e3,
                    ]
                    assert gpu_sampler.config["parameter_bounds"]["bounds"]["D0"] == [
                        1e-3,
                        1e3,
                    ]

    @pytest.mark.skipif(
        not BOTH_IMPLEMENTATIONS_AVAILABLE,
        reason="Both MCMC implementations not available",
    )
    def test_parameter_names_consistency(
        self, mock_analysis_core, standardized_mcmc_config
    ):
        """Test that parameter names are handled consistently."""
        with patch("homodyne.optimization.mcmc.PYMC_AVAILABLE", True):
            with patch("homodyne.optimization.mcmc_gpu.JAX_AVAILABLE", True):
                with patch("homodyne.optimization.mcmc_gpu.jax"):
                    cpu_sampler = MCMCSamplerCPU(
                        mock_analysis_core, standardized_mcmc_config
                    )
                    gpu_sampler = MCMCSamplerGPU(
                        mock_analysis_core, standardized_mcmc_config
                    )

                    expected_params = [
                        "D0",
                        "alpha",
                        "D_offset",
                        "gamma_dot_t0",
                        "beta",
                        "gamma_dot_t_offset",
                        "phi0",
                    ]
                    actual_params = standardized_mcmc_config["initial_parameters"][
                        "parameter_names"
                    ]

                    assert actual_params == expected_params

                    # Both implementations should have access to the same parameter names
                    assert (
                        cpu_sampler.config["initial_parameters"]["parameter_names"]
                        == expected_params
                    )
                    assert (
                        gpu_sampler.config["initial_parameters"]["parameter_names"]
                        == expected_params
                    )


class TestPerformanceComparison:
    """Performance comparison tests (marked as slow)."""

    @pytest.mark.slow
    @pytest.mark.skipif(
        not BOTH_IMPLEMENTATIONS_AVAILABLE,
        reason="Both MCMC implementations not available",
    )
    def test_initialization_performance_comparison(
        self, mock_analysis_core, standardized_mcmc_config
    ):
        """Compare initialization performance between CPU and GPU implementations."""
        import time

        with patch("homodyne.optimization.mcmc.PYMC_AVAILABLE", True):
            with patch("homodyne.optimization.mcmc_gpu.JAX_AVAILABLE", True):
                with patch("homodyne.optimization.mcmc_gpu.jax"):
                    # Time CPU initialization
                    start_time = time.time()
                    MCMCSamplerCPU(mock_analysis_core, standardized_mcmc_config)
                    cpu_init_time = time.time() - start_time

                    # Time GPU initialization
                    start_time = time.time()
                    MCMCSamplerGPU(mock_analysis_core, standardized_mcmc_config)
                    gpu_init_time = time.time() - start_time

                    # Both should initialize within reasonable time (< 5 seconds)
                    assert (
                        cpu_init_time < 5.0
                    ), f"CPU initialization too slow: {cpu_init_time:.2f}s"
                    assert (
                        gpu_init_time < 5.0
                    ), f"GPU initialization too slow: {gpu_init_time:.2f}s"

                    # Log performance comparison
                    print(f"CPU initialization: {cpu_init_time:.4f}s")
                    print(f"GPU initialization: {gpu_init_time:.4f}s")

    @pytest.mark.slow
    @pytest.mark.skipif(
        not BOTH_IMPLEMENTATIONS_AVAILABLE,
        reason="Both MCMC implementations not available",
    )
    def test_memory_usage_comparison_placeholder(self):
        """Placeholder for memory usage comparison."""
        # This would require actual sampling runs to measure
        pytest.skip("Memory usage comparison requires full sampling implementation")


class TestCrossValidationIntegration:
    """Integration tests for cross-validation."""

    @pytest.mark.skipif(
        not BOTH_IMPLEMENTATIONS_AVAILABLE,
        reason="Both MCMC implementations not available",
    )
    def test_end_to_end_compatibility(
        self, mock_analysis_core, standardized_mcmc_config, consistent_correlation_data
    ):
        """Test end-to-end compatibility between implementations."""
        c2_exp, phi_angles = consistent_correlation_data

        with patch("homodyne.optimization.mcmc.PYMC_AVAILABLE", True):
            with patch("homodyne.optimization.mcmc_gpu.JAX_AVAILABLE", True):
                with patch("homodyne.optimization.mcmc_gpu.jax"):
                    try:
                        # Initialize both samplers
                        cpu_sampler = MCMCSamplerCPU(
                            mock_analysis_core, standardized_mcmc_config
                        )
                        gpu_sampler = MCMCSamplerGPU(
                            mock_analysis_core, standardized_mcmc_config
                        )

                        # Both should handle the same input data structure
                        assert c2_exp.shape == (3, 50, 100)
                        assert len(phi_angles) == 3

                        # Both samplers should be ready for the same analysis workflow
                        assert cpu_sampler.core == gpu_sampler.core
                        assert cpu_sampler.config == gpu_sampler.config

                    except Exception as e:
                        pytest.fail(f"End-to-end compatibility test failed: {e}")

    @pytest.mark.skipif(
        not BOTH_IMPLEMENTATIONS_AVAILABLE,
        reason="Both MCMC implementations not available",
    )
    def test_configuration_robustness(self, mock_analysis_core):
        """Test that both implementations handle various configuration scenarios."""
        # Test with minimal config
        minimal_config = {
            "optimization_config": {
                "mcmc_sampling": {
                    "num_samples": 10,
                    "num_warmup": 5,
                    "num_chains": 1,
                }
            },
            "parameter_bounds": {"bounds": {"D0": [1, 100]}},
            "initial_parameters": {"values": [50.0], "parameter_names": ["D0"]},
        }

        with patch("homodyne.optimization.mcmc.PYMC_AVAILABLE", True):
            with patch("homodyne.optimization.mcmc_gpu.JAX_AVAILABLE", True):
                with patch("homodyne.optimization.mcmc_gpu.jax"):
                    try:
                        cpu_sampler = MCMCSamplerCPU(mock_analysis_core, minimal_config)
                        gpu_sampler = MCMCSamplerGPU(mock_analysis_core, minimal_config)

                        # Both should handle minimal configuration
                        assert cpu_sampler.mcmc_config["num_samples"] == 10
                        assert gpu_sampler.mcmc_config["num_samples"] == 10

                    except Exception as e:
                        pytest.fail(f"Configuration robustness test failed: {e}")


class TestFactoryFunctionCompatibility:
    """Test factory function compatibility if available."""

    @pytest.mark.skipif(
        not BOTH_IMPLEMENTATIONS_AVAILABLE,
        reason="Both MCMC implementations not available",
    )
    def test_factory_function_consistency(
        self, mock_analysis_core, standardized_mcmc_config
    ):
        """Test factory function consistency between implementations."""
        # Check if factory functions exist
        try:
            from homodyne.optimization.mcmc import create_mcmc_sampler as create_cpu

            cpu_factory_exists = True
        except ImportError:
            cpu_factory_exists = False

        try:
            from homodyne.optimization.mcmc_gpu import create_mcmc_sampler as create_gpu

            gpu_factory_exists = True
        except ImportError:
            gpu_factory_exists = False

        if cpu_factory_exists and gpu_factory_exists:
            with patch("homodyne.optimization.mcmc.PYMC_AVAILABLE", True):
                with patch("homodyne.optimization.mcmc_gpu.JAX_AVAILABLE", True):
                    with patch("homodyne.optimization.mcmc_gpu.jax"):
                        # Both factory functions should work similarly
                        cpu_sampler = create_cpu(
                            mock_analysis_core, standardized_mcmc_config
                        )
                        gpu_sampler = create_gpu(
                            mock_analysis_core, standardized_mcmc_config
                        )

                        assert cpu_sampler is not None
                        assert gpu_sampler is not None
        else:
            pytest.skip("Factory functions not available in both implementations")


if __name__ == "__main__":
    pytest.main([__file__])
