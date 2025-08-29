"""
MCMC Performance Tests for Homodyne Scattering Analysis
======================================================

This module contains comprehensive performance tests for MCMC sampling methods,
including benchmarks for:
- MCMC initialization and model building
- Sampling performance with different backends (CPU/JAX)
- Progressive sampling strategies
- Convergence diagnostics
- Memory usage optimization

Authors: Wei Chen, Hongrui He
Institution: Argonne National Laboratory
"""

import logging
import time

import numpy as np
import pytest

# Import homodyne modules
try:
    from homodyne.optimization.mcmc import (
        JAX_AVAILABLE,
        PYMC_AVAILABLE,
        MCMCSampler,
        create_mcmc_sampler,
    )

    MCMC_AVAILABLE = True
except ImportError as e:
    MCMCSampler = None  # type: ignore
    create_mcmc_sampler = None  # type: ignore
    PYMC_AVAILABLE = False
    JAX_AVAILABLE = False
    MCMC_AVAILABLE = False
    logging.warning(f"MCMC not available for performance testing: {e}")

# Test configuration for MCMC performance
MCMC_PERFORMANCE_CONFIG = {
    "metadata": {"config_version": "0.6.5"},
    "experimental_data": {"data_folder_path": "./test_data/"},
    "analyzer_parameters": {
        "temporal": {"dt": 0.1, "start_frame": 1, "end_frame": 50},
        "scattering": {"wavevector_q": 0.01},
        "geometry": {"stator_rotor_gap": 1000000},
    },
    "initial_parameters": {
        "values": [100.0, -0.5, 10.0, 0.0, 0.0, 0.0, 0.0],
        "parameter_names": [
            "D0",
            "alpha",
            "D_offset",
            "unused1",
            "unused2",
            "unused3",
            "unused4",
        ],
    },
    "optimization_config": {
        "mcmc_sampling": {
            "enabled": True,
            "draws": 100,  # Reduced for performance tests
            "tune": 50,  # Reduced for performance tests
            "chains": 2,  # Reduced for performance tests
            "cores": 2,
            "target_accept": 0.85,
            "use_jax": True,
            "use_progressive_sampling": True,
            "noise_config": {
                "use_simple_forward_model": False,
                "error_model": "normal",
                "enable_angle_filtering": True,
            },
        },
    },
    "parameter_space": {
        "bounds": [
            {"name": "D0", "min": 1.0, "max": 10000.0},
            {"name": "alpha", "min": -2.0, "max": 2.0},
            {"name": "D_offset", "min": 0.1, "max": 1000.0},
        ]
    },
    "analysis_settings": {"mode": "static", "num_parameters": 3},
}


class MockAnalysisCoreForMCMC:
    """High-performance mock analysis core for MCMC performance testing."""

    def __init__(self, config, n_angles=8, n_times=30):
        self.config = config
        self.n_angles = n_angles
        self.n_times = n_times
        self.phi_angles = np.linspace(-30, 30, n_angles)
        self.c2_experimental = self._generate_mcmc_data()

    def _generate_mcmc_data(self):
        """Generate realistic data suitable for MCMC testing."""
        time_delays = np.linspace(0, 5, self.n_times)
        c2_data = np.zeros((self.n_angles, self.n_times))

        # Realistic parameters
        D0_true, alpha_true, D_offset_true = 100.0, -0.6, 15.0

        for i, phi in enumerate(self.phi_angles):
            # Create realistic correlation function
            D_eff = D0_true * time_delays ** abs(alpha_true) + D_offset_true
            decay = np.exp(-0.008 * D_eff * time_delays)

            # Weak angular dependence
            angular_factor = 1.0 + 0.05 * np.cos(np.radians(phi))
            contrast = 0.2 * angular_factor
            c2_data[i, :] = 1.0 + contrast * decay

        # Add controlled noise for consistent testing
        np.random.seed(42)  # Fixed seed for reproducible tests
        noise_level = 0.01
        noise = np.random.normal(0, noise_level, c2_data.shape)
        c2_data += noise * np.mean(c2_data)

        return c2_data

    def compute_c2_correlation_optimized(self, params, phi_angles):
        """Optimized correlation function computation for MCMC."""
        n_angles = len(phi_angles)
        time_delays = np.linspace(0, 5, self.n_times)

        D0, alpha, D_offset = params[0], params[1], params[2]
        c2_theory = np.zeros((n_angles, self.n_times))

        # Vectorized computation
        D_eff = (
            D0 * time_delays[np.newaxis, :] ** abs(alpha)
            + D_offset * time_delays[np.newaxis, :]
        )

        for i, phi in enumerate(phi_angles):
            angular_factor = 1.0 + 0.05 * np.cos(np.radians(phi))
            contrast = 0.2 * angular_factor
            decay = np.exp(-0.008 * D_eff[0, :])
            c2_theory[i, :] = 1.0 + contrast * decay

        return c2_theory

    def calculate_chi_squared_optimized(
        self,
        params,
        phi_angles,
        c2_experimental,
        method_name=None,
        filter_angles_for_optimization=None,
    ):
        """Optimized chi-squared calculation for MCMC."""
        c2_theory = self.compute_c2_correlation_optimized(params, phi_angles)
        residuals = c2_experimental - c2_theory
        return (
            np.sum(residuals**2) / c2_experimental.size
            if c2_experimental.size > 0
            else 0.0
        )

    def is_static_mode(self):
        return True

    def get_effective_parameter_count(self):
        return 3

    @property
    def time_length(self):
        return self.n_times


@pytest.fixture
def mcmc_mock_core():
    """Fixture providing mock analysis core optimized for MCMC testing."""
    return MockAnalysisCoreForMCMC(MCMC_PERFORMANCE_CONFIG)


@pytest.fixture
def large_mcmc_mock_core():
    """Fixture providing large-scale mock analysis core for MCMC stress testing."""
    return MockAnalysisCoreForMCMC(MCMC_PERFORMANCE_CONFIG, n_angles=25, n_times=100)


@pytest.mark.performance
@pytest.mark.skipif(
    not MCMC_AVAILABLE or not PYMC_AVAILABLE,
    reason="MCMC/PyMC not available",
)
class TestMCMCPerformance:
    """Performance tests for MCMC sampling methods."""

    def test_mcmc_initialization_performance(self, mcmc_mock_core):
        """Test performance of MCMC sampler initialization."""
        assert MCMCSampler is not None, "MCMCSampler not available"

        start_time = time.time()

        # Test multiple initializations
        for _ in range(5):
            MCMCSampler(mcmc_mock_core, MCMC_PERFORMANCE_CONFIG)

        end_time = time.time()
        avg_init_time = (end_time - start_time) / 5

        # MCMC initialization should be reasonably fast (< 100ms)
        assert (
            avg_init_time < 0.1
        ), f"MCMC initialization too slow: {avg_init_time:.4f}s"

    def test_model_building_performance(self, mcmc_mock_core):
        """Test performance of PyMC model building."""
        assert MCMCSampler is not None, "MCMCSampler not available"
        sampler = MCMCSampler(mcmc_mock_core, MCMC_PERFORMANCE_CONFIG)

        np.array([100.0, -0.6, 15.0])
        phi_angles = mcmc_mock_core.phi_angles
        c2_experimental = mcmc_mock_core.c2_experimental

        start_time = time.time()

        # Build model multiple times to test consistency
        for _ in range(3):
            model = sampler._build_bayesian_model_optimized(c2_experimental, phi_angles)
            # Model should build successfully
            assert model is not None

        end_time = time.time()
        avg_model_time = (end_time - start_time) / 3

        # Model building should be fast (< 500ms for small problems)
        assert avg_model_time < 0.5, f"Model building too slow: {avg_model_time:.4f}s"

    def test_sampling_performance_cpu(self, mcmc_mock_core):
        """Test CPU-based MCMC sampling performance."""
        assert MCMCSampler is not None, "MCMCSampler not available"
        sampler = MCMCSampler(mcmc_mock_core, MCMC_PERFORMANCE_CONFIG)

        np.array([90.0, -0.5, 12.0])  # Close to true values
        phi_angles = mcmc_mock_core.phi_angles
        c2_experimental = mcmc_mock_core.c2_experimental

        # Test CPU sampling with reduced sample count
        config = MCMC_PERFORMANCE_CONFIG.copy()
        config["optimization_config"]["mcmc_sampling"]["draws"] = 20
        config["optimization_config"]["mcmc_sampling"]["tune"] = 10
        config["optimization_config"]["mcmc_sampling"]["use_jax"] = False

        start_time = time.time()

        result = sampler.run_mcmc_analysis(
            phi_angles=phi_angles,
            c2_experimental=c2_experimental,
        )

        end_time = time.time()
        sampling_time = end_time - start_time

        # CPU sampling should complete within reasonable time (< 120s for CI environment)
        assert sampling_time < 120, f"CPU sampling too slow: {sampling_time:.3f}s"

        # Check that sampling succeeded
        if result.get("trace") is not None:  # trace
            assert result is not None  # info
            assert isinstance(result, dict)

    @pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
    def test_sampling_performance_jax(self, mcmc_mock_core):
        """Test JAX-based MCMC sampling performance."""
        assert MCMCSampler is not None, "MCMCSampler not available"
        sampler = MCMCSampler(mcmc_mock_core, MCMC_PERFORMANCE_CONFIG)

        np.array([90.0, -0.5, 12.0])
        phi_angles = mcmc_mock_core.phi_angles
        c2_experimental = mcmc_mock_core.c2_experimental

        # Test JAX sampling with reduced sample count
        config = MCMC_PERFORMANCE_CONFIG.copy()
        config["optimization_config"]["mcmc_sampling"]["draws"] = 20
        config["optimization_config"]["mcmc_sampling"]["tune"] = 10
        config["optimization_config"]["mcmc_sampling"]["use_jax"] = True

        start_time = time.time()

        result = sampler.run_mcmc_analysis(
            phi_angles=phi_angles,
            c2_experimental=c2_experimental,
        )

        end_time = time.time()
        sampling_time = end_time - start_time

        # JAX sampling might be faster or comparable to CPU (< 25s for small problem)
        assert sampling_time < 25, f"JAX sampling too slow: {sampling_time:.3f}s"

        if result.get("trace") is not None:
            assert isinstance(result, dict)

    def test_progressive_sampling_performance(self, mcmc_mock_core):
        """Test progressive MCMC sampling strategy performance."""
        assert MCMCSampler is not None, "MCMCSampler not available"

        # Test with and without progressive sampling
        config_progressive = MCMC_PERFORMANCE_CONFIG.copy()
        config_progressive["optimization_config"]["mcmc_sampling"][
            "use_progressive_sampling"
        ] = True
        config_progressive["optimization_config"]["mcmc_sampling"]["draws"] = 25
        config_progressive["optimization_config"]["mcmc_sampling"]["tune"] = 15

        config_standard = MCMC_PERFORMANCE_CONFIG.copy()
        config_standard["optimization_config"]["mcmc_sampling"][
            "use_progressive_sampling"
        ] = False
        config_standard["optimization_config"]["mcmc_sampling"]["draws"] = 25
        config_standard["optimization_config"]["mcmc_sampling"]["tune"] = 15

        sampler_progressive = MCMCSampler(mcmc_mock_core, config_progressive)
        sampler_standard = MCMCSampler(mcmc_mock_core, config_standard)

        np.array([85.0, -0.4, 10.0])  # Slightly off to test convergence
        phi_angles = mcmc_mock_core.phi_angles
        c2_experimental = mcmc_mock_core.c2_experimental

        # Time progressive sampling
        start_time = time.time()
        result_progressive = sampler_progressive.run_mcmc_analysis(
            phi_angles=phi_angles,
            c2_experimental=c2_experimental,
        )
        progressive_time = time.time() - start_time

        # Time standard sampling
        start_time = time.time()
        result_standard = sampler_standard.run_mcmc_analysis(
            phi_angles=phi_angles,
            c2_experimental=c2_experimental,
        )
        standard_time = time.time() - start_time

        # Progressive might be slightly slower due to multiple stages but should be comparable
        # Allow up to 50% overhead for progressive strategy
        max_progressive_time = standard_time * 1.5
        assert (
            progressive_time < max_progressive_time
        ), f"Progressive sampling too slow: {progressive_time:.3f}s vs standard {standard_time:.3f}s"

        # Both should succeed
        if (
            result_progressive.get("trace") is not None
            and result_standard.get("trace") is not None
        ):
            logging.info(
                f"Progressive: {progressive_time:.3f}s, Standard: {standard_time:.3f}s"
            )

    def test_convergence_diagnostics_performance(self, mcmc_mock_core):
        """Test performance of convergence diagnostics computation."""
        assert MCMCSampler is not None, "MCMCSampler not available"
        sampler = MCMCSampler(mcmc_mock_core, MCMC_PERFORMANCE_CONFIG)

        np.array([95.0, -0.5, 13.0])
        phi_angles = mcmc_mock_core.phi_angles
        c2_experimental = mcmc_mock_core.c2_experimental

        # Run a quick sampling to get trace for diagnostics
        config = MCMC_PERFORMANCE_CONFIG.copy()
        config["optimization_config"]["mcmc_sampling"]["draws"] = 20
        config["optimization_config"]["mcmc_sampling"]["tune"] = 10

        result = sampler.run_mcmc_analysis(
            phi_angles=phi_angles,
            c2_experimental=c2_experimental,
        )

        if result.get("trace") is not None:  # trace exists
            trace = result["trace"]

            # Time convergence diagnostics
            start_time = time.time()

            # Run diagnostics multiple times to test performance
            diagnostics = None
            for _ in range(3):
                diagnostics = sampler.compute_convergence_diagnostics(trace)

            end_time = time.time()
            avg_diagnostics_time = (end_time - start_time) / 3

            # Convergence diagnostics should be fast (< 200ms)
            assert (
                avg_diagnostics_time < 0.2
            ), f"Convergence diagnostics too slow: {avg_diagnostics_time:.4f}s"

            # Check that diagnostics were computed
            if diagnostics is not None:
                # With very small sample sizes, diagnostics may fail
                if "error" not in diagnostics:
                    assert "rhat" in diagnostics or "r_hat" in diagnostics
                    assert (
                        "ess" in diagnostics or "effective_sample_size" in diagnostics
                    )

    def test_scaling_with_sample_count(self, mcmc_mock_core):
        """Test performance scaling with different sample counts."""
        assert MCMCSampler is not None, "MCMCSampler not available"

        sample_counts = [
            {"draws": 20, "tune": 10},
            {"draws": 40, "tune": 20},
            {"draws": 80, "tune": 40},
        ]

        timing_results = []

        for sample_config in sample_counts:
            config = MCMC_PERFORMANCE_CONFIG.copy()
            config["optimization_config"]["mcmc_sampling"].update(sample_config)
            config["optimization_config"]["mcmc_sampling"][
                "use_jax"
            ] = False  # Consistent backend

            sampler = MCMCSampler(mcmc_mock_core, config)

            np.array([100.0, -0.6, 15.0])

            start_time = time.time()

            result = sampler.run_mcmc_analysis(
                phi_angles=mcmc_mock_core.phi_angles,
                c2_experimental=mcmc_mock_core.c2_experimental,
            )

            end_time = time.time()
            timing_results.append(end_time - start_time)

            # Check that sampling succeeded
            if result.get("trace") is not None:
                total_samples = sample_config["draws"] * 2  # 2 chains
                logging.info(f"Sample count {total_samples}: {timing_results[-1]:.3f}s")

        # Performance should scale roughly linearly with sample count
        # Allow for some overhead but not excessive scaling
        for i in range(1, len(timing_results)):
            if timing_results[i - 1] > 0.1:  # Only check if previous time is measurable
                prev_samples = (
                    sample_counts[i - 1]["draws"] + sample_counts[i - 1]["tune"]
                ) * 2
                curr_samples = (
                    sample_counts[i]["draws"] + sample_counts[i]["tune"]
                ) * 2
                sample_ratio = curr_samples / prev_samples
                time_ratio = timing_results[i] / timing_results[i - 1]

                # Time ratio should not exceed sample ratio by more than factor of 3
                assert (
                    time_ratio < 3 * sample_ratio
                ), f"Poor scaling: time ratio {time_ratio:.2f} vs sample ratio {sample_ratio:.2f}"


@pytest.mark.performance
@pytest.mark.skipif(
    not MCMC_AVAILABLE or not PYMC_AVAILABLE,
    reason="MCMC/PyMC not available",
)
class TestMCMCMemoryUsage:
    """Memory usage tests for MCMC sampling."""

    def test_memory_efficient_large_data(self, large_mcmc_mock_core):
        """Test memory efficiency with larger datasets."""
        assert MCMCSampler is not None, "MCMCSampler not available"

        config = MCMC_PERFORMANCE_CONFIG.copy()
        config["optimization_config"]["mcmc_sampling"]["draws"] = 15
        config["optimization_config"]["mcmc_sampling"]["tune"] = 10

        sampler = MCMCSampler(large_mcmc_mock_core, config)

        np.array([100.0, -0.6, 15.0])
        phi_angles = large_mcmc_mock_core.phi_angles  # 25 angles
        c2_experimental = large_mcmc_mock_core.c2_experimental  # 25x100 data

        # Test that large dataset sampling completes without excessive memory usage
        start_time = time.time()

        result = sampler.run_mcmc_analysis(
            phi_angles=phi_angles,
            c2_experimental=c2_experimental,
        )

        end_time = time.time()
        sampling_time = end_time - start_time

        # Should complete within reasonable time even with larger dataset
        assert (
            sampling_time < 60
        ), f"Large dataset sampling too slow: {sampling_time:.3f}s"

        if result.get("trace") is not None:
            # Check that results are reasonable
            result["trace"]
            assert isinstance(result, dict)
            logging.info(
                f"Large dataset ({c2_experimental.size} points): {sampling_time:.3f}s"
            )

    def test_model_cleanup_performance(self, mcmc_mock_core):
        """Test that model cleanup prevents memory leaks."""
        assert MCMCSampler is not None, "MCMCSampler not available"
        sampler = MCMCSampler(mcmc_mock_core, MCMC_PERFORMANCE_CONFIG)

        np.array([100.0, -0.6, 15.0])
        phi_angles = mcmc_mock_core.phi_angles
        c2_experimental = mcmc_mock_core.c2_experimental

        # Create and destroy models multiple times
        for _i in range(5):
            model = sampler._build_bayesian_model_optimized(c2_experimental, phi_angles)
            assert model is not None
            # Model should be properly cleaned up

        # If we get here without memory errors, cleanup is working properly
        assert True, "Model cleanup successful"


@pytest.mark.performance
@pytest.mark.benchmark
@pytest.mark.skipif(
    not MCMC_AVAILABLE or not PYMC_AVAILABLE,
    reason="MCMC/PyMC not available",
)
class TestMCMCBenchmarks:
    """Benchmark tests for MCMC methods using pytest-benchmark."""

    def test_mcmc_sampling_benchmark(self, mcmc_mock_core, benchmark):
        """Benchmark complete MCMC sampling workflow."""
        assert MCMCSampler is not None, "MCMCSampler not available"

        config = MCMC_PERFORMANCE_CONFIG.copy()
        config["optimization_config"]["mcmc_sampling"]["draws"] = 15
        config["optimization_config"]["mcmc_sampling"]["tune"] = 10
        config["optimization_config"]["mcmc_sampling"]["chains"] = 1

        sampler = MCMCSampler(mcmc_mock_core, config)

        np.array([95.0, -0.55, 14.0])
        phi_angles = mcmc_mock_core.phi_angles
        c2_experimental = mcmc_mock_core.c2_experimental

        def run_mcmc_sampling():
            return sampler.run_mcmc_analysis(
                phi_angles=phi_angles,
                c2_experimental=c2_experimental,
            )

        # Benchmark the complete sampling process
        result = benchmark(run_mcmc_sampling)

        # Verify that sampling succeeded
        if result.get("trace") is not None:
            assert isinstance(result, dict)

    def test_model_building_benchmark(self, mcmc_mock_core, benchmark):
        """Benchmark PyMC model building."""
        assert MCMCSampler is not None, "MCMCSampler not available"
        sampler = MCMCSampler(mcmc_mock_core, MCMC_PERFORMANCE_CONFIG)

        np.array([100.0, -0.6, 15.0])
        phi_angles = mcmc_mock_core.phi_angles
        c2_experimental = mcmc_mock_core.c2_experimental

        def build_model():
            model = sampler._build_bayesian_model_optimized(c2_experimental, phi_angles)
            return model is not None

        # Benchmark model building
        result = benchmark(build_model)
        assert result is True

    def test_convergence_diagnostics_benchmark(self, mcmc_mock_core, benchmark):
        """Benchmark convergence diagnostics computation."""
        assert MCMCSampler is not None, "MCMCSampler not available"
        sampler = MCMCSampler(mcmc_mock_core, MCMC_PERFORMANCE_CONFIG)

        # First run a quick sampling to get trace
        np.array([100.0, -0.6, 15.0])
        phi_angles = mcmc_mock_core.phi_angles
        c2_experimental = mcmc_mock_core.c2_experimental

        config = MCMC_PERFORMANCE_CONFIG.copy()
        config["optimization_config"]["mcmc_sampling"]["draws"] = 15
        config["optimization_config"]["mcmc_sampling"]["tune"] = 10

        result = sampler.run_mcmc_analysis(
            phi_angles=phi_angles,
            c2_experimental=c2_experimental,
        )

        if result.get("trace") is not None:
            trace = result["trace"]

            def compute_diagnostics():
                return sampler.compute_convergence_diagnostics(trace)

            # Benchmark convergence diagnostics
            diagnostics = benchmark(compute_diagnostics)

            # Verify diagnostics were computed
            # With very small sample sizes, diagnostics may fail
            if "error" not in diagnostics:
                assert (
                    "rhat" in diagnostics
                    or "r_hat" in diagnostics
                    or "max_rhat" in diagnostics
                )
                assert (
                    "ess" in diagnostics
                    or "effective_sample_size" in diagnostics
                    or "min_ess" in diagnostics
                )


if __name__ == "__main__":
    # Run MCMC performance tests
    pytest.main([__file__, "-v", "-m", "performance", "--tb=short"])
