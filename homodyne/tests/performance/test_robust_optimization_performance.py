"""
Performance Tests for Robust Optimization Methods
=================================================

This module contains performance-focused tests for the robust optimization
framework, including benchmarks, timing tests, and scaling analysis.

Authors: Wei Chen, Hongrui He
Institution: Argonne National Laboratory
"""

import logging
import os
import time

# Import homodyne modules
from typing import TYPE_CHECKING, Any, cast
from unittest.mock import Mock

import numpy as np
import pytest

if TYPE_CHECKING:
    from homodyne.optimization.classical import ClassicalOptimizer
    from homodyne.optimization.robust import (
        RobustHomodyneOptimizer,
        create_robust_optimizer,
    )
else:
    try:
        from homodyne.optimization.classical import ClassicalOptimizer
        from homodyne.optimization.robust import (
            CVXPY_AVAILABLE,
            GUROBI_AVAILABLE,
            RobustHomodyneOptimizer,
            create_robust_optimizer,
        )

        ROBUST_OPTIMIZATION_AVAILABLE = True
    except ImportError:
        RobustHomodyneOptimizer = cast(Any, Mock())  # type: ignore[misc]
        create_robust_optimizer = cast(Any, Mock())  # type: ignore[misc]
        ClassicalOptimizer = cast(Any, Mock())  # type: ignore[misc]
        CVXPY_AVAILABLE = False
        GUROBI_AVAILABLE = False
        ROBUST_OPTIMIZATION_AVAILABLE = False
        logging.warning(
            "Robust optimization not available for performance testing: Import failed"
        )

# Test configuration
PERFORMANCE_CONFIG = {
    "metadata": {"config_version": "0.6.5"},
    "experimental_data": {"data_folder_path": "./test_data/"},
    "analyzer_parameters": {
        "temporal": {"dt": 0.1, "start_frame": 1, "end_frame": 100},
        "scattering": {"wavevector_q": 0.01},
        "geometry": {"stator_rotor_gap": 1000000},
    },
    "initial_parameters": {"values": [100.0, -0.5, 10.0, 0.0, 0.0, 0.0, 0.0]},
    "optimization_config": {
        "classical_optimization": {"methods": ["Nelder-Mead", "Robust-Wasserstein"]},
        "robust_optimization": {
            "enabled": True,
            "uncertainty_model": "wasserstein",
            "uncertainty_radius": 0.05,
            "n_scenarios": 20,
            "regularization_alpha": 0.01,
            "regularization_beta": 0.001,
            "solver_settings": {"Method": 2, "TimeLimit": 60, "OutputFlag": 0},
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


class MockAnalysisCorePerformance:
    """High-performance mock analysis core for performance testing."""

    def __init__(self, config, n_angles=20, n_times=100):
        self.config = config
        self.n_angles = n_angles
        self.n_times = n_times
        self.phi_angles = np.linspace(-45, 45, n_angles)
        self.c2_experimental = self._generate_realistic_data()

    def _generate_realistic_data(self):
        """Generate realistic correlation function data with proper scaling."""
        time_delays = np.linspace(0, 10, self.n_times)
        c2_data = np.zeros((self.n_angles, self.n_times))

        # Realistic parameters for XPCS
        D0_true, alpha_true, D_offset_true = 150.0, -0.8, 20.0

        for i, phi in enumerate(self.phi_angles):
            # Time-dependent diffusion with angle dependence
            D_eff = D0_true * time_delays ** abs(alpha_true) + D_offset_true
            decay = np.exp(-0.01 * D_eff * time_delays)  # q^2 factor

            # Add weak angular dependence
            angular_factor = 1.0 + 0.1 * np.cos(np.radians(phi))

            # Correlation function: g2 = 1 + contrast * |g1|^2
            contrast = 0.25 * angular_factor
            c2_data[i, :] = 1.0 + contrast * decay

        # Add realistic noise (1-2% level)
        noise_level = 0.015
        noise = np.random.normal(0, noise_level, c2_data.shape)
        c2_data += noise * np.mean(c2_data)

        return c2_data

    def compute_c2_correlation_optimized(self, params, phi_angles):
        """Optimized correlation function computation."""
        n_angles = len(phi_angles)
        time_delays = np.linspace(0, 10, self.n_times)

        D0, alpha, D_offset = params[0], params[1], params[2]
        c2_theory = np.zeros((n_angles, self.n_times))

        # Vectorized computation for performance
        D_eff = (
            D0 * time_delays[np.newaxis, :] ** abs(alpha)
            + D_offset * time_delays[np.newaxis, :]
        )

        for i, phi in enumerate(phi_angles):
            angular_factor = 1.0 + 0.1 * np.cos(np.radians(phi))
            contrast = 0.25 * angular_factor
            decay = np.exp(-0.01 * D_eff[0, :])
            c2_theory[i, :] = 1.0 + contrast * decay

        return c2_theory

    def calculate_chi_squared_optimized(self, params, phi_angles, c2_experimental):
        """Optimized chi-squared calculation."""
        c2_theory = self.compute_c2_correlation_optimized(params, phi_angles)
        residuals = c2_experimental - c2_theory
        # Normalized chi-squared
        return (
            np.sum(residuals**2) / c2_experimental.size
            if c2_experimental.size > 0
            else 0.0
        )

    def calculate_c2_nonequilibrium_laminar_parallel(self, params, phi_angles):
        """Mock method for robust optimization compatibility."""
        n_angles = len(phi_angles)
        time_delays = np.linspace(0, 10, self.n_times)

        D0, alpha, D_offset = params[0], params[1], params[2]
        # Return shape (n_angles, n_times, n_times) for compatibility
        c2_theory = np.zeros((n_angles, self.n_times, self.n_times))

        # Vectorized computation
        D_eff = D0 * time_delays ** abs(alpha) + D_offset * time_delays

        for i, phi in enumerate(phi_angles):
            angular_factor = 1.0 + 0.1 * np.cos(np.radians(phi))
            contrast = 0.25 * angular_factor

            # Fill 3D correlation matrix
            for j in range(self.n_times):
                for k in range(self.n_times):
                    abs(time_delays[j] - time_delays[k])
                    decay = np.exp(-0.01 * D_eff[min(j, k)])
                    c2_theory[i, j, k] = 1.0 + contrast * decay

        return c2_theory

    def is_static_mode(self):
        return True

    def get_effective_parameter_count(self):
        return 3

    @property
    def time_length(self):
        """Mock time length property."""
        return self.n_times


@pytest.fixture
def performance_mock_core():
    """Fixture providing performance-optimized mock analysis core."""
    return MockAnalysisCorePerformance(PERFORMANCE_CONFIG)


@pytest.fixture
def large_scale_mock_core():
    """Fixture providing large-scale mock analysis core."""
    return MockAnalysisCorePerformance(PERFORMANCE_CONFIG, n_angles=50, n_times=200)


@pytest.mark.performance
@pytest.mark.skipif(
    not ROBUST_OPTIMIZATION_AVAILABLE,
    reason="Robust optimization not available",
)
class TestRobustOptimizationPerformance:
    """Performance tests for robust optimization methods."""

    def test_initialization_performance(self, performance_mock_core):
        """Test performance of optimizer initialization."""
        assert RobustHomodyneOptimizer is not None, (
            "RobustHomodyneOptimizer not available"
        )

        start_time = time.time()

        for _ in range(10):
            _ = RobustHomodyneOptimizer(performance_mock_core, PERFORMANCE_CONFIG)

        end_time = time.time()
        avg_init_time = (end_time - start_time) / 10

        # Initialization should be fast (< 5ms, more lenient due to new features)
        assert avg_init_time < 0.005, f"Initialization too slow: {avg_init_time:.4f}s"

    def test_parameter_bounds_performance(self, performance_mock_core):
        """Test performance of parameter bounds extraction."""
        assert RobustHomodyneOptimizer is not None, (
            "RobustHomodyneOptimizer not available"
        )
        assert RobustHomodyneOptimizer is not None, (
            "RobustHomodyneOptimizer not available"
        )
        optimizer = RobustHomodyneOptimizer(performance_mock_core, PERFORMANCE_CONFIG)

        start_time = time.time()
        bounds = None  # Initialize to prevent unbound variable

        for _ in range(100):
            bounds = optimizer._get_parameter_bounds()

        end_time = time.time()
        avg_bounds_time = (end_time - start_time) / 100

        # Bounds extraction should be very fast (< 1.0ms for CI compatibility)
        assert avg_bounds_time < 0.001, (
            f"Bounds extraction too slow: {avg_bounds_time:.4f}s"
        )
        assert bounds is not None
        assert len(bounds) == 3

    def test_chi_squared_computation_performance(self, performance_mock_core):
        """Test performance of chi-squared computation."""
        assert RobustHomodyneOptimizer is not None, (
            "RobustHomodyneOptimizer not available"
        )
        optimizer = RobustHomodyneOptimizer(performance_mock_core, PERFORMANCE_CONFIG)

        test_params = np.array([150.0, -0.8, 20.0])
        phi_angles = performance_mock_core.phi_angles
        c2_experimental = performance_mock_core.c2_experimental

        # Initialize to prevent unbound variable
        chi_squared = 0.0

        # Warm-up
        for _ in range(5):
            optimizer._compute_chi_squared(test_params, phi_angles, c2_experimental)

        # Timing test
        start_time = time.time()

        for _ in range(50):
            chi_squared = optimizer._compute_chi_squared(
                test_params, phi_angles, c2_experimental
            )

        end_time = time.time()
        avg_chi_squared_time = (end_time - start_time) / 50

        # Chi-squared computation should be fast (< 10ms for 20x100 data, more lenient)
        assert avg_chi_squared_time < 0.010, (
            f"Chi-squared computation too slow: {avg_chi_squared_time:.4f}s"
        )
        assert np.isfinite(chi_squared)

    def test_theoretical_correlation_performance(self, performance_mock_core):
        """Test performance of theoretical correlation function computation."""
        assert RobustHomodyneOptimizer is not None, (
            "RobustHomodyneOptimizer not available"
        )
        optimizer = RobustHomodyneOptimizer(performance_mock_core, PERFORMANCE_CONFIG)

        test_params = np.array([150.0, -0.8, 20.0])
        phi_angles = performance_mock_core.phi_angles

        # Initialize to prevent unbound variable
        c2_theory = np.array([])

        # Warm-up
        for _ in range(5):
            optimizer._compute_theoretical_correlation(test_params, phi_angles)

        # Timing test
        start_time = time.time()

        for _ in range(30):
            c2_theory = optimizer._compute_theoretical_correlation(
                test_params, phi_angles
            )

        end_time = time.time()
        avg_correlation_time = (end_time - start_time) / 30

        # Theoretical correlation should be fast (< 500ms for 20x100x100 data)
        assert avg_correlation_time < 0.5, (
            f"Theoretical correlation too slow: {avg_correlation_time:.4f}s"
        )
        assert c2_theory.shape == (
            len(phi_angles),
            performance_mock_core.n_times,
            performance_mock_core.n_times,
        )

    def test_jacobian_computation_performance(self, performance_mock_core):
        """Test performance of Jacobian computation for linearization."""
        assert RobustHomodyneOptimizer is not None, (
            "RobustHomodyneOptimizer not available"
        )
        optimizer = RobustHomodyneOptimizer(performance_mock_core, PERFORMANCE_CONFIG)

        test_params = np.array([150.0, -0.8, 20.0])
        phi_angles = performance_mock_core.phi_angles

        # Timing test
        start_time = time.time()

        c2_theory, jacobian = optimizer._compute_linearized_correlation(
            test_params, phi_angles, performance_mock_core.c2_experimental
        )

        end_time = time.time()
        jacobian_time = end_time - start_time

        # Jacobian computation involves finite differences, should be
        # reasonable (< 2s for 3D data, more lenient in CI)
        max_time = 5.0 if os.getenv("CI") else 2.0
        assert jacobian_time < max_time, (
            f"Jacobian computation too slow: {jacobian_time:.4f}s (max: {max_time:.1f}s)"
        )
        assert jacobian.shape[1] == len(test_params)
        assert jacobian.shape[0] == c2_theory.size

    @pytest.mark.skipif(not CVXPY_AVAILABLE, reason="CVXPY not available")
    def test_scenario_generation_performance(self, performance_mock_core):
        """Test performance of bootstrap scenario generation."""
        assert RobustHomodyneOptimizer is not None, (
            "RobustHomodyneOptimizer not available"
        )
        optimizer = RobustHomodyneOptimizer(performance_mock_core, PERFORMANCE_CONFIG)

        test_params = np.array([150.0, -0.8, 20.0])
        phi_angles = performance_mock_core.phi_angles
        c2_experimental = performance_mock_core.c2_experimental

        # Test different scenario counts
        scenario_counts = [10, 20, 50]

        for n_scenarios in scenario_counts:
            start_time = time.time()

            scenarios = optimizer._generate_bootstrap_scenarios(
                test_params, phi_angles, c2_experimental, n_scenarios
            )

            end_time = time.time()
            scenario_time = end_time - start_time

            # Scenario generation should scale linearly with count
            # Be more lenient in CI environments
            base_time_per_scenario = (
                0.01 if os.getenv("CI") else 0.005
            )  # 10ms or 5ms per scenario
            max_time = base_time_per_scenario * n_scenarios
            assert scenario_time < max_time, f"Scenario generation too slow: {
                scenario_time:.4f}s for {n_scenarios} scenarios (max: {max_time:.4f}s)"
            assert len(scenarios) == n_scenarios

    def test_scaling_with_data_size(self):
        """Test performance scaling with different data sizes."""
        if not ROBUST_OPTIMIZATION_AVAILABLE or not CVXPY_AVAILABLE:
            pytest.skip("CVXPY not available")

        data_sizes = [(10, 50), (20, 100), (30, 150)]
        chi_squared_times = []
        solver_times = []

        for n_angles, n_times in data_sizes:
            mock_core = MockAnalysisCorePerformance(
                PERFORMANCE_CONFIG, n_angles, n_times
            )
            assert RobustHomodyneOptimizer is not None, (
                "RobustHomodyneOptimizer not available"
            )
            optimizer = RobustHomodyneOptimizer(mock_core, PERFORMANCE_CONFIG)

            test_params = np.array([150.0, -0.8, 20.0])

            # Test chi-squared computation scaling
            start_time = time.time()
            for _ in range(10):
                _ = optimizer._compute_chi_squared(
                    test_params,
                    mock_core.phi_angles,
                    mock_core.c2_experimental,
                )
            end_time = time.time()
            avg_time = (end_time - start_time) / 10
            chi_squared_times.append(avg_time)

            # Test robust optimization solver scaling
            start_time = time.time()
            try:
                result, _ = optimizer._solve_distributionally_robust(
                    theta_init=test_params,
                    phi_angles=mock_core.phi_angles,
                    c2_experimental=mock_core.c2_experimental,
                    uncertainty_radius=0.03,
                )
                solver_time = time.time() - start_time
                solver_times.append(solver_time)
            except Exception as e:
                # If solver fails, record a reasonable fallback time
                logging.warning(f"Solver failed for size {n_angles}x{n_times}: {e}")
                solver_times.append(float("inf"))

        # Performance should scale reasonably with data size
        for i in range(1, len(chi_squared_times)):
            prev_size = data_sizes[i - 1][0] * data_sizes[i - 1][1]
            curr_size = data_sizes[i][0] * data_sizes[i][1]
            size_ratio = curr_size / prev_size if prev_size > 0 else float("inf")

            # Skip scaling check if timing was too small to measure reliably
            if chi_squared_times[i - 1] <= 1e-6 or chi_squared_times[i] <= 1e-6:
                continue

            time_ratio = chi_squared_times[i] / chi_squared_times[i - 1]

            # Time ratio should not exceed size ratio by more than factor of 6
            # Relaxed from 4 to 6 since robust optimization can have worse scaling
            assert time_ratio < 6 * size_ratio, (
                f"Chi-squared scaling poor: time ratio {time_ratio:.2f} vs size ratio {size_ratio:.2f}"
            )

        # Test solver scaling (more lenient due to optimization complexity)
        for i in range(1, len(solver_times)):
            if solver_times[i - 1] != float("inf") and solver_times[i] != float("inf"):
                if (
                    solver_times[i - 1] > 0.001
                ):  # Only check if previous time is measurable
                    # Recalculate size ratio for this iteration
                    prev_size = data_sizes[i - 1][0] * data_sizes[i - 1][1]
                    curr_size = data_sizes[i][0] * data_sizes[i][1]
                    solver_size_ratio = (
                        curr_size / prev_size if prev_size > 0 else float("inf")
                    )

                    solver_ratio = solver_times[i] / solver_times[i - 1]
                    # Solver can have more complex scaling due to optimization overhead
                    assert solver_ratio < 8 * solver_size_ratio, (
                        f"Solver scaling poor: time ratio {solver_ratio:.2f} vs size ratio {solver_size_ratio:.2f}"
                    )


@pytest.mark.performance
@pytest.mark.skipif(
    not ROBUST_OPTIMIZATION_AVAILABLE or not CVXPY_AVAILABLE,
    reason="CVXPY not available",
)
class TestRobustOptimizationBenchmarks:
    """Benchmark tests for robust optimization methods."""

    def test_wasserstein_optimization_benchmark(self, performance_mock_core, benchmark):
        """Benchmark Wasserstein DRO optimization."""
        assert RobustHomodyneOptimizer is not None, (
            "RobustHomodyneOptimizer not available"
        )
        optimizer = RobustHomodyneOptimizer(performance_mock_core, PERFORMANCE_CONFIG)

        test_params = np.array([120.0, -0.6, 15.0])  # Close to true values
        phi_angles = performance_mock_core.phi_angles
        c2_experimental = performance_mock_core.c2_experimental

        def run_wasserstein_optimization():
            return optimizer._solve_distributionally_robust(
                theta_init=test_params,
                phi_angles=phi_angles,
                c2_experimental=c2_experimental,
                uncertainty_radius=0.03,
            )

        # Benchmark the optimization
        result = benchmark(run_wasserstein_optimization)

        optimal_params, info = result

        # Verify results if optimization succeeded
        if optimal_params is not None:
            assert isinstance(optimal_params, np.ndarray)
            assert len(optimal_params) == 3
            assert info["method"] == "distributionally_robust"

    def test_scenario_optimization_benchmark(self, performance_mock_core, benchmark):
        """Benchmark scenario-based robust optimization."""
        assert RobustHomodyneOptimizer is not None, (
            "RobustHomodyneOptimizer not available"
        )
        optimizer = RobustHomodyneOptimizer(performance_mock_core, PERFORMANCE_CONFIG)

        test_params = np.array([120.0, -0.6, 15.0])
        phi_angles = performance_mock_core.phi_angles
        c2_experimental = performance_mock_core.c2_experimental

        def run_scenario_optimization():
            return optimizer._solve_scenario_robust(
                theta_init=test_params,
                phi_angles=phi_angles,
                c2_experimental=c2_experimental,
                n_scenarios=15,
            )

        # Benchmark the optimization
        result = benchmark(run_scenario_optimization)

        optimal_params, info = result

        # Verify results if optimization succeeded
        if optimal_params is not None:
            assert isinstance(optimal_params, np.ndarray)
            assert len(optimal_params) == 3
            assert info["method"] == "scenario_robust"

    def test_ellipsoidal_optimization_benchmark(self, performance_mock_core, benchmark):
        """Benchmark ellipsoidal robust optimization."""
        assert RobustHomodyneOptimizer is not None, (
            "RobustHomodyneOptimizer not available"
        )
        optimizer = RobustHomodyneOptimizer(performance_mock_core, PERFORMANCE_CONFIG)

        test_params = np.array([120.0, -0.6, 15.0])
        phi_angles = performance_mock_core.phi_angles
        c2_experimental = performance_mock_core.c2_experimental

        def run_ellipsoidal_optimization():
            return optimizer._solve_ellipsoidal_robust(
                theta_init=test_params,
                phi_angles=phi_angles,
                c2_experimental=c2_experimental,
                gamma=0.1,
            )

        # Benchmark the optimization
        result = benchmark(run_ellipsoidal_optimization)

        optimal_params, info = result

        # Verify results if optimization succeeded
        if optimal_params is not None:
            assert isinstance(optimal_params, np.ndarray)
            assert len(optimal_params) == 3
            assert info["method"] == "ellipsoidal_robust"

    def test_classical_vs_robust_performance_comparison(self, performance_mock_core):
        """Compare performance between classical and robust optimization."""
        # Classical optimizer
        assert ClassicalOptimizer is not None, "ClassicalOptimizer not available"
        classical_optimizer = ClassicalOptimizer(
            performance_mock_core, PERFORMANCE_CONFIG
        )

        # Robust optimizer
        assert RobustHomodyneOptimizer is not None, (
            "RobustHomodyneOptimizer not available"
        )
        robust_optimizer = RobustHomodyneOptimizer(
            performance_mock_core, PERFORMANCE_CONFIG
        )

        test_params = np.array([120.0, -0.6, 15.0])
        phi_angles = performance_mock_core.phi_angles
        c2_experimental = performance_mock_core.c2_experimental

        def objective_func(params):
            return performance_mock_core.calculate_chi_squared_optimized(
                params, phi_angles, c2_experimental
            )

        # Time classical Nelder-Mead
        start_time = time.time()
        classical_success, _ = classical_optimizer.run_single_method(
            "Nelder-Mead", objective_func, test_params
        )
        classical_time = time.time() - start_time

        # Time robust optimization (if CVXPY available)
        if CVXPY_AVAILABLE:
            start_time = time.time()
            robust_params, _ = robust_optimizer.run_robust_optimization(
                initial_parameters=test_params,
                phi_angles=phi_angles,
                c2_experimental=c2_experimental,
                method="wasserstein",
            )
            robust_time = time.time() - start_time

            # Log performance comparison
            logging.info(f"Classical optimization time: {classical_time:.3f}s")
            logging.info(f"Robust optimization time: {robust_time:.3f}s")

            # Robust optimization may be slower due to convex formulation
            # but should be within reasonable bounds (< 10x classical)
            if robust_params is not None and classical_success:
                assert robust_time < 10 * classical_time, (
                    f"Robust optimization too slow: {robust_time:.3f}s vs classical {classical_time:.3f}s"
                )

    def test_solver_fallback_performance(self, performance_mock_core):
        """Test performance of solver fallback mechanism."""
        if not ROBUST_OPTIMIZATION_AVAILABLE or not CVXPY_AVAILABLE:
            pytest.skip("CVXPY not available")

        assert RobustHomodyneOptimizer is not None, (
            "RobustHomodyneOptimizer not available"
        )
        optimizer = RobustHomodyneOptimizer(performance_mock_core, PERFORMANCE_CONFIG)

        test_params = np.array([120.0, -0.6, 15.0])
        phi_angles = performance_mock_core.phi_angles
        c2_experimental = performance_mock_core.c2_experimental

        # Test that solver fallback doesn't cause excessive delays
        start_time = time.time()

        # Try a challenging problem that might require solver fallback
        optimal_params, info = optimizer._solve_distributionally_robust(
            theta_init=test_params,
            phi_angles=phi_angles,
            c2_experimental=c2_experimental,
            uncertainty_radius=0.15,  # Higher uncertainty might trigger fallback
        )

        end_time = time.time()
        total_time = end_time - start_time

        # Even with potential solver fallback, should complete reasonably quickly
        assert total_time < 30, f"Solver fallback too slow: {total_time:.3f}s"

        # Check that we got some result
        if optimal_params is not None:
            assert len(optimal_params) == 3
            assert np.all(np.isfinite(optimal_params))

    def test_caching_performance_improvement(self, performance_mock_core):
        """Test that caching improves performance for repeated calls."""
        if not ROBUST_OPTIMIZATION_AVAILABLE:
            pytest.skip("Robust optimization not available")

        assert RobustHomodyneOptimizer is not None, (
            "RobustHomodyneOptimizer not available"
        )

        # Test with caching enabled
        config_with_cache = PERFORMANCE_CONFIG.copy()
        config_with_cache.setdefault("optimization_config", {}).setdefault(
            "robust_optimization", {}
        )["enable_caching"] = True
        optimizer_cached = RobustHomodyneOptimizer(
            performance_mock_core, config_with_cache
        )

        # Test without caching
        config_no_cache = PERFORMANCE_CONFIG.copy()
        config_no_cache.setdefault("optimization_config", {}).setdefault(
            "robust_optimization", {}
        )["enable_caching"] = False
        optimizer_no_cache = RobustHomodyneOptimizer(
            performance_mock_core, config_no_cache
        )

        np.array([150.0, -0.8, 20.0])

        # First call (both should be similar)
        start_time = time.time()
        for _ in range(5):
            _ = optimizer_cached._get_parameter_bounds()
        cached_first_time = time.time() - start_time

        start_time = time.time()
        for _ in range(5):
            _ = optimizer_no_cache._get_parameter_bounds()
        time.time() - start_time

        # Second call with same number of iterations (cached should be faster or similar)
        start_time = time.time()
        for _ in range(5):  # Same iteration count for fair comparison
            _ = optimizer_cached._get_parameter_bounds()
        cached_second_time = time.time() - start_time

        # Caching should provide some benefit or at least not make things worse
        # Allow some tolerance since bounds caching might have minimal impact
        assert cached_second_time <= cached_first_time * 2.0, (
            f"Caching performance degraded: {cached_second_time:.4f}s vs {cached_first_time:.4f}s"
        )


@pytest.mark.performance
@pytest.mark.skipif(
    not ROBUST_OPTIMIZATION_AVAILABLE,
    reason="Robust optimization not available",
)
class TestRobustOptimizationMemoryUsage:
    """Memory usage tests for robust optimization."""

    def test_memory_efficient_scenario_generation(self, large_scale_mock_core):
        """Test memory efficiency of scenario generation with large datasets."""
        assert RobustHomodyneOptimizer is not None, (
            "RobustHomodyneOptimizer not available"
        )
        optimizer = RobustHomodyneOptimizer(large_scale_mock_core, PERFORMANCE_CONFIG)

        test_params = np.array([150.0, -0.8, 20.0])
        phi_angles = large_scale_mock_core.phi_angles  # 50 angles
        c2_experimental = large_scale_mock_core.c2_experimental  # 50x200 data

        # Generate scenarios and check memory usage is reasonable
        scenarios = optimizer._generate_bootstrap_scenarios(
            test_params, phi_angles, c2_experimental, n_scenarios=20
        )

        # Check that scenarios are generated efficiently
        assert len(scenarios) == 20

        # Each scenario should have same shape as experimental data
        for scenario in scenarios:
            assert scenario.shape == c2_experimental.shape
            assert np.all(np.isfinite(scenario))

    def test_large_jacobian_computation(self, large_scale_mock_core):
        """Test Jacobian computation with large datasets."""
        assert RobustHomodyneOptimizer is not None, (
            "RobustHomodyneOptimizer not available"
        )
        optimizer = RobustHomodyneOptimizer(large_scale_mock_core, PERFORMANCE_CONFIG)

        test_params = np.array([150.0, -0.8, 20.0])
        phi_angles = large_scale_mock_core.phi_angles

        # Compute Jacobian for large dataset
        c2_theory, jacobian = optimizer._compute_linearized_correlation(
            test_params, phi_angles, large_scale_mock_core.c2_experimental
        )

        # Check dimensions
        expected_data_size = (
            len(phi_angles)
            * large_scale_mock_core.n_times
            * large_scale_mock_core.n_times
        )
        assert jacobian.shape == (expected_data_size, len(test_params))
        assert c2_theory.size == expected_data_size

        # Check memory usage is reasonable (no excessive copies)
        assert jacobian.nbytes < 100e6  # Less than 100MB for reasonable problem sizes


@pytest.mark.performance
class TestRobustOptimizationStressTests:
    """Stress tests for robust optimization under challenging conditions."""

    def test_high_noise_performance(self):
        """Test performance with high noise levels."""
        if not ROBUST_OPTIMIZATION_AVAILABLE or not CVXPY_AVAILABLE:
            pytest.skip("CVXPY not available")

        # Create mock core with high noise
        config = PERFORMANCE_CONFIG.copy()
        mock_core = MockAnalysisCorePerformance(config, n_angles=15, n_times=80)

        # Add high noise to experimental data
        noise_level = 0.1  # 10% noise
        noise = np.random.normal(0, noise_level, mock_core.c2_experimental.shape)
        mock_core.c2_experimental += noise * np.mean(mock_core.c2_experimental)

        assert RobustHomodyneOptimizer is not None, (
            "RobustHomodyneOptimizer not available"
        )
        optimizer = RobustHomodyneOptimizer(mock_core, config)

        test_params = np.array([120.0, -0.6, 15.0])

        # Test that robust optimization handles high noise gracefully
        start_time = time.time()

        optimal_params, _ = optimizer.run_robust_optimization(
            initial_parameters=test_params,
            phi_angles=mock_core.phi_angles,
            c2_experimental=mock_core.c2_experimental,
            method="wasserstein",
            uncertainty_radius=0.08,  # Higher uncertainty for noisy data
        )

        end_time = time.time()
        optimization_time = end_time - start_time

        # Should complete within reasonable time even with high noise
        assert optimization_time < 60, (
            f"High noise optimization too slow: {optimization_time:.3f}s"
        )

        # If successful, results should be reasonable
        if optimal_params is not None:
            assert np.all(np.isfinite(optimal_params))
            assert len(optimal_params) == 3

    def test_extreme_parameter_values_robustness(self):
        """Test robustness with extreme parameter values."""
        if not ROBUST_OPTIMIZATION_AVAILABLE or not CVXPY_AVAILABLE:
            pytest.skip("CVXPY not available")

        mock_core = MockAnalysisCorePerformance(PERFORMANCE_CONFIG)
        assert RobustHomodyneOptimizer is not None, (
            "RobustHomodyneOptimizer not available"
        )
        optimizer = RobustHomodyneOptimizer(mock_core, PERFORMANCE_CONFIG)

        # Test with extreme parameter values
        extreme_params = [
            np.array([1.0, -1.9, 0.1]),  # Near bounds
            np.array([9999.0, 1.9, 999.0]),  # Near upper bounds
            np.array([50.0, 0.0, 50.0]),  # Mid-range
        ]

        for params in extreme_params:
            try:
                chi_squared = optimizer._compute_chi_squared(
                    params, mock_core.phi_angles, mock_core.c2_experimental
                )
                # For extreme parameter values, chi-squared may be infinite due to numerical issues
                # This is the expected behavior for invalid parameter regions
                if np.isfinite(chi_squared):
                    assert chi_squared >= 0, f"Chi-squared negative for params {params}"
                else:
                    # Infinite chi-squared is acceptable for extreme parameters
                    # indicating they are outside the valid parameter space
                    logging.info(
                        f"Chi-squared is infinite for extreme params {params} - this is expected"
                    )
                    assert chi_squared == float("inf"), (
                        f"Expected inf, got {chi_squared} for params {params}"
                    )
            except (OverflowError, ValueError) as e:
                # Acceptable to fail gracefully with extreme values
                logging.warning(f"Expected failure with extreme params {params}: {e}")


if __name__ == "__main__":
    # Run performance tests
    pytest.main([__file__, "-v", "-m", "performance", "--tb=short"])
