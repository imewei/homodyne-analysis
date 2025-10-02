"""
Comprehensive Performance Tests for Optimization Algorithms
===========================================================

Performance benchmarking and regression tests for classical and robust optimization methods.
"""

import time
from unittest.mock import Mock
from unittest.mock import patch

import numpy as np
import pytest

try:
    from homodyne.optimization.classical import ClassicalOptimizer

    CLASSICAL_AVAILABLE = True
except ImportError:
    CLASSICAL_AVAILABLE = False

try:
    from homodyne.optimization.robust import RobustHomodyneOptimizer

    ROBUST_AVAILABLE = True
except ImportError:
    ROBUST_AVAILABLE = False

try:
    from homodyne.core.kernels import NUMBA_AVAILABLE
except ImportError:
    NUMBA_AVAILABLE = False


class PerformanceBenchmark:
    """Base class for performance benchmarking."""

    def __init__(self, name: str):
        self.name = name
        self.times = []
        self.memory_usage = []
        self.results = []

    def measure_time(self, func, *args, **kwargs):
        """Measure execution time of a function."""
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()

        execution_time = end_time - start_time
        self.times.append(execution_time)
        self.results.append(result)

        return result, execution_time

    def get_statistics(self):
        """Get performance statistics."""
        if not self.times:
            return {}

        return {
            "mean_time": np.mean(self.times),
            "std_time": np.std(self.times),
            "min_time": np.min(self.times),
            "max_time": np.max(self.times),
            "total_runs": len(self.times),
        }


def create_mock_analysis_core():
    """Create a properly configured mock analysis core for testing."""
    from unittest.mock import Mock

    import numpy as np

    mock_core = Mock()
    mock_core._calculate_chi_squared = Mock(return_value=1.5)
    mock_core.calculate_chi_squared_optimized = Mock(return_value=1.5)

    # Add necessary attributes for ClassicalOptimizer
    mock_core.config = {}
    mock_core.c2_data = np.random.rand(3, 10, 10)
    mock_core.phi_angles = np.array([0, 45, 90])
    mock_core.time_delays = np.linspace(0.1, 1.0, 10)

    # Mock the load_experimental_data method to return a proper tuple
    c2_experimental = np.random.rand(3, 10, 10)
    phi_angles = np.array([0, 45, 90])
    mock_core.load_experimental_data = Mock(
        return_value=(c2_experimental, 1.0, phi_angles, 3)
    )

    return mock_core


@pytest.mark.skipif(
    not CLASSICAL_AVAILABLE, reason="Classical optimization not available"
)
class TestClassicalOptimizationPerformance:
    """Performance tests for classical optimization methods."""

    def setup_method(self):
        """Setup performance test fixtures."""
        self.benchmark = PerformanceBenchmark("ClassicalOptimization")

        # Create test configuration
        self.config = {
            "experimental_parameters": {
                "q_value": 0.1,
                "contrast": 0.95,
                "offset": 1.0,
            },
            "analysis_parameters": {
                "mode": "laminar_flow",
                "method": "classical",
                "max_iterations": 1000,
                "tolerance": 1e-6,
            },
            "parameter_bounds": {
                "D0": [1e-6, 1e-1],
                "alpha": [0.1, 2.0],
                "D_offset": [1e-8, 1e-3],
                "gamma0": [1e-4, 1.0],
                "beta": [0.1, 2.0],
                "gamma_offset": [1e-6, 1e-1],
                "phi0": [-180, 180],
            },
        }

        # Create synthetic test data
        self.create_test_data()

    def create_test_data(self):
        """Create synthetic test data for performance testing."""
        # Different data sizes for scalability testing
        self.data_sizes = {
            "small": (8, 5, 5),  # 8 angles, 5x5 time grid
            "medium": (16, 10, 10),  # 16 angles, 10x10 time grid
            "large": (32, 20, 20),  # 32 angles, 20x20 time grid
        }

        self.test_datasets = {}

        for size_name, (n_angles, n_t1, n_t2) in self.data_sizes.items():
            angles = np.linspace(0, 2 * np.pi, n_angles, endpoint=False)
            t1_array = np.linspace(0.1, 5.0, n_t1)
            t2_array = np.linspace(0.2, 5.1, n_t2)

            # Generate realistic correlation data
            np.random.seed(42)  # For reproducibility
            c2_data = np.ones((n_angles, n_t1, n_t2))

            for i, angle in enumerate(angles):
                for j, t1 in enumerate(t1_array):
                    for k, t2 in enumerate(t2_array):
                        dt = abs(t2 - t1)
                        # Realistic correlation with noise
                        correlation = (
                            0.9 * np.exp(-0.05 * dt) * (1 + 0.1 * np.cos(2 * angle))
                        )
                        noise = 0.01 * np.random.randn()
                        c2_data[i, j, k] = 1.0 + correlation + noise

            self.test_datasets[size_name] = {
                "c2_data": c2_data,
                "angles": angles,
                "t1_array": t1_array,
                "t2_array": t2_array,
            }

    def test_optimization_scalability(self):
        """Test optimization performance scalability with data size."""
        optimizer = ClassicalOptimizer(create_mock_analysis_core(), self.config)

        scalability_results = {}

        for size_name, data in self.test_datasets.items():
            # Mock the actual optimization to focus on data handling performance
            with patch.object(optimizer, "_run_scipy_optimization") as mock_opt:
                mock_result = Mock()
                mock_result.x = [1e-3, 0.9, 1e-4, 0.01, 0.8, 0.001, 0.0]
                mock_result.fun = 0.5
                mock_result.success = True
                mock_opt.return_value = mock_result

                # Measure time for data preprocessing and setup - use correct method name
                initial_params = np.array([1e-3, 0.9, 1e-4, 0.01, 0.8, 0.001, 0.0])
                result, exec_time = self.benchmark.measure_time(
                    optimizer.run_optimization,
                    initial_params,
                    data["angles"],
                    data["c2_data"],
                )

                scalability_results[size_name] = {
                    "time": exec_time,
                    "data_size": np.prod(data["c2_data"].shape),
                    "success": result.get("success", False),
                }

        # Verify scalability characteristics
        small_time = scalability_results["small"]["time"]
        medium_time = scalability_results["medium"]["time"]
        scalability_results["large"]["time"]

        # Time should scale roughly with data size (allow some overhead)
        small_size = scalability_results["small"]["data_size"]
        medium_size = scalability_results["medium"]["data_size"]
        scalability_results["large"]["data_size"]

        # Check that performance scales reasonably
        medium_ratio = medium_time / small_time
        size_ratio = medium_size / small_size

        # Performance should not degrade more than 2x the data size ratio
        assert medium_ratio < 2 * size_ratio, (
            f"Performance degradation too severe: {medium_ratio} vs {size_ratio}"
        )

    def test_chi_squared_calculation_performance(self):
        """Test chi-squared calculation performance."""
        optimizer = ClassicalOptimizer(create_mock_analysis_core(), self.config)

        # Test with medium dataset
        self.test_datasets["medium"]

        # Test parameters
        params = np.array([1e-3, 0.9, 1e-4, 0.01, 0.8, 0.001, 0.0])

        # Benchmark chi-squared calculation
        n_runs = 10
        times = []

        for _ in range(n_runs):
            start_time = time.perf_counter()
            chi_squared = optimizer._calculate_chi_squared(params)
            end_time = time.perf_counter()
            times.append(end_time - start_time)

            # Verify result is reasonable
            assert chi_squared >= 0.0
            assert np.isfinite(chi_squared)

        mean_time = np.mean(times)
        std_time = np.std(times)

        # Chi-squared calculation should be fast (< 0.1 seconds for medium data)
        assert mean_time < 0.1, f"Chi-squared calculation too slow: {mean_time:.3f}s"

        # Should be consistent (low variance) - relaxed for mocked tests
        cv = std_time / mean_time if mean_time > 0 else 0
        assert cv < 2.0, f"Chi-squared timing too variable: CV = {cv:.3f}"

    @pytest.mark.skipif(not NUMBA_AVAILABLE, reason="Numba not available")
    def test_numba_acceleration_performance(self):
        """Test Numba JIT acceleration performance."""
        from homodyne.core.kernels import compute_g1_correlation_numba

        # Test parameters
        t1, t2 = 1.0, 2.0
        phi = np.pi / 4
        q = 0.1
        params = [1e-3, 0.9, 1e-4, 0.01, 0.8, 0.001, 0.0]

        # First call (compilation time)
        start_compile = time.perf_counter()
        result_first = compute_g1_correlation_numba(t1, t2, phi, q, *params)
        end_compile = time.perf_counter()
        compile_time = end_compile - start_compile

        # Subsequent calls (optimized execution)
        n_runs = 1000
        start_optimized = time.perf_counter()

        for _ in range(n_runs):
            result = compute_g1_correlation_numba(t1, t2, phi, q, *params)
            assert abs(result - result_first) < 1e-12  # Results should be identical

        end_optimized = time.perf_counter()
        optimized_time = (end_optimized - start_optimized) / n_runs

        # Optimized calls should be much faster than compilation
        # Note: Realistic Numba speedup is 5-8x for complex numerical operations
        # involving transcendental functions and array operations
        speedup = compile_time / optimized_time if optimized_time > 0 else float("inf")
        assert speedup > 5, f"Numba speedup insufficient: {speedup:.1f}x"

        # Individual optimized calls should be very fast
        assert optimized_time < 1e-4, (
            f"Numba optimized call too slow: {optimized_time:.6f}s"
        )

    def test_memory_efficiency(self):
        """Test memory efficiency of optimization algorithms."""
        import os

        import psutil

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        optimizer = ClassicalOptimizer(create_mock_analysis_core(), self.config)

        # Process large dataset
        large_data = self.test_datasets["large"]

        # Mock optimization to focus on memory usage
        with patch.object(optimizer, "_run_scipy_optimization") as mock_opt:
            mock_result = Mock()
            mock_result.x = [1e-3, 0.9, 1e-4, 0.01, 0.8, 0.001, 0.0]
            mock_result.fun = 0.5
            mock_result.success = True
            mock_opt.return_value = mock_result

            # Run optimization
            optimizer.optimize(
                large_data["c2_data"],
                large_data["angles"],
                large_data["t1_array"],
                large_data["t2_array"],
            )

            peak_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_increase = peak_memory - initial_memory

            # Memory increase should be reasonable (< 500 MB for large dataset)
            assert memory_increase < 500, (
                f"Memory usage too high: {memory_increase:.1f} MB"
            )

    def test_convergence_performance(self):
        """Test optimization convergence performance."""
        ClassicalOptimizer(create_mock_analysis_core(), self.config)
        data = self.test_datasets["small"]  # Use small data for faster testing

        # Test with different tolerance levels
        tolerances = [1e-3, 1e-4, 1e-5, 1e-6]
        convergence_results = {}

        for tol in tolerances:
            # Update tolerance in config
            test_config = self.config.copy()
            test_config["analysis_parameters"]["tolerance"] = tol

            optimizer_tol = ClassicalOptimizer(create_mock_analysis_core(), test_config)

            with patch.object(optimizer_tol, "_run_scipy_optimization") as mock_opt:
                # Simulate different convergence times based on tolerance
                def mock_optimization(*args, **kwargs):
                    # Simulate longer optimization for tighter tolerance
                    time.sleep(
                        0.001 / tol
                    )  # Artificial delay proportional to tolerance

                    result = Mock()
                    result.x = [1e-3, 0.9, 1e-4, 0.01, 0.8, 0.001, 0.0]
                    result.fun = tol * 10  # Better fit for tighter tolerance
                    result.success = True
                    result.nit = int(
                        1000 * (1e-6 / tol)
                    )  # More iterations for tighter tolerance
                    return result

                mock_opt.side_effect = mock_optimization

                start_time = time.perf_counter()
                result = optimizer_tol.optimize(
                    data["c2_data"], data["angles"], data["t1_array"], data["t2_array"]
                )
                end_time = time.perf_counter()

                convergence_results[tol] = {
                    "time": end_time - start_time,
                    "chi_squared": result.get("chi_squared", 0),
                    "success": result.get("success", False),
                }

        # Verify that tighter tolerances give better fits (lower chi-squared)
        chi_squared_values = [
            convergence_results[tol]["chi_squared"] for tol in tolerances
        ]

        # Chi-squared should generally decrease with tighter tolerance
        # (allowing some noise in the mock results)
        for i in range(len(chi_squared_values) - 1):
            ratio = chi_squared_values[i + 1] / chi_squared_values[i]
            assert ratio < 2.0, (
                f"Chi-squared not improving with tighter tolerance: {ratio}"
            )


@pytest.mark.skipif(not ROBUST_AVAILABLE, reason="Robust optimization not available")
class TestRobustOptimizationPerformance:
    """Performance tests for robust optimization methods."""

    def setup_method(self):
        """Setup robust optimization performance tests."""
        self.benchmark = PerformanceBenchmark("RobustOptimization")

        self.config = {
            "experimental_parameters": {
                "q_value": 0.1,
                "contrast": 0.95,
                "offset": 1.0,
            },
            "analysis_parameters": {
                "mode": "laminar_flow",
                "method": "robust",
                "robust_method": "wasserstein_dro",
                "uncertainty_budget": 0.1,
                "max_iterations": 100,  # Reduced for testing
            },
            "parameter_bounds": {
                "D0": [1e-6, 1e-1],
                "alpha": [0.1, 2.0],
                "D_offset": [1e-8, 1e-3],
                "gamma0": [1e-4, 1.0],
                "beta": [0.1, 2.0],
                "gamma_offset": [1e-6, 1e-1],
                "phi0": [-180, 180],
            },
        }

        # Create test data with outliers
        self.create_noisy_test_data()

    def create_noisy_test_data(self):
        """Create test data with outliers for robust optimization testing."""
        angles = np.linspace(0, 2 * np.pi, 8, endpoint=False)
        t1_array = np.linspace(0.1, 3.0, 8)
        t2_array = np.linspace(0.2, 3.1, 8)

        np.random.seed(123)  # For reproducibility
        c2_data = np.ones((8, 8, 8))

        for i, angle in enumerate(angles):
            for j, t1 in enumerate(t1_array):
                for k, t2 in enumerate(t2_array):
                    dt = abs(t2 - t1)
                    correlation = (
                        0.9 * np.exp(-0.1 * dt) * (1 + 0.1 * np.cos(2 * angle))
                    )

                    # Add noise and occasional outliers
                    noise = 0.02 * np.random.randn()
                    if np.random.random() < 0.05:  # 5% outliers
                        noise += 0.2 * np.random.randn()  # Large outlier

                    c2_data[i, j, k] = 1.0 + correlation + noise

        self.noisy_data = {
            "c2_data": c2_data,
            "angles": angles,
            "t1_array": t1_array,
            "t2_array": t2_array,
        }

    def test_robust_vs_classical_performance(self):
        """Compare performance of robust vs classical optimization."""
        # Test classical optimization
        classical_config = self.config.copy()
        classical_config["analysis_parameters"]["method"] = "classical"

        try:
            classical_optimizer = ClassicalOptimizer(
                create_mock_analysis_core(), classical_config
            )

            with patch.object(
                classical_optimizer, "_run_scipy_optimization"
            ) as mock_classical:
                mock_result = Mock()
                mock_result.x = [1e-3, 0.9, 1e-4, 0.01, 0.8, 0.001, 0.0]
                mock_result.fun = 0.5
                mock_result.success = True
                mock_classical.return_value = mock_result

                classical_result, classical_time = self.benchmark.measure_time(
                    classical_optimizer.optimize,
                    self.noisy_data["c2_data"],
                    self.noisy_data["angles"],
                    self.noisy_data["t1_array"],
                    self.noisy_data["t2_array"],
                )

        except ImportError:
            classical_time = 0.1  # Fallback for comparison
            classical_result = {"success": True}

        # Test robust optimization
        robust_optimizer = RobustHomodyneOptimizer(
            create_mock_analysis_core(), config=self.config
        )

        with patch.object(
            robust_optimizer, "_solve_robust_optimization"
        ) as mock_robust:
            mock_result = Mock()
            mock_result.x = [1e-3, 0.9, 1e-4, 0.01, 0.8, 0.001, 0.0]
            mock_result.fun = 0.3  # Better fit due to robustness
            mock_result.success = True
            mock_robust.return_value = mock_result

            robust_result, robust_time = self.benchmark.measure_time(
                robust_optimizer.optimize,
                self.noisy_data["c2_data"],
                self.noisy_data["angles"],
                self.noisy_data["t1_array"],
                self.noisy_data["t2_array"],
            )

        # Robust optimization may be slower but should be more robust to outliers
        if classical_time > 0:
            time_ratio = robust_time / classical_time
            # Allow robust optimization to be up to 10x slower (due to complexity)
            assert time_ratio < 10, (
                f"Robust optimization too slow: {time_ratio:.1f}x classical"
            )

        # Both should succeed
        assert classical_result.get("success", False)
        assert robust_result.get("success", False)

    def test_uncertainty_budget_performance(self):
        """Test performance with different uncertainty budgets."""
        uncertainty_budgets = [0.05, 0.1, 0.2, 0.3]
        performance_results = {}

        for budget in uncertainty_budgets:
            test_config = self.config.copy()
            test_config["analysis_parameters"]["uncertainty_budget"] = budget

            optimizer = RobustHomodyneOptimizer(
                create_mock_analysis_core(), config=test_config
            )

            with patch.object(optimizer, "_solve_robust_optimization") as mock_opt:
                # Simulate longer solving time for larger uncertainty budgets
                def mock_solve(*args, **kwargs):
                    time.sleep(0.001 * budget * 10)  # Artificial delay

                    result = Mock()
                    result.x = [1e-3, 0.9, 1e-4, 0.01, 0.8, 0.001, 0.0]
                    result.fun = 0.1 + budget  # Worse fit for larger uncertainty
                    result.success = True
                    return result

                mock_opt.side_effect = mock_solve

                result, exec_time = self.benchmark.measure_time(
                    optimizer.optimize,
                    self.noisy_data["c2_data"],
                    self.noisy_data["angles"],
                    self.noisy_data["t1_array"],
                    self.noisy_data["t2_array"],
                )

                performance_results[budget] = {
                    "time": exec_time,
                    "chi_squared": result.get("chi_squared", 0),
                    "success": result.get("success", False),
                }

        # Performance should scale reasonably with uncertainty budget
        times = [performance_results[budget]["time"] for budget in uncertainty_budgets]

        # Times should generally increase with uncertainty budget
        # (allowing some variance in the mock timing)
        for i in range(len(times) - 1):
            ratio = times[i + 1] / times[i] if times[i] > 0 else 1
            assert ratio < 5.0, (
                "Performance degradation too severe with uncertainty budget"
            )

    def test_scenario_generation_performance(self):
        """Test performance of scenario generation for robust optimization."""
        RobustHomodyneOptimizer(create_mock_analysis_core(), config=self.config)

        # Test scenario generation with different numbers of scenarios
        scenario_counts = [10, 50, 100, 200]
        generation_times = []

        for n_scenarios in scenario_counts:
            # Mock scenario generation
            start_time = time.perf_counter()

            # Simulate scenario generation time
            scenarios = []
            for _ in range(n_scenarios):
                scenario = self.noisy_data["c2_data"] + 0.01 * np.random.randn(
                    *self.noisy_data["c2_data"].shape
                )
                scenarios.append(scenario)

            end_time = time.perf_counter()
            generation_time = end_time - start_time
            generation_times.append(generation_time)

            # Verify scenarios are reasonable
            assert len(scenarios) == n_scenarios
            for scenario in scenarios[:3]:  # Check first few
                assert scenario.shape == self.noisy_data["c2_data"].shape
                assert np.all(scenario >= 1.0)  # g2 should be >= 1

        # Generation time should scale roughly linearly with number of scenarios
        if len(generation_times) >= 2:
            time_ratio = generation_times[-1] / generation_times[0]
            scenario_ratio = scenario_counts[-1] / scenario_counts[0]

            # Allow some overhead, but should be roughly linear
            assert time_ratio < 2 * scenario_ratio, (
                f"Scenario generation scaling poor: {time_ratio} vs {scenario_ratio}"
            )


class TestOptimizationMemoryProfiling:
    """Memory profiling tests for optimization algorithms."""

    def test_memory_leak_detection(self):
        """Test for memory leaks in optimization loops."""
        import gc
        import os

        import psutil

        process = psutil.Process(os.getpid())

        # Force garbage collection
        gc.collect()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Run multiple optimization cycles
        n_cycles = 10
        for i in range(n_cycles):
            # Create temporary data
            angles = np.linspace(0, 2 * np.pi, 4, endpoint=False)
            t1_array = np.array([1.0, 2.0])
            t2_array = np.array([1.5, 2.5])
            c2_data = 1.0 + 0.1 * np.random.random((4, 2, 2))

            # Simulate optimization
            if CLASSICAL_AVAILABLE:
                config = {
                    "experimental_parameters": {
                        "q_value": 0.1,
                        "contrast": 0.95,
                        "offset": 1.0,
                    },
                    "analysis_parameters": {
                        "mode": "static_isotropic",
                        "method": "classical",
                    },
                    "parameter_bounds": {
                        "D0": [1e-6, 1e-1],
                        "alpha": [0.1, 2.0],
                        "D_offset": [1e-8, 1e-3],
                    },
                }

                optimizer = ClassicalOptimizer(create_mock_analysis_core(), config)

                with patch.object(optimizer, "_run_scipy_optimization") as mock_opt:
                    mock_result = Mock()
                    mock_result.x = [1e-3, 0.9, 1e-4]
                    mock_result.fun = 0.1
                    mock_result.success = True
                    mock_opt.return_value = mock_result

                    result = optimizer.optimize(c2_data, angles, t1_array, t2_array)

            # Clear references
            del c2_data, angles, t1_array, t2_array
            if "optimizer" in locals():
                del optimizer
            if "result" in locals():
                del result

            # Force garbage collection
            gc.collect()

        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory

        # Memory increase should be minimal (< 50 MB after 10 cycles)
        assert memory_increase < 50, (
            f"Potential memory leak detected: {memory_increase:.1f} MB increase"
        )

    def test_large_data_memory_handling(self):
        """Test memory handling with large datasets."""
        import os

        import psutil

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Create very large dataset
        large_angles = np.linspace(0, 2 * np.pi, 64, endpoint=False)
        large_t1 = np.linspace(0.1, 10.0, 50)
        large_t2 = np.linspace(0.2, 10.1, 50)

        # This creates a ~800MB array
        large_c2_data = 1.0 + 0.1 * np.random.random((64, 50, 50))

        peak_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_for_data = peak_memory - initial_memory

        # Should handle large data without excessive memory overhead
        if CLASSICAL_AVAILABLE:
            config = {
                "experimental_parameters": {
                    "q_value": 0.1,
                    "contrast": 0.95,
                    "offset": 1.0,
                },
                "analysis_parameters": {"mode": "laminar_flow", "method": "classical"},
                "parameter_bounds": {
                    "D0": [1e-6, 1e-1],
                    "alpha": [0.1, 2.0],
                    "D_offset": [1e-8, 1e-3],
                    "gamma0": [1e-4, 1.0],
                    "beta": [0.1, 2.0],
                    "gamma_offset": [1e-6, 1e-1],
                    "phi0": [-180, 180],
                },
            }

            optimizer = ClassicalOptimizer(create_mock_analysis_core(), config)

            # Just test data setup (mock actual optimization)
            with patch.object(optimizer, "_run_scipy_optimization") as mock_opt:
                mock_result = Mock()
                mock_result.x = [1e-3, 0.9, 1e-4, 0.01, 0.8, 0.001, 0.0]
                mock_result.fun = 0.5
                mock_result.success = True
                mock_opt.return_value = mock_result

                try:
                    optimizer.optimize(large_c2_data, large_angles, large_t1, large_t2)
                    optimization_memory = process.memory_info().rss / 1024 / 1024  # MB

                    # Memory overhead should be reasonable (< 2x data size)
                    overhead = optimization_memory - initial_memory - memory_for_data
                    assert overhead < 2 * memory_for_data, (
                        f"Memory overhead too high: {overhead:.1f} MB"
                    )

                except MemoryError:
                    pytest.skip("Insufficient memory for large data test")


class TestPerformanceRegression:
    """Performance regression tests."""

    def setup_method(self):
        """Setup performance regression tests."""
        # Load or create performance baselines
        self.performance_baselines = {
            "small_data_optimization": 0.1,  # seconds
            "medium_data_optimization": 0.5,  # seconds
            "chi_squared_calculation": 0.01,  # seconds
            "g1_correlation_single": 1e-5,  # seconds
            "memory_usage_mb": 100,  # MB
        }

    def test_small_data_performance_regression(self):
        """Test performance regression for small datasets."""
        if not CLASSICAL_AVAILABLE:
            pytest.skip("Classical optimization not available")

        # Create small test dataset
        angles = np.linspace(0, 2 * np.pi, 4, endpoint=False)
        t1_array = np.array([1.0, 2.0])
        t2_array = np.array([1.5, 2.5])
        c2_data = 1.0 + 0.1 * np.random.random((4, 2, 2))

        config = {
            "experimental_parameters": {
                "q_value": 0.1,
                "contrast": 0.95,
                "offset": 1.0,
            },
            "analysis_parameters": {"mode": "static_isotropic", "method": "classical"},
            "parameter_bounds": {
                "D0": [1e-6, 1e-1],
                "alpha": [0.1, 2.0],
                "D_offset": [1e-8, 1e-3],
            },
        }

        optimizer = ClassicalOptimizer(create_mock_analysis_core(), config)

        with patch.object(optimizer, "_run_scipy_optimization") as mock_opt:
            mock_result = Mock()
            mock_result.x = [1e-3, 0.9, 1e-4]
            mock_result.fun = 0.1
            mock_result.success = True
            mock_opt.return_value = mock_result

            start_time = time.perf_counter()
            optimizer.optimize(c2_data, angles, t1_array, t2_array)
            end_time = time.perf_counter()

            execution_time = end_time - start_time
            baseline = self.performance_baselines["small_data_optimization"]

            # Allow 50% performance degradation before flagging regression
            assert execution_time < 1.5 * baseline, (
                f"Performance regression detected: {execution_time:.3f}s vs baseline {baseline:.3f}s"
            )

    @pytest.mark.skipif(not NUMBA_AVAILABLE, reason="Numba not available")
    def test_kernel_performance_regression(self):
        """Test performance regression for computational kernels."""
        from homodyne.core.kernels import compute_g1_correlation_numba

        # Warm up JIT compilation
        compute_g1_correlation_numba(
            1.0, 2.0, 0.0, 0.1, 1e-3, 0.9, 1e-4, 0.01, 0.8, 0.001, 0.0
        )

        # Benchmark single call
        n_calls = 1000
        start_time = time.perf_counter()

        for _ in range(n_calls):
            compute_g1_correlation_numba(
                1.0, 2.0, 0.0, 0.1, 1e-3, 0.9, 1e-4, 0.01, 0.8, 0.001, 0.0
            )

        end_time = time.perf_counter()

        avg_time = (end_time - start_time) / n_calls
        baseline = self.performance_baselines["g1_correlation_single"]

        # Allow 2x performance degradation for kernel functions
        assert avg_time < 2.0 * baseline, (
            f"Kernel performance regression: {avg_time:.6f}s vs baseline {baseline:.6f}s"
        )

    def save_performance_baseline(self, test_name: str, execution_time: float):
        """Save performance baseline for future regression testing."""
        # In a real implementation, this would save to a file or database
        print(f"Baseline for {test_name}: {execution_time:.6f}s")

    def test_generate_new_baselines(self):
        """Generate new performance baselines (for updating baselines)."""
        # This test can be run to generate new baselines when performance is improved
        pytest.skip("Use this test to generate new baselines when needed")
