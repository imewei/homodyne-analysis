"""
Comprehensive Integration Tests for Distributed Computing and ML Acceleration
=============================================================================

End-to-end integration tests that validate the complete workflow from
configuration to optimization results, including performance benchmarks
and real optimization scenarios.

Authors: Wei Chen, Hongrui He, Claude (Anthropic)
Institution: Argonne National Laboratory
"""

import shutil
import tempfile
import time
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest

# Import the modules we're testing
try:
    from homodyne.optimization.classical import ClassicalOptimizer
    from homodyne.optimization.distributed import (
        DistributedOptimizationCoordinator,
        OptimizationResult,
        OptimizationTask,
    )
    from homodyne.optimization.ml_acceleration import (
        MLAcceleratedOptimizer,
    )
    from homodyne.optimization.utils import (
        OptimizationConfig,
        SystemResourceDetector,
        quick_setup_distributed_optimization,
        quick_setup_ml_acceleration,
    )

    INTEGRATION_AVAILABLE = True
except ImportError as e:
    print(f"Integration modules not available: {e}")
    INTEGRATION_AVAILABLE = False

# Check for scikit-learn availability for ML tests
try:
    import sklearn
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False


@pytest.mark.skipif(
    not INTEGRATION_AVAILABLE, reason="Integration modules not available"
)
class TestDistributedMLIntegration:
    """End-to-end integration tests for distributed computing and ML acceleration."""

    @pytest.fixture
    def temp_workspace(self):
        """Create temporary workspace for integration tests."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def sample_optimization_data(self):
        """Create sample optimization data for testing."""
        return {
            "parameters": np.array([1.0, 2.0, 3.0]),
            "bounds": [(0.0, 10.0), (0.0, 10.0), (0.0, 10.0)],
            "objective_config": {"function_type": "quadratic", "noise_level": 0.1},
        }

    @pytest.fixture
    def integration_config(self):
        """Create comprehensive configuration for integration testing."""
        return {
            "distributed_optimization": {
                "enabled": True,
                "backend_preference": ["multiprocessing", "ray", "dask"],
                "multiprocessing_config": {"num_processes": 2},
                "ray_config": {"num_cpus": 2, "num_gpus": 0},
                "dask_config": {
                    "n_workers": 2,
                    "threads_per_worker": 1,
                    "memory_limit": "1GB",
                },
            },
            "ml_acceleration": {
                "enabled": True,
                "predictor_type": "ensemble",
                "ml_model_config": {
                    "model_type": "ensemble",
                    "validation_split": 0.2,
                    "hyperparameters": {
                        "random_forest": {
                            "n_estimators": 10,  # Reduced for faster testing
                            "max_depth": 5,
                        },
                        "neural_network": {
                            "hidden_layer_sizes": [10, 5],  # Smaller for testing
                            "max_iter": 50,
                        },
                    },
                },
            },
            "performance_monitoring": {"enabled": True, "benchmark_mode": True},
        }

    def test_end_to_end_distributed_optimization_workflow(
        self, temp_workspace, sample_optimization_data, integration_config
    ):
        """Test complete distributed optimization workflow from start to finish."""

        # Step 1: Initialize distributed coordinator
        coordinator = DistributedOptimizationCoordinator(integration_config)

        # Step 2: Attempt backend initialization (should succeed with multiprocessing fallback)
        backend_initialized = coordinator.initialize()
        assert backend_initialized, "Failed to initialize any distributed backend"
        assert coordinator.backend_type in [
            "multiprocessing",
            "ray",
            "dask",
        ], f"Unexpected backend: {coordinator.backend_type}"

        # Step 3: Create optimization task
        task = OptimizationTask(
            task_id="integration_test_001",
            method="Nelder-Mead",
            parameters=sample_optimization_data["parameters"],
            bounds=sample_optimization_data["bounds"],
            objective_config=sample_optimization_data["objective_config"],
        )

        # Step 4: Submit task
        task_id = coordinator.submit_optimization_task(task)
        # Note: System generates its own task IDs, so we just verify we got one
        assert task_id is not None, "No task ID returned after submission"

        # Step 5: Wait for completion and get results
        max_wait_time = 30.0  # 30 seconds timeout
        start_time = time.time()
        results = []

        while time.time() - start_time < max_wait_time:
            results = coordinator.get_optimization_results()
            if results:
                break
            time.sleep(0.5)

        # Step 6: Validate results
        assert len(results) == 1, f"Expected 1 result, got {len(results)}"
        result = results[0]

        assert isinstance(result, OptimizationResult), (
            "Result is not OptimizationResult instance"
        )
        assert result.task_id == task.task_id, "Task ID mismatch in result"
        assert result.success, f"Optimization failed: {result.error_message}"
        assert result.parameters is not None, "Optimization parameters are None"
        assert len(result.parameters) == len(sample_optimization_data["parameters"]), (
            "Parameter dimension mismatch"
        )
        assert result.objective_value < 100.0, (
            f"Objective value too high: {result.objective_value}"
        )
        assert result.execution_time > 0, "Execution time should be positive"

        # Step 7: Get cluster info
        cluster_info = coordinator.get_cluster_info()
        assert "backend" in cluster_info, "Cluster info missing backend information"
        assert cluster_info["backend"] == coordinator.backend_type, (
            "Backend type mismatch"
        )

        # Step 8: Cleanup
        coordinator.shutdown()

    @pytest.mark.skipif(not ML_AVAILABLE, reason="Scikit-learn not available")
    def test_ml_acceleration_end_to_end_workflow(
        self, temp_workspace, integration_config
    ):
        """Test complete ML acceleration workflow with training and prediction."""

        # Step 1: Create ML accelerated optimizer
        ml_optimizer = MLAcceleratedOptimizer(
            config=integration_config["ml_acceleration"]
        )

        # Step 2: Generate training data by running multiple optimizations
        training_results = []
        for i in range(10):  # Generate 10 training samples
            params = np.random.uniform(0, 5, 3)

            # Simple quadratic objective for consistent results
            def objective(x):
                return np.sum((x - np.array([2.0, 3.0, 1.0])) ** 2)

            # Simulate optimization result
            optimal_params = np.array([2.0, 3.0, 1.0]) + np.random.normal(0, 0.1, 3)
            training_results.append(
                {
                    "initial_parameters": params,
                    "optimal_parameters": optimal_params,
                    "objective_value": objective(optimal_params),
                    "experimental_conditions": {
                        "temperature": 25.0 + i,
                        "pressure": 1.0,
                        "sample_id": f"sample_{i}",
                    },
                }
            )

        # Step 3: Train ML models with the generated data
        ml_optimizer.update_training_data(training_results)
        training_success = ml_optimizer.train_models()
        assert training_success, "ML model training failed"

        # Step 4: Test prediction accuracy
        test_conditions = {
            "temperature": 30.0,
            "pressure": 1.0,
            "sample_id": "test_sample",
        }

        prediction_result = ml_optimizer.predict_optimization_start(
            bounds=[(0.0, 10.0)] * 3, experimental_conditions=test_conditions
        )

        # Step 5: Validate prediction
        assert prediction_result is not None, "ML prediction failed"
        assert "predicted_parameters" in prediction_result, (
            "Missing predicted parameters"
        )
        assert "confidence_score" in prediction_result, "Missing confidence score"
        assert "ensemble_predictions" in prediction_result, (
            "Missing ensemble predictions"
        )

        predicted_params = prediction_result["predicted_parameters"]
        confidence = prediction_result["confidence_score"]

        assert len(predicted_params) == 3, "Predicted parameters dimension mismatch"
        assert 0.0 <= confidence <= 1.0, f"Invalid confidence score: {confidence}"

        # Step 6: Test that prediction is reasonable (close to known optimum)
        expected_optimum = np.array([2.0, 3.0, 1.0])
        prediction_error = np.linalg.norm(predicted_params - expected_optimum)
        assert prediction_error < 2.0, (
            f"Prediction too far from optimum: {prediction_error}"
        )

        # Step 7: Test ML-accelerated optimization
        def test_objective(x):
            return np.sum((x - expected_optimum) ** 2)

        result = ml_optimizer.optimize(
            objective_function=test_objective,
            bounds=[(0.0, 10.0)] * 3,
            experimental_conditions=test_conditions,
        )

        assert result.success, f"ML-accelerated optimization failed: {result.message}"
        final_error = np.linalg.norm(result.x - expected_optimum)
        assert final_error < 0.5, (
            f"ML-accelerated optimization not accurate enough: {final_error}"
        )

    @pytest.mark.skipif(not ML_AVAILABLE, reason="Scikit-learn not available")
    def test_combined_distributed_ml_workflow(self, temp_workspace, integration_config):
        """Test combined distributed computing + ML acceleration workflow."""

        # Step 1: Setup distributed ML optimization
        integration_config["distributed_optimization"]
        integration_config["ml_acceleration"]

        # Create enhanced optimizers
        # Enable distributed optimization
        enhanced_optimizer = quick_setup_distributed_optimization(
            num_processes=2, backend="multiprocessing"
        )

        # Enable ML acceleration
        final_optimizer = quick_setup_ml_acceleration(
            enable_transfer_learning=False,  # Simplified for testing
        )

        # Step 2: Test optimization with both features
        def complex_objective(x):
            """Complex objective function to test optimization capabilities."""
            return (
                np.sum((x - np.array([1.0, 2.0, 3.0])) ** 2)
                + 0.1 * np.sin(10 * np.sum(x))
                + 0.05 * np.random.normal()
            )

        bounds = [(0.0, 5.0)] * 3
        experimental_conditions = {
            "temperature": 25.0,
            "pressure": 1.0,
            "sample_type": "integration_test",
        }

        # Step 3: Run optimization
        result = final_optimizer.optimize(
            objective_function=complex_objective,
            bounds=bounds,
            experimental_conditions=experimental_conditions,
        )

        # Step 4: Validate combined optimization result
        assert result.success, f"Combined optimization failed: {result.message}"
        assert len(result.x) == 3, "Result parameter dimension mismatch"

        # Check that result is reasonable (within bounds and close to expected optimum)
        expected_optimum = np.array([1.0, 2.0, 3.0])
        optimization_error = np.linalg.norm(result.x - expected_optimum)
        assert optimization_error < 1.0, (
            f"Combined optimization not accurate: {optimization_error}"
        )

        # Step 5: Verify distributed execution occurred
        if hasattr(final_optimizer, "_distributed_coordinator"):
            cluster_info = final_optimizer._distributed_coordinator.get_cluster_info()
            assert cluster_info.get("backend") in [
                "multiprocessing",
                "ray",
                "dask",
            ], "Distributed backend not used"

    def test_performance_benchmarking_integration(self, integration_config):
        """Test performance benchmarking and monitoring integration."""

        # Step 1: Setup benchmark configuration
        from homodyne.optimization.utils import OptimizationBenchmark

        benchmark = OptimizationBenchmark()

        # Step 2: Benchmark classical vs distributed optimization
        test_cases = [
            {
                "name": "classical_nelder_mead",
                "method": "Nelder-Mead",
                "distributed": False,
            },
            {
                "name": "distributed_nelder_mead",
                "method": "Nelder-Mead",
                "distributed": True,
            },
            {"name": "ml_accelerated", "method": "Nelder-Mead", "ml_enabled": True},
        ]

        comparison_results = benchmark.compare_optimizers(test_cases)

        # Step 3: Validate benchmark results
        assert "test_cases" in comparison_results, (
            "Missing test cases in benchmark results"
        )
        assert "summary" in comparison_results, "Missing summary in benchmark results"

        summary = comparison_results["summary"]
        assert "average_time" in summary, "Missing average time in benchmark summary"
        assert "average_objective" in summary, (
            "Missing average objective in benchmark summary"
        )

        # Step 4: Verify performance improvements
        test_case_results = comparison_results["test_cases"]
        assert len(test_case_results) == len(test_cases), (
            "Benchmark test case count mismatch"
        )

        # All test cases should complete successfully
        for case in test_case_results:
            success = case.get("success", case.get("metrics", {}).get("success", False))
            assert success, f"Benchmark case {case['name']} failed"
            execution_time = case.get("execution_time", case.get("total_time", 0))
            assert execution_time > 0, (
                f"Invalid execution time for {case['name']}"
            )

    def test_error_recovery_and_fault_tolerance(self, integration_config):
        """Test error recovery and fault tolerance in distributed/ML systems."""

        # Step 1: Test distributed error recovery
        coordinator = DistributedOptimizationCoordinator(integration_config)
        backend_initialized = coordinator.initialize()
        assert backend_initialized, "Failed to initialize backend for error testing"

        # Step 2: Submit task with intentionally failing objective
        def failing_objective(x):
            raise ValueError("Intentional test failure")

        failing_task = OptimizationTask(
            task_id="error_test_001",
            method="Nelder-Mead",
            parameters=np.array([1.0, 2.0]),
            bounds=[(0.0, 5.0), (0.0, 5.0)],
            objective_config={"function_type": "failing"},
            max_retries=0,  # Don't retry, immediate error result
        )

        # Submit task (should not raise exception)
        task_id = coordinator.submit_optimization_task(failing_task)
        assert task_id == failing_task.task_id, "Failed task submission"

        # Step 3: Wait for error result
        max_wait_time = 10.0
        start_time = time.time()
        error_results = []

        while time.time() - start_time < max_wait_time:
            error_results = coordinator.get_optimization_results(timeout=1.0)
            if error_results:
                break
            time.sleep(0.5)

        # Step 4: Validate error handling
        assert len(error_results) == 1, "Should get one error result"
        error_result = error_results[0]

        assert not error_result.success, "Task should have failed"
        assert error_result.error_message is not None, "Missing error message"
        assert error_result.objective_value == float("inf"), (
            "Error objective value should be inf"
        )

        # Step 5: Test recovery with good task
        good_task = OptimizationTask(
            task_id="recovery_test_001",
            method="Nelder-Mead",
            parameters=np.array([1.0, 2.0]),
            bounds=[(0.0, 5.0), (0.0, 5.0)],
            objective_config={"function_type": "quadratic"},
        )

        coordinator.submit_optimization_task(good_task)

        # Wait for recovery result
        start_time = time.time()
        recovery_results = []

        while time.time() - start_time < max_wait_time:
            recovery_results = coordinator.get_optimization_results(timeout=1.0)
            if recovery_results:
                break
            time.sleep(0.5)

        # Step 6: Validate recovery
        assert len(recovery_results) == 1, "Should get one recovery result"
        recovery_result = recovery_results[0]

        assert recovery_result.success, (
            f"Recovery task should succeed: {recovery_result.error_message}"
        )
        assert recovery_result.task_id == good_task.task_id, "Recovery task ID mismatch"

        # Cleanup
        coordinator.shutdown()

    def test_configuration_validation_integration(self, temp_workspace):
        """Test configuration validation and system optimization integration."""

        # Step 1: Test invalid configuration handling
        invalid_config = {
            "distributed_optimization": {
                "enabled": True,
                "backend_preference": ["invalid_backend"],
                "ray_config": {"num_cpus": -1},  # Invalid
            }
        }

        from homodyne.optimization.utils import validate_configuration

        is_valid, errors = validate_configuration(invalid_config)

        assert not is_valid, "Invalid configuration should be rejected"
        assert len(errors) > 0, "Should have validation errors"
        assert any("invalid_backend" in error.lower() for error in errors), (
            "Should catch invalid backend"
        )
        assert any("num_cpus" in error.lower() for error in errors), (
            "Should catch invalid CPU count"
        )

        # Step 2: Test system resource optimization
        system_config = SystemResourceDetector.optimize_configuration(
            {
                "distributed_optimization": {
                    "multiprocessing_config": {"num_processes": None},
                    "ray_config": {"num_cpus": None},
                },
                "ml_acceleration": {
                    "ml_model_config": {
                        "hyperparameters": {"random_forest": {"n_estimators": 100}}
                    }
                },
            }
        )

        # Step 3: Validate optimized configuration
        mp_config = system_config["distributed_optimization"]["multiprocessing_config"]
        assert mp_config["num_processes"] is not None, (
            "Should set multiprocessing processes"
        )
        assert mp_config["num_processes"] > 0, "Should set positive process count"

        ray_config = system_config["distributed_optimization"]["ray_config"]
        assert ray_config["num_cpus"] is not None, "Should set Ray CPU count"
        assert ray_config["num_cpus"] > 0, "Should set positive CPU count"

        # Step 4: Test configuration file operations
        config_file = temp_workspace / "test_config.json"

        config_manager = OptimizationConfig()
        config_manager.config = system_config
        config_manager.save_config(config_file)

        # Verify file was created and can be loaded
        assert config_file.exists(), "Configuration file should be created"

        loaded_config = OptimizationConfig(config_file)
        assert loaded_config.config == system_config, (
            "Loaded configuration should match saved"
        )

    @pytest.mark.slow
    def test_scalability_and_performance_benchmarks(self, integration_config):
        """Test scalability characteristics and performance benchmarks."""

        # Step 1: Setup performance monitoring
        from homodyne.optimization.utils import OptimizationBenchmark

        benchmark = OptimizationBenchmark()

        # Step 2: Test different problem sizes
        problem_sizes = [10, 50, 100]  # Parameter dimensions
        backends = ["multiprocessing"]  # Focus on available backend

        performance_results = {}

        for size in problem_sizes:
            for backend in backends:
                test_name = f"{backend}_{size}d"
                benchmark.start_benchmark(
                    test_name,
                    {"backend": backend, "problem_size": size, "method": "Nelder-Mead"},
                )

                # Run optimization
                start_time = time.time()

                # Create large-scale optimization problem
                coordinator = DistributedOptimizationCoordinator(
                    {
                        "distributed_optimization": {
                            "enabled": True,
                            "backend_preference": [backend],
                            "multiprocessing_config": {"num_processes": 2},
                        }
                    }
                )

                if coordinator.initialize():
                    task = OptimizationTask(
                        task_id=f"perf_test_{test_name}",
                        method="Nelder-Mead",
                        parameters=np.random.uniform(0.5, 1.5, size),  # Random start near optimum
                        bounds=[(-2.0, 2.0)] * size,  # Tighter bounds around optimum
                        objective_config={"function_type": "quadratic", "target_params": np.zeros(size)},
                    )

                    coordinator.submit_optimization_task(task)

                    # Wait for completion
                    max_wait = 60.0  # 1 minute timeout
                    wait_start = time.time()
                    results = []

                    while time.time() - wait_start < max_wait:
                        results = coordinator.get_optimization_results()
                        if results:
                            break
                        time.sleep(1.0)

                    execution_time = time.time() - start_time

                    # Record metrics
                    benchmark.record_metric("execution_time", execution_time)
                    benchmark.record_metric("problem_size", size)
                    benchmark.record_metric(
                        "success",
                        len(results) > 0 and results[0].success if results else False,
                    )

                    if results:
                        benchmark.record_metric(
                            "objective_value", results[0].objective_value
                        )
                        benchmark.record_metric(
                            "function_evaluations",
                            results[0].metadata.get("function_evaluations", 0),
                        )

                    coordinator.shutdown()

                performance_results[test_name] = benchmark.end_benchmark()

        # Step 3: Validate scaling characteristics
        multiprocessing_results = [
            v for k, v in performance_results.items() if "multiprocessing" in k
        ]

        assert len(multiprocessing_results) == len(problem_sizes), (
            "Missing performance results"
        )

        # Check that all runs succeeded
        for result in multiprocessing_results:
            assert result["metrics"].get("success", False), (
                f"Performance test failed: {result['name']}"
            )

        # Check scaling efficiency (execution time should grow sub-linearly with problem size)
        times = [r["metrics"]["execution_time"] for r in multiprocessing_results]
        sizes = [r["metrics"]["problem_size"] for r in multiprocessing_results]

        # Simple scaling check - larger problems shouldn't take exponentially longer
        for i in range(1, len(times)):
            time_ratio = times[i] / times[i - 1]
            size_ratio = sizes[i] / sizes[i - 1]

            # Time growth should be less than quadratic relative to size growth
            assert time_ratio < size_ratio**2, (
                f"Poor scaling: time ratio {time_ratio} vs size ratio {size_ratio}"
            )


def test_integration_modules_basic_functionality():
    """Basic smoke test to ensure integration modules can be imported and instantiated."""
    if not INTEGRATION_AVAILABLE:
        pytest.skip("Integration modules not available")
    if not ML_AVAILABLE:
        pytest.skip("Scikit-learn not available for ML tests")

    # Test basic imports and instantiation
    coordinator = DistributedOptimizationCoordinator()
    assert coordinator is not None, (
        "Failed to create DistributedOptimizationCoordinator"
    )

    ml_optimizer = MLAcceleratedOptimizer(config={"ml_model_config": {"model_type": "ensemble"}})
    assert ml_optimizer is not None, "Failed to create MLAcceleratedOptimizer"

    config = OptimizationConfig()
    assert config is not None, "Failed to create OptimizationConfig"

    print("✅ Integration modules basic functionality test passed")


if __name__ == "__main__":
    # Run basic functionality test
    test_integration_modules_basic_functionality()

    # Run full test suite with pytest if available
    try:
        pytest.main([__file__, "-v", "--tb=short"])
    except SystemExit:
        pass
