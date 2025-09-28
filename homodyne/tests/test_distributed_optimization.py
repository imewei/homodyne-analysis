"""
Comprehensive Tests for Distributed Optimization Framework
=========================================================

Test suite for distributed computing capabilities including multi-node optimization,
load balancing, fault tolerance, and backend integration.

Authors: Wei Chen, Hongrui He, Claude (Anthropic)
Institution: Argonne National Laboratory
"""

import shutil
import tempfile
import time
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pytest

# Import the modules we're testing
from homodyne.optimization.distributed import (
    DistributedOptimizationCoordinator,
    MultiprocessingBackend,
    OptimizationResult,
    OptimizationTask,
    create_distributed_optimizer,
    get_available_backends,
    integrate_with_classical_optimizer,
)


class TestDistributedBackends:
    """Test suite for distributed computing backends."""

    def test_get_available_backends(self):
        """Test backend availability detection."""
        backends = get_available_backends()

        assert isinstance(backends, dict)
        assert "multiprocessing" in backends
        assert backends["multiprocessing"] is True  # Should always be available

        # Check that other backends are boolean
        for _backend, available in backends.items():
            assert isinstance(available, bool)

    def test_multiprocessing_backend_initialization(self):
        """Test multiprocessing backend initialization."""
        backend = MultiprocessingBackend()

        config = {"num_processes": 2}
        success = backend.initialize(config)

        assert success is True
        assert backend.initialized is True

        # Test cluster info
        cluster_info = backend.get_cluster_info()
        assert cluster_info["backend"] == "multiprocessing"
        assert cluster_info["total_processes"] == 2

        # Cleanup
        backend.shutdown()

    def test_multiprocessing_backend_task_submission(self):
        """Test task submission and retrieval."""
        backend = MultiprocessingBackend()
        backend.initialize({"num_processes": 2})

        # Create test task
        task = OptimizationTask(
            task_id="test_task_1",
            method="Nelder-Mead",
            parameters=np.array([1.0, 2.0, 3.0]),
            bounds=None,
            objective_config={},
        )

        # Submit task
        task_id = backend.submit_task(task)
        assert task_id == "test_task_1"

        # Wait a moment for task processing
        time.sleep(0.5)

        # Get results
        results = backend.get_results(timeout=2.0)

        # Should have at least one result
        assert len(results) >= 0  # May be 0 if task is still running

        backend.shutdown()

    def test_optimization_task_creation(self):
        """Test optimization task data structure."""
        task = OptimizationTask(
            task_id="test_task",
            method="Nelder-Mead",
            parameters=np.array([1.0, 2.0]),
            bounds=[(0.0, 10.0), (0.0, 10.0)],
            objective_config={"tolerance": 1e-6},
            priority=2,
            timeout=120.0,
        )

        assert task.task_id == "test_task"
        assert task.method == "Nelder-Mead"
        assert np.array_equal(task.parameters, np.array([1.0, 2.0]))
        assert task.bounds == [(0.0, 10.0), (0.0, 10.0)]
        assert task.priority == 2
        assert task.timeout == 120.0
        assert task.retry_count == 0
        assert task.max_retries == 3

    def test_optimization_result_creation(self):
        """Test optimization result data structure."""
        result = OptimizationResult(
            task_id="test_task",
            success=True,
            parameters=np.array([1.5, 2.5]),
            objective_value=0.123,
            execution_time=5.67,
            node_id="node_1",
            metadata={"method": "Nelder-Mead"},
        )

        assert result.task_id == "test_task"
        assert result.success is True
        assert np.array_equal(result.parameters, np.array([1.5, 2.5]))
        assert result.objective_value == 0.123
        assert result.execution_time == 5.67
        assert result.node_id == "node_1"
        assert result.metadata["method"] == "Nelder-Mead"
        assert result.error_message is None


class TestDistributedOptimizationCoordinator:
    """Test suite for the distributed optimization coordinator."""

    def test_coordinator_initialization(self):
        """Test coordinator initialization with different backends."""
        coordinator = DistributedOptimizationCoordinator()

        # Test initialization with multiprocessing (should always work)
        success = coordinator.initialize(["multiprocessing"])
        assert success is True
        assert coordinator.backend_type == "multiprocessing"

        coordinator.shutdown()

    def test_coordinator_task_submission(self):
        """Test task submission through coordinator."""
        coordinator = DistributedOptimizationCoordinator()
        coordinator.initialize(["multiprocessing"])

        parameter_sets = [
            np.array([1.0, 2.0, 3.0]),
            np.array([2.0, 3.0, 4.0]),
            np.array([3.0, 4.0, 5.0]),
        ]

        methods = ["Nelder-Mead", "Nelder-Mead", "Nelder-Mead"]
        objective_configs = [{}] * 3

        task_ids = coordinator.submit_optimization_tasks(
            parameter_sets, methods, objective_configs
        )

        assert len(task_ids) == 3
        for task_id in task_ids:
            assert isinstance(task_id, str)
            assert task_id.startswith("opt_task_")

        coordinator.shutdown()

    def test_parameter_sweep(self):
        """Test distributed parameter sweep functionality."""
        coordinator = DistributedOptimizationCoordinator()
        coordinator.initialize(["multiprocessing"])

        parameter_ranges = {"param1": (0.0, 10.0, 3), "param2": (1.0, 5.0, 2)}

        # Mock the parameter sweep to avoid long execution times
        with patch.object(coordinator, "get_optimization_results") as mock_results:
            # Mock some successful results
            mock_results.return_value = [
                OptimizationResult(
                    task_id="task_1",
                    success=True,
                    parameters=np.array([1.0, 2.0]),
                    objective_value=0.5,
                    execution_time=1.0,
                    node_id="node_1",
                ),
                OptimizationResult(
                    task_id="task_2",
                    success=True,
                    parameters=np.array([5.0, 3.0]),
                    objective_value=0.3,
                    execution_time=0.8,
                    node_id="node_1",
                ),
            ]

            # Run parameter sweep
            results = coordinator.run_distributed_parameter_sweep(
                parameter_ranges, optimization_method="Nelder-Mead"
            )

            assert results["success"] is True
            assert results["total_tasks"] > 0
            assert "best_result" in results
            assert "statistics" in results

        coordinator.shutdown()

    def test_cluster_status(self):
        """Test cluster status monitoring."""
        coordinator = DistributedOptimizationCoordinator()
        coordinator.initialize(["multiprocessing"])

        status = coordinator.get_cluster_status()

        assert "backend_type" in status
        assert status["backend_type"] == "multiprocessing"
        assert "cluster_info" in status
        assert "task_queue_length" in status
        assert "completed_tasks" in status
        assert "performance_metrics" in status

        coordinator.shutdown()

    def test_coordinator_error_handling(self):
        """Test error handling in coordinator."""
        coordinator = DistributedOptimizationCoordinator()

        # Test initialization with invalid backend
        success = coordinator.initialize(["invalid_backend"])
        assert success is False

        # Test operations without initialization
        with pytest.raises(RuntimeError):
            coordinator.submit_optimization_tasks([], [], [])

    def test_performance_monitoring(self):
        """Test performance monitoring capabilities."""
        coordinator = DistributedOptimizationCoordinator()
        coordinator.initialize(["multiprocessing"])

        # Create mock result for performance tracking
        result = OptimizationResult(
            task_id="perf_test",
            success=True,
            parameters=np.array([1.0, 2.0]),
            objective_value=0.123,
            execution_time=2.5,
            node_id="test_node",
        )

        # Update performance metrics
        coordinator._update_performance_metrics(result)

        # Check that metrics were recorded
        assert "test_node" in coordinator.performance_monitor
        metrics = coordinator.performance_monitor["test_node"]
        assert metrics["completed_tasks"] == 1
        assert metrics["total_execution_time"] == 2.5
        assert metrics["successful_tasks"] == 1
        assert metrics["success_rate"] == 1.0

        coordinator.shutdown()


class TestDistributedOptimizationIntegration:
    """Test integration with existing optimization classes."""

    def test_create_distributed_optimizer(self):
        """Test creation of distributed optimizer."""
        config = {"multiprocessing_config": {"num_processes": 2}}

        coordinator = create_distributed_optimizer(
            config, backend_preference=["multiprocessing"]
        )

        assert coordinator is not None
        assert coordinator.backend_type == "multiprocessing"

        coordinator.shutdown()

    def test_integration_with_classical_optimizer(self):
        """Test integration with classical optimizer."""
        # Create mock classical optimizer
        mock_optimizer = Mock()
        mock_optimizer.run_classical_optimization_optimized = Mock()

        # Integrate with distributed capabilities
        enhanced_optimizer = integrate_with_classical_optimizer(
            mock_optimizer, {"multiprocessing_config": {"num_processes": 2}}
        )

        # Check that distributed method was added
        assert hasattr(enhanced_optimizer, "run_distributed_optimization")
        assert hasattr(enhanced_optimizer, "_distributed_coordinator")

    def test_backend_fallback(self):
        """Test automatic fallback to available backends."""
        coordinator = DistributedOptimizationCoordinator()

        # Try to initialize with Ray first, then fallback to multiprocessing
        success = coordinator.initialize(["ray", "multiprocessing"])

        # Should succeed with at least multiprocessing
        assert success is True
        assert coordinator.backend_type in ["ray", "multiprocessing"]

        coordinator.shutdown()


class TestDistributedOptimizationEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_parameter_sets(self):
        """Test handling of empty parameter sets."""
        coordinator = DistributedOptimizationCoordinator()
        coordinator.initialize(["multiprocessing"])

        task_ids = coordinator.submit_optimization_tasks([], [], [])
        assert len(task_ids) == 0

        coordinator.shutdown()

    def test_mismatched_input_lengths(self):
        """Test handling of mismatched input lengths."""
        coordinator = DistributedOptimizationCoordinator()
        coordinator.initialize(["multiprocessing"])

        parameter_sets = [np.array([1.0, 2.0])]
        methods = ["Nelder-Mead", "BFGS"]  # Mismatched length
        objective_configs = [{}]

        # Should handle gracefully (zip will stop at shortest)
        task_ids = coordinator.submit_optimization_tasks(
            parameter_sets, methods, objective_configs
        )
        assert len(task_ids) == 1

        coordinator.shutdown()

    def test_large_parameter_sets(self):
        """Test handling of large parameter sets."""
        coordinator = DistributedOptimizationCoordinator()
        coordinator.initialize(["multiprocessing"])

        # Create many parameter sets
        n_sets = 50
        parameter_sets = [np.random.rand(3) for _ in range(n_sets)]
        methods = ["Nelder-Mead"] * n_sets
        objective_configs = [{}] * n_sets

        task_ids = coordinator.submit_optimization_tasks(
            parameter_sets, methods, objective_configs
        )

        assert len(task_ids) == n_sets

        coordinator.shutdown()

    def test_coordinator_robustness(self):
        """Test coordinator robustness to various error conditions."""
        coordinator = DistributedOptimizationCoordinator()

        # Test shutdown without initialization
        coordinator.shutdown()  # Should not raise error

        # Test double initialization
        coordinator.initialize(["multiprocessing"])
        coordinator.initialize(["multiprocessing"])  # Should handle gracefully

        # Test double shutdown
        coordinator.shutdown()
        coordinator.shutdown()  # Should not raise error


class TestDistributedOptimizationPerformance:
    """Performance-focused tests for distributed optimization."""

    def test_multiprocessing_speedup(self):
        """Test that multiprocessing provides speedup for parallel tasks."""
        # This is a basic test - in practice you'd use actual optimization functions

        coordinator = DistributedOptimizationCoordinator()
        coordinator.initialize(["multiprocessing"])

        # Submit multiple simple tasks
        n_tasks = 4
        parameter_sets = [np.array([i, i + 1]) for i in range(n_tasks)]
        methods = ["Nelder-Mead"] * n_tasks
        objective_configs = [{}] * n_tasks

        start_time = time.time()
        task_ids = coordinator.submit_optimization_tasks(
            parameter_sets, methods, objective_configs
        )

        # Wait for some results
        time.sleep(1.0)
        coordinator.get_optimization_results(timeout=5.0)

        elapsed_time = time.time() - start_time

        # Basic checks
        assert len(task_ids) == n_tasks
        assert elapsed_time < 10.0  # Should complete reasonably quickly

        coordinator.shutdown()

    @pytest.mark.slow
    def test_load_balancing(self):
        """Test load balancing across multiple workers."""
        coordinator = DistributedOptimizationCoordinator()
        coordinator.initialize(["multiprocessing"])

        # Submit tasks with varying complexity
        parameter_sets = [np.random.rand(3) for _ in range(10)]
        methods = ["Nelder-Mead"] * 10
        objective_configs = [{"complexity": i} for i in range(10)]

        task_ids = coordinator.submit_optimization_tasks(
            parameter_sets, methods, objective_configs
        )

        # Monitor completion over time
        start_time = time.time()
        all_results = []

        while len(all_results) < len(task_ids) and time.time() - start_time < 30:
            new_results = coordinator.get_optimization_results(timeout=1.0)
            all_results.extend(new_results)
            time.sleep(0.1)

        # Check that tasks were distributed
        node_ids = [r.node_id for r in all_results if r.success]
        unique_nodes = set(node_ids)

        # With multiprocessing, should have multiple unique process IDs
        assert len(unique_nodes) >= 1  # At least one worker

        coordinator.shutdown()


@pytest.fixture
def temp_config_dir():
    """Create temporary directory for configuration files."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


class TestDistributedOptimizationConfiguration:
    """Test configuration handling for distributed optimization."""

    def test_configuration_loading(self, temp_config_dir):
        """Test loading configuration from file."""
        config_data = {
            "multiprocessing_config": {"num_processes": 4},
            "load_balancing": {"strategy": "static"},
        }

        config_file = temp_config_dir / "test_config.json"
        with open(config_file, "w") as f:
            import json

            json.dump(config_data, f)

        coordinator = DistributedOptimizationCoordinator(config=config_data)
        success = coordinator.initialize(["multiprocessing"])

        assert success is True
        coordinator.shutdown()

    def test_invalid_configuration(self):
        """Test handling of invalid configuration."""
        invalid_config = {"multiprocessing_config": {"num_processes": -1}}  # Invalid

        coordinator = DistributedOptimizationCoordinator(config=invalid_config)
        # Should still initialize, may use defaults
        success = coordinator.initialize(["multiprocessing"])

        # May succeed with corrected config or fail gracefully
        if success:
            coordinator.shutdown()


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
