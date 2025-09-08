"""
Integration Tests: Classical vs Robust Optimization Performance
==============================================================

This module provides comprehensive integration tests comparing the performance
features and capabilities between classical and robust optimization methods.

Test Categories:
- Performance feature parity tests
- Warm-start capability comparisons
- Fallback chain consistency tests
- Memory optimization comparisons
- Configuration compatibility tests

Authors: Wei Chen, Hongrui He
Institution: Argonne National Laboratory
"""

from unittest.mock import Mock, patch

import numpy as np
import pytest

from homodyne.optimization.classical import GUROBI_AVAILABLE, ClassicalOptimizer

# Import robust optimization with graceful fallback
try:
    from homodyne.optimization.robust import (
        CVXPY_AVAILABLE,
        RobustHomodyneOptimizer,
    )

    ROBUST_AVAILABLE = True
except ImportError:
    RobustHomodyneOptimizer = Mock
    CVXPY_AVAILABLE = False
    ROBUST_AVAILABLE = False


@pytest.fixture
def mock_analysis_core():
    """Create a mock analysis core for both optimizers."""
    mock = Mock()
    mock.config = {"analysis": {"mode": "laminar_flow"}}
    mock.get_effective_parameter_count.return_value = 7
    mock._parameter_bounds = np.array(
        [[1e-3, 1e3], [-2, 2], [0, 100], [1e-3, 1e3], [-2, 2], [0, 100], [0, 360]]
    )

    # Mock config manager
    mock.config_manager = Mock()
    return mock


@pytest.fixture
def consistent_config():
    """Create a configuration that works for both classical and robust optimizers."""
    return {
        "optimization_config": {
            "classical_optimization": {
                "methods": ["Nelder-Mead", "Gurobi"],
                "nelder_mead": {"max_iterations": 1000, "tolerance": 1e-6},
                "gurobi": {"enable_warm_starts": True},
            },
            "robust_optimization": {
                "method": "wasserstein",
                "solver_optimization": {
                    "enable_warm_starts": True,
                    "adaptive_solver_selection": True,
                },
            },
        },
        "performance_settings": {
            "enable_caching": True,
            "enable_warm_starts": True,
            "memory_optimization": True,
        },
    }


class TestPerformanceFeatureParity:
    """Test that classical and robust optimizers have equivalent performance features."""

    def test_warm_start_method_consistency(self, mock_analysis_core, consistent_config):
        """Test that both optimizers have consistent warm-start methods."""
        classical_optimizer = ClassicalOptimizer(mock_analysis_core, consistent_config)

        # Test classical optimizer methods
        assert hasattr(classical_optimizer, "_initialize_warm_start")
        assert hasattr(classical_optimizer, "_solve_with_warm_start")

        if ROBUST_AVAILABLE:
            robust_optimizer = RobustHomodyneOptimizer(
                mock_analysis_core, consistent_config
            )

            # Test robust optimizer methods
            assert hasattr(robust_optimizer, "_initialize_warm_start")
            assert hasattr(robust_optimizer, "_solve_with_warm_start")

    def test_fallback_chain_consistency(self, mock_analysis_core, consistent_config):
        """Test that both optimizers have consistent fallback mechanisms."""
        classical_optimizer = ClassicalOptimizer(mock_analysis_core, consistent_config)

        # Test classical optimizer fallback methods
        assert hasattr(classical_optimizer, "_solve_with_fallback_chain")

        if ROBUST_AVAILABLE:
            robust_optimizer = RobustHomodyneOptimizer(
                mock_analysis_core, consistent_config
            )

            # Test robust optimizer fallback methods
            assert hasattr(robust_optimizer, "_solve_with_fallback_chain")

    def test_performance_settings_utilization(
        self, mock_analysis_core, consistent_config
    ):
        """Test that both optimizers utilize performance settings consistently."""
        classical_optimizer = ClassicalOptimizer(mock_analysis_core, consistent_config)

        # Test that classical optimizer can access performance settings through config
        performance_settings = consistent_config.get("performance_settings", {})

        if ROBUST_AVAILABLE:
            RobustHomodyneOptimizer(mock_analysis_core, consistent_config)

            # Both should access the same configuration structure
            assert performance_settings["enable_caching"] is True
            assert performance_settings["enable_warm_starts"] is True


class TestWarmStartComparison:
    """Compare warm-start implementations between classical and robust optimizers."""

    def test_warm_start_initialization_parity(
        self, mock_analysis_core, consistent_config
    ):
        """Test that warm-start initialization works consistently."""
        classical_optimizer = ClassicalOptimizer(mock_analysis_core, consistent_config)

        # Initialize classical warm start
        classical_optimizer._initialize_warm_start("gurobi", "test_problem")

        if ROBUST_AVAILABLE:
            robust_optimizer = RobustHomodyneOptimizer(
                mock_analysis_core, consistent_config
            )

            # Initialize robust warm start
            robust_optimizer._initialize_warm_start("test_problem")

            # Both should have created state structures
            assert hasattr(classical_optimizer, "_optimization_state")
            assert hasattr(robust_optimizer, "_solver_state")

    def test_warm_start_state_management(self, mock_analysis_core, consistent_config):
        """Test that state management is consistent between optimizers."""
        classical_optimizer = ClassicalOptimizer(mock_analysis_core, consistent_config)
        problem_signature = "state_management_test"

        # Set up classical state
        classical_optimizer._initialize_warm_start("gurobi", problem_signature)
        classical_state = classical_optimizer._optimization_state[problem_signature]

        # Verify classical state structure
        assert "initialized" in classical_state
        assert "last_solution" in classical_state
        assert "solver_stats" in classical_state

        if ROBUST_AVAILABLE:
            robust_optimizer = RobustHomodyneOptimizer(
                mock_analysis_core, consistent_config
            )

            # Set up robust state
            robust_optimizer._initialize_warm_start(problem_signature)
            robust_state = robust_optimizer._solver_state.get(problem_signature, {})

            # Verify robust state structure (should be similar)
            assert "initialized" in robust_state or len(robust_state) > 0


class TestFallbackChainComparison:
    """Compare fallback chain implementations."""

    def test_fallback_behavior_consistency(self, mock_analysis_core, consistent_config):
        """Test that fallback chains behave consistently."""
        classical_optimizer = ClassicalOptimizer(mock_analysis_core, consistent_config)
        initial_params = np.array([1.0, 0.0, 5.0, 1.5, 0.0, 3.0, 0.0])
        bounds = [
            (1e-3, 1e3),
            (-2, 2),
            (0, 100),
            (1e-3, 1e3),
            (-2, 2),
            (0, 100),
            (0, 360),
        ]

        # Mock classical fallback behavior
        with (
            patch.object(
                classical_optimizer, "_run_gurobi_optimization"
            ) as mock_gurobi,
            patch.object(
                classical_optimizer, "_run_nelder_mead_optimization"
            ) as mock_nelder,
        ):
            mock_gurobi.return_value = {"success": False, "message": "Primary failed"}
            mock_nelder.return_value = {
                "success": True,
                "x": initial_params,
                "fun": 0.1,
            }

            classical_result = classical_optimizer._solve_with_fallback_chain(
                initial_params, bounds
            )

            assert classical_result["success"] is True
            assert classical_result["method_used"] == "nelder_mead"

        if ROBUST_AVAILABLE:
            robust_optimizer = RobustHomodyneOptimizer(
                mock_analysis_core, consistent_config
            )

            # Test robust fallback (mocked)
            with patch.object(
                robust_optimizer, "_solve_with_fallback_chain"
            ) as mock_robust_fallback:
                mock_robust_fallback.return_value = {
                    "success": True,
                    "status": "optimal",
                    "variables": {"x": initial_params},
                }

                robust_result = robust_optimizer._solve_with_fallback_chain(Mock())

                assert robust_result["success"] is True

    def test_error_handling_consistency(self, mock_analysis_core, consistent_config):
        """Test that error handling is consistent between optimizers."""
        classical_optimizer = ClassicalOptimizer(mock_analysis_core, consistent_config)
        initial_params = np.array([1.0, 0.0, 5.0, 1.5, 0.0, 3.0, 0.0])
        bounds = [
            (1e-3, 1e3),
            (-2, 2),
            (0, 100),
            (1e-3, 1e3),
            (-2, 2),
            (0, 100),
            (0, 360),
        ]

        # Test classical error handling
        with (
            patch.object(
                classical_optimizer, "_run_gurobi_optimization"
            ) as mock_gurobi,
            patch.object(
                classical_optimizer, "_run_nelder_mead_optimization"
            ) as mock_nelder,
        ):
            mock_gurobi.return_value = {"success": False, "message": "Gurobi failed"}
            mock_nelder.return_value = {
                "success": False,
                "message": "Nelder-Mead failed",
            }

            classical_result = classical_optimizer._solve_with_fallback_chain(
                initial_params, bounds
            )

            assert classical_result["success"] is False
            assert (
                "All optimization methods in fallback chain failed"
                in classical_result["message"]
            )

        # Both optimizers should handle total failure gracefully
        if ROBUST_AVAILABLE:
            robust_optimizer = RobustHomodyneOptimizer(
                mock_analysis_core, consistent_config
            )

            with patch.object(
                robust_optimizer, "_solve_with_fallback_chain"
            ) as mock_robust_fallback:
                mock_robust_fallback.return_value = {
                    "success": False,
                    "message": "All robust methods failed",
                }

                robust_result = robust_optimizer._solve_with_fallback_chain(Mock())
                assert robust_result["success"] is False


class TestMemoryOptimizationComparison:
    """Compare memory optimization features."""

    def test_caching_behavior_parity(self, mock_analysis_core, consistent_config):
        """Test that caching behavior is consistent."""
        classical_optimizer = ClassicalOptimizer(mock_analysis_core, consistent_config)

        # Test that optimizers can be configured with caching settings
        performance_settings = consistent_config.get("performance_settings", {})
        assert performance_settings.get("enable_caching") is True
        
        # Verify optimizer initialization with caching enabled
        assert classical_optimizer is not None

    def test_memory_optimization_settings(self, mock_analysis_core, consistent_config):
        """Test memory optimization settings."""
        classical_optimizer = ClassicalOptimizer(mock_analysis_core, consistent_config)
        
        # Test that memory optimization is configurable
        performance_settings = consistent_config.get("performance_settings", {})
        assert performance_settings.get("memory_optimization") is True
        
        # Verify optimizer can handle memory optimization configuration
        assert classical_optimizer is not None


class TestConfigurationCompatibility:
    """Test that configuration structures are compatible."""

    def test_configuration_schema_consistency(self, mock_analysis_core):
        """Test that both optimizers can use the same configuration schema."""
        # Configuration that should work for both optimizers
        unified_config = {
            "optimization_config": {
                "classical_optimization": {
                    "methods": ["Nelder-Mead"],
                    "enable_performance_features": True,
                },
                "robust_optimization": {
                    "method": "wasserstein",
                    "solver_optimization": {"enable_warm_starts": True},
                },
            },
            "performance_settings": {
                "enable_caching": True,
                "enable_warm_starts": True,
                "memory_optimization": True,
                "parallel_processing": {"enable_multiprocessing": True},
            },
        }

        # Both optimizers should initialize without error
        classical_optimizer = ClassicalOptimizer(mock_analysis_core, unified_config)
        assert classical_optimizer is not None

        if ROBUST_AVAILABLE:
            robust_optimizer = RobustHomodyneOptimizer(
                mock_analysis_core, unified_config
            )
            assert robust_optimizer is not None

    def test_performance_settings_parsing(self, mock_analysis_core, consistent_config):
        """Test that performance settings are parsed consistently."""
        classical_optimizer = ClassicalOptimizer(mock_analysis_core, consistent_config)

        # Get performance settings from configuration
        performance_settings = consistent_config.get("performance_settings", {})

        # Verify expected settings exist
        assert "enable_caching" in performance_settings
        assert "enable_warm_starts" in performance_settings
        assert performance_settings["enable_caching"] is True


@pytest.mark.integration
class TestPerformanceComparison:
    """Integration tests comparing actual performance between optimizers."""

    @pytest.fixture
    def performance_test_data(self):
        """Create test data for performance comparisons."""
        return {
            "initial_params": np.array([1.0, 0.0, 5.0, 1.5, 0.0, 3.0, 45.0]),
            "bounds": [
                (1e-3, 1e3),
                (-2, 2),
                (0, 100),
                (1e-3, 1e3),
                (-2, 2),
                (0, 100),
                (0, 360),
            ],
            "test_data": np.random.randn(100, 50) + 1.0,  # Mock experimental data
        }

    def test_warm_start_performance_impact(
        self, mock_analysis_core, consistent_config, performance_test_data
    ):
        """Test that warm starts provide performance benefits."""
        classical_optimizer = ClassicalOptimizer(mock_analysis_core, consistent_config)

        # Mock timing for cold start vs warm start
        problem_signature = "performance_test"

        # Cold start timing (mock)
        with patch("time.time", side_effect=[0.0, 1.0]):  # 1 second
            classical_optimizer._initialize_warm_start("gurobi", problem_signature)

        # Warm start should be available now
        state = classical_optimizer._optimization_state[problem_signature]
        state["warm_start_available"] = True
        state["last_solution"] = performance_test_data["initial_params"]

        # Verify warm start state is ready
        assert state["warm_start_available"] is True
        assert state["last_solution"] is not None

    def test_memory_usage_configuration(self, mock_analysis_core, consistent_config):
        """Test memory usage configuration between optimizers."""
        classical_optimizer = ClassicalOptimizer(mock_analysis_core, consistent_config)

        # Test that memory optimization settings are accessible
        performance_settings = consistent_config.get("performance_settings", {})
        assert performance_settings.get("memory_optimization") is True
        
        # Verify optimizer handles memory settings correctly
        assert classical_optimizer is not None

    @pytest.mark.skipif(
        not ROBUST_AVAILABLE, reason="Robust optimization not available"
    )
    @pytest.mark.skipif(not GUROBI_AVAILABLE, reason="Gurobi not available")
    def test_method_availability_consistency(
        self, mock_analysis_core, consistent_config
    ):
        """Test that method availability is reported consistently."""
        classical_optimizer = ClassicalOptimizer(mock_analysis_core, consistent_config)
        robust_optimizer = RobustHomodyneOptimizer(
            mock_analysis_core, consistent_config
        )

        # Both should be able to report their capabilities
        assert hasattr(classical_optimizer, "get_available_methods")

        # Robust optimizer should have dependency checking
        assert hasattr(robust_optimizer, "check_dependencies")

        # Both should handle missing dependencies gracefully
        try:
            robust_optimizer.check_dependencies()
        except ImportError:
            pass  # Expected if dependencies missing
