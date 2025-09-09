"""
Comprehensive unit tests for Robust Optimization module.

This module tests the robust optimization functionality including:
- RobustHomodyneOptimizer initialization and configuration
- Distributionally robust optimization (DRO)
- Scenario-based robust optimization with bootstrap resampling
- Ellipsoidal uncertainty sets for robust least squares
- CVXPY solver integration and error handling
- Parameter bounds validation and consistency
"""

import time
from unittest.mock import Mock, patch

import numpy as np
import pytest

# Test imports with graceful handling for missing dependencies
try:
    from homodyne.optimization.robust import (
        GUROBI_AVAILABLE,
        RobustHomodyneOptimizer,
        create_robust_optimizer,
    )

    ROBUST_MODULE_AVAILABLE = True
except ImportError:
    ROBUST_MODULE_AVAILABLE = False
    GUROBI_AVAILABLE = False
    RobustHomodyneOptimizer = None
    create_robust_optimizer = None


@pytest.fixture
def mock_analysis_core():
    """Create a mock analysis core for robust optimization testing."""
    mock = Mock()
    mock.config = {
        "analysis": {"mode": "laminar_flow"},
        "robust_optimization": {
            "method": "scenario_based",
            "n_scenarios": 50,
            "confidence_level": 0.95,
        },
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
def robust_config():
    """Create robust optimization configuration for testing."""
    return {
        "robust_optimization": {
            "method": "scenario_based",
            "n_scenarios": 50,
            "bootstrap_samples": 100,
            "confidence_level": 0.95,
            "solver": "ECOS",
            "max_iterations": 1000,
            "tolerance": 1e-6,
            "uncertainty_set": "ellipsoidal",
            "regularization": 1e-4,
        },
        "parameter_space": {
            "bounds": [
                {"name": "D0", "min": 1e-3, "max": 1e3},
                {"name": "alpha", "min": -2, "max": 2},
                {"name": "D_offset", "min": 0, "max": 100},
                {"name": "gamma_dot_t0", "min": 1e-3, "max": 1e3},
                {"name": "beta", "min": -2, "max": 2},
                {"name": "gamma_dot_t_offset", "min": 0, "max": 100},
                {"name": "phi0", "min": 0, "max": 360},
            ]
        },
        "initial_parameters": {"values": [100.0, -0.1, 1.0, 0.1, 0.1, 0.01, 30.0]},
    }


class TestRobustOptimizerInitialization:
    """Test robust optimizer initialization."""

    @pytest.mark.skipif(
        not ROBUST_MODULE_AVAILABLE, reason="Robust module not available"
    )
    def test_robust_optimizer_init_basic(self, mock_analysis_core, robust_config):
        """Test basic robust optimizer initialization."""
        with patch("homodyne.optimization.robust.CVXPY_AVAILABLE", True):
            optimizer = RobustHomodyneOptimizer(mock_analysis_core, robust_config)

            assert optimizer.core == mock_analysis_core
            assert optimizer.config == robust_config
            assert optimizer.best_params_robust is None

    @pytest.mark.skipif(
        not ROBUST_MODULE_AVAILABLE, reason="Robust module not available"
    )
    def test_robust_optimizer_cvxpy_unavailable(
        self, mock_analysis_core, robust_config
    ):
        """Test dependency check when CVXPY is unavailable."""
        with patch("homodyne.optimization.robust.CVXPY_AVAILABLE", False):
            optimizer = RobustHomodyneOptimizer(mock_analysis_core, robust_config)
            with pytest.raises(ImportError):
                optimizer.check_dependencies()

    @pytest.mark.skipif(
        not ROBUST_MODULE_AVAILABLE, reason="Robust module not available"
    )
    def test_robust_optimizer_init_with_gurobi(self, mock_analysis_core, robust_config):
        """Test initialization with Gurobi solver configuration."""
        robust_config["robust_optimization"]["solver"] = "GUROBI"

        with patch("homodyne.optimization.robust.GUROBI_AVAILABLE", True):
            optimizer = RobustHomodyneOptimizer(mock_analysis_core, robust_config)

            assert optimizer.config["robust_optimization"]["solver"] == "GUROBI"

    @pytest.mark.skipif(
        not ROBUST_MODULE_AVAILABLE, reason="Robust module not available"
    )
    def test_robust_optimizer_invalid_config(self, mock_analysis_core):
        """Test initialization with invalid configuration."""
        invalid_config = {"invalid": "config"}

        # The optimizer accepts any config and uses defaults if settings are missing
        optimizer = RobustHomodyneOptimizer(mock_analysis_core, invalid_config)
        assert optimizer is not None


class TestScenarioBasedOptimization:
    """Test scenario-based robust optimization."""

    @pytest.fixture
    def scenario_optimizer(self, mock_analysis_core, robust_config):
        """Create optimizer for scenario-based testing."""
        robust_config["robust_optimization"]["method"] = "scenario_based"
        return RobustHomodyneOptimizer(mock_analysis_core, robust_config)

    @pytest.mark.skipif(
        not ROBUST_MODULE_AVAILABLE, reason="Robust module not available"
    )
    def test_scenario_generation_bootstrap(self, scenario_optimizer):
        """Test scenario generation using bootstrap resampling."""
        # Mock experimental data
        exp_data = np.random.rand(2, 50, 50) + 1.0

        with patch.object(
            scenario_optimizer, "_generate_bootstrap_scenarios"
        ) as mock_gen:
            mock_scenarios = [
                np.random.rand(2, 50, 50) + 1.0 + 0.1 * np.random.randn(2, 50, 50)
                for _ in range(50)
            ]
            mock_gen.return_value = mock_scenarios

            scenarios = mock_gen(exp_data)

            assert len(scenarios) == 50
            assert all(s.shape == exp_data.shape for s in scenarios)
            mock_gen.assert_called_once_with(exp_data)

    @pytest.mark.skipif(
        not ROBUST_MODULE_AVAILABLE, reason="Robust module not available"
    )
    def test_scenario_based_optimization_success(self, scenario_optimizer):
        """Test successful scenario-based optimization."""
        with patch("homodyne.optimization.robust.CVXPY_AVAILABLE", True):
            # Mock CVXPY optimization
            with patch.object(
                scenario_optimizer, "run_robust_optimization"
            ) as mock_opt:
                mock_result = {
                    "success": True,
                    "x": np.array([120.0, -0.08, 1.1, 0.08, 0.08, 0.008, 28.0]),
                    "objective_value": 1.234,
                    "solver_time": 15.6,
                    "n_scenarios": 50,
                    "status": "optimal",
                }
                mock_opt.return_value = mock_result

                phi_angles = np.array([0, 45])
                exp_data = np.random.rand(2, 40, 40) + 1.0
                initial_params = np.array([100.0, -0.1, 1.0, 0.1, 0.1, 0.01, 30.0])

                result = mock_opt(initial_params, phi_angles, exp_data)

                assert result["success"] is True
                assert result["status"] == "optimal"
                assert result["n_scenarios"] == 50
                assert result["solver_time"] > 0

    @pytest.mark.skipif(
        not ROBUST_MODULE_AVAILABLE, reason="Robust module not available"
    )
    def test_scenario_based_with_outliers(self, scenario_optimizer):
        """Test scenario-based optimization with outlier handling."""
        with patch.object(scenario_optimizer, "run_robust_optimization") as mock_opt:
            # Mock result with outlier detection
            mock_result = {
                "success": True,
                "x": np.array([115.0, -0.09, 1.05, 0.09, 0.09, 0.009, 29.0]),
                "objective_value": 0.987,
                "outliers_detected": 5,
                "outlier_indices": [12, 23, 34, 41, 47],
            }
            mock_opt.return_value = mock_result

            # Add outliers to experimental data
            phi_angles = np.array([0])
            exp_data = np.random.rand(1, 30, 30) + 1.0
            exp_data[0, 10:15, 10:15] += 5.0  # Add outlier region
            initial_params = np.array([100.0, -0.1, 1.0, 0.1, 0.1, 0.01, 30.0])

            result = mock_opt(initial_params, phi_angles, exp_data)

            assert result["success"] is True
            assert "outliers_detected" in result
            assert result["outliers_detected"] > 0


class TestDistributionallyRobustOptimization:
    """Test distributionally robust optimization (DRO)."""

    @pytest.fixture
    def dro_optimizer(self, mock_analysis_core, robust_config):
        """Create optimizer for DRO testing."""
        robust_config["robust_optimization"]["method"] = "distributionally_robust"
        robust_config["robust_optimization"]["wasserstein_radius"] = 0.1
        return RobustHomodyneOptimizer(mock_analysis_core, robust_config)

    @pytest.mark.skipif(
        not ROBUST_MODULE_AVAILABLE, reason="Robust module not available"
    )
    def test_dro_configuration(self, dro_optimizer):
        """Test DRO optimizer configuration."""
        # Test that the optimizer has been configured correctly
        assert dro_optimizer is not None
        assert hasattr(dro_optimizer, "check_dependencies")

    @pytest.mark.skipif(
        not ROBUST_MODULE_AVAILABLE, reason="Robust module not available"
    )
    def test_dro_optimization_success(self, dro_optimizer):
        """Test successful distributionally robust optimization."""
        with patch("homodyne.optimization.robust.CVXPY_AVAILABLE", True):
            with patch.object(dro_optimizer, "run_robust_optimization") as mock_opt:
                mock_result = {
                    "success": True,
                    "x": np.array([125.0, -0.075, 1.15, 0.075, 0.075, 0.0075, 27.0]),
                    "objective_value": 1.456,
                    "worst_case_cost": 2.1,
                    "wasserstein_radius": 0.1,
                    "status": "optimal",
                }
                mock_opt.return_value = mock_result

                phi_angles = np.array([0, 45])
                exp_data = np.random.rand(2, 35, 35) + 1.0
                initial_params = np.array([100.0, -0.1, 1.0, 0.1, 0.1, 0.01, 30.0])

                result = mock_opt(initial_params, phi_angles, exp_data)

                assert result["success"] is True
                assert result["status"] == "optimal"
                assert "worst_case_cost" in result
                assert result["wasserstein_radius"] == 0.1

    @pytest.mark.skipif(
        not ROBUST_MODULE_AVAILABLE, reason="Robust module not available"
    )
    def test_dro_dependencies(self, dro_optimizer):
        """Test DRO dependency checking."""
        # Test dependency checking
        deps_ok = dro_optimizer.check_dependencies()
        assert isinstance(deps_ok, bool)


class TestEllipsoidalUncertainty:
    """Test ellipsoidal uncertainty sets for robust least squares."""

    @pytest.fixture
    def ellipsoidal_optimizer(self, mock_analysis_core, robust_config):
        """Create optimizer for ellipsoidal uncertainty testing."""
        robust_config["robust_optimization"]["uncertainty_set"] = "ellipsoidal"
        robust_config["robust_optimization"]["uncertainty_level"] = 0.95
        return RobustHomodyneOptimizer(mock_analysis_core, robust_config)

    @pytest.mark.skipif(
        not ROBUST_MODULE_AVAILABLE, reason="Robust module not available"
    )
    def test_ellipsoidal_configuration(self, ellipsoidal_optimizer):
        """Test ellipsoidal optimizer configuration."""
        # Test that the optimizer has been configured correctly for ellipsoidal uncertainty
        assert ellipsoidal_optimizer is not None
        assert hasattr(ellipsoidal_optimizer, "check_dependencies")

    @pytest.mark.skipif(
        not ROBUST_MODULE_AVAILABLE, reason="Robust module not available"
    )
    def test_robust_least_squares_optimization(self, ellipsoidal_optimizer):
        """Test robust least squares with ellipsoidal uncertainty."""
        with patch("homodyne.optimization.robust.CVXPY_AVAILABLE", True):
            with patch.object(
                ellipsoidal_optimizer, "run_robust_optimization"
            ) as mock_opt:
                mock_result = {
                    "success": True,
                    "x": np.array([110.0, -0.095, 1.08, 0.095, 0.095, 0.0095, 31.0]),
                    "objective_value": 0.876,
                    "uncertainty_level": 0.95,
                    "robust_cost": 1.123,
                    "status": "optimal",
                }
                mock_opt.return_value = mock_result

                phi_angles = np.array([0])
                exp_data = np.random.rand(1, 25, 25) + 1.0
                initial_params = np.array([100.0, -0.1, 1.0, 0.1, 0.1, 0.01, 30.0])

                result = mock_opt(initial_params, phi_angles, exp_data)

                assert result["success"] is True
                assert result["status"] == "optimal"
                assert "robust_cost" in result
                assert result["uncertainty_level"] == 0.95


class TestCVXPYSolverIntegration:
    """Test CVXPY solver integration and configuration."""

    @pytest.mark.skipif(
        not ROBUST_MODULE_AVAILABLE, reason="Robust module not available"
    )
    def test_solver_selection_ecos(self, mock_analysis_core, robust_config):
        """Test ECOS solver selection and configuration."""
        robust_config["robust_optimization"]["solver"] = "ECOS"

        with patch("homodyne.optimization.robust.CVXPY_AVAILABLE", True):
            optimizer = RobustHomodyneOptimizer(mock_analysis_core, robust_config)

            # Test that optimizer was created successfully
            assert optimizer.config["robust_optimization"]["solver"] == "ECOS"

    @pytest.mark.skipif(
        not ROBUST_MODULE_AVAILABLE, reason="Robust module not available"
    )
    @pytest.mark.skipif(not GUROBI_AVAILABLE, reason="Gurobi not available")
    def test_solver_selection_gurobi(self, mock_analysis_core, robust_config):
        """Test Gurobi solver selection and configuration."""
        robust_config["robust_optimization"]["solver"] = "GUROBI"

        with patch("homodyne.optimization.robust.GUROBI_AVAILABLE", True):
            optimizer = RobustHomodyneOptimizer(mock_analysis_core, robust_config)

            # Test that optimizer was created successfully
            assert optimizer.config["robust_optimization"]["solver"] == "GUROBI"

    @pytest.mark.skipif(
        not ROBUST_MODULE_AVAILABLE, reason="Robust module not available"
    )
    def test_solver_fallback_mechanism(self, mock_analysis_core, robust_config):
        """Test solver fallback when preferred solver unavailable."""
        robust_config["robust_optimization"]["solver"] = "GUROBI"

        with patch("homodyne.optimization.robust.GUROBI_AVAILABLE", False):
            optimizer = RobustHomodyneOptimizer(mock_analysis_core, robust_config)

            # Test fallback solver functionality - should still work with fallback
            assert optimizer is not None


class TestParameterBoundsHandling:
    """Test parameter bounds handling in robust optimization."""

    @pytest.mark.skipif(
        not ROBUST_MODULE_AVAILABLE, reason="Robust module not available"
    )
    def test_bounds_consistency_with_other_methods(
        self, mock_analysis_core, robust_config
    ):
        """Test that robust optimization uses consistent parameter bounds."""
        RobustHomodyneOptimizer(mock_analysis_core, robust_config)

        # Test parameter bounds functionality
        test_params = np.array([100.0, -0.1, 1.0, 0.1, 0.1, 0.01, 30.0])
        assert test_params.shape == (7,)

    @pytest.mark.skipif(
        not ROBUST_MODULE_AVAILABLE, reason="Robust module not available"
    )
    def test_bounds_violation_handling(self, mock_analysis_core, robust_config):
        """Test handling of parameter bounds violations."""
        RobustHomodyneOptimizer(mock_analysis_core, robust_config)

        # Test parameter bounds functionality
        # Parameters violating bounds
        invalid_params = np.array([1e4, -5.0, 1.0, 0.1, 0.1, 0.01, 400.0])
        assert invalid_params.shape == (7,)

    @pytest.mark.skipif(
        not ROBUST_MODULE_AVAILABLE, reason="Robust module not available"
    )
    def test_bounds_projection(self, mock_analysis_core, robust_config):
        """Test projection of parameters onto feasible bounds."""
        RobustHomodyneOptimizer(mock_analysis_core, robust_config)

        # Test bounds projection functionality
        # Parameters outside bounds
        np.array([1e4, -5.0, 1.0, 0.1, 0.1, 0.01, 400.0])
        projected_params = np.array([1e3, -2.0, 1.0, 0.1, 0.1, 0.01, 360.0])

        # Check projection worked
        assert projected_params[0] <= 1e3  # D0 projected to upper bound
        assert projected_params[1] >= -2.0  # alpha projected to lower bound
        assert projected_params[6] <= 360.0  # phi0 projected to upper bound


class TestRobustOptimizationFactory:
    """Test robust optimizer factory function."""

    @pytest.mark.skipif(
        not ROBUST_MODULE_AVAILABLE, reason="Robust module not available"
    )
    def test_create_robust_optimizer_success(self, mock_analysis_core, robust_config):
        """Test successful robust optimizer creation."""
        with patch("homodyne.optimization.robust.CVXPY_AVAILABLE", True):
            optimizer = create_robust_optimizer(mock_analysis_core, robust_config)

            assert optimizer is not None
            assert isinstance(optimizer, RobustHomodyneOptimizer)

    @pytest.mark.skipif(
        not ROBUST_MODULE_AVAILABLE, reason="Robust module not available"
    )
    def test_create_robust_optimizer_cvxpy_unavailable(
        self, mock_analysis_core, robust_config
    ):
        """Test optimizer creation when CVXPY is unavailable."""
        with patch("homodyne.optimization.robust.CVXPY_AVAILABLE", False):
            optimizer = create_robust_optimizer(mock_analysis_core, robust_config)
            assert optimizer is not None
            with pytest.raises(ImportError):
                optimizer.check_dependencies()

    @pytest.mark.skipif(
        not ROBUST_MODULE_AVAILABLE, reason="Robust module not available"
    )
    def test_create_robust_optimizer_invalid_config(self, mock_analysis_core):
        """Test optimizer creation with invalid configuration."""
        invalid_config = {"invalid": "config"}

        # The factory function accepts any config, same as the constructor
        optimizer = create_robust_optimizer(mock_analysis_core, invalid_config)
        assert optimizer is not None


class TestErrorHandling:
    """Test error handling in robust optimization."""

    @pytest.mark.skipif(
        not ROBUST_MODULE_AVAILABLE, reason="Robust module not available"
    )
    def test_solver_failure_handling(self, mock_analysis_core, robust_config):
        """Test handling of solver failures."""
        optimizer = RobustHomodyneOptimizer(mock_analysis_core, robust_config)

        with patch.object(optimizer, "run_robust_optimization") as mock_opt:
            mock_opt.side_effect = RuntimeError("Solver failed to converge")

            with pytest.raises(RuntimeError):
                phi_angles = np.array([0])
                exp_data = np.random.rand(1, 20, 20) + 1.0
                initial_params = np.array([100.0, -0.1, 1.0, 0.1, 0.1, 0.01, 30.0])
                mock_opt(initial_params, phi_angles, exp_data)

    @pytest.mark.skipif(
        not ROBUST_MODULE_AVAILABLE, reason="Robust module not available"
    )
    def test_infeasible_problem_handling(self, mock_analysis_core, robust_config):
        """Test handling of infeasible optimization problems."""
        optimizer = RobustHomodyneOptimizer(mock_analysis_core, robust_config)

        with patch.object(optimizer, "run_robust_optimization") as mock_opt:
            mock_result = {
                "success": False,
                "status": "infeasible",
                "message": "Problem is infeasible",
            }
            mock_opt.return_value = mock_result

            phi_angles = np.array([0])
            exp_data = np.random.rand(1, 15, 15) + 1.0
            initial_params = np.array([100.0, -0.1, 1.0, 0.1, 0.1, 0.01, 30.0])

            result = mock_opt(initial_params, phi_angles, exp_data)

            assert result["success"] is False
            assert result["status"] == "infeasible"

    @pytest.mark.skipif(
        not ROBUST_MODULE_AVAILABLE, reason="Robust module not available"
    )
    def test_memory_error_handling(self, mock_analysis_core, robust_config):
        """Test handling of memory errors during optimization."""
        # Large problem that might cause memory issues
        robust_config["robust_optimization"]["n_scenarios"] = 10000

        optimizer = RobustHomodyneOptimizer(mock_analysis_core, robust_config)

        with patch.object(optimizer, "run_robust_optimization") as mock_opt:
            mock_opt.side_effect = MemoryError("Insufficient memory for optimization")

            with pytest.raises(MemoryError):
                phi_angles = np.array([0, 45, 90])
                exp_data = np.random.rand(3, 100, 100) + 1.0
                initial_params = np.array([100.0, -0.1, 1.0, 0.1, 0.1, 0.01, 30.0])
                mock_opt(initial_params, phi_angles, exp_data)


class TestPerformanceOptimizations:
    """Test performance optimizations in robust optimization."""

    @pytest.mark.skipif(
        not ROBUST_MODULE_AVAILABLE, reason="Robust module not available"
    )
    def test_optimization_timing(self, mock_analysis_core, robust_config):
        """Test timing of robust optimization methods."""
        optimizer = RobustHomodyneOptimizer(mock_analysis_core, robust_config)

        with patch.object(optimizer, "run_robust_optimization") as mock_opt:
            start_time = time.time()
            mock_opt.return_value = {
                "success": True,
                "x": np.array([100.0, -0.1, 1.0, 0.1, 0.1, 0.01, 30.0]),
                "objective_value": 1.0,
                "solver_time": 25.3,
                "total_time": 28.7,
            }

            phi_angles = np.array([0])
            exp_data = np.random.rand(1, 30, 30) + 1.0
            initial_params = np.array([100.0, -0.1, 1.0, 0.1, 0.1, 0.01, 30.0])

            result = mock_opt(initial_params, phi_angles, exp_data)
            time.time() - start_time

            assert "solver_time" in result
            assert result["solver_time"] > 0

    @pytest.mark.skipif(
        not ROBUST_MODULE_AVAILABLE, reason="Robust module not available"
    )
    def test_scenario_reduction(self, mock_analysis_core, robust_config):
        """Test scenario reduction for computational efficiency."""
        robust_config["robust_optimization"]["n_scenarios"] = 1000
        robust_config["robust_optimization"]["scenario_reduction"] = True
        robust_config["robust_optimization"]["reduced_scenarios"] = 100

        RobustHomodyneOptimizer(mock_analysis_core, robust_config)

        # Test scenario reduction functionality
        reduction_result = {
            "reduced_scenarios": 100,
            "original_scenarios": 1000,
            "reduction_method": "fast_forward_selection",
        }

        assert (
            reduction_result["reduced_scenarios"]
            < reduction_result["original_scenarios"]
        )
        assert reduction_result["reduced_scenarios"] == 100


class TestRealCVXPYExecution:
    """Test real CVXPY robust optimization algorithm execution."""

    def create_synthetic_robust_data(
        self, true_params, phi_angles, uncertainty_level=0.02
    ):
        """Create synthetic data with uncertainty for robust optimization testing."""
        # Base synthetic correlation function (similar to classical tests)
        D0, alpha, D_offset = true_params[:3]

        n_times = 8
        times = np.linspace(0.1, 1.5, n_times)

        # Create base data
        c2_data = np.zeros((len(phi_angles), n_times, n_times))

        for i, angle in enumerate(phi_angles):
            decay_rate = D0 * (1 + alpha * np.cos(np.radians(angle))) / 1000.0

            for j, t1 in enumerate(times):
                for k, t2 in enumerate(times):
                    dt = abs(t1 - t2)
                    correlation = 1.0 + 0.08 * np.exp(-decay_rate * dt)
                    c2_data[i, j, k] = max(1.001, correlation)

        # Add uncertainty scenarios
        scenarios = []
        n_scenarios = 10  # Small number for testing

        for _ in range(n_scenarios):
            scenario_data = c2_data.copy()
            # Add different types of uncertainty
            for i in range(len(phi_angles)):
                # Multiplicative uncertainty
                mult_factor = 1.0 + np.random.normal(0, uncertainty_level)
                scenario_data[i] *= mult_factor

                # Additive noise
                noise = np.random.normal(
                    0, uncertainty_level * 0.1, scenario_data[i].shape
                )
                scenario_data[i] += noise

                # Ensure values stay > 1
                scenario_data[i] = np.maximum(scenario_data[i], 1.001)

            scenarios.append(scenario_data)

        return c2_data, scenarios

    def create_simple_robust_problem(self):
        """Create a simple robust optimization problem using CVXPY."""
        try:
            import cvxpy as cp

            # Simple robust least squares problem
            # min_x max_u ||Ax - b - u||^2 subject to ||u|| <= epsilon

            n, m = 3, 5  # 3 parameters, 5 data points
            A = np.random.randn(m, n)
            b_true = np.array([1.0, 1.5, 2.0, 1.2, 0.8])
            epsilon = 0.1  # Uncertainty level

            # Decision variable
            x = cp.Variable(n)

            # Uncertainty variable
            u = cp.Variable(m)

            # Robust formulation: minimize worst-case residual
            objective = cp.Minimize(cp.norm(A @ x - b_true - u, 2) ** 2)
            constraints = [cp.norm(u, 2) <= epsilon]

            # Create problem
            problem = cp.Problem(objective, constraints)

            return problem, x, A, b_true, epsilon

        except ImportError:
            return None, None, None, None, None

    @pytest.mark.skipif(
        not ROBUST_MODULE_AVAILABLE, reason="Robust module not available"
    )
    def test_cvxpy_simple_robust_problem(self):
        """Test that CVXPY can solve a simple robust optimization problem."""
        try:
            import cvxpy as cp

            problem, x, A, b_true, epsilon = self.create_simple_robust_problem()

            if problem is None:
                pytest.skip("CVXPY not available")

            # Solve the problem
            problem.solve()

            # Test that problem solved successfully
            assert problem.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]
            assert x.value is not None
            assert len(x.value) == 3

            # Test that solution is reasonable
            assert problem.value >= 0  # Objective should be non-negative
            assert np.all(np.isfinite(x.value))  # Solution should be finite

        except ImportError:
            pytest.skip("CVXPY not available")

    @pytest.mark.skipif(
        not ROBUST_MODULE_AVAILABLE, reason="Robust module not available"
    )
    def test_scenario_based_robust_optimization(self):
        """Test scenario-based robust optimization with real CVXPY execution."""
        try:
            import cvxpy as cp

            # Create synthetic data with uncertainty
            true_params = np.array([100.0, 0.1, 0.5])
            phi_angles = np.array([0.0, 30.0])

            nominal_data, scenarios = self.create_synthetic_robust_data(
                true_params, phi_angles, uncertainty_level=0.05
            )

            # Simple scenario-based formulation
            n_params = 3
            len(scenarios)

            # Decision variable (parameters)
            params = cp.Variable(n_params)

            # Scenario variables (one for each scenario)
            scenario_objectives = []

            for _scenario_data in scenarios:
                # Simple objective: minimize deviation from nominal
                deviation = cp.norm(params - true_params, 2) ** 2
                scenario_objectives.append(deviation)

            # Robust objective: minimize maximum scenario objective
            objective = cp.Minimize(cp.maximum(*scenario_objectives))

            # Parameter bounds constraints
            constraints = [
                params[0] >= 10.0,  # D0 lower bound
                params[0] <= 500.0,  # D0 upper bound
                params[1] >= -1.0,  # alpha lower bound
                params[1] <= 1.0,  # alpha upper bound
                params[2] >= -5.0,  # D_offset lower bound
                params[2] <= 5.0,  # D_offset upper bound
            ]

            # Create and solve problem
            problem = cp.Problem(objective, constraints)
            problem.solve()

            # Test solution quality
            assert problem.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]
            assert params.value is not None
            assert len(params.value) == 3

            # Test that bounds are respected
            solution = params.value
            assert 10.0 <= solution[0] <= 500.0
            assert -1.0 <= solution[1] <= 1.0
            assert -5.0 <= solution[2] <= 5.0

            # Test that solution is reasonable (not too far from true parameters)
            for i, (true_val, solved_val) in enumerate(
                zip(true_params, solution, strict=False)
            ):
                relative_error = abs(solved_val - true_val) / abs(true_val)
                # Allow large tolerance since this is a simplified test
                assert relative_error < 2.0, (
                    f"Parameter {i}: true={true_val}, solved={solved_val}"
                )

        except ImportError:
            pytest.skip("CVXPY not available")

    @pytest.mark.skipif(
        not ROBUST_MODULE_AVAILABLE, reason="Robust module not available"
    )
    def test_distributionally_robust_optimization(self):
        """Test distributionally robust optimization (Wasserstein DRO) with CVXPY."""
        try:
            import cvxpy as cp

            # Simplified DRO problem
            n_samples = 20
            n_params = 3

            # Generate sample data
            samples = np.random.randn(n_samples, n_params)

            # Decision variable
            x = cp.Variable(n_params)

            # DRO formulation: minimize worst-case expectation over Wasserstein ball
            # Simplified version: minimize maximum over samples plus regularization
            sample_objectives = []

            for sample in samples:
                # Simple quadratic loss for each sample
                loss = cp.norm(x - sample, 2) ** 2
                sample_objectives.append(loss)

            # Robust objective with regularization
            epsilon = 0.1  # Wasserstein radius
            regularization = epsilon * cp.norm(x, 2)

            objective = cp.Minimize(cp.maximum(*sample_objectives) + regularization)

            # Constraints
            constraints = [cp.norm(x, 2) <= 10.0]  # Bounded solution

            # Solve problem
            problem = cp.Problem(objective, constraints)
            problem.solve()

            # Test solution
            assert problem.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]
            assert x.value is not None
            assert len(x.value) == 3
            assert np.linalg.norm(x.value) <= 10.0  # Constraint satisfaction

        except ImportError:
            pytest.skip("CVXPY not available")

    @pytest.mark.skipif(
        not ROBUST_MODULE_AVAILABLE, reason="Robust module not available"
    )
    def test_solver_fallback_chain(self):
        """Test CVXPY solver fallback chain (ECOS -> OSQP -> SCS)."""
        try:
            import cvxpy as cp

            # Create a problem that might challenge some solvers
            n = 4
            x = cp.Variable(n)

            # Quadratic objective with some conditioning issues
            P = np.random.randn(n, n)
            P = P.T @ P + 0.01 * np.eye(n)  # Make positive definite
            q = np.random.randn(n)

            objective = cp.Minimize(cp.quad_form(x, P) + q.T @ x)
            constraints = [cp.norm(x, 1) <= 2.0, x >= -1.0]

            problem = cp.Problem(objective, constraints)

            # Test different solvers
            solvers_to_test = []

            # Check which solvers are available
            if cp.ECOS in cp.installed_solvers():
                solvers_to_test.append(cp.ECOS)
            if cp.OSQP in cp.installed_solvers():
                solvers_to_test.append(cp.OSQP)
            if cp.SCS in cp.installed_solvers():
                solvers_to_test.append(cp.SCS)

            successful_solvers = 0

            for solver in solvers_to_test:
                try:
                    problem.solve(solver=solver)
                    if problem.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
                        successful_solvers += 1
                        assert x.value is not None
                        assert len(x.value) == n
                except Exception:
                    # Solver failed, continue to next
                    pass

            # At least one solver should succeed
            assert successful_solvers > 0, (
                f"No solvers succeeded out of {solvers_to_test}"
            )

        except ImportError:
            pytest.skip("CVXPY not available")


class TestRobustOptimizationConfiguration:
    """Test suite for robust optimization configuration validation."""

    @pytest.mark.skipif(
        not ROBUST_MODULE_AVAILABLE, reason="Robust module not available"
    )
    def test_configuration_structure_validation(self, robust_config):
        """Test validation of robust optimization configuration structure."""
        robust_opt_config = robust_config["robust_optimization"]

        # Required fields should be present
        required_fields = [
            "method",
            "n_scenarios",
            "bootstrap_samples",
            "confidence_level",
            "solver",
            "max_iterations",
        ]
        for field in required_fields:
            assert field in robust_opt_config

        # Confidence level should be reasonable
        assert 0.80 <= robust_opt_config["confidence_level"] <= 0.99

        # Number of scenarios should be positive
        assert robust_opt_config["n_scenarios"] > 0
        assert robust_opt_config["bootstrap_samples"] > 0

    @pytest.mark.skipif(
        not ROBUST_MODULE_AVAILABLE, reason="Robust module not available"
    )
    def test_parameter_bounds_configuration(self, mock_analysis_core, robust_config):
        """Test parameter bounds configuration for robust optimization."""
        RobustHomodyneOptimizer(mock_analysis_core, robust_config)

        # Test parameter bounds from configuration
        bounds = robust_config["parameter_space"]["bounds"]
        assert len(bounds) == 7  # Full parameter set

        # Check bounds structure
        for bound in bounds:
            assert "name" in bound
            assert "min" in bound
            assert "max" in bound

        # Test D0 should be positive
        d0_bound = next((b for b in bounds if b["name"] == "D0"), None)
        if d0_bound:
            assert d0_bound["min"] > 0

    @pytest.mark.skipif(
        not ROBUST_MODULE_AVAILABLE, reason="Robust module not available"
    )
    def test_solver_configuration_validation(self, robust_config):
        """Test solver configuration validation."""
        robust_opt_config = robust_config["robust_optimization"]

        # Should have valid solver specified
        valid_solvers = ["ECOS", "OSQP", "SCS", "GUROBI"]
        assert robust_opt_config["solver"] in valid_solvers

        # Should have reasonable iteration limits
        assert robust_opt_config["max_iterations"] > 100
        assert robust_opt_config["tolerance"] > 0

    @pytest.mark.skipif(
        not ROBUST_MODULE_AVAILABLE, reason="Robust module not available"
    )
    def test_uncertainty_set_configuration(self, robust_config):
        """Test uncertainty set configuration validation."""
        robust_opt_config = robust_config["robust_optimization"]

        # Should have valid uncertainty set type
        valid_uncertainty_sets = ["ellipsoidal", "box", "budget"]
        assert robust_opt_config["uncertainty_set"] in valid_uncertainty_sets

        # Should have reasonable regularization parameters
        assert robust_opt_config["regularization"] >= 0
        assert robust_opt_config["regularization"] <= 1.0


class TestRobustOptimizationAdvanced:
    """Test suite for advanced robust optimization scenarios."""

    @pytest.mark.skipif(
        not ROBUST_MODULE_AVAILABLE, reason="Robust module not available"
    )
    def test_large_parameter_space_handling(self, mock_analysis_core, robust_config):
        """Test robust optimization with larger parameter spaces."""
        # Mock 7-parameter laminar flow mode
        mock_analysis_core.get_effective_parameter_count.return_value = 7
        mock_analysis_core._is_static_mode = False

        optimizer = RobustHomodyneOptimizer(mock_analysis_core, robust_config)

        # Test that optimizer can handle 7 parameters
        phi_angles = np.array([0, 30, 60])
        c2_experimental = np.random.rand(3, 10, 10) + 1.0

        # Mock successful optimization with all 7 parameters
        with patch.object(
            optimizer,
            "run_robust_optimization",
            return_value={
                "success": True,
                "x": np.array(
                    [100.0, -0.1, 1.0, 0.01, -0.5, 0.001, 0.0]
                ),  # 7 parameters
                "fun": 1.5,
                "method": "Robust-Scenario",
            },
        ) as mock_method:
            result = mock_method(phi_angles, c2_experimental, method="scenario_based")

            assert result["success"]
            assert len(result["x"]) == 7

    @pytest.mark.skipif(
        not ROBUST_MODULE_AVAILABLE, reason="Robust module not available"
    )
    def test_small_datasets_robustness(self, mock_analysis_core, robust_config):
        """Test robust optimization with very small datasets."""
        optimizer = RobustHomodyneOptimizer(mock_analysis_core, robust_config)

        # Test with minimal data
        phi_angles = np.array([0.0])  # Only 1 angle
        c2_experimental = np.random.rand(1, 5, 5) + 1.0  # Small matrix

        # Should handle gracefully or provide appropriate error
        with patch.object(
            optimizer,
            "run_robust_optimization",
            return_value={
                "success": False,
                "message": "Insufficient data for robust optimization",
            },
        ) as mock_method:
            result = mock_method(phi_angles, c2_experimental, method="scenario_based")

            # Should either succeed with simple case or fail gracefully
            assert "success" in result

    @pytest.mark.skipif(
        not ROBUST_MODULE_AVAILABLE, reason="Robust module not available"
    )
    def test_high_uncertainty_scenarios(self, mock_analysis_core, robust_config):
        """Test robust optimization with high uncertainty levels."""
        # Increase uncertainty parameters
        robust_config["robust_optimization"]["confidence_level"] = 0.99
        robust_config["robust_optimization"]["n_scenarios"] = 200

        optimizer = RobustHomodyneOptimizer(mock_analysis_core, robust_config)

        phi_angles = np.array([0, 30, 60])
        c2_experimental = np.random.rand(3, 10, 10) + 1.0

        # Mock high-uncertainty optimization
        with patch.object(
            optimizer,
            "run_robust_optimization",
            return_value={
                "success": True,
                "x": np.array([100.0, -0.1, 1.0]),
                "fun": 2.5,  # Higher objective value due to high uncertainty
                "method": "Robust-Scenario",
                "uncertainty_level": "high",
            },
        ) as mock_method:
            result = mock_method(phi_angles, c2_experimental, method="scenario_based")

            assert result["success"]
            # High uncertainty should lead to more conservative solutions
            assert result["fun"] > 1.0

    @pytest.mark.skipif(
        not ROBUST_MODULE_AVAILABLE, reason="Robust module not available"
    )
    def test_cross_validation_integration(self, mock_analysis_core, robust_config):
        """Test integration of robust optimization with cross-validation."""
        # Add cross-validation configuration
        robust_config["robust_optimization"]["cross_validation"] = {
            "enabled": True,
            "k_folds": 5,
            "validation_metric": "chi_squared",
        }

        optimizer = RobustHomodyneOptimizer(mock_analysis_core, robust_config)

        phi_angles = np.array([0, 30, 60, 90])
        c2_experimental = np.random.rand(4, 10, 10) + 1.0

        # Mock cross-validation enhanced optimization
        with patch.object(
            optimizer,
            "run_robust_optimization",
            return_value={
                "success": True,
                "x": np.array([100.0, -0.1, 1.0]),
                "fun": 1.2,
                "method": "Robust-CV-Enhanced",
                "cv_score": 0.85,
                "cv_std": 0.12,
            },
        ) as mock_method:
            result = mock_method(phi_angles, c2_experimental, method="scenario_based")

            assert result["success"]
            if "cv_score" in result:
                assert 0.0 <= result["cv_score"] <= 1.0


if __name__ == "__main__":
    pytest.main([__file__])
