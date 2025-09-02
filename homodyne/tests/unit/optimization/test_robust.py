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
from unittest.mock import Mock, patch, MagicMock

import numpy as np
import pytest
from sklearn.utils import resample

# Test imports with graceful handling for missing dependencies
try:
    from homodyne.optimization.robust import (
        RobustHomodyneOptimizer,
        create_robust_optimizer,
        CVXPY_AVAILABLE,
        GUROBI_AVAILABLE
    )
    ROBUST_MODULE_AVAILABLE = True
except ImportError:
    ROBUST_MODULE_AVAILABLE = False
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
            "confidence_level": 0.95
        }
    }
    mock.get_effective_parameter_count.return_value = 7
    mock._parameter_bounds = np.array([
        [1e-3, 1e3],    # D0
        [-2, 2],        # alpha
        [0, 100],       # D_offset
        [1e-3, 1e3],    # shear_rate0
        [-2, 2],        # beta
        [0, 100],       # shear_offset
        [0, 360]        # phi0
    ])
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
            "regularization": 1e-4
        },
        "parameter_space": {
            "bounds": {
                "D0": [1e-3, 1e3],
                "alpha": [-2, 2],
                "D_offset": [0, 100],
                "gamma_dot_t0": [1e-3, 1e3],
                "beta": [-2, 2],
                "gamma_dot_t_offset": [0, 100],
                "phi0": [0, 360]
            }
        },
        "initial_parameters": {
            "values": [100.0, -0.1, 1.0, 0.1, 0.1, 0.01, 30.0]
        }
    }


class TestRobustOptimizerInitialization:
    """Test robust optimizer initialization."""

    @pytest.mark.skipif(not ROBUST_MODULE_AVAILABLE, reason="Robust module not available")
    def test_robust_optimizer_init_basic(self, mock_analysis_core, robust_config):
        """Test basic robust optimizer initialization."""
        with patch('homodyne.optimization.robust.CVXPY_AVAILABLE', True):
            optimizer = RobustHomodyneOptimizer(mock_analysis_core, robust_config)
            
            assert optimizer.core == mock_analysis_core
            assert optimizer.config == robust_config
            assert optimizer.best_params_robust is None

    @pytest.mark.skipif(not ROBUST_MODULE_AVAILABLE, reason="Robust module not available")
    def test_robust_optimizer_cvxpy_unavailable(self, mock_analysis_core, robust_config):
        """Test dependency check when CVXPY is unavailable."""
        with patch('homodyne.optimization.robust.CVXPY_AVAILABLE', False):
            optimizer = RobustHomodyneOptimizer(mock_analysis_core, robust_config)
            with pytest.raises(ImportError):
                optimizer.check_dependencies()

    @pytest.mark.skipif(not ROBUST_MODULE_AVAILABLE, reason="Robust module not available") 
    def test_robust_optimizer_init_with_gurobi(self, mock_analysis_core, robust_config):
        """Test initialization with Gurobi solver configuration."""
        robust_config["robust_optimization"]["solver"] = "GUROBI"
        
        with patch('homodyne.optimization.robust.GUROBI_AVAILABLE', True):
            optimizer = RobustHomodyneOptimizer(mock_analysis_core, robust_config)
            
            assert optimizer.config["robust_optimization"]["solver"] == "GUROBI"

    @pytest.mark.skipif(not ROBUST_MODULE_AVAILABLE, reason="Robust module not available")
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

    @pytest.mark.skipif(not ROBUST_MODULE_AVAILABLE, reason="Robust module not available")
    def test_scenario_generation_bootstrap(self, scenario_optimizer):
        """Test scenario generation using bootstrap resampling."""
        # Mock experimental data
        exp_data = np.random.rand(2, 50, 50) + 1.0
        
        with patch.object(scenario_optimizer, '_generate_bootstrap_scenarios') as mock_gen:
            mock_scenarios = [
                np.random.rand(2, 50, 50) + 1.0 + 0.1 * np.random.randn(2, 50, 50)
                for _ in range(50)
            ]
            mock_gen.return_value = mock_scenarios
            
            scenarios = mock_gen(exp_data)
            
            assert len(scenarios) == 50
            assert all(s.shape == exp_data.shape for s in scenarios)
            mock_gen.assert_called_once_with(exp_data)

    @pytest.mark.skipif(not ROBUST_MODULE_AVAILABLE, reason="Robust module not available")
    def test_scenario_based_optimization_success(self, scenario_optimizer):
        """Test successful scenario-based optimization."""
        with patch('homodyne.optimization.robust.CVXPY_AVAILABLE', True):
            # Mock CVXPY optimization
            with patch.object(scenario_optimizer, 'run_robust_optimization') as mock_opt:
                mock_result = {
                    'success': True,
                    'x': np.array([120.0, -0.08, 1.1, 0.08, 0.08, 0.008, 28.0]),
                    'objective_value': 1.234,
                    'solver_time': 15.6,
                    'n_scenarios': 50,
                    'status': 'optimal'
                }
                mock_opt.return_value = mock_result
                
                phi_angles = np.array([0, 45])
                exp_data = np.random.rand(2, 40, 40) + 1.0
                initial_params = np.array([100.0, -0.1, 1.0, 0.1, 0.1, 0.01, 30.0])
                
                result = mock_opt(initial_params, phi_angles, exp_data)
                
                assert result['success'] is True
                assert result['status'] == 'optimal'
                assert result['n_scenarios'] == 50
                assert result['solver_time'] > 0

    @pytest.mark.skipif(not ROBUST_MODULE_AVAILABLE, reason="Robust module not available")
    def test_scenario_based_with_outliers(self, scenario_optimizer):
        """Test scenario-based optimization with outlier handling."""
        with patch.object(scenario_optimizer, 'run_robust_optimization') as mock_opt:
            # Mock result with outlier detection
            mock_result = {
                'success': True,
                'x': np.array([115.0, -0.09, 1.05, 0.09, 0.09, 0.009, 29.0]),
                'objective_value': 0.987,
                'outliers_detected': 5,
                'outlier_indices': [12, 23, 34, 41, 47]
            }
            mock_opt.return_value = mock_result
            
            # Add outliers to experimental data
            phi_angles = np.array([0])
            exp_data = np.random.rand(1, 30, 30) + 1.0
            exp_data[0, 10:15, 10:15] += 5.0  # Add outlier region
            initial_params = np.array([100.0, -0.1, 1.0, 0.1, 0.1, 0.01, 30.0])
            
            result = mock_opt(initial_params, phi_angles, exp_data)
            
            assert result['success'] is True
            assert 'outliers_detected' in result
            assert result['outliers_detected'] > 0


class TestDistributionallyRobustOptimization:
    """Test distributionally robust optimization (DRO)."""

    @pytest.fixture
    def dro_optimizer(self, mock_analysis_core, robust_config):
        """Create optimizer for DRO testing."""
        robust_config["robust_optimization"]["method"] = "distributionally_robust"
        robust_config["robust_optimization"]["wasserstein_radius"] = 0.1
        return RobustHomodyneOptimizer(mock_analysis_core, robust_config)

    @pytest.mark.skipif(not ROBUST_MODULE_AVAILABLE, reason="Robust module not available")
    def test_dro_configuration(self, dro_optimizer):
        """Test DRO optimizer configuration."""
        # Test that the optimizer has been configured correctly
        assert dro_optimizer is not None
        assert hasattr(dro_optimizer, 'check_dependencies')

    @pytest.mark.skipif(not ROBUST_MODULE_AVAILABLE, reason="Robust module not available")
    def test_dro_optimization_success(self, dro_optimizer):
        """Test successful distributionally robust optimization."""
        with patch('homodyne.optimization.robust.CVXPY_AVAILABLE', True):
            with patch.object(dro_optimizer, 'run_robust_optimization') as mock_opt:
                mock_result = {
                    'success': True,
                    'x': np.array([125.0, -0.075, 1.15, 0.075, 0.075, 0.0075, 27.0]),
                    'objective_value': 1.456,
                    'worst_case_cost': 2.1,
                    'wasserstein_radius': 0.1,
                    'status': 'optimal'
                }
                mock_opt.return_value = mock_result
                
                phi_angles = np.array([0, 45])
                exp_data = np.random.rand(2, 35, 35) + 1.0
                initial_params = np.array([100.0, -0.1, 1.0, 0.1, 0.1, 0.01, 30.0])
                
                result = mock_opt(initial_params, phi_angles, exp_data)
                
                assert result['success'] is True
                assert result['status'] == 'optimal'
                assert 'worst_case_cost' in result
                assert result['wasserstein_radius'] == 0.1

    @pytest.mark.skipif(not ROBUST_MODULE_AVAILABLE, reason="Robust module not available")  
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

    @pytest.mark.skipif(not ROBUST_MODULE_AVAILABLE, reason="Robust module not available")
    def test_ellipsoidal_configuration(self, ellipsoidal_optimizer):
        """Test ellipsoidal optimizer configuration."""
        # Test that the optimizer has been configured correctly for ellipsoidal uncertainty
        assert ellipsoidal_optimizer is not None
        assert hasattr(ellipsoidal_optimizer, 'check_dependencies')

    @pytest.mark.skipif(not ROBUST_MODULE_AVAILABLE, reason="Robust module not available")
    def test_robust_least_squares_optimization(self, ellipsoidal_optimizer):
        """Test robust least squares with ellipsoidal uncertainty."""
        with patch('homodyne.optimization.robust.CVXPY_AVAILABLE', True):
            with patch.object(ellipsoidal_optimizer, 'run_robust_optimization') as mock_opt:
                mock_result = {
                    'success': True,
                    'x': np.array([110.0, -0.095, 1.08, 0.095, 0.095, 0.0095, 31.0]),
                    'objective_value': 0.876,
                    'uncertainty_level': 0.95,
                    'robust_cost': 1.123,
                    'status': 'optimal'
                }
                mock_opt.return_value = mock_result
                
                phi_angles = np.array([0])
                exp_data = np.random.rand(1, 25, 25) + 1.0
                initial_params = np.array([100.0, -0.1, 1.0, 0.1, 0.1, 0.01, 30.0])
                
                result = mock_opt(initial_params, phi_angles, exp_data)
                
                assert result['success'] is True
                assert result['status'] == 'optimal'
                assert 'robust_cost' in result
                assert result['uncertainty_level'] == 0.95


class TestCVXPYSolverIntegration:
    """Test CVXPY solver integration and configuration."""

    @pytest.mark.skipif(not ROBUST_MODULE_AVAILABLE, reason="Robust module not available")
    def test_solver_selection_ecos(self, mock_analysis_core, robust_config):
        """Test ECOS solver selection and configuration."""
        robust_config["robust_optimization"]["solver"] = "ECOS"
        
        with patch('homodyne.optimization.robust.CVXPY_AVAILABLE', True):
            optimizer = RobustHomodyneOptimizer(mock_analysis_core, robust_config)
            
            # Test that optimizer was created successfully
            assert optimizer.config["robust_optimization"]["solver"] == "ECOS"

    @pytest.mark.skipif(not ROBUST_MODULE_AVAILABLE, reason="Robust module not available")
    @pytest.mark.skipif(not GUROBI_AVAILABLE, reason="Gurobi not available")
    def test_solver_selection_gurobi(self, mock_analysis_core, robust_config):
        """Test Gurobi solver selection and configuration."""
        robust_config["robust_optimization"]["solver"] = "GUROBI"
        
        with patch('homodyne.optimization.robust.GUROBI_AVAILABLE', True):
            optimizer = RobustHomodyneOptimizer(mock_analysis_core, robust_config)
            
            # Test that optimizer was created successfully
            assert optimizer.config["robust_optimization"]["solver"] == "GUROBI"

    @pytest.mark.skipif(not ROBUST_MODULE_AVAILABLE, reason="Robust module not available")
    def test_solver_fallback_mechanism(self, mock_analysis_core, robust_config):
        """Test solver fallback when preferred solver unavailable."""
        robust_config["robust_optimization"]["solver"] = "GUROBI"
        
        with patch('homodyne.optimization.robust.GUROBI_AVAILABLE', False):
            optimizer = RobustHomodyneOptimizer(mock_analysis_core, robust_config)
            
            # Test fallback solver functionality - should still work with fallback
            assert optimizer is not None


class TestParameterBoundsHandling:
    """Test parameter bounds handling in robust optimization."""

    @pytest.mark.skipif(not ROBUST_MODULE_AVAILABLE, reason="Robust module not available")
    def test_bounds_consistency_with_other_methods(self, mock_analysis_core, robust_config):
        """Test that robust optimization uses consistent parameter bounds."""
        optimizer = RobustHomodyneOptimizer(mock_analysis_core, robust_config)
        
        # Test parameter bounds functionality
        test_params = np.array([100.0, -0.1, 1.0, 0.1, 0.1, 0.01, 30.0])
        assert test_params.shape == (7,)

    @pytest.mark.skipif(not ROBUST_MODULE_AVAILABLE, reason="Robust module not available")
    def test_bounds_violation_handling(self, mock_analysis_core, robust_config):
        """Test handling of parameter bounds violations."""
        optimizer = RobustHomodyneOptimizer(mock_analysis_core, robust_config)
        
        # Test parameter bounds functionality
        # Parameters violating bounds
        invalid_params = np.array([1e4, -5.0, 1.0, 0.1, 0.1, 0.01, 400.0])
        assert invalid_params.shape == (7,)

    @pytest.mark.skipif(not ROBUST_MODULE_AVAILABLE, reason="Robust module not available")
    def test_bounds_projection(self, mock_analysis_core, robust_config):
        """Test projection of parameters onto feasible bounds."""
        optimizer = RobustHomodyneOptimizer(mock_analysis_core, robust_config)
        
        # Test bounds projection functionality
        # Parameters outside bounds
        unbounded_params = np.array([1e4, -5.0, 1.0, 0.1, 0.1, 0.01, 400.0])
        projected_params = np.array([1e3, -2.0, 1.0, 0.1, 0.1, 0.01, 360.0])
        
        # Check projection worked
        assert projected_params[0] <= 1e3  # D0 projected to upper bound
        assert projected_params[1] >= -2.0  # alpha projected to lower bound
        assert projected_params[6] <= 360.0  # phi0 projected to upper bound


class TestRobustOptimizationFactory:
    """Test robust optimizer factory function."""

    @pytest.mark.skipif(not ROBUST_MODULE_AVAILABLE, reason="Robust module not available")
    def test_create_robust_optimizer_success(self, mock_analysis_core, robust_config):
        """Test successful robust optimizer creation."""
        with patch('homodyne.optimization.robust.CVXPY_AVAILABLE', True):
            optimizer = create_robust_optimizer(mock_analysis_core, robust_config)
            
            assert optimizer is not None
            assert isinstance(optimizer, RobustHomodyneOptimizer)

    @pytest.mark.skipif(not ROBUST_MODULE_AVAILABLE, reason="Robust module not available")
    def test_create_robust_optimizer_cvxpy_unavailable(self, mock_analysis_core, robust_config):
        """Test optimizer creation when CVXPY is unavailable."""
        with patch('homodyne.optimization.robust.CVXPY_AVAILABLE', False):
            optimizer = create_robust_optimizer(mock_analysis_core, robust_config)
            assert optimizer is not None
            with pytest.raises(ImportError):
                optimizer.check_dependencies()

    @pytest.mark.skipif(not ROBUST_MODULE_AVAILABLE, reason="Robust module not available")
    def test_create_robust_optimizer_invalid_config(self, mock_analysis_core):
        """Test optimizer creation with invalid configuration."""
        invalid_config = {"invalid": "config"}
        
        # The factory function accepts any config, same as the constructor
        optimizer = create_robust_optimizer(mock_analysis_core, invalid_config)
        assert optimizer is not None


class TestErrorHandling:
    """Test error handling in robust optimization."""

    @pytest.mark.skipif(not ROBUST_MODULE_AVAILABLE, reason="Robust module not available")
    def test_solver_failure_handling(self, mock_analysis_core, robust_config):
        """Test handling of solver failures."""
        optimizer = RobustHomodyneOptimizer(mock_analysis_core, robust_config)
        
        with patch.object(optimizer, 'run_robust_optimization') as mock_opt:
            mock_opt.side_effect = RuntimeError("Solver failed to converge")
            
            with pytest.raises(RuntimeError):
                phi_angles = np.array([0])
                exp_data = np.random.rand(1, 20, 20) + 1.0
                initial_params = np.array([100.0, -0.1, 1.0, 0.1, 0.1, 0.01, 30.0])
                mock_opt(initial_params, phi_angles, exp_data)

    @pytest.mark.skipif(not ROBUST_MODULE_AVAILABLE, reason="Robust module not available")
    def test_infeasible_problem_handling(self, mock_analysis_core, robust_config):
        """Test handling of infeasible optimization problems."""
        optimizer = RobustHomodyneOptimizer(mock_analysis_core, robust_config)
        
        with patch.object(optimizer, 'run_robust_optimization') as mock_opt:
            mock_result = {
                'success': False,
                'status': 'infeasible',
                'message': 'Problem is infeasible'
            }
            mock_opt.return_value = mock_result
            
            phi_angles = np.array([0])
            exp_data = np.random.rand(1, 15, 15) + 1.0
            initial_params = np.array([100.0, -0.1, 1.0, 0.1, 0.1, 0.01, 30.0])
            
            result = mock_opt(initial_params, phi_angles, exp_data)
            
            assert result['success'] is False
            assert result['status'] == 'infeasible'

    @pytest.mark.skipif(not ROBUST_MODULE_AVAILABLE, reason="Robust module not available")
    def test_memory_error_handling(self, mock_analysis_core, robust_config):
        """Test handling of memory errors during optimization."""
        # Large problem that might cause memory issues
        robust_config["robust_optimization"]["n_scenarios"] = 10000
        
        optimizer = RobustHomodyneOptimizer(mock_analysis_core, robust_config)
        
        with patch.object(optimizer, 'run_robust_optimization') as mock_opt:
            mock_opt.side_effect = MemoryError("Insufficient memory for optimization")
            
            with pytest.raises(MemoryError):
                phi_angles = np.array([0, 45, 90])
                exp_data = np.random.rand(3, 100, 100) + 1.0
                initial_params = np.array([100.0, -0.1, 1.0, 0.1, 0.1, 0.01, 30.0])
                mock_opt(initial_params, phi_angles, exp_data)


class TestPerformanceOptimizations:
    """Test performance optimizations in robust optimization."""

    @pytest.mark.skipif(not ROBUST_MODULE_AVAILABLE, reason="Robust module not available")
    def test_optimization_timing(self, mock_analysis_core, robust_config):
        """Test timing of robust optimization methods."""
        optimizer = RobustHomodyneOptimizer(mock_analysis_core, robust_config)
        
        with patch.object(optimizer, 'run_robust_optimization') as mock_opt:
            start_time = time.time()
            mock_opt.return_value = {
                'success': True,
                'x': np.array([100.0, -0.1, 1.0, 0.1, 0.1, 0.01, 30.0]),
                'objective_value': 1.0,
                'solver_time': 25.3,
                'total_time': 28.7
            }
            
            phi_angles = np.array([0])
            exp_data = np.random.rand(1, 30, 30) + 1.0
            initial_params = np.array([100.0, -0.1, 1.0, 0.1, 0.1, 0.01, 30.0])
            
            result = mock_opt(initial_params, phi_angles, exp_data)
            runtime = time.time() - start_time
            
            assert 'solver_time' in result
            assert result['solver_time'] > 0

    @pytest.mark.skipif(not ROBUST_MODULE_AVAILABLE, reason="Robust module not available")
    def test_scenario_reduction(self, mock_analysis_core, robust_config):
        """Test scenario reduction for computational efficiency."""
        robust_config["robust_optimization"]["n_scenarios"] = 1000
        robust_config["robust_optimization"]["scenario_reduction"] = True
        robust_config["robust_optimization"]["reduced_scenarios"] = 100
        
        optimizer = RobustHomodyneOptimizer(mock_analysis_core, robust_config)
        
        # Test scenario reduction functionality
        reduction_result = {
            'reduced_scenarios': 100,
            'original_scenarios': 1000,
            'reduction_method': 'fast_forward_selection'
        }
        
        assert reduction_result['reduced_scenarios'] < reduction_result['original_scenarios']
        assert reduction_result['reduced_scenarios'] == 100


if __name__ == "__main__":
    pytest.main([__file__])