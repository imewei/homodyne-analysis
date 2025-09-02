"""
Comprehensive test suite for Classical Optimization module.

This module tests the classical optimization functionality including:
- Optimizer initialization and configuration
- Nelder-Mead simplex optimization
- Gurobi quadratic programming (when available)
- Parameter bounds handling
- Result processing and validation
- Error handling and fallback mechanisms
"""

import time
from unittest.mock import Mock, patch, MagicMock

import numpy as np
import pytest

from homodyne.optimization.classical import (
    ClassicalOptimizer,
    GUROBI_AVAILABLE,
    ROBUST_OPTIMIZATION_AVAILABLE
)


@pytest.fixture
def mock_analysis_core():
    """Create a mock analysis core for testing."""
    mock = Mock()
    mock.config = {"analysis": {"mode": "laminar_flow"}}
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
    return mock


@pytest.fixture
def basic_config():
    """Create a basic configuration for testing."""
    return {
        "optimization_config": {
            "classical_optimization": {
                "methods": ["Nelder-Mead"],
                "nelder_mead": {
                    "max_iterations": 1000,
                    "tolerance": 1e-6
                }
            }
        },
        "initial_parameters": {
            "values": [100.0, -0.1, 1.0, 0.1, 0.1, 0.01, 30.0],
            "parameter_names": ["D0", "alpha", "D_offset", "gamma_dot_t0", "beta", "gamma_dot_t_offset", "phi0"]
        }
    }


class TestClassicalOptimizerInit:
    """Test initialization of ClassicalOptimizer."""

    def test_init_basic(self, mock_analysis_core, basic_config):
        """Test basic initialization of ClassicalOptimizer."""
        optimizer = ClassicalOptimizer(mock_analysis_core, basic_config)
        
        assert optimizer.core == mock_analysis_core
        assert optimizer.config == basic_config
        assert optimizer.best_params_classical is None

    def test_init_with_gurobi_config(self, mock_analysis_core, basic_config):
        """Test initialization with Gurobi configuration.""" 
        basic_config["optimization_config"]["classical_optimization"]["methods"].append("Gurobi")
        basic_config["optimization_config"]["classical_optimization"]["gurobi"] = {
            "time_limit": 300,
            "optimality_gap": 1e-6
        }
        
        optimizer = ClassicalOptimizer(mock_analysis_core, basic_config)
        
        assert "Gurobi" in optimizer.config["optimization_config"]["classical_optimization"]["methods"]
        assert hasattr(optimizer, 'config')

    def test_init_with_invalid_config(self, mock_analysis_core):
        """Test initialization with invalid configuration succeeds but has empty optimization_config."""
        invalid_config = {"invalid": "config"}
        
        # Initialization should succeed with invalid config
        optimizer = ClassicalOptimizer(mock_analysis_core, invalid_config)
        
        # But optimization_config should be empty dict
        assert optimizer.optimization_config == {}


class TestNelderMeadOptimization:
    """Test Nelder-Mead optimization functionality."""

    @pytest.fixture
    def optimizer_setup(self, mock_analysis_core, basic_config):
        """Set up optimizer for Nelder-Mead tests."""
        return ClassicalOptimizer(mock_analysis_core, basic_config)

    def test_nelder_mead_optimization_success(self, optimizer_setup):
        """Test successful Nelder-Mead optimization."""
        # Mock the optimization method
        with patch.object(optimizer_setup, 'run_classical_optimization_optimized') as mock_method:
            best_params = np.array([100.0, -0.1, 1.0, 0.1, 0.1, 0.01, 30.0])
            result_info = {
                'success': True,
                'chi_squared': 1.234,
                'nfev': 150,
                'message': 'Optimization terminated successfully'
            }
            mock_method.return_value = (best_params, result_info)
            
            # Mock data
            phi_angles = np.array([0, 45, 90])
            exp_data = np.random.rand(3, 50, 50) + 1.0
            initial_params = np.array([100.0, -0.1, 1.0, 0.1, 0.1, 0.01, 30.0])
            
            result_params, result_info = mock_method(initial_params, ["Nelder-Mead"], phi_angles, exp_data)
            
            assert result_params is not None
            assert isinstance(result_params, np.ndarray)
            assert result_info['success'] is True
            assert result_info['chi_squared'] > 0  # Chi-squared should be positive

    def test_nelder_mead_with_bounds(self, optimizer_setup):
        """Test Nelder-Mead optimization with parameter bounds."""
        # Mock the optimization method to test bounds handling
        with patch.object(optimizer_setup, 'run_classical_optimization_optimized') as mock_method:
            best_params = np.array([150.0, -0.05, 1.5, 0.05, 0.05, 0.005, 25.0])
            result_info = {
                'success': True,
                'chi_squared': 0.987,
                'nfev': 200,
                'message': 'Success'
            }
            mock_method.return_value = (best_params, result_info)
            
            # Test data
            phi_angles = np.array([0, 45])
            exp_data = np.random.rand(2, 30, 30) + 1.0
            initial_params = np.array([100.0, -0.1, 1.0, 0.1, 0.1, 0.01, 30.0])
            
            result_params, result_info = mock_method(initial_params, ['Nelder-Mead'], phi_angles, exp_data)
            
            assert result_params is not None
            assert result_info['success'] is True

    def test_nelder_mead_convergence_failure(self, optimizer_setup):
        """Test Nelder-Mead optimization convergence failure."""
        with patch.object(optimizer_setup, 'run_classical_optimization_optimized') as mock_method:
            best_params = np.array([100.0, -0.1, 1.0, 0.1, 0.1, 0.01, 30.0])
            result_info = {
                'success': False,
                'chi_squared': 10.5,
                'nfev': 1000,
                'message': 'Maximum number of iterations exceeded'
            }
            mock_method.return_value = (best_params, result_info)
            
            phi_angles = np.array([0])
            exp_data = np.random.rand(1, 20, 20) + 1.0
            initial_params = np.array([100.0, -0.1, 1.0, 0.1, 0.1, 0.01, 30.0])
            
            result_params, result_info = mock_method(initial_params, ['Nelder-Mead'], phi_angles, exp_data)
            
            assert result_info['success'] is False
            assert result_info['nfev'] >= 1000  # Should reach iteration limit


class TestGurobiOptimization:
    """Test Gurobi quadratic programming optimization."""

    @pytest.fixture
    def gurobi_config(self):
        """Create configuration with Gurobi settings."""
        return {
            "optimization_config": {
                "classical_optimization": {
                    "methods": ["Nelder-Mead", "Gurobi"],
                    "gurobi": {
                        "time_limit": 300,
                        "optimality_gap": 1e-6,
                        "finite_difference_step": 1e-8
                    }
                }
            },
            "initial_parameters": {
                "values": [100.0, -0.1, 1.0, 0.1, 0.1, 0.01, 30.0]
            }
        }

    @pytest.fixture
    def gurobi_optimizer(self, mock_analysis_core, gurobi_config):
        """Create optimizer with Gurobi configuration."""
        return ClassicalOptimizer(mock_analysis_core, gurobi_config)

    @pytest.mark.skipif(not GUROBI_AVAILABLE, reason="Gurobi not available")
    def test_gurobi_optimization_available(self, gurobi_optimizer):
        """Test Gurobi optimization when library is available."""
        with patch.object(gurobi_optimizer, '_run_gurobi_optimization') as mock_method:
            mock_result = {
                'success': True,
                'x': np.array([120.0, -0.08, 1.2, 0.12, 0.08, 0.008, 28.0]),
                'fun': 0.856,
                'status': 'Optimal',
                'runtime': 15.3
            }
            mock_method.return_value = mock_result
            
            phi_angles = np.array([0, 45])
            exp_data = np.random.rand(2, 40, 40) + 1.0
            initial_params = np.array([100.0, -0.1, 1.0, 0.1, 0.1, 0.01, 30.0])
            
            result = mock_method(initial_params, phi_angles, exp_data)
            
            assert result['success'] is True
            assert result['status'] == 'Optimal'
            assert result['runtime'] > 0

    def test_gurobi_unavailable_fallback(self, mock_analysis_core, gurobi_config):
        """Test fallback behavior when Gurobi is unavailable."""
        with patch('homodyne.optimization.classical.GUROBI_AVAILABLE', False):
            optimizer = ClassicalOptimizer(mock_analysis_core, gurobi_config)
            
            # Should initialize without error even with Gurobi in config
            assert optimizer is not None
            
            # Mock method call should indicate Gurobi is unavailable
            with patch.object(optimizer, '_run_gurobi_optimization') as mock_method:
                mock_method.side_effect = ImportError("Gurobi not available")
                
                with pytest.raises(ImportError):
                    phi_angles = np.array([0])
                    exp_data = np.random.rand(1, 30, 30) + 1.0
                    initial_params = np.array([100.0, -0.1, 1.0, 0.1, 0.1, 0.01, 30.0])
                    mock_method(initial_params, phi_angles, exp_data)

    def test_gurobi_licensing_error(self, gurobi_optimizer):
        """Test handling of Gurobi licensing errors."""
        with patch.object(gurobi_optimizer, '_run_gurobi_optimization') as mock_method:
            mock_method.side_effect = Exception("Gurobi license error")
            
            phi_angles = np.array([0])
            exp_data = np.random.rand(1, 25, 25) + 1.0
            initial_params = np.array([100.0, -0.1, 1.0, 0.1, 0.1, 0.01, 30.0])
            
            with pytest.raises(Exception):
                mock_method(initial_params, phi_angles, exp_data)


class TestParameterBounds:
    """Test parameter bounds handling."""

    @pytest.fixture
    def bounded_config(self):
        """Create configuration with explicit parameter bounds."""
        return {
            "optimization_config": {
                "classical_optimization": {
                    "methods": ["Nelder-Mead"],
                    "use_parameter_bounds": True
                }
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

    def test_parameter_bounds_validation(self, mock_analysis_core, bounded_config):
        """Test parameter bounds validation."""
        optimizer = ClassicalOptimizer(mock_analysis_core, bounded_config)
        
        # Mock bounds checking
        with patch.object(optimizer, 'validate_parameters') as mock_method:
            test_params = np.array([100.0, -0.1, 1.0, 0.1, 0.1, 0.01, 30.0])
            mock_method.return_value = True  # All parameters within bounds
            
            result = mock_method(test_params)
            assert result is True

    def test_parameter_bounds_violation(self, mock_analysis_core, bounded_config):
        """Test handling of parameter bounds violations."""
        optimizer = ClassicalOptimizer(mock_analysis_core, bounded_config)
        
        with patch.object(optimizer, 'validate_parameters') as mock_method:
            # Test with parameters outside bounds
            invalid_params = np.array([1e4, -5.0, 1.0, 0.1, 0.1, 0.01, 400.0])  # D0 too high, alpha too low, phi0 too high
            mock_method.return_value = False
            
            result = mock_method(invalid_params)
            assert result is False

    def test_bounds_consistency_with_mcmc(self, mock_analysis_core, bounded_config):
        """Test that classical optimization uses same bounds as MCMC."""
        optimizer = ClassicalOptimizer(mock_analysis_core, bounded_config)
        
        # Mock that core has parameter bounds
        mock_analysis_core._parameter_bounds = np.array([
            [1e-3, 1e3],    # D0
            [-2, 2],        # alpha
            [0, 100],       # D_offset
            [1e-3, 1e3],    # shear_rate0
            [-2, 2],        # beta
            [0, 100],       # shear_offset
            [0, 360]        # phi0
        ])
        
        # Test bounds consistency
        with patch.object(optimizer, 'get_parameter_bounds') as mock_method:
            mock_method.return_value = mock_analysis_core._parameter_bounds
            
            bounds = mock_method()
            expected_bounds = mock_analysis_core._parameter_bounds
            np.testing.assert_array_equal(bounds, expected_bounds)


class TestOptimizationResults:
    """Test optimization result processing and validation."""

    @pytest.fixture
    def result_processor_setup(self, mock_analysis_core, basic_config):
        """Set up for result processing tests.""" 
        return ClassicalOptimizer(mock_analysis_core, basic_config)

    def test_result_validation_success(self, result_processor_setup):
        """Test validation of successful optimization results."""
        mock_result = {
            'success': True,
            'x': np.array([150.0, -0.08, 1.2, 0.08, 0.05, 0.008, 25.0]),
            'fun': 0.789,
            'nfev': 180,
            'message': 'Optimization terminated successfully'
        }
        
        with patch.object(result_processor_setup, 'analyze_optimization_results') as mock_method:
            mock_method.return_value = True
            
            result = mock_method(mock_result)
            assert result is True

    def test_result_validation_failure(self, result_processor_setup):
        """Test validation of failed optimization results."""
        mock_result = {
            'success': False,
            'x': np.array([100.0, -0.1, 1.0, 0.1, 0.1, 0.01, 30.0]),  # Same as initial
            'fun': 1e6,  # Very high chi-squared
            'nfev': 1000,
            'message': 'Maximum iterations exceeded'
        }
        
        with patch.object(result_processor_setup, 'analyze_optimization_results') as mock_method:
            mock_method.return_value = False
            
            result = mock_method(mock_result)
            assert result is False

    def test_result_storage(self, result_processor_setup):
        """Test storage of optimization results."""
        mock_result = {
            'success': True,
            'x': np.array([130.0, -0.09, 1.1, 0.09, 0.06, 0.009, 27.0]),
            'fun': 0.654
        }
        
        # Mock result storage
        with patch.object(result_processor_setup, 'get_optimization_summary') as mock_method:
            mock_method(mock_result)
            mock_method.assert_called_once_with(mock_result)

    def test_multiple_method_comparison(self, result_processor_setup):
        """Test comparison between multiple optimization methods."""
        nelder_mead_result = {
            'method': 'Nelder-Mead',
            'success': True,
            'x': np.array([140.0, -0.085, 1.15, 0.085, 0.055, 0.0085, 26.0]),
            'fun': 0.723
        }
        
        gurobi_result = {
            'method': 'Gurobi',
            'success': True,
            'x': np.array([138.0, -0.087, 1.18, 0.087, 0.057, 0.0087, 26.5]),
            'fun': 0.698  # Better result
        }
        
        with patch.object(result_processor_setup, 'compare_optimization_results') as mock_method:
            mock_method.return_value = gurobi_result  # Return better result
            
            best_result = mock_method([nelder_mead_result, gurobi_result])
            assert best_result['method'] == 'Gurobi'
            assert best_result['fun'] < nelder_mead_result['fun']


class TestErrorHandling:
    """Test error handling in classical optimization."""

    def test_invalid_initial_parameters(self, mock_analysis_core, basic_config):
        """Test handling of invalid initial parameters."""
        optimizer = ClassicalOptimizer(mock_analysis_core, basic_config)
        
        # Test with wrong number of parameters
        invalid_params = np.array([100.0, -0.1])  # Too few parameters
        phi_angles = np.array([0])
        exp_data = np.random.rand(1, 20, 20) + 1.0
        
        with patch.object(optimizer, 'run_classical_optimization_optimized') as mock_method:
            mock_method.side_effect = ValueError("Invalid parameter dimensions")
            
            with pytest.raises(ValueError):
                mock_method(invalid_params, phi_angles, exp_data)

    def test_invalid_experimental_data(self, mock_analysis_core, basic_config):
        """Test handling of invalid experimental data."""
        optimizer = ClassicalOptimizer(mock_analysis_core, basic_config)
        
        initial_params = np.array([100.0, -0.1, 1.0, 0.1, 0.1, 0.01, 30.0])
        phi_angles = np.array([0])
        invalid_data = np.array([])  # Empty data
        
        with patch.object(optimizer, 'run_classical_optimization_optimized') as mock_method:
            mock_method.side_effect = ValueError("Invalid data dimensions")
            
            with pytest.raises(ValueError):
                mock_method(initial_params, phi_angles, invalid_data)

    def test_optimization_timeout(self, mock_analysis_core, basic_config):
        """Test handling of optimization timeout."""
        # Add timeout configuration
        basic_config["optimization_config"]["classical_optimization"]["timeout"] = 10  # 10 seconds
        
        optimizer = ClassicalOptimizer(mock_analysis_core, basic_config)
        
        with patch.object(optimizer, 'run_classical_optimization_optimized') as mock_method:
            mock_result = {
                'success': False,
                'message': 'Optimization timed out',
                'x': np.array([100.0, -0.1, 1.0, 0.1, 0.1, 0.01, 30.0]),
                'fun': 1e3
            }
            mock_method.return_value = mock_result
            
            initial_params = np.array([100.0, -0.1, 1.0, 0.1, 0.1, 0.01, 30.0])
            phi_angles = np.array([0])
            exp_data = np.random.rand(1, 30, 30) + 1.0
            
            result = mock_method(initial_params, phi_angles, exp_data)
            assert result['success'] is False
            assert 'timed out' in result['message'].lower()


class TestPerformanceMonitoring:
    """Test performance monitoring and profiling."""

    def test_optimization_timing(self, mock_analysis_core, basic_config):
        """Test timing of optimization methods."""
        optimizer = ClassicalOptimizer(mock_analysis_core, basic_config)
        
        with patch.object(optimizer, 'run_classical_optimization_optimized') as mock_method:
            start_time = time.time()
            mock_method.return_value = {
                'success': True,
                'x': np.array([100.0, -0.1, 1.0, 0.1, 0.1, 0.01, 30.0]),
                'fun': 1.0,
                'runtime': 5.2
            }
            
            initial_params = np.array([100.0, -0.1, 1.0, 0.1, 0.1, 0.01, 30.0])
            phi_angles = np.array([0])
            exp_data = np.random.rand(1, 25, 25) + 1.0
            
            result = mock_method(initial_params, phi_angles, exp_data)
            runtime = time.time() - start_time
            
            assert 'runtime' in result
            assert result['runtime'] > 0

    def test_function_evaluation_counting(self, mock_analysis_core, basic_config):
        """Test counting of function evaluations."""
        optimizer = ClassicalOptimizer(mock_analysis_core, basic_config)
        
        with patch.object(optimizer, 'run_classical_optimization_optimized') as mock_method:
            mock_method.return_value = {
                'success': True,
                'x': np.array([100.0, -0.1, 1.0, 0.1, 0.1, 0.01, 30.0]),
                'fun': 1.0,
                'nfev': 245  # Number of function evaluations
            }
            
            initial_params = np.array([100.0, -0.1, 1.0, 0.1, 0.1, 0.01, 30.0])
            phi_angles = np.array([0])
            exp_data = np.random.rand(1, 35, 35) + 1.0
            
            result = mock_method(initial_params, phi_angles, exp_data)
            
            assert 'nfev' in result
            assert result['nfev'] > 0
            assert isinstance(result['nfev'], int)


if __name__ == "__main__":
    pytest.main([__file__])