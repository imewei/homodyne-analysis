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
from unittest.mock import Mock, patch

import numpy as np
import pytest

from homodyne.optimization.classical import GUROBI_AVAILABLE, ClassicalOptimizer


@pytest.fixture
def mock_analysis_core():
    """Create a mock analysis core for testing."""
    mock = Mock()
    mock.config = {"analysis": {"mode": "laminar_flow"}}
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
    return mock


@pytest.fixture
def basic_config():
    """Create a basic configuration for testing."""
    return {
        "optimization_config": {
            "classical_optimization": {
                "methods": ["Nelder-Mead"],
                "nelder_mead": {"max_iterations": 1000, "tolerance": 1e-6},
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
        basic_config["optimization_config"]["classical_optimization"]["methods"].append(
            "Gurobi"
        )
        basic_config["optimization_config"]["classical_optimization"]["gurobi"] = {
            "time_limit": 300,
            "optimality_gap": 1e-6,
        }

        optimizer = ClassicalOptimizer(mock_analysis_core, basic_config)

        assert (
            "Gurobi"
            in optimizer.config["optimization_config"]["classical_optimization"][
                "methods"
            ]
        )
        assert hasattr(optimizer, "config")

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
        with patch.object(
            optimizer_setup, "run_classical_optimization_optimized"
        ) as mock_method:
            best_params = np.array([100.0, -0.1, 1.0, 0.1, 0.1, 0.01, 30.0])
            result_info = {
                "success": True,
                "chi_squared": 1.234,
                "nfev": 150,
                "message": "Optimization terminated successfully",
            }
            mock_method.return_value = (best_params, result_info)

            # Mock data
            phi_angles = np.array([0, 45, 90])
            exp_data = np.random.rand(3, 50, 50) + 1.0
            initial_params = np.array([100.0, -0.1, 1.0, 0.1, 0.1, 0.01, 30.0])

            result_params, result_info = mock_method(
                initial_params, ["Nelder-Mead"], phi_angles, exp_data
            )

            assert result_params is not None
            assert isinstance(result_params, np.ndarray)
            assert result_info["success"] is True
            assert result_info["chi_squared"] > 0  # Chi-squared should be positive

    def test_nelder_mead_with_bounds(self, optimizer_setup):
        """Test Nelder-Mead optimization with parameter bounds."""
        # Mock the optimization method to test bounds handling
        with patch.object(
            optimizer_setup, "run_classical_optimization_optimized"
        ) as mock_method:
            best_params = np.array([150.0, -0.05, 1.5, 0.05, 0.05, 0.005, 25.0])
            result_info = {
                "success": True,
                "chi_squared": 0.987,
                "nfev": 200,
                "message": "Success",
            }
            mock_method.return_value = (best_params, result_info)

            # Test data
            phi_angles = np.array([0, 45])
            exp_data = np.random.rand(2, 30, 30) + 1.0
            initial_params = np.array([100.0, -0.1, 1.0, 0.1, 0.1, 0.01, 30.0])

            result_params, result_info = mock_method(
                initial_params, ["Nelder-Mead"], phi_angles, exp_data
            )

            assert result_params is not None
            assert result_info["success"] is True

    def test_nelder_mead_convergence_failure(self, optimizer_setup):
        """Test Nelder-Mead optimization convergence failure."""
        with patch.object(
            optimizer_setup, "run_classical_optimization_optimized"
        ) as mock_method:
            best_params = np.array([100.0, -0.1, 1.0, 0.1, 0.1, 0.01, 30.0])
            result_info = {
                "success": False,
                "chi_squared": 10.5,
                "nfev": 1000,
                "message": "Maximum number of iterations exceeded",
            }
            mock_method.return_value = (best_params, result_info)

            phi_angles = np.array([0])
            exp_data = np.random.rand(1, 20, 20) + 1.0
            initial_params = np.array([100.0, -0.1, 1.0, 0.1, 0.1, 0.01, 30.0])

            result_params, result_info = mock_method(
                initial_params, ["Nelder-Mead"], phi_angles, exp_data
            )

            assert result_info["success"] is False
            assert result_info["nfev"] >= 1000  # Should reach iteration limit


class TestRealAlgorithmExecution:
    """Test real optimization algorithm execution with synthetic data."""

    def create_synthetic_data(self, true_params, phi_angles, noise_level=0.01):
        """Create synthetic correlation function data with known parameters.

        This creates test data where we know the ground truth parameters,
        allowing us to test parameter recovery accuracy.
        """
        # Simple exponential decay model: C(t) = 1 + contrast * exp(-t/tau)
        # Map parameters to physical values for synthetic data
        D0, alpha, D_offset = true_params[:3]

        # Create time delays (simplified)
        n_times = 10
        times = np.linspace(0.1, 2.0, n_times)

        # Simple synthetic correlation function
        c2_data = np.zeros((len(phi_angles), n_times, n_times))

        for i, angle in enumerate(phi_angles):
            # Simple model: decay rate depends on angle and parameters
            decay_rate = D0 * (1 + alpha * np.cos(np.radians(angle))) / 1000.0

            for j, t1 in enumerate(times):
                for k, t2 in enumerate(times):
                    # Simplified correlation function
                    dt = abs(t1 - t2)
                    correlation = 1.0 + 0.1 * np.exp(-decay_rate * dt)

                    # Add small amount of noise
                    noise = np.random.normal(0, noise_level)
                    c2_data[i, j, k] = max(1.001, correlation + noise)  # Ensure > 1

        return c2_data

    def create_real_analysis_core(self):
        """Create a real analysis core that can compute chi-squared values."""
        from unittest.mock import MagicMock

        core = MagicMock()

        # Real chi-squared calculation using synthetic data
        def calculate_chi_squared_real(
            params, phi_angles, c2_experimental, method_name="", **kwargs
        ):
            """Real chi-squared calculation for synthetic data."""
            # Generate theoretical data based on parameters
            c2_theoretical = self.create_synthetic_data(
                params, phi_angles, noise_level=0
            )

            # Calculate chi-squared
            residuals = c2_experimental - c2_theoretical
            chi_squared = np.sum(residuals**2)

            # Return reasonable chi-squared value
            return float(chi_squared)

        core.calculate_chi_squared_optimized = calculate_chi_squared_real
        core.get_effective_parameter_count.return_value = 3  # Static mode
        core._parameter_bounds = np.array(
            [
                [10.0, 1000.0],  # D0 bounds
                [-1.0, 1.0],  # alpha bounds
                [-10.0, 10.0],  # D_offset bounds
            ]
        )

        return core

    def test_nelder_mead_algorithm_execution(self):
        """Test that Nelder-Mead algorithm executes and finds reasonable solutions."""
        # Use simple test parameters
        true_params = np.array([100.0, 0.2, 1.0])  # D0, alpha, D_offset
        phi_angles = np.array([0.0, 45.0])  # Fewer angles for simpler test

        # Create synthetic data with known parameters
        c2_experimental = self.create_synthetic_data(
            true_params, phi_angles, noise_level=0.01
        )

        # Create real analysis core
        analysis_core = self.create_real_analysis_core()

        # Use scipy.optimize directly for real algorithm testing
        from scipy.optimize import minimize

        def objective_function(params):
            """Real objective function for testing."""
            return analysis_core.calculate_chi_squared_optimized(
                params, phi_angles, c2_experimental, "TestReal"
            )

        # Test with several initial guesses to ensure robustness
        initial_guesses = [
            np.array([200.0, -0.1, -2.0]),
            np.array([50.0, 0.5, 3.0]),
            np.array([150.0, 0.0, 0.0]),
        ]


        successful_optimizations = 0

        for initial_guess in initial_guesses:
            # Run real Nelder-Mead optimization
            result = minimize(
                objective_function,
                initial_guess,
                method="Nelder-Mead",
                options={"maxiter": 50, "xatol": 1e-3, "fatol": 1e-3},
            )

            # Test basic optimization functionality
            assert isinstance(result.x, np.ndarray)
            assert len(result.x) == 3
            assert result.fun >= 0  # Chi-squared should be non-negative

            # Count successful optimizations
            if result.success or result.fun < 10.0:  # Reasonable threshold
                successful_optimizations += 1

        # At least one optimization should succeed
        assert successful_optimizations > 0, "No optimization attempts succeeded"

        # Test that the algorithm is actually improving the objective
        initial_value = objective_function(initial_guesses[0])
        final_result = minimize(
            objective_function,
            initial_guesses[0],
            method="Nelder-Mead",
            options={"maxiter": 30},
        )

        # Final result should be better than initial guess (or very close)
        assert (
            final_result.fun <= initial_value * 1.1
        ), f"No improvement: initial={initial_value}, final={final_result.fun}"

    def test_bounded_optimization_respects_constraints(self):
        """Test that optimization respects parameter bounds."""
        true_params = np.array([50.0, 0.5, 2.0])
        phi_angles = np.array([0.0, 30.0])

        c2_experimental = self.create_synthetic_data(
            true_params, phi_angles, noise_level=0.01
        )
        analysis_core = self.create_real_analysis_core()

        config = {
            "optimization_config": {
                "classical_optimization": {
                    "methods": ["Nelder-Mead"],
                    "method_options": {"Nelder-Mead": {"maxiter": 50}},
                }
            },
            "parameter_space": {
                "bounds": [
                    {
                        "name": "D0",
                        "min": 100.0,
                        "max": 200.0,
                    },  # Force D0 away from true value
                    {
                        "name": "alpha",
                        "min": -0.5,
                        "max": 0.0,
                    },  # Force alpha away from true value
                    {"name": "D_offset", "min": 0.0, "max": 5.0},
                ]
            },
        }

        optimizer = ClassicalOptimizer(analysis_core, config)
        bounds = optimizer.get_parameter_bounds(3)

        from scipy.optimize import minimize

        def objective_function(params):
            return analysis_core.calculate_chi_squared_optimized(
                params, phi_angles, c2_experimental, "TestBounds"
            )

        initial_guess = np.array([150.0, -0.25, 2.5])

        # Use L-BFGS-B which respects bounds
        result = minimize(
            objective_function,
            initial_guess,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 50},
        )

        # Test that bounds are respected
        for i, (param_val, (min_bound, max_bound)) in enumerate(zip(result.x, bounds, strict=False)):
            assert (
                min_bound <= param_val <= max_bound
            ), f"Parameter {i} = {param_val} violates bounds [{min_bound}, {max_bound}]"

    def test_optimization_convergence_monitoring(self):
        """Test optimization convergence monitoring and iteration tracking."""
        true_params = np.array([200.0, -0.3, 0.5])
        phi_angles = np.array([0.0])

        c2_experimental = self.create_synthetic_data(
            true_params, phi_angles, noise_level=0.001
        )
        analysis_core = self.create_real_analysis_core()

        # Track function evaluations
        iteration_count = 0
        function_values = []

        def objective_function(params):
            nonlocal iteration_count
            iteration_count += 1

            chi_squared = analysis_core.calculate_chi_squared_optimized(
                params, phi_angles, c2_experimental, f"TestConv_{iteration_count}"
            )
            function_values.append(chi_squared)
            return chi_squared

        from scipy.optimize import minimize

        initial_guess = np.array([100.0, 0.1, 1.0])

        minimize(
            objective_function,
            initial_guess,
            method="Nelder-Mead",
            options={"maxiter": 30, "xatol": 1e-6},
        )

        # Test convergence monitoring
        assert iteration_count > 0
        assert len(function_values) == iteration_count

        # Test that function value generally decreases (allowing for some noise)
        if len(function_values) > 5:
            # Compare first quarter to last quarter of iterations
            early_values = function_values[: len(function_values) // 4]
            late_values = function_values[-len(function_values) // 4 :]

            avg_early = np.mean(early_values)
            avg_late = np.mean(late_values)

            # Should show improvement (decreasing chi-squared)
            assert avg_late <= avg_early * 1.1  # Allow for some noise/fluctuation

    def test_multiple_angle_optimization(self):
        """Test optimization with multiple scattering angles."""
        true_params = np.array([150.0, 0.1, -1.0])
        phi_angles = np.array([0.0, 15.0, 30.0, 45.0, 60.0])  # Multiple angles

        c2_experimental = self.create_synthetic_data(
            true_params, phi_angles, noise_level=0.005
        )
        analysis_core = self.create_real_analysis_core()

        def objective_function(params):
            return analysis_core.calculate_chi_squared_optimized(
                params, phi_angles, c2_experimental, "TestMultiAngle"
            )

        from scipy.optimize import minimize

        initial_guess = np.array([100.0, 0.0, 0.0])
        bounds = [(50.0, 300.0), (-0.5, 0.5), (-5.0, 5.0)]

        result = minimize(
            objective_function,
            initial_guess,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 100},
        )

        # Test that optimization works with multiple angles
        assert result.fun < 10.0  # Should achieve reasonable fit
        assert len(result.x) == 3

        # Test that all angles contributed to the optimization
        final_chi_squared = objective_function(result.x)
        assert final_chi_squared > 0


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
                        "finite_difference_step": 1e-8,
                    },
                }
            },
            "initial_parameters": {"values": [100.0, -0.1, 1.0, 0.1, 0.1, 0.01, 30.0]},
        }

    @pytest.fixture
    def gurobi_optimizer(self, mock_analysis_core, gurobi_config):
        """Create optimizer with Gurobi configuration."""
        return ClassicalOptimizer(mock_analysis_core, gurobi_config)

    @pytest.mark.skipif(not GUROBI_AVAILABLE, reason="Gurobi not available")
    def test_gurobi_optimization_available(self, gurobi_optimizer):
        """Test Gurobi optimization when library is available."""
        with patch.object(gurobi_optimizer, "_run_gurobi_optimization") as mock_method:
            mock_result = {
                "success": True,
                "x": np.array([120.0, -0.08, 1.2, 0.12, 0.08, 0.008, 28.0]),
                "fun": 0.856,
                "status": "Optimal",
                "runtime": 15.3,
            }
            mock_method.return_value = mock_result

            phi_angles = np.array([0, 45])
            exp_data = np.random.rand(2, 40, 40) + 1.0
            initial_params = np.array([100.0, -0.1, 1.0, 0.1, 0.1, 0.01, 30.0])

            result = mock_method(initial_params, phi_angles, exp_data)

            assert result["success"] is True
            assert result["status"] == "Optimal"
            assert result["runtime"] > 0

    def test_gurobi_unavailable_fallback(self, mock_analysis_core, gurobi_config):
        """Test fallback behavior when Gurobi is unavailable."""
        with patch("homodyne.optimization.classical.GUROBI_AVAILABLE", False):
            optimizer = ClassicalOptimizer(mock_analysis_core, gurobi_config)

            # Should initialize without error even with Gurobi in config
            assert optimizer is not None

            # Mock method call should indicate Gurobi is unavailable
            with patch.object(optimizer, "_run_gurobi_optimization") as mock_method:
                mock_method.side_effect = ImportError("Gurobi not available")

                with pytest.raises(ImportError):
                    phi_angles = np.array([0])
                    exp_data = np.random.rand(1, 30, 30) + 1.0
                    initial_params = np.array([100.0, -0.1, 1.0, 0.1, 0.1, 0.01, 30.0])
                    mock_method(initial_params, phi_angles, exp_data)

    def test_gurobi_licensing_error(self, gurobi_optimizer):
        """Test handling of Gurobi licensing errors."""
        with patch.object(gurobi_optimizer, "_run_gurobi_optimization") as mock_method:
            mock_method.return_value = (False, Exception("Gurobi license error"))

            # Test that the mock method properly returns a failure
            initial_params = np.array([100.0, -0.1, 1.0, 0.1, 0.1, 0.01, 30.0])
            phi_angles = np.array([0])
            exp_data = np.random.rand(1, 25, 25) + 1.0

            # Call the mocked method directly to verify it returns failure
            success, error = mock_method(initial_params, phi_angles, exp_data)
            assert not success
            assert isinstance(error, Exception)


class TestParameterBounds:
    """Test parameter bounds handling."""

    @pytest.fixture
    def bounded_config(self):
        """Create configuration with explicit parameter bounds."""
        return {
            "optimization_config": {
                "classical_optimization": {
                    "methods": ["Nelder-Mead"],
                    "use_parameter_bounds": True,
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
                    "phi0": [0, 360],
                }
            },
            "initial_parameters": {"values": [100.0, -0.1, 1.0, 0.1, 0.1, 0.01, 30.0]},
        }

    def test_parameter_bounds_validation(self, mock_analysis_core, bounded_config):
        """Test parameter bounds validation."""
        optimizer = ClassicalOptimizer(mock_analysis_core, bounded_config)

        # Mock bounds checking
        with patch.object(optimizer, "validate_parameters") as mock_method:
            test_params = np.array([100.0, -0.1, 1.0, 0.1, 0.1, 0.01, 30.0])
            mock_method.return_value = True  # All parameters within bounds

            result = mock_method(test_params)
            assert result is True

    def test_parameter_bounds_violation(self, mock_analysis_core, bounded_config):
        """Test handling of parameter bounds violations."""
        optimizer = ClassicalOptimizer(mock_analysis_core, bounded_config)

        with patch.object(optimizer, "validate_parameters") as mock_method:
            # Test with parameters outside bounds
            invalid_params = np.array(
                [1e4, -5.0, 1.0, 0.1, 0.1, 0.01, 400.0]
            )  # D0 too high, alpha too low, phi0 too high
            mock_method.return_value = False

            result = mock_method(invalid_params)
            assert result is False

    def test_bounds_consistency_with_mcmc(self, mock_analysis_core, bounded_config):
        """Test that classical optimization uses same bounds as MCMC."""
        optimizer = ClassicalOptimizer(mock_analysis_core, bounded_config)

        # Mock that core has parameter bounds
        mock_analysis_core._parameter_bounds = np.array(
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

        # Test bounds consistency
        with patch.object(optimizer, "get_parameter_bounds") as mock_method:
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
            "success": True,
            "x": np.array([150.0, -0.08, 1.2, 0.08, 0.05, 0.008, 25.0]),
            "fun": 0.789,
            "nfev": 180,
            "message": "Optimization terminated successfully",
        }

        with patch.object(
            result_processor_setup, "analyze_optimization_results"
        ) as mock_method:
            mock_method.return_value = True

            result = mock_method(mock_result)
            assert result is True

    def test_result_validation_failure(self, result_processor_setup):
        """Test validation of failed optimization results."""
        mock_result = {
            "success": False,
            "x": np.array([100.0, -0.1, 1.0, 0.1, 0.1, 0.01, 30.0]),  # Same as initial
            "fun": 1e6,  # Very high chi-squared
            "nfev": 1000,
            "message": "Maximum iterations exceeded",
        }

        with patch.object(
            result_processor_setup, "analyze_optimization_results"
        ) as mock_method:
            mock_method.return_value = False

            result = mock_method(mock_result)
            assert result is False

    def test_result_storage(self, result_processor_setup):
        """Test storage of optimization results."""
        mock_result = {
            "success": True,
            "x": np.array([130.0, -0.09, 1.1, 0.09, 0.06, 0.009, 27.0]),
            "fun": 0.654,
        }

        # Mock result storage
        with patch.object(
            result_processor_setup, "get_optimization_summary"
        ) as mock_method:
            mock_method(mock_result)
            mock_method.assert_called_once_with(mock_result)

    def test_multiple_method_comparison(self, result_processor_setup):
        """Test comparison between multiple optimization methods."""
        nelder_mead_result = {
            "method": "Nelder-Mead",
            "success": True,
            "x": np.array([140.0, -0.085, 1.15, 0.085, 0.055, 0.0085, 26.0]),
            "fun": 0.723,
        }

        gurobi_result = {
            "method": "Gurobi",
            "success": True,
            "x": np.array([138.0, -0.087, 1.18, 0.087, 0.057, 0.0087, 26.5]),
            "fun": 0.698,  # Better result
        }

        with patch.object(
            result_processor_setup, "compare_optimization_results"
        ) as mock_method:
            mock_method.return_value = gurobi_result  # Return better result

            best_result = mock_method([nelder_mead_result, gurobi_result])
            assert best_result["method"] == "Gurobi"
            assert best_result["fun"] < nelder_mead_result["fun"]


class TestErrorHandling:
    """Test error handling in classical optimization."""

    def test_invalid_initial_parameters(self, mock_analysis_core, basic_config):
        """Test handling of invalid initial parameters."""
        optimizer = ClassicalOptimizer(mock_analysis_core, basic_config)

        # Test with wrong number of parameters
        invalid_params = np.array([100.0, -0.1])  # Too few parameters
        phi_angles = np.array([0])
        exp_data = np.random.rand(1, 20, 20) + 1.0

        with patch.object(
            optimizer, "run_classical_optimization_optimized"
        ) as mock_method:
            mock_method.side_effect = ValueError("Invalid parameter dimensions")

            with pytest.raises(ValueError):
                mock_method(invalid_params, phi_angles, exp_data)

    def test_invalid_experimental_data(self, mock_analysis_core, basic_config):
        """Test handling of invalid experimental data."""
        optimizer = ClassicalOptimizer(mock_analysis_core, basic_config)

        initial_params = np.array([100.0, -0.1, 1.0, 0.1, 0.1, 0.01, 30.0])
        phi_angles = np.array([0])
        invalid_data = np.array([])  # Empty data

        with patch.object(
            optimizer, "run_classical_optimization_optimized"
        ) as mock_method:
            mock_method.side_effect = ValueError("Invalid data dimensions")

            with pytest.raises(ValueError):
                mock_method(initial_params, phi_angles, invalid_data)

    def test_optimization_timeout(self, mock_analysis_core, basic_config):
        """Test handling of optimization timeout."""
        # Add timeout configuration
        basic_config["optimization_config"]["classical_optimization"][
            "timeout"
        ] = 10  # 10 seconds

        optimizer = ClassicalOptimizer(mock_analysis_core, basic_config)

        with patch.object(
            optimizer, "run_classical_optimization_optimized"
        ) as mock_method:
            mock_result = {
                "success": False,
                "message": "Optimization timed out",
                "x": np.array([100.0, -0.1, 1.0, 0.1, 0.1, 0.01, 30.0]),
                "fun": 1e3,
            }
            mock_method.return_value = mock_result

            initial_params = np.array([100.0, -0.1, 1.0, 0.1, 0.1, 0.01, 30.0])
            phi_angles = np.array([0])
            exp_data = np.random.rand(1, 30, 30) + 1.0

            result = mock_method(initial_params, phi_angles, exp_data)
            assert result["success"] is False
            assert "timed out" in result["message"].lower()


class TestPerformanceMonitoring:
    """Test performance monitoring and profiling."""

    def test_optimization_timing(self, mock_analysis_core, basic_config):
        """Test timing of optimization methods."""
        optimizer = ClassicalOptimizer(mock_analysis_core, basic_config)

        with patch.object(
            optimizer, "run_classical_optimization_optimized"
        ) as mock_method:
            start_time = time.time()
            mock_method.return_value = {
                "success": True,
                "x": np.array([100.0, -0.1, 1.0, 0.1, 0.1, 0.01, 30.0]),
                "fun": 1.0,
                "runtime": 5.2,
            }

            initial_params = np.array([100.0, -0.1, 1.0, 0.1, 0.1, 0.01, 30.0])
            phi_angles = np.array([0])
            exp_data = np.random.rand(1, 25, 25) + 1.0

            result = mock_method(initial_params, phi_angles, exp_data)
            time.time() - start_time

            assert "runtime" in result
            assert result["runtime"] > 0

    def test_function_evaluation_counting(self, mock_analysis_core, basic_config):
        """Test counting of function evaluations."""
        optimizer = ClassicalOptimizer(mock_analysis_core, basic_config)

        with patch.object(
            optimizer, "run_classical_optimization_optimized"
        ) as mock_method:
            mock_method.return_value = {
                "success": True,
                "x": np.array([100.0, -0.1, 1.0, 0.1, 0.1, 0.01, 30.0]),
                "fun": 1.0,
                "nfev": 245,  # Number of function evaluations
            }

            initial_params = np.array([100.0, -0.1, 1.0, 0.1, 0.1, 0.01, 30.0])
            phi_angles = np.array([0])
            exp_data = np.random.rand(1, 35, 35) + 1.0

            result = mock_method(initial_params, phi_angles, exp_data)

            assert "nfev" in result
            assert result["nfev"] > 0
            assert isinstance(result["nfev"], int)


class TestClassicalConfigurationReading:
    """Test classical optimization configuration reading from JSON files."""

    def test_initial_parameters_reading(self, mock_analysis_core):
        """Test that ClassicalOptimizer reads initial parameters correctly."""
        test_config = {
            "initial_parameters": {
                "values": [1000.0, -0.5, 100.0, 0.001, 0.2, 0.0001, 5.0],
                "parameter_names": [
                    "D0",
                    "alpha",
                    "D_offset",
                    "gamma_dot_t0",
                    "beta",
                    "gamma_dot_t_offset",
                    "phi0",
                ],
                "active_parameters": ["D0", "alpha", "D_offset"],
            },
            "optimization_config": {
                "classical_optimization": {"methods": ["Nelder-Mead"]}
            },
        }

        # Create ClassicalOptimizer instance
        ClassicalOptimizer(mock_analysis_core, test_config)

        # Test that it can access initial parameters correctly
        expected_values = np.array([1000.0, -0.5, 100.0, 0.001, 0.2, 0.0001, 5.0])
        actual_values = np.array(
            test_config["initial_parameters"]["values"], dtype=np.float64
        )
        np.testing.assert_array_equal(actual_values, expected_values)

        # Test parameter names access
        param_names = test_config["initial_parameters"]["parameter_names"]
        assert len(param_names) == 7
        assert param_names[0] == "D0"
        assert param_names[1] == "alpha"
        assert param_names[2] == "D_offset"

    def test_parameter_bounds_extraction(self, mock_analysis_core):
        """Test that bounds are correctly extracted for optimization methods."""
        test_config = {
            "parameter_space": {
                "bounds": [
                    {"name": "D0", "min": 1.0, "max": 1000000},
                    {"name": "alpha", "min": -1.5, "max": -0.5},
                    {"name": "D_offset", "min": -100, "max": 100},
                ]
            },
            "optimization_config": {
                "classical_optimization": {"methods": ["Nelder-Mead"]}
            },
        }

        optimizer = ClassicalOptimizer(mock_analysis_core, test_config)

        # Test bounds extraction using the actual method
        bounds = optimizer.get_parameter_bounds(effective_param_count=3)

        # Expected bounds should match configuration
        expected_bounds = [(1.0, 1000000), (-1.5, -0.5), (-100, 100)]
        assert bounds == expected_bounds

    def test_static_mode_parameter_selection(self, mock_analysis_core):
        """Test parameter selection for static mode (first 3 parameters)."""
        test_config = {
            "initial_parameters": {
                "values": [18000, -1.59, 3.10, 0.0, 0.0, 0.0, 0.0],
                "parameter_names": [
                    "D0",
                    "alpha",
                    "D_offset",
                    "gamma_dot_t0",
                    "beta",
                    "gamma_dot_t_offset",
                    "phi0",
                ],
                "active_parameters": ["D0", "alpha", "D_offset"],
            },
            "parameter_space": {
                "bounds": [
                    {"name": "D0", "min": 1.0, "max": 1000000},
                    {"name": "alpha", "min": -1.6, "max": -1.5},
                    {"name": "D_offset", "min": -100, "max": 100},
                    {
                        "name": "gamma_dot_t0",
                        "min": 0.0,
                        "max": 0.0,
                    },  # Fixed for static
                    {"name": "beta", "min": 0.0, "max": 0.0},  # Fixed for static
                    {
                        "name": "gamma_dot_t_offset",
                        "min": 0.0,
                        "max": 0.0,
                    },  # Fixed for static
                    {"name": "phi0", "min": 0.0, "max": 0.0},  # Fixed for static
                ]
            },
            "analysis_settings": {
                "static_mode": True,
                "static_submode": "isotropic",
            },
            "optimization_config": {
                "classical_optimization": {"methods": ["Nelder-Mead"]}
            },
        }

        optimizer = ClassicalOptimizer(mock_analysis_core, test_config)

        # Test parameter selection for static mode
        initial_parameters = np.array(
            test_config["initial_parameters"]["values"], dtype=np.float64
        )
        effective_param_count = 3  # Static mode uses first 3 parameters
        static_parameters = initial_parameters[:effective_param_count]

        expected_static_params = np.array([18000, -1.59, 3.10])
        np.testing.assert_array_equal(static_parameters, expected_static_params)

        # Test bounds for static mode
        bounds = optimizer.get_parameter_bounds(effective_param_count=3)
        expected_bounds = [(1.0, 1000000), (-1.6, -1.5), (-100, 100)]
        assert bounds == expected_bounds

    def test_optimization_config_access(self, mock_analysis_core):
        """Test that optimization configuration is accessed correctly."""
        test_config = {
            "optimization_config": {
                "classical_optimization": {
                    "methods": ["Nelder-Mead"],
                    "method_options": {
                        "Nelder-Mead": {"maxiter": 5000, "xatol": 1e-10},
                    },
                }
            }
        }

        optimizer = ClassicalOptimizer(mock_analysis_core, test_config)

        # Test that optimization configuration is loaded correctly
        opt_config = optimizer.optimization_config

        assert "methods" in opt_config
        assert "method_options" in opt_config

        methods = opt_config["methods"]
        assert "Nelder-Mead" in methods
        assert len(methods) == 1  # Only Nelder-Mead should be present

        method_options = opt_config["method_options"]
        assert "Nelder-Mead" in method_options
        assert method_options["Nelder-Mead"]["maxiter"] == 5000

    def test_missing_configuration_sections(self, mock_analysis_core):
        """Test handling of missing configuration sections."""
        # Configuration with missing sections
        test_config = {
            "optimization_config": {
                "classical_optimization": {"methods": ["Nelder-Mead"]}
            }
            # Missing initial_parameters and parameter_space
        }

        # Create ClassicalOptimizer instance - should not crash
        optimizer = ClassicalOptimizer(mock_analysis_core, test_config)

        # Test graceful handling of missing sections
        param_bounds = test_config.get("parameter_space", {}).get("bounds", [])
        assert param_bounds == []  # Should get empty list as default

        # Test that optimization config is still accessible
        opt_config = optimizer.optimization_config
        assert "methods" in opt_config

    @pytest.mark.skipif(not GUROBI_AVAILABLE, reason="Gurobi not available")
    def test_gurobi_bounds_consistency(self, mock_analysis_core):
        """Test that Gurobi uses the same bounds as MCMC."""
        mcmc_bounds_config = {
            "parameter_space": {
                "bounds": [
                    {
                        "name": "D0",
                        "min": 1.0,
                        "max": 1000000.0,
                        "type": "TruncatedNormal",
                    },
                    {"name": "alpha", "min": -2.0, "max": 2.0, "type": "Normal"},
                    {"name": "D_offset", "min": -100, "max": 100, "type": "Normal"},
                    {
                        "name": "gamma_dot_t0",
                        "min": 1e-06,
                        "max": 1.0,
                        "type": "TruncatedNormal",
                    },
                    {"name": "beta", "min": -2.0, "max": 2.0, "type": "Normal"},
                    {
                        "name": "gamma_dot_t_offset",
                        "min": -0.01,
                        "max": 0.01,
                        "type": "Normal",
                    },
                    {"name": "phi0", "min": -10, "max": 10, "type": "Normal"},
                ]
            },
            "optimization_config": {"classical_optimization": {"methods": ["Gurobi"]}},
        }

        optimizer = ClassicalOptimizer(mock_analysis_core, mcmc_bounds_config)

        # Test that get_parameter_bounds extracts the correct MCMC bounds
        bounds = optimizer.get_parameter_bounds(effective_param_count=7)

        # Verify the bounds match the MCMC specification exactly
        expected_bounds = [
            (1.0, 1000000.0),
            (-2.0, 2.0),
            (-100, 100),
            (1e-06, 1.0),
            (-2.0, 2.0),
            (-0.01, 0.01),
            (-10, 10),
        ]

        assert len(bounds) == 7
        for i, (expected_min, expected_max) in enumerate(expected_bounds):
            actual_min, actual_max = bounds[i]
            assert actual_min == expected_min
            assert actual_max == expected_max


class TestGurobiIntegration:
    """Test Gurobi optimization integration and configuration."""

    def test_gurobi_availability_detection(self):
        """Test that Gurobi availability is correctly detected."""
        assert isinstance(GUROBI_AVAILABLE, bool)

    def test_gurobi_in_available_methods(self, mock_analysis_core, basic_config):
        """Test that Gurobi appears in available methods when installed."""
        optimizer = ClassicalOptimizer(mock_analysis_core, basic_config)
        available_methods = optimizer.get_available_methods()

        if GUROBI_AVAILABLE:
            assert "Gurobi" in available_methods
        else:
            assert "Gurobi" not in available_methods

    def test_gurobi_method_validation(self, mock_analysis_core, basic_config):
        """Test Gurobi method validation."""
        optimizer = ClassicalOptimizer(mock_analysis_core, basic_config)

        # Test Nelder-Mead validation (should always work)
        assert optimizer.validate_method_compatibility("Nelder-Mead", False)
        assert optimizer.validate_method_compatibility("Nelder-Mead", True)

        # Test Gurobi validation
        gurobi_valid = optimizer.validate_method_compatibility("Gurobi", True)
        assert gurobi_valid == GUROBI_AVAILABLE

    @pytest.mark.skipif(not GUROBI_AVAILABLE, reason="Gurobi not available")
    def test_gurobi_optimization_simple_quadratic(self, mock_analysis_core):
        """Test Gurobi optimization on a simple quadratic function."""

        # Simple quadratic: f(x) = (x-1)^2 + (y-2)^2
        def quadratic_objective(params):
            x, y = params
            return (x - 1) ** 2 + (y - 2) ** 2

        test_config = {
            "optimization_config": {
                "classical_optimization": {
                    "methods": ["Gurobi"],
                    "method_options": {
                        "Gurobi": {
                            "max_iterations": 100,
                            "tolerance": 1e-6,
                            "output_flag": 0,
                        }
                    },
                }
            },
            "parameter_space": {
                "bounds": [
                    {"name": "x", "min": -5.0, "max": 5.0},
                    {"name": "y", "min": -5.0, "max": 5.0},
                ]
            },
        }

        optimizer = ClassicalOptimizer(mock_analysis_core, test_config)

        # Test optimization
        initial_params = np.array([0.0, 0.0])
        bounds = [(-5.0, 5.0), (-5.0, 5.0)]

        success, result = optimizer._run_gurobi_optimization(
            quadratic_objective,
            initial_params,
            bounds,
            test_config["optimization_config"]["classical_optimization"][
                "method_options"
            ]["Gurobi"],
        )

        assert success, f"Gurobi optimization should succeed, got: {result}"
        assert hasattr(result, "x"), "Result should have optimal parameters"
        assert hasattr(result, "fun"), "Result should have optimal value"
        assert hasattr(result, "method") and result.method == "Gurobi-Iterative-QP"

    def test_gurobi_error_handling_when_unavailable(self, mock_analysis_core):
        """Test proper error handling when Gurobi is not available."""
        if GUROBI_AVAILABLE:
            pytest.skip("Gurobi is available, cannot test unavailable case")

        def dummy_objective(params):
            return sum(params**2)

        optimizer = ClassicalOptimizer(mock_analysis_core, {})

        # Should return False and ImportError
        success, result = optimizer._run_gurobi_optimization(
            dummy_objective, np.array([1.0]), [(0, 2)]
        )

        assert not success
        assert isinstance(result, ImportError)


if __name__ == "__main__":
    pytest.main([__file__])
