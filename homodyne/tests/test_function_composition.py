"""
Tests for Function Composition Patterns
=======================================

Comprehensive test suite demonstrating the improved testability achieved
through function composition patterns in the homodyne analysis package.

These tests show how composable functions are easier to test, debug,
and maintain compared to monolithic implementations.

Authors: Wei Chen, Hongrui He
Institution: Argonne National Laboratory
"""

from unittest.mock import patch

import numpy as np
import pytest

# Import our composition framework
from homodyne.core.composition import ConfigurablePipeline
from homodyne.core.composition import Pipeline
from homodyne.core.composition import Result
from homodyne.core.composition import compose
from homodyne.core.composition import create_validation_chain
from homodyne.core.composition import curry
from homodyne.core.composition import memoize
from homodyne.core.composition import pipe
from homodyne.core.composition import retry_on_failure
from homodyne.core.composition import safe_divide
from homodyne.core.composition import safe_sqrt
from homodyne.core.workflows import DataProcessor
from homodyne.core.workflows import ExperimentalData
from homodyne.core.workflows import ParameterValidator
from homodyne.core.workflows import SimulationWorkflow


class TestCompositionFramework:
    """Test suite for the core composition framework."""

    def test_result_success_flow(self):
        """Test Result type success flow."""
        result = Result.success(42)

        assert result.is_success
        assert not result.is_failure
        assert result.value == 42
        assert result.error is None

    def test_result_failure_flow(self):
        """Test Result type failure flow."""
        error = ValueError("Test error")
        result = Result.failure(error)

        assert not result.is_success
        assert result.is_failure
        assert result.error == error

        with pytest.raises(ValueError):
            _ = result.value

    def test_result_map_operation(self):
        """Test Result map operation for transformations."""
        result = Result.success(10)
        mapped = result.map(lambda x: x * 2)

        assert mapped.is_success
        assert mapped.value == 20

    def test_result_map_with_error(self):
        """Test Result map operation with error handling."""
        result = Result.success("not_a_number")
        mapped = result.map(lambda x: int(x))

        assert mapped.is_failure
        assert isinstance(mapped.error, ValueError)

    def test_result_flat_map_operation(self):
        """Test Result flat_map operation for chaining."""
        result = (
            Result.success(16).flat_map(safe_sqrt).flat_map(lambda x: safe_divide(x, 2))
        )

        assert result.is_success
        assert result.value == 2.0

    def test_result_flat_map_with_error(self):
        """Test Result flat_map operation with error propagation."""
        result = (
            Result.success(-16)
            .flat_map(safe_sqrt)
            .flat_map(lambda x: safe_divide(x, 2))
        )

        assert result.is_failure
        assert isinstance(result.error, ValueError)

    def test_result_filter_operation(self):
        """Test Result filter operation."""
        result = Result.success(10)
        filtered = result.filter(lambda x: x > 5, "Value too small")

        assert filtered.is_success
        assert filtered.value == 10

    def test_result_filter_failure(self):
        """Test Result filter operation with failure."""
        result = Result.success(3)
        filtered = result.filter(lambda x: x > 5, "Value too small")

        assert filtered.is_failure
        assert "Value too small" in str(filtered.error)

    def test_function_composition(self):
        """Test basic function composition."""
        add_one = lambda x: x + 1
        multiply_two = lambda x: x * 2

        composed = compose(add_one, multiply_two)
        result = composed(5)  # add_one(multiply_two(5)) = add_one(10) = 11

        assert result == 11

    def test_function_piping(self):
        """Test function piping (left to right composition)."""
        add_one = lambda x: x + 1
        multiply_two = lambda x: x * 2

        piped = pipe(add_one, multiply_two)
        result = piped(5)  # multiply_two(add_one(5)) = multiply_two(6) = 12

        assert result == 12

    def test_curry_function(self):
        """Test function currying."""

        def add_three(x, y, z):
            return x + y + z

        curried_add = curry(add_three)
        add_5_and_3 = curried_add(5)(3)
        result = add_5_and_3(2)

        assert result == 10

    def test_memoization(self):
        """Test function memoization."""
        call_count = 0

        @memoize
        def expensive_function(x):
            nonlocal call_count
            call_count += 1
            return x * x

        # First call
        result1 = expensive_function(5)
        assert result1 == 25
        assert call_count == 1

        # Second call with same argument (should use cache)
        result2 = expensive_function(5)
        assert result2 == 25
        assert call_count == 1  # Should not increment

        # Third call with different argument
        result3 = expensive_function(6)
        assert result3 == 36
        assert call_count == 2

    def test_retry_decorator(self):
        """Test retry decorator functionality."""
        attempt_count = 0

        @retry_on_failure(max_attempts=3, delay=0.01)
        def flaky_function():
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count < 3:
                raise RuntimeError("Temporary failure")
            return "success"

        result = flaky_function()
        assert result == "success"
        assert attempt_count == 3


class TestPipelineFramework:
    """Test suite for the Pipeline framework."""

    def test_pipeline_basic_execution(self):
        """Test basic pipeline execution."""
        pipeline = Pipeline().add_step(lambda x: x * 2).add_step(lambda x: x + 1)

        result = pipeline.execute(5)
        assert result.is_success
        assert result.value == 11  # (5 * 2) + 1

    def test_pipeline_with_validation(self):
        """Test pipeline with validation steps."""
        pipeline = (
            Pipeline()
            .add_validation(lambda x: x > 0, "Value must be positive")
            .add_transform(lambda x: x * 2)
            .add_validation(lambda x: x < 20, "Value too large")
        )

        # Test successful execution
        result = pipeline.execute(5)
        assert result.is_success
        assert result.value == 10

        # Test validation failure
        result_fail = pipeline.execute(-1)
        assert result_fail.is_failure
        assert "positive" in str(result_fail.error)

    def test_pipeline_with_side_effects(self):
        """Test pipeline with side effects."""
        logged_values = []

        pipeline = (
            Pipeline()
            .add_transform(lambda x: x * 2)
            .add_side_effect(lambda x: logged_values.append(x))
            .add_transform(lambda x: x + 1)
        )

        result = pipeline.execute(5)
        assert result.is_success
        assert result.value == 11
        assert logged_values == [10]  # Side effect should capture intermediate value

    def test_pipeline_error_handling(self):
        """Test pipeline error handling."""

        def error_handler(error):
            return f"Handled: {error}"

        pipeline = (
            Pipeline()
            .add_transform(lambda x: x / 0)  # This will raise ZeroDivisionError
            .with_error_handler(error_handler)
        )

        result = pipeline.execute(5)
        assert result.is_success
        assert "Handled:" in result.value

    def test_configurable_pipeline(self):
        """Test configurable pipeline creation."""
        config = {
            "steps": [
                {
                    "type": "validation",
                    "function": "is_positive",
                    "error_message": "Value must be positive",
                },
                {"type": "transform", "function": "sqrt"},
                {"type": "transform", "function": "multiply", "args": [2]},
            ]
        }

        configurable_pipeline = ConfigurablePipeline(config)
        pipeline = configurable_pipeline.build_pipeline()

        result = pipeline.execute(np.array([16]))
        assert result.is_success
        assert np.allclose(result.value, [8])  # sqrt(16) * 2 = 4 * 2 = 8

    def test_validation_chain(self):
        """Test validation chain creation."""
        validators = [
            lambda x: isinstance(x, (int, float)),
            lambda x: x > 0,
            lambda x: x < 100,
        ]

        validation_chain = create_validation_chain(*validators)

        # Test successful validation
        result = validation_chain(50)
        assert result.is_success
        assert result.value == 50

        # Test validation failure
        result_fail = validation_chain(-5)
        assert result_fail.is_failure


class TestWorkflowComponents:
    """Test suite for workflow components."""

    def test_parameter_validator(self):
        """Test ParameterValidator functionality."""
        validator = (
            ParameterValidator()
            .add_positivity_check("D0")
            .add_range_check("alpha", -2.0, 2.0)
            .add_finite_check("gamma_dot_t0")
        )

        # Test valid parameters
        valid_params = {"D0": 1e-11, "alpha": 0.5, "gamma_dot_t0": 0.01}
        result = validator.validate(valid_params)
        assert result.is_success

        # Test invalid parameters (negative D0)
        invalid_params = {"D0": -1e-11, "alpha": 0.5, "gamma_dot_t0": 0.01}
        result = validator.validate(invalid_params)
        assert result.is_failure
        assert "positive" in str(result.error)

    def test_data_processor_normalization(self):
        """Test DataProcessor normalization functionality."""
        test_data = np.array([[1, 2, 3], [4, 5, 6]])
        result = DataProcessor.normalize_correlation_data(test_data)

        assert result.is_success
        normalized = result.value
        assert np.min(normalized) == 0.0
        assert np.max(normalized) == 1.0

    def test_data_processor_angle_filtering(self):
        """Test DataProcessor angle filtering functionality."""
        angles = np.array([0, 45, 90, 135, 180, 225, 270, 315])
        result = DataProcessor.filter_angles_by_range(angles, 45, 135)

        assert result.is_success
        filtered = result.value
        expected = np.array([45, 90, 135])
        np.testing.assert_array_equal(filtered, expected)

    def test_data_processor_angle_statistics(self):
        """Test DataProcessor angle statistics calculation."""
        angles = np.array([0, 90, 180, 270])
        result = DataProcessor.calculate_angle_statistics(angles)

        assert result.is_success
        stats = result.value
        assert stats["mean"] == 135.0
        assert stats["min"] == 0.0
        assert stats["max"] == 270.0
        assert stats["count"] == 4

    def test_data_processor_scaling(self):
        """Test DataProcessor scaling transformation."""
        data = np.array([1, 2, 3, 4, 5])
        result = DataProcessor.apply_scaling_transformation(data, 2.0, 1.0)

        assert result.is_success
        scaled = result.value
        expected = np.array([3, 5, 7, 9, 11])  # 2 * [1,2,3,4,5] + 1
        np.testing.assert_array_equal(scaled, expected)

    @patch("homodyne.core.workflows.logger")
    def test_simulation_workflow_phi_angles(self, mock_logger):
        """Test SimulationWorkflow phi angles creation."""
        # Test custom angles
        pipeline = SimulationWorkflow.create_phi_angles_pipeline("0,45,90,135")
        result = pipeline.execute("0,45,90,135")

        assert result.is_success
        angles = result.value
        expected = np.array([0, 45, 90, 135])
        np.testing.assert_array_equal(angles, expected)

    def test_simulation_workflow_time_arrays(self):
        """Test SimulationWorkflow time arrays creation."""
        config = {
            "analyzer_parameters": {
                "temporal": {"dt": 0.1, "start_frame": 1, "end_frame": 11}
            }
        }

        pipeline = SimulationWorkflow.create_time_arrays_pipeline(config)
        result = pipeline.execute(config)

        assert result.is_success
        t1, t2, n_time = result.value
        # Frame counting: start_frame=1, end_frame=11 → n_time = 11 - 1 + 1 = 11 (inclusive)
        assert n_time == 11
        assert len(t1) == 11
        assert len(t2) == 11
        assert np.allclose(t1, np.arange(11) * 0.1)


class TestComposedFunctionBenefits:
    """
    Test suite demonstrating the benefits of composed functions.

    This class shows how composition patterns improve:
    - Testability: Individual components can be tested in isolation
    - Reusability: Components can be reused in different contexts
    - Maintainability: Changes to one component don't affect others
    - Readability: Complex operations are broken into understandable steps
    """

    def test_isolated_component_testing(self):
        """Demonstrate how composed functions enable isolated testing."""
        # Each component can be tested independently
        validator = ParameterValidator().add_positivity_check("value")
        processor = DataProcessor()

        # Test validator in isolation
        valid_result = validator.validate({"value": 5})
        invalid_result = validator.validate({"value": -5})

        assert valid_result.is_success
        assert invalid_result.is_failure

        # Test processor in isolation
        data = np.array([1, 2, 3])
        norm_result = processor.normalize_correlation_data(data)

        assert norm_result.is_success
        assert np.min(norm_result.value) == 0.0

    def test_component_reusability(self):
        """Demonstrate how composed functions can be reused."""
        # Same validator can be used in multiple contexts
        basic_validator = ParameterValidator().add_positivity_check("D0")

        # Reuse in different workflows
        static_validator = basic_validator.add_range_check("alpha", -1.0, 1.0)
        flow_validator = basic_validator.add_positivity_check("gamma_dot_t0")

        test_params = {"D0": 1e-11, "alpha": 0.5, "gamma_dot_t0": 0.01}

        # Both validators can validate the same parameters
        static_result = static_validator.validate(test_params)
        flow_result = flow_validator.validate(test_params)

        assert static_result.is_success
        assert flow_result.is_success

    def test_error_propagation_consistency(self):
        """Demonstrate consistent error handling across composed functions."""
        # All composed functions use Result type for consistent error handling

        # Test error propagation through multiple steps
        result = (
            Result.success(-16)
            .flat_map(safe_sqrt)  # This will fail
            .map(lambda x: x * 2)  # This won't execute
            .map(lambda x: x + 1)
        )  # This won't execute either

        assert result.is_failure
        assert isinstance(result.error, ValueError)

        # Error is preserved through the entire chain
        assert "negative" in str(result.error)

    def test_pipeline_modularity(self):
        """Demonstrate how pipelines can be built modularly."""
        # Build pipeline incrementally
        base_pipeline = Pipeline().add_validation(lambda x: x > 0, "Must be positive")

        # Add different transformations for different use cases
        sqrt_pipeline = base_pipeline.add_transform(lambda x: x**0.5)
        square_pipeline = base_pipeline.add_transform(lambda x: x**2)

        # Test different pipelines with same base validation
        sqrt_result = sqrt_pipeline.execute(16)
        square_result = square_pipeline.execute(4)

        assert sqrt_result.is_success
        assert sqrt_result.value == 4.0

        assert square_result.is_success
        assert square_result.value == 16

    def test_functional_vs_imperative_readability(self):
        """
        Demonstrate improved readability of functional composition
        compared to imperative style.
        """
        # Functional style (composed)
        functional_workflow = pipe(lambda x: x * 2, lambda x: x + 1, lambda x: x**0.5)

        # Imperative style (traditional)
        def imperative_workflow(x):
            step1 = x * 2
            step2 = step1 + 1
            step3 = step2**0.5
            return step3

        # Both produce the same result
        test_value = 4
        functional_result = functional_workflow(test_value)
        imperative_result = imperative_workflow(test_value)

        assert functional_result == imperative_result == 3.0

        # But functional style is more composable and testable


class TestExperimentalData:
    """Test the ExperimentalData dataclass with built-in validation."""

    def test_valid_experimental_data(self):
        """Test creation of valid ExperimentalData."""
        c2_exp = np.random.rand(5, 10, 10)
        phi_angles = np.array([0, 45, 90, 135, 180])

        data = ExperimentalData(c2_exp, phi_angles, 1.0, 5)

        assert data.num_angles == 5
        assert len(data.phi_angles) == 5
        assert data.c2_exp.shape[0] == 5

    def test_invalid_experimental_data(self):
        """Test validation failure in ExperimentalData."""
        c2_exp = np.random.rand(5, 10, 10)
        phi_angles = np.array([0, 45, 90])  # Wrong number of angles

        with pytest.raises(ValueError, match="Expected 5 angles"):
            ExperimentalData(c2_exp, phi_angles, 1.0, 5)

    def test_non_finite_data_validation(self):
        """Test validation of non-finite values."""
        c2_exp = np.array([[[1, 2], [3, np.inf]]])  # Contains infinity
        phi_angles = np.array([0])

        with pytest.raises(ValueError, match="non-finite values"):
            ExperimentalData(c2_exp, phi_angles, 1.0, 1)


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v"])
