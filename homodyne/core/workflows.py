"""
Composable Analysis Workflows for Homodyne Analysis
===================================================

This module implements function composition patterns for creating modular,
testable, and readable analysis workflows in the homodyne analysis package.

Features:
- Composable analysis pipelines using functional patterns
- Type-safe parameter validation chains
- Modular data processing workflows
- Error-safe computational pipelines
- Configurable optimization workflows

Authors: Wei Chen, Hongrui He
Institution: Argonne National Laboratory
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .composition import ConfigurablePipeline
from .composition import Pipeline
from .composition import Result
from .composition import curry
from .composition import pipe

logger = logging.getLogger(__name__)


@dataclass
class AnalysisConfig:
    """Configuration container for analysis workflows."""

    config_path: Path
    output_dir: Path
    method: str = "classical"
    static_mode: bool = True
    plot_experimental_data: bool = False
    distributed: bool = False
    ml_accelerated: bool = False


@dataclass
class ExperimentalData:
    """Container for experimental data with validation."""

    c2_exp: np.ndarray
    phi_angles: np.ndarray
    time_length: float
    num_angles: int

    def __post_init__(self):
        """Validate experimental data upon creation."""
        self._validate_data()

    def _validate_data(self):
        """Validate experimental data integrity."""
        if self.c2_exp is None or self.phi_angles is None:
            raise ValueError("Experimental data and phi angles cannot be None")

        if len(self.phi_angles) != self.num_angles:
            raise ValueError(
                f"Expected {self.num_angles} angles, got {len(self.phi_angles)}"
            )

        if not np.all(np.isfinite(self.c2_exp)):
            raise ValueError("Experimental data contains non-finite values")

        if not np.all(np.isfinite(self.phi_angles)):
            raise ValueError("Phi angles contain non-finite values")


@dataclass
class OptimizationResult:
    """Container for optimization results with metadata."""

    parameters: np.ndarray
    chi_squared: float
    success: bool
    method: str
    iterations: int
    function_evaluations: int
    message: str
    metadata: dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class ParameterValidator:
    """
    Composable parameter validation using functional patterns.

    This class provides a fluent interface for building complex parameter
    validation chains that can be reused across different analysis workflows.
    """

    def __init__(self):
        self.validators = []

    def add_range_check(
        self, param_name: str, min_val: float, max_val: float
    ) -> ParameterValidator:
        """Add range validation for a parameter."""

        def range_validator(params: dict) -> bool:
            if param_name not in params:
                return False
            value = params[param_name]
            return min_val <= value <= max_val

        self.validators.append((f"{param_name}_range", range_validator))
        return self

    def add_positivity_check(self, param_name: str) -> ParameterValidator:
        """Add positivity validation for a parameter."""

        def positivity_validator(params: dict) -> bool:
            if param_name not in params:
                return False
            return params[param_name] > 0

        self.validators.append((f"{param_name}_positive", positivity_validator))
        return self

    def add_finite_check(self, param_name: str) -> ParameterValidator:
        """Add finite value validation for a parameter."""

        def finite_validator(params: dict) -> bool:
            if param_name not in params:
                return False
            return np.isfinite(params[param_name])

        self.validators.append((f"{param_name}_finite", finite_validator))
        return self

    def add_custom_check(
        self, name: str, validator: Callable[[dict], bool]
    ) -> ParameterValidator:
        """Add custom validation function."""
        self.validators.append((name, validator))
        return self

    def validate(self, params: dict) -> Result[dict]:
        """Execute all validation checks on parameters."""
        for name, validator in self.validators:
            try:
                if not validator(params):
                    return Result.failure(
                        ValueError(f"Parameter validation failed: {name}")
                    )
            except Exception as e:
                return Result.failure(
                    ValueError(f"Parameter validation error in {name}: {e}")
                )

        return Result.success(params)

    def build_pipeline(self) -> Pipeline:
        """Build a validation pipeline from the configured validators."""
        pipeline = Pipeline()

        for name, validator in self.validators:
            error_msg = f"Parameter validation failed: {name}"
            pipeline = pipeline.add_validation(validator, error_msg)

        return pipeline


class DataProcessor:
    """
    Composable data processing workflows using functional patterns.

    This class provides reusable data processing components that can be
    composed into complex analysis pipelines.
    """

    @staticmethod
    def normalize_correlation_data(c2_data: np.ndarray) -> Result[np.ndarray]:
        """Normalize correlation data to [0, 1] range."""
        try:
            if not np.all(np.isfinite(c2_data)):
                return Result.failure(ValueError("Data contains non-finite values"))

            c2_min = np.min(c2_data)
            c2_max = np.max(c2_data)

            if c2_max == c2_min:
                return Result.failure(ValueError("Data has zero variance"))

            normalized = (c2_data - c2_min) / (c2_max - c2_min)
            return Result.success(normalized)

        except Exception as e:
            return Result.failure(e)

    @staticmethod
    def filter_angles_by_range(
        phi_angles: np.ndarray, min_angle: float, max_angle: float
    ) -> Result[np.ndarray]:
        """Filter phi angles to specified range."""
        try:
            mask = (phi_angles >= min_angle) & (phi_angles <= max_angle)
            filtered_angles = phi_angles[mask]

            if len(filtered_angles) == 0:
                return Result.failure(
                    ValueError(f"No angles found in range [{min_angle}, {max_angle}]")
                )

            return Result.success(filtered_angles)

        except Exception as e:
            return Result.failure(e)

    @staticmethod
    def calculate_angle_statistics(phi_angles: np.ndarray) -> Result[dict[str, float]]:
        """Calculate statistical properties of angle distribution."""
        try:
            stats = {
                "mean": float(np.mean(phi_angles)),
                "std": float(np.std(phi_angles)),
                "min": float(np.min(phi_angles)),
                "max": float(np.max(phi_angles)),
                "range": float(np.max(phi_angles) - np.min(phi_angles)),
                "count": len(phi_angles),
            }
            return Result.success(stats)

        except Exception as e:
            return Result.failure(e)

    @staticmethod
    def apply_scaling_transformation(
        data: np.ndarray, contrast: float, offset: float
    ) -> Result[np.ndarray]:
        """Apply linear scaling transformation: scaled = contrast * data + offset."""
        try:
            if not np.isfinite(contrast) or not np.isfinite(offset):
                return Result.failure(ValueError("Contrast and offset must be finite"))

            scaled_data = contrast * data + offset
            return Result.success(scaled_data)

        except Exception as e:
            return Result.failure(e)


class OptimizationWorkflow:
    """
    Composable optimization workflows using functional patterns.

    This class provides building blocks for creating complex optimization
    pipelines that can be easily tested and modified.
    """

    def __init__(self, analyzer):
        self.analyzer = analyzer

    def create_parameter_validation_pipeline(self, analysis_mode: str) -> Pipeline:
        """Create parameter validation pipeline based on analysis mode."""
        validator = ParameterValidator()

        # Common validations for all modes
        validator = (
            validator.add_positivity_check("D0")
            .add_finite_check("D0")
            .add_range_check("alpha", -2.0, 2.0)
            .add_finite_check("alpha")
        )

        # Mode-specific validations
        if analysis_mode == "laminar_flow":
            validator = (
                validator.add_positivity_check("gamma_dot_t0")
                .add_range_check("beta", -2.0, 2.0)
                .add_range_check("phi0", -15.0, 15.0)
            )

        return validator.build_pipeline()

    def create_data_preprocessing_pipeline(self) -> Pipeline:
        """Create data preprocessing pipeline."""

        def log_data_info(data):
            logger.info(f"Processing experimental data with shape: {data.c2_exp.shape}")
            logger.info(
                f"Phi angles range: {data.phi_angles.min():.1f}° to {data.phi_angles.max():.1f}°"
            )
            return data

        def validate_data_integrity(data):
            if not isinstance(data, ExperimentalData):
                raise ValueError("Expected ExperimentalData object")
            return data

        return (
            Pipeline()
            .add_validation(lambda x: x is not None, "Data cannot be None")
            .add_step(validate_data_integrity)
            .add_side_effect(log_data_info)
        )

    def create_optimization_pipeline(self, method: str) -> Pipeline:
        """Create optimization execution pipeline."""

        def select_optimizer(config):
            if method == "classical":
                from ..optimization.classical import ClassicalOptimizer

                return ClassicalOptimizer(self.analyzer, config)
            if method == "robust":
                from ..optimization.robust import create_robust_optimizer

                return create_robust_optimizer(self.analyzer, config)
            raise ValueError(f"Unknown optimization method: {method}")

        def log_optimization_start(optimizer):
            logger.info(
                f"Starting {method} optimization with {type(optimizer).__name__}"
            )
            return optimizer

        return (
            Pipeline()
            .add_transform(select_optimizer)
            .add_side_effect(log_optimization_start)
        )

    def create_results_processing_pipeline(self) -> Pipeline:
        """Create results processing and validation pipeline."""

        def validate_optimization_result(result):
            if not hasattr(result, "x") or not hasattr(result, "fun"):
                raise ValueError("Invalid optimization result format")
            return result

        def log_results(result):
            logger.info(f"Optimization completed: χ² = {result.fun:.6e}")
            logger.info(
                f"Success: {result.success}, Iterations: {getattr(result, 'nit', 'N/A')}"
            )
            return result

        def create_result_object(result):
            return OptimizationResult(
                parameters=result.x,
                chi_squared=result.fun,
                success=result.success,
                method=getattr(result, "method", "unknown"),
                iterations=getattr(result, "nit", 0),
                function_evaluations=getattr(result, "nfev", 0),
                message=getattr(result, "message", ""),
            )

        return (
            Pipeline()
            .add_step(validate_optimization_result)
            .add_side_effect(log_results)
            .add_transform(create_result_object)
        )


class SimulationWorkflow:
    """
    Composable simulation workflows using functional patterns.

    This class provides building blocks for creating simulation pipelines
    that generate theoretical data with proper validation and error handling.
    """

    @staticmethod
    def create_phi_angles_pipeline(custom_angles: str | None = None) -> Pipeline:
        """Create phi angles generation/parsing pipeline."""

        def parse_custom_angles(angles_str):
            if angles_str is None:
                # Generate default angles
                return np.linspace(0, 180, 5, endpoint=False)
            # Parse custom angles
            try:
                angles_list = [float(angle.strip()) for angle in angles_str.split(",")]
                return np.array(angles_list)
            except ValueError as e:
                raise ValueError(f"Failed to parse phi angles: {e}")

        def validate_angles(angles):
            if len(angles) == 0:
                raise ValueError("No phi angles provided")
            if not np.all(np.isfinite(angles)):
                raise ValueError("Phi angles must be finite")
            if np.any(angles < 0) or np.any(angles >= 360):
                raise ValueError("Phi angles must be in range [0, 360)")
            return angles

        def log_angles(angles):
            logger.info(f"Using {len(angles)} phi angles: {angles}")
            return angles

        return (
            Pipeline()
            .add_transform(parse_custom_angles)
            .add_step(validate_angles)
            .add_side_effect(log_angles)
        )

    @staticmethod
    def create_time_arrays_pipeline(config: dict) -> Pipeline:
        """Create time arrays generation pipeline."""

        def extract_temporal_config(config):
            temporal_config = config.get("analyzer_parameters", {}).get("temporal", {})
            dt = temporal_config.get("dt", 0.1)
            start_frame = temporal_config.get("start_frame", 1)
            end_frame = temporal_config.get("end_frame", 50)
            return dt, start_frame, end_frame

        def create_time_arrays(temporal_params):
            dt, start_frame, end_frame = temporal_params
            n_time = end_frame - start_frame + 1  # Inclusive counting

            if n_time <= 0:
                raise ValueError("Invalid time range: end_frame must be > start_frame")

            t1 = np.arange(n_time) * dt
            t2 = np.arange(n_time) * dt
            return t1, t2, n_time

        def log_time_info(time_data):
            t1, t2, n_time = time_data
            dt = t1[1] - t1[0] if len(t1) > 1 else 0.1
            logger.info(f"Created time arrays: {n_time} points, dt={dt}")
            return time_data

        return (
            Pipeline()
            .add_transform(extract_temporal_config)
            .add_step(create_time_arrays)
            .add_side_effect(log_time_info)
        )

    @staticmethod
    def create_c2_generation_pipeline(core, initial_params: np.ndarray) -> Pipeline:
        """Create C2 correlation function generation pipeline."""

        def validate_parameters(params):
            if params is None or len(params) == 0:
                raise ValueError("Initial parameters cannot be None or empty")
            if not np.all(np.isfinite(params)):
                raise ValueError("Initial parameters must be finite")
            return params

        def generate_c2_for_angles(data):
            params, phi_angles, n_time = data
            n_angles = len(phi_angles)
            c2_theoretical = np.zeros((n_angles, n_time, n_time))

            for i, phi_angle in enumerate(phi_angles):
                logger.debug(f"Computing C2 for phi angle {phi_angle:.1f}°")
                c2_single = core.calculate_c2_single_angle_optimized(params, phi_angle)
                c2_theoretical[i] = c2_single

            return c2_theoretical

        def log_generation_complete(c2_data):
            logger.info(f"Generated C2 data with shape: {c2_data.shape}")
            return c2_data

        return (
            Pipeline()
            .add_step(
                lambda phi_angles: (
                    validate_parameters(initial_params),
                    phi_angles[0],
                    phi_angles[1],
                )
            )
            .add_transform(generate_c2_for_angles)
            .add_side_effect(log_generation_complete)
        )


# Higher-order functions for workflow composition
def create_analysis_workflow(
    config: AnalysisConfig,
) -> Callable[[Any], Result[OptimizationResult]]:
    """
    Create a complete analysis workflow as a composed function.

    This function demonstrates how to use function composition to create
    complex analysis pipelines that are both readable and testable.

    Parameters
    ----------
    config : AnalysisConfig
        Analysis configuration

    Returns
    -------
    Callable[[Any], Result[OptimizationResult]]
        Composed analysis workflow function
    """

    # Create individual pipeline components
    def load_and_validate_config(analyzer_class):
        # Initialize analyzer with config
        analyzer = analyzer_class(str(config.config_path))
        return analyzer

    def load_experimental_data(analyzer):
        c2_exp, time_length, phi_angles, num_angles = analyzer.load_experimental_data()
        return ExperimentalData(c2_exp, phi_angles, time_length, num_angles)

    def run_optimization(data_and_analyzer):
        analyzer, data = data_and_analyzer
        workflow = OptimizationWorkflow(analyzer)

        # Create optimization pipeline
        opt_pipeline = (
            workflow.create_parameter_validation_pipeline("static")
            .add_step(lambda x: workflow.create_optimization_pipeline(config.method))
            .add_step(lambda x: workflow.create_results_processing_pipeline())
        )

        return opt_pipeline.execute(data)

    # Compose the complete workflow
    return pipe(
        load_and_validate_config,
        lambda analyzer: (analyzer, load_experimental_data(analyzer)),
        run_optimization,
    )


def create_simulation_workflow(
    config: AnalysisConfig, phi_angles_str: str | None = None
) -> Callable:
    """
    Create a complete simulation workflow as a composed function.

    Parameters
    ----------
    config : AnalysisConfig
        Analysis configuration
    phi_angles_str : Optional[str]
        Custom phi angles specification

    Returns
    -------
    Callable
        Composed simulation workflow function
    """
    # Curry the simulation pipeline creation functions
    phi_pipeline = curry(SimulationWorkflow.create_phi_angles_pipeline)(phi_angles_str)
    time_pipeline = SimulationWorkflow.create_time_arrays_pipeline

    # Compose simulation workflow
    def simulation_workflow(analyzer_and_config):
        analyzer, config_dict = analyzer_and_config

        # Get initial parameters
        initial_params = np.array(config_dict["initial_parameters"]["values"])

        # Create and execute simulation pipeline
        phi_result = phi_pipeline().execute(phi_angles_str)
        if phi_result.is_failure:
            return phi_result

        time_result = time_pipeline(config_dict).execute(config_dict)
        if time_result.is_failure:
            return time_result

        # Generate C2 data
        phi_angles = phi_result.value
        t1, t2, n_time = time_result.value

        c2_pipeline = SimulationWorkflow.create_c2_generation_pipeline(
            analyzer, initial_params
        )
        c2_result = c2_pipeline.execute((phi_angles, n_time))

        return c2_result

    return simulation_workflow


# Demonstration and testing functions
def demonstrate_workflow_composition():
    """
    Demonstrate composable workflow patterns.

    This function shows how the composition patterns improve readability
    and testability in real analysis scenarios.
    """
    print("Composable Workflow Demonstration")
    print("=" * 50)

    # 1. Parameter validation demonstration
    print("\n1. Parameter Validation Chain:")
    validator = (
        ParameterValidator()
        .add_positivity_check("D0")
        .add_range_check("alpha", -2.0, 2.0)
        .add_finite_check("gamma_dot_t0")
    )

    test_params = {"D0": 1e-11, "alpha": 0.5, "gamma_dot_t0": 0.01}
    validation_result = validator.validate(test_params)

    if validation_result.is_success:
        print(f"Parameters valid: {validation_result.value}")
    else:
        print(f"Validation failed: {validation_result.error}")

    # 2. Data processing pipeline demonstration
    print("\n2. Data Processing Pipeline:")
    test_data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

    processing_pipeline = (
        Pipeline()
        .add_validation(lambda x: x.size > 0, "Data cannot be empty")
        .add_transform(lambda x: x * 2)  # Scale by 2
        .add_transform(np.mean)  # Calculate mean
        .add_side_effect(lambda x: print(f"Processed result: {x}"))
    )

    result = processing_pipeline.execute(test_data)
    if result.is_success:
        print(f"Final result: {result.value}")

    # 3. Configurable pipeline demonstration
    print("\n3. Configurable Pipeline:")
    config = {
        "steps": [
            {
                "type": "validation",
                "function": "is_not_empty",
                "error_message": "Data cannot be empty",
            },
            {"type": "transform", "function": "mean"},
            {"type": "transform", "function": "sqrt"},
        ]
    }

    configurable_pipeline = ConfigurablePipeline(config)
    pipeline = configurable_pipeline.build_pipeline()

    test_array = np.array([1, 4, 9, 16, 25])
    config_result = pipeline.execute(test_array)

    if config_result.is_success:
        print(f"Configurable pipeline result: {config_result.value}")


if __name__ == "__main__":
    demonstrate_workflow_composition()
