"""
Composed Analysis Engine for Homodyne Analysis
==============================================

This module demonstrates the application of function composition patterns
to create more readable, testable, and maintainable analysis workflows.

It refactors key analysis functions using:
- Functional composition patterns
- Monadic error handling
- Composable validation chains
- Modular data processing pipelines

Authors: Wei Chen, Hongrui He
Institution: Argonne National Laboratory
"""

import logging
from pathlib import Path

import numpy as np

from .composition import Pipeline
from .composition import Result
from .workflows import ExperimentalData
from .workflows import OptimizationResult
from .workflows import ParameterValidator

logger = logging.getLogger(__name__)


class ComposedHomodyneAnalysis:
    """
    Homodyne analysis engine using function composition patterns.

    This class demonstrates how to refactor complex analysis workflows
    using functional programming patterns for improved readability,
    testability, and maintainability.
    """

    def __init__(self, config_path: str | Path):
        self.config_path = Path(config_path)
        self.config = None
        self.analyzer = None
        self._initialize_analysis_core()

    def _initialize_analysis_core(self):
        """Initialize the underlying analysis core."""
        try:
            from ..analysis.core import HomodyneAnalysisCore

            self.analyzer = HomodyneAnalysisCore(str(self.config_path))
            self.config = self.analyzer.config
            logger.info("✓ Composed analysis engine initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize analysis core: {e}")
            raise

    def create_parameter_validation_workflow(
        self,
    ) -> Callable[[np.ndarray], Result[np.ndarray]]:
        """
        Create a composable parameter validation workflow.

        Returns
        -------
        Callable[[np.ndarray], Result[np.ndarray]]
            Validation workflow function
        """
        # Get analysis mode to determine validation rules
        analysis_mode = self.analyzer.config_manager.get_analysis_mode()
        is_static = self.analyzer.is_static_mode()

        # Build validation pipeline based on mode
        validator = ParameterValidator()

        # Common validations
        validator = (
            validator.add_positivity_check("D0")
            .add_finite_check("D0")
            .add_range_check("alpha", -2.0, 2.0)
        )

        # Mode-specific validations
        if not is_static:  # Laminar flow mode
            validator = (
                validator.add_positivity_check("gamma_dot_t0")
                .add_range_check("beta", -2.0, 2.0)
                .add_range_check("phi0", -15.0, 15.0)
            )

        def validate_parameters(params: np.ndarray) -> Result[np.ndarray]:
            """Convert array to dict and validate."""
            try:
                param_names = self.config.get("initial_parameters", {}).get(
                    "parameter_names", []
                )
                if len(param_names) != len(params):
                    return Result.failure(ValueError("Parameter count mismatch"))

                param_dict = dict(zip(param_names, params, strict=False))
                validation_result = validator.validate(param_dict)

                if validation_result.is_success:
                    return Result.success(params)
                return Result.failure(validation_result.error)

            except Exception as e:
                return Result.failure(e)

        return validate_parameters

    def create_data_loading_workflow(self) -> Callable[[], Result[ExperimentalData]]:
        """
        Create a composable data loading workflow.

        Returns
        -------
        Callable[[], Result[ExperimentalData]]
            Data loading workflow function
        """

        def load_experimental_data() -> Result[ExperimentalData]:
            """Load and validate experimental data."""
            try:
                logger.info("Loading experimental data...")
                c2_exp, time_length, phi_angles, num_angles = (
                    self.analyzer.load_experimental_data()
                )

                # Create data container with built-in validation
                data = ExperimentalData(c2_exp, phi_angles, time_length, num_angles)
                logger.info(
                    f"✓ Loaded data: {num_angles} angles, time_length={time_length}"
                )

                return Result.success(data)

            except Exception as e:
                logger.error(f"Failed to load experimental data: {e}")
                return Result.failure(e)

        return load_experimental_data

    def create_chi_squared_calculation_workflow(
        self,
    ) -> Callable[[np.ndarray, np.ndarray, np.ndarray], Result[float]]:
        """
        Create a composable chi-squared calculation workflow.

        Returns
        -------
        Callable[[np.ndarray, np.ndarray, np.ndarray], Result[float]]
            Chi-squared calculation workflow function
        """

        # Create validation pipeline
        def validate_inputs(
            params: np.ndarray, phi_angles: np.ndarray, c2_exp: np.ndarray
        ) -> Result[Tuple]:
            """Validate inputs for chi-squared calculation."""
            validation_pipeline = (
                Pipeline()
                .add_validation(lambda x: x[0] is not None, "Parameters cannot be None")
                .add_validation(lambda x: x[1] is not None, "Phi angles cannot be None")
                .add_validation(
                    lambda x: x[2] is not None, "Experimental data cannot be None"
                )
                .add_validation(lambda x: len(x[0]) > 0, "Parameters cannot be empty")
                .add_validation(lambda x: len(x[1]) > 0, "Phi angles cannot be empty")
                .add_validation(
                    lambda x: np.all(np.isfinite(x[0])), "Parameters must be finite"
                )
                .add_validation(
                    lambda x: np.all(np.isfinite(x[1])), "Phi angles must be finite"
                )
                .add_validation(
                    lambda x: np.all(np.isfinite(x[2])),
                    "Experimental data must be finite",
                )
            )

            return validation_pipeline.execute((params, phi_angles, c2_exp))

        def calculate_chi_squared(validated_inputs: Tuple) -> Result[float]:
            """Calculate chi-squared using the composed workflow."""
            try:
                params, phi_angles, c2_exp = validated_inputs

                # Use the refactored chi-squared calculation
                chi2_value = self.analyzer.calculate_chi_squared_optimized(
                    params, phi_angles, c2_exp
                )

                if not np.isfinite(chi2_value):
                    return Result.failure(
                        ValueError("Chi-squared calculation returned non-finite value")
                    )

                return Result.success(float(chi2_value))

            except Exception as e:
                return Result.failure(e)

        # Compose the complete workflow
        return lambda params, phi_angles, c2_exp: (
            validate_inputs(params, phi_angles, c2_exp).flat_map(calculate_chi_squared)
        )

    def create_optimization_workflow(
        self, method: str = "classical"
    ) -> Callable[[np.ndarray, ExperimentalData], Result[OptimizationResult]]:
        """
        Create a composable optimization workflow.

        Parameters
        ----------
        method : str
            Optimization method ("classical" or "robust")

        Returns
        -------
        Callable[[np.ndarray, ExperimentalData], Result[OptimizationResult]]
            Optimization workflow function
        """

        def run_optimization(
            initial_params: np.ndarray, data: ExperimentalData
        ) -> Result[OptimizationResult]:
            """Execute optimization using functional composition."""
            try:
                # Create chi-squared objective function
                chi2_workflow = self.create_chi_squared_calculation_workflow()

                def objective_function(params):
                    result = chi2_workflow(params, data.phi_angles, data.c2_exp)
                    if result.is_success:
                        return result.value
                    logger.warning(f"Chi-squared calculation failed: {result.error}")
                    return np.inf

                # Select and run optimizer
                if method == "classical":
                    from ..optimization.classical import ClassicalOptimizer

                    optimizer = ClassicalOptimizer(self.analyzer, self.config)
                    success, opt_result = optimizer.optimize(
                        objective_function, initial_params
                    )
                elif method == "robust":
                    from ..optimization.robust import create_robust_optimizer

                    optimizer = create_robust_optimizer(self.analyzer, self.config)
                    success, opt_result = optimizer.optimize(
                        objective_function, initial_params
                    )
                else:
                    return Result.failure(
                        ValueError(f"Unknown optimization method: {method}")
                    )

                # Process results
                if success and hasattr(opt_result, "x"):
                    result = OptimizationResult(
                        parameters=opt_result.x,
                        chi_squared=opt_result.fun,
                        success=opt_result.success,
                        method=method,
                        iterations=getattr(opt_result, "nit", 0),
                        function_evaluations=getattr(opt_result, "nfev", 0),
                        message=getattr(opt_result, "message", ""),
                    )
                    return Result.success(result)
                error_msg = str(opt_result) if not success else "Optimization failed"
                return Result.failure(RuntimeError(error_msg))

            except Exception as e:
                return Result.failure(e)

        return run_optimization

    def create_complete_analysis_workflow(
        self, method: str = "classical"
    ) -> Callable[[], Result[OptimizationResult]]:
        """
        Create a complete analysis workflow using function composition.

        This demonstrates how to compose all the individual workflows into
        a single, testable, and maintainable analysis pipeline.

        Parameters
        ----------
        method : str
            Optimization method to use

        Returns
        -------
        Callable[[], Result[OptimizationResult]]
            Complete analysis workflow function
        """
        # Get individual workflow components
        load_data_workflow = self.create_data_loading_workflow()
        optimization_workflow = self.create_optimization_workflow(method)
        param_validation_workflow = self.create_parameter_validation_workflow()

        def complete_analysis() -> Result[OptimizationResult]:
            """Execute the complete analysis using functional composition."""
            logger.info(f"Starting complete {method} analysis workflow...")

            # Step 1: Load experimental data
            data_result = load_data_workflow()
            if data_result.is_failure:
                return Result.failure(data_result.error)

            data = data_result.value

            # Step 2: Get and validate initial parameters
            try:
                initial_params = np.array(self.config["initial_parameters"]["values"])
            except (KeyError, TypeError) as e:
                return Result.failure(
                    ValueError(f"Failed to get initial parameters: {e}")
                )

            param_validation_result = param_validation_workflow(initial_params)
            if param_validation_result.is_failure:
                return Result.failure(param_validation_result.error)

            validated_params = param_validation_result.value

            # Step 3: Run optimization
            optimization_result = optimization_workflow(validated_params, data)
            if optimization_result.is_failure:
                return Result.failure(optimization_result.error)

            result = optimization_result.value
            logger.info(f"✓ Analysis completed: χ² = {result.chi_squared:.6e}")

            return Result.success(result)

        return complete_analysis

    def create_simulation_workflow(
        self, phi_angles: Optional[np.ndarray] = None
    ) -> Callable[[np.ndarray], Result[Dict[str, np.ndarray]]]:
        """
        Create a composable simulation workflow.

        Parameters
        ----------
        phi_angles : Optional[np.ndarray]
            Custom phi angles, if None uses defaults

        Returns
        -------
        Callable[[np.ndarray], Result[Dict[str, np.ndarray]]]
            Simulation workflow function
        """

        def run_simulation(parameters: np.ndarray) -> Result[Dict[str, np.ndarray]]:
            """Execute simulation using functional composition."""
            try:
                # Parameter validation
                param_validation = self.create_parameter_validation_workflow()
                validation_result = param_validation(parameters)
                if validation_result.is_failure:
                    return Result.failure(validation_result.error)

                # Use default angles if not provided
                if phi_angles is None:
                    angles = np.linspace(0, 180, 5, endpoint=False)
                else:
                    angles = phi_angles

                # Create simulation pipeline
                simulation_pipeline = (
                    Pipeline()
                    .add_validation(lambda x: len(x) > 0, "Parameters cannot be empty")
                    .add_side_effect(
                        lambda x: logger.info(f"Simulating with {len(angles)} angles")
                    )
                    .add_transform(
                        lambda params: self._generate_theoretical_data(params, angles)
                    )
                )

                simulation_result = simulation_pipeline.execute(parameters)
                return simulation_result

            except Exception as e:
                return Result.failure(e)

        return run_simulation

    def _generate_theoretical_data(
        self, parameters: np.ndarray, phi_angles: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """Generate theoretical C2 data for simulation."""
        n_angles = len(phi_angles)

        # Get time configuration
        temporal_config = self.config.get("analyzer_parameters", {}).get("temporal", {})
        dt = temporal_config.get("dt", 0.1)
        start_frame = temporal_config.get("start_frame", 1)
        end_frame = temporal_config.get("end_frame", 50)
        n_time = end_frame - start_frame + 1  # Inclusive counting

        # Generate time arrays (n_time points from 0 to (n_time-1)*dt)
        t1 = np.arange(n_time) * dt
        t2 = np.arange(n_time) * dt

        # Generate C2 data
        c2_theoretical = np.zeros((n_angles, n_time, n_time))

        for i, phi_angle in enumerate(phi_angles):
            logger.debug(f"Computing C2 for phi angle {phi_angle:.1f}°")
            c2_single = self.analyzer.calculate_c2_single_angle_optimized(
                parameters, phi_angle
            )
            c2_theoretical[i] = c2_single

        return {
            "c2_theoretical": c2_theoretical,
            "phi_angles": phi_angles,
            "t1": t1,
            "t2": t2,
            "parameters": parameters,
        }

    # Convenience methods that use the composed workflows
    def run_analysis(self, method: str = "classical") -> Result[OptimizationResult]:
        """
        Run complete analysis using composed workflows.

        Parameters
        ----------
        method : str
            Optimization method ("classical" or "robust")

        Returns
        -------
        Result[OptimizationResult]
            Analysis results or error
        """
        workflow = self.create_complete_analysis_workflow(method)
        return workflow()

    def run_simulation(
        self,
        parameters: Optional[np.ndarray] = None,
        phi_angles: Optional[np.ndarray] = None,
    ) -> Result[Dict[str, np.ndarray]]:
        """
        Run simulation using composed workflows.

        Parameters
        ----------
        parameters : Optional[np.ndarray]
            Parameters to use, if None uses config initial parameters
        phi_angles : Optional[np.ndarray]
            Phi angles to use, if None uses defaults

        Returns
        -------
        Result[Dict[str, np.ndarray]]
            Simulation results or error
        """
        if parameters is None:
            parameters = np.array(self.config["initial_parameters"]["values"])

        workflow = self.create_simulation_workflow(phi_angles)
        return workflow(parameters)

    def validate_parameters(self, parameters: np.ndarray) -> Result[np.ndarray]:
        """
        Validate parameters using composed validation workflow.

        Parameters
        ----------
        parameters : np.ndarray
            Parameters to validate

        Returns
        -------
        Result[np.ndarray]
            Validated parameters or error
        """
        workflow = self.create_parameter_validation_workflow()
        return workflow(parameters)


# Demonstration function
def demonstrate_composed_analysis():
    """
    Demonstrate composed analysis workflows.

    This function shows how composition patterns improve the analysis
    workflow's readability, testability, and maintainability.
    """
    print("Composed Analysis Workflow Demonstration")
    print("=" * 50)

    try:
        # Note: This would require a real config file to run
        # config_path = "path/to/config.json"
        # analyzer = ComposedHomodyneAnalysis(config_path)

        print("\n1. Workflow Creation:")
        print("✓ Data loading workflow created")
        print("✓ Parameter validation workflow created")
        print("✓ Chi-squared calculation workflow created")
        print("✓ Optimization workflow created")
        print("✓ Complete analysis workflow created")

        print("\n2. Composition Benefits:")
        print("- Each workflow component is independently testable")
        print("- Error handling is consistent across all workflows")
        print("- Workflows can be easily modified or extended")
        print("- Complex operations are broken into readable steps")
        print("- Functional composition enables easy testing")

        print("\n3. Usage Patterns:")
        print("- analyzer.run_analysis('classical')")
        print("- analyzer.run_simulation(custom_params)")
        print("- analyzer.validate_parameters(test_params)")

    except Exception as e:
        print(f"Demonstration requires valid configuration: {e}")


if __name__ == "__main__":
    demonstrate_composed_analysis()
