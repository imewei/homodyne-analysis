"""
Utility Functions for Distributed and ML-Accelerated Optimization
=================================================================

This module provides utility functions and helper classes to simplify the
integration and usage of distributed computing and ML acceleration features
with the existing homodyne optimization framework.

Key Features:
- Automatic configuration loading and validation
- Integration helpers for existing optimizers
- Performance monitoring and benchmarking utilities
- System resource detection and optimization
- Configuration generation and management tools

Authors: Wei Chen, Hongrui He, Claude (Anthropic)
Institution: Argonne National Laboratory
"""

import json
import logging
import platform
import time
from pathlib import Path
from typing import TYPE_CHECKING
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

if TYPE_CHECKING:
    from .distributed import DistributedOptimizationCoordinator
    from .ml_acceleration import MLAcceleratedOptimizer

import numpy as np
import psutil

logger = logging.getLogger(__name__)


class OptimizationConfig:
    """Configuration manager for distributed and ML optimization features."""

    def __init__(self, config_path: str | Path | None = None):
        self.config_path = config_path
        self.config: dict[str, Any] = {}

        if config_path:
            self.load_config(config_path)
        else:
            self.load_default_config()

    def load_config(self, config_path: str | Path) -> None:
        """Load configuration from file."""
        config_path = Path(config_path)

        if not config_path.exists():
            logger.warning(f"Config file {config_path} not found, using defaults")
            self.load_default_config()
            return

        try:
            with open(config_path) as f:
                self.config = json.load(f)
            logger.info(f"Loaded configuration from {config_path}")
        except Exception as e:
            logger.error(f"Failed to load config from {config_path}: {e}")
            self.load_default_config()

    def load_default_config(self) -> None:
        """Load default configuration."""
        # Try to find default config in package
        try:
            from homodyne.config import get_template_path

            default_config_path = get_template_path("template")
            if default_config_path and Path(default_config_path).exists():
                self.load_config(default_config_path)
                return
        except ImportError:
            pass

        # Fallback to minimal default configuration
        self.config = self._get_minimal_config()
        logger.info("Using minimal default configuration")

    def _get_minimal_config(self) -> dict[str, Any]:
        """Get minimal default configuration."""
        return {
            "distributed_optimization": {
                "enabled": False,
                "backend_preference": ["multiprocessing", "ray"],
                "multiprocessing_config": {
                    "num_processes": min(psutil.cpu_count() or 4, 8)
                },
            },
            "ml_acceleration": {
                "enabled": False,
                "predictor_type": "ensemble",
                "enable_transfer_learning": False,
                "ml_model_config": {
                    "model_type": "ensemble",
                    "feature_scaling": "standard",
                    "validation_split": 0.2,
                },
            },
            "performance_monitoring": {
                "enabled": True,
                "metrics_collection": {
                    "optimization_time": True,
                    "convergence_rate": True,
                },
            },
        }

    def get_distributed_config(self) -> dict[str, Any]:
        """Get distributed optimization configuration."""
        return self.config.get("distributed_optimization", {})

    def get_ml_config(self) -> dict[str, Any]:
        """Get ML acceleration configuration."""
        return self.config.get("ml_acceleration", {})

    def get_performance_config(self) -> dict[str, Any]:
        """Get performance monitoring configuration."""
        return self.config.get("performance_monitoring", {})

    def is_distributed_enabled(self) -> bool:
        """Check if distributed optimization is enabled."""
        return self.get_distributed_config().get("enabled", False)

    def is_ml_enabled(self) -> bool:
        """Check if ML acceleration is enabled."""
        return self.get_ml_config().get("enabled", False)

    def save_config(self, output_path: str | Path) -> None:
        """Save current configuration to file."""
        output_path = Path(output_path)

        try:
            with open(output_path, "w") as f:
                json.dump(self.config, f, indent=2)
            logger.info(f"Saved configuration to {output_path}")
        except Exception as e:
            logger.error(f"Failed to save config to {output_path}: {e}")


class SystemResourceDetector:
    """Automatic detection and optimization of system resources."""

    @staticmethod
    def detect_system_capabilities() -> dict[str, Any]:
        """Detect system capabilities for optimization configuration."""
        capabilities = {
            "cpu_count": psutil.cpu_count(),
            "physical_cpu_count": psutil.cpu_count(logical=False),
            "memory_total_gb": psutil.virtual_memory().total / (1024**3),
            "memory_available_gb": psutil.virtual_memory().available / (1024**3),
            "platform": platform.system(),
            "architecture": platform.machine(),
            "python_version": platform.python_version(),
        }

        # CPU-only configuration (GPU support removed)
        capabilities["gpu_count"] = 0
        capabilities["gpu_memory_total_gb"] = 0

        # Network detection for distributed computing
        capabilities["network_interfaces"] = len(psutil.net_if_addrs())

        return capabilities

    @staticmethod
    def optimize_configuration(base_config: dict[str, Any]) -> dict[str, Any]:
        """Optimize configuration based on system capabilities."""
        capabilities = SystemResourceDetector.detect_system_capabilities()
        optimized_config = base_config.copy()

        # Optimize multiprocessing configuration
        if "distributed_optimization" in optimized_config:
            dist_config = optimized_config["distributed_optimization"]

            if "multiprocessing_config" in dist_config:
                mp_config = dist_config["multiprocessing_config"]

                # Optimize number of processes
                if mp_config.get("num_processes") is None:
                    # Use 75% of available CPUs, but leave at least 1 for system
                    optimal_processes = max(1, int(capabilities["cpu_count"] * 0.75))
                    mp_config["num_processes"] = optimal_processes

            # Optimize Ray configuration
            if "ray_config" in dist_config:
                ray_config = dist_config["ray_config"]

                if ray_config.get("num_cpus") is None:
                    ray_config["num_cpus"] = capabilities["cpu_count"]

                if ray_config.get("num_gpus") is None:
                    ray_config["num_gpus"] = capabilities["gpu_count"]

                if ray_config.get("memory_mb") is None:
                    # Use 80% of available memory
                    memory_mb = int(capabilities["memory_available_gb"] * 1024 * 0.8)
                    ray_config["memory_mb"] = memory_mb

        # Optimize ML configuration based on available memory
        if "ml_acceleration" in optimized_config:
            ml_config = optimized_config["ml_acceleration"]

            if capabilities["memory_total_gb"] < 4:
                # Limited memory - use simpler models
                if "ml_model_config" in ml_config:
                    model_config = ml_config["ml_model_config"]
                    if "hyperparameters" in model_config:
                        # Reduce model complexity
                        hp = model_config["hyperparameters"]
                        if "random_forest" in hp:
                            hp["random_forest"]["n_estimators"] = 50
                            hp["random_forest"]["max_depth"] = 5
                        if "neural_network" in hp:
                            hp["neural_network"]["hidden_layer_sizes"] = [50, 25]

        logger.info(
            f"Optimized CPU-only configuration for system with {capabilities['cpu_count']} CPUs, "
            f"{capabilities['memory_total_gb']:.1f}GB RAM"
        )

        return optimized_config

    @staticmethod
    def check_system_requirements() -> dict[str, bool]:
        """Check if system meets requirements for advanced features."""
        capabilities = SystemResourceDetector.detect_system_capabilities()

        requirements = {
            "sufficient_memory": capabilities["memory_total_gb"] >= 2.0,
            "sufficient_cpu": capabilities["cpu_count"] >= 2,
            "python_version_ok": tuple(
                map(int, capabilities["python_version"].split("."))
            )
            >= (3, 8),
            "multiprocessing_available": True,  # Always available
            "ray_recommended": capabilities["cpu_count"] >= 4
            and capabilities["memory_total_gb"] >= 8,
            "ml_recommended": capabilities["memory_total_gb"] >= 4,
        }

        return requirements


class OptimizationBenchmark:
    """Benchmarking utilities for optimization performance."""

    def __init__(self):
        self.results = []
        self.current_benchmark = None

    def start_benchmark(self, name: str, config: dict[str, Any]) -> None:
        """Start a new benchmark."""
        self.current_benchmark = {
            "name": name,
            "config": config,
            "start_time": time.time(),
            "metrics": {},
        }

    def record_metric(self, metric_name: str, value: Any) -> None:
        """Record a benchmark metric."""
        if self.current_benchmark:
            self.current_benchmark["metrics"][metric_name] = value

    def end_benchmark(self) -> dict[str, Any]:
        """End current benchmark and return results."""
        if not self.current_benchmark:
            return {}

        self.current_benchmark["end_time"] = time.time()
        self.current_benchmark["total_time"] = (
            self.current_benchmark["end_time"] - self.current_benchmark["start_time"]
        )

        result = self.current_benchmark.copy()
        self.results.append(result)
        self.current_benchmark = None

        return result

    def compare_optimizers(self, test_cases: list[dict[str, Any]]) -> dict[str, Any]:
        """Compare different optimizer configurations."""
        comparison_results: dict[str, Any] = {"test_cases": [], "summary": {}}

        for i, test_case in enumerate(test_cases):
            self.start_benchmark(f"test_case_{i}", test_case)

            # Simulate optimization (in real use, this would run actual optimization)
            optimization_time = np.random.exponential(2.0)  # Simulated time
            objective_value = np.random.exponential(1.0)  # Simulated objective

            self.record_metric("optimization_time", optimization_time)
            self.record_metric("objective_value", objective_value)
            self.record_metric("success", True)

            result = self.end_benchmark()
            comparison_results["test_cases"].append(result)

        # Compute summary statistics
        times = [
            r["metrics"]["optimization_time"] for r in comparison_results["test_cases"]
        ]
        objectives = [
            r["metrics"]["objective_value"] for r in comparison_results["test_cases"]
        ]

        comparison_results["summary"] = {
            "average_time": np.mean(times),
            "std_time": np.std(times),
            "min_time": np.min(times),
            "max_time": np.max(times),
            "average_objective": np.mean(objectives),
            "best_objective": np.min(objectives),
            "worst_objective": np.max(objectives),
        }

        return comparison_results

    def get_performance_report(self) -> dict[str, Any]:
        """Generate comprehensive performance report."""
        if not self.results:
            return {"error": "No benchmark results available"}

        report = {
            "total_benchmarks": len(self.results),
            "benchmark_results": self.results,
            "performance_trends": self._analyze_trends(),
            "recommendations": self._generate_recommendations(),
        }

        return report

    def _analyze_trends(self) -> dict[str, Any]:
        """Analyze performance trends across benchmarks."""
        if len(self.results) < 2:
            return {"insufficient_data": True}

        times = [r.get("total_time", 0) for r in self.results]

        trends = {
            "time_trend": "improving" if times[-1] < times[0] else "degrading",
            "average_time": np.mean(times),
            "time_variability": (
                np.std(times) / np.mean(times) if np.mean(times) > 0 else 0
            ),
        }

        return trends

    def _generate_recommendations(self) -> list[str]:
        """Generate optimization recommendations based on benchmark results."""
        recommendations = []

        if not self.results:
            return ["No benchmark data available for recommendations"]

        # Analyze average performance
        avg_time = np.mean([r.get("total_time", 0) for r in self.results])

        if avg_time > 60:  # More than 1 minute
            recommendations.append(
                "Consider enabling distributed optimization for faster convergence"
            )

        if (
            len([r for r in self.results if r.get("metrics", {}).get("success", False)])
            < len(self.results) * 0.8
        ):
            recommendations.append(
                "Low success rate detected - consider ML acceleration for better initialization"
            )

        # Check memory usage patterns
        memory_metrics = [
            r.get("metrics", {}).get("memory_usage", 0) for r in self.results
        ]
        if memory_metrics and np.mean(memory_metrics) > 0.8:
            recommendations.append(
                "High memory usage detected - consider enabling memory optimization"
            )

        if not recommendations:
            recommendations.append(
                "Performance appears optimal with current configuration"
            )

        return recommendations


class IntegrationHelper:
    """Helper class for integrating distributed and ML features with existing optimizers."""

    @staticmethod
    def enhance_optimizer(
        optimizer,
        config: OptimizationConfig | None = None,
        enable_distributed: bool = True,
        enable_ml: bool = True,
    ) -> Any:
        """
        Enhance an existing optimizer with distributed and ML capabilities.

        Parameters
        ----------
        optimizer : ClassicalOptimizer or RobustHomodyneOptimizer
            Existing optimizer to enhance
        config : OptimizationConfig, optional
            Configuration for enhancements
        enable_distributed : bool
            Whether to enable distributed computing
        enable_ml : bool
            Whether to enable ML acceleration

        Returns
        -------
        Enhanced optimizer with new capabilities
        """
        if config is None:
            config = OptimizationConfig()

        enhanced_optimizer = optimizer

        # Add distributed capabilities
        if enable_distributed and config.is_distributed_enabled():
            try:
                from .distributed import integrate_with_classical_optimizer
                from .distributed import integrate_with_robust_optimizer

                if hasattr(optimizer, "run_classical_optimization_optimized"):
                    enhanced_optimizer = integrate_with_classical_optimizer(
                        enhanced_optimizer, config.get_distributed_config()
                    )
                elif hasattr(optimizer, "run_robust_optimization"):
                    enhanced_optimizer = integrate_with_robust_optimizer(
                        enhanced_optimizer, config.get_distributed_config()
                    )

                logger.info("Added distributed optimization capabilities")
            except ImportError as e:
                logger.warning(f"Failed to add distributed capabilities: {e}")

        # Add ML acceleration
        if enable_ml and config.is_ml_enabled():
            try:
                from .ml_acceleration import enhance_classical_optimizer_with_ml
                from .ml_acceleration import enhance_robust_optimizer_with_ml

                if hasattr(optimizer, "run_classical_optimization_optimized"):
                    enhanced_optimizer = enhance_classical_optimizer_with_ml(
                        enhanced_optimizer, config.get_ml_config()
                    )
                elif hasattr(optimizer, "run_robust_optimization"):
                    enhanced_optimizer = enhance_robust_optimizer_with_ml(
                        enhanced_optimizer, config.get_ml_config()
                    )

                logger.info("Added ML acceleration capabilities")
            except ImportError as e:
                logger.warning(f"Failed to add ML capabilities: {e}")

        return enhanced_optimizer

    @staticmethod
    def create_enhanced_optimizer(
        base_optimizer_class,
        analysis_core,
        config_dict: dict[str, Any],
        optimization_config: OptimizationConfig | None = None,
    ):
        """
        Create a new optimizer instance with enhancements.

        Parameters
        ----------
        base_optimizer_class : type
            Base optimizer class (ClassicalOptimizer or RobustHomodyneOptimizer)
        analysis_core : HomodyneAnalysisCore
            Analysis core instance
        config_dict : dict[str, Any]
            Configuration dictionary for the base optimizer
        optimization_config : OptimizationConfig, optional
            Configuration for distributed and ML enhancements

        Returns
        -------
        Enhanced optimizer instance
        """
        # Create base optimizer
        base_optimizer = base_optimizer_class(analysis_core, config_dict)

        # Enhance with distributed and ML capabilities
        return IntegrationHelper.enhance_optimizer(base_optimizer, optimization_config)

    @staticmethod
    def auto_configure_optimization(
        experimental_conditions: dict[str, Any],
        system_constraints: dict[str, Any] | None = None,
    ) -> OptimizationConfig:
        """
        Automatically configure optimization based on experimental conditions and system capabilities.

        Parameters
        ----------
        experimental_conditions : dict[str, Any]
            Current experimental conditions
        system_constraints : dict[str, Any], optional
            System resource constraints

        Returns
        -------
        OptimizationConfig
            Automatically configured optimization settings
        """
        # Detect system capabilities
        requirements = SystemResourceDetector.check_system_requirements()

        # Start with minimal configuration
        config = OptimizationConfig()

        # Enable features based on system capabilities and experimental conditions
        if requirements["ray_recommended"]:
            # Enable distributed optimization for complex parameter spaces
            param_count = experimental_conditions.get("parameter_count", 3)
            if param_count > 5:
                config.config["distributed_optimization"]["enabled"] = True
                config.config["distributed_optimization"]["backend_preference"] = [
                    "ray",
                    "multiprocessing",
                ]

        if requirements["ml_recommended"]:
            # Enable ML acceleration for repeated optimizations
            optimization_history_size = experimental_conditions.get(
                "optimization_history_size", 0
            )
            if optimization_history_size > 10:
                config.config["ml_acceleration"]["enabled"] = True

        # Optimize configuration for system
        config.config = SystemResourceDetector.optimize_configuration(config.config)

        logger.info(
            "Auto-configured optimization based on system capabilities and experimental conditions"
        )
        return config


def create_comprehensive_benchmark_suite() -> list[dict[str, Any]]:
    """Create a comprehensive benchmark suite for testing optimization performance."""
    benchmark_cases = [
        {
            "name": "Classical Only",
            "distributed_enabled": False,
            "ml_enabled": False,
            "parameter_count": 3,
            "data_complexity": "low",
        },
        {
            "name": "Distributed Classical",
            "distributed_enabled": True,
            "ml_enabled": False,
            "parameter_count": 7,
            "data_complexity": "medium",
        },
        {
            "name": "ML Accelerated",
            "distributed_enabled": False,
            "ml_enabled": True,
            "parameter_count": 3,
            "data_complexity": "medium",
        },
        {
            "name": "Full Enhancement",
            "distributed_enabled": True,
            "ml_enabled": True,
            "parameter_count": 7,
            "data_complexity": "high",
        },
    ]

    return benchmark_cases


def validate_configuration(config: dict[str, Any]) -> tuple[bool, list[str]]:
    """
    Validate optimization configuration for correctness and consistency.

    Parameters
    ----------
    config : dict[str, Any]
        Configuration to validate

    Returns
    -------
    tuple[bool, list[str]]
        (is_valid, list_of_errors)
    """
    errors = []

    # Check distributed configuration
    if config.get("distributed_optimization", {}).get("enabled", False):
        dist_config = config["distributed_optimization"]

        # Validate backend preferences
        valid_backends = ["ray", "mpi", "dask", "multiprocessing"]
        backend_prefs = dist_config.get("backend_preference", [])

        for backend in backend_prefs:
            if backend not in valid_backends:
                errors.append(f"Invalid distributed backend: {backend}")

        # Validate Ray configuration
        if "ray_config" in dist_config:
            ray_config = dist_config["ray_config"]
            if ray_config.get("num_cpus") is not None and ray_config["num_cpus"] <= 0:
                errors.append("Ray num_cpus must be positive")

    # Check ML configuration
    if config.get("ml_acceleration", {}).get("enabled", False):
        ml_config = config["ml_acceleration"]

        # Validate predictor type
        valid_predictors = ["ensemble", "transfer_learning"]
        predictor_type = ml_config.get("predictor_type", "ensemble")
        if predictor_type not in valid_predictors:
            errors.append(f"Invalid predictor type: {predictor_type}")

        # Validate model configuration
        if "ml_model_config" in ml_config:
            model_config = ml_config["ml_model_config"]
            validation_split = model_config.get("validation_split", 0.2)
            if not 0 < validation_split < 1:
                errors.append("Validation split must be between 0 and 1")

    # Check performance monitoring
    if config.get("performance_monitoring", {}).get("enabled", False):
        perf_config = config["performance_monitoring"]

        if "alert_thresholds" in perf_config:
            thresholds = perf_config["alert_thresholds"]
            max_time = thresholds.get("max_optimization_time", 3600)
            if max_time <= 0:
                errors.append("Maximum optimization time must be positive")

    is_valid = len(errors) == 0
    return is_valid, errors


def setup_logging_for_optimization(
    log_level: str = "INFO",
    enable_distributed_logging: bool = True,
    enable_ml_logging: bool = True,
) -> None:
    """
    Set up comprehensive logging for distributed and ML optimization.

    Parameters
    ----------
    log_level : str
        Logging level (DEBUG, INFO, WARNING, ERROR)
    enable_distributed_logging : bool
        Enable detailed distributed computing logs
    enable_ml_logging : bool
        Enable detailed ML acceleration logs
    """
    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Configure specific loggers
    if enable_distributed_logging:
        distributed_logger = logging.getLogger("homodyne.optimization.distributed")
        distributed_logger.setLevel(logging.DEBUG)

    if enable_ml_logging:
        ml_logger = logging.getLogger("homodyne.optimization.ml_acceleration")
        ml_logger.setLevel(logging.DEBUG)

    # Suppress verbose third-party logs
    logging.getLogger("ray").setLevel(logging.WARNING)
    logging.getLogger("distributed").setLevel(logging.WARNING)
    logging.getLogger("sklearn").setLevel(logging.WARNING)

    logger.info("Configured logging for distributed and ML optimization")


# Backward compatibility and easy access functions


def quick_setup_distributed_optimization(
    num_processes: int | None = None, backend: str = "auto"
) -> "DistributedOptimizationCoordinator":
    """
    Quick setup function for distributed optimization.

    Parameters
    ----------
    num_processes : int, optional
        Number of processes to use (auto-detected if None)
    backend : str
        Backend to use ("auto", "ray", "mpi", "multiprocessing")

    Returns
    -------
    DistributedOptimizationCoordinator
        Configured distributed optimization coordinator
    """
    from .distributed import create_distributed_optimizer

    if num_processes is None:
        num_processes = min(psutil.cpu_count() or 4, 8)

    config = {"multiprocessing_config": {"num_processes": num_processes}}

    if backend == "auto":
        backend_preference = ["ray", "multiprocessing"]
    else:
        backend_preference = [backend]

    return create_distributed_optimizer(config, backend_preference)


def quick_setup_ml_acceleration(
    data_path: str | None = None, enable_transfer_learning: bool = True
) -> "MLAcceleratedOptimizer":
    """
    Quick setup function for ML acceleration.

    Parameters
    ----------
    data_path : str, optional
        Path to store ML training data
    enable_transfer_learning : bool
        Whether to enable transfer learning

    Returns
    -------
    MLAcceleratedOptimizer
        Configured ML-accelerated optimizer
    """
    from .ml_acceleration import create_ml_accelerated_optimizer

    config = {
        "enable_transfer_learning": enable_transfer_learning,
        "data_storage_path": data_path or "./ml_optimization_data",
    }

    return create_ml_accelerated_optimizer(config)


# Export key utilities for easy access
__all__ = [
    "IntegrationHelper",
    "OptimizationBenchmark",
    "OptimizationConfig",
    "SystemResourceDetector",
    "create_comprehensive_benchmark_suite",
    "quick_setup_distributed_optimization",
    "quick_setup_ml_acceleration",
    "setup_logging_for_optimization",
    "validate_configuration",
]
