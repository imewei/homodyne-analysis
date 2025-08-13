"""
Configuration Management for Homodyne Scattering Analysis

This module provides centralized configuration management through the ConfigManager
class and associated logging utilities.

Created for: Rheo-SAXS-XPCS Homodyne Analysis
Authors: Wei Chen, Hongrui He
Institution: Argonne National Laboratory & University of Chicago
"""

import json
import logging
import multiprocessing as mp
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Default parallelization setting
DEFAULT_NUM_THREADS = min(16, mp.cpu_count())

# Create module-level logger
logger = logging.getLogger(__name__)


def configure_logging(cfg: Dict[str, Any]) -> logging.Logger:
    """
    Configure centralized logging system with hierarchy and handlers.

    This function sets up a complete logging infrastructure:
    - Creates a logger hierarchy (root + module logger)
    - Sets up RotatingFileHandler with size-based rotation
    - Optionally creates StreamHandler for console output
    - Applies consistent formatting and log levels

    Parameters
    ----------
    cfg : dict
        Logging configuration dictionary with keys:
        - log_to_file: bool, enable file logging
        - log_to_console: bool, enable console logging
        - log_filename: str, log file path
        - level: str, logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR')
        - format: str, log message format string
        - rotation: dict with 'max_bytes' and 'backup_count'

    Returns
    -------
    logging.Logger
        Configured logger instance for reuse
    """
    # Clear any existing handlers to avoid conflicts
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Get or create module logger
    module_logger = logging.getLogger(__name__)
    for handler in module_logger.handlers[:]:
        module_logger.removeHandler(handler)

    # Parse configuration
    log_level = getattr(
        logging, cfg.get("level", "INFO").upper(), logging.INFO
    )
    format_str = cfg.get(
        "format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    formatter = logging.Formatter(format_str)

    # Set up root logger level
    root_logger.setLevel(log_level)
    module_logger.setLevel(log_level)

    # Suppress matplotlib font debug messages to reduce log noise
    logging.getLogger("matplotlib.font_manager").setLevel(logging.WARNING)

    handlers_created = []

    # File handler with rotation
    if cfg.get("log_to_file", False):
        filename = cfg.get("log_filename", "homodyne_analysis.log")
        rotation_config = cfg.get("rotation", {})
        max_bytes = rotation_config.get(
            "max_bytes", 10 * 1024 * 1024
        )  # 10MB default
        backup_count = rotation_config.get("backup_count", 3)

        try:
            file_handler = RotatingFileHandler(
                filename=filename,
                maxBytes=max_bytes,
                backupCount=backup_count,
                encoding="utf-8",
            )
            file_handler.setLevel(log_level)
            file_handler.setFormatter(formatter)

            # Add to both root and module logger
            root_logger.addHandler(file_handler)
            module_logger.addHandler(file_handler)
            handlers_created.append(
                f"RotatingFileHandler({filename}, {max_bytes//1024//1024}MB, {backup_count} backups)"
            )

        except (OSError, IOError) as e:
            logger.warning(f"Failed to create file handler: {e}")
            logger.info("Continuing with console logging only...")

    # Console handler
    if cfg.get("log_to_console", False):
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)

        # Add to both root and module logger
        root_logger.addHandler(console_handler)
        module_logger.addHandler(console_handler)
        handlers_created.append("StreamHandler(console)")

    # Prevent propagation to avoid duplicate messages
    module_logger.propagate = False

    if handlers_created:
        handler_list = ", ".join(handlers_created)
        logger.info(
            f"Logging configured: {handler_list} (level={cfg.get('level', 'INFO')})"
        )

        # Log initial message to verify setup
        module_logger.info(f"Logging system initialized: {handler_list}")
        module_logger.debug(f"Logger hierarchy: root -> {__name__}")
    else:
        logger.info("No logging handlers configured")

    return module_logger


class ConfigManager:
    """
    Manages JSON-based configuration for the analysis pipeline.

    This class provides a centralized configuration system that:
    - Loads and validates JSON configuration files
    - Provides hierarchical parameter organization
    - Supports runtime configuration overrides
    - Manages test configurations for different scenarios

    Configuration Structure:
    - analyzer_parameters: Core analysis settings
    - experimental_data: Data paths and loading options
    - optimization_config: Method settings and hyperparameters
    - parameter_space: Physical bounds and priors
    - performance_settings: Computational optimizations
    """

    def __init__(self, config_file: str = "homodyne_config.json"):
        """
        Initialize configuration manager.

        Parameters
        ----------
        config_file : str
            Path to JSON configuration file
        """
        self.config_file = config_file
        self.config = None
        self.load_config()
        self.validate_config()
        self.setup_logging()

    def load_config(self):
        """Load configuration from JSON file with error handling."""
        try:
            config_path = Path(self.config_file)
            if not config_path.exists():
                raise FileNotFoundError(
                    f"Configuration file not found: {self.config_file}"
                )

            with open(config_path, "r") as f:
                self.config = json.load(f)

            logger.info(f"Configuration loaded from: {self.config_file}")

            # Display version information if available
            if "metadata" in self.config:
                version = self.config["metadata"].get(
                    "config_version", "Unknown"
                )
                logger.info(f"Configuration version: {version}")

        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error: {e}")
            logger.info("Using default configuration...")
            self.config = self._get_default_config()
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            logger.exception(
                "Full traceback for configuration loading failure:"
            )
            logger.info("Using default configuration...")
            self.config = self._get_default_config()

    def validate_config(self):
        """
        Validate configuration parameters for consistency.

        Checks:
        - Required sections are present
        - Frame ranges are valid
        - Physical parameters are reasonable
        - File paths exist (if enabled)
        """
        if not self.config:
            raise ValueError("Configuration is None")

        # Check required sections
        required_sections = [
            "analyzer_parameters",
            "experimental_data",
            "optimization_config",
        ]
        missing = [s for s in required_sections if s not in self.config]
        if missing:
            raise ValueError(f"Missing required sections: {missing}")

        # Validate frame range
        analyzer = self.config["analyzer_parameters"]
        start = analyzer.get("start_frame", 1)
        end = analyzer.get("end_frame", 100)

        if start >= end:
            raise ValueError(f"Invalid frame range: {start} >= {end}")

        # Check minimum frame count
        min_frames = (
            self.config.get("validation_rules", {})
            .get("frame_range", {})
            .get("minimum_frames", 10)
        )
        if end - start < min_frames:
            raise ValueError(
                f"Insufficient frames: {end-start} < {min_frames}"
            )

        # Validate physical parameters
        self._validate_physical_parameters()

        logger.info(
            f"Configuration validated: frames {start}-{end} ({end-start} frames)"
        )

    def _validate_physical_parameters(self):
        """Validate physical parameter ranges."""
        if self.config is None or "analyzer_parameters" not in self.config:
            raise ValueError(
                "Configuration or 'analyzer_parameters' section is missing."
            )

        params = self.config["analyzer_parameters"]

        # Wavevector validation
        q = params.get("wavevector_q", 0.0054)
        if q <= 0:
            raise ValueError(f"Wavevector must be positive: {q}")
        if q > 1.0:
            logger.warning(f"Large wavevector: {q} Å⁻¹ (typical: 0.001-0.1)")

        # Time step validation
        dt = params.get("dt", 0.1)
        if dt <= 0:
            raise ValueError(f"Time step must be positive: {dt}")

        # Gap size validation
        h = params.get("stator_rotor_gap", 2000000)
        if h <= 0:
            raise ValueError(f"Gap size must be positive: {h}")

    def setup_logging(self) -> Optional[logging.Logger]:
        """Configure logging based on configuration using centralized configure_logging()."""
        if self.config is None:
            logger.warning("Configuration is None, skipping logging setup.")
            return None

        log_config = self.config.get("logging", {})

        # Skip logging setup if neither file nor console logging is enabled
        if not log_config.get("log_to_file", False) and not log_config.get(
            "log_to_console", False
        ):
            return None

        # Use the centralized configure_logging function
        try:
            configured_logger = configure_logging(log_config)
            return configured_logger
        except Exception as e:
            logger.warning(f"Failed to configure logging: {e}")
            logger.exception(
                "Full traceback for logging configuration failure:"
            )
            logger.info("Continuing without logging...")
            return None

    def get(self, *keys, default=None):
        """
        Get nested configuration value.

        Parameters
        ----------
        *keys : str
            Sequence of nested keys
        default : any
            Default value if key not found

        Returns
        -------
        Configuration value or default
        """
        try:
            value = self.config
            for key in keys:
                if value is None or not isinstance(value, dict):
                    return default
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default

    def get_angle_filtering_config(self) -> Dict[str, Any]:
        """
        Get angle filtering configuration with defaults.

        Returns
        -------
        dict
            Angle filtering configuration including:
            - enabled: bool, whether angle filtering is enabled
            - target_ranges: list of dicts with min_angle and max_angle
            - fallback_to_all_angles: bool, whether to use all angles if no targets found
        """
        angle_filtering = self.get(
            "optimization_config", "angle_filtering", default={}
        )

        # Ensure angle_filtering is a dictionary for unpacking
        if not isinstance(angle_filtering, dict):
            angle_filtering = {}

        # Provide sensible defaults if configuration is missing or incomplete
        default_config = {
            "enabled": True,
            "target_ranges": [
                {"min_angle": -10.0, "max_angle": 10.0},
                {"min_angle": 170.0, "max_angle": 190.0},
            ],
            "fallback_to_all_angles": True,
        }

        # Merge with defaults
        result = {**default_config, **angle_filtering}

        # Validate target_ranges structure
        if "target_ranges" in result:
            valid_ranges = []
            for range_config in result["target_ranges"]:
                if (
                    isinstance(range_config, dict)
                    and "min_angle" in range_config
                    and "max_angle" in range_config
                ):
                    valid_ranges.append(
                        {
                            "min_angle": float(range_config["min_angle"]),
                            "max_angle": float(range_config["max_angle"]),
                        }
                    )
                else:
                    logger.warning(
                        f"Invalid angle range configuration: {range_config}"
                    )
            result["target_ranges"] = valid_ranges

        return result

    def is_angle_filtering_enabled(self) -> bool:
        """
        Check if angle filtering is enabled in configuration.

        Returns
        -------
        bool
            True if angle filtering should be used, False otherwise
        """
        return self.get_angle_filtering_config().get("enabled", True)

    def get_target_angle_ranges(self) -> List[Tuple[float, float]]:
        """
        Get list of target angle ranges for optimization.

        Returns
        -------
        list of tuple
            List of (min_angle, max_angle) tuples in degrees
        """
        config = self.get_angle_filtering_config()
        ranges = config.get("target_ranges", [])

        return [(r["min_angle"], r["max_angle"]) for r in ranges]

    def should_fallback_to_all_angles(self) -> bool:
        """
        Check if system should fallback to all angles when no targets found.

        Returns
        -------
        bool
            True if should fallback to all angles, False to raise error
        """
        return self.get_angle_filtering_config().get(
            "fallback_to_all_angles", True
        )

    def get_test_config(self, test_name: str) -> Dict[str, Any]:
        """
        Get predefined test configuration.

        Parameters
        ----------
        test_name : str
            Name of test configuration

        Returns
        -------
        dict
            Test-specific configuration
        """
        if self.config is None:
            raise ValueError(
                "Configuration is None. Cannot retrieve test configurations."
            )
        configs = self.config.get("test_configurations", {})

        if test_name not in configs:
            available = list(configs.keys())
            raise ValueError(
                f"Test '{test_name}' not found. Available: {available}"
            )

        return configs[test_name]

    def is_static_mode_enabled(self) -> bool:
        """
        Check if static mode is enabled in configuration.

        Returns
        -------
        bool
            True if static mode is enabled, False otherwise
        """
        return self.get("analysis_settings", "static_mode", default=False)

    def get_analysis_mode(self) -> str:
        """
        Get the current analysis mode.

        Returns
        -------
        str
            "static" if static mode is enabled, "laminar_flow" otherwise
        """
        return "static" if self.is_static_mode_enabled() else "laminar_flow"

    def get_effective_parameter_count(self) -> int:
        """
        Get the effective number of model parameters based on analysis mode.

        Returns
        -------
        int
            Number of parameters used in the analysis:
            - Static mode: 3 (only diffusion parameters)
            - Laminar flow mode: 7 (all parameters)
        """
        return 3 if self.is_static_mode_enabled() else 7

    def get_analysis_settings(self) -> Dict[str, Any]:
        """
        Get analysis settings with defaults.

        Returns
        -------
        Dict[str, Any]
            Analysis settings including static_mode flag and descriptions
        """
        analysis_settings = self.get("analysis_settings", default={})

        # Provide sensible defaults
        default_settings = {
            "static_mode": False,
            "model_description": {
                "static_case": (
                    "g₁(t₁,t₂) = exp(-q² ∫ᵗ²ᵗ¹ D(t)dt) = g₁_diff(t₁,t₂), g₂(t₁,t₂) = [g₁(t₁,t₂)]²"
                ),
                "laminar_flow_case": (
                    "g₁(t₁,t₂) = g₁_diff(t₁,t₂) × g₁_shear(t₁,t₂) where g₁_shear = [sinc(Φ)]² and Φ = (1/2π)qL cos(φ₀-φ) ∫|t₂-t₁| γ̇(t')dt'"
                ),
            },
        }

        # Merge with defaults
        result = {**default_settings, **analysis_settings}
        return result

    def _get_default_config(self) -> Dict[str, Any]:
        """Generate minimal default configuration."""
        return {
            "metadata": {
                "config_version": "5.1-default",
                "description": "Emergency fallback configuration",
            },
            "analyzer_parameters": {
                "temporal": {
                    "dt": 0.1,
                    "start_frame": 1001,
                    "end_frame": 2000,
                },
                "scattering": {"wavevector_q": 0.0054},
                "geometry": {"stator_rotor_gap": 2000000},
                "computational": {
                    "num_threads": DEFAULT_NUM_THREADS,
                    "auto_detect_cores": False,
                    "max_threads_limit": 128,
                },
            },
            "experimental_data": {
                "data_folder_path": "./data/C020/",
                "data_file_name": "default_data.hdf",
                "phi_angles_path": "./data/C020/",
                "phi_angles_file": "phi_list.txt",
                "exchange_key": "exchange",
                "cache_file_path": ".",
                "cache_filename_template": (
                    "cached_c2_frames_{start_frame}_{end_frame}.npz"
                ),
            },
            "analysis_settings": {
                "static_mode": False,
                "model_description": {
                    "static_case": (
                        "g₁(t₁,t₂) = exp(-q² ∫ᵗ²ᵗ¹ D(t)dt) = g₁_diff(t₁,t₂), g₂(t₁,t₂) = [g₁(t₁,t₂)]²"
                    ),
                    "laminar_flow_case": (
                        "g₁(t₁,t₂) = g₁_diff(t₁,t₂) × g₁_shear(t₁,t₂) where g₁_shear = [sinc(Φ)]² and Φ = (1/2π)qL cos(φ₀-φ) ∫|t₂-t₁| γ̇(t')dt'"
                    ),
                },
            },
            "initial_parameters": {
                "values": [1324.1, -0.014, -0.674361, 0.003, -0.909, 0.0, 0.0],
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
            "optimization_config": {
                "angle_filtering": {
                    "enabled": True,
                    "target_ranges": [
                        {"min_angle": -10.0, "max_angle": 10.0},
                        {"min_angle": 170.0, "max_angle": 190.0},
                    ],
                    "fallback_to_all_angles": True,
                },
                "classical_optimization": {
                    "methods": ["Nelder-Mead"],
                    "method_options": {
                        "Nelder-Mead": {
                            "maxiter": 5000,
                            "xatol": 1e-8,
                            "fatol": 1e-8,
                        }
                    },
                },
                "bayesian_optimization": {
                    "n_calls": 20,
                    "n_initial_points": 5,
                },
                "bayesian_inference": {"mcmc_draws": 1000, "mcmc_tune": 500},
            },
            "parameter_space": {
                "bounds": [
                    {
                        "name": "D0",
                        "min": 1e-3,
                        "max": 1e6,
                        "type": "log-uniform",
                    },
                    {
                        "name": "alpha",
                        "min": -2.0,
                        "max": 2.0,
                        "type": "uniform",
                    },
                    {
                        "name": "D_offset",
                        "min": -5000,
                        "max": 5000,
                        "type": "uniform",
                    },
                    {
                        "name": "gamma_dot_t0",
                        "min": 1e-6,
                        "max": 1.0,
                        "type": "log-uniform",
                    },
                    {
                        "name": "beta",
                        "min": -2.0,
                        "max": 2.0,
                        "type": "uniform",
                    },
                    {
                        "name": "gamma_dot_t_offset",
                        "min": -0.1,
                        "max": 0.1,
                        "type": "uniform",
                    },
                    {
                        "name": "phi0",
                        "min": -15.0,
                        "max": 15.0,
                        "type": "uniform",
                    },
                ]
            },
            "validation_rules": {"frame_range": {"minimum_frames": 10}},
            "performance_settings": {
                "parallel_execution": True,
                "use_threading": True,
                "optimization_counter_log_frequency": 100,
            },
            "advanced_settings": {
                "data_loading": {
                    "use_diagonal_correction": True,
                    "vectorized_diagonal_fix": True,
                },
                "chi_squared_calculation": {
                    "scaling_optimization": True,
                    "uncertainty_estimation_factor": 0.1,
                    "minimum_sigma": 1e-10,
                    "validity_check": {
                        "check_positive_D0": True,
                        "check_positive_gamma_dot_t0": True,
                        "check_positive_time_dependent": True,
                        "check_parameter_bounds": True,
                    },
                },
            },
            "test_configurations": {
                "production": {
                    "description": "Standard production configuration",
                    "classical_methods": ["Nelder-Mead"],
                    "bo_n_calls": 20,
                    "mcmc_draws": 1000,
                }
            },
        }
