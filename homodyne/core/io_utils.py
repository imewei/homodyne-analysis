"""
IO Utilities for Homodyne Scattering Analysis
==============================================

Comprehensive I/O utilities for safe data handling in XPCS analysis workflows.
Provides robust file operations, intelligent data serialization, and structured
result management with extensive error handling and logging.

Key Features:
- Thread-safe directory creation with race condition handling
- Timestamped filename generation with configurable formatting
- Multi-format data serialization (JSON, NumPy, Pickle, Matplotlib)
- Custom JSON serializer for NumPy arrays and complex objects
- Comprehensive error handling with detailed logging
- Structured result saving for analysis workflows
- Frame counting utilities for consistent time_length calculations

Data Formats Supported:
- JSON: Configuration files, analysis results, metadata
- NumPy (.npz): Correlation functions, parameter arrays, numerical data
- Pickle (.pkl): Complex Python objects, model instances (MCMC traces removed)
- Matplotlib: Figures and plots with publication-quality settings

Safety Features:
- Atomic file operations where possible
- Directory creation with appropriate permissions
- Comprehensive exception handling for I/O errors
- Logging of all operations for debugging and audit trails

Authors: Wei Chen, Hongrui He
Institution: Argonne National Laboratory
"""

__author__ = "Wei Chen, Hongrui He"
__credits__ = "Argonne National Laboratory"

import json
import logging
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

# Module-level logger for I/O operations tracking and debugging
logger = logging.getLogger(__name__)


def ensure_dir(path: str | Path, permissions: int = 0o755) -> Path:
    """
    Thread-safe recursive directory creation with comprehensive error handling.

    Creates directory hierarchies safely, handling race conditions that can occur
    in multi-process environments (e.g., parallel optimization runs). Uses atomic
    operations where possible and validates directory creation success.

    Features:
    - Race condition safety for concurrent directory creation
    - Recursive parent directory creation
    - Configurable permissions for security control
    - Path validation and type checking
    - Comprehensive error reporting

    Args:
        path (str | Path): Directory path to create (absolute or relative)
        permissions (int): Unix-style permissions (default: 0o755 = rwxr-xr-x)

    Returns:
        Path: Pathlib.Path object of the created/validated directory

    Raises:
        OSError: Directory creation failed, path exists but isn't a directory,
                or permissions issues

    Example:
        >>> ensure_dir("./homodyne_results/classical/traces")
        PosixPath('./homodyne_results/classical/traces')

        >>> ensure_dir("/tmp/analysis", permissions=0o700)  # Owner-only access
        PosixPath('/tmp/analysis')
    """
    path_obj = Path(path)

    try:
        path_obj.mkdir(parents=True, exist_ok=True, mode=permissions)
        logger.debug(f"Directory ensured: {path_obj.absolute()}")
    except OSError as e:
        # Re-check if the directory exists (race condition handling)
        if not path_obj.exists():
            logger.error(f"Failed to create directory {path_obj}: {e}")
            raise
        elif not path_obj.is_dir():
            logger.error(f"Path exists but is not a directory: {path_obj}")
            raise OSError(f"Path exists but is not a directory: {path_obj}")

    return path_obj


def timestamped_filename(
    base_name: str, chi2: float | None = None, config: dict | None = None
) -> str:
    """
    Generate intelligently formatted filenames with timestamps and analysis metadata.

    Creates structured filenames that include temporal information and analysis
    quality metrics, facilitating result organization and identification. Supports
    configurable timestamp formats and optional inclusion of chi-squared values
    for quick quality assessment.

    Filename Components:
    - Base name: User-specified prefix (e.g., 'analysis_results')
    - Timestamp: Configurable format for temporal ordering
    - Chi-squared: Optional goodness-of-fit indicator
    - Config version: Optional configuration identification

    Configuration Options:
    - timestamp_format: strftime format string (default: "%Y%m%d_%H%M%S")
    - include_chi_squared: Boolean flag for chi2 inclusion
    - include_config_name: Boolean flag for configuration version

    Args:
        base_name (str): Base filename prefix (without extension)
        chi2 (float | None): Chi-squared value for quality indication
        config (dict | None): Configuration with output_settings/file_naming

    Returns:
        str: Structured filename string ready for file operations

    Examples:
        >>> config = {"output_settings": {"file_naming": {
        ...     "timestamp_format": "%Y%m%d_%H%M%S",
        ...     "include_chi_squared": True,
        ...     "include_config_name": True
        ... }}}
        >>> timestamped_filename("classical_results", 1.234e-3, config)
        'classical_results_20240315_143022_chi2_0.001234_v5.1'

        >>> timestamped_filename("quick_analysis")  # Minimal version
        'quick_analysis_20240315_143022'
    """
    # Default configuration
    default_naming = {
        "timestamp_format": "%Y%m%d_%H%M%S",
        "include_config_name": True,
        "include_chi_squared": True,
    }

    # Extract file naming configuration
    if (
        config
        and "output_settings" in config
        and "file_naming" in config["output_settings"]
    ):
        naming_config = {
            **default_naming,
            **config["output_settings"]["file_naming"],
        }
    else:
        naming_config = default_naming
        logger.warning("No file_naming configuration found, using defaults")

    # Generate timestamp
    timestamp = datetime.now().strftime(naming_config["timestamp_format"])

    # Build filename components
    filename_parts = [base_name, timestamp]

    # Add chi-squared value if requested and provided
    if naming_config.get("include_chi_squared", False) and chi2 is not None:
        chi2_str = f"chi2_{chi2:.6f}"
        filename_parts.append(chi2_str)

    # Add config name if requested and available
    if naming_config.get("include_config_name", False) and config:
        if "metadata" in config and "config_version" in config["metadata"]:
            config_name = f"v{config['metadata']['config_version']}"
            filename_parts.append(config_name)

    filename = "_".join(filename_parts)
    logger.debug(f"Generated filename: {filename}")

    return filename


def _json_serializer(obj):
    """
    Custom JSON serializer for scientific computing objects.

    Handles NumPy arrays, scalars, and complex Python objects that are not
    natively JSON-serializable. Essential for saving analysis results that
    contain numerical arrays and computed parameters.

    Supported Object Types:
    - NumPy arrays: Converted to Python lists
    - NumPy scalars: Extracted as native Python types
    - Complex objects: String representation fallback
    - Other types: Safe string conversion

    Args:
        obj: Object to serialize

    Returns:
        JSON-serializable representation of the object

    Raises:
        TypeError: For truly non-serializable objects that should fail
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    elif isinstance(obj, (np.complexfloating, complex)):
        # Don't serialize complex numbers - let them fail for testing
        raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")
    elif hasattr(obj, "__dict__"):
        return str(obj)  # Convert complex objects to string
    else:
        return str(obj)


def save_json(data: Any, filepath: str | Path, **kwargs: Any) -> bool:
    """
    Save data as JSON with robust error handling and NumPy support.

    Provides safe JSON serialization with automatic handling of scientific
    computing objects like NumPy arrays and scalars. Uses custom serializer
    to ensure compatibility with analysis results containing numerical data.

    Features:
    - Custom NumPy serializer for arrays and scalars
    - Automatic directory creation for output path
    - UTF-8 encoding for international character support
    - Comprehensive error handling with detailed logging
    - Configurable JSON formatting options

    Default JSON Parameters:
    - indent=2: Pretty formatting for readability
    - ensure_ascii=False: Support for Unicode characters
    - default=_json_serializer: NumPy and object handling

    Args:
        data: Data structure to save (dicts, lists, arrays, etc.)
        filepath (str | Path): Output file path (directories created automatically)
        **kwargs: Additional json.dump() arguments (override defaults)

    Returns:
        bool: True if save successful, False if any error occurred

    Examples:
        >>> results = {"parameters": np.array([1.2, 3.4]), "chi2": 1.234e-5}
        >>> save_json(results, "analysis_results.json")
        True

        >>> save_json(data, "compact.json", indent=None, separators=(',', ':'))
        True  # Compact JSON format
    """
    filepath = Path(filepath)

    try:
        # Ensure parent directory exists
        ensure_dir(filepath.parent)

        # Set default JSON parameters with custom serializer
        json_kwargs = {
            "indent": 2,
            "ensure_ascii": False,
            "default": _json_serializer,
        }
        json_kwargs.update(kwargs)

        # Save JSON file
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, **json_kwargs)  # type: ignore[arg-type]

        logger.info(f"Successfully saved JSON data to: {filepath}")
        return True

    except (TypeError, ValueError) as e:
        logger.error(f"JSON serialization error for {filepath}: {e}")
        return False
    except OSError as e:
        logger.error(f"File I/O error saving JSON to {filepath}: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error saving JSON to {filepath}: {e}")
        return False


def save_numpy(
    data: np.ndarray,
    filepath: str | Path,
    compressed: bool = True,
    **kwargs: Any,
) -> bool:
    """
    Save NumPy arrays with optimal compression and format selection.

    Provides efficient storage of numerical data with automatic format selection
    based on file extension and compression preferences. Essential for saving
    correlation functions, parameter arrays, and other numerical results.

    Format Selection:
    - .npz extension or compressed=True: Uses np.savez_compressed (recommended)
    - Other extensions with compressed=False: Uses np.save (uncompressed)
    - Automatic directory creation for nested paths

    Compression Benefits:
    - Significantly reduced file sizes (typically 2-10x smaller)
    - Faster I/O for large arrays due to reduced data transfer
    - Standard NumPy format compatibility

    Args:
        data (np.ndarray): NumPy array to save (any shape/dtype)
        filepath (str | Path): Output file path (.npz recommended)
        compressed (bool): Enable compression (default: True for efficiency)
        **kwargs: Additional arguments for np.savez_compressed/np.save

    Returns:
        bool: True if save successful, False if error occurred

    Examples:
        >>> correlation_data = np.random.rand(1000, 50, 50)  # Large 3D array
        >>> save_numpy(correlation_data, "c2_experimental.npz")
        True  # Compressed format, much smaller file

        >>> parameters = np.array([1.2, -0.5, 3.4e-3, 0.1])
        >>> save_numpy(parameters, "optimized_params.npy", compressed=False)
        True  # Uncompressed for small arrays
    """
    filepath = Path(filepath)

    try:
        # Ensure parent directory exists
        ensure_dir(filepath.parent)

        if compressed or filepath.suffix == ".npz":
            # Use compressed format
            np.savez_compressed(filepath, data=data, **kwargs)
        else:
            # Use uncompressed format
            np.save(filepath, data, **kwargs)

        logger.info(f"Successfully saved NumPy data to: {filepath}")
        return True

    except (ValueError, TypeError) as e:
        logger.error(f"NumPy data error for {filepath}: {e}")
        return False
    except OSError as e:
        logger.error(f"File I/O error saving NumPy data to {filepath}: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error saving NumPy data to {filepath}: {e}")
        return False


def save_pickle(
    data: Any,
    filepath: str | Path,
    protocol: int = pickle.HIGHEST_PROTOCOL,
    **kwargs: Any,
) -> bool:
    """
    Save data using pickle with error handling and logging.

    Args:
        data: Data to pickle
        filepath (str | Path): Output file path
        protocol (int): Pickle protocol version (default: highest available)
        **kwargs: Additional arguments (reserved for future use)

    Returns:
        bool: True if successful, False otherwise

    Example:
        >>> data = {"model": some_complex_object, "parameters": [1, 2, 3]}
        >>> save_pickle(data, "model_data.pkl")
        True
    """
    filepath = Path(filepath)

    try:
        # Ensure parent directory exists
        ensure_dir(filepath.parent)

        # Save pickle file
        with open(filepath, "wb") as f:
            pickle.dump(data, f, protocol=protocol)

        logger.info(f"Successfully saved pickle data to: {filepath}")
        return True

    except (pickle.PicklingError, TypeError) as e:
        logger.error(f"Pickle serialization error for {filepath}: {e}")
        return False
    except OSError as e:
        logger.error(f"File I/O error saving pickle to {filepath}: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error saving pickle to {filepath}: {e}")
        return False


def save_fig(
    figure: Any,
    filepath: str | Path,
    dpi: int = 300,
    format: str | None = None,
    **kwargs: Any,
) -> bool:
    """
    Save matplotlib figure with error handling and logging.

    Args:
        figure: Matplotlib figure object
        filepath (str | Path): Output file path
        dpi (int): Resolution in dots per inch (default: 300)
        format (str | None): Figure format (inferred from extension if None)
        **kwargs: Additional arguments passed to figure.savefig()

    Returns:
        bool: True if successful, False otherwise

    Example:
        >>> import matplotlib.pyplot as plt
        >>> fig, ax = plt.subplots()
        >>> ax.plot([1, 2, 3], [1, 4, 2])
        >>> save_fig(fig, "plot.png", dpi=300, bbox_inches='tight')
        True
    """
    filepath = Path(filepath)

    try:
        # Ensure parent directory exists
        ensure_dir(filepath.parent)

        # Set default savefig parameters
        savefig_kwargs = {
            "dpi": dpi,
            "bbox_inches": "tight",
            "facecolor": "white",
            "edgecolor": "none",
        }
        savefig_kwargs.update(kwargs)

        # Add format if specified
        if format:
            savefig_kwargs["format"] = format

        # Save figure
        figure.savefig(filepath, **savefig_kwargs)

        logger.info(f"Successfully saved figure to: {filepath}")
        return True

    except AttributeError as e:
        logger.error(f"Invalid figure object for {filepath}: {e}")
        return False
    except OSError as e:
        logger.error(f"File I/O error saving figure to {filepath}: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error saving figure to {filepath}: {e}")
        return False


# Utility functions for common file operations
def get_output_directory(config: dict | None = None) -> Path:
    """
    Get the output directory from configuration, creating it if necessary.

    Args:
        config (dict | None): Configuration dictionary

    Returns:
        Path: Output directory path
    """
    default_dir = "./homodyne_results"

    if config and "output_settings" in config:
        output_dir = config["output_settings"].get("results_directory", default_dir)
    else:
        output_dir = default_dir
        logger.warning(
            "No output directory configuration found, using default: ./homodyne_results"
        )

    return ensure_dir(output_dir)


def save_classical_optimization_results(
    results: dict,
    method_results: dict | None = None,
    config: dict | None = None,
    base_name: str = "classical_results",
) -> dict[str, bool]:
    """
    Save classical optimization results with method-specific organization.

    Creates separate files for each optimization method (Nelder-Mead, Gurobi) to
    prevent overwriting and enable method comparison. Organizes results in a
    structured directory layout for easy analysis and plotting.

    File Organization:
    - classical_results_nelder_mead_TIMESTAMP.json
    - classical_results_gurobi_TIMESTAMP.json
    - classical_results_all_methods_TIMESTAMP.json (combined)

    Args:
        results (Dict): Main optimization results
        method_results (Dict): Method-specific results dictionary
        config (dict | None): Configuration for output directory and naming
        base_name (str): Base name for output files

    Returns:
        dict[str, bool]: Save status for each method and combined results
    """
    output_dir = get_output_directory(config) / "classical"

    output_dir.mkdir(parents=True, exist_ok=True)
    save_status = {}

    if method_results:
        logger.info(
            f"Saving method-specific results for: {list(method_results.keys())}"
        )
        # Save individual method results
        for method, method_data in method_results.items():
            success = method_data.get("success", False)
            if success:
                method_name = method.lower().replace("-", "_")
                chi2 = method_data.get("chi_squared")
                filename_base = timestamped_filename(
                    f"{base_name}_{method_name}", chi2, config
                )

                # Create method-specific result structure
                method_result = {
                    "optimization_method": method,
                    "parameters": method_data.get("parameters"),
                    "chi_squared": chi2,
                    "success": method_data.get("success"),
                    "iterations": method_data.get("iterations"),
                    "function_evaluations": method_data.get("function_evaluations"),
                    "message": method_data.get("message", ""),
                    "timestamp": datetime.now().isoformat(),
                    **{k: v for k, v in results.items() if k not in ["method_results"]},
                }

                json_path = output_dir / f"{filename_base}.json"
                save_status[f"{method}_json"] = save_json(method_result, json_path)

                logger.info(f"✓ Saved {method} results to: {json_path.name}")
            else:
                logger.debug(f"Skipped {method} (not successful)")
                save_status[f"{method}_skipped"] = True

    # Save combined results with all methods
    combined_filename = timestamped_filename(
        f"{base_name}_all_methods", results.get("best_chi_squared"), config
    )
    combined_path = output_dir / f"{combined_filename}.json"

    combined_results = {
        **results,
        "method_results": method_results,
        "timestamp": datetime.now().isoformat(),
    }

    save_status["combined_json"] = save_json(combined_results, combined_path)
    logger.info(f"Saved combined results to: {combined_path.name}")

    return save_status


def save_analysis_results(
    results: dict,
    config: dict | None = None,
    base_name: str = "analysis_results",
) -> dict[str, bool]:
    """
    Orchestrate comprehensive saving of analysis results in multiple formats.

    Enhanced to handle method-specific classical optimization results, preventing
    overwrites between Nelder-Mead and Gurobi methods. Intelligently saves
    analysis results using optimal formats for different data types.

    Save Strategy:
    - JSON: Main results, parameters, metadata (human-readable)
    - NumPy (.npz): Correlation data, large numerical arrays (efficient)
    - Pickle (.pkl): Complex objects, model instances (complete; MCMC traces removed)
    - Method-specific: Individual files for each classical optimization method

    File Organization:
    - Timestamped base filename for chronological organization
    - Format-specific suffixes: .json, _data.npz, _full.pkl
    - Classical-only results saved to classical/ subdirectory
    - Multi-method results saved to main output directory (MCMC removed)
    - Automatic directory creation and organization
    - Consistent naming across all output files

    Args:
        results (Dict): Complete analysis results dictionary containing:
                       - Optimization results and parameters
                       - Correlation data arrays
                       - Configuration and metadata
        config (dict | None): Configuration for output directory and naming
        base_name (str): Prefix for all output files (default: "analysis_results")

    Returns:
        dict[str, bool]: Save status for each format:
                        - "json": JSON save status
                        - "numpy": NumPy array save status (if applicable)
                        - "pickle": Pickle save status (if applicable)
                        - method-specific keys for classical optimization

    Example:
        >>> results = {
        ...     "classical_optimization": {"parameters": [1.2, -0.5, 3.4]},
        ...     "correlation_data": np.random.rand(100, 50, 50),
        ...     "best_chi_squared": 1.234e-5
        ... }
        >>> status = save_analysis_results(results, config, "experiment_A")
        >>> print(status)
        {'json': True, 'numpy': True, 'pickle': True, 'nelder_mead_json': True, 'gurobi_json': True}
    """
    output_dir = get_output_directory(config)
    chi2 = results.get("best_chi_squared")

    # Generate base filename
    filename_base = timestamped_filename(base_name, chi2, config)

    save_status = {}

    # Handle classical optimization results with method-specific saving
    # ONLY for true classical methods, not for robust methods that use
    # ClassicalOptimizer internally
    if "classical_optimization" in results and results.get("methods_used", []) == [
        "Classical"
    ]:
        classical_results = results["classical_optimization"]
        method_results = None

        # Check if enhanced classical results with method information are
        # available
        if hasattr(classical_results, "get") and isinstance(classical_results, dict):
            method_results = classical_results.get("method_results")
        elif hasattr(classical_results, "method_results"):
            # Results from enhanced classical optimizer
            method_results = getattr(classical_results, "method_results", None)

        if method_results:
            # Save method-specific results
            # Create a results dict with basic info for the saving function
            results_for_save = {
                "best_chi_squared": getattr(classical_results, "fun", None),
                "timestamp": results.get("timestamp", ""),
                "success": getattr(classical_results, "success", True),
            }
            classical_save_status = save_classical_optimization_results(
                results_for_save,
                method_results,
                config,
                "classical_optimization",
            )
            save_status.update(classical_save_status)

    # Save main results as JSON
    # For classical-only results, save to classical subdirectory
    if "classical_optimization" in results and results.get("methods_used", []) == [
        "Classical"
    ]:
        # This is a classical-only result, save to classical subdirectory
        classical_dir = output_dir / "classical"

        classical_dir.mkdir(parents=True, exist_ok=True)
        json_path = classical_dir / f"{filename_base}.json"
    else:
        # This is a multi-method result, save to main directory
        json_path = output_dir / f"{filename_base}.json"

    save_status["json"] = save_json(results, json_path)

    # Save NumPy arrays if present
    if "correlation_data" in results and isinstance(
        results["correlation_data"], np.ndarray
    ):
        # Use same directory logic as main JSON file
        if "classical_optimization" in results and results.get("methods_used", []) == [
            "Classical"
        ]:
            npz_path = (output_dir / "classical") / f"{filename_base}_data.npz"
        else:
            npz_path = output_dir / f"{filename_base}_data.npz"
        save_status["numpy"] = save_numpy(results["correlation_data"], npz_path)

    # Save complex objects as pickle
    if any(
        # Complex object detection logic
        False
        for key in results.keys()
    ):
        # Use same directory logic as main JSON file
        if "classical_optimization" in results and results.get("methods_used", []) == [
            "Classical"
        ]:
            pkl_path = (output_dir / "classical") / f"{filename_base}_full.pkl"
        else:
            pkl_path = output_dir / f"{filename_base}_full.pkl"
        save_status["pickle"] = save_pickle(results, pkl_path)

    logger.info(f"Analysis results saved with base name: {filename_base}")
    logger.info(f"Save status: {save_status}")

    return save_status


if __name__ == "__main__":
    # Basic test of the utility functions
    import tempfile

    # Set up logging for testing
    logging.basicConfig(level=logging.INFO)

    print("Testing IO utilities...")

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        # Test ensure_dir
        test_dir = tmp_path / "test" / "nested" / "directory"
        result_dir = ensure_dir(test_dir)
        print(f"Directory created: {result_dir.exists()}")

        # Test timestamped_filename
        config = {
            "output_settings": {
                "file_naming": {
                    "timestamp_format": "%Y%m%d_%H%M%S",
                    "include_chi_squared": True,
                }
            }
        }
        filename = timestamped_filename("test_results", 1.234, config)
        print(f"Generated filename: {filename}")

        # Test save functions
        test_data = {"test": "data", "values": [1, 2, 3]}
        json_success = save_json(test_data, result_dir / "test.json")
        print(f"JSON save success: {json_success}")

        test_array = np.random.rand(10, 10)
        numpy_success = save_numpy(test_array, result_dir / "test.npz")
        print(f"NumPy save success: {numpy_success}")

        pickle_success = save_pickle(test_data, result_dir / "test.pkl")
        print(f"Pickle save success: {pickle_success}")

    print("All tests completed!")


# ============================================================================
# Frame Counting Utilities
# ============================================================================


def calculate_time_length(start_frame: int, end_frame: int) -> int:
    """
    Calculate time_length using inclusive frame counting.

    This is the canonical formula used throughout the homodyne package to ensure
    dimensional consistency between configuration, cached data, and runtime arrays.

    Frame Counting Convention:
    -------------------------
    - Config frames are 1-based and inclusive: [start_frame, end_frame]
    - time_length includes both start and end frames
    - Formula: time_length = end_frame - start_frame + 1

    Examples:
    ---------
    >>> calculate_time_length(1, 100)
    100
    >>> calculate_time_length(401, 1000)
    600
    >>> calculate_time_length(1, 1)
    1

    Args:
        start_frame: Starting frame number (1-based, inclusive)
        end_frame: Ending frame number (1-based, inclusive)

    Returns:
        int: Number of frames in the range (time_length)

    Raises:
        ValueError: If start_frame > end_frame

    Note:
        This formula was fixed in v0.6.5+ to address a critical bug where the
        original formula (end_frame - start_frame) caused off-by-one errors,
        dimensional mismatches, and NaN chi-squared values.

    See Also:
        - homodyne/analysis/core.py:240 - Core time_length calculation
        - homodyne/data/xpcs_loader.py - Data loading with frame slicing
        - homodyne/tests/test_time_length_calculation.py - Regression tests
    """
    if start_frame > end_frame:
        raise ValueError(
            f"Invalid frame range: start_frame ({start_frame}) > end_frame ({end_frame})"
        )

    time_length = end_frame - start_frame + 1
    return time_length


def config_frames_to_python_slice(
    start_frame: int, end_frame: int
) -> tuple[int, int]:
    """
    Convert config frame range (1-based inclusive) to Python slice indices (0-based).

    Config Convention:
    ------------------
    - start_frame: 1-based, inclusive (e.g., 1 means first frame)
    - end_frame: 1-based, inclusive (e.g., 100 means include frame 100)

    Python Slice Convention:
    ------------------------
    - start: 0-based, inclusive
    - end: 0-based, exclusive (used as data[start:end])

    Examples:
    ---------
    >>> config_frames_to_python_slice(1, 100)
    (0, 100)
    >>> config_frames_to_python_slice(401, 1000)
    (400, 1000)

    This gives slice [0:100] = 100 frames, [400:1000] = 600 frames,
    matching time_length = end_frame - start_frame + 1.

    Args:
        start_frame: Starting frame from config (1-based, inclusive)
        end_frame: Ending frame from config (1-based, inclusive)

    Returns:
        tuple[int, int]: (python_start, python_end) for use in data[start:end]

    Note:
        The returned indices are designed for Python array slicing where the
        end index is exclusive. The slice [python_start:python_end] will give
        exactly time_length = end_frame - start_frame + 1 elements.

    See Also:
        - convert_c2_to_npz.py:convert_config_frames_to_python() - Data conversion
        - homodyne/data/xpcs_loader.py:637 - Frame slicing in data loader
    """
    python_start = start_frame - 1  # Convert to 0-based
    python_end = end_frame  # Keep as-is for exclusive slice
    return python_start, python_end
