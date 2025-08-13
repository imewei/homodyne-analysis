"""
IO Utilities for Homodyne Scattering Analysis

This module provides utility functions for safe file system operations,
timestamped filename generation, and robust data saving with error handling.

Created for: Rheo-SAXS-XPCS Homodyne Analysis
Authors: Wei Chen, Hongrui He
Institution: Argonne National Laboratory & University of Chicago
"""

import json
import pickle
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Union
import numpy as np

# Set up logging
logger = logging.getLogger(__name__)


def ensure_dir(path: Union[str, Path], permissions: int = 0o755) -> Path:
    """
    Create directories recursively with race-condition safety.

    This function safely creates directories, handling race conditions
    where multiple processes might try to create the same directory
    simultaneously.

    Args:
        path (Union[str, Path]): Directory path to create
        permissions (int): Directory permissions (default: 0o755)

    Returns:
        Path: Path object of the created directory

    Raises:
        OSError: If directory creation fails for reasons other than already existing

    Example:
        >>> ensure_dir("./results/data")
        PosixPath('./results/data')
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
    base_name: str, chi2: Optional[float] = None, config: Optional[Dict] = None
) -> str:
    """
    Build timestamped filenames based on output_settings["file_naming"] configuration.

    Creates filenames with timestamps and optional chi-squared values according
    to the configuration settings.

    Args:
        base_name (str): Base filename (without extension)
        chi2 (Optional[float]): Chi-squared value to include in filename
        config (Optional[Dict]): Configuration dictionary containing output_settings

    Returns:
        str: Formatted filename with timestamp and optional chi2 value

    Example:
        >>> config = {"output_settings": {"file_naming": {"timestamp_format": "%Y%m%d_%H%M%S",
        ...                                               "include_chi_squared": True}}}
        >>> timestamped_filename("results", 1.234, config)
        'results_20240315_143022_chi2_1.234000'
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


def save_json(data: Any, filepath: Union[str, Path], **kwargs) -> bool:
    """
    Save data as JSON with error handling and logging.

    Args:
        data: Data to save (must be JSON serializable)
        filepath (Union[str, Path]): Output file path
        **kwargs: Additional arguments passed to json.dump()

    Returns:
        bool: True if successful, False otherwise

    Example:
        >>> data = {"results": [1, 2, 3], "chi2": 1.234}
        >>> save_json(data, "results.json", indent=2)
        True
    """
    filepath = Path(filepath)

    try:
        # Ensure parent directory exists
        ensure_dir(filepath.parent)

        # Set default JSON parameters
        json_kwargs = {"indent": 2, "ensure_ascii": False}
        json_kwargs.update(kwargs)

        # Save JSON file
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, **json_kwargs)

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
    filepath: Union[str, Path],
    compressed: bool = True,
    **kwargs,
) -> bool:
    """
    Save NumPy array with error handling and logging.

    Args:
        data (np.ndarray): NumPy array to save
        filepath (Union[str, Path]): Output file path
        compressed (bool): Whether to use compression (default: True)
        **kwargs: Additional arguments passed to np.savez_compressed or np.save

    Returns:
        bool: True if successful, False otherwise

    Example:
        >>> data = np.random.rand(100, 100)
        >>> save_numpy(data, "data.npz", compressed=True)
        True
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
    filepath: Union[str, Path],
    protocol: int = pickle.HIGHEST_PROTOCOL,
    **kwargs,
) -> bool:
    """
    Save data using pickle with error handling and logging.

    Args:
        data: Data to pickle
        filepath (Union[str, Path]): Output file path
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
    figure,
    filepath: Union[str, Path],
    dpi: int = 300,
    format: Optional[str] = None,
    **kwargs,
) -> bool:
    """
    Save matplotlib figure with error handling and logging.

    Args:
        figure: Matplotlib figure object
        filepath (Union[str, Path]): Output file path
        dpi (int): Resolution in dots per inch (default: 300)
        format (Optional[str]): Figure format (inferred from extension if None)
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
def get_output_directory(config: Optional[Dict] = None) -> Path:
    """
    Get the output directory from configuration, creating it if necessary.

    Args:
        config (Optional[Dict]): Configuration dictionary

    Returns:
        Path: Output directory path
    """
    default_dir = "./results"

    if config and "output_settings" in config:
        output_dir = config["output_settings"].get("results_directory", default_dir)
    else:
        output_dir = default_dir
        logger.warning(
            "No output directory configuration found, using default: ./results"
        )

    return ensure_dir(output_dir)


def save_analysis_results(
    results: Dict,
    config: Optional[Dict] = None,
    base_name: str = "analysis_results",
) -> Dict[str, bool]:
    """
    Save complete analysis results using appropriate formats.

    Args:
        results (Dict): Analysis results dictionary
        config (Optional[Dict]): Configuration dictionary
        base_name (str): Base filename for outputs

    Returns:
        Dict[str, bool]: Status of each save operation
    """
    output_dir = get_output_directory(config)
    chi2 = results.get("best_chi_squared")

    # Generate base filename
    filename_base = timestamped_filename(base_name, chi2, config)

    save_status = {}

    # Save main results as JSON
    json_path = output_dir / f"{filename_base}.json"
    save_status["json"] = save_json(results, json_path)

    # Save NumPy arrays if present
    if "correlation_data" in results and isinstance(
        results["correlation_data"], np.ndarray
    ):
        npz_path = output_dir / f"{filename_base}_data.npz"
        save_status["numpy"] = save_numpy(results["correlation_data"], npz_path)

    # Save complex objects as pickle
    if any(
        key.startswith("mcmc_") or key.startswith("bayesian_") for key in results.keys()
    ):
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
