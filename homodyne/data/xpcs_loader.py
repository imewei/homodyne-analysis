"""
XPCS Data Loader for Homodyne Analysis
======================================

Enhanced XPCS data loader supporting both APS (old) and APS-U (new) HDF5 formats
with intelligent caching and robust error handling.

This module provides:
- Auto-detection of APS vs APS-U format
- Direct h5py-based HDF5 reading (no pyxpcsviewer dependency)
- Half-matrix reconstruction for correlation matrices
- Mandatory diagonal correction applied post-load
- Smart NPZ caching to avoid reloading large HDF5 files
- Integration with existing homodyne configuration system

Key Features:
- Format Support: APS old format and APS-U new format
- Configuration: JSON-based (compatible with existing configs)
- Caching: Intelligent NPZ caching with compression
- Output: NumPy arrays optimized for homodyne analysis
- Validation: Basic data quality checks
"""

import json
import logging
import os
from pathlib import Path
from typing import Any

import numpy as np

# Handle h5py dependency
try:
    import h5py

    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False
    h5py = None

# Set up logging
logger = logging.getLogger(__name__)


class XPCSDataFormatError(Exception):
    """Raised when XPCS data format is not recognized or invalid."""

    pass


class XPCSDependencyError(Exception):
    """Raised when required dependencies are not available."""

    pass


class XPCSConfigurationError(Exception):
    """Raised when configuration is invalid or missing required parameters."""

    pass


def load_xpcs_config(config_path: str | Path) -> dict[str, Any]:
    """
    Load XPCS configuration from JSON file.

    Args:
        config_path: Path to JSON configuration file

    Returns:
        Configuration dictionary

    Raises:
        XPCSConfigurationError: If configuration format is unsupported or invalid
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise XPCSConfigurationError(f"Configuration file not found: {config_path}")

    try:
        if config_path.suffix.lower() == ".json":
            with open(config_path) as f:
                config = json.load(f)
            logger.info(f"Loaded JSON configuration: {config_path}")
            return config
        else:
            raise XPCSConfigurationError(
                f"Unsupported configuration format: {config_path.suffix}. "
                f"Supported formats: .json"
            )

    except json.JSONDecodeError as e:
        raise XPCSConfigurationError(
            f"Failed to parse configuration file {config_path}: {e}"
        )


class XPCSDataLoader:
    """
    XPCS data loader for Homodyne Analysis.

    Supports both APS (old) and APS-U (new) formats with auto-detection,
    intelligent caching, and integration with existing homodyne configurations.

    Features:
    - JSON configuration compatibility with existing homodyne configs
    - Auto-detection of HDF5 format (APS vs APS-U)
    - Smart NPZ caching with compression
    - Half-matrix reconstruction for correlation matrices
    - Mandatory diagonal correction applied consistently
    - NumPy array output optimized for homodyne analysis
    """

    def __init__(
        self,
        config_path: str | None = None,
        config_dict: dict | None = None,
    ):
        """
        Initialize XPCS data loader with JSON configuration.

        Args:
            config_path: Path to JSON configuration file
            config_dict: Configuration dictionary (alternative to config_path)

        Raises:
            XPCSDependencyError: If required dependencies are not available
            XPCSConfigurationError: If configuration is invalid
        """
        # Check for required dependencies
        self._check_dependencies()

        if config_path and config_dict:
            raise ValueError("Provide either config_path or config_dict, not both")

        if config_path:
            self.config = load_xpcs_config(config_path)
        elif config_dict:
            self.config = config_dict
        else:
            raise ValueError("Must provide either config_path or config_dict")

        # Extract main configuration sections
        self.exp_config = self.config.get("experimental_data", {})
        self.analyzer_config = self.config.get("analyzer_parameters", {})

        # Validate configuration
        self._validate_configuration()

        logger.info("XPCS data loader initialized for HDF5 format auto-detection")

    def _check_dependencies(self) -> None:
        """Check for required dependencies and raise error if missing."""
        missing_deps = []

        if not HAS_H5PY:
            missing_deps.append("h5py")

        if missing_deps:
            error_msg = f"Missing required dependencies: {', '.join(missing_deps)}. "
            error_msg += "Please install them with: pip install " + " ".join(
                missing_deps
            )
            logger.error(error_msg)
            raise XPCSDependencyError(error_msg)

    def _validate_configuration(self) -> None:
        """Validate configuration parameters."""
        required_exp_data = ["data_folder_path", "data_file_name"]

        # Validate experimental data parameters
        for key in required_exp_data:
            if key not in self.exp_config:
                raise XPCSConfigurationError(
                    f"Missing required experimental_data parameter: {key}"
                )

        # Validate analyzer parameters - handle nested structure
        # Check if parameters are in temporal subsection (new structure)
        if "temporal" in self.analyzer_config:
            temporal_config = self.analyzer_config["temporal"]
            required_temporal = ["dt", "start_frame", "end_frame"]

            for key in required_temporal:
                if key not in temporal_config:
                    raise XPCSConfigurationError(
                        f"Missing required analyzer_parameters.temporal parameter: {key}"
                    )
        else:
            # Check for direct parameters (old structure)
            required_analyzer = ["dt", "start_frame", "end_frame"]

            for key in required_analyzer:
                if key not in self.analyzer_config:
                    raise XPCSConfigurationError(
                        f"Missing required analyzer_parameters parameter: {key}"
                    )

        # Validate file existence
        data_file_path = os.path.join(
            self.exp_config["data_folder_path"], self.exp_config["data_file_name"]
        )

        if not os.path.exists(data_file_path):
            logger.warning(f"Data file not found: {data_file_path}")
            logger.info("File will be checked again during data loading")

    def _get_temporal_param(self, param_name: str, default=None):
        """Get temporal parameter from nested or direct structure."""
        if "temporal" in self.analyzer_config:
            return self.analyzer_config["temporal"].get(param_name, default)
        else:
            return self.analyzer_config.get(param_name, default)

    def _get_scattering_param(self, param_name: str, default=None):
        """Get scattering parameter from nested or direct structure."""
        if "scattering" in self.analyzer_config:
            return self.analyzer_config["scattering"].get(param_name, default)
        else:
            return self.analyzer_config.get(param_name, default)

    def load_experimental_data(self) -> tuple[np.ndarray, int, np.ndarray, int]:
        """
        Load experimental data with priority: cache NPZ → raw HDF → error.

        Returns:
            Tuple containing:
            - c2_experimental: Correlation data array (num_angles, time_length, time_length)
            - time_length: Length of time dimension
            - phi_angles: Array of phi angles
            - num_angles: Number of phi angles
        """
        # Construct file paths
        data_folder = self.exp_config.get("data_folder_path", "./")
        data_file = self.exp_config.get("data_file_name", "")
        cache_folder = self.exp_config.get("cache_file_path", data_folder)

        # Get frame parameters
        start_frame = self._get_temporal_param("start_frame", 1)
        end_frame = self._get_temporal_param("end_frame", 8000)

        # Construct cache filename
        cache_template = self.exp_config.get(
            "cache_filename_template", "cached_c2_frames_{start_frame}_{end_frame}.npz"
        )

        cache_filename = f"{cache_template.replace('{start_frame}', str(start_frame)).replace('{end_frame}', str(end_frame))}" if '{' in cache_template else f"cached_c2_frames_{start_frame}_{end_frame}.npz"
        cache_path = os.path.join(cache_folder, cache_filename)

        # Check cache first
        if os.path.exists(cache_path):
            logger.info(f"Loading cached data from: {cache_path}")
            data = self._load_from_cache(cache_path)
        else:
            # Load from raw HDF file
            hdf_path = os.path.join(data_folder, data_file)
            if not os.path.exists(hdf_path):
                raise FileNotFoundError(
                    f"Neither cache file {cache_path} nor HDF file {hdf_path} exists"
                )

            logger.info(f"Loading raw data from: {hdf_path}")
            data = self._load_from_hdf(hdf_path)

            # Save to cache
            logger.info(f"Saving processed data to cache: {cache_path}")
            self._save_to_cache(data, cache_path)

            # Generate text files for phi angles
            self._save_text_files(data)

        # Apply mandatory diagonal correction (post-load for consistent behavior)
        logger.debug("Applying mandatory diagonal correction to correlation matrices")
        c2_exp_corrected = []
        for i in range(len(data["c2_exp"])):
            c2_corrected = self._correct_diagonal(data["c2_exp"][i])
            c2_exp_corrected.append(c2_corrected)

        c2_experimental = np.array(c2_exp_corrected)

        # Perform basic data quality checks
        self._validate_loaded_data(c2_experimental, data["phi_angles_list"])

        logger.info(
            f"Data loaded successfully - shapes: phi{data['phi_angles_list'].shape}, "
            f"c2{c2_experimental.shape}"
        )

        # Return in format expected by HomodyneAnalysis
        time_length = c2_experimental.shape[-1]
        num_angles = len(data["phi_angles_list"])

        return c2_experimental, time_length, data["phi_angles_list"], num_angles

    def _load_from_cache(self, cache_path: str) -> dict[str, Any]:
        """Load data from NPZ cache file."""
        with np.load(cache_path, allow_pickle=True) as data:
            # Validate cache metadata if available
            if "cache_metadata" in data:
                metadata = data["cache_metadata"].item()
                logger.debug(f"Cache metadata found: {metadata}")

            return {
                "phi_angles_list": data["phi_angles_list"],
                "c2_exp": data["c2_exp"],
            }

    def _load_from_hdf(self, hdf_path: str) -> dict[str, Any]:
        """Load and process data from HDF5 file."""
        # Detect format
        logger.debug("Starting HDF5 format detection")
        format_type = self._detect_format(hdf_path)
        logger.info(f"Detected format: {format_type}")

        # Load based on format
        if format_type == "aps_old":
            return self._load_aps_old_format(hdf_path)
        elif format_type == "aps_u":
            return self._load_aps_u_format(hdf_path)
        else:
            raise XPCSDataFormatError(f"Unsupported format: {format_type}")

    def _detect_format(self, hdf_path: str) -> str:
        """Detect whether HDF5 file is APS old or APS-U new format."""
        with h5py.File(hdf_path, "r") as f:
            # Check for APS-U format keys
            if (
                "xpcs" in f
                and "qmap" in f["xpcs"]
                and "dynamic_v_list_dim0" in f["xpcs/qmap"]
                and "twotime" in f["xpcs"]
                and "correlation_map" in f["xpcs/twotime"]
            ):
                return "aps_u"

            # Check for APS old format keys
            elif (
                "xpcs" in f
                and "dqlist" in f["xpcs"]
                and "dphilist" in f["xpcs"]
                and "exchange" in f
                and "C2T_all" in f["exchange"]
            ):
                return "aps_old"

            else:
                available_keys = list(f.keys())
                raise XPCSDataFormatError(
                    f"Cannot determine HDF5 format - missing expected keys. "
                    f"Available root keys: {available_keys}"
                )

    def _load_aps_old_format(self, hdf_path: str) -> dict[str, Any]:
        """Load data from APS old format HDF5 file."""
        with h5py.File(hdf_path, "r") as f:
            # Load q and phi lists
            # dqlist = f["xpcs/dqlist"][0, :]  # Available but not currently used
            dphilist = f["xpcs/dphilist"][0, :]  # Shape (1, N) -> (N,)

            # Load correlation data from exchange/C2T_all
            c2t_group = f["exchange/C2T_all"]
            c2_keys = list(c2t_group.keys())

            logger.debug(
                f"Loading {len(c2_keys)} correlation matrices from APS old format"
            )

            # Load all correlation matrices
            c2_matrices = []
            selected_phi_angles = []

            for i, key in enumerate(c2_keys):
                c2_half = c2t_group[key][()]
                # Reconstruct full matrix from half matrix
                c2_full = self._reconstruct_full_matrix(c2_half)
                c2_matrices.append(c2_full)
                # Use corresponding phi angle
                if i < len(dphilist):
                    selected_phi_angles.append(dphilist[i])

            # Convert to numpy arrays
            c2_matrices_array = np.array(c2_matrices)
            phi_angles_array = np.array(selected_phi_angles)

            # Apply frame slicing
            c2_exp = self._apply_frame_slicing(c2_matrices_array)

            return {
                "phi_angles_list": phi_angles_array,
                "c2_exp": c2_exp,
            }

    def _load_aps_u_format(self, hdf_path: str) -> dict[str, Any]:
        """Load data from APS-U new format HDF5 file using processed_bins mapping."""
        with h5py.File(hdf_path, "r") as f:
            # Load the processed_bins mapping
            processed_bins = f["xpcs/twotime/processed_bins"][()]

            # Load the q and phi lists
            q_values = f["xpcs/qmap/dynamic_v_list_dim0"][()]
            phi_values = f["xpcs/qmap/dynamic_v_list_dim1"][()]

            n_q = len(q_values)
            n_phi = len(phi_values)

            logger.debug(f"APS-U format: {n_q} q-values, {n_phi} phi-values")
            logger.debug(
                f"Processed bins: {len(processed_bins)} correlation matrices available"
            )

            # Map processed_bins to (q,phi) pairs
            qphi_pairs = []
            valid_bin_indices = []

            for i, processed_bin in enumerate(processed_bins):
                bin_idx = processed_bin - 1  # Convert to 0-based
                q_idx = bin_idx // n_phi
                phi_idx = bin_idx % n_phi

                # Check if indices are valid
                if 0 <= q_idx < n_q and 0 <= phi_idx < n_phi:
                    phi_val = phi_values[phi_idx]
                    qphi_pairs.append(phi_val)
                    valid_bin_indices.append(i)

            if len(qphi_pairs) == 0:
                raise XPCSDataFormatError(
                    "No valid (q,phi) pairs found from processed_bins mapping"
                )

            # Load correlation matrices
            corr_group = f["xpcs/twotime/correlation_map"]
            c2_keys = sorted(corr_group.keys())

            logger.debug(
                f"Loading {len(valid_bin_indices)} correlation matrices from APS-U format"
            )

            c2_matrices = []
            for bin_idx in valid_bin_indices:
                if bin_idx < len(c2_keys):
                    key = c2_keys[bin_idx]
                    c2_half = corr_group[key][()]
                    # Reconstruct full matrix from half matrix
                    c2_full = self._reconstruct_full_matrix(c2_half)
                    c2_matrices.append(c2_full)

            # Convert to numpy arrays
            c2_matrices_array = np.array(c2_matrices)
            phi_angles_array = np.array(qphi_pairs)

            # Apply frame slicing
            c2_exp = self._apply_frame_slicing(c2_matrices_array)

            return {
                "phi_angles_list": phi_angles_array,
                "c2_exp": c2_exp,
            }

    def _reconstruct_full_matrix(self, c2_half: np.ndarray) -> np.ndarray:
        """
        Reconstruct full correlation matrix from half matrix (APS storage format).

        Based on pyXPCSViewer's approach:
        c2 = c2_half + c2_half.T
        c2[diag] /= 2

        Note: Diagonal correction is applied separately post-load.
        """
        c2_full = c2_half + c2_half.T
        # Correct diagonal (was doubled in addition)
        diag_indices = np.diag_indices(c2_half.shape[0])
        c2_full[diag_indices] /= 2

        return c2_full

    def _correct_diagonal(self, c2_mat: np.ndarray) -> np.ndarray:
        """
        Apply diagonal correction to correlation matrix.

        Based on pyXPCSViewer's correct_diagonal_c2 function.
        """
        size = c2_mat.shape[0]
        side_band = c2_mat[(np.arange(size - 1), np.arange(1, size))]

        # Create diagonal values
        diag_val = np.zeros(size)
        diag_val[:-1] += side_band
        diag_val[1:] += side_band
        norm = np.ones(size)
        norm[1:-1] = 2

        # Create a copy to avoid modifying input
        c2_corrected = c2_mat.copy()
        c2_corrected[np.diag_indices(size)] = diag_val / norm
        return c2_corrected

    def _apply_frame_slicing(self, c2_matrices: np.ndarray) -> np.ndarray:
        """
        Apply frame slicing to correlation matrices.

        Args:
            c2_matrices: Correlation matrices, shape (n_phi, full_frames, full_frames)

        Returns:
            Frame-sliced correlation matrices, shape (n_phi, sliced_frames, sliced_frames)
        """
        start_frame = (
            self._get_temporal_param("start_frame", 1) - 1
        )  # Convert to 0-based
        end_frame = self._get_temporal_param("end_frame", c2_matrices.shape[-1])

        # Validate frame parameters
        max_frames = c2_matrices.shape[-1]
        if start_frame < 0:
            start_frame = 0
            logger.warning("start_frame adjusted to 0")
        if end_frame > max_frames:
            end_frame = max_frames
            logger.warning(f"end_frame adjusted to {max_frames}")

        # Apply frame slicing if needed
        if start_frame > 0 or end_frame < max_frames:
            c2_exp = c2_matrices[:, start_frame:end_frame, start_frame:end_frame]
            logger.debug(
                f"Applied frame slicing: [{start_frame}:{end_frame}] -> shape {c2_exp.shape}"
            )
        else:
            c2_exp = c2_matrices
            logger.debug("No frame slicing needed - using full range")

        return c2_exp

    def _save_to_cache(self, data: dict[str, Any], cache_path: str) -> None:
        """Save processed data to NPZ cache file with metadata."""
        # Ensure cache directory exists
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)

        # Add cache metadata
        start_frame = self._get_temporal_param("start_frame", 1)
        end_frame = self._get_temporal_param(
            "end_frame", data["c2_exp"].shape[-1] + start_frame - 1
        )

        cache_metadata = {
            "start_frame": start_frame,
            "end_frame": end_frame,
            "phi_count": len(data["phi_angles_list"]),
            "cache_version": "1.0",
            "format": "homodyne_analysis",
        }

        cache_data = {
            "phi_angles_list": data["phi_angles_list"],
            "c2_exp": data["c2_exp"],
            "cache_metadata": cache_metadata,
        }

        # Save with compression
        if self.exp_config.get("cache_compression", True):
            np.savez_compressed(cache_path, **cache_data)
        else:
            np.savez(cache_path, **cache_data)

        # Log cache statistics
        file_size_mb = os.path.getsize(cache_path) / (1024 * 1024)
        logger.info(f"Cache saved: {cache_path}")
        logger.info(
            f"Cache size: {file_size_mb:.2f} MB, Phi angles: {cache_metadata['phi_count']}"
        )

    def _save_text_files(self, data: dict[str, Any]) -> None:
        """Save phi_angles list to text file."""
        # Get output directory
        phi_folder = self.exp_config.get("phi_angles_path", "./")

        # Save phi angles list
        phi_file = os.path.join(phi_folder, "phi_angles_list.txt")
        os.makedirs(os.path.dirname(phi_file), exist_ok=True)
        np.savetxt(
            phi_file,
            data["phi_angles_list"],
            fmt="%.6f",
            header="Phi angles (degrees)",
            comments="# ",
        )

        logger.debug(f"Text file saved: {phi_file}")

    def _validate_loaded_data(self, c2_exp: np.ndarray, phi_angles: np.ndarray) -> None:
        """Perform basic validation on loaded data."""
        # Basic checks
        if np.any(~np.isfinite(c2_exp)):
            logger.error("Correlation data contains non-finite values (NaN or Inf)")

        if np.any(c2_exp < 0):
            logger.warning("Correlation data contains negative values")

        # Check for reasonable correlation values (should be around 1.0 at t=0)
        diagonal_values = np.array([c2_exp[i].diagonal() for i in range(len(c2_exp))])
        mean_diagonal = np.mean(diagonal_values[:, 0])  # t=0 correlation
        if not (0.5 < mean_diagonal < 2.0):
            logger.warning(
                f"Unusual t=0 correlation value: {mean_diagonal:.3f} (expected ~1.0)"
            )

        logger.info("Basic data quality validation completed")


# Convenience function for simple usage
def load_xpcs_data(config_path: str) -> tuple[np.ndarray, int, np.ndarray, int]:
    """
    Convenience function to load XPCS data from configuration file.

    Args:
        config_path: Path to JSON configuration file

    Returns:
        Tuple containing:
        - c2_experimental: Correlation data array (num_angles, time_length, time_length)
        - time_length: Length of time dimension
        - phi_angles: Array of phi angles
        - num_angles: Number of phi angles

    Example:
        >>> c2_exp, time_len, phi_angles, num_angles = load_xpcs_data("config.json")
        >>> print(f"Loaded data shape: {c2_exp.shape}")
    """
    loader = XPCSDataLoader(config_path=config_path)
    return loader.load_experimental_data()


# Export main classes and functions
__all__ = [
    "XPCSConfigurationError",
    "XPCSDataFormatError",
    "XPCSDataLoader",
    "XPCSDependencyError",
    "load_xpcs_config",
    "load_xpcs_data",
]
