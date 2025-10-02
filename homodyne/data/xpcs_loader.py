"""
XPCS Data Loader for Homodyne Analysis
======================================

Enhanced XPCS data loader supporting both APS (old) and APS-U (new) HDF5 formats
with intelligent caching, comprehensive filtering, and robust error handling.

This module provides:
- Auto-detection of APS vs APS-U format
- Direct h5py-based HDF5 reading (no pyxpcsviewer dependency)
- Complete APS-U processed_bins mapping implementation
- Half-matrix reconstruction for correlation matrices
- Mandatory diagonal correction applied post-load
- Smart NPZ caching to avoid reloading large HDF5 files
- Integration with cache-aware phi angles loading system

APS-U Format Support (Complete Implementation):
------------------------------------------------
The APS-U format uses a grid-based structure with processed_bins mapping:

HDF5 Structure:
  xpcs/
  ├── qmap/
  │   ├── dynamic_v_list_dim0   # All q-values (wavevector magnitudes)
  │   └── dynamic_v_list_dim1   # All phi-values (angles in degrees)
  └── twotime/
      ├── processed_bins        # Maps which (q,phi) pairs have data
      └── correlation_map/
          ├── c2_00001          # Correlation matrix 1 (half-matrix format)
          ├── c2_00002          # Correlation matrix 2
          └── ...

Loading Algorithm:
  1. Load processed_bins mapping to identify available (q,phi) pairs
  2. Map bins to actual (q,phi) coordinates using grid formula:
     - bin_idx = processed_bin - 1 (convert to 0-based)
     - q_idx = bin_idx // n_phi (row index)
     - phi_idx = bin_idx % n_phi (column index)
  3. Load correlation matrices for valid bins
  4. Apply data filtering (quality, phi-range)
  5. Select optimal q-vector (closest to config value)
  6. Extract all phi angles for selected q-vector (exact matching)
  7. Apply frame slicing and generate time arrays
  8. Return complete data structure with q-list, phi-list, time arrays, and c2 data

Key Differences from APS Old Format:
  - Q-Phi Storage: Grid structure with processed_bins vs flat (q,phi) pairs
  - Correlation Data: xpcs/twotime/correlation_map vs exchange/C2T_all
  - Q-Vector Selection: Exact matching (< 1e-10) vs tolerance-based (10%)
  - Phi Angles: Structured grid per q-vector vs variable per q-vector

Cache-Aware Workflow:
  - When cache exists: Load from NPZ (includes phi_angles_list)
  - When no cache: Extract from HDF5, save phi_angles_list.txt + wavevector_q_list.txt

Key Features:
- Format Support: Complete APS old and APS-U new format
- Configuration: JSON-based (compatible with existing configs)
- Caching: Intelligent NPZ caching with compression
- Output: NumPy arrays optimized for homodyne analysis
- Validation: Basic data quality checks
- Text Files: Generates phi_angles_list.txt and wavevector_q_list.txt
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


class XPCSDependencyError(Exception):
    """Raised when required dependencies are not available."""


class XPCSConfigurationError(Exception):
    """Raised when configuration is invalid or missing required parameters."""


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
        """
        Check for required dependencies for XPCS data loading.

        Raises
        ------
        XPCSDependencyError
            If required dependencies are not available
        """
        # Check for h5py dependency
        if not HAS_H5PY:
            raise XPCSDependencyError(
                "h5py is required for XPCS data loading but was not found. "
                "Please install h5py: pip install h5py"
            )

        # Check for numpy (should always be available, but validate version)
        try:
            import numpy as np

            # Check minimum numpy version for scientific computing
            np_version = tuple(map(int, np.__version__.split(".")[:2]))
            if np_version < (1, 20):
                raise XPCSDependencyError(
                    f"NumPy version {np.__version__} is too old. "
                    "Please upgrade to NumPy >= 1.20.0: pip install --upgrade numpy"
                )
        except ImportError:
            raise XPCSDependencyError(
                "NumPy is required for XPCS data loading but was not found. "
                "Please install numpy: pip install numpy"
            )

        logger.debug("✓ All required dependencies available for XPCS data loading")

    def _get_temporal_param(self, param_name: str, default_value: Any) -> Any:
        """Get temporal parameter from configuration, handling both flat and nested structures."""
        # Check if parameters are in temporal subsection (new structure)
        if "temporal" in self.analyzer_config:
            temporal_config = self.analyzer_config["temporal"]
            return temporal_config.get(param_name, default_value)
        # Fallback to flat structure (old format)
        return self.analyzer_config.get(param_name, default_value)

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

        cache_filename = (
            f"{cache_template.replace('{start_frame}', str(start_frame)).replace('{end_frame}', str(end_frame))}"
            if "{" in cache_template
            else f"cached_c2_frames_{start_frame}_{end_frame}.npz"
        )
        cache_path = os.path.join(cache_folder, cache_filename)

        # Check cache first
        if os.path.exists(cache_path):
            logger.info(f"Loading cached data from: {cache_path}")
            data = self._load_from_cache(cache_path)
        else:
            # Load from raw data file (HDF5 or NPZ)
            data_path = os.path.join(data_folder, data_file)
            if not os.path.exists(data_path):
                raise FileNotFoundError(
                    f"Neither cache file {cache_path} nor data file {data_path} exists"
                )

            logger.info(f"Loading raw data from: {data_path}")

            # Detect file format and load accordingly
            if data_file.lower().endswith(".npz"):
                data = self._load_from_npz(data_path)
            elif data_file.lower().endswith((".h5", ".hdf5")):
                data = self._load_from_hdf(data_path)
            else:
                # Try HDF5 format as default
                data = self._load_from_hdf(data_path)

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

        # Get shape info for logging
        phi_shape = getattr(
            data["phi_angles_list"], "shape", f"list[{len(data['phi_angles_list'])}]"
        )
        logger.info(
            f"Data loaded successfully - shapes: phi{phi_shape}, "
            f"c2{c2_experimental.shape}"
        )

        # Return in format expected by HomodyneAnalysis
        time_length = c2_experimental.shape[-1]
        num_angles = len(data["phi_angles_list"])

        return c2_experimental, time_length, data["phi_angles_list"], num_angles

    def _load_from_hdf(self, hdf_path: str) -> dict[str, Any]:
        """Load and process data from HDF5 file."""
        # Detect format
        logger.debug("Starting HDF5 format detection")
        format_type = self._detect_format(hdf_path)
        logger.info(f"Detected format: {format_type}")

        # Load based on format
        if format_type == "aps_old":
            return self._load_aps_old_format(hdf_path)
        if format_type == "aps_u":
            return self._load_aps_u_format(hdf_path)
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
            if (
                "xpcs" in f
                and "dqlist" in f["xpcs"]
                and "dphilist" in f["xpcs"]
                and "exchange" in f
                and "C2T_all" in f["exchange"]
            ):
                return "aps_old"

            available_keys = list(f.keys())
            raise XPCSDataFormatError(
                f"Cannot determine HDF5 format - missing expected keys. "
                f"Available root keys: {available_keys}"
            )

    def _load_aps_old_format(self, hdf_path: str) -> dict[str, Any]:
        """Load data from APS old format HDF5 file."""
        with h5py.File(hdf_path, "r") as f:
            # Step 1: Load q and phi lists (2D arrays - extract first row)
            if "xpcs/dqlist" not in f:
                raise XPCSDataFormatError("Required path 'xpcs/dqlist' not found")
            if "xpcs/dphilist" not in f:
                raise XPCSDataFormatError("Required path 'xpcs/dphilist' not found")

            dqlist_2d = f["xpcs/dqlist"][()]
            dphilist_2d = f["xpcs/dphilist"][()]

            # Extract 1D arrays from 2D structure [0, :]
            if dqlist_2d.ndim == 2:
                dqlist = dqlist_2d[0, :]
                dphilist = dphilist_2d[0, :]
                logger.debug(
                    f"Extracted 1D arrays from 2D structure: {len(dqlist)} (q,phi) pairs"
                )
            else:
                dqlist = dqlist_2d
                dphilist = dphilist_2d
                logger.debug(f"Using 1D arrays directly: {len(dqlist)} (q,phi) pairs")

            # Step 2: Access C2T_all as a GROUP (not dataset!)
            if "exchange" not in f:
                raise XPCSDataFormatError("'exchange' group not found in HDF5 file")

            exchange = f["exchange"]
            if "C2T_all" not in exchange:
                raise XPCSDataFormatError("'C2T_all' not found in exchange group")

            c2t_obj = exchange["C2T_all"]

            # Step 3: Verify it's a group and get dataset keys
            if not isinstance(c2t_obj, h5py.Group):
                raise XPCSDataFormatError(
                    f"'C2T_all' should be a Group but is {type(c2t_obj)}. "
                    f"Expected structure: exchange/C2T_all/<c2_00001, c2_00002, ...>"
                )

            c2_keys = sorted(c2t_obj.keys())
            logger.debug(f"Found {len(c2_keys)} correlation matrices in C2T_all group")

            # Validate data consistency
            if len(c2_keys) != len(dqlist):
                logger.warning(
                    f"Mismatch: {len(c2_keys)} correlation matrices but {len(dqlist)} (q,phi) pairs"
                )
                min_len = min(len(c2_keys), len(dqlist))
                c2_keys = c2_keys[:min_len]
                dqlist = dqlist[:min_len]
                dphilist = dphilist[:min_len]
                logger.debug(f"Truncated to {min_len} entries for consistency")

            # Step 4: Load all correlation matrices
            logger.debug(f"Loading {len(c2_keys)} correlation matrices")
            c2_matrices_all = []
            for key in c2_keys:
                c2_half = c2t_obj[key][()]
                c2_full = self._reconstruct_full_matrix(c2_half)
                c2_matrices_all.append(c2_full)

            # Step 5: Select optimal q-vector
            selected_q_idx = self._select_optimal_wavevector(dqlist)
            selected_q = dqlist[selected_q_idx]

            # Step 6: Find (q,phi) pairs near selected q-vector with tolerance
            q_tolerance = selected_q * 0.1  # 10% tolerance
            q_matching_indices = np.where(np.abs(dqlist - selected_q) <= q_tolerance)[0]

            # Ensure minimum coverage
            if len(q_matching_indices) < 5 and len(dqlist) >= 5:
                q_distances = np.abs(dqlist - selected_q)
                closest_indices = np.argsort(q_distances)
                n_desired = min(10, len(closest_indices))
                q_matching_indices = closest_indices[:n_desired]
                logger.debug(
                    f"Expanded to {len(q_matching_indices)} q-vectors for phi coverage"
                )

            logger.debug(
                f"Selected {len(q_matching_indices)} (q,phi) pairs, "
                f"q-range: {dqlist[q_matching_indices].min():.6f}-{dqlist[q_matching_indices].max():.6f} Å⁻¹"
            )

            # Step 7: Apply phi filtering if enabled
            selected_indices = self._get_selected_indices_simple(dphilist)

            if selected_indices is not None:
                final_indices = np.intersect1d(q_matching_indices, selected_indices)
                logger.debug(
                    f"After phi filtering: {len(q_matching_indices)} -> {len(final_indices)}"
                )
            else:
                final_indices = q_matching_indices

            # Step 8: Extract filtered data
            filtered_dqlist = dqlist[final_indices]
            filtered_dphilist = dphilist[final_indices]
            selected_c2_matrices = [c2_matrices_all[i] for i in final_indices]
            c2_matrices_array = np.array(selected_c2_matrices)

            # Step 9: Apply frame slicing
            c2_exp = self._apply_frame_slicing(c2_matrices_array)

            # Step 10: Calculate time arrays
            dt = self._get_temporal_param("dt", 0.1)
            start_frame = self._get_temporal_param("start_frame", 1)
            end_frame = self._get_temporal_param(
                "end_frame", start_frame + c2_exp.shape[-1] - 1
            )

            time_max = dt * (end_frame - start_frame)
            time_1d = np.linspace(0, time_max, c2_exp.shape[-1])
            t1, t2 = np.meshgrid(time_1d, time_1d, indexing="ij")

            return {
                "wavevector_q_list": filtered_dqlist,
                "phi_angles_list": filtered_dphilist,
                "t1": t1,
                "t2": t2,
                "c2_exp": c2_exp,
            }

    def _load_aps_u_format(self, hdf_path: str) -> dict[str, Any]:
        """
        Load data from APS-U new format HDF5 file using processed_bins mapping.

        This implementation follows the comprehensive APS-U loading algorithm:
        1. Load processed_bins mapping to identify which (q,phi) pairs have data
        2. Map bins to actual (q,phi) coordinates using grid structure
        3. Load correlation matrices for valid bins
        4. Apply data filtering (quality, phi-range)
        5. Select optimal q-vector
        6. Extract all phi angles for selected q-vector
        7. Apply frame slicing and generate time arrays

        Returns complete data structure compatible with cache-aware loading.
        """
        with h5py.File(hdf_path, "r") as f:
            # Step 1: Load processed_bins mapping - indicates which (q,phi) have data
            processed_bins = f["xpcs/twotime/processed_bins"][()]

            # Step 2: Load q and phi value grids
            q_values = f["xpcs/qmap/dynamic_v_list_dim0"][()]  # All q values
            phi_values = f["xpcs/qmap/dynamic_v_list_dim1"][()]  # All phi values

            n_q = len(q_values)
            n_phi = len(phi_values)

            logger.debug(f"APS-U format: {n_q} q-values, {n_phi} phi-values")
            logger.debug(f"Q range: {q_values.min():.6f} to {q_values.max():.6f} Å⁻¹")
            logger.debug(f"Phi values: {phi_values}")
            logger.debug(
                f"Processed bins: {len(processed_bins)} correlation matrices available"
            )

            # Step 3: Map processed_bins to (q,phi) pairs using grid structure
            # Formula: bin_idx = processed_bin - 1; q_idx = bin_idx // n_phi; phi_idx = bin_idx % n_phi
            qphi_pairs = []
            valid_bin_indices = []

            for i, processed_bin in enumerate(processed_bins):
                bin_idx = processed_bin - 1  # Convert to 0-based indexing
                q_idx = bin_idx // n_phi
                phi_idx = bin_idx % n_phi

                # Validate indices are within grid bounds
                if 0 <= q_idx < n_q and 0 <= phi_idx < n_phi:
                    q_val = q_values[q_idx]
                    phi_val = phi_values[phi_idx]
                    qphi_pairs.append((q_val, phi_val))  # Store both q and phi
                    valid_bin_indices.append(i)
                else:
                    logger.warning(
                        f"Invalid bin mapping: processed_bin={processed_bin}, "
                        f"q_idx={q_idx}, phi_idx={phi_idx}"
                    )

            if len(qphi_pairs) == 0:
                raise XPCSDataFormatError(
                    "No valid (q,phi) pairs found from processed_bins mapping"
                )

            # Step 4: Convert to arrays for processing
            qphi_array = np.array(qphi_pairs)
            filtered_dqlist = qphi_array[:, 0]  # q values for valid pairs
            filtered_dphilist = qphi_array[:, 1]  # phi values for valid pairs

            logger.debug(
                f"Extracted {len(valid_bin_indices)} valid (q,phi) pairs from processed_bins"
            )

            # Step 5: Load correlation matrices for valid bins
            corr_group = f["xpcs/twotime/correlation_map"]
            c2_keys = sorted(
                corr_group.keys()
            )  # Sorted alphabetically (c2_00001, c2_00002, ...)

            logger.debug(
                f"Loading {len(valid_bin_indices)} correlation matrices corresponding to valid (q,phi) pairs"
            )
            c2_matrices_for_filtering = []

            for bin_idx in valid_bin_indices:
                if bin_idx < len(c2_keys):
                    key = c2_keys[bin_idx]
                    c2_half = corr_group[key][()]
                    # Reconstruct full matrix from half-matrix storage
                    c2_full = self._reconstruct_full_matrix(c2_half)
                    c2_matrices_for_filtering.append(c2_full)
                else:
                    logger.warning(
                        f"Matrix index {bin_idx} exceeds available matrices ({len(c2_keys)})"
                    )

            # Step 6: Ensure consistency between matrices and (q,phi) pairs
            min_count = min(len(c2_matrices_for_filtering), len(filtered_dqlist))
            if len(c2_matrices_for_filtering) != len(filtered_dqlist):
                logger.warning(
                    f"Matrix count ({len(c2_matrices_for_filtering)}) != "
                    f"(q,phi) pair count ({len(filtered_dqlist)})"
                )
                c2_matrices_for_filtering = c2_matrices_for_filtering[:min_count]
                filtered_dqlist = filtered_dqlist[:min_count]
                filtered_dphilist = filtered_dphilist[:min_count]
                logger.debug(f"Truncated to {min_count} entries for consistency")

            # Step 7: Select optimal q-vector (closest match to config)
            selected_q_idx = self._select_optimal_wavevector(filtered_dqlist)
            selected_q = filtered_dqlist[selected_q_idx]

            logger.debug(
                f"Selected optimal q-vector: {selected_q:.6f} Å⁻¹ (index {selected_q_idx})"
            )

            # Step 8: Find all (q,phi) pairs matching selected q-vector
            # APS-U uses exact matching (< 1e-10 tolerance) unlike APS old (10% tolerance)
            q_matching_indices = np.where(np.abs(filtered_dqlist - selected_q) < 1e-10)[
                0
            ]
            logger.debug(
                f"Found {len(q_matching_indices)} (q,phi) pairs matching selected q-vector"
            )

            # Step 9: Apply phi filtering if enabled
            selected_indices = self._get_selected_indices_simple(filtered_dphilist)

            if selected_indices is not None:
                # Intersect q-vector selection with phi filtering
                final_indices = np.intersect1d(q_matching_indices, selected_indices)
                logger.debug(
                    f"After intersecting with phi filtering: {len(final_indices)} pairs remain"
                )
            else:
                # No phi filtering - use all pairs for selected q-vector
                final_indices = q_matching_indices
                logger.debug(
                    f"No phi filtering applied - using all {len(final_indices)} pairs for selected q-vector"
                )

            # Step 10: Fallback if no valid indices (safety check)
            if len(final_indices) == 0:
                logger.warning(
                    "No valid indices found, using first available entry as fallback"
                )
                final_indices = [0]

            # Step 11: Extract final data for selected indices
            final_dqlist = filtered_dqlist[final_indices]
            final_dphilist = filtered_dphilist[final_indices]
            c2_matrices = [c2_matrices_for_filtering[i] for i in final_indices]

            logger.debug(f"Final selection: {len(c2_matrices)} correlation matrices")

            # Step 12: Convert to numpy array for frame slicing
            c2_matrices_array = np.array(c2_matrices)

            # Step 13: Apply frame slicing to selected q-vector data
            c2_exp = self._apply_frame_slicing(c2_matrices_array)

            # Step 14: Calculate time arrays (2D meshgrids for correlation analysis)
            dt = self._get_temporal_param("dt", 0.1)
            start_frame = self._get_temporal_param("start_frame", 1)
            end_frame = self._get_temporal_param(
                "end_frame", start_frame + c2_exp.shape[-1] - 1
            )

            time_max = dt * (end_frame - start_frame)
            time_1d = np.linspace(0, time_max, c2_exp.shape[-1])
            t1, t2 = np.meshgrid(time_1d, time_1d, indexing="ij")

            # Step 15: Return complete data structure
            return {
                "wavevector_q_list": final_dqlist,  # Selected q-values
                "phi_angles_list": final_dphilist,  # Corresponding phi angles
                "t1": t1,  # 2D time meshgrid (first dimension)
                "t2": t2,  # 2D time meshgrid (second dimension)
                "c2_exp": c2_exp,  # Shape: (n_phi, n_frames, n_frames)
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

    def _select_optimal_wavevector(self, dqlist: np.ndarray) -> int:
        """Select q-vector index closest to config value."""
        # Get from nested scattering config
        scattering_config = self.analyzer_config.get("scattering", {})
        config_q = scattering_config.get("wavevector_q", 0.0054)

        # Find closest q-vector
        closest_idx = np.argmin(np.abs(dqlist - config_q))
        selected_q = dqlist[closest_idx]
        deviation = abs(selected_q - config_q)

        logger.info(
            f"Selected closest q-vector: {selected_q:.6f} Å⁻¹ "
            f"(target: {config_q:.6f} Å⁻¹, deviation: {deviation:.6f} Å⁻¹)"
        )

        return closest_idx

    def _get_selected_indices_simple(self, dphilist: np.ndarray) -> np.ndarray | None:
        """Simple phi angle filtering based on configuration."""
        opt_config = self.config.get("optimization_config", {})
        angle_config = opt_config.get("angle_filtering", {})

        if not angle_config.get("enabled", False):
            logger.debug("Angle filtering disabled")
            return None

        target_ranges = angle_config.get("target_ranges", [])
        if not target_ranges:
            return None

        selected_mask = np.zeros(len(dphilist), dtype=bool)
        for range_spec in target_ranges:
            min_angle = range_spec.get("min_angle", -180)
            max_angle = range_spec.get("max_angle", 180)
            in_range = (dphilist >= min_angle) & (dphilist <= max_angle)
            selected_mask |= in_range

        selected_indices = np.where(selected_mask)[0]

        if len(selected_indices) == 0 and angle_config.get(
            "fallback_to_all_angles", True
        ):
            logger.warning("No angles in target ranges - using all angles")
            return None

        logger.debug(
            f"Angle filtering: {len(selected_indices)}/{len(dphilist)} angles selected"
        )
        return selected_indices

    def _apply_frame_slicing(self, c2_matrices: np.ndarray) -> np.ndarray:
        """Apply frame slicing to correlation matrices."""
        start_frame = self._get_temporal_param("start_frame", 1) - 1  # 0-based
        end_frame = self._get_temporal_param("end_frame", c2_matrices.shape[-1])

        max_frames = c2_matrices.shape[-1]
        start_frame = max(0, start_frame)
        end_frame = min(max_frames, end_frame)

        if start_frame > 0 or end_frame < max_frames:
            c2_exp = c2_matrices[:, start_frame:end_frame, start_frame:end_frame]
            logger.debug(
                f"Applied frame slicing: [{start_frame}:{end_frame}] -> shape {c2_exp.shape}"
            )
        else:
            c2_exp = c2_matrices
            logger.debug("No frame slicing needed - using full range")

        return c2_exp

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

    def _load_from_npz(self, npz_path: str) -> dict[str, Any]:
        """Load data from NPZ file (for testing purposes)."""
        logger.info(f"Loading NPZ data from: {npz_path}")

        try:
            data = np.load(npz_path)

            # Extract data arrays
            c2_data = data["c2_data"]
            angles = data["angles"]
            t1_array = data["t1_array"] if "t1_array" in data else None
            t2_array = data["t2_array"] if "t2_array" in data else None

            logger.debug(f"NPZ data shape: {c2_data.shape}")
            logger.debug(f"Angles: {len(angles)}")

            # Create correlation matrices list
            c2_list = []
            for i in range(len(angles)):
                c2_list.append(c2_data[i])

            time_length = c2_data.shape[1] if len(c2_data.shape) > 1 else 1

            logger.info(
                f"✓ Loaded NPZ data: {len(c2_list)} angles, time_length={time_length}"
            )

            return {
                "c2_exp": c2_list,
                "phi_angles_list": [angles] * len(c2_list),
                "time_length": time_length,
                "num_angles": len(angles),
            }

        except Exception as e:
            logger.error(f"Failed to load NPZ file: {e}")
            raise XPCSDataFormatError(f"Failed to load NPZ file {npz_path}: {e}")

    def _load_from_cache(self, cache_path: str) -> dict[str, Any]:
        """Load data from cache NPZ file."""
        logger.info(f"Loading cached data from: {cache_path}")

        try:
            data = np.load(cache_path, allow_pickle=True)

            result = {
                "c2_exp": data["c2_exp"],
                "phi_angles_list": data["phi_angles_list"],
            }

            # Add optional fields if present (new format)
            if "wavevector_q_list" in data:
                result["wavevector_q_list"] = data["wavevector_q_list"]
            if "t1" in data:
                result["t1"] = data["t1"]
            if "t2" in data:
                result["t2"] = data["t2"]

            return result
        except Exception as e:
            logger.error(f"Failed to load cache file: {e}")
            raise XPCSDataFormatError(f"Failed to load cache file {cache_path}: {e}")

    def _save_to_cache(self, data: dict[str, Any], cache_path: str) -> None:
        """Save data to cache NPZ file."""
        logger.info(f"Saving data to cache: {cache_path}")

        try:
            # Ensure directory exists
            cache_dir = os.path.dirname(cache_path)
            if cache_dir:
                os.makedirs(cache_dir, exist_ok=True)

            # Prepare data for caching
            cache_data = {
                "c2_exp": data["c2_exp"],
                "phi_angles_list": data["phi_angles_list"],
            }

            # Add optional fields if present
            if "wavevector_q_list" in data:
                cache_data["wavevector_q_list"] = data["wavevector_q_list"]
            if "t1" in data:
                cache_data["t1"] = data["t1"]
            if "t2" in data:
                cache_data["t2"] = data["t2"]

            np.savez_compressed(cache_path, **cache_data)
            logger.debug(f"✓ Cached data saved to: {cache_path}")
        except Exception as e:
            logger.warning(f"Failed to save cache file: {e}")
            # Don't raise exception for cache save failures

    def _save_text_files(self, data: dict[str, Any]) -> None:
        """Save phi angles and wavevector q-values to text files."""
        # Get output directories
        phi_folder = self.exp_config.get(
            "phi_angles_path", self.exp_config.get("data_folder_path", ".")
        )
        data_folder = self.exp_config.get("data_folder_path", ".")

        # Ensure directories exist
        os.makedirs(phi_folder, exist_ok=True)
        os.makedirs(data_folder, exist_ok=True)

        # Save phi angles list
        phi_file = os.path.join(phi_folder, "phi_angles_list.txt")
        np.savetxt(
            phi_file,
            data["phi_angles_list"],
            fmt="%.6f",
            header="Phi angles (degrees)",
            comments="# ",
        )
        logger.debug(f"Saved phi_angles_list.txt to {phi_file}")

        # Save wavevector q list (if available from HDF5 loading)
        if "wavevector_q_list" in data:
            q_file = os.path.join(data_folder, "wavevector_q_list.txt")
            np.savetxt(
                q_file,
                data["wavevector_q_list"],
                fmt="%.8e",
                header="Wavevector q (1/Angstrom)",
                comments="# ",
            )
            logger.debug(f"Saved wavevector_q_list.txt to {q_file}")


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
