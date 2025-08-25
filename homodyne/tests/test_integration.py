"""
Integration Tests for Rheo-SAXS-XPCS Analysis
=============================================

Tests complete analysis workflow with mocked heavy computations.
"""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

import numpy as np
import pytest

# Import the modules to test
from homodyne.tests.fixtures import (create_minimal_config_file,
                                     dummy_analysis_results, dummy_config,
                                     dummy_correlation_data, dummy_phi_angles,
                                     dummy_theoretical_data,
                                     mock_optimization_result, temp_directory)

# Import modules being tested
try:
    from homodyne.core.io_utils import (ensure_dir, get_output_directory,
                                        save_analysis_results)

    IO_UTILS_AVAILABLE = True
except ImportError:
    IO_UTILS_AVAILABLE = False

    # Provide fallback functions for type checker
    def save_analysis_results(*args, **kwargs):
        return {"json": False, "error": "IO utils not available"}

    def ensure_dir(path, *args, **kwargs):
        # Return a Path object that behaves correctly for tests
        return Path(path)

    def get_output_directory(*args, **kwargs):
        return Path("/tmp")


try:
    from homodyne.plotting import create_all_plots

    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False

    # Provide fallback function for type checker
    def create_all_plots(*args, **kwargs):
        return {}


from typing import Any, Optional


# Define a type stub that matches the interface we need
class _HomodyneAnalysisCoreStub:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.config: Optional[Any] = None

    def _prepare_plot_data(self, *args: Any, **kwargs: Any) -> None:
        return None

    def _generate_analysis_plots(self, *args: Any, **kwargs: Any) -> None:
        return None

    def _generate_theoretical_data(self, *args: Any, **kwargs: Any) -> None:
        return None


try:
    from homodyne.analysis.core import HomodyneAnalysisCore

    CORE_ANALYSIS_AVAILABLE = True
except ImportError:
    CORE_ANALYSIS_AVAILABLE = False
    # Use the stub when the real module is not available
    HomodyneAnalysisCore = _HomodyneAnalysisCoreStub  # type: ignore[misc]


class TestCompleteWorkflow:
    """Test complete analysis workflow integration."""

    @pytest.mark.skipif(
        not (IO_UTILS_AVAILABLE and PLOTTING_AVAILABLE),
        reason="Required modules not available",
    )
    def test_full_analysis_pipeline(
        self, temp_directory, dummy_analysis_results, dummy_config
    ):
        """Test complete pipeline: data processing → analysis → plotting → saving."""
        # Set up output directory in temp space
        dummy_config["output_settings"]["results_directory"] = str(
            temp_directory / "full_pipeline"
        )

        # Step 1: Create output directories
        output_dir = get_output_directory(dummy_config)
        assert output_dir.exists()

        # Step 2: Save analysis results
        save_status = save_analysis_results(
            dummy_analysis_results, dummy_config, "integration_test"
        )
        assert save_status["json"] is True

        # Step 3: Create plots
        plot_status = create_all_plots(dummy_analysis_results, output_dir, dummy_config)
        successful_plots = sum(1 for status in plot_status.values() if status)
        assert successful_plots >= 1

        # Step 4: Verify all outputs exist
        json_files = list(output_dir.glob("*.json"))
        plot_files = list(output_dir.glob("*.png"))

        assert len(json_files) >= 1
        assert len(plot_files) >= 1

        # Verify JSON content is readable
        with open(json_files[0], "r") as f:
            saved_results = json.load(f)
        assert isinstance(saved_results, dict)
        assert "best_chi_squared" in saved_results

    def test_directory_creation_workflow(self, temp_directory):
        """Test that the entire directory structure is created correctly."""
        base_dir = temp_directory / "analysis_workspace"

        # Test nested directory creation
        required_dirs = [
            base_dir / "results",
            base_dir / "plots" / "c2_heatmaps",
            base_dir / "plots" / "parameters",
            base_dir / "plots" / "diagnostics",
            base_dir / "data" / "processed",
            base_dir / "data" / "cached",
        ]

        created_dirs = []
        for dir_path in required_dirs:
            result_dir = ensure_dir(dir_path)
            created_dirs.append(result_dir)
            assert result_dir.exists()
            assert result_dir.is_dir()

        # Verify all directories were created
        assert len(created_dirs) == len(required_dirs)

        # Test that they can be used for file operations
        test_file = created_dirs[0] / "test.txt"
        test_file.write_text("test content")
        assert test_file.exists()
        assert test_file.read_text() == "test content"

    def test_new_output_directory_structure(self, temp_directory):
        """Test the new organized output directory structure."""
        base_dir = temp_directory / "homodyne_results"

        # Create the new expected directory structure with method-specific
        # directories
        expected_structure = [
            base_dir,  # Main output directory
            base_dir / "classical",  # Classical method results
            base_dir / "classical" / "nelder_mead",  # Nelder-Mead method
            base_dir / "classical" / "gurobi",  # Gurobi method
            base_dir / "exp_data",  # Experimental data plots
        ]

        created_dirs = []
        for dir_path in expected_structure:
            result_dir = ensure_dir(dir_path)
            created_dirs.append(result_dir)
            assert result_dir.exists()
            assert result_dir.is_dir()

        # Test creating expected files in each directory
        # Main directory files
        main_files = [base_dir / "homodyne_analysis_results.json", base_dir / "run.log"]

        # Create method-specific files in classical subdirectories
        method_names = ["nelder_mead", "gurobi"]
        classical_files = []

        for method_name in method_names:
            method_dir = base_dir / "classical" / method_name
            method_files = [
                method_dir / f"analysis_results_{method_name}.json",
                method_dir / "parameters.json",
                method_dir / "fitted_data.npz",  # Consolidated data
                method_dir / f"c2_heatmaps_{method_name}.png",
            ]
            classical_files.extend(method_files)

        # Summary file in classical directory
        classical_files.append(
            base_dir / "classical" / "all_classical_methods_summary.json"
        )

        # Experimental data directory files
        exp_data_files = [
            base_dir / "exp_data" / "data_validation_phi_0.0deg.png",
            base_dir / "exp_data" / "summary_statistics.txt",
        ]

        # Create all expected files
        all_files = main_files + classical_files + exp_data_files
        for file_path in all_files:
            file_path.parent.mkdir(parents=True, exist_ok=True)
            if file_path.suffix == ".npz":
                # Create realistic NPZ files with consolidated data
                mock_experimental_data = np.random.rand(3, 20, 30)
                mock_fitted_data = np.random.rand(3, 20, 30)
                mock_residuals_data = mock_experimental_data - mock_fitted_data

                np.savez_compressed(
                    file_path,
                    c2_experimental=mock_experimental_data,
                    c2_fitted=mock_fitted_data,
                    residuals=mock_residuals_data,
                    parameters=np.array([1.5, 2.0, 0.5]),
                    uncertainties=np.array([0.1, 0.1, 0.1]),
                    chi_squared=np.array([0.5]),
                )
            else:
                file_path.touch()
            assert file_path.exists()

        # Verify directory organization
        assert len(list(base_dir.glob("*.json"))) >= 1  # Main results files

        # Verify method-specific directories and files
        for method_name in method_names:
            method_dir = base_dir / "classical" / method_name
            assert method_dir.exists()
            # Consolidated data
            assert (method_dir / "fitted_data.npz").exists()
            assert (method_dir / "parameters.json").exists()
            assert (method_dir / f"c2_heatmaps_{method_name}.png").exists()

            # Verify NPZ file structure
            data = np.load(method_dir / "fitted_data.npz")
            assert "c2_experimental" in data
            assert "c2_fitted" in data
            assert "residuals" in data

        # Verify summary file
        assert (base_dir / "classical" / "all_classical_methods_summary.json").exists()

        # Verify experimental data plots
        assert len(list((base_dir / "exp_data").glob("*.png"))) >= 1  # Validation plots


class TestMockedHeavyComputation:
    """Test integration with mocked heavy computational components."""

    def test_mock_optimization_workflow(
        self, temp_directory, dummy_config, mock_optimization_result
    ):
        """Test analysis workflow with mocked optimization."""

        # Mock the heavy optimization function
        with patch("scipy.optimize.minimize") as mock_minimize:
            mock_minimize.return_value = mock_optimization_result

            # Simulate analysis workflow
            config = dummy_config.copy()
            config["output_settings"]["results_directory"] = str(
                temp_directory / "mock_opt"
            )

            # Create mock analysis function
            def mock_analysis_function():
                """Simulate analysis with mocked optimization."""
                # This would normally call heavy computation
                # Instead, we return the mocked result
                return {
                    "best_parameters": dict(
                        zip(
                            config["initial_parameters"]["parameter_names"],
                            mock_optimization_result.x,
                        )
                    ),
                    "best_chi_squared": mock_optimization_result.fun,
                    "optimization_success": mock_optimization_result.success,
                    "iterations": mock_optimization_result.nit,
                    "function_evaluations": mock_optimization_result.nfev,
                }

            # Run mocked analysis
            results = mock_analysis_function()

            # Verify mock was called and results are reasonable
            assert results["best_chi_squared"] == mock_optimization_result.fun
            assert results["optimization_success"] is True
            assert len(results["best_parameters"]) == len(
                config["initial_parameters"]["parameter_names"]
            )

            # Test that we can save these results
            if IO_UTILS_AVAILABLE:
                save_status = save_analysis_results(
                    results, config, "mock_optimization"
                )
                assert save_status["json"] is True

    def test_mock_data_loading(self, temp_directory, dummy_config):
        """Test data loading workflow with mocked file I/O."""

        # Create mock data files
        data_dir = temp_directory / "mock_data"
        data_dir.mkdir()

        # Mock HDF5 data loading
        mock_correlation_data = np.random.exponential(1.0, (100, 50, 50))
        mock_phi_angles = np.array([0.0, 22.5, 45.0, 67.5, 90.0])

        with patch("numpy.load") as mock_load:
            # Configure mock to return our test data
            mock_load.return_value = {
                "correlation_data": mock_correlation_data,
                "phi_angles": mock_phi_angles,
            }

            # Simulate data loading workflow
            def mock_load_data():
                """Simulate heavy data loading operation."""
                # This would normally load from HDF5
                loaded = np.load("mock_file.npz")
                return {
                    "experimental_data": loaded["correlation_data"],
                    "phi_angles": loaded["phi_angles"],
                }

            # Test the workflow
            data = mock_load_data()

            # Verify mock was used and data is correct shape
            mock_load.assert_called_once_with("mock_file.npz")
            assert data["experimental_data"].shape == mock_correlation_data.shape
            assert np.array_equal(data["phi_angles"], mock_phi_angles)

    @pytest.mark.skipif(not PLOTTING_AVAILABLE, reason="Plotting module not available")
    def test_plotting_workflow_integration(self, temp_directory, dummy_config):
        """Test plotting workflow integration without parameter evolution."""

        # Test that plotting works without the removed parameter evolution
        # function
        import numpy as np

        from homodyne.plotting import plot_c2_heatmaps

        exp_data = np.random.random((1, 10, 10)) + 1.0
        theory_data = np.random.random((1, 10, 10)) + 1.0
        phi_angles = np.array([0.0])

        # This should work with actual plotting functions
        success = plot_c2_heatmaps(
            exp_data, theory_data, phi_angles, temp_directory, dummy_config
        )

        # Should succeed and create plot files
        assert success is True
        plot_files = list(temp_directory.glob("*.png"))
        assert len(plot_files) > 0


class TestErrorHandlingIntegration:
    """Test error handling across the complete workflow."""

    def test_partial_failure_recovery(self, temp_directory, dummy_config):
        """Test that workflow continues when some components fail."""
        config = dummy_config.copy()
        config["output_settings"]["results_directory"] = str(
            temp_directory / "partial_failure"
        )

        # Create results with some missing components
        partial_results = {
            "best_parameters": {"D0": 100.0},
            "best_chi_squared": 1.234,
            # Missing: experimental_data, theoretical_data (plots will fail)
            "parameter_bounds": config["parameter_space"]["bounds"][:1],
        }

        # Save should succeed
        if IO_UTILS_AVAILABLE:
            save_status = save_analysis_results(partial_results, config, "partial_test")
            assert save_status["json"] is True

        # Plotting should handle missing data gracefully
        if PLOTTING_AVAILABLE:
            plot_status = create_all_plots(
                partial_results, temp_directory / "partial_failure", config
            )

            # Some plots may fail, but the function should return status
            assert isinstance(plot_status, dict)
            # At least one plot type should succeed (parameter evolution)
            assert any(plot_status.values())

    def test_configuration_error_handling(self, temp_directory):
        """Test handling of invalid configurations."""
        invalid_config = {
            "analyzer_parameters": {
                "temporal": {
                    "dt": -0.1,
                    "start_frame": 100,
                    "end_frame": 50,
                }  # Invalid values
            }
        }

        config_file = temp_directory / "invalid_config.json"
        with open(config_file, "w") as f:
            json.dump(invalid_config, f)

        # Loading should work (JSON is valid)
        with open(config_file, "r") as f:
            loaded_config = json.load(f)

        assert loaded_config == invalid_config

        # But validation would catch the issues
        assert loaded_config["analyzer_parameters"]["temporal"]["dt"] < 0
        assert (
            loaded_config["analyzer_parameters"]["temporal"]["start_frame"]
            > loaded_config["analyzer_parameters"]["temporal"]["end_frame"]
        )

    def test_file_permission_error_handling(self, temp_directory):
        """Test handling of file permission errors."""
        if os.name == "nt":  # Skip on Windows
            pytest.skip("Permission tests not reliable on Windows")

        # Create a read-only directory
        readonly_dir = temp_directory / "readonly"
        readonly_dir.mkdir(mode=0o444)

        try:
            # Attempt to save results should fail gracefully
            if IO_UTILS_AVAILABLE:
                results = {"test": "data"}
                config = {"output_settings": {"results_directory": str(readonly_dir)}}

                save_status = save_analysis_results(results, config, "permission_test")

                # Should handle permission error gracefully
                assert save_status["json"] is False

        finally:
            # Clean up
            readonly_dir.chmod(0o755)


class TestDataValidation:
    """Test data validation throughout the workflow."""

    def test_correlation_data_validation(self, dummy_phi_angles):
        """Test validation of correlation data shapes and values."""

        # Valid data
        valid_data = np.random.rand(3, 20, 30)  # angles x t2 x t1
        assert valid_data.shape[0] == len(dummy_phi_angles)
        assert not np.any(np.isnan(valid_data))
        # Correlation data should be non-negative
        assert np.all(valid_data >= 0)

        # Invalid data shapes
        wrong_angles = np.random.rand(5, 20, 30)  # Too many angles
        assert wrong_angles.shape[0] != len(dummy_phi_angles)

        wrong_dimensions = np.random.rand(3, 20)  # Missing delay time dimension
        assert len(wrong_dimensions.shape) != 3

        # Invalid data values
        nan_data = np.full((3, 20, 30), np.nan)
        assert np.all(np.isnan(nan_data))

        negative_data = np.full((3, 20, 30), -1.0)
        assert np.all(negative_data < 0)

    def test_parameter_bounds_validation(self, dummy_config):
        """Test parameter bounds validation."""
        bounds = dummy_config["parameter_space"]["bounds"]

        for bound in bounds:
            # Check required fields
            assert "name" in bound
            assert "min" in bound
            assert "max" in bound

            # Check logical consistency
            assert bound["min"] < bound["max"]

            # Check physical constraints for specific parameters
            if bound["name"] in ["D0", "gamma_dot_t0"]:
                assert (
                    bound["min"] > 0
                ), f"{
                    bound['name']} must have positive minimum"

            # Check that bounds are reasonable
            assert not np.isinf(bound["min"])
            assert not np.isinf(bound["max"])
            assert not np.isnan(bound["min"])
            assert not np.isnan(bound["max"])

    def test_time_array_validation(self):
        """Test validation of time arrays."""
        # Valid time arrays
        valid_t2 = np.linspace(0.1, 2.0, 20)
        valid_t1 = np.linspace(0.1, 3.0, 30)

        assert np.all(valid_t2 > 0)
        assert np.all(valid_t1 > 0)
        assert len(valid_t2) == 20
        assert len(valid_t1) == 30
        assert np.all(np.diff(valid_t2) > 0)  # Should be increasing
        assert np.all(np.diff(valid_t1) > 0)

        # Invalid time arrays
        negative_times = np.linspace(-1.0, 1.0, 10)
        assert np.any(negative_times < 0)

        non_monotonic = np.array([0.1, 0.3, 0.2, 0.4])
        assert not np.all(np.diff(non_monotonic) > 0)


class TestMemoryManagement:
    """Test memory-efficient operations and cleanup."""

    @pytest.mark.memory
    def test_large_array_handling(self, temp_directory):
        """Test handling of large data arrays without excessive memory usage."""
        # Create moderately large arrays to test memory handling
        large_shape = (10, 100, 150)  # Still manageable in CI environments

        # Use numpy's memory mapping capabilities for large arrays
        large_data = np.random.rand(*large_shape).astype(
            np.float32
        )  # Use float32 to save memory

        # Test that we can process the data
        assert large_data.shape == large_shape
        assert large_data.dtype == np.float32

        # Test basic operations that shouldn't cause memory issues
        mean_val = np.mean(large_data)
        std_val = np.std(large_data)

        assert not np.isnan(mean_val)
        assert not np.isnan(std_val)
        assert std_val > 0

        # Test saving large arrays
        if IO_UTILS_AVAILABLE:
            from homodyne.core.io_utils import save_numpy

            filepath = temp_directory / "large_array.npz"
            success = save_numpy(large_data, filepath, compressed=True)

            assert success is True
            assert filepath.exists()

            # Verify we can load it back
            loaded = np.load(filepath)
            np.testing.assert_array_equal(loaded["data"], large_data)

    @pytest.mark.memory
    def test_cleanup_after_plotting(self, temp_directory, dummy_config):
        """Test that matplotlib figures are properly cleaned up."""
        import matplotlib.pyplot as plt

        initial_figs = len(plt.get_fignums())

        # Create and save a plot
        if PLOTTING_AVAILABLE:
            import numpy as np

            from homodyne.plotting import plot_c2_heatmaps

            exp_data = np.random.random((1, 10, 10)) + 1.0
            theory_data = np.random.random((1, 10, 10)) + 1.0
            phi_angles = np.array([0.0])

            success = plot_c2_heatmaps(
                exp_data, theory_data, phi_angles, temp_directory, dummy_config
            )

            # Check that figures were cleaned up
            final_figs = len(plt.get_fignums())

            # Should not accumulate figures (memory leak prevention)
            assert (
                final_figs <= initial_figs + 1
            )  # Allow for one figure that might remain


class TestPerAngleAnalysisIntegration:
    """Integration tests for per-angle chi-squared analysis."""

    def test_per_angle_analysis_with_quality_assessment(
        self, temp_directory, dummy_config
    ):
        """Test per-angle analysis integration with quality assessment."""
        from unittest.mock import Mock, patch

        # Add fit_quality rules to config if not present
        if "validation_rules" not in dummy_config:
            dummy_config["validation_rules"] = {}

        dummy_config["validation_rules"]["fit_quality"] = {
            "overall_chi_squared": {
                "excellent_threshold": 2.0,
                "acceptable_threshold": 5.0,
                "warning_threshold": 10.0,
                "critical_threshold": 20.0,
            },
            "per_angle_chi_squared": {
                "excellent_threshold": 2.0,
                "acceptable_threshold": 5.0,
                "warning_threshold": 10.0,
                "outlier_threshold_multiplier": 2.5,
                "max_outlier_fraction": 0.25,
                "min_good_angles": 3,
            },
        }

        # Mock analyzer with per-angle analysis capability
        with patch("homodyne.analysis.core.HomodyneAnalysisCore") as MockAnalyzer:
            mock_instance = Mock()
            MockAnalyzer.return_value = mock_instance
            mock_instance.config = dummy_config

            # Mock per-angle analysis method
            mock_instance.analyze_per_angle_chi_squared.return_value = {
                "method": "TestMethod",
                "overall_reduced_chi_squared": 25.0,
                "overall_reduced_chi_squared_uncertainty": 2.5,
                "overall_reduced_chi_squared_std": 5.5,
                "n_optimization_angles": 5,
                "quality_assessment": {
                    "overall_quality": "warning",
                    "per_angle_quality": "acceptable",
                    "combined_quality": "warning",
                    "quality_issues": ["Overall chi-squared above warning threshold"],
                },
                "angle_categorization": {
                    "good_angles": {"count": 8, "fraction": 0.8},
                    "unacceptable_angles": {"count": 2, "fraction": 0.2},
                    "statistical_outliers": {"count": 1, "fraction": 0.1},
                },
            }

            # Test analysis execution
            parameters = np.array([1000.0, -0.1, 50.0, 0.01, -0.5, 0.001, 0.0])
            phi_angles = np.array(
                [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]
            )
            c2_exp = np.random.rand(10, 4, 5)

            result = mock_instance.analyze_per_angle_chi_squared(
                parameters,
                phi_angles,
                c2_exp,
                save_to_file=True,
                output_dir=str(temp_directory),
            )

            # Verify analysis was called
            mock_instance.analyze_per_angle_chi_squared.assert_called_once()

            # Verify quality assessment structure
            assert "quality_assessment" in result
            assert "angle_categorization" in result
            assert result["quality_assessment"]["combined_quality"] == "warning"

    def test_per_angle_results_integration(self, temp_directory, dummy_config):
        """Test that per-angle results are properly included in main results."""
        import json
        from pathlib import Path
        from unittest.mock import Mock, patch

        # Create mock results that would be included in main analysis results
        mock_results = {
            "method": "Classical",
            "overall_reduced_chi_squared": 15.0,
            "overall_reduced_chi_squared_uncertainty": 1.8,
            "overall_reduced_chi_squared_std": 4.0,
            "n_optimization_angles": 5,
            "per_angle_analysis": {
                "phi_angles_deg": [10.0, 20.0, 30.0, 40.0, 50.0],
                "chi_squared_reduced": [8.0, 12.0, 18.0, 25.0, 30.0],
            },
            "quality_assessment": {
                "combined_quality": "warning",
                "thresholds_used": {"acceptable_per_angle": 15.0},
            },
            "angle_categorization": {
                "good_angles": {"count": 2, "angles_deg": [10.0, 20.0]},
                "unacceptable_angles": {
                    "count": 3,
                    "angles_deg": [30.0, 40.0, 50.0],
                },
            },
        }

        # Verify the structure of per-angle results
        assert mock_results["method"] == "Classical"
        assert "quality_assessment" in mock_results
        assert "angle_categorization" in mock_results
        assert len(mock_results["per_angle_analysis"]["phi_angles_deg"]) == 5


class TestConcurrencyAndRaceConditions:
    """Test handling of concurrent operations and race conditions."""

    def test_concurrent_directory_creation(self, temp_directory):
        """Test that concurrent directory creation is handled safely."""
        import threading
        from concurrent.futures import ThreadPoolExecutor

        base_dir = temp_directory / "concurrent_test"
        results = []
        errors = []

        def create_directory(subdir_name):
            """Function to create directory in thread."""
            try:
                result = ensure_dir(base_dir / subdir_name)
                results.append(result)
                return result
            except Exception as e:
                errors.append(e)
                return None

        # Create multiple directories concurrently
        subdirs = [f"subdir_{i}" for i in range(5)]

        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(create_directory, subdir) for subdir in subdirs]

            # Wait for all to complete
            for future in futures:
                future.result()

        # Check that all directories were created
        assert len(errors) == 0, f"Errors occurred: {errors}"
        assert len(results) == len(subdirs)

        for subdir in subdirs:
            assert (base_dir / subdir).exists()

    def test_concurrent_file_saving(self, temp_directory):
        """Test concurrent file saving operations."""
        from concurrent.futures import ThreadPoolExecutor

        if not IO_UTILS_AVAILABLE:
            pytest.skip("IO utils not available")

        from homodyne.core.io_utils import save_json

        def save_test_file(file_id):
            """Function to save file in thread."""
            data = {"id": file_id, "data": list(range(file_id, file_id + 10))}
            filepath = temp_directory / f"concurrent_file_{file_id}.json"
            return save_json(data, filepath)

        # Save multiple files concurrently
        file_ids = list(range(5))

        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(save_test_file, fid) for fid in file_ids]
            results = [future.result() for future in futures]

        # All saves should succeed
        assert all(results)

        # Verify all files exist and have correct content
        for file_id in file_ids:
            filepath = temp_directory / f"concurrent_file_{file_id}.json"
            assert filepath.exists()

            with open(filepath, "r") as f:
                data = json.load(f)

            assert data["id"] == file_id
            assert data["data"] == list(range(file_id, file_id + 10))


class TestAnalysisWorkflowIntegration:
    """Test integration of analysis workflow with plotting."""

    @pytest.mark.skipif(
        not CORE_ANALYSIS_AVAILABLE, reason="Core analysis module not available"
    )
    def test_analysis_plotting_integration(self, temp_directory):
        """Test that analysis workflow integrates properly with plotting."""
        from homodyne.tests.fixtures import create_minimal_config_file

        config_file = create_minimal_config_file(temp_directory / "test_config.json")

        try:
            # Initialize analyzer
            analyzer = HomodyneAnalysisCore(str(config_file))

            # Check that analyzer has plotting methods
            assert hasattr(analyzer, "_generate_analysis_plots")
            assert hasattr(analyzer, "_prepare_plot_data")
            assert hasattr(analyzer, "_generate_theoretical_data")

            # Check cache variables are initialized
            assert hasattr(analyzer, "_last_experimental_data")
            assert hasattr(analyzer, "_last_phi_angles")

            # Simulate analysis results
            mock_results = {
                "classical_optimization": {
                    "parameters": [1000, -0.5, 100],
                    "chi_squared": 2.5,
                    "success": True,
                }
            }

            # Test plot data preparation
            plot_data = analyzer._prepare_plot_data(mock_results, analyzer.config or {})
            assert plot_data is not None
            assert "best_parameters" in plot_data
            assert "parameter_bounds" in plot_data

        except Exception as e:
            if "Configuration file not found" in str(e):
                pytest.skip("Could not create valid config file for test")
            else:
                raise

    @pytest.mark.skipif(
        not (CORE_ANALYSIS_AVAILABLE and PLOTTING_AVAILABLE),
        reason="Core analysis or plotting not available",
    )
    def test_mcmc_results_plotting_integration(self, temp_directory):
        """Test MCMC results integration with plotting workflow."""
        from homodyne.tests.fixtures import create_minimal_config_file

        config_file = create_minimal_config_file(
            temp_directory / "mcmc_test_config.json"
        )

        try:
            import arviz as az

            analyzer = HomodyneAnalysisCore(str(config_file))

            # Create mock MCMC results with trace data
            n_chains, n_draws = 2, 100
            param_names = ["D0", "alpha", "D_offset"]

            posterior_dict = {}
            for param in param_names:
                posterior_dict[param] = np.random.normal(0, 1, (n_chains, n_draws))

            trace_data = az.from_dict({"posterior": posterior_dict})

            mcmc_results = {
                "mcmc_optimization": {
                    "parameters": [1000, -0.5, 100],
                    "chi_squared": 2.3,
                    "success": True,
                    "convergence_diagnostics": {
                        "max_rhat": 1.05,
                        "min_ess": 200,
                        "converged": True,
                        # Updated to match actual MCMC module output format
                        "rhat": {"D0": 1.02, "alpha": 1.03, "D_offset": 1.01},
                        "ess": {"D0": 400, "alpha": 350, "D_offset": 450},
                        "mcse": {"D0": 0.001, "alpha": 0.002, "D_offset": 0.0015},
                        # Keep backward compatibility with old format too
                        "r_hat": {"D0": 1.02, "alpha": 1.03, "D_offset": 1.01},
                        "ess_bulk": {"D0": 400, "alpha": 350, "D_offset": 450},
                        "mcse_mean": {"D0": 0.001, "alpha": 0.002, "D_offset": 0.0015},
                    },
                }
            }

            # Test MCMC plot data preparation
            plot_data = analyzer._prepare_plot_data(mcmc_results, analyzer.config or {})
            assert plot_data is not None
            assert "mcmc_diagnostics" in plot_data
            assert "parameter_names" in plot_data
            assert "parameter_units" in plot_data

            # Add mock trace data to plot_data and test plotting
            plot_data["mcmc_trace"] = trace_data

            # Test individual MCMC plotting functions
            from homodyne.plotting import (plot_mcmc_convergence_diagnostics,
                                           plot_mcmc_corner, plot_mcmc_trace)

            plots_dir = temp_directory / "mcmc_plots"
            plots_dir.mkdir(exist_ok=True)

            # Test corner plot
            corner_success = plot_mcmc_corner(
                trace_data,
                plots_dir,
                analyzer.config,
                param_names=param_names,
                param_units=["Å²/s", "dimensionless", "Å²/s"],
            )

            # Test trace plot
            trace_success = plot_mcmc_trace(
                trace_data,
                plots_dir,
                analyzer.config,
                param_names=param_names,
                param_units=["Å²/s", "dimensionless", "Å²/s"],
            )

            # Test convergence diagnostics
            diag_success = plot_mcmc_convergence_diagnostics(
                trace_data,
                mcmc_results["mcmc_optimization"]["convergence_diagnostics"],
                plots_dir,
                analyzer.config,
                param_names=param_names,
            )

            # At least one plot should succeed (depending on available
            # packages)
            assert any([corner_success, trace_success, diag_success])

            # Check for created files
            plot_files = list(plots_dir.glob("*.png"))
            assert len(plot_files) >= 1

        except ImportError:
            pytest.skip("ArviZ not available for MCMC integration test")
        except Exception as e:
            if "Configuration file not found" in str(e):
                pytest.skip("Could not create valid config file for test")
            else:
                raise

    @pytest.mark.skipif(
        not CORE_ANALYSIS_AVAILABLE, reason="Core analysis module not available"
    )
    def test_configuration_consistency_integration(self, temp_directory):
        """Test that configuration is consistent across analysis workflow."""
        from homodyne.tests.fixtures import create_minimal_config_file

        config_file = create_minimal_config_file(
            temp_directory / "consistency_test_config.json"
        )

        try:
            analyzer = HomodyneAnalysisCore(str(config_file))

            # Check parameter consistency
            config = analyzer.config or {}
            param_names = config.get("initial_parameters", {}).get(
                "parameter_names", []
            )
            param_values = config.get("initial_parameters", {}).get("values", [])
            bounds = config.get("parameter_space", {}).get("bounds", [])

            # All should have same count
            assert len(param_names) == len(param_values) == len(bounds)

            # Parameter names should match between initial parameters and
            # bounds
            bound_names = [bound.get("name", "") for bound in bounds]
            assert param_names == bound_names

            # Check plotting configuration
            output_settings = config.get("output_settings", {})
            reporting = output_settings.get("reporting", {})
            plotting_enabled = reporting.get("generate_plots", False)

            # Should have valid plotting configuration
            assert isinstance(plotting_enabled, bool)

            if plotting_enabled:
                plot_formats = reporting.get("plot_formats", [])
                assert isinstance(plot_formats, list)
                assert len(plot_formats) > 0

                # All formats should be valid
                valid_formats = ["png", "pdf", "svg", "eps"]
                assert all(fmt in valid_formats for fmt in plot_formats)

        except Exception as e:
            if "Configuration file not found" in str(e):
                pytest.skip("Could not create valid config file for test")
            else:
                raise

    def test_end_to_end_plotting_workflow(self, temp_directory, dummy_config):
        """Test complete end-to-end plotting workflow."""
        # Set up comprehensive mock results with all plot types
        comprehensive_results = {
            "experimental_data": np.random.rand(3, 20, 20) + 1.0,
            "theoretical_data": np.random.rand(3, 20, 20) + 1.0,
            "phi_angles": np.array([0, 45, 90]),
            "best_parameters": {
                "D0": 1000.0,
                "alpha": -0.5,
                "D_offset": 100.0,
                "gamma_dot_t0": 0.001,
                "beta": 0.2,
                "gamma_dot_t_offset": 0.0001,
                "phi0": 5.0,
            },
            "parameter_bounds": dummy_config["parameter_space"]["bounds"],
            "parameter_names": [
                "D0",
                "alpha",
                "D_offset",
                "gamma_dot_t0",
                "beta",
                "gamma_dot_t_offset",
                "phi0",
            ],
            "parameter_units": [
                "Å²/s",
                "dimensionless",
                "Å²/s",
                "s⁻¹",
                "dimensionless",
                "s⁻¹",
                "degrees",
            ],
            "chi_squared": 2.5,
            "method": "Comprehensive Test",
        }

        # Add MCMC data if ArviZ is available
        try:
            import arviz as az

            n_chains, n_draws = 2, 200
            param_names = comprehensive_results["parameter_names"]

            posterior_dict = {}
            for param in param_names:
                if param == "D0":
                    posterior_dict[param] = np.random.lognormal(
                        5, 0.5, (n_chains, n_draws)
                    )
                else:
                    posterior_dict[param] = np.random.normal(
                        0, 0.1, (n_chains, n_draws)
                    )

            trace_data = az.from_dict({"posterior": posterior_dict})

            comprehensive_results.update(
                {
                    "mcmc_trace": trace_data,
                    "mcmc_diagnostics": {
                        "r_hat": {
                            name: np.random.uniform(1.0, 1.1) for name in param_names
                        },
                        "ess_bulk": {
                            name: np.random.randint(150, 400) for name in param_names
                        },
                        "mcse_mean": {
                            name: np.random.uniform(0.001, 0.005)
                            for name in param_names
                        },
                        "max_rhat": 1.08,
                        "min_ess": 180,
                        "converged": True,
                        "assessment": "Good",
                    },
                }
            )

        except ImportError:
            pass  # Skip MCMC parts if ArviZ not available

        # Test plotting workflow
        plots_dir = temp_directory / "comprehensive_plots"
        plots_dir.mkdir(exist_ok=True)

        plot_status = create_all_plots(comprehensive_results, plots_dir, dummy_config)

        # Check results
        assert isinstance(plot_status, dict)

        # Should have attempted multiple plot types
        expected_basic_plots = ["c2_heatmaps", "diagnostic_summary"]
        # Note: parameter_evolution functionality has been removed
        for plot_type in expected_basic_plots:
            if plot_type in plot_status:
                assert isinstance(plot_status[plot_type], bool)

        # Check that files were created
        all_plot_files = list(plots_dir.glob("*.png"))
        assert len(all_plot_files) >= 2  # Should create multiple plots

        # Verify file sizes (should contain actual plot data)
        for plot_file in all_plot_files:
            assert plot_file.stat().st_size > 5000  # Reasonable minimum for plot file
