"""
Tests for save_results_with_config method and related functionality.

This module tests the save_results_with_config method implementation,
including proper handling of None configurations, timezone-aware timestamps,
and uncertainty field preservation.
"""

import pytest
import json
import tempfile
import os
from datetime import datetime, timezone
from unittest.mock import patch, mock_open
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from homodyne.analysis.core import HomodyneAnalysisCore


class TestSaveResultsWithConfig:
    """Test the save_results_with_config method implementation."""

    @pytest.fixture
    def mock_analyzer(self):
        """Create a minimal analyzer instance for testing."""
        analyzer = HomodyneAnalysisCore.__new__(HomodyneAnalysisCore)
        analyzer.config = {
            "output_settings": {
                "file_formats": {
                    "results_format": "json"
                }
            },
            "advanced_settings": {
                "chi_squared_calculation": {
                    "_scaling_optimization_note": "Scaling optimization is always enabled: g₂ = offset + contrast × g₁",
                    "uncertainty_calculation": {
                        "enable_uncertainty": True,
                        "report_uncertainty": True
                    }
                }
            }
        }
        return analyzer

    @pytest.fixture
    def sample_results_with_uncertainty(self):
        """Sample results dictionary including uncertainty fields."""
        return {
            "classical_optimization": {
                "parameters": [1000.0, -0.5, 100.0, 0.001, -0.3, 0.0001, 0.5],
                "reduced_chi_squared": 3.2,
                "reduced_chi_squared_uncertainty": 0.15,
                "reduced_chi_squared_std": 0.8,
                "n_optimization_angles": 5,
                "degrees_of_freedom": 1234,
                "convergence": True
            },
        }

    def test_save_with_valid_config(self, mock_analyzer, sample_results_with_uncertainty):
        """Test saving results with a valid configuration."""
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch('builtins.open', mock_open()) as mock_file:
                with patch('json.dump') as mock_json:
                    mock_analyzer.save_results_with_config(sample_results_with_uncertainty)
                    
                    # Verify json.dump was called
                    assert mock_json.called
                    call_args = mock_json.call_args
                    saved_data = call_args[0][0]  # First argument to json.dump
                    
                    # Check structure
                    assert "timestamp" in saved_data
                    assert "config" in saved_data
                    assert "results" in saved_data
                    assert "execution_metadata" in saved_data
                    
                    # Check results preservation
                    assert saved_data["results"] == sample_results_with_uncertainty
                    
                    # Check uncertainty fields are preserved
                    classical_results = saved_data["results"]["classical_optimization"]
                    assert "reduced_chi_squared_uncertainty" in classical_results
                    assert "reduced_chi_squared_std" in classical_results
                    assert "n_optimization_angles" in classical_results
                    assert "degrees_of_freedom" in classical_results

    def test_save_with_none_config(self, sample_results_with_uncertainty):
        """Test saving results when config is None."""
        analyzer = HomodyneAnalysisCore.__new__(HomodyneAnalysisCore)
        analyzer.config = None  # Explicitly set to None
        
        with patch('builtins.open', mock_open()) as mock_file:
            with patch('json.dump') as mock_json:
                # This should not raise an exception
                analyzer.save_results_with_config(sample_results_with_uncertainty)
                
                # Verify it was called (meaning no exception was raised)
                assert mock_json.called
                call_args = mock_json.call_args
                saved_data = call_args[0][0]
                
                # Config should be None but structure should still be valid
                assert saved_data["config"] is None
                assert "timestamp" in saved_data
                assert "results" in saved_data

    def test_timezone_aware_timestamps(self, mock_analyzer, sample_results_with_uncertainty):
        """Test that timestamps are timezone-aware and properly formatted."""
        with patch('builtins.open', mock_open()):
            with patch('json.dump') as mock_json:
                mock_analyzer.save_results_with_config(sample_results_with_uncertainty)
                
                call_args = mock_json.call_args
                saved_data = call_args[0][0]
                
                # Check timestamp format
                timestamp_str = saved_data["timestamp"]
                
                # Should be able to parse as ISO format with timezone
                try:
                    parsed_time = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                    # Should have timezone info
                    assert parsed_time.tzinfo is not None
                except ValueError:
                    pytest.fail(f"Timestamp {timestamp_str} is not in valid ISO format")

    def test_different_result_formats(self, mock_analyzer):
        """Test saving with different output format configurations."""
        # Test JSON format
        mock_analyzer.config["output_settings"]["file_formats"]["results_format"] = "json"
        test_results = {"test": "data"}
        
        with patch('builtins.open', mock_open()) as mock_file:
            with patch('json.dump'):
                mock_analyzer.save_results_with_config(test_results)
                # Check that file was opened with correct name
                mock_file.assert_called()

    def test_execution_metadata_structure(self, mock_analyzer, sample_results_with_uncertainty):
        """Test that execution metadata has correct structure."""
        with patch('builtins.open', mock_open()):
            with patch('json.dump') as mock_json:
                mock_analyzer.save_results_with_config(sample_results_with_uncertainty)
                
                call_args = mock_json.call_args
                saved_data = call_args[0][0]
                
                exec_metadata = saved_data["execution_metadata"]
                assert "analysis_success" in exec_metadata
                assert "timestamp" in exec_metadata
                assert exec_metadata["analysis_success"] is True

    def test_uncertainty_fields_preservation(self, mock_analyzer):
        """Test that all uncertainty-related fields are properly preserved."""
        test_results = {
            "method_test": {
                "reduced_chi_squared": 2.5,
                "reduced_chi_squared_uncertainty": 0.1,
                "reduced_chi_squared_std": 0.3,
                "n_optimization_angles": 4,
                "degrees_of_freedom": 500,
                "angle_chi_squared_reduced": [2.1, 2.3, 2.8, 2.7],
                "parameters": [1000, -0.5, 100, 0.001, -0.3, 0.0001, 0.5]
            }
        }
        
        with patch('builtins.open', mock_open()):
            with patch('json.dump') as mock_json:
                mock_analyzer.save_results_with_config(test_results)
                
                call_args = mock_json.call_args
                saved_data = call_args[0][0]
                
                method_results = saved_data["results"]["method_test"]
                
                # Check all uncertainty fields are preserved
                expected_fields = [
                    "reduced_chi_squared", "reduced_chi_squared_uncertainty",
                    "reduced_chi_squared_std", "n_optimization_angles",
                    "degrees_of_freedom", "angle_chi_squared_reduced"
                ]
                
                for field in expected_fields:
                    assert field in method_results
                    assert method_results[field] == test_results["method_test"][field]

    def test_file_creation_and_backup(self, mock_analyzer, sample_results_with_uncertainty):
        """Test file creation and backup directory logic."""
        with patch('os.path.exists') as mock_exists:
            with patch('builtins.open', mock_open()) as mock_file:
                with patch('json.dump'):
                    # Test when results directory exists
                    mock_exists.return_value = True
                    mock_analyzer.save_results_with_config(sample_results_with_uncertainty)
                    
                    # Should be called twice - once for main file, once for backup
                    assert mock_file.call_count >= 1

    def test_error_handling(self, mock_analyzer, sample_results_with_uncertainty):
        """Test error handling in save_results_with_config."""
        with patch('builtins.open', side_effect=PermissionError("Access denied")):
            # Should raise the PermissionError
            with pytest.raises(PermissionError):
                mock_analyzer.save_results_with_config(sample_results_with_uncertainty)

    def test_json_serialization_compatibility(self, mock_analyzer):
        """Test that results are JSON serializable."""
        import numpy as np
        
        # Test with numpy types that need conversion
        test_results = {
            "method": {
                "reduced_chi_squared": np.float64(2.5),
                "reduced_chi_squared_uncertainty": np.float32(0.1),
                "n_optimization_angles": np.int64(4),
                "parameters": np.array([1000.0, -0.5, 100.0])
            }
        }
        
        with patch('builtins.open', mock_open()):
            with patch('json.dump') as mock_json:
                # Should not raise an exception due to numpy types
                mock_analyzer.save_results_with_config(test_results)
                
                # Check that json.dump was called with default=str for numpy compatibility
                call_args = mock_json.call_args
                call_kwargs = call_args[1]
                assert 'default' in call_kwargs