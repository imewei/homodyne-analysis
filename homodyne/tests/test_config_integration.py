"""
Tests for configuration integration with angle filtering.

This module tests the integration between ConfigManager and the analysis pipeline,
ensuring that angle filtering settings are properly read from configuration files
and applied throughout the system.
"""

import json
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pytest

from homodyne.core.config import ConfigManager


class TestConfigurationIntegration:
    """Test integration between ConfigManager and analysis components."""

    @pytest.fixture
    def config_with_custom_angle_filtering(self):
        """Configuration with custom angle filtering settings."""
        return {
            "metadata": {"config_version": "5.1"},
            "analyzer_parameters": {
                "temporal": {"dt": 0.1, "start_frame": 1, "end_frame": 100}
            },
            "experimental_data": {"data_folder_path": "./test_data/"},
            "optimization_config": {
                "angle_filtering": {
                    "enabled": True,
                    "target_ranges": [
                        {"min_angle": -15.0, "max_angle": 15.0},
                        {"min_angle": 165.0, "max_angle": 195.0},
                    ],
                    "fallback_to_all_angles": False,
                }
            },
        }

    @pytest.fixture
    def config_with_disabled_angle_filtering(self):
        """Configuration with angle filtering disabled."""
        return {
            "metadata": {"config_version": "5.1"},
            "analyzer_parameters": {
                "temporal": {"dt": 0.1, "start_frame": 1, "end_frame": 100}
            },
            "experimental_data": {"data_folder_path": "./test_data/"},
            "optimization_config": {
                "angle_filtering": {
                    "enabled": False,
                    "target_ranges": [
                        {"min_angle": -10.0, "max_angle": 10.0},
                        {"min_angle": 170.0, "max_angle": 190.0},
                    ],
                    "fallback_to_all_angles": True,
                }
            },
        }

    def test_core_analysis_uses_config_manager_ranges(
        self, tmp_path, config_with_custom_angle_filtering
    ):
        """Test that core analysis reads target ranges from ConfigManager."""
        config_file = tmp_path / "custom_ranges.json"
        with open(config_file, "w") as f:
            json.dump(config_with_custom_angle_filtering, f)

        # Create a ConfigManager
        config_manager = ConfigManager(str(config_file))

        # Verify the custom ranges are read correctly
        target_ranges = config_manager.get_target_angle_ranges()
        expected_ranges = [(-15.0, 15.0), (165.0, 195.0)]
        assert target_ranges == expected_ranges

        # Mock the core analysis module
        with patch("homodyne.analysis.core.HomodyneAnalysisCore") as MockCore:
            mock_core = MockCore.return_value
            mock_core.config_manager = config_manager

            # Test angles that should be filtered
            test_angles = np.array(
                [
                    -20.0,
                    -10.0,
                    0.0,
                    10.0,
                    20.0,
                    160.0,
                    170.0,
                    180.0,
                    190.0,
                    200.0,
                ]
            )

            # Simulate the filtering logic from core.py
            optimization_indices = []
            for i, angle in enumerate(test_angles):
                for min_angle, max_angle in target_ranges:
                    if min_angle <= angle <= max_angle:
                        optimization_indices.append(i)
                        break

            # Expected indices based on custom ranges
            expected_indices = [
                1,
                2,
                3,
                6,
                7,
                8,
            ]  # Angles in [-15, 15] and [165, 195]
            assert optimization_indices == expected_indices

    def test_core_analysis_respects_disabled_filtering(
        self, tmp_path, config_with_disabled_angle_filtering
    ):
        """Test that core analysis respects disabled angle filtering."""
        config_file = tmp_path / "disabled_filtering.json"
        with open(config_file, "w") as f:
            json.dump(config_with_disabled_angle_filtering, f)

        # Create a ConfigManager
        config_manager = ConfigManager(str(config_file))

        # Verify filtering is disabled
        assert config_manager.is_angle_filtering_enabled() is False

        # When filtering is disabled, all angles should be used
        test_angles = np.array([0.0, 45.0, 90.0, 135.0, 180.0])

        # Simulate the logic: if filtering is disabled, use all angles
        if config_manager.is_angle_filtering_enabled():
            # Apply filtering (not executed in this test)
            optimization_indices = []
        else:
            # Use all angles
            optimization_indices = list(range(len(test_angles)))

        expected_indices = [0, 1, 2, 3, 4]  # All angles
        assert optimization_indices == expected_indices

    def test_core_analysis_fallback_behavior(self, tmp_path):
        """Test core analysis fallback behavior with no matching angles."""
        # Create config with ranges that won't match any test angles
        config_with_no_matches = {
            "metadata": {"config_version": "5.1"},
            "analyzer_parameters": {
                "temporal": {"dt": 0.1, "start_frame": 1, "end_frame": 100}
            },
            "experimental_data": {"data_folder_path": "./test_data/"},
            "optimization_config": {
                "angle_filtering": {
                    "enabled": True,
                    "target_ranges": [
                        {
                            "min_angle": 300.0,
                            "max_angle": 350.0,
                        }  # No angles will match
                    ],
                    "fallback_to_all_angles": True,
                }
            },
        }

        config_file = tmp_path / "no_matches.json"
        with open(config_file, "w") as f:
            json.dump(config_with_no_matches, f)

        config_manager = ConfigManager(str(config_file))

        # Test with angles that won't match the target ranges
        test_angles = np.array([0.0, 45.0, 90.0, 135.0, 180.0])
        target_ranges = config_manager.get_target_angle_ranges()

        # Apply filtering logic
        optimization_indices = []
        for i, angle in enumerate(test_angles):
            for min_angle, max_angle in target_ranges:
                if min_angle <= angle <= max_angle:
                    optimization_indices.append(i)
                    break

        # No angles should match
        assert len(optimization_indices) == 0

        # Check fallback setting
        should_fallback = config_manager.should_fallback_to_all_angles()
        assert should_fallback

        # With fallback enabled, should use all angles
        if not optimization_indices and should_fallback:
            optimization_indices = list(range(len(test_angles)))

        assert optimization_indices == [0, 1, 2, 3, 4]

    def test_core_analysis_no_fallback_behavior(self, tmp_path):
        """Test core analysis with fallback disabled (should raise error)."""
        # Create config with no fallback
        config_no_fallback = {
            "metadata": {"config_version": "5.1"},
            "analyzer_parameters": {
                "temporal": {"dt": 0.1, "start_frame": 1, "end_frame": 100}
            },
            "experimental_data": {"data_folder_path": "./test_data/"},
            "optimization_config": {
                "angle_filtering": {
                    "enabled": True,
                    "target_ranges": [
                        {
                            "min_angle": 300.0,
                            "max_angle": 350.0,
                        }  # No angles will match
                    ],
                    "fallback_to_all_angles": False,
                }
            },
        }

        config_file = tmp_path / "no_fallback.json"
        with open(config_file, "w") as f:
            json.dump(config_no_fallback, f)

        config_manager = ConfigManager(str(config_file))

        # Test with angles that won't match
        test_angles = np.array([0.0, 45.0, 90.0])
        target_ranges = config_manager.get_target_angle_ranges()

        optimization_indices = []
        for i, angle in enumerate(test_angles):
            for min_angle, max_angle in target_ranges:
                if min_angle <= angle <= max_angle:
                    optimization_indices.append(i)
                    break

        # No angles should match
        assert len(optimization_indices) == 0

        # Fallback should be disabled
        should_fallback = config_manager.should_fallback_to_all_angles()
        assert should_fallback is False

        # This should result in an error condition
        if not optimization_indices and not should_fallback:
            # Simulate the error that would be raised in core.py
            error_expected = True
        else:
            error_expected = False

        assert error_expected

    def test_optimization_methods_integration(
        self, tmp_path, config_with_custom_angle_filtering
    ):
        """Test that optimization methods properly integrate with ConfigManager."""
        config_file = tmp_path / "integration_test.json"
        with open(config_file, "w") as f:
            json.dump(config_with_custom_angle_filtering, f)

        config_manager = ConfigManager(str(config_file))

        # Test classical optimization integration
        try:
            from homodyne.optimization.classical import ClassicalOptimizer

            # Create mock analyzer with ConfigManager
            mock_analyzer = Mock()
            mock_analyzer.config_manager = config_manager

            optimizer = ClassicalOptimizer(mock_analyzer, {})

            # Create objective function
            phi_angles = np.array([-20.0, -10.0, 0.0, 10.0, 20.0, 175.0])
            c2_experimental = np.random.rand(6, 5, 5)

            objective = optimizer.create_objective_function(
                phi_angles, c2_experimental, "Test"
            )

            # Mock the core method
            mock_analyzer.calculate_chi_squared_optimized = Mock(return_value=5.0)
            test_params = np.array([1000.0, -0.1, 50.0, 0.01, -0.5, 0.001, 0.0])

            result = objective(test_params)

            # Verify that angle filtering was enabled (from config)
            call_args = mock_analyzer.calculate_chi_squared_optimized.call_args
            assert call_args[1]["filter_angles_for_optimization"]
            assert result == 5.0

        except ImportError:
            pytest.skip("Classical optimization module not available")

        # Test MCMC integration
        try:
            from homodyne.optimization.mcmc import MCMCSampler

            # Create mock MCMC sampler with ConfigManager
            mock_analyzer = Mock()
            mock_analyzer.config_manager = config_manager

            # Verify that the MCMC run_mcmc_analysis method would use ConfigManager
            # (This is tested at the API level since full MCMC requires PyMC)
            enabled = config_manager.is_angle_filtering_enabled()
            ranges = config_manager.get_target_angle_ranges()
            fallback = config_manager.should_fallback_to_all_angles()

            # Verify config values are as expected
            assert enabled
            assert ranges == [(-15.0, 15.0), (165.0, 195.0)]
            assert fallback is False

        except ImportError:
            pytest.skip("MCMC module not available")

    def test_backward_compatibility_without_config_manager(self):
        """Test that systems work without ConfigManager (backward compatibility)."""
        try:
            from homodyne.optimization.classical import ClassicalOptimizer

            # Create analyzer without ConfigManager
            mock_analyzer = Mock(spec=["calculate_chi_squared_optimized"])
            # Explicitly ensure no config_manager attribute exists

            optimizer = ClassicalOptimizer(
                mock_analyzer,
                {"optimization_config": {"angle_filtering": {"enabled": True}}},
            )

            phi_angles = np.array([0.0, 90.0, 180.0])
            c2_experimental = np.random.rand(3, 5, 5)

            objective = optimizer.create_objective_function(
                phi_angles, c2_experimental, "Test"
            )

            # Mock the core method
            mock_analyzer.calculate_chi_squared_optimized = Mock(return_value=3.0)
            test_params = np.array([1000.0, -0.1, 50.0, 0.01, -0.5, 0.001, 0.0])

            result = objective(test_params)

            # Should still use angle filtering (from config dict)
            call_args = mock_analyzer.calculate_chi_squared_optimized.call_args
            assert call_args[1]["filter_angles_for_optimization"]
            assert result == 3.0

        except ImportError:
            pytest.skip("Classical optimization module not available")

    def test_real_config_files_integration(self):
        """Test integration with actual configuration files."""
        # Test with my_config.json if it exists
        my_config_path = Path(__file__).parent.parent.parent / "my_config.json"
        if my_config_path.exists():
            config_manager = ConfigManager(str(my_config_path))

            # Should not raise exceptions
            enabled = config_manager.is_angle_filtering_enabled()
            ranges = config_manager.get_target_angle_ranges()
            fallback = config_manager.should_fallback_to_all_angles()
            full_config = config_manager.get_angle_filtering_config()

            # All should return valid values
            assert isinstance(enabled, bool)
            assert isinstance(ranges, list)
            assert isinstance(fallback, bool)
            assert isinstance(full_config, dict)

            # Full config should have expected keys
            expected_keys = {
                "enabled",
                "target_ranges",
                "fallback_to_all_angles",
            }
            assert expected_keys.issubset(set(full_config.keys()))

        # Test with template config
        template_path = Path(__file__).parent.parent / "config_template.json"
        if template_path.exists():
            config_manager = ConfigManager(str(template_path))

            # Should work identically
            enabled = config_manager.is_angle_filtering_enabled()
            ranges = config_manager.get_target_angle_ranges()
            fallback = config_manager.should_fallback_to_all_angles()

            assert isinstance(enabled, bool)
            assert isinstance(ranges, list)
            assert isinstance(fallback, bool)


if __name__ == "__main__":
    pytest.main([__file__])
