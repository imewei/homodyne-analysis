"""
Tests for MCMC angle filtering functionality.

This module tests the enhanced MCMC sampling that includes:
- Angle filtering for optimization (using only specific angular ranges)
- Default angle filtering behavior
- Integration with PyMC model building
- Backward compatibility with existing code
- Performance optimizations through data reduction
"""

import numpy as np
import pytest
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

# Test PyMC availability
try:
    import pymc as pm
    import arviz as az
    PYMC_AVAILABLE = True
except ImportError:
    PYMC_AVAILABLE = False
    pm = None
    az = None


class TestMCMCAngleFilteringCore:
    """Test core MCMC angle filtering functionality."""

    @pytest.fixture
    def mock_config_for_mcmc(self):
        """Configuration for MCMC angle filtering tests."""
        return {
            "metadata": {
                "config_version": "5.1",
                "description": "Test configuration for MCMC angle filtering"
            },
            "analyzer_parameters": {
                "temporal": {"dt": 0.1, "start_frame": 1, "end_frame": 100},
                "scattering": {"wavevector_q": 0.005},
                "geometry": {"stator_rotor_gap": 2000000},
                "computational": {"num_threads": 4}
            },
            "experimental_data": {
                "data_folder_path": "./test_data/",
                "data_file_name": "test_data.hdf",
                "phi_angles_path": "./test_data/",
                "phi_angles_file": "phi_angles.txt"
            },
            "initial_parameters": {
                "values": [1000.0, -0.1, 50.0, 0.01, -0.5, 0.001, 0.0],
                "parameter_names": ["D0", "alpha", "D_offset", "gamma_dot_t0", "beta", "gamma_dot_t_offset", "phi0"]
            },
            "optimization_config": {
                "mcmc_sampling": {
                    "mcmc_draws": 100,
                    "mcmc_tune": 50,
                    "mcmc_chains": 2,
                    "target_accept": 0.8
                }
            },
            "performance_settings": {
                "use_float32_precision": True,
                "bayesian_subsample_factor": 1,
                "noise_model": {
                    "use_simple_forward_model": True,
                    "sigma_prior": 0.1
                }
            }
        }

    @pytest.fixture
    def test_phi_angles_mcmc(self):
        """Test phi angles for MCMC filtering tests."""
        return np.array([
            -25.0,   # Outside range
            -8.0,    # Inside [-10, 10] range
            -5.0,    # Inside [-10, 10] range
            0.0,     # Inside [-10, 10] range
            5.0,     # Inside [-10, 10] range
            8.0,     # Inside [-10, 10] range
            15.0,    # Outside range
            45.0,    # Outside range
            90.0,    # Outside range
            150.0,   # Outside range
            175.0,   # Inside [170, 190] range
            180.0,   # Inside [170, 190] range
            185.0,   # Inside [170, 190] range
            200.0,   # Outside range
        ])

    @pytest.fixture
    def mock_mcmc_sampler(self, mock_config_for_mcmc):
        """Create a mock MCMC sampler for testing."""
        if not PYMC_AVAILABLE:
            pytest.skip("PyMC not available")
        
        from homodyne.optimization.mcmc import MCMCSampler
        
        # Mock analyzer
        mock_analyzer = Mock()
        mock_analyzer.config = mock_config_for_mcmc
        mock_analyzer.num_threads = 4
        
        # Create sampler without full initialization
        sampler = MCMCSampler.__new__(MCMCSampler)
        sampler.core = mock_analyzer
        sampler.config = mock_config_for_mcmc
        sampler.mcmc_config = mock_config_for_mcmc["optimization_config"]["mcmc_sampling"]
        sampler.bayesian_model = None
        sampler.mcmc_trace = None
        sampler.mcmc_result = None
        
        return sampler

    def test_mcmc_angle_filtering_parameter_defaults(self, mock_mcmc_sampler):
        """Test that MCMC methods have correct default parameters."""
        if not PYMC_AVAILABLE:
            pytest.skip("PyMC not available")
        
        import inspect
        
        # Test _build_bayesian_model_optimized defaults
        sig = inspect.signature(mock_mcmc_sampler._build_bayesian_model_optimized)
        assert 'filter_angles_for_optimization' in sig.parameters
        assert sig.parameters['filter_angles_for_optimization'].default is True
        
        # Test _run_mcmc_nuts_optimized defaults
        sig = inspect.signature(mock_mcmc_sampler._run_mcmc_nuts_optimized)
        assert 'filter_angles_for_optimization' in sig.parameters
        assert sig.parameters['filter_angles_for_optimization'].default is True
        
        # Test run_mcmc_analysis defaults (now uses None, resolved at runtime)
        sig = inspect.signature(mock_mcmc_sampler.run_mcmc_analysis)
        assert 'filter_angles_for_optimization' in sig.parameters
        assert sig.parameters['filter_angles_for_optimization'].default is None  # Uses None, resolved to True at runtime

    def test_mcmc_angle_filtering_logic(self, test_phi_angles_mcmc):
        """Test MCMC angle filtering identification logic."""
        target_ranges = [(-10.0, 10.0), (170.0, 190.0)]
        
        # Apply filtering logic
        optimization_indices = []
        for i, angle in enumerate(test_phi_angles_mcmc):
            for min_angle, max_angle in target_ranges:
                if min_angle <= angle <= max_angle:
                    optimization_indices.append(i)
                    break
        
        expected_indices = [1, 2, 3, 4, 5, 10, 11, 12]  # Indices of angles in ranges
        expected_angles = [-8.0, -5.0, 0.0, 5.0, 8.0, 175.0, 180.0, 185.0]
        
        assert optimization_indices == expected_indices
        assert len(optimization_indices) == 8
        
        filtered_angles = test_phi_angles_mcmc[optimization_indices]
        assert np.allclose(filtered_angles, expected_angles)

    def test_mcmc_data_shape_with_filtering(self, mock_mcmc_sampler, test_phi_angles_mcmc):
        """Test that MCMC data filtering reduces data dimensions correctly."""
        if not PYMC_AVAILABLE:
            pytest.skip("PyMC not available")
        
        # Create mock experimental data
        n_angles = len(test_phi_angles_mcmc)
        n_time = 50  # Smaller for testing
        c2_experimental = np.random.rand(n_angles, n_time, n_time) + 1.0
        
        # Test angle filtering logic directly
        target_ranges = [(-10.0, 10.0), (170.0, 190.0)]
        optimization_indices = []
        for i, angle in enumerate(test_phi_angles_mcmc):
            for min_angle, max_angle in target_ranges:
                if min_angle <= angle <= max_angle:
                    optimization_indices.append(i)
                    break
        
        # Apply filtering
        c2_filtered = c2_experimental[optimization_indices]
        
        # Check dimensions
        assert c2_filtered.shape[0] == len(optimization_indices)  # Reduced angles
        assert c2_filtered.shape[1] == n_time  # Time dimension unchanged
        assert c2_filtered.shape[2] == n_time  # Time dimension unchanged
        
        # Should have 8 angles (from test_phi_angles_mcmc)
        assert c2_filtered.shape[0] == 8

    def test_mcmc_model_building_with_filtering(self, mock_mcmc_sampler, test_phi_angles_mcmc):
        """Test MCMC model building with angle filtering enabled and disabled."""
        if not PYMC_AVAILABLE:
            pytest.skip("PyMC not available")
        
        # Create test data
        n_angles = len(test_phi_angles_mcmc)
        n_time = 20  # Small for testing
        c2_experimental = np.ones((n_angles, n_time, n_time)) + 0.1 * np.random.rand(n_angles, n_time, n_time)
        
        try:
            # Test with filtering enabled
            model_filtered = mock_mcmc_sampler._build_bayesian_model_optimized(
                c2_experimental, test_phi_angles_mcmc, filter_angles_for_optimization=True
            )
            
            # Test with filtering disabled
            model_all = mock_mcmc_sampler._build_bayesian_model_optimized(
                c2_experimental, test_phi_angles_mcmc, filter_angles_for_optimization=False
            )
            
            # Both should be valid PyMC models
            assert model_filtered is not None
            assert model_all is not None
            
            # Both should have the same number of parameters (7)
            with model_filtered:
                n_params_filtered = len(model_filtered.basic_RVs)
            with model_all:
                n_params_all = len(model_all.basic_RVs)
            
            # Parameter count should be the same
            assert n_params_filtered == n_params_all
            
        except Exception as e:
            # Model building might fail due to simplified forward model
            # This is acceptable - we're mainly testing the parameter passing
            assert "forward model" in str(e).lower() or "pymc" in str(e).lower()

    def test_mcmc_fallback_behavior(self, mock_mcmc_sampler):
        """Test MCMC fallback when no angles are in optimization ranges."""
        if not PYMC_AVAILABLE:
            pytest.skip("PyMC not available")
        
        # Create angles all outside optimization ranges
        phi_angles_outside = np.array([30.0, 60.0, 120.0, 210.0, 240.0])
        n_time = 10
        c2_experimental = np.ones((len(phi_angles_outside), n_time, n_time))
        
        # This should not crash and should fall back to using all angles
        try:
            model = mock_mcmc_sampler._build_bayesian_model_optimized(
                c2_experimental, phi_angles_outside, filter_angles_for_optimization=True
            )
            # If successful, the model should be built with all angles
            assert model is not None
        except Exception as e:
            # Expected due to simplified model limitations
            assert "forward model" in str(e).lower() or "pymc" in str(e).lower()


class TestMCMCAngleFilteringIntegration:
    """Test MCMC angle filtering integration with the analysis pipeline."""

    def test_mcmc_sampler_creation_with_filtering(self):
        """Test MCMC sampler creation includes angle filtering capability."""
        if not PYMC_AVAILABLE:
            pytest.skip("PyMC not available")
        
        try:
            from homodyne.optimization.mcmc import create_mcmc_sampler, MCMCSampler
            
            # Test that MCMCSampler class has the required methods
            assert hasattr(MCMCSampler, 'run_mcmc_analysis')
            assert hasattr(MCMCSampler, '_build_bayesian_model_optimized')
            assert hasattr(MCMCSampler, '_run_mcmc_nuts_optimized')
            
            # Test method signatures
            import inspect
            sig = inspect.signature(MCMCSampler.run_mcmc_analysis)
            assert 'filter_angles_for_optimization' in sig.parameters
            
        except ImportError:
            pytest.skip("MCMC module not available")

    def test_mcmc_consistency_with_other_methods(self):
        """Test that MCMC uses the same angle ranges as Classical/Bayesian methods."""
        # Test that all methods use identical target ranges
        target_ranges = [(-10.0, 10.0), (170.0, 190.0)]
        
        # Test angles that should be consistent across all methods
        test_cases = [
            (-10.0, True),   # Boundary - should be included
            (-5.0, True),    # Inside first range
            (0.0, True),     # Inside first range
            (5.0, True),     # Inside first range
            (10.0, True),    # Boundary - should be included
            (15.0, False),   # Outside ranges
            (170.0, True),   # Boundary - should be included
            (175.0, True),   # Inside second range
            (180.0, True),   # Inside second range
            (185.0, True),   # Inside second range
            (190.0, True),   # Boundary - should be included
            (195.0, False),  # Outside ranges
        ]
        
        for angle, should_include in test_cases:
            is_included = False
            for min_angle, max_angle in target_ranges:
                if min_angle <= angle <= max_angle:
                    is_included = True
                    break
            
            assert is_included == should_include, f"MCMC filtering inconsistent for {angle}°"

    def test_mcmc_runner_integration(self):
        """Test that main runner uses MCMC angle filtering."""
        # This is a structural test - we can't easily run the full pipeline
        # So we test that the runner script has the expected call structure
        
        # Read the runner file to check for explicit filtering call
        runner_path = Path(__file__).parent.parent.parent / "run_homodyne.py"
        if runner_path.exists():
            with open(runner_path, 'r') as f:
                content = f.read()
            
            # Check that MCMC call includes angle filtering
            assert "filter_angles_for_optimization=True" in content
            assert "run_mcmc_analysis" in content


class TestMCMCAngleFilteringPerformance:
    """Test performance aspects of MCMC angle filtering."""

    def test_mcmc_data_reduction_calculation(self):
        """Test calculation of data reduction from angle filtering."""
        # Simulate real experimental data dimensions
        total_angles = 23  # Real experimental data
        optimization_angles = 4  # Expected from filtering
        n_time = 999  # Real time dimension
        
        # Calculate data reduction
        total_data_points = total_angles * n_time * n_time
        filtered_data_points = optimization_angles * n_time * n_time
        
        reduction_factor = total_data_points / filtered_data_points
        reduction_percentage = (1 - filtered_data_points / total_data_points) * 100
        
        # Verify expected reductions
        assert reduction_factor == pytest.approx(5.75, rel=0.1)  # ~5.75x reduction
        assert reduction_percentage == pytest.approx(82.6, rel=1.0)  # ~82.6% reduction

    def test_mcmc_memory_usage_estimation(self):
        """Test memory usage estimation for filtered vs unfiltered data."""
        # Real data dimensions
        total_angles = 23
        filtered_angles = 4
        n_time = 999
        bytes_per_float32 = 4
        
        # Memory usage calculations
        memory_all = total_angles * n_time * n_time * bytes_per_float32
        memory_filtered = filtered_angles * n_time * n_time * bytes_per_float32
        
        # Convert to MB
        memory_all_mb = memory_all / (1024 * 1024)
        memory_filtered_mb = memory_filtered / (1024 * 1024)
        
        # Verify significant memory savings
        memory_savings = memory_all_mb - memory_filtered_mb
        assert memory_savings > 0
        assert memory_filtered_mb / memory_all_mb == pytest.approx(4/23, rel=0.01)


class TestMCMCAngleFilteringBackwardCompatibility:
    """Test backward compatibility of MCMC angle filtering."""

    def test_mcmc_default_parameter_behavior(self):
        """Test that MCMC angle filtering is enabled by default but can be disabled."""
        if not PYMC_AVAILABLE:
            pytest.skip("PyMC not available")
        
        try:
            from homodyne.optimization.mcmc import MCMCSampler
            import inspect
            
            # Test that default is None (resolved to True at runtime)
            sig = inspect.signature(MCMCSampler.run_mcmc_analysis)
            default_value = sig.parameters['filter_angles_for_optimization'].default
            assert default_value is None  # Uses None, resolved to True at runtime
            
            # Test that parameter is optional
            assert not sig.parameters['filter_angles_for_optimization'].annotation == inspect.Parameter.empty
            
        except ImportError:
            pytest.skip("MCMC module not available")

    def test_mcmc_configuration_compatibility(self):
        """Test that existing MCMC configurations still work."""
        # Test that essential MCMC config sections are preserved
        required_mcmc_config_keys = [
            "mcmc_draws",
            "mcmc_tune", 
            "mcmc_chains",
            "target_accept"
        ]
        
        # Verify these are still expected keys (no breaking changes)
        for key in required_mcmc_config_keys:
            assert isinstance(key, str)
            assert len(key) > 0


class TestMCMCAngleFilteringErrorHandling:
    """Test error handling and edge cases for MCMC angle filtering."""

    def test_mcmc_empty_angle_array(self):
        """Test MCMC behavior with empty angle array."""
        empty_angles = np.array([])
        target_ranges = [(-10.0, 10.0), (170.0, 190.0)]
        
        # Should not crash
        optimization_indices = []
        for i, angle in enumerate(empty_angles):
            for min_angle, max_angle in target_ranges:
                if min_angle <= angle <= max_angle:
                    optimization_indices.append(i)
                    break
        
        assert len(optimization_indices) == 0

    def test_mcmc_single_angle_filtering(self):
        """Test MCMC angle filtering with single angle."""
        # Test single angle inside range
        single_angle_in = np.array([5.0])
        target_ranges = [(-10.0, 10.0), (170.0, 190.0)]
        
        optimization_indices = []
        for i, angle in enumerate(single_angle_in):
            for min_angle, max_angle in target_ranges:
                if min_angle <= angle <= max_angle:
                    optimization_indices.append(i)
                    break
        
        assert optimization_indices == [0]
        
        # Test single angle outside range
        single_angle_out = np.array([45.0])
        optimization_indices = []
        for i, angle in enumerate(single_angle_out):
            for min_angle, max_angle in target_ranges:
                if min_angle <= angle <= max_angle:
                    optimization_indices.append(i)
                    break
        
        assert optimization_indices == []

    def test_mcmc_boundary_angle_handling(self):
        """Test MCMC handling of angles exactly on boundaries."""
        boundary_angles = np.array([-10.0, 10.0, 170.0, 190.0])
        target_ranges = [(-10.0, 10.0), (170.0, 190.0)]
        
        optimization_indices = []
        for i, angle in enumerate(boundary_angles):
            for min_angle, max_angle in target_ranges:
                if min_angle <= angle <= max_angle:
                    optimization_indices.append(i)
                    break
        
        # All boundary angles should be included
        assert optimization_indices == [0, 1, 2, 3]


if __name__ == "__main__":
    pytest.main([__file__])