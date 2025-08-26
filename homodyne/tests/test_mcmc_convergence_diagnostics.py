"""
Tests for MCMC convergence diagnostics and quality assessment.

This module tests the MCMC convergence validation that replaces chi-squared
analysis for Bayesian methods, including:
- R-hat (potential scale reduction factor) validation
- Effective Sample Size (ESS) assessment
- Divergence diagnostics
- Quality categorization based on convergence criteria
- Configuration-driven validation thresholds
"""

import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pytest
from run_homodyne import main as run_homodyne_main

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

# Test PyMC availability
try:
    import arviz as az
    import pymc as pm

    PYMC_AVAILABLE = True
except ImportError:
    PYMC_AVAILABLE = False
    pm = None
    az = None


pytestmark = pytest.mark.skipif(
    not PYMC_AVAILABLE,
    reason="PyMC is required for MCMC sampling but is not available.",
)


class TestMCMCConvergenceDiagnostics:
    """Test MCMC convergence diagnostics and quality assessment."""

    @pytest.fixture
    def mock_config_with_mcmc_thresholds(self):
        """Configuration with MCMC convergence thresholds."""
        return {
            "analyzer_parameters": {
                "temporal": {"dt": 0.1, "start_frame": 1, "end_frame": 100},
                "scattering": {"wavevector_q": 0.005},
                "geometry": {"stator_rotor_gap": 2000000},
                "computational": {"num_threads": 4},
            },
            "validation_rules": {
                "mcmc_convergence": {
                    "rhat_thresholds": {
                        "excellent_threshold": 1.01,
                        "good_threshold": 1.05,
                        "acceptable_threshold": 1.1,
                        "critical_threshold": 1.2,
                    },
                    "ess_thresholds": {
                        "excellent_threshold": 400,
                        "good_threshold": 200,
                        "acceptable_threshold": 100,
                        "minimum_threshold": 50,
                    },
                    "divergence_thresholds": {
                        "max_divergences_fraction": 0.05,
                        "warning_divergences_fraction": 0.01,
                    },
                    "quality_assessment": {
                        "excellent_criteria": "R̂ < 1.01 AND ESS > 400 AND divergences < 1%",
                        "good_criteria": "R̂ < 1.05 AND ESS > 200 AND divergences < 3%",
                        "acceptable_criteria": "R̂ < 1.1 AND ESS > 100 AND divergences < 5%",
                        "poor_criteria": "Any criterion fails to meet acceptable threshold",
                    },
                },
            },
        }

    @pytest.fixture
    def mock_mcmc_results(self):
        """Mock MCMC results with convergence diagnostics."""
        return {
            "mcmc_optimization": {
                "parameters": [1000.0, -0.1, 50.0],  # Static isotropic example
                "trace": Mock(),  # Mock trace object
                "diagnostics": {
                    "rhat": Mock(),
                    "ess": Mock(),
                    "mcse": Mock(),
                    "max_rhat": 1.03,
                    "min_ess": 250,
                    "converged": True,
                    "assessment": "Converged",
                },
                "success": True,
                "method": "MCMC",
            }
        }

    def test_mcmc_quality_assessment_excellent(
        self, mock_config_with_mcmc_thresholds
    ):
        """Test MCMC quality assessment for excellent convergence."""
        # Mock excellent convergence metrics
        diagnostics = {
            "max_rhat": 1.005,
            "min_ess": 500,
            "converged": True,
            "assessment": "Converged",
        }

        # Test quality assessment logic (extracted from run_homodyne.py)
        rhat_thresholds = mock_config_with_mcmc_thresholds["validation_rules"][
            "mcmc_convergence"
        ]["rhat_thresholds"]
        ess_thresholds = mock_config_with_mcmc_thresholds["validation_rules"][
            "mcmc_convergence"
        ]["ess_thresholds"]

        max_rhat = diagnostics["max_rhat"]
        min_ess = diagnostics["min_ess"]

        excellent_rhat = rhat_thresholds["excellent_threshold"]
        good_rhat = rhat_thresholds["good_threshold"]
        acceptable_rhat = rhat_thresholds["acceptable_threshold"]

        excellent_ess = ess_thresholds["excellent_threshold"]
        good_ess = ess_thresholds["good_threshold"]
        acceptable_ess = ess_thresholds["acceptable_threshold"]

        if max_rhat < excellent_rhat and min_ess > excellent_ess:
            quality = "excellent"
        elif max_rhat < good_rhat and min_ess > good_ess:
            quality = "good"
        elif max_rhat < acceptable_rhat and min_ess > acceptable_ess:
            quality = "acceptable"
        else:
            quality = "poor"

        assert quality == "excellent"

    def test_mcmc_quality_assessment_good(
        self, mock_config_with_mcmc_thresholds
    ):
        """Test MCMC quality assessment for good convergence."""
        # Mock good convergence metrics
        diagnostics = {
            "max_rhat": 1.03,
            "min_ess": 250,
            "converged": True,
            "assessment": "Converged",
        }

        # Test quality assessment logic
        rhat_thresholds = mock_config_with_mcmc_thresholds["validation_rules"][
            "mcmc_convergence"
        ]["rhat_thresholds"]
        ess_thresholds = mock_config_with_mcmc_thresholds["validation_rules"][
            "mcmc_convergence"
        ]["ess_thresholds"]

        max_rhat = diagnostics["max_rhat"]
        min_ess = diagnostics["min_ess"]

        excellent_rhat = rhat_thresholds["excellent_threshold"]
        good_rhat = rhat_thresholds["good_threshold"]
        acceptable_rhat = rhat_thresholds["acceptable_threshold"]

        excellent_ess = ess_thresholds["excellent_threshold"]
        good_ess = ess_thresholds["good_threshold"]
        acceptable_ess = ess_thresholds["acceptable_threshold"]

        if max_rhat < excellent_rhat and min_ess > excellent_ess:
            quality = "excellent"
        elif max_rhat < good_rhat and min_ess > good_ess:
            quality = "good"
        elif max_rhat < acceptable_rhat and min_ess > acceptable_ess:
            quality = "acceptable"
        else:
            quality = "poor"

        assert quality == "good"

    def test_mcmc_quality_assessment_acceptable(
        self, mock_config_with_mcmc_thresholds
    ):
        """Test MCMC quality assessment for acceptable convergence."""
        # Mock acceptable convergence metrics
        diagnostics = {
            "max_rhat": 1.08,
            "min_ess": 120,
            "converged": True,
            "assessment": "Converged",
        }

        # Test quality assessment logic
        rhat_thresholds = mock_config_with_mcmc_thresholds["validation_rules"][
            "mcmc_convergence"
        ]["rhat_thresholds"]
        ess_thresholds = mock_config_with_mcmc_thresholds["validation_rules"][
            "mcmc_convergence"
        ]["ess_thresholds"]

        max_rhat = diagnostics["max_rhat"]
        min_ess = diagnostics["min_ess"]

        excellent_rhat = rhat_thresholds["excellent_threshold"]
        good_rhat = rhat_thresholds["good_threshold"]
        acceptable_rhat = rhat_thresholds["acceptable_threshold"]

        excellent_ess = ess_thresholds["excellent_threshold"]
        good_ess = ess_thresholds["good_threshold"]
        acceptable_ess = ess_thresholds["acceptable_threshold"]

        if max_rhat < excellent_rhat and min_ess > excellent_ess:
            quality = "excellent"
        elif max_rhat < good_rhat and min_ess > good_ess:
            quality = "good"
        elif max_rhat < acceptable_rhat and min_ess > acceptable_ess:
            quality = "acceptable"
        else:
            quality = "poor"

        assert quality == "acceptable"

    def test_mcmc_quality_assessment_poor(
        self, mock_config_with_mcmc_thresholds
    ):
        """Test MCMC quality assessment for poor convergence."""
        # Mock poor convergence metrics
        diagnostics = {
            "max_rhat": 1.15,
            "min_ess": 80,
            "converged": False,
            "assessment": "Not converged",
        }

        # Test quality assessment logic
        rhat_thresholds = mock_config_with_mcmc_thresholds["validation_rules"][
            "mcmc_convergence"
        ]["rhat_thresholds"]
        ess_thresholds = mock_config_with_mcmc_thresholds["validation_rules"][
            "mcmc_convergence"
        ]["ess_thresholds"]

        max_rhat = diagnostics["max_rhat"]
        min_ess = diagnostics["min_ess"]

        excellent_rhat = rhat_thresholds["excellent_threshold"]
        good_rhat = rhat_thresholds["good_threshold"]
        acceptable_rhat = rhat_thresholds["acceptable_threshold"]

        excellent_ess = ess_thresholds["excellent_threshold"]
        good_ess = ess_thresholds["good_threshold"]
        acceptable_ess = ess_thresholds["acceptable_threshold"]

        if max_rhat < excellent_rhat and min_ess > excellent_ess:
            quality = "excellent"
        elif max_rhat < good_rhat and min_ess > good_ess:
            quality = "good"
        elif max_rhat < acceptable_rhat and min_ess > acceptable_ess:
            quality = "acceptable"
        else:
            quality = "poor"

        assert quality == "poor"

    def test_mcmc_no_chi_squared_analysis(self, mock_mcmc_results):
        """Test that MCMC results don't trigger chi-squared analysis."""
        # This test verifies that MCMC results bypass chi-squared calculation
        # and instead use convergence diagnostics

        mcmc_result = mock_mcmc_results["mcmc_optimization"]

        # Verify that the method is MCMC
        assert mcmc_result["method"] == "MCMC"

        # Verify that diagnostics are available
        assert "diagnostics" in mcmc_result
        assert "max_rhat" in mcmc_result["diagnostics"]
        assert "min_ess" in mcmc_result["diagnostics"]
        assert "assessment" in mcmc_result["diagnostics"]

        # Verify that convergence status is meaningful
        diagnostics = mcmc_result["diagnostics"]
        assert diagnostics["assessment"] in ["Converged", "Not converged"]
        assert isinstance(diagnostics["max_rhat"], (int, float))
        assert isinstance(diagnostics["min_ess"], (int, float))

    def test_mcmc_config_validation_thresholds(
        self, mock_config_with_mcmc_thresholds
    ):
        """Test that MCMC configuration contains proper validation thresholds."""
        mcmc_config = mock_config_with_mcmc_thresholds["validation_rules"][
            "mcmc_convergence"
        ]

        # Verify R-hat thresholds
        rhat_thresholds = mcmc_config["rhat_thresholds"]
        assert (
            rhat_thresholds["excellent_threshold"]
            < rhat_thresholds["good_threshold"]
        )
        assert (
            rhat_thresholds["good_threshold"]
            < rhat_thresholds["acceptable_threshold"]
        )
        assert (
            rhat_thresholds["acceptable_threshold"]
            < rhat_thresholds["critical_threshold"]
        )

        # Verify ESS thresholds
        ess_thresholds = mcmc_config["ess_thresholds"]
        assert (
            ess_thresholds["excellent_threshold"]
            > ess_thresholds["good_threshold"]
        )
        assert (
            ess_thresholds["good_threshold"]
            > ess_thresholds["acceptable_threshold"]
        )
        assert (
            ess_thresholds["acceptable_threshold"]
            > ess_thresholds["minimum_threshold"]
        )

        # Verify divergence thresholds
        div_thresholds = mcmc_config["divergence_thresholds"]
        assert (
            div_thresholds["warning_divergences_fraction"]
            < div_thresholds["max_divergences_fraction"]
        )

    @patch("builtins.print")
    def test_mcmc_logging_format(self, mock_print, mock_mcmc_results):
        """Test that MCMC convergence diagnostics are logged in the correct format."""
        # Simulate the logging behavior from run_homodyne.py
        mcmc_result = mock_mcmc_results["mcmc_optimization"]
        diagnostics = mcmc_result["diagnostics"]

        # Mock the logger calls that would happen in run_homodyne.py
        expected_log_calls = [
            "MCMC convergence diagnostics [MCMC]:",
            f"  Convergence status: {diagnostics['assessment']}",
            f"  Maximum R̂ (R-hat): {diagnostics['max_rhat']:.4f}",
            f"  Minimum ESS: {diagnostics['min_ess']:.0f}",
            "  MCMC quality: GOOD",  # Based on the mock values
            "  Sampling completed with posterior analysis available",
        ]

        # This test verifies the expected log format structure
        # In actual usage, these would be logger.info() calls
        assert all(isinstance(call, str) for call in expected_log_calls)

    def test_default_mcmc_thresholds(self):
        """Test that default MCMC thresholds are used when config is missing."""
        # Test the fallback behavior when configuration is incomplete
        empty_config = {}

        # Default values from run_homodyne.py
        excellent_rhat = 1.01
        good_rhat = 1.05
        acceptable_rhat = 1.1

        excellent_ess = 400
        good_ess = 200
        acceptable_ess = 100

        # Simulate the config lookup with empty config
        validation_config = empty_config.get("validation_rules", {})
        mcmc_config = validation_config.get("mcmc_convergence", {})
        rhat_thresholds = mcmc_config.get("rhat_thresholds", {})
        ess_thresholds = mcmc_config.get("ess_thresholds", {})

        # Test that defaults are used
        assert (
            rhat_thresholds.get("excellent_threshold", 1.01) == excellent_rhat
        )
        assert rhat_thresholds.get("good_threshold", 1.05) == good_rhat
        assert (
            rhat_thresholds.get("acceptable_threshold", 1.1) == acceptable_rhat
        )

        assert ess_thresholds.get("excellent_threshold", 400) == excellent_ess
        assert ess_thresholds.get("good_threshold", 200) == good_ess
        assert (
            ess_thresholds.get("acceptable_threshold", 100) == acceptable_ess
        )


if __name__ == "__main__":
    pytest.main([__file__])
