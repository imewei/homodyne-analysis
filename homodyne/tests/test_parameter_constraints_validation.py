"""
Test Parameter Constraints Validation System
============================================

Tests for the parameter constraints consistency checking system that ensures
documentation and configuration files maintain consistent parameter ranges,
distributions, and physical constraints across the homodyne package.

This test suite validates:
1. Authoritative parameter constraints definition
2. Documentation constraint checking
3. Configuration file constraint validation
4. Constraint synchronization tools
5. Gap analysis and reporting functionality

Coverage:
- check_constraints.py functionality
- fix_constraints.py functionality  
- Parameter bounds consistency
- Prior distribution validation
- Physical constraint enforcement
- Static vs flow mode parameter handling
"""

import pytest
import json
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch, mock_open
import sys

# Add the project root to the path to import our constraint checking tools
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    from check_constraints import (
        PARAMETER_CONSTRAINTS,
        check_documentation_constraints,
        check_config_constraints,
        find_all_files,
        main as check_main
    )
    from fix_constraints import (
        generate_parameter_table_markdown,
        fix_readme_constraints,
        fix_config_constraints,
        create_gap_analysis_report,
        backup_file,
        main as fix_main
    )
except ImportError:
    pytest.skip("Constraint checking modules not available", allow_module_level=True)


class TestAuthorativeConstraints:
    """Test the authoritative parameter constraints definition."""
    
    def test_parameter_constraints_structure(self):
        """Test that PARAMETER_CONSTRAINTS has the expected structure."""
        assert "core_parameters" in PARAMETER_CONSTRAINTS
        assert "scaling_parameters" in PARAMETER_CONSTRAINTS
        assert "physical_functions" in PARAMETER_CONSTRAINTS
        assert "mcmc_config" in PARAMETER_CONSTRAINTS
    
    def test_core_parameters_completeness(self):
        """Test that all expected core parameters are defined."""
        expected_params = {
            "D0", "alpha", "D_offset", "gamma_dot_t0", 
            "beta", "gamma_dot_t_offset", "phi0"
        }
        actual_params = set(PARAMETER_CONSTRAINTS["core_parameters"].keys())
        assert actual_params == expected_params
    
    def test_parameter_constraint_fields(self):
        """Test that each parameter has required constraint fields."""
        required_fields = {
            "range", "unit", "distribution", "prior_mu", 
            "prior_sigma", "physical_constraint", "description"
        }
        
        for param_name, constraints in PARAMETER_CONSTRAINTS["core_parameters"].items():
            actual_fields = set(constraints.keys())
            assert required_fields.issubset(actual_fields), \
                f"Parameter {param_name} missing fields: {required_fields - actual_fields}"
    
    def test_parameter_ranges_validity(self):
        """Test that parameter ranges are physically reasonable."""
        for param_name, constraints in PARAMETER_CONSTRAINTS["core_parameters"].items():
            min_val, max_val = constraints["range"]
            assert min_val < max_val, f"Invalid range for {param_name}: [{min_val}, {max_val}]"
            
            # Test specific parameter constraints
            if param_name == "D0":
                assert min_val > 0, "D0 must have positive minimum"
                assert max_val >= 1e6, "D0 should allow large diffusion coefficients"
                
            elif param_name == "gamma_dot_t0":
                assert min_val > 0, "gamma_dot_t0 must have positive minimum"
                assert min_val <= 1e-6, "gamma_dot_t0 minimum should allow very small shear rates"
                
            elif param_name in ["alpha", "beta"]:
                assert min_val >= -2.0, f"{param_name} minimum should be >= -2.0"
                assert max_val <= 2.0, f"{param_name} maximum should be <= 2.0"
    
    def test_scaling_parameters_structure(self):
        """Test scaling parameters structure."""
        expected_scaling = {"contrast", "offset", "c2_fitted", "c2_theory"}
        actual_scaling = set(PARAMETER_CONSTRAINTS["scaling_parameters"].keys())
        assert actual_scaling == expected_scaling
        
        # Test derived vs fitted parameters
        assert PARAMETER_CONSTRAINTS["scaling_parameters"]["c2_fitted"]["type"] == "derived"
        assert PARAMETER_CONSTRAINTS["scaling_parameters"]["c2_theory"]["type"] == "derived"
    
    def test_physical_functions_completeness(self):
        """Test physical function constraints."""
        expected_functions = {"D_time", "gamma_dot_time"}
        actual_functions = set(PARAMETER_CONSTRAINTS["physical_functions"].keys())
        assert actual_functions == expected_functions
        
        for func_name, func_info in PARAMETER_CONSTRAINTS["physical_functions"].items():
            assert "formula" in func_info
            assert "constraint" in func_info
            assert "purpose" in func_info


class TestDocumentationConstraintChecking:
    """Test documentation constraint validation functionality."""
    
    @pytest.fixture
    def temp_directory(self):
        """Create temporary directory for test files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)
    
    def create_valid_documentation(self):
        """Create a valid documentation content with parameter constraints."""
        return """
# Test Documentation

## Physical Constraints and Parameter Ranges

### Core Model Parameters

| Parameter | Range | Distribution | Physical Constraint |
|-----------|-------|--------------|-------------------|
| `D0` | [1.0, 1000000.0] Å²/s | TruncatedNormal(μ=10000.0, σ=1000.0) | positive |
| `alpha` | [-2.0, 2.0] dimensionless | Normal(μ=-1.5, σ=0.1) | none |
| `D_offset` | [-100.0, 100.0] Å²/s | Normal(μ=0.0, σ=10.0) | none |
| `gamma_dot_t0` | [1e-06, 1.0] s⁻¹ | TruncatedNormal(μ=0.001, σ=0.01) | positive |
| `beta` | [-2.0, 2.0] dimensionless | Normal(μ=0.0, σ=0.1) | none |
| `gamma_dot_t_offset` | [-0.01, 0.01] s⁻¹ | Normal(μ=0.0, σ=0.001) | none |
| `phi0` | [-10.0, 10.0] degrees | Normal(μ=0.0, σ=5.0) | none |

### Physical Function Constraints

The package automatically enforces positivity for time-dependent functions.

### Scaling Parameters for Correlation Functions

The relationship **c2_fitted = c2_theory × contrast + offset** uses bounded parameters.
"""
    
    def create_invalid_documentation(self):
        """Create documentation with incorrect parameter ranges."""
        return """
# Test Documentation

## Physical Constraints and Parameter Ranges

### Core Model Parameters

| Parameter | Range | Distribution | Physical Constraint |
|-----------|-------|--------------|-------------------|
| `D0` | [1.0, 1000.0] Å²/s | TruncatedNormal(μ=10000.0, σ=1000.0) | positive |
| `alpha` | [-1.0, 1.0] dimensionless | Normal(μ=-1.5, σ=0.1) | none |

### Physical Function Constraints

Functions are constrained.
"""
    
    def test_valid_documentation_passes(self, temp_directory):
        """Test that valid documentation passes constraint checking."""
        doc_file = temp_directory / "valid_doc.md"
        doc_file.write_text(self.create_valid_documentation())
        
        issues = check_documentation_constraints(str(doc_file))
        assert len(issues) == 0, f"Valid documentation should pass, but got: {issues}"
    
    def test_missing_sections_detected(self, temp_directory):
        """Test detection of missing documentation sections."""
        doc_file = temp_directory / "missing_sections.md"
        doc_file.write_text("# Documentation without parameter constraints")
        
        issues = check_documentation_constraints(str(doc_file))
        assert any("Missing 'Core Model Parameters' section" in issue for issue in issues)
    
    def test_incorrect_ranges_detected(self, temp_directory):
        """Test detection of incorrect parameter ranges."""
        doc_file = temp_directory / "wrong_ranges.md"
        doc_file.write_text(self.create_invalid_documentation())
        
        issues = check_documentation_constraints(str(doc_file))
        
        # Should detect incorrect D0 range (1000.0 instead of 1000000.0)
        d0_range_issue = any("D0" in issue and "incorrect range" in issue for issue in issues)
        assert d0_range_issue, f"Should detect incorrect D0 range, issues: {issues}"
    
    def test_file_read_error_handling(self, temp_directory):
        """Test handling of file read errors."""
        nonexistent_file = temp_directory / "nonexistent.md"
        issues = check_documentation_constraints(str(nonexistent_file))
        assert any("Error reading" in issue for issue in issues)


class TestConfigurationConstraintChecking:
    """Test configuration file constraint validation functionality."""
    
    @pytest.fixture
    def temp_directory(self):
        """Create temporary directory for test files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)
    
    def create_valid_config(self):
        """Create a valid configuration with correct parameter constraints."""
        return {
            "analysis_settings": {"static_mode": False},
            "parameter_space": {
                "bounds": [
                    {
                        "name": "D0",
                        "min": 1.0,
                        "max": 1000000.0,
                        "type": "TruncatedNormal",
                        "prior_mu": 10000.0,
                        "prior_sigma": 1000.0,
                        "unit": "Å²/s",
                        "physical_constraint": "positive"
                    },
                    {
                        "name": "alpha", 
                        "min": -2.0,
                        "max": 2.0,
                        "type": "Normal",
                        "prior_mu": -1.5,
                        "prior_sigma": 0.1,
                        "unit": "dimensionless",
                        "physical_constraint": "none"
                    }
                ]
            },
            "optimization_config": {
                "scaling_parameters": {
                    "contrast": {
                        "min": 0.05,
                        "max": 0.5,
                        "prior_mu": 0.3,
                        "prior_sigma": 0.1,
                        "type": "TruncatedNormal"
                    },
                    "offset": {
                        "min": 0.05,
                        "max": 1.95,
                        "prior_mu": 1.0,
                        "prior_sigma": 0.2,
                        "type": "TruncatedNormal"
                    }
                }
            }
        }
    
    def create_static_config(self):
        """Create a static mode configuration with fixed flow parameters."""
        return {
            "analysis_settings": {"static_mode": True},
            "parameter_space": {
                "bounds": [
                    {
                        "name": "D0",
                        "min": 1.0,
                        "max": 1000000.0,
                        "type": "TruncatedNormal",
                        "prior_mu": 10000.0,
                        "prior_sigma": 1000.0
                    },
                    {
                        "name": "gamma_dot_t0",
                        "min": 0.0,
                        "max": 0.0,
                        "type": "fixed",
                        "_reference_min": 1e-6,
                        "_reference_max": 1.0,
                        "_reference_type": "TruncatedNormal",
                        "_reference_prior_mu": 0.001,
                        "_reference_prior_sigma": 0.01
                    }
                ]
            }
        }
    
    def create_invalid_config(self):
        """Create configuration with incorrect parameter constraints."""
        return {
            "analysis_settings": {"static_mode": False},
            "parameter_space": {
                "bounds": [
                    {
                        "name": "D0",
                        "min": 1.0,
                        "max": 1000.0,  # Wrong max value
                        "type": "TruncatedNormal",
                        "prior_mu": 5000.0,  # Wrong prior
                        "prior_sigma": 500.0  # Wrong prior
                    }
                ]
            }
        }
    
    def test_valid_config_passes(self, temp_directory):
        """Test that valid configuration passes constraint checking."""
        config_file = temp_directory / "valid_config.json"
        config = self.create_valid_config()
        
        with open(config_file, 'w') as f:
            json.dump(config, f)
        
        issues = check_config_constraints(str(config_file))
        assert len(issues) == 0, f"Valid config should pass, but got: {issues}"
    
    def test_static_mode_handling(self, temp_directory):
        """Test correct handling of static mode configurations."""
        config_file = temp_directory / "static_config.json" 
        config = self.create_static_config()
        
        with open(config_file, 'w') as f:
            json.dump(config, f)
        
        issues = check_config_constraints(str(config_file))
        assert len(issues) == 0, f"Static config should pass, but got: {issues}"
    
    def test_incorrect_bounds_detected(self, temp_directory):
        """Test detection of incorrect parameter bounds."""
        config_file = temp_directory / "invalid_config.json"
        config = self.create_invalid_config()
        
        with open(config_file, 'w') as f:
            json.dump(config, f)
        
        issues = check_config_constraints(str(config_file))
        
        # Should detect incorrect bounds and priors
        assert any("D0 max should be 1000000.0" in issue for issue in issues)
        assert any("D0 prior_mu should be 10000.0" in issue for issue in issues)
    
    def test_missing_parameter_space(self, temp_directory):
        """Test handling of missing parameter_space section."""
        config_file = temp_directory / "no_bounds.json"
        config = {"analysis_settings": {"static_mode": False}}
        
        with open(config_file, 'w') as f:
            json.dump(config, f)
        
        issues = check_config_constraints(str(config_file))
        assert any("Missing parameter_space.bounds" in issue for issue in issues)
    
    def test_json_parsing_error(self, temp_directory):
        """Test handling of invalid JSON files."""
        config_file = temp_directory / "invalid.json"
        config_file.write_text("{ invalid json }")
        
        issues = check_config_constraints(str(config_file))
        assert any("Error reading" in issue for issue in issues)


class TestConstraintSynchronization:
    """Test constraint synchronization and fixing functionality."""
    
    @pytest.fixture
    def temp_directory(self):
        """Create temporary directory for test files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)
    
    def test_markdown_table_generation(self):
        """Test generation of parameter constraints markdown table."""
        table = generate_parameter_table_markdown()
        
        # Should contain all core parameters
        for param in PARAMETER_CONSTRAINTS["core_parameters"].keys():
            assert f"`{param}`" in table
        
        # Should contain physical function constraints
        assert "Physical Function Constraints" in table
        assert "D(t) = D₀(t)^α + D_offset" in table
        
        # Should contain scaling parameters
        assert "Scaling Parameters for Correlation Functions" in table
        assert "`contrast`" in table
        assert "`offset`" in table
    
    def test_backup_file_creation(self, temp_directory):
        """Test backup file creation during fixes."""
        original_file = temp_directory / "original.txt"
        original_file.write_text("Original content")
        
        backup_path = backup_file(str(original_file))
        
        assert Path(backup_path).exists()
        assert Path(backup_path).read_text() == "Original content"
        assert "backup_" in backup_path
    
    def test_config_constraint_fixing(self, temp_directory):
        """Test fixing of configuration file constraints."""
        config_file = temp_directory / "broken_config.json"
        config = {
            "analysis_settings": {"static_mode": False},
            "parameter_space": {
                "bounds": [
                    {
                        "name": "D0",
                        "min": 1.0,
                        "max": 1000.0,  # Wrong value
                        "type": "Normal",  # Wrong type
                        "prior_mu": 5000.0,  # Wrong prior
                        "prior_sigma": 500.0  # Wrong prior
                    }
                ]
            }
        }
        
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        changes = fix_config_constraints(str(config_file))
        
        # Should report changes made
        assert any("Updated D0 max bound" in change for change in changes)
        assert any("Updated D0 distribution type" in change for change in changes)
        assert any("Updated D0 prior_mu" in change for change in changes)
        
        # Verify config was actually fixed
        with open(config_file, 'r') as f:
            fixed_config = json.load(f)
        
        d0_bound = fixed_config["parameter_space"]["bounds"][0]
        assert d0_bound["max"] == 1000000.0
        assert d0_bound["type"] == "TruncatedNormal"
        assert d0_bound["prior_mu"] == 10000.0
    
    def test_static_mode_reference_bounds_addition(self, temp_directory):
        """Test addition of reference bounds to static mode configs."""
        config_file = temp_directory / "static_config.json"
        config = {
            "analysis_settings": {"static_mode": True},
            "parameter_space": {
                "bounds": [
                    {
                        "name": "gamma_dot_t0",
                        "min": 0.0,
                        "max": 0.0,
                        "type": "fixed"
                    }
                ]
            }
        }
        
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        changes = fix_config_constraints(str(config_file))
        
        # Should report addition of reference bounds
        assert any("Added reference bounds for fixed parameter gamma_dot_t0" in change for change in changes)
        
        # Verify reference bounds were added
        with open(config_file, 'r') as f:
            fixed_config = json.load(f)
        
        gamma_bound = fixed_config["parameter_space"]["bounds"][0]
        assert "_reference_min" in gamma_bound
        assert "_reference_max" in gamma_bound
        assert gamma_bound["_reference_min"] == 1e-6
        assert gamma_bound["_reference_max"] == 1.0
    
    def test_gap_analysis_report_generation(self):
        """Test gap analysis report generation."""
        mock_changes = [
            "Updated D0 max bound in config1.json",
            "Updated alpha prior_mu in config2.json", 
            "Added reference bounds for fixed parameter gamma_dot_t0 in static.json"
        ]
        
        report = create_gap_analysis_report(mock_changes)
        
        assert "Parameter Constraints Synchronization Report" in report
        assert "Total changes: 3" in report
        assert "config1.json" in report
        assert "config2.json" in report
        assert "static.json" in report


class TestConstraintCheckingIntegration:
    """Integration tests for the complete constraint checking system."""
    
    @pytest.fixture
    def temp_directory(self):
        """Create temporary directory for test files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)
    
    def test_find_all_files_detection(self, temp_directory):
        """Test file discovery functionality."""
        # Create test files
        (temp_directory / "README.md").write_text("readme content")
        (temp_directory / "docs").mkdir()
        (temp_directory / "docs" / "config.rst").write_text("doc content")
        (temp_directory / "test_config.json").write_text('{"test": "config"}')
        
        with patch('glob.glob') as mock_glob:
            # Mock glob to return our test files
            mock_glob.side_effect = [
                [str(temp_directory / "README.md")],  # README pattern
                [],  # .md pattern
                [str(temp_directory / "docs" / "config.rst")],  # docs rst pattern
                [],  # docs md pattern
                [str(temp_directory / "test_config.json")],  # first json pattern
                [str(temp_directory / "test_config.json")],  # second json pattern
            ]
            
            files = find_all_files()
            
            assert len(files["readme"]) >= 1
            assert len(files["configs"]) >= 1
    
    @patch('check_constraints.find_all_files')
    @patch('check_constraints.check_documentation_constraints')
    @patch('check_constraints.check_config_constraints')
    def test_main_function_execution(self, mock_config_check, mock_doc_check, mock_find_files):
        """Test main constraint checking function."""
        # Setup mocks
        mock_find_files.return_value = {
            "readme": ["README.md"],
            "docs": ["docs/config.rst"], 
            "configs": ["config.json"]
        }
        mock_doc_check.return_value = []  # No issues
        mock_config_check.return_value = []  # No issues
        
        # Mock os.path.exists to return True
        with patch('os.path.exists', return_value=True):
            exit_code = check_main()
        
        assert exit_code == 0  # Should succeed with no issues
        mock_doc_check.assert_called()
        mock_config_check.assert_called()
    
    def test_constraint_checking_with_issues(self):
        """Test constraint checking when issues are found."""
        with patch('check_constraints.find_all_files') as mock_find:
            with patch('check_constraints.check_documentation_constraints') as mock_doc_check:
                with patch('check_constraints.check_config_constraints') as mock_config_check:
                    with patch('os.path.exists', return_value=True):
                        # Setup mocks with issues
                        mock_find.return_value = {
                            "readme": ["README.md"],
                            "docs": [],
                            "configs": ["config.json"]
                        }
                        mock_doc_check.return_value = ["README.md: Missing parameter table"]
                        mock_config_check.return_value = ["config.json: Wrong D0 bounds"]
                        
                        exit_code = check_main()
                        
                        assert exit_code == 2  # Should return number of issues found


class TestConstraintCheckingCoverage:
    """Test coverage of constraint checking functionality."""
    
    def test_all_core_parameters_covered(self):
        """Test that constraint checking covers all core parameters."""
        from check_constraints import PARAMETER_CONSTRAINTS
        
        # Verify all parameters have comprehensive constraint definitions
        for param_name, constraints in PARAMETER_CONSTRAINTS["core_parameters"].items():
            assert "range" in constraints
            assert "distribution" in constraints
            assert "prior_mu" in constraints
            assert "prior_sigma" in constraints
            assert "physical_constraint" in constraints
    
    def test_constraint_checking_parameter_coverage(self):
        """Test that check_config_constraints handles all parameters."""
        # Create a comprehensive config with all parameters
        config = {
            "analysis_settings": {"static_mode": False},
            "parameter_space": {
                "bounds": [
                    {
                        "name": param_name,
                        "min": constraints["range"][0],
                        "max": constraints["range"][1],
                        "type": constraints["distribution"],
                        "prior_mu": constraints["prior_mu"],
                        "prior_sigma": constraints["prior_sigma"]
                    }
                    for param_name, constraints in PARAMETER_CONSTRAINTS["core_parameters"].items()
                ]
            },
            "optimization_config": {
                "scaling_parameters": {
                    "contrast": {
                        "min": 0.05,
                        "max": 0.5,
                        "prior_mu": 0.3,
                        "prior_sigma": 0.1,
                        "type": "TruncatedNormal"
                    },
                    "offset": {
                        "min": 0.05,
                        "max": 1.95,
                        "prior_mu": 1.0,
                        "prior_sigma": 0.2,
                        "type": "TruncatedNormal"
                    }
                }
            }
        }
        
        # Write to temporary file and test
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config, f)
            temp_path = f.name
        
        try:
            issues = check_config_constraints(temp_path)
            # Should pass with no issues since all constraints match
            assert len(issues) == 0, f"Comprehensive config should pass, but got: {issues}"
        finally:
            os.unlink(temp_path)


@pytest.mark.integration
class TestEndToEndConstraintWorkflow:
    """End-to-end tests for the complete constraint management workflow."""
    
    @pytest.fixture
    def temp_project(self):
        """Create a temporary project structure for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir)
            
            # Create project structure
            (project_dir / "docs").mkdir()
            (project_dir / "configs").mkdir()
            
            # Create sample files
            (project_dir / "README.md").write_text("""
# Test Project

## Physical Constraints and Parameter Ranges

### Core Model Parameters

| Parameter | Range | Distribution | Physical Constraint |
|-----------|-------|--------------|-------------------|
| `D0` | [1.0, 1000.0] Å²/s | TruncatedNormal(μ=5000.0, σ=500.0) | positive |

### Physical Function Constraints
### Scaling Parameters for Correlation Functions
""")
            
            (project_dir / "configs" / "test.json").write_text(json.dumps({
                "analysis_settings": {"static_mode": False},
                "parameter_space": {
                    "bounds": [
                        {
                            "name": "D0",
                            "min": 1.0,
                            "max": 1000.0,  # Wrong
                            "type": "Normal",  # Wrong
                            "prior_mu": 5000.0,  # Wrong
                            "prior_sigma": 500.0  # Wrong
                        }
                    ]
                }
            }, indent=2))
            
            yield project_dir
    
    def test_detection_and_fixing_workflow(self, temp_project):
        """Test the complete detection → fixing → validation workflow."""
        # Step 1: Detect issues
        readme_issues = check_documentation_constraints(str(temp_project / "README.md"))
        config_issues = check_config_constraints(str(temp_project / "configs" / "test.json"))
        
        # Should find issues
        assert len(readme_issues) > 0
        assert len(config_issues) > 0
        
        # Step 2: Fix issues
        readme_changes = fix_readme_constraints(str(temp_project / "README.md"))
        config_changes = fix_config_constraints(str(temp_project / "configs" / "test.json"))
        
        # Should report changes
        assert len(readme_changes) > 0
        assert len(config_changes) > 0
        
        # Step 3: Verify fixes worked
        readme_issues_after = check_documentation_constraints(str(temp_project / "README.md"))
        config_issues_after = check_config_constraints(str(temp_project / "configs" / "test.json"))
        
        # Should have no issues after fixing
        assert len(readme_issues_after) == 0, f"README should be fixed, but still has: {readme_issues_after}"
        assert len(config_issues_after) == 0, f"Config should be fixed, but still has: {config_issues_after}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
