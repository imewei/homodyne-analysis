"""
Authoritative Parameter Constraints for Homodyne Package
========================================================

This module defines the single source of truth for parameter constraints, ranges,
prior distributions, and physical constraints used throughout the homodyne package.

Used by:
- Documentation generation
- Configuration validation
- Test suite validation
- Parameter synchronization tools

Authors: Wei Chen, Hongrui He
Institution: Argonne National Laboratory & University of Chicago
"""

# Authoritative parameter constraints - single source of truth
PARAMETER_CONSTRAINTS = {
    "core_parameters": {
        "D0": {
            "range": [1.0, 1000000.0],
            "unit": "Å²/s",
            "distribution": "TruncatedNormal",
            "prior_mu": 10000.0,
            "prior_sigma": 1000.0,
            "physical_constraint": "positive",
            "description": "Diffusion coefficient at reference time"
        },
        "alpha": {
            "range": [-2.0, 2.0],
            "unit": "dimensionless",
            "distribution": "Normal",
            "prior_mu": -1.5,
            "prior_sigma": 0.1,
            "physical_constraint": "none",
            "description": "Time-dependence exponent for diffusion"
        },
        "D_offset": {
            "range": [-100.0, 100.0],
            "unit": "Å²/s",
            "distribution": "Normal",
            "prior_mu": 0.0,
            "prior_sigma": 10.0,
            "physical_constraint": "none",
            "description": "Diffusion offset parameter"
        },
        "gamma_dot_t0": {
            "range": [1e-6, 1.0],
            "unit": "s⁻¹",
            "distribution": "TruncatedNormal",
            "prior_mu": 0.001,
            "prior_sigma": 0.01,
            "physical_constraint": "positive",
            "description": "Shear rate at reference time"
        },
        "beta": {
            "range": [-2.0, 2.0],
            "unit": "dimensionless",
            "distribution": "Normal",
            "prior_mu": 0.0,
            "prior_sigma": 0.1,
            "physical_constraint": "none",
            "description": "Time-dependence exponent for shear rate"
        },
        "gamma_dot_t_offset": {
            "range": [-0.01, 0.01],
            "unit": "s⁻¹",
            "distribution": "Normal",
            "prior_mu": 0.0,
            "prior_sigma": 0.001,
            "physical_constraint": "none",
            "description": "Shear rate offset parameter"
        },
        "phi0": {
            "range": [-10.0, 10.0],
            "unit": "degrees",
            "distribution": "Normal",
            "prior_mu": 0.0,
            "prior_sigma": 5.0,
            "physical_constraint": "none",
            "description": "Phase offset for angular correlation"
        }
    },
    "scaling_parameters": {
        "contrast": {
            "range": [0.05, 0.5],
            "unit": "dimensionless",
            "distribution": "TruncatedNormal",
            "prior_mu": 0.3,
            "prior_sigma": 0.1,
            "type": "fitted",
            "description": "Scaling factor for correlation strength"
        },
        "offset": {
            "range": [0.05, 1.95],
            "unit": "dimensionless",
            "distribution": "TruncatedNormal",
            "prior_mu": 1.0,
            "prior_sigma": 0.2,
            "type": "fitted",
            "description": "Baseline correlation level"
        },
        "c2_fitted": {
            "range": [1.0, 2.0],
            "unit": "dimensionless",
            "formula": "c2_theory × contrast + offset",
            "type": "derived",
            "description": "Fitted correlation function values"
        },
        "c2_theory": {
            "range": [0.0, 1.0],
            "unit": "dimensionless",
            "type": "derived",
            "description": "Theoretical correlation function values"
        }
    },
    "physical_functions": {
        "D_time": {
            "formula": "D(t) = D₀(t)^α + D_offset",
            "constraint": "positive for all t",
            "purpose": "Time-dependent diffusion coefficient"
        },
        "gamma_dot_time": {
            "formula": "γ̇(t) = γ̇₀(t)^β + γ̇_offset",
            "constraint": "positive for all t in flow analysis",
            "purpose": "Time-dependent shear rate"
        }
    },
    "mcmc_config": {
        "draws": {
            "default": 2000,
            "range": [500, 10000],
            "description": "Number of MCMC samples"
        },
        "tune": {
            "default": 500,
            "range": [100, 2000],
            "description": "Number of tuning steps"
        },
        "chains": {
            "default": 2,
            "range": [1, 8],
            "description": "Number of parallel chains"
        }
    }
}


def get_parameter_constraint(param_name):
    """Get constraint information for a specific parameter."""
    if param_name in PARAMETER_CONSTRAINTS["core_parameters"]:
        return PARAMETER_CONSTRAINTS["core_parameters"][param_name]
    elif param_name in PARAMETER_CONSTRAINTS["scaling_parameters"]:
        return PARAMETER_CONSTRAINTS["scaling_parameters"][param_name]
    else:
        raise ValueError(f"Unknown parameter: {param_name}")


def get_all_core_parameters():
    """Get list of all core parameter names."""
    return list(PARAMETER_CONSTRAINTS["core_parameters"].keys())


def get_all_scaling_parameters():
    """Get list of all scaling parameter names."""
    return list(PARAMETER_CONSTRAINTS["scaling_parameters"].keys())


def validate_parameter_constraints():
    """Validate that parameter constraints are internally consistent."""
    errors = []
    
    # Check core parameters
    for param_name, constraints in PARAMETER_CONSTRAINTS["core_parameters"].items():
        # Check required fields
        required_fields = ["range", "unit", "distribution", "prior_mu", "prior_sigma", "physical_constraint"]
        for field in required_fields:
            if field not in constraints:
                errors.append(f"{param_name} missing required field: {field}")
        
        # Check range validity
        if "range" in constraints:
            min_val, max_val = constraints["range"]
            if min_val >= max_val:
                errors.append(f"{param_name} has invalid range: [{min_val}, {max_val}]")
        
        # Check physical constraint consistency
        if constraints.get("physical_constraint") == "positive" and constraints["range"][0] <= 0:
            errors.append(f"{param_name} marked positive but range includes non-positive values")
    
    return errors


import json
import re
import os
import glob
from pathlib import Path


def check_documentation_constraints(file_path):
    """Check if documentation file contains correct parameter constraints."""
    issues = []
    
    try:
        with open(file_path, 'r') as f:
            content = f.read()
    except Exception as e:
        return [f"Error reading {file_path}: {str(e)}"]
    
    # Check for required sections
    if "Core Model Parameters" not in content:
        issues.append(f"{file_path}: Missing 'Core Model Parameters' section")
    
    if "Physical Function Constraints" not in content:
        issues.append(f"{file_path}: Missing 'Physical Function Constraints' section")
    
    if "Scaling Parameters for Correlation Functions" not in content:
        issues.append(f"{file_path}: Missing 'Scaling Parameters for Correlation Functions' section")
    
    # Check parameter ranges in table format
    for param_name, constraints in PARAMETER_CONSTRAINTS["core_parameters"].items():
        param_pattern = rf"`{param_name}`.*?\[({constraints['range'][0]}.*?{constraints['range'][1]})\]"
        if not re.search(param_pattern, content, re.DOTALL):
            issues.append(f"{file_path}: Parameter {param_name} has incorrect range or is missing")
    
    return issues


def check_config_constraints(file_path):
    """Check if configuration file has correct parameter constraints."""
    issues = []
    
    try:
        with open(file_path, 'r') as f:
            config = json.load(f)
    except Exception as e:
        return [f"Error reading {file_path}: {str(e)}"]
    
    # Check if parameter_space exists
    if "parameter_space" not in config or "bounds" not in config["parameter_space"]:
        return [f"{file_path}: Missing parameter_space.bounds section"]
    
    # Check static mode setting
    is_static = config.get("analysis_settings", {}).get("static_mode", False)
    
    # Check each parameter bound
    bounds = config["parameter_space"]["bounds"]
    param_dict = {param["name"]: param for param in bounds}
    
    for param_name, expected_constraints in PARAMETER_CONSTRAINTS["core_parameters"].items():
        if param_name not in param_dict:
            continue  # Parameter may not be present in all configs
        
        param_config = param_dict[param_name]
        
        # For fixed parameters in static mode, check reference bounds
        if is_static and param_config.get("type") == "fixed":
            # Check reference bounds exist
            if "_reference_min" not in param_config or "_reference_max" not in param_config:
                issues.append(f"{file_path}: Fixed parameter {param_name} missing reference bounds")
            else:
                # Check reference bounds match expected
                if param_config["_reference_min"] != expected_constraints["range"][0]:
                    issues.append(f"{file_path}: {param_name} _reference_min should be {expected_constraints['range'][0]}")
                if param_config["_reference_max"] != expected_constraints["range"][1]:
                    issues.append(f"{file_path}: {param_name} _reference_max should be {expected_constraints['range'][1]}")
        else:
            # Check regular bounds
            if param_config["min"] != expected_constraints["range"][0]:
                issues.append(f"{file_path}: {param_name} min should be {expected_constraints['range'][0]}")
            if param_config["max"] != expected_constraints["range"][1]:
                issues.append(f"{file_path}: {param_name} max should be {expected_constraints['range'][1]}")
            
            # Check prior parameters
            if "prior_mu" in param_config and param_config["prior_mu"] != expected_constraints["prior_mu"]:
                issues.append(f"{file_path}: {param_name} prior_mu should be {expected_constraints['prior_mu']}")
            if "prior_sigma" in param_config and param_config["prior_sigma"] != expected_constraints["prior_sigma"]:
                issues.append(f"{file_path}: {param_name} prior_sigma should be {expected_constraints['prior_sigma']}")
            
            # Check distribution type
            if "type" in param_config and param_config["type"] != expected_constraints["distribution"]:
                issues.append(f"{file_path}: {param_name} type should be {expected_constraints['distribution']}")
    
    # Check scaling parameters if present
    if "optimization_config" in config and "scaling_parameters" in config["optimization_config"]:
        scaling_params = config["optimization_config"]["scaling_parameters"]
        for param_name, expected_constraints in PARAMETER_CONSTRAINTS["scaling_parameters"].items():
            if param_name in scaling_params and "range" in expected_constraints:
                param_config = scaling_params[param_name]
                if param_config["min"] != expected_constraints["range"][0]:
                    issues.append(f"{file_path}: {param_name} min should be {expected_constraints['range'][0]}")
                if param_config["max"] != expected_constraints["range"][1]:
                    issues.append(f"{file_path}: {param_name} max should be {expected_constraints['range'][1]}")
    
    return issues


def find_all_files():
    """Find all relevant documentation and configuration files."""
    files = {
        "readme": [],
        "docs": [],
        "configs": []
    }
    
    # Find README files
    files["readme"].extend(glob.glob("README.md"))
    files["readme"].extend(glob.glob("*.md"))
    
    # Find documentation files
    files["docs"].extend(glob.glob("docs/**/*.rst", recursive=True))
    files["docs"].extend(glob.glob("docs/**/*.md", recursive=True))
    
    # Find configuration files
    files["configs"].extend(glob.glob("*.json"))
    files["configs"].extend(glob.glob("**/*.json", recursive=True))
    
    # Filter out backup files and other non-config JSONs
    files["configs"] = [f for f in files["configs"] if "backup" not in f and "package" not in f]
    
    return files


def main():
    """Main constraint checking function."""
    print("Checking parameter constraints across documentation and configuration files...")
    
    # Find all files
    files = find_all_files()
    
    all_issues = []
    
    # Check README files
    print("\nChecking README files...")
    for file_path in files["readme"]:
        if os.path.exists(file_path):
            issues = check_documentation_constraints(file_path)
            all_issues.extend(issues)
            if issues:
                print(f"  {file_path}: {len(issues)} issues")
            else:
                print(f"  {file_path}: OK")
    
    # Check documentation files (only key ones)
    print("\nChecking documentation files...")
    key_docs = [f for f in files["docs"] if "configuration.rst" in f or "user-guide" in f]
    for file_path in key_docs:
        if os.path.exists(file_path):
            issues = check_documentation_constraints(file_path)
            all_issues.extend(issues)
            if issues:
                print(f"  {file_path}: {len(issues)} issues")
            else:
                print(f"  {file_path}: OK")
    
    # Check configuration files
    print("\nChecking configuration files...")
    for file_path in files["configs"]:
        if os.path.exists(file_path):
            issues = check_config_constraints(file_path)
            all_issues.extend(issues)
            if issues:
                print(f"  {file_path}: {len(issues)} issues")
            else:
                print(f"  {file_path}: OK")
    
    # Summary
    print(f"\nTotal issues found: {len(all_issues)}")
    if all_issues:
        print("\nIssues:")
        for issue in all_issues:
            print(f"  - {issue}")
        return len(all_issues)
    else:
        print("All constraint checks passed!")
        return 0


if __name__ == "__main__":
    # Validate constraints first
    errors = validate_parameter_constraints()
    if errors:
        print("Parameter constraint validation errors:")
        for error in errors:
            print(f"  - {error}")
        exit(1)
    
    # Run constraint checking
    exit_code = main()
    exit(exit_code)
