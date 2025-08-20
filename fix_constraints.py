"""
Parameter Constraints Synchronization and Fixing Tools
======================================================

Tools for automatically fixing parameter constraints in documentation and 
configuration files to match the authoritative constraints defined in 
check_constraints.py.

Authors: Wei Chen, Hongrui He
Institution: Argonne National Laboratory & University of Chicago
"""

import json
import shutil
import os
import re
from datetime import datetime
from pathlib import Path
from check_constraints import PARAMETER_CONSTRAINTS


def generate_parameter_table_markdown():
    """Generate markdown table for parameter constraints documentation."""
    markdown_lines = []
    
    # Core parameters table
    markdown_lines.append("### Core Model Parameters")
    markdown_lines.append("")
    markdown_lines.append("| Parameter | Range | Distribution | Physical Constraint |")
    markdown_lines.append("|-----------|-------|--------------|-------------------|")
    
    for param_name, constraints in PARAMETER_CONSTRAINTS["core_parameters"].items():
        range_str = f"[{constraints['range'][0]}, {constraints['range'][1]}] {constraints['unit']}"
        dist_str = f"{constraints['distribution']}(μ={constraints['prior_mu']}, σ={constraints['prior_sigma']})"
        physical_str = constraints['physical_constraint']
        
        markdown_lines.append(f"| `{param_name}` | {range_str} | {dist_str} | {physical_str} |")
    
    markdown_lines.append("")
    
    # Physical function constraints section
    markdown_lines.append("### Physical Function Constraints")
    markdown_lines.append("")
    markdown_lines.append("The package automatically enforces positivity for time-dependent functions:")
    markdown_lines.append("")
    
    for func_name, func_info in PARAMETER_CONSTRAINTS["physical_functions"].items():
        markdown_lines.append(f"- **{func_info['purpose']}**: {func_info['formula']}")
        markdown_lines.append(f"  - Constraint: {func_info['constraint']}")
    
    markdown_lines.append("")
    
    # Scaling parameters section
    markdown_lines.append("### Scaling Parameters for Correlation Functions")
    markdown_lines.append("")
    markdown_lines.append("The relationship **c2_fitted = c2_theory × contrast + offset** uses bounded parameters:")
    markdown_lines.append("")
    
    scaling_params = PARAMETER_CONSTRAINTS["scaling_parameters"]
    for param_name in ["contrast", "offset"]:
        if param_name in scaling_params:
            constraints = scaling_params[param_name]
            if "range" in constraints:
                range_str = f"[{constraints['range'][0]}, {constraints['range'][1]}]"
                dist_str = f"{constraints['distribution']}(μ={constraints['prior_mu']}, σ={constraints['prior_sigma']})"
                markdown_lines.append(f"- `{param_name}`: {range_str} {dist_str}")
    
    markdown_lines.append("")
    
    return "\n".join(markdown_lines)


def backup_file(file_path):
    """Create a backup of the file before modifying."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = f"{file_path}.backup_{timestamp}"
    shutil.copy2(file_path, backup_path)
    return backup_path


def fix_readme_constraints(file_path):
    """Fix parameter constraints in README file."""
    changes = []
    
    try:
        with open(file_path, 'r') as f:
            content = f.read()
    except Exception as e:
        return [f"Error reading {file_path}: {str(e)}"]
    
    # Create backup
    backup_path = backup_file(file_path)
    changes.append(f"Backed up {file_path} to {backup_path}")
    
    # Generate new parameter table
    new_table = generate_parameter_table_markdown()
    
    # Find and replace existing parameter constraints section
    # Look for the section starting with "## Physical Constraints and Parameter Ranges"
    pattern = r"(## Physical Constraints and Parameter Ranges.*?)(?=\n## |\Z)"
    
    if re.search(pattern, content, re.DOTALL):
        # Replace existing section
        replacement = f"## Physical Constraints and Parameter Ranges\n\n{new_table}"
        content = re.sub(pattern, replacement, content, flags=re.DOTALL)
        changes.append(f"Updated parameter constraints section in {file_path}")
    else:
        # Add new section if not found
        if "## Physical Constraints and Parameter Ranges" not in content:
            # Add before the first ## section or at the end
            first_section_match = re.search(r"\n## ", content)
            if first_section_match:
                insertion_point = first_section_match.start()
                new_section = f"\n## Physical Constraints and Parameter Ranges\n\n{new_table}\n"
                content = content[:insertion_point] + new_section + content[insertion_point:]
            else:
                content += f"\n\n## Physical Constraints and Parameter Ranges\n\n{new_table}"
            changes.append(f"Added parameter constraints section to {file_path}")
    
    # Write updated content
    try:
        with open(file_path, 'w') as f:
            f.write(content)
        changes.append(f"Successfully wrote updated {file_path}")
    except Exception as e:
        changes.append(f"Error writing {file_path}: {str(e)}")
    
    return changes


def fix_config_constraints(file_path):
    """Fix parameter constraints in configuration file."""
    changes = []
    
    try:
        with open(file_path, 'r') as f:
            config = json.load(f)
    except Exception as e:
        return [f"Error reading {file_path}: {str(e)}"]
    
    # Create backup
    backup_path = backup_file(file_path)
    changes.append(f"Backed up {file_path} to {backup_path}")
    
    # Check if parameter_space exists
    if "parameter_space" not in config or "bounds" not in config["parameter_space"]:
        return changes + [f"Skipping {file_path}: Missing parameter_space.bounds section"]
    
    # Check static mode setting
    is_static = config.get("analysis_settings", {}).get("static_mode", False)
    
    # Fix each parameter bound
    bounds = config["parameter_space"]["bounds"]
    param_dict = {param["name"]: i for i, param in enumerate(bounds)}
    
    for param_name, expected_constraints in PARAMETER_CONSTRAINTS["core_parameters"].items():
        if param_name not in param_dict:
            continue  # Parameter not present in this config
        
        param_idx = param_dict[param_name]
        param_config = bounds[param_idx]
        
        # For fixed parameters in static mode, add reference bounds
        if is_static and param_config.get("type") == "fixed":
            # Add reference bounds if missing
            if "_reference_min" not in param_config:
                param_config["_reference_min"] = expected_constraints["range"][0]
                changes.append(f"Added reference bounds for fixed parameter {param_name} in {file_path}")
            if "_reference_max" not in param_config:
                param_config["_reference_max"] = expected_constraints["range"][1]
            if "_reference_type" not in param_config:
                param_config["_reference_type"] = expected_constraints["distribution"]
            if "_reference_prior_mu" not in param_config:
                param_config["_reference_prior_mu"] = expected_constraints["prior_mu"]
            if "_reference_prior_sigma" not in param_config:
                param_config["_reference_prior_sigma"] = expected_constraints["prior_sigma"]
            
            # Fix reference bounds if incorrect
            if param_config.get("_reference_min") != expected_constraints["range"][0]:
                param_config["_reference_min"] = expected_constraints["range"][0]
                changes.append(f"Fixed {param_name} _reference_min in {file_path}")
            if param_config.get("_reference_max") != expected_constraints["range"][1]:
                param_config["_reference_max"] = expected_constraints["range"][1]
                changes.append(f"Fixed {param_name} _reference_max in {file_path}")
        else:
            # Fix regular bounds
            if param_config["min"] != expected_constraints["range"][0]:
                param_config["min"] = expected_constraints["range"][0]
                changes.append(f"Updated {param_name} min bound in {file_path}")
            if param_config["max"] != expected_constraints["range"][1]:
                param_config["max"] = expected_constraints["range"][1]
                changes.append(f"Updated {param_name} max bound in {file_path}")
            
            # Fix prior parameters
            if "prior_mu" in param_config and param_config["prior_mu"] != expected_constraints["prior_mu"]:
                param_config["prior_mu"] = expected_constraints["prior_mu"]
                changes.append(f"Updated {param_name} prior_mu in {file_path}")
            if "prior_sigma" in param_config and param_config["prior_sigma"] != expected_constraints["prior_sigma"]:
                param_config["prior_sigma"] = expected_constraints["prior_sigma"]
                changes.append(f"Updated {param_name} prior_sigma in {file_path}")
            
            # Fix distribution type
            if "type" in param_config and param_config["type"] != expected_constraints["distribution"]:
                param_config["type"] = expected_constraints["distribution"]
                changes.append(f"Updated {param_name} distribution type in {file_path}")
    
    # Fix scaling parameters if present
    if "optimization_config" in config and "scaling_parameters" in config["optimization_config"]:
        scaling_params = config["optimization_config"]["scaling_parameters"]
        for param_name, expected_constraints in PARAMETER_CONSTRAINTS["scaling_parameters"].items():
            if param_name in scaling_params and "range" in expected_constraints:
                param_config = scaling_params[param_name]
                if param_config["min"] != expected_constraints["range"][0]:
                    param_config["min"] = expected_constraints["range"][0]
                    changes.append(f"Updated {param_name} min in {file_path}")
                if param_config["max"] != expected_constraints["range"][1]:
                    param_config["max"] = expected_constraints["range"][1]
                    changes.append(f"Updated {param_name} max in {file_path}")
                if "prior_mu" in param_config and param_config["prior_mu"] != expected_constraints["prior_mu"]:
                    param_config["prior_mu"] = expected_constraints["prior_mu"]
                    changes.append(f"Updated {param_name} prior_mu in {file_path}")
                if "prior_sigma" in param_config and param_config["prior_sigma"] != expected_constraints["prior_sigma"]:
                    param_config["prior_sigma"] = expected_constraints["prior_sigma"]
                    changes.append(f"Updated {param_name} prior_sigma in {file_path}")
                if "type" in param_config and param_config["type"] != expected_constraints["distribution"]:
                    param_config["type"] = expected_constraints["distribution"]
                    changes.append(f"Updated {param_name} distribution type in {file_path}")
    
    # Write updated config
    try:
        with open(file_path, 'w') as f:
            json.dump(config, f, indent=2)
        changes.append(f"Successfully wrote updated {file_path}")
    except Exception as e:
        changes.append(f"Error writing {file_path}: {str(e)}")
    
    return changes


def create_gap_analysis_report(all_changes):
    """Create a gap analysis report of all changes made."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    report_lines = [
        "# Parameter Constraints Synchronization Report",
        "",
        f"**Generated:** {timestamp}",
        "",
        "## Overview",
        "",
        "This report documents the synchronization of parameter constraints and ranges across the homodyne package to ensure consistency between documentation and configuration files.",
        "",
        "## Authoritative Parameter Constraints",
        "",
        "The following parameter constraints were established as the single source of truth:",
        "",
        "### Core Model Parameters",
        ""
    ]
    
    # Add parameter list
    for param_name, constraints in PARAMETER_CONSTRAINTS["core_parameters"].items():
        range_str = f"[{constraints['range'][0]}, {constraints['range'][1]}] {constraints['unit']}"
        dist_str = f"({constraints['distribution']})"
        report_lines.append(f"- **{param_name}**: {range_str} {dist_str}")
    
    report_lines.extend([
        "",
        "### Scaling Parameters",
        ""
    ])
    
    for param_name, constraints in PARAMETER_CONSTRAINTS["scaling_parameters"].items():
        if "range" in constraints:
            range_str = f"[{constraints['range'][0]}, {constraints['range'][1]}]"
            dist_str = f"({constraints['distribution']})" if "distribution" in constraints else ""
            report_lines.append(f"- **{param_name}**: {range_str} {dist_str}")
    
    report_lines.extend([
        "",
        "",
        "## Changes Made",
        "",
        f"Total changes: {len(all_changes)}",
        ""
    ])
    
    # Group changes by file
    changes_by_file = {}
    for change in all_changes:
        # Try to extract file name
        if " in " in change:
            file_part = change.split(" in ")[-1]
        elif change.startswith("Backed up "):
            file_part = change.split(" ")[2]
        else:
            file_part = "general"
        
        if file_part not in changes_by_file:
            changes_by_file[file_part] = []
        changes_by_file[file_part].append(change)
    
    for file_name, file_changes in changes_by_file.items():
        report_lines.append(f"### {file_name}")
        report_lines.append("")
        for change in file_changes:
            report_lines.append(f"- {change}")
        report_lines.append("")
    
    report_lines.extend([
        "",
        "## Validation",
        "",
        "After applying these changes, all parameter constraints should be consistent across:",
        "",
        "- Main README.md documentation",
        "- All configuration template files",
        "- All configuration example files",
        "",
        "## Next Steps",
        "",
        "1. Review the backup files created during this process",
        "2. Test the updated configurations",
        "3. Commit the changes to version control",
        "4. Update the changelog",
        "",
        "## Files Modified",
        ""
    ])
    
    modified_files = set()
    for change in all_changes:
        if "Successfully wrote updated" in change:
            file_name = change.split("Successfully wrote updated ")[-1]
            modified_files.add(file_name)
    
    for file_name in sorted(modified_files):
        report_lines.append(f"- {file_name}")
    
    return "\n".join(report_lines)


def main():
    """Main function to fix all constraint issues."""
    from check_constraints import find_all_files
    
    print("Fixing parameter constraint issues...")
    
    # Find all files
    files = find_all_files()
    
    all_changes = []
    
    # Fix README files
    print("\nFixing README files...")
    for file_path in files["readme"]:
        if os.path.exists(file_path) and file_path.endswith(".md"):
            changes = fix_readme_constraints(file_path)
            all_changes.extend(changes)
            if changes:
                print(f"  {file_path}: {len(changes)} changes")
            else:
                print(f"  {file_path}: No changes needed")
    
    # Fix configuration files
    print("\nFixing configuration files...")
    for file_path in files["configs"]:
        if os.path.exists(file_path):
            changes = fix_config_constraints(file_path)
            all_changes.extend(changes)
            if changes:
                print(f"  {file_path}: {len(changes)} changes")
            else:
                print(f"  {file_path}: No changes needed")
    
    # Generate gap analysis report
    print("\nGenerating gap analysis report...")
    report_content = create_gap_analysis_report(all_changes)
    
    os.makedirs("docs/dev", exist_ok=True)
    report_path = "docs/dev/constraint_sync_report.md"
    with open(report_path, 'w') as f:
        f.write(report_content)
    
    print(f"Gap analysis report saved to: {report_path}")
    print(f"\nTotal changes made: {len(all_changes)}")
    
    return all_changes


if __name__ == "__main__":
    changes = main()
    print(f"Constraint fixing completed with {len(changes)} total changes.")
