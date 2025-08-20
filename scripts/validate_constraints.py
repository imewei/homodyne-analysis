#!/usr/bin/env python3
"""
CI/CD Integration Script for Parameter Constraints Validation
=============================================================

This script validates parameter constraints consistency across the homodyne
package and can be integrated into continuous integration workflows.

Usage:
    python scripts/validate_constraints.py [--fix] [--report-only]

Options:
    --fix         Apply automatic fixes to detected constraint inconsistencies
    --report-only Generate validation report without failing on issues
    --verbose     Show detailed validation output

Exit Codes:
    0 - All constraints are consistent
    1 - Constraint validation errors found
    2 - Script execution errors

Authors: Wei Chen, Hongrui He
Institution: Argonne National Laboratory & University of Chicago
"""

import sys
import os
import argparse
from pathlib import Path

# Add project root to Python path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from check_constraints import main as check_main, validate_parameter_constraints
    from fix_constraints import main as fix_main
except ImportError as e:
    print(f"Error importing constraint validation modules: {e}")
    print("Make sure check_constraints.py and fix_constraints.py are in the project root.")
    sys.exit(2)


def validate_constraints_internal():
    """Validate internal consistency of parameter constraints definition."""
    print("Validating internal constraint consistency...")
    errors = validate_parameter_constraints()
    if errors:
        print("❌ Parameter constraint definition errors:")
        for error in errors:
            print(f"  - {error}")
        return False
    else:
        print("✅ Parameter constraint definitions are internally consistent")
        return True


def check_project_constraints(verbose=False):
    """Check constraint consistency across project files."""
    print("Checking parameter constraints across project files...")
    
    # Change to project root for consistent file discovery
    original_cwd = os.getcwd()
    os.chdir(PROJECT_ROOT)
    
    try:
        if verbose:
            # Run with full output
            issues_count = check_main()
        else:
            # Capture output for summary
            import subprocess
            result = subprocess.run([sys.executable, "check_constraints.py"], 
                                 capture_output=True, text=True)
            issues_count = result.returncode
            
            if issues_count == 0:
                print("✅ All parameter constraints are consistent")
            else:
                print(f"❌ Found {issues_count} constraint inconsistencies")
                if not verbose:
                    print("Run with --verbose for detailed output")
        
        return issues_count
    finally:
        os.chdir(original_cwd)


def fix_project_constraints():
    """Apply automatic fixes to constraint inconsistencies."""
    print("Applying automatic fixes to constraint inconsistencies...")
    
    # Change to project root
    original_cwd = os.getcwd()
    os.chdir(PROJECT_ROOT)
    
    try:
        changes = fix_main()
        if changes:
            print(f"✅ Applied {len(changes)} fixes successfully")
            print("Review the changes and backup files before committing")
        else:
            print("ℹ️  No fixes were needed")
        return True
    except Exception as e:
        print(f"❌ Error applying fixes: {e}")
        return False
    finally:
        os.chdir(original_cwd)


def main():
    """Main validation script entry point."""
    parser = argparse.ArgumentParser(
        description="Validate parameter constraints consistency across the homodyne package",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/validate_constraints.py                    # Check constraints only
    python scripts/validate_constraints.py --fix             # Check and apply fixes
    python scripts/validate_constraints.py --report-only     # Generate report without failing
    python scripts/validate_constraints.py --verbose         # Show detailed output
        """
    )
    
    parser.add_argument("--fix", action="store_true",
                       help="Apply automatic fixes to detected inconsistencies")
    parser.add_argument("--report-only", action="store_true",
                       help="Generate validation report without failing on issues")
    parser.add_argument("--verbose", action="store_true",
                       help="Show detailed validation output")
    
    args = parser.parse_args()
    
    print("🔍 Homodyne Parameter Constraints Validation")
    print("=" * 50)
    
    # Step 1: Validate internal constraint definitions
    if not validate_constraints_internal():
        print("\n❌ Validation failed due to internal constraint definition errors")
        return 2
    
    print()
    
    # Step 2: Check project-wide constraints
    issues_count = check_project_constraints(verbose=args.verbose)
    
    if issues_count == 0:
        print("\n✅ All parameter constraints validation passed!")
        return 0
    
    # Step 3: Handle issues
    if args.fix:
        print()
        if fix_project_constraints():
            print("\n✅ Constraint fixes applied successfully")
            print("Please review the changes and run validation again")
            return 0
        else:
            print("\n❌ Failed to apply constraint fixes")
            return 1
    elif args.report_only:
        print(f"\n⚠️  Found {issues_count} constraint issues (report-only mode)")
        return 0
    else:
        print(f"\n❌ Found {issues_count} constraint inconsistencies")
        print("Run with --fix to apply automatic fixes or --report-only to ignore failures")
        return 1


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n⚠️  Validation interrupted by user")
        sys.exit(130)  # Standard exit code for Ctrl+C
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(2)
