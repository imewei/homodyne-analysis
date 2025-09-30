#!/usr/bin/env python3
"""
Homodyne Advanced Completion System Uninstaller
===============================================

Command-line tool to safely remove the upgraded completion system
with backup restoration and cleanup verification.

Usage:
    python uninstall_completion.py [options]
    homodyne-uninstall-completion [options]

Examples:
    # Safe uninstall with confirmation
    python uninstall_completion.py

    # Force uninstall without confirmation
    python uninstall_completion.py --force

    # Show what would be removed without actually removing
    python uninstall_completion.py --dry-run

    # Uninstall with verbose output
    python uninstall_completion.py --verbose
"""

import argparse
import sys
from pathlib import Path

from .installer import CompletionInstaller
from .installer import InstallationConfig


def create_parser() -> argparse.ArgumentParser:
    """Create command line argument parser."""
    parser = argparse.ArgumentParser(
        prog="homodyne-uninstall-completion",
        description="Uninstall Homodyne Advanced Completion System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Uninstallation Process:
  1. Detect current installation
  2. Remove completion scripts and activation hooks
  3. Clean up cache and temporary files
  4. Remove environment-specific integration
  5. Verify complete removal

What Gets Removed:
  - Completion engine and plugin files
  - Shell-specific completion scripts
  - Activation scripts (conda/venv)
  - Cache files and directories
  - Command aliases and shortcuts
  - Configuration files

Safety Features:
  - Pre-uninstall validation
  - Dry-run mode for testing
  - Detailed removal reporting
  - Error handling and recovery

Examples:
  # Safe uninstall with confirmation
  homodyne-uninstall-completion

  # Quick uninstall without prompts
  homodyne-uninstall-completion --force --quiet

  # See what would be removed
  homodyne-uninstall-completion --dry-run --verbose

  # Thorough uninstall with verification
  homodyne-uninstall-completion --verify --verbose
        """,
    )

    # Uninstall behavior
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force uninstall without confirmation prompts",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be removed without actually removing",
    )

    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify complete removal after uninstall",
    )

    # Output control
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose output",
    )

    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Suppress all output except errors",
    )

    return parser


def validate_args(args: argparse.Namespace) -> bool:
    """Validate command line arguments."""
    if args.verbose and args.quiet:
        print("Error: --verbose and --quiet are mutually exclusive", file=sys.stderr)
        return False

    return True


def print_status(message: str, level: str = "info", quiet: bool = False) -> None:
    """Print status message with appropriate formatting."""
    if quiet and level != "error":
        return

    if level == "error":
        print(f"❌ {message}", file=sys.stderr)
    elif level == "warning":
        print(f"⚠️  {message}", file=sys.stderr)
    elif level == "success":
        print(f"✅ {message}")
    elif level == "info":
        print(f"i  {message}")
    else:
        print(message)


def show_installation_status(
    installer: CompletionInstaller, verbose: bool = False
) -> None:
    """Show current installation status."""
    info = installer.get_installation_info()

    print("🔍 Current Installation:")
    print(f"   Status: {'Installed' if info['installed'] else 'Not installed'}")

    if info["installed"]:
        print(f"   Environment: {info['environment_type']}")
        print(f"   Location: {info['install_base']}")

        if verbose and "version" in info:
            print(f"   Version: {info.get('version', 'unknown')}")
            features = info.get("features", {})
            if features:
                print("   Features:")
                for feature, enabled in features.items():
                    status = "enabled" if enabled else "disabled"
                    print(f"     - {feature}: {status}")

    elif verbose:
        print(f"   Environment: {info['environment_type']}")
        print(f"   Would install to: {info['install_base']}")


def confirm_uninstall(installer: CompletionInstaller, force: bool = False) -> bool:
    """Get user confirmation for uninstall."""
    if force:
        return True

    info = installer.get_installation_info()
    if not info["installed"]:
        return True

    print("\n⚠️  This will remove the Homodyne Advanced Completion System:")
    print("   • All completion scripts and activation hooks")
    print("   • Cached completion data")
    print("   • Command aliases (hma, hmc, hmr, etc.)")
    print("   • Environment-specific integration")

    print(f"\n📍 Installation location: {info['install_base']}")

    try:
        response = (
            input("\n❓ Are you sure you want to proceed? [y/N]: ").strip().lower()
        )
        return response in ["y", "yes"]
    except (KeyboardInterrupt, EOFError):
        print("\n⚠️  Uninstall cancelled by user")
        return False


def show_dry_run_info(installer: CompletionInstaller) -> None:
    """Show what would be removed in dry-run mode."""
    info = installer.get_installation_info()

    if not info["installed"]:
        print("🧪 Dry Run Result: Nothing to remove (not installed)")
        return

    print("🧪 Dry Run - Would remove:")
    print(f"   📁 Installation directory: {info['install_base']}")

    # Try to get a preview of what would be removed
    try:
        install_base = Path(info["install_base"])
        if install_base.exists():
            print("   📄 Files that would be removed:")

            # Count files by type
            py_files = list(install_base.rglob("*.py"))
            sh_files = list(install_base.rglob("*.sh"))
            other_files = list(install_base.rglob("*"))
            total_files = len([f for f in other_files if f.is_file()])

            if py_files:
                print(f"      • {len(py_files)} Python files")
            if sh_files:
                print(f"      • {len(sh_files)} Shell scripts")
            if total_files > len(py_files) + len(sh_files):
                print(
                    f"      • {total_files - len(py_files) - len(sh_files)} Other files"
                )

            print(f"   📊 Total: {total_files} files")

            # Show directory structure
            dirs = [d for d in install_base.rglob("*") if d.is_dir()]
            if dirs:
                print(f"   📁 Directories: {len(dirs)}")

    except Exception:
        print("   ⚠️  Could not preview files (permission or access issue)")

    # Check for activation scripts
    env_path = Path(info["environment_path"])
    activation_locations = [
        env_path / "etc" / "conda" / "activate.d",
        env_path / "bin",
        env_path / "Scripts",
    ]

    activation_scripts = []
    for location in activation_locations:
        if location.exists():
            scripts = list(location.glob("*homodyne-completion*"))
            scripts.extend(location.glob("*activate-homodyne*"))
            activation_scripts.extend(scripts)

    if activation_scripts:
        print(f"   🔗 Activation scripts: {len(activation_scripts)}")
        for script in activation_scripts:
            print(f"      • {script}")

    print("\n💡 Use --force to skip confirmation prompt")


def verify_removal(installer: CompletionInstaller, verbose: bool = False) -> bool:
    """Verify that the completion system was completely removed."""
    info = installer.get_installation_info()

    if info["installed"]:
        print_status(
            "Verification failed: System still appears to be installed", "error"
        )
        return False

    # Check for leftover files
    install_base = Path(info["install_base"])
    if install_base.exists():
        remaining_files = list(install_base.rglob("*"))
        if remaining_files:
            print_status(
                f"Warning: {len(remaining_files)} files remain in install directory",
                "warning",
            )
            if verbose:
                for file_path in remaining_files[:10]:  # Show first 10
                    print(f"   • {file_path}")
                if len(remaining_files) > 10:
                    print(f"   • ... and {len(remaining_files) - 10} more")
            return False

    # Check for activation scripts
    env_path = Path(info["environment_path"])
    activation_locations = [
        env_path / "etc" / "conda" / "activate.d",
        env_path / "bin",
        env_path / "Scripts",
    ]

    remaining_scripts = []
    for location in activation_locations:
        if location.exists():
            scripts = list(location.glob("*homodyne-completion*"))
            scripts.extend(location.glob("*activate-homodyne*"))
            remaining_scripts.extend(scripts)

    if remaining_scripts:
        print_status(
            f"Warning: {len(remaining_scripts)} activation scripts remain", "warning"
        )
        if verbose:
            for script in remaining_scripts:
                print(f"   • {script}")
        return False

    print_status("Verification successful: System completely removed", "success")
    return True


def main() -> int:
    """Main uninstallation routine."""
    parser = create_parser()
    args = parser.parse_args()

    # Validate arguments
    if not validate_args(args):
        return 1

    try:
        # Create installer for uninstall operations
        config = InstallationConfig()  # Default config for uninstall
        installer = CompletionInstaller(config)

        # Show current status
        if args.verbose:
            show_installation_status(installer, verbose=True)
        elif not args.quiet:
            show_installation_status(installer, verbose=False)

        # Handle dry run
        if args.dry_run:
            show_dry_run_info(installer)
            return 0

        # Check if actually installed
        info = installer.get_installation_info()
        if not info["installed"]:
            print_status("Completion system is not installed", "info", args.quiet)
            return 0

        # Get confirmation
        if not confirm_uninstall(installer, args.force):
            print_status("Uninstall cancelled", "info", args.quiet)
            return 0

        # Perform uninstallation
        if not args.quiet:
            print_status("Starting uninstallation...", "info", args.quiet)

        result = installer.uninstall()

        # Handle result
        if result.success:
            print_status(result.message, "success", args.quiet)

            if args.verbose and result.installed_files:
                print(f"\n🗑️  Removed {len(result.installed_files)} files:")
                for file_path in result.installed_files[:20]:  # Show first 20
                    print(f"   • {file_path}")
                if len(result.installed_files) > 20:
                    print(f"   • ... and {len(result.installed_files) - 20} more")

            # Verify removal if requested
            if args.verify:
                if not args.quiet:
                    print_status("Verifying complete removal...", "info", args.quiet)
                verify_removal(installer, args.verbose)

            if not args.quiet:
                print("\n✨ Uninstallation complete!")
                print("\n📋 Next steps:")
                print("   1. Restart your shell to clear any cached completions")
                print(
                    "   2. Check that 'homodyne <TAB>' no longer provides completions"
                )
                print(
                    "   3. Aliases (hma, hmc, hmr, etc.) should no longer be available"
                )

            return 0

        print_status(result.message, "error", args.quiet)

        if result.errors:
            print("\n💥 Errors encountered:")
            for error in result.errors:
                print_status(error, "error", args.quiet)

        if result.warnings and args.verbose:
            print("\n⚠️  Warnings:")
            for warning in result.warnings:
                print_status(warning, "warning", args.quiet)

        return 1

    except KeyboardInterrupt:
        print_status("Uninstall cancelled by user", "warning", args.quiet)
        return 1

    except Exception as e:
        print_status(f"Unexpected error: {e}", "error", args.quiet)
        if args.verbose:
            import traceback

            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
