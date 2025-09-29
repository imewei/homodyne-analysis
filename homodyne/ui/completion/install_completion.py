#!/usr/bin/env python3
"""
Homodyne Advanced Completion System Installer
=============================================

Command-line tool to install the upgraded completion system with
virtual environment isolation, intelligent caching, and plugin support.

Usage:
    python install_completion.py [options]
    homodyne-install-completion [options]

Examples:
    # Install with default settings (advanced mode)
    python install_completion.py

    # Install for specific shells
    python install_completion.py --shell bash --shell zsh

    # Install in simple mode without caching
    python install_completion.py --mode simple --no-cache

    # Force reinstall
    python install_completion.py --force

    # Development mode with debugging
    python install_completion.py --mode development --verbose
"""

import argparse
import sys

from .installer import CompletionInstaller
from .installer import InstallationConfig
from .installer import InstallationMode
from .installer import ShellType


def create_parser() -> argparse.ArgumentParser:
    """Create command line argument parser."""
    parser = argparse.ArgumentParser(
        prog="homodyne-install-completion",
        description="Install Homodyne Advanced Completion System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Installation Modes:
  simple      - Basic completion only, minimal resource usage
  advanced    - Full completion with caching and smart features (default)
  development - Development mode with debugging and verbose output

Shell Support:
  auto        - Detect and install for available shells (default)
  bash        - Install for bash shell
  zsh         - Install for zsh shell
  fish        - Install for fish shell

Environment Detection:
  The installer automatically detects your virtual environment type:
  - Conda/Mamba environments
  - Python venv/virtualenv
  - Poetry environments
  - Pipenv environments

Examples:
  # Quick install with auto-detection
  homodyne-install-completion

  # Install for specific shells with caching disabled
  homodyne-install-completion --shell bash --shell zsh --no-cache

  # Development installation with verbose output
  homodyne-install-completion --mode development --verbose --enable-debug

  # Force reinstall existing installation
  homodyne-install-completion --force --backup

Performance Options:
  --cache-size MB     Set cache size limit (default: 50MB)
  --timeout MS        Set completion timeout (default: 1000ms)
  --no-background     Disable background cache warming

Advanced Features:
  --enable-project    Enable project-aware completions (default)
  --enable-smart      Enable smart context-aware completions (default)
  --enable-aliases    Enable command aliases (hmv, hmm, etc.) (default)
        """,
    )

    # Installation mode
    parser.add_argument(
        "--mode",
        choices=["simple", "advanced", "development"],
        default="advanced",
        help="Installation mode (default: advanced)",
    )

    # Shell selection
    parser.add_argument(
        "--shell",
        action="append",
        choices=["auto", "bash", "zsh", "fish"],
        help="Target shell(s) for completion (can be specified multiple times)",
    )

    # Feature flags
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable completion caching",
    )

    parser.add_argument(
        "--no-aliases",
        action="store_true",
        help="Disable command aliases (hmv, hmm, etc.)",
    )

    parser.add_argument(
        "--no-project",
        action="store_true",
        help="Disable project-aware completions",
    )

    parser.add_argument(
        "--no-smart",
        action="store_true",
        help="Disable smart context-aware completions",
    )

    parser.add_argument(
        "--no-background",
        action="store_true",
        help="Disable background cache warming",
    )

    # Performance settings
    parser.add_argument(
        "--cache-size",
        type=int,
        default=50,
        metavar="MB",
        help="Maximum cache size in MB (default: 50)",
    )

    parser.add_argument(
        "--timeout",
        type=int,
        default=1000,
        metavar="MS",
        help="Completion timeout in milliseconds (default: 1000)",
    )

    # Installation behavior
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force reinstall if already installed",
    )

    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Don't backup existing completion files",
    )

    parser.add_argument(
        "--no-atomic",
        action="store_true",
        help="Disable atomic installation (less safe but faster)",
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

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be installed without actually installing",
    )

    return parser


def validate_args(args: argparse.Namespace) -> bool:
    """Validate command line arguments."""
    if args.verbose and args.quiet:
        print("Error: --verbose and --quiet are mutually exclusive", file=sys.stderr)
        return False

    if args.cache_size < 1 or args.cache_size > 1000:
        print("Error: Cache size must be between 1 and 1000 MB", file=sys.stderr)
        return False

    if args.timeout < 100 or args.timeout > 10000:
        print("Error: Timeout must be between 100 and 10000 ms", file=sys.stderr)
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


def show_installation_info(
    installer: CompletionInstaller, verbose: bool = False
) -> None:
    """Show information about the installation environment."""
    info = installer.get_installation_info()

    print("🔍 Installation Environment:")
    print(f"   Environment Type: {info['environment_type']}")
    print(f"   Environment Path: {info['environment_path']}")
    print(f"   Detected Shells: {', '.join(info['detected_shells'])}")

    if verbose:
        print(f"   Install Base: {info['install_base']}")
        if info["installed"]:
            print("   Current Status: Already installed")
        else:
            print("   Current Status: Not installed")


def show_dry_run_info(
    config: InstallationConfig, installer: CompletionInstaller
) -> None:
    """Show what would be installed in dry-run mode."""
    print("🧪 Dry Run - Would install:")
    print(f"   Mode: {config.mode.value}")
    print(f"   Shells: {[shell.value for shell in config.shells]}")
    print("   Features:")
    print(f"     - Caching: {config.enable_caching}")
    print(f"     - Aliases: {config.enable_aliases}")
    print(f"     - Project Detection: {config.enable_project_detection}")
    print(f"     - Smart Completion: {config.enable_smart_completion}")
    print(f"     - Background Warming: {config.enable_background_warming}")
    print("   Performance:")
    print(f"     - Cache Size: {config.cache_size_mb}MB")
    print(f"     - Timeout: {config.completion_timeout_ms}ms")
    print("   Installation:")
    print(f"     - Force Install: {config.force_install}")
    print(f"     - Backup Existing: {config.backup_existing}")
    print(f"     - Atomic Install: {config.atomic_install}")


def main() -> int:
    """Main installation routine."""
    parser = create_parser()
    args = parser.parse_args()

    # Validate arguments
    if not validate_args(args):
        return 1

    try:
        # Create installation configuration
        config = InstallationConfig(
            mode=InstallationMode(args.mode),
            shells=[ShellType(shell) for shell in (args.shell or ["auto"])],
            enable_aliases=not args.no_aliases,
            enable_caching=not args.no_cache,
            force_install=args.force,
            backup_existing=not args.no_backup,
            atomic_install=not args.no_atomic,
            enable_project_detection=not args.no_project,
            enable_smart_completion=not args.no_smart,
            enable_background_warming=not args.no_background,
            cache_size_mb=args.cache_size,
            completion_timeout_ms=args.timeout,
        )

        # Create installer
        installer = CompletionInstaller(config)

        # Show environment info
        if args.verbose:
            show_installation_info(installer, verbose=True)
        elif not args.quiet:
            show_installation_info(installer, verbose=False)

        # Handle dry run
        if args.dry_run:
            show_dry_run_info(config, installer)
            return 0

        # Check if already installed
        if installer.is_installed() and not config.force_install:
            print_status(
                "Completion system is already installed. Use --force to reinstall.",
                "warning",
                args.quiet,
            )
            return 0

        # Perform installation
        if not args.quiet:
            print_status("Starting installation...", "info", args.quiet)

        result = installer.install()

        # Handle result
        if result.success:
            print_status(result.message, "success", args.quiet)

            if args.verbose and result.installed_files:
                print("\n📁 Installed files:")
                for file_path in result.installed_files:
                    print(f"   {file_path}")

            if result.backup_files and not args.quiet:
                print(f"\n💾 Backed up {len(result.backup_files)} existing files")

            if not args.quiet:
                print("\n🚀 Installation complete!")
                print("\n📋 Next steps:")
                print("   1. Restart your shell or run: source ~/.bashrc (or ~/.zshrc)")
                print("   2. Try: homodyne <TAB> to test completion")
                print("   3. Use aliases: hmv, hmm, hmh for quick analysis")

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
        print_status("Installation cancelled by user", "warning", args.quiet)
        return 1

    except Exception as e:
        print_status(f"Unexpected error: {e}", "error", args.quiet)
        if args.verbose:
            import traceback

            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
