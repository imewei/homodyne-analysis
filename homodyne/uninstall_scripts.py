#!/usr/bin/env python3
"""
Homodyne Script Cleanup - Remove conda environment scripts
===========================================================

This script removes homodyne-related scripts from conda environment directories
that were installed during pip installation but are not tracked for removal
during pip uninstall.

Usage:
    python -m homodyne.uninstall_scripts
    homodyne-cleanup
"""

import os
import platform
import sys
from pathlib import Path


def is_virtual_environment():
    """Check if running in a virtual environment."""
    return (
        hasattr(sys, "real_prefix")
        or (hasattr(sys, "base_prefix") and sys.base_prefix != sys.prefix)
        or os.environ.get("CONDA_DEFAULT_ENV") is not None
    )


def cleanup_conda_scripts():
    """Remove conda environment scripts."""
    if not is_virtual_environment():
        print("⚠️  Not running in a virtual environment, skipping cleanup")
        return False

    if platform.system() != "Linux":
        print(f"ℹ️  Platform is {platform.system()}, no conda scripts to clean")
        return True

    try:
        conda_env_dir = Path(sys.prefix)

        # Define script paths to remove
        scripts_to_remove = [
            conda_env_dir / "etc" / "conda" / "activate.d" / "homodyne-gpu-activate.sh",
            conda_env_dir
            / "etc"
            / "conda"
            / "deactivate.d"
            / "homodyne-gpu-deactivate.sh",
            conda_env_dir / "etc" / "homodyne" / "gpu_activation.sh",
            conda_env_dir / "etc" / "homodyne" / "homodyne_completion_bypass.zsh",
            conda_env_dir / "etc" / "homodyne" / "homodyne_config.sh",
        ]

        print(f"🧹 Cleaning up Homodyne scripts in: {conda_env_dir}")
        print()

        # Remove scripts if they exist
        removed_count = 0
        for script_path in scripts_to_remove:
            if script_path.exists():
                script_path.unlink()
                print(f"✓ Removed: {script_path}")
                removed_count += 1
            else:
                print(f"ℹ️  Not found: {script_path}")

        # Remove empty directories
        homodyne_etc_dir = conda_env_dir / "etc" / "homodyne"
        if homodyne_etc_dir.exists() and not any(homodyne_etc_dir.iterdir()):
            homodyne_etc_dir.rmdir()
            print(f"✓ Removed empty directory: {homodyne_etc_dir}")
            removed_count += 1

        print()
        if removed_count > 0:
            print(f"✅ Successfully cleaned up {removed_count} files/directories")
            print(
                "🔄 Restart your shell or reactivate the conda environment to complete cleanup"
            )
        else:
            print("✅ No homodyne scripts found to remove")

        return True

    except Exception as e:
        print(f"❌ Failed to clean up conda scripts: {e}")
        return False


def main():
    """Main cleanup routine."""
    print("=" * 60)
    print("🧹 Homodyne Script Cleanup")
    print("=" * 60)

    try:
        success = cleanup_conda_scripts()
        if success:
            print("\n✅ Cleanup completed successfully")
        else:
            print("\n❌ Cleanup encountered issues")
            return 1

    except KeyboardInterrupt:
        print("\n⚠️  Cleanup cancelled by user")
        return 1
    except Exception as e:
        print(f"\n❌ Unexpected error during cleanup: {e}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
