#!/bin/bash
#=============================================================================
# Standalone Homodyne Environment Cleanup Script
#=============================================================================
# Purpose: Remove homodyne-related scripts from conda environment
# Usage: bash standalone_cleanup.sh
# 
# This script works even after the homodyne package has been uninstalled.
# Use this if you forgot to run `homodyne-cleanup` before uninstalling.
#=============================================================================

echo "============================================================"
echo "🧹 Standalone Homodyne Environment Cleanup"
echo "============================================================"

# Check if we're in a virtual environment
if [[ -z "$CONDA_PREFIX" && -z "$VIRTUAL_ENV" ]]; then
    echo "⚠️  No virtual environment detected."
    echo "   Please activate your conda/virtual environment first:"
    echo "   conda activate <env-name>"
    echo "   # or"
    echo "   source activate <venv-path>/bin/activate"
    exit 1
fi

# Determine environment prefix
ENV_PREFIX=""
if [[ -n "$CONDA_PREFIX" ]]; then
    ENV_PREFIX="$CONDA_PREFIX"
    echo "📦 Conda environment: $(basename $CONDA_PREFIX)"
elif [[ -n "$VIRTUAL_ENV" ]]; then
    ENV_PREFIX="$VIRTUAL_ENV"
    echo "📦 Virtual environment: $(basename $VIRTUAL_ENV)"
fi

echo "🔍 Scanning for homodyne scripts in: $ENV_PREFIX"
echo ""

# Define script paths to remove
SCRIPTS_TO_REMOVE=(
    "$ENV_PREFIX/etc/conda/activate.d/homodyne-gpu-activate.sh"
    "$ENV_PREFIX/etc/conda/deactivate.d/homodyne-gpu-deactivate.sh"
    "$ENV_PREFIX/etc/homodyne/gpu_activation.sh"
    "$ENV_PREFIX/etc/homodyne/homodyne_aliases.sh"
    "$ENV_PREFIX/etc/homodyne/homodyne_completion_bypass.zsh"
    "$ENV_PREFIX/etc/homodyne/homodyne_config.sh"
)

# Remove scripts if they exist
REMOVED_COUNT=0
for script_path in "${SCRIPTS_TO_REMOVE[@]}"; do
    if [[ -f "$script_path" ]]; then
        rm -f "$script_path"
        echo "✓ Removed: $script_path"
        ((REMOVED_COUNT++))
    else
        echo "ℹ️  Not found: $script_path"
    fi
done

# Remove empty directories
HOMODYNE_ETC_DIR="$ENV_PREFIX/etc/homodyne"
if [[ -d "$HOMODYNE_ETC_DIR" ]] && [[ -z "$(ls -A "$HOMODYNE_ETC_DIR" 2>/dev/null)" ]]; then
    rmdir "$HOMODYNE_ETC_DIR"
    echo "✓ Removed empty directory: $HOMODYNE_ETC_DIR"
    ((REMOVED_COUNT++))
fi

echo ""
if [[ $REMOVED_COUNT -gt 0 ]]; then
    echo "✅ Successfully cleaned up $REMOVED_COUNT files/directories"
    echo "🔄 Restart your shell or reactivate the environment to complete cleanup"
else
    echo "✅ No homodyne scripts found to remove"
fi

echo ""
echo "============================================================"
echo "✨ Cleanup completed!"
echo "============================================================"