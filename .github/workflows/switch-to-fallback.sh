#!/bin/bash

# Script to switch from main workflow to fallback workflow
# Run this if the main GitHub Actions workflow continues to fail

echo "🔄 Switching to fallback documentation workflow..."

# Check if files exist
if [ ! -f ".github/workflows/docs.yml" ]; then
    echo "❌ Main workflow file docs.yml not found"
    exit 1
fi

if [ ! -f ".github/workflows/docs-fallback.yml.disabled" ]; then
    echo "❌ Fallback workflow file not found"
    exit 1
fi

# Disable main workflow
echo "📝 Disabling main workflow..."
mv .github/workflows/docs.yml .github/workflows/docs.yml.disabled
echo "✅ Renamed docs.yml to docs.yml.disabled"

# Enable fallback workflow
echo "📝 Enabling fallback workflow..."
mv .github/workflows/docs-fallback.yml.disabled .github/workflows/docs-fallback.yml
echo "✅ Renamed docs-fallback.yml.disabled to docs-fallback.yml"

echo ""
echo "🎉 Fallback workflow is now active!"
echo ""
echo "The fallback workflow uses the traditional gh-pages approach which is more"
echo "forgiving of repository configuration issues. It will:"
echo "  - Build documentation the same way"
echo "  - Deploy to gh-pages branch instead of GitHub Pages Actions"
echo "  - Work without requiring GitHub Pages to be pre-configured"
echo ""
echo "Next steps:"
echo "1. Commit and push these changes"
echo "2. The workflow will trigger on the next push to main"
echo "3. Check the Actions tab to monitor the fallback deployment"
echo ""
echo "To switch back to the main workflow later:"
echo "  mv .github/workflows/docs.yml.disabled .github/workflows/docs.yml"
echo "  mv .github/workflows/docs-fallback.yml .github/workflows/docs-fallback.yml.disabled"