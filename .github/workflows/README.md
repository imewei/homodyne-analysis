# GitHub Actions Workflows

This directory contains GitHub Actions workflows for the homodyne repository.

## 🚀 Active Workflows

### [`deploy-docs.yml`](./deploy-docs.yml) - Documentation Deployment
- **Purpose**: Deploy documentation to GitHub Pages
- **Trigger**: 
  - Push to `main` branch
  - Manual workflow dispatch
- **Method**: Uses `peaceiris/actions-gh-pages@v4` for reliable deployment
- **Output**: https://imewei.github.io/homodyne/
- **Features**:
  - Builds documentation with Sphinx
  - Deploys to `gh-pages` branch
  - Comprehensive build verification
  - Performance statistics

### [`docs.yml`](./docs.yml) - Documentation Testing
- **Purpose**: Test documentation builds on PRs and feature branches
- **Trigger**:
  - Pull requests to `main`
  - Push to `develop` or `feature/*` branches
  - Manual workflow dispatch
- **Method**: Build-only testing (no deployment)
- **Features**:
  - Validates documentation builds correctly
  - Uploads build artifacts for review
  - Fast feedback for contributors
  - Python 3.12+ compatibility testing

## 📋 Workflow Strategy

1. **Production Deployment**: `deploy-docs.yml` handles all main branch deployments
2. **Quality Assurance**: `docs.yml` validates changes before merging
3. **Single Responsibility**: Each workflow has a clear, focused purpose
4. **Reliability**: Uses proven peaceiris action for GitHub Pages deployment

## 🛠️ Setup Requirements

### GitHub Pages Configuration
1. Go to: https://github.com/imewei/homodyne/settings/pages
2. Under "Source": Select "Deploy from a branch"  
3. Branch: "gh-pages"
4. Path: "/ (root)"
5. Click "Save"

### Repository Requirements
- Repository must be public or have GitHub Pages enabled
- GitHub Actions must be enabled
- Python 3.12+ required
- Sphinx documentation dependencies in `pyproject.toml`

## 📖 Documentation Build Process

Both workflows use the standard documentation build process:

```bash
cd docs
make clean
make html
```

The build process:
1. Installs package with `[docs]` dependencies
2. Cleans previous builds
3. Generates HTML documentation
4. Verifies `index.html` exists
5. Provides build statistics

## 🔧 Troubleshooting

If documentation deployment fails:

1. **Check GitHub Pages Settings**:
   - Source should be "Deploy from a branch" → "gh-pages"
   - NOT "GitHub Actions"

2. **Verify Repository Status**:
   - Repository is public or has Pages enabled
   - No branch protection blocking deployments

3. **Check Workflow Logs**:
   - Look for build errors in the workflow runs
   - Verify all dependencies install correctly

4. **Manual Deployment**:
   - Use "Run workflow" button on `deploy-docs.yml`
   - Check Actions tab for detailed error messages

## 📊 Performance

- **Testing workflow** (`docs.yml`): ~2-3 minutes
- **Deployment workflow** (`deploy-docs.yml`): ~3-5 minutes  
- **GitHub Pages propagation**: 5-10 minutes after deployment

## 🎯 Best Practices

1. **Test First**: Always test documentation changes with PRs
2. **Clean Builds**: Workflows use `make clean` for consistency
3. **Artifact Storage**: Test builds are saved for 7 days
4. **Minimal Permissions**: Each workflow uses minimal required permissions
5. **Clear Naming**: Workflow names clearly indicate their purpose
