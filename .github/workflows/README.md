# GitHub Actions Workflows

This directory contains GitHub Actions workflows for the homodyne repository.

## 🚀 Active Workflows

### [`deploy-docs.yml`](./deploy-docs.yml) - Documentation Build Testing
- **Purpose**: Test documentation builds to ensure ReadTheDocs deployment will succeed
- **Trigger**:
  - Push to `main` branch
  - Pull requests to `main`
  - Manual workflow dispatch
- **Method**: Build-only testing (no deployment - handled by ReadTheDocs)
- **Output**: Documentation available at https://homodyne.readthedocs.io/
- **Features**:
  - Builds documentation with Sphinx
  - Comprehensive build verification
  - Performance statistics
  - Validates compatibility before ReadTheDocs deployment

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

1. **ReadTheDocs Deployment**: Documentation is automatically deployed via ReadTheDocs on push to `main`
2. **Build Validation**: `deploy-docs.yml` validates builds will succeed before ReadTheDocs attempts deployment
3. **Quality Assurance**: `docs.yml` validates changes before merging
4. **Single Responsibility**: Each workflow has a clear, focused purpose

## 🛠️ Setup Requirements

### ReadTheDocs Configuration
1. Documentation is automatically built and deployed via ReadTheDocs
2. Configuration file: `.readthedocs.yaml`
3. Builds triggered automatically on push to `main` branch
4. Live documentation: https://homodyne.readthedocs.io/

### Repository Requirements
- GitHub Actions must be enabled
- Python 3.12+ required
- Sphinx documentation dependencies in `pyproject.toml`
- ReadTheDocs webhook configured (automatic)

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

1. **Check ReadTheDocs Build Logs**:
   - Visit https://readthedocs.org/projects/homodyne/builds/
   - Look for build errors and warnings
   - Verify all dependencies are correctly specified

2. **Verify Repository Status**:
   - Ensure `.readthedocs.yaml` configuration is correct
   - Check that all required files are committed to `main`
   - Verify webhook is properly configured

3. **Check GitHub Actions Logs**:
   - Look for build errors in the workflow runs
   - Verify all dependencies install correctly
   - Use workflow to validate builds before ReadTheDocs attempts

4. **Manual Build Testing**:
   - Use "Run workflow" button on `deploy-docs.yml` to test builds
   - Check Actions tab for detailed error messages

## 📊 Performance

- **Testing workflow** (`docs.yml`): ~2-3 minutes
- **Build validation** (`deploy-docs.yml`): ~3-5 minutes  
- **ReadTheDocs deployment**: ~5-10 minutes after push to main

## 🎯 Best Practices

1. **Test First**: Always test documentation changes with PRs
2. **Clean Builds**: Workflows use `make clean` for consistency
3. **Artifact Storage**: Test builds are saved for 7 days
4. **Minimal Permissions**: Each workflow uses minimal required permissions
5. **Clear Naming**: Workflow names clearly indicate their purpose
