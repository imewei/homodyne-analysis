# GitHub Pages Setup and Troubleshooting Guide

This guide helps resolve GitHub Pages deployment issues for the homodyne documentation.

## 🚀 Quick Setup Instructions

### Step 1: Enable GitHub Pages
1. Go to your repository: `https://github.com/imewei/homodyne`
2. Navigate to **Settings** → **Pages** (left sidebar)
3. Under **Source**, select **"GitHub Actions"** (NOT "Deploy from a branch")
4. Click **Save**

### Step 2: Verify Repository Settings
- Repository must be **public** OR have GitHub Pages enabled for private repos
- Ensure you have admin/write access to the repository
- Check that Actions are enabled: **Settings** → **Actions** → **General**

### Step 3: Run the Workflow
- Go to **Actions** tab in your repository
- Find "Build and Deploy Documentation" workflow
- Click **Run workflow** to trigger manually
- Or push to the `main` branch to trigger automatically

## 🔧 Common Issues and Solutions

### Issue 1: "Pages deployment failed" 
**Cause**: Repository not configured for GitHub Actions deployment

**Solution**:
```
Repository Settings → Pages → Source → Select "GitHub Actions"
```

### Issue 2: "Permission denied" or "Forbidden"
**Cause**: Insufficient permissions or wrong repository settings

**Solutions**:
- Ensure you're the repository owner or have admin access
- Check if organization policies restrict GitHub Pages
- Verify workflow permissions in `.github/workflows/docs.yml`

### Issue 3: "Repository not found" or "404 errors"
**Cause**: Repository privacy settings or wrong URLs

**Solutions**:
- Make repository public, or enable Pages for private repos (requires Pro/Team plan)
- Verify repository name: `imewei/homodyne`
- Check all URLs in `pyproject.toml` point to correct repository

### Issue 4: Build succeeds but deployment fails
**Cause**: GitHub Pages service issues or configuration problems

**Solutions**:
1. Try the simple workflow: `.github/workflows/docs-simple.yml`
2. Run manually from Actions tab
3. Check GitHub Status page for Pages service issues
4. Verify artifact upload completed successfully

### Issue 5: "Branch protection" errors
**Cause**: Branch protection rules blocking Actions

**Solutions**:
- Go to **Settings** → **Branches**
- Edit protection rules for `main` branch
- Ensure "Restrict pushes that create files" allows Actions
- Add Actions bot to bypass restrictions if needed

## 🛠️ Advanced Troubleshooting

### Check Workflow Permissions
Ensure `.github/workflows/docs.yml` has proper permissions:
```yaml
permissions:
  contents: read
  pages: write
  id-token: write
  actions: read
```

### Verify Environment Setup
The workflow uses:
- **Python 3.12** (updated from 3.8)
- **Sphinx** documentation generator
- **GitHub Actions** for deployment (not branch-based)

### Manual Deployment Test
If automated deployment fails, test locally:
```bash
cd docs
make clean
make html
# Check that _build/html/index.html exists
```

### Alternative Deployment Methods

#### Option 1: Use Simple Workflow
Enable the simple workflow for basic deployment:
- Manually run `.github/workflows/docs-simple.yml`
- This uses minimal dependencies and direct Sphinx build

#### Option 2: Branch-Based Deployment (Fallback)
If Actions deployment continues failing:
1. **Settings** → **Pages** → **Source** → **"Deploy from a branch"**
2. Select **gh-pages** branch
3. Use actions to build and push to gh-pages branch

## 📋 Verification Checklist

- [ ] Repository is public or has Pages enabled for private repos
- [ ] **Settings** → **Pages** → **Source** = "GitHub Actions"  
- [ ] **Settings** → **Actions** → **General** = Actions enabled
- [ ] Workflow has proper permissions (contents: read, pages: write, id-token: write)
- [ ] No branch protection rules blocking deployments
- [ ] Documentation builds locally (`make html` in docs/ directory)
- [ ] Repository owner has GitHub Pages feature available

## 🌐 Expected Results

When working correctly:
- Workflow runs on pushes to `main` branch
- Documentation builds successfully
- Pages deploys to: `https://imewei.github.io/homodyne/`
- Deployment status shows "✅ Active" in repository Pages settings

## 📞 Getting Help

If issues persist:

1. **Check GitHub Status**: https://www.githubstatus.com/
2. **Review Workflow Logs**: Actions tab → Failed workflow → Click on failed step
3. **GitHub Community**: https://github.community/
4. **Repository Issues**: https://github.com/imewei/homodyne/issues

## 🔄 Alternative Workflows Available

- **Main workflow**: `.github/workflows/docs.yml` (comprehensive with error handling)
- **Simple workflow**: `.github/workflows/docs-simple.yml` (minimal, manual trigger only)
- **Fallback**: Branch-based deployment to `gh-pages` branch

Choose the workflow that best fits your needs and repository setup.
