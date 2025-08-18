#!/usr/bin/env python3
"""
GitHub Repository Setup Checker for GitHub Pages

This script helps verify that your repository is properly configured
for GitHub Pages deployment with GitHub Actions.
"""

import sys
import subprocess
import json
import urllib.request
import urllib.error


def run_command(cmd):
    """Run a shell command and return the output."""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        return result.stdout.strip(), result.returncode == 0
    except Exception as e:
        return str(e), False


def check_git_remote():
    """Check if the git remote is set correctly."""
    print("🔍 Checking git remote configuration...")
    
    output, success = run_command("git remote -v")
    if not success:
        print("❌ Not a git repository or git not available")
        return False
    
    lines = output.split('\n')
    remote_url = None
    for line in lines:
        if 'origin' in line and 'github.com' in line:
            # Extract URL - handle both SSH and HTTPS formats
            if 'git@github.com:' in line:
                remote_url = line.split('git@github.com:')[1].split()[0].replace('.git', '')
            elif 'https://github.com/' in line:
                remote_url = line.split('https://github.com/')[1].split()[0].replace('.git', '')
            break
    
    if remote_url:
        print(f"✅ Git remote found: {remote_url}")
        return remote_url
    else:
        print("❌ No GitHub remote found")
        return False


def check_repository_accessibility(repo_path):
    """Check if the repository is accessible via GitHub API."""
    print(f"🌐 Checking repository accessibility: {repo_path}")
    
    try:
        url = f"https://api.github.com/repos/{repo_path}"
        with urllib.request.urlopen(url) as response:
            data = json.loads(response.read())
            
        print(f"✅ Repository exists: {data['full_name']}")
        print(f"   Private: {'Yes' if data['private'] else 'No'}")
        print(f"   Has Pages: {'Yes' if data.get('has_pages', False) else 'No'}")
        print(f"   Default branch: {data['default_branch']}")
        
        return data
        
    except urllib.error.HTTPError as e:
        if e.code == 404:
            print("❌ Repository not found or not accessible")
        elif e.code == 403:
            print("❌ Access forbidden - check repository permissions")
        else:
            print(f"❌ HTTP Error {e.code}: {e.reason}")
        return False
    except Exception as e:
        print(f"❌ Error accessing repository: {e}")
        return False


def check_local_docs():
    """Check if documentation can be built locally."""
    print("📖 Checking local documentation build...")
    
    # Check if docs directory exists
    import os
    if not os.path.exists('docs'):
        print("❌ docs/ directory not found")
        return False
    
    # Check if Makefile or conf.py exists
    if not (os.path.exists('docs/Makefile') or os.path.exists('docs/conf.py')):
        print("❌ No Makefile or conf.py found in docs/")
        return False
    
    print("✅ Documentation structure found")
    
    # Try to build docs
    print("🔨 Attempting to build documentation...")
    output, success = run_command("cd docs && make html")
    
    if success and os.path.exists('docs/_build/html/index.html'):
        print("✅ Documentation builds successfully")
        return True
    else:
        print("❌ Documentation build failed")
        print("   Try running: cd docs && make html")
        return False


def check_workflow_file():
    """Check if GitHub Actions workflow file exists and is valid."""
    print("⚙️ Checking GitHub Actions workflow...")
    
    import os
    import yaml
    
    workflow_path = '.github/workflows/docs.yml'
    if not os.path.exists(workflow_path):
        print("❌ Workflow file not found: .github/workflows/docs.yml")
        return False
    
    try:
        with open(workflow_path, 'r') as f:
            workflow = yaml.safe_load(f)
        
        # Check key components
        has_permissions = 'permissions' in workflow
        has_pages_write = workflow.get('permissions', {}).get('pages') == 'write'
        has_deploy_job = any('deploy' in job_name.lower() for job_name in workflow.get('jobs', {}))
        
        print(f"✅ Workflow file exists")
        print(f"   Has permissions: {'Yes' if has_permissions else 'No'}")
        print(f"   Has pages:write: {'Yes' if has_pages_write else 'No'}")
        print(f"   Has deploy job: {'Yes' if has_deploy_job else 'No'}")
        
        return has_permissions and has_pages_write and has_deploy_job
        
    except Exception as e:
        print(f"❌ Error reading workflow file: {e}")
        return False


def main():
    """Main function to run all checks."""
    print("🚀 GitHub Pages Setup Checker")
    print("=" * 50)
    
    all_checks_passed = True
    
    # Check 1: Git remote
    repo_path = check_git_remote()
    if not repo_path:
        all_checks_passed = False
    
    print()
    
    # Check 2: Repository accessibility
    if repo_path:
        repo_data = check_repository_accessibility(repo_path)
        if not repo_data:
            all_checks_passed = False
    else:
        all_checks_passed = False
    
    print()
    
    # Check 3: Local docs build
    if not check_local_docs():
        all_checks_passed = False
    
    print()
    
    # Check 4: Workflow file
    if not check_workflow_file():
        all_checks_passed = False
    
    print()
    print("=" * 50)
    
    if all_checks_passed:
        print("🎉 All checks passed! Your repository should be ready for GitHub Pages.")
        print()
        print("Next steps:")
        print("1. Go to https://github.com/{}/settings/pages".format(repo_path))
        print("2. Set Source to 'GitHub Actions'")
        print("3. Push to main branch or run workflow manually")
    else:
        print("❌ Some checks failed. Please review the issues above.")
        print()
        print("Common solutions:")
        print("- Ensure repository is public or has Pages enabled")
        print("- Check repository permissions")
        print("- Verify workflow file permissions")
        print("- Test documentation build locally")
    
    return 0 if all_checks_passed else 1


if __name__ == "__main__":
    try:
        import yaml
    except ImportError:
        print("❌ PyYAML not installed. Install with: pip install PyYAML")
        sys.exit(1)
    
    sys.exit(main())
