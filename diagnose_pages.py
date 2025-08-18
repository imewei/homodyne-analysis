#!/usr/bin/env python3
"""
GitHub Pages Deployment Diagnostic Tool

This script diagnoses common GitHub Pages deployment issues and provides
specific fixes for the homodyne repository.
"""

import sys
import subprocess
import json
import urllib.request
import urllib.error
import os


def run_command(cmd):
    """Run a shell command and return the output."""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        return result.stdout.strip(), result.stderr.strip(), result.returncode == 0
    except Exception as e:
        return "", str(e), False


def check_github_api(endpoint, description):
    """Check a GitHub API endpoint."""
    print(f"🌐 Checking {description}...")
    
    try:
        with urllib.request.urlopen(endpoint) as response:
            data = json.loads(response.read())
        return data, True
    except urllib.error.HTTPError as e:
        print(f"   ❌ HTTP {e.code}: {e.reason}")
        return None, False
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return None, False


def diagnose_repository():
    """Diagnose repository configuration."""
    print("🔍 REPOSITORY DIAGNOSIS")
    print("=" * 50)
    
    # Get repository info
    repo_data, success = check_github_api(
        "https://api.github.com/repos/imewei/homodyne",
        "repository accessibility"
    )
    
    if not success:
        print("❌ Cannot access repository. Possible causes:")
        print("   - Repository doesn't exist")
        print("   - Repository is private and you don't have access")
        print("   - Network connectivity issues")
        return False
    
    print(f"✅ Repository found: {repo_data['full_name']}")
    print(f"   Private: {'Yes' if repo_data['private'] else 'No'}")
    print(f"   Default branch: {repo_data['default_branch']}")
    print(f"   Has Pages: {'Yes' if repo_data.get('has_pages', False) else 'No'}")
    print(f"   Owner type: {repo_data['owner']['type']}")
    
    return repo_data


def diagnose_pages_settings():
    """Diagnose GitHub Pages settings."""
    print("\n📄 GITHUB PAGES SETTINGS")
    print("=" * 50)
    
    pages_data, success = check_github_api(
        "https://api.github.com/repos/imewei/homodyne/pages",
        "GitHub Pages configuration"
    )
    
    if not success:
        print("❌ GitHub Pages not configured or not accessible")
        print("   SOLUTION: Enable GitHub Pages in repository settings")
        print("   Go to: https://github.com/imewei/homodyne/settings/pages")
        return False
    
    print("✅ GitHub Pages is configured")
    print(f"   Status: {pages_data.get('status', 'unknown')}")
    print(f"   URL: {pages_data.get('html_url', 'N/A')}")
    
    source = pages_data.get('source', {})
    print(f"   Source type: {source.get('branch', 'unknown')}")
    print(f"   Source path: {source.get('path', 'unknown')}")
    
    # Check if using GitHub Actions
    if source.get('branch') == 'github-pages':
        print("✅ Using GitHub Actions deployment")
    elif source.get('branch'):
        print(f"⚠️  Using branch deployment: {source.get('branch')}")
        print("   For Actions deployment, change source to 'GitHub Actions'")
    
    return pages_data


def diagnose_actions():
    """Diagnose GitHub Actions."""
    print("\n⚙️ GITHUB ACTIONS DIAGNOSIS")
    print("=" * 50)
    
    # Check if Actions are enabled
    actions_data, success = check_github_api(
        "https://api.github.com/repos/imewei/homodyne/actions/permissions",
        "GitHub Actions permissions"
    )
    
    if success and actions_data:
        enabled = actions_data.get('enabled', False)
        print(f"✅ GitHub Actions enabled: {enabled}")
        if not enabled:
            print("❌ GitHub Actions are disabled")
            print("   SOLUTION: Enable Actions in Settings → Actions → General")
            return False
    else:
        print("⚠️  Cannot check Actions permissions (might be OK)")
    
    # Check recent workflow runs
    runs_data, success = check_github_api(
        "https://api.github.com/repos/imewei/homodyne/actions/runs?per_page=5",
        "recent workflow runs"
    )
    
    if success and runs_data:
        runs = runs_data.get('workflow_runs', [])
        print(f"📊 Recent workflow runs: {len(runs)}")
        
        for run in runs[:3]:  # Show last 3 runs
            name = run.get('name', 'Unknown')
            status = run.get('status', 'unknown')
            conclusion = run.get('conclusion', 'unknown')
            branch = run.get('head_branch', 'unknown')
            created = run.get('created_at', 'unknown')[:10]  # Just date
            
            print(f"   • {name} ({branch}): {status}/{conclusion} - {created}")
            
            if conclusion == 'failure':
                print(f"     🔗 Check logs: {run.get('html_url', 'N/A')}")
    
    return True


def diagnose_permissions():
    """Diagnose repository permissions."""
    print("\n🔐 PERMISSIONS DIAGNOSIS")
    print("=" * 50)
    
    # Check if we can get current user info
    user_data, success = check_github_api(
        "https://api.github.com/user",
        "current user authentication"
    )
    
    if success:
        print(f"✅ Authenticated as: {user_data.get('login', 'unknown')}")
    else:
        print("⚠️  Not authenticated with GitHub API (using anonymous)")
    
    # Check repository collaborators (might fail for public repos)
    collab_data, success = check_github_api(
        "https://api.github.com/repos/imewei/homodyne/collaborators",
        "repository collaborators"
    )
    
    if success:
        print(f"✅ Repository has {len(collab_data)} collaborators")
    else:
        print("ℹ️  Cannot check collaborators (might be normal for public repos)")


def diagnose_local_setup():
    """Diagnose local repository setup."""
    print("\n🔧 LOCAL SETUP DIAGNOSIS")
    print("=" * 50)
    
    # Check git status
    stdout, stderr, success = run_command("git status --porcelain")
    if success:
        if stdout:
            print(f"⚠️  {len(stdout.split())} uncommitted changes")
        else:
            print("✅ Working directory clean")
    else:
        print("❌ Not a git repository or git not available")
        return False
    
    # Check current branch
    stdout, stderr, success = run_command("git branch --show-current")
    if success:
        print(f"📍 Current branch: {stdout}")
    
    # Check remote
    stdout, stderr, success = run_command("git remote -v")
    if success:
        print("✅ Git remote configured:")
        for line in stdout.split('\n')[:2]:  # Show first 2 lines
            print(f"   {line}")
    
    # Check documentation build
    print("\n📖 Testing documentation build...")
    if os.path.exists('docs/Makefile'):
        stdout, stderr, success = run_command("cd docs && make html")
        if success and os.path.exists('docs/_build/html/index.html'):
            print("✅ Documentation builds successfully")
        else:
            print("❌ Documentation build failed")
            if stderr:
                print(f"   Error: {stderr[:200]}...")  # First 200 chars
            return False
    else:
        print("❌ No docs/Makefile found")
        return False
    
    return True


def provide_solutions():
    """Provide specific solutions based on diagnosis."""
    print("\n🎯 RECOMMENDED SOLUTIONS")
    print("=" * 50)
    
    print("1. 📋 IMMEDIATE ACTIONS:")
    print("   • Go to: https://github.com/imewei/homodyne/settings/pages")
    print("   • Set Source to 'GitHub Actions' (not 'Deploy from a branch')")
    print("   • Ensure repository is public or has Pages enabled")
    print("")
    
    print("2. 🔄 ALTERNATIVE WORKFLOWS AVAILABLE:")
    print("   • Main workflow: .github/workflows/docs.yml")
    print("   • Robust workflow: .github/workflows/docs-robust.yml")
    print("   • Peaceiris workflow: .github/workflows/docs-peaceiris.yml")
    print("")
    
    print("3. 🧪 TESTING STEPS:")
    print("   • Run: python diagnose_pages.py")
    print("   • Test locally: cd docs && make html")
    print("   • Manual trigger: GitHub → Actions → Run workflow")
    print("")
    
    print("4. 🆘 IF ALL ELSE FAILS:")
    print("   • Use branch-based deployment:")
    print("     - Set Pages source to 'gh-pages' branch")
    print("     - Run docs-robust.yml workflow")
    print("   • Check GitHub Status: https://githubstatus.com")
    print("   • Create issue: https://github.com/imewei/homodyne/issues")


def main():
    """Main diagnostic function."""
    print("🔍 GITHUB PAGES DEPLOYMENT DIAGNOSTICS")
    print("🌐 Repository: imewei/homodyne")
    print("=" * 60)
    
    all_good = True
    
    # Run diagnostics
    repo_data = diagnose_repository()
    if not repo_data:
        all_good = False
    
    pages_data = diagnose_pages_settings()
    if not pages_data:
        all_good = False
    
    actions_ok = diagnose_actions()
    if not actions_ok:
        all_good = False
    
    diagnose_permissions()
    
    local_ok = diagnose_local_setup()
    if not local_ok:
        all_good = False
    
    provide_solutions()
    
    print("\n" + "=" * 60)
    if all_good:
        print("🎉 DIAGNOSIS COMPLETE: Configuration looks good!")
        print("   If deployment still fails, try the alternative workflows.")
    else:
        print("⚠️  DIAGNOSIS COMPLETE: Issues found above")
        print("   Follow the recommended solutions to fix the problems.")
    
    return 0 if all_good else 1


if __name__ == "__main__":
    sys.exit(main())
