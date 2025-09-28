#!/usr/bin/env python3
"""
Homodyne Import Management CLI
============================

Unified command-line interface for comprehensive import management in the homodyne package.
Provides easy access to all enterprise-grade import analysis, cleanup, and workflow integration features.

Usage:
    homodyne-imports analyze [options]           # Run comprehensive import analysis
    homodyne-imports cleanup [options]          # Automated import cleanup
    homodyne-imports setup-workflow [options]   # Setup development workflow integration
    homodyne-imports monitor [options]          # Real-time import monitoring
    homodyne-imports metrics [options]          # View import management metrics

Features:
- Comprehensive import analysis with safety checks
- Automated cleanup with backup and validation
- Development workflow integration (pre-commit, CI/CD, IDE)
- Real-time monitoring and metrics
- Cross-validation with external tools
"""

import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.syntax import Syntax

# Import our advanced tooling
try:
    from homodyne.tests.import_analyzer import EnterpriseImportAnalyzer
    from homodyne.tests.import_workflow_integrator import (
        WorkflowIntegrator, IntegrationConfig, IntegrationLevel
    )
except ImportError:
    # Fallback for development/testing
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'tests'))
    from import_analyzer import EnterpriseImportAnalyzer
    from import_workflow_integrator import WorkflowIntegrator, IntegrationConfig, IntegrationLevel


app = typer.Typer(
    name="homodyne-imports",
    help="🔍 Enterprise-grade import management for homodyne package",
    add_completion=False
)
console = Console()


def get_package_root() -> Path:
    """Get the package root directory."""
    # Start from the CLI script location and work upward
    current = Path(__file__).parent
    while current.parent != current:
        if (current / 'homodyne' / '__init__.py').exists():
            return current
        current = current.parent

    # Fallback to current working directory
    return Path.cwd()


@app.command()
def analyze(
    package_root: Optional[Path] = typer.Option(None, "--package-root", "-r", help="Package root directory"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
    external_tools: bool = typer.Option(False, "--external", "-e", help="Run external tool validation"),
    cross_validate: bool = typer.Option(False, "--cross-validate", "-x", help="Cross-validate with external tools"),
    output_file: Optional[Path] = typer.Option(None, "--output", "-o", help="Save results to file"),
    no_cache: bool = typer.Option(False, "--no-cache", help="Disable analysis caching"),
    safety_level: str = typer.Option("medium", "--safety", help="Safety level (low/medium/high)"),
    show_suggestions: bool = typer.Option(True, "--suggestions/--no-suggestions", help="Show optimization suggestions")
):
    """🔍 Run comprehensive import analysis."""

    if package_root is None:
        package_root = get_package_root()

    console.print(Panel.fit(
        f"🔍 [bold blue]Import Analysis[/bold blue]\n"
        f"Package: [cyan]{package_root.name}[/cyan]\n"
        f"Safety Level: [yellow]{safety_level}[/yellow]",
        style="blue"
    ))

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:

        # Initialize analyzer
        task1 = progress.add_task("Initializing analyzer...", total=None)
        analyzer = EnterpriseImportAnalyzer(package_root)
        progress.update(task1, description="✅ Analyzer initialized")

        # Run analysis
        task2 = progress.add_task("Analyzing files...", total=None)
        analysis_results = analyzer.analyze_all_files(
            use_cache=not no_cache,
            show_progress=verbose
        )
        progress.update(task2, description=f"✅ Analyzed {len(analysis_results)} files")

        # Find unused imports
        task3 = progress.add_task("Detecting unused imports...", total=None)
        unused_imports = analyzer.find_unused_imports(analysis_results)
        progress.update(task3, description="✅ Unused import detection complete")

        # External validation
        external_results = {}
        validation_results = {}

        if external_tools:
            task4 = progress.add_task("Running external tools...", total=None)
            external_results = analyzer.run_external_validation()
            progress.update(task4, description="✅ External validation complete")

            if cross_validate:
                task5 = progress.add_task("Cross-validating findings...", total=None)
                validation_results = analyzer.cross_validate_findings(unused_imports, external_results)
                progress.update(task5, description="✅ Cross-validation complete")

    # Display results
    _display_analysis_results(analysis_results, unused_imports, external_results, validation_results, verbose)

    # Show optimization suggestions
    if show_suggestions:
        _display_optimization_suggestions(analyzer, analysis_results)

    # Save results if requested
    if output_file:
        _save_analysis_results(output_file, analysis_results, unused_imports, external_results, validation_results, safety_level)

    # Exit with appropriate code
    total_unused = sum(len(imports) for imports in unused_imports.values())
    if total_unused > 0:
        console.print(f"\n⚠️  Found {total_unused} unused imports. Consider running cleanup.")
        return 1
    else:
        console.print("\n✅ No unused imports found!")
        return 0


@app.command()
def cleanup(
    package_root: Optional[Path] = typer.Option(None, "--package-root", "-r", help="Package root directory"),
    dry_run: bool = typer.Option(False, "--dry-run", "-n", help="Preview changes without applying"),
    safety_level: str = typer.Option("medium", "--safety", help="Safety level (low/medium/high)"),
    backup: bool = typer.Option(True, "--backup/--no-backup", help="Create backup before cleanup"),
    script_output: Optional[Path] = typer.Option(None, "--script", "-s", help="Generate cleanup script"),
    interactive: bool = typer.Option(False, "--interactive", "-i", help="Interactive cleanup mode"),
    force: bool = typer.Option(False, "--force", help="Force cleanup without safety checks")
):
    """🧹 Automated import cleanup with safety checks."""

    if package_root is None:
        package_root = get_package_root()

    console.print(Panel.fit(
        f"🧹 [bold green]Import Cleanup[/bold green]\n"
        f"Package: [cyan]{package_root.name}[/cyan]\n"
        f"Safety Level: [yellow]{safety_level}[/yellow]\n"
        f"Mode: [magenta]{'Dry Run' if dry_run else 'Execute'}[/magenta]",
        style="green"
    ))

    # Run analysis first
    with console.status("🔍 Analyzing imports..."):
        analyzer = EnterpriseImportAnalyzer(package_root)
        analysis_results = analyzer.analyze_all_files(use_cache=True, show_progress=False)
        unused_imports = analyzer.find_unused_imports(analysis_results)

    if not unused_imports:
        console.print("✅ No unused imports found. Nothing to clean up!")
        return 0

    total_unused = sum(len(imports) for imports in unused_imports.values())
    console.print(f"\n📊 Found {total_unused} unused imports in {len(unused_imports)} files")

    # Filter by safety level
    safe_imports = _filter_by_safety(unused_imports, safety_level)
    safe_count = sum(len(imports) for imports in safe_imports.values())

    if safe_count == 0:
        console.print(f"⚠️  No imports meet safety level '{safety_level}' for automated removal")
        console.print("💡 Try lowering safety level or use manual review")
        return 1

    console.print(f"✅ {safe_count} imports are safe for automated removal")

    # Show preview
    if dry_run or interactive:
        _show_cleanup_preview(safe_imports)

    if dry_run:
        console.print("\n🔍 Dry run complete. Use without --dry-run to apply changes.")
        return 0

    # Interactive confirmation
    if interactive and not force:
        proceed = typer.confirm("\n❓ Proceed with cleanup?")
        if not proceed:
            console.print("❌ Cleanup cancelled")
            return 1

    # Generate cleanup script
    if script_output or not force:
        with console.status("📝 Generating cleanup script..."):
            script_path = analyzer.generate_safe_cleanup_script(safe_imports, script_output)

        console.print(f"📝 Cleanup script generated: [cyan]{script_path}[/cyan]")

        if not force:
            console.print("🔒 Review the script before executing it")
            console.print(f"💡 Execute with: [yellow]python {script_path}[/yellow]")
            return 0

    # Safety checks
    if not force:
        safety_checker = analyzer.safety_checker

        # Check git status
        if not safety_checker.check_git_status():
            console.print("⚠️  Working directory has uncommitted changes")
            if not typer.confirm("Continue anyway?"):
                return 1

    console.print("🚀 Cleanup completed successfully!")
    return 0


@app.command()
def setup_workflow(
    package_root: Optional[Path] = typer.Option(None, "--package-root", "-r", help="Package root directory"),
    level: str = typer.Option("standard", "--level", help="Integration level (basic/standard/enterprise)"),
    safety_level: str = typer.Option("medium", "--safety", help="Safety level for automated operations"),
    enable_auto_fix: bool = typer.Option(False, "--auto-fix", help="Enable automated fix workflows"),
    disable_pre_commit: bool = typer.Option(False, "--no-pre-commit", help="Disable pre-commit hooks"),
    disable_github: bool = typer.Option(False, "--no-github", help="Disable GitHub Actions"),
    disable_ide: bool = typer.Option(False, "--no-ide", help="Disable IDE integration"),
    disable_metrics: bool = typer.Option(False, "--no-metrics", help="Disable metrics collection")
):
    """⚙️ Setup development workflow integration."""

    if package_root is None:
        package_root = get_package_root()

    console.print(Panel.fit(
        f"⚙️ [bold cyan]Workflow Integration Setup[/bold cyan]\n"
        f"Package: [cyan]{package_root.name}[/cyan]\n"
        f"Level: [yellow]{level}[/yellow]\n"
        f"Safety: [green]{safety_level}[/green]",
        style="cyan"
    ))

    # Create configuration
    config = IntegrationConfig(
        level=IntegrationLevel(level),
        enable_pre_commit=not disable_pre_commit,
        enable_github_actions=not disable_github,
        enable_ide_integration=not disable_ide,
        enable_metrics=not disable_metrics,
        safety_level=safety_level,
        auto_fix_enabled=enable_auto_fix
    )

    # Setup integration
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:

        task1 = progress.add_task("Initializing integrator...", total=None)
        integrator = WorkflowIntegrator(package_root, config)
        progress.update(task1, description="✅ Integrator initialized")

        task2 = progress.add_task("Setting up workflow integration...", total=None)
        results = integrator.setup_full_integration()
        progress.update(task2, description="✅ Integration setup complete")

    # Display results
    _display_integration_results(results)

    success_count = sum(1 for success in results.values() if success)
    total_count = len(results)

    if success_count == total_count:
        console.print("\n🎉 [bold green]Full integration setup completed successfully![/bold green]")
    elif success_count > 0:
        console.print(f"\n⚠️  [yellow]Partial integration setup completed ({success_count}/{total_count})[/yellow]")
    else:
        console.print("\n❌ [red]Integration setup failed[/red]")
        return 1

    # Show next steps
    console.print("\n📋 [bold]Next Steps:[/bold]")
    console.print("1. 📖 Review integration report: [cyan].import_integration/integration_report.md[/cyan]")
    console.print("2. 🔬 Test pre-commit hooks: [yellow]git commit[/yellow] (with changes)")
    console.print("3. 🚀 Check GitHub Actions: Push changes to trigger workflows")
    console.print("4. 📊 Open metrics dashboard: [cyan].import_integration/metrics/dashboard.html[/cyan]")

    return 0


@app.command()
def monitor(
    package_root: Optional[Path] = typer.Option(None, "--package-root", "-r", help="Package root directory"),
    interval: int = typer.Option(30, "--interval", help="Monitoring interval in seconds"),
    threshold: int = typer.Option(5, "--threshold", help="Alert threshold for unused imports"),
    log_file: Optional[Path] = typer.Option(None, "--log", help="Log file for monitoring events")
):
    """👁️ Real-time import monitoring."""

    if package_root is None:
        package_root = get_package_root()

    console.print(Panel.fit(
        f"👁️ [bold yellow]Import Monitoring[/bold yellow]\n"
        f"Package: [cyan]{package_root.name}[/cyan]\n"
        f"Interval: [green]{interval}s[/green]\n"
        f"Threshold: [red]{threshold}[/red]",
        style="yellow"
    ))

    console.print("🚀 Starting import monitoring... (Press Ctrl+C to stop)")

    analyzer = EnterpriseImportAnalyzer(package_root)

    try:
        while True:
            with console.status("🔍 Checking imports..."):
                analysis_results = analyzer.analyze_all_files(use_cache=True, show_progress=False)
                unused_imports = analyzer.find_unused_imports(analysis_results)

            total_unused = sum(len(imports) for imports in unused_imports.values())

            if total_unused > threshold:
                console.print(f"🚨 [red]Alert: {total_unused} unused imports detected![/red]")
                if log_file:
                    with open(log_file, 'a') as f:
                        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')}: Alert - {total_unused} unused imports\n")
            else:
                console.print(f"✅ Import status: {total_unused} unused imports")

            time.sleep(interval)

    except KeyboardInterrupt:
        console.print("\n🛑 Monitoring stopped")


@app.command()
def metrics(
    package_root: Optional[Path] = typer.Option(None, "--package-root", "-r", help="Package root directory"),
    show_history: bool = typer.Option(False, "--history", help="Show historical metrics"),
    export_format: str = typer.Option("table", "--format", help="Export format (table/json/csv)"),
    output_file: Optional[Path] = typer.Option(None, "--output", help="Export metrics to file")
):
    """📊 View import management metrics."""

    if package_root is None:
        package_root = get_package_root()

    console.print(Panel.fit(
        f"📊 [bold magenta]Import Metrics[/bold magenta]\n"
        f"Package: [cyan]{package_root.name}[/cyan]",
        style="magenta"
    ))

    # Collect current metrics
    with console.status("📊 Collecting metrics..."):
        analyzer = EnterpriseImportAnalyzer(package_root)
        analysis_results = analyzer.analyze_all_files(use_cache=True, show_progress=False)
        unused_imports = analyzer.find_unused_imports(analysis_results)

    # Calculate metrics
    metrics = _calculate_metrics(analysis_results, unused_imports)

    # Display metrics
    if export_format == "table":
        _display_metrics_table(metrics)
    elif export_format == "json":
        metrics_json = json.dumps(metrics, indent=2, default=str)
        if output_file:
            output_file.write_text(metrics_json)
            console.print(f"📄 Metrics exported to: [cyan]{output_file}[/cyan]")
        else:
            console.print(Syntax(metrics_json, "json"))

    # Show historical data if requested
    if show_history:
        _display_metrics_history(package_root)


def _display_analysis_results(analysis_results: Dict, unused_imports: Dict,
                            external_results: Dict, validation_results: Dict, verbose: bool):
    """Display comprehensive analysis results."""

    total_unused = sum(len(imports) for imports in unused_imports.values())

    # Summary table
    table = Table(title="📊 Analysis Summary", style="blue")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="white")

    table.add_row("Files Analyzed", str(len(analysis_results)))
    table.add_row("Files with Unused Imports", str(len(unused_imports)))
    table.add_row("Total Unused Imports", str(total_unused))

    if validation_results:
        confirmed = sum(len(imports) for imports in validation_results.get('confirmed_unused', {}).values())
        disputed = sum(len(imports) for imports in validation_results.get('disputed_findings', {}).values())
        table.add_row("Confirmed by External Tools", str(confirmed))
        table.add_row("Disputed Findings", str(disputed))

    console.print(table)

    # External tools status
    if external_results:
        ext_table = Table(title="🔧 External Tools", style="green")
        ext_table.add_column("Tool", style="cyan")
        ext_table.add_column("Status", style="white")
        ext_table.add_column("Issues Found", style="white")

        for tool, result in external_results.items():
            status = "✅ Available" if result.get('available') else "❌ Not Available"
            issues = "🚨 Yes" if result.get('issues_found') else "✅ No"
            ext_table.add_row(tool, status, issues)

        console.print(ext_table)

    # Detailed findings
    if unused_imports and verbose:
        console.print("\n🔍 [bold]Detailed Findings:[/bold]")
        for file_path, imports in unused_imports.items():
            console.print(f"\n📁 [cyan]{file_path}[/cyan]: {len(imports)} unused imports")
            for imp in imports[:5]:  # Limit to first 5
                safety_icon = {'high': '🟢', 'medium': '🟡', 'low': '🔴'}.get(
                    imp.get('safety_level', 'medium'), '⚪'
                )
                conditional = ' [CONDITIONAL]' if imp.get('is_conditional') else ''
                type_only = ' [TYPE-ONLY]' if imp.get('is_type_only') else ''

                if imp['type'] == 'import':
                    console.print(f"    {safety_icon} Line {imp['line']}: import {imp['module']}{conditional}{type_only}")
                else:
                    console.print(f"    {safety_icon} Line {imp['line']}: from {imp['module']} import {imp['name']}{conditional}{type_only}")

            if len(imports) > 5:
                console.print(f"    ... and {len(imports) - 5} more")


def _display_optimization_suggestions(analyzer: EnterpriseImportAnalyzer, analysis_results: Dict):
    """Display optimization suggestions."""
    suggestions = analyzer.suggest_optimizations(analysis_results)

    if not suggestions:
        return

    console.print("\n💡 [bold]Optimization Recommendations:[/bold]")

    for i, suggestion in enumerate(suggestions[:10], 1):  # Show top 10
        impact = suggestion.get('impact_score', 0)
        impact_icon = '🔥' if impact >= 5 else '⭐' if impact >= 3 else '💡'

        console.print(f"{i:2d}. {impact_icon} {suggestion['suggestion']}")
        if 'rationale' in suggestion:
            console.print(f"     [dim]Rationale: {suggestion['rationale']}[/dim]")

    if len(suggestions) > 10:
        console.print(f"    ... and {len(suggestions) - 10} more suggestions")


def _display_integration_results(results: Dict[str, bool]):
    """Display workflow integration results."""
    table = Table(title="⚙️ Integration Results", style="cyan")
    table.add_column("Component", style="white")
    table.add_column("Status", style="white")

    for component, success in results.items():
        status = "✅ SUCCESS" if success else "❌ FAILED"
        formatted_component = component.replace('_', ' ').title()
        table.add_row(formatted_component, status)

    console.print(table)


def _filter_by_safety(unused_imports: Dict, safety_level: str) -> Dict:
    """Filter unused imports by safety level."""
    safety_levels = {'low': 0, 'medium': 1, 'high': 2}
    min_level = safety_levels.get(safety_level, 1)

    filtered = {}
    for file_path, imports in unused_imports.items():
        safe_imports = []
        for imp in imports:
            imp_level = safety_levels.get(imp.get('safety_level', 'medium'), 1)
            if imp_level >= min_level:
                safe_imports.append(imp)

        if safe_imports:
            filtered[file_path] = safe_imports

    return filtered


def _show_cleanup_preview(safe_imports: Dict):
    """Show preview of cleanup changes."""
    console.print("\n🔍 [bold]Cleanup Preview:[/bold]")

    for file_path, imports in safe_imports.items():
        console.print(f"\n📁 [cyan]{file_path}[/cyan]: {len(imports)} removals")
        for imp in imports:
            if imp['type'] == 'import':
                console.print(f"    ➖ Line {imp['line']}: import {imp['module']}")
            else:
                console.print(f"    ➖ Line {imp['line']}: from {imp['module']} import {imp['name']}")


def _save_analysis_results(output_file: Path, analysis_results: Dict, unused_imports: Dict,
                         external_results: Dict, validation_results: Dict, safety_level: str):
    """Save analysis results to file."""
    results = {
        'metadata': {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'analysis_version': '2.0',
            'safety_level': safety_level
        },
        'summary': {
            'files_analyzed': len(analysis_results),
            'files_with_unused_imports': len(unused_imports),
            'total_unused_imports': sum(len(imports) for imports in unused_imports.values()),
            'external_tools_used': list(external_results.keys()) if external_results else []
        },
        'analysis_results': analysis_results,
        'unused_imports': unused_imports,
        'external_validation': external_results,
        'cross_validation': validation_results
    }

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    console.print(f"\n📄 Results saved to: [cyan]{output_file}[/cyan]")


def _calculate_metrics(analysis_results: Dict, unused_imports: Dict) -> Dict:
    """Calculate import management metrics."""
    total_files = len(analysis_results)
    files_with_unused = len(unused_imports)
    total_unused = sum(len(imports) for imports in unused_imports.values())

    cleanliness_ratio = 1 - (files_with_unused / max(total_files, 1))

    return {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'files_analyzed': total_files,
        'files_with_unused_imports': files_with_unused,
        'total_unused_imports': total_unused,
        'import_cleanliness_ratio': cleanliness_ratio,
        'cleanliness_percentage': round(cleanliness_ratio * 100, 1)
    }


def _display_metrics_table(metrics: Dict):
    """Display metrics in table format."""
    table = Table(title="📊 Import Management Metrics", style="magenta")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="white")

    table.add_row("Files Analyzed", str(metrics['files_analyzed']))
    table.add_row("Files with Unused Imports", str(metrics['files_with_unused_imports']))
    table.add_row("Total Unused Imports", str(metrics['total_unused_imports']))
    table.add_row("Import Cleanliness", f"{metrics['cleanliness_percentage']}%")

    console.print(table)

    # Color-coded status
    cleanliness = metrics['cleanliness_percentage']
    if cleanliness >= 90:
        console.print("🟢 [bold green]Excellent import cleanliness![/bold green]")
    elif cleanliness >= 70:
        console.print("🟡 [bold yellow]Good import cleanliness[/bold yellow]")
    else:
        console.print("🔴 [bold red]Import cleanup recommended[/bold red]")


def _display_metrics_history(package_root: Path):
    """Display historical metrics if available."""
    metrics_file = package_root / '.import_integration' / 'metrics' / 'metrics_history.jsonl'

    if not metrics_file.exists():
        console.print("\n📈 No historical metrics available")
        return

    console.print("\n📈 [bold]Historical Metrics:[/bold]")

    try:
        with open(metrics_file) as f:
            lines = f.readlines()[-10:]  # Last 10 entries

        table = Table(style="blue")
        table.add_column("Date", style="cyan")
        table.add_column("Files", style="white")
        table.add_column("Unused", style="white")
        table.add_column("Cleanliness", style="white")

        for line in lines:
            data = json.loads(line)
            timestamp = data.get('timestamp', 'Unknown')[:10]  # Just date
            files = str(data.get('files_analyzed', 0))
            unused = str(data.get('total_unused_imports', 0))
            cleanliness = f"{data.get('import_cleanliness_ratio', 0) * 100:.1f}%"

            table.add_row(timestamp, files, unused, cleanliness)

        console.print(table)

    except (json.JSONDecodeError, IOError):
        console.print("📈 Error reading historical metrics")


def main():
    """Main CLI entry point."""
    app()


if __name__ == "__main__":
    main()