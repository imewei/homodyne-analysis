"""
Advanced CLI Enhancement and User Experience Optimization
========================================================

Enhanced command-line interface with real-time feedback, intelligent progress tracking,
and optimized user workflows for scientific computing applications.

Features:
- Interactive command-line interface with rich formatting
- Real-time progress tracking with ETA calculation
- Intelligent error reporting with contextual suggestions
- Performance monitoring and optimization recommendations
- Smart configuration validation and assistance
- Multi-level verbosity control
- Cross-platform compatibility
"""

import argparse
import logging
import time
from pathlib import Path
from typing import Any

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.prompt import Confirm
    from rich.prompt import Prompt
    from rich.table import Table
    from rich.text import Text

    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    Console = None

from .progress import ProgressTracker


class CLIEnhancer:
    """
    Advanced CLI enhancement system for scientific computing applications.

    Provides intelligent user experience improvements including real-time feedback,
    progress tracking, error guidance, and performance optimization suggestions.
    """

    def __init__(self, enable_rich: bool = True, verbosity: int = 1):
        self.enable_rich = enable_rich and RICH_AVAILABLE
        self.verbosity = verbosity
        self.console = Console() if self.enable_rich else None
        self.start_time = time.time()
        self.performance_metrics = {}

        # Initialize logging
        self.logger = logging.getLogger(__name__)

    def print_header(self, title: str, subtitle: str = "") -> None:
        """Print application header with styling."""
        if self.enable_rich and self.console:
            title_text = Text(title, style="bold blue")
            if subtitle:
                subtitle_text = Text(subtitle, style="dim")
                content = f"{title_text}\n{subtitle_text}"
            else:
                content = title_text

            panel = Panel(content, border_style="blue", padding=(1, 2))
            self.console.print(panel)
        else:
            print(f"\n{'=' * 60}")
            print(f"{title}")
            if subtitle:
                print(f"{subtitle}")
            print(f"{'=' * 60}\n")

    def print_configuration_summary(self, config: dict[str, Any]) -> None:
        """Print configuration summary with validation status."""
        if self.verbosity < 1:
            return

        if self.enable_rich and self.console:
            table = Table(
                title="Configuration Summary",
                show_header=True,
                header_style="bold blue",
            )
            table.add_column("Setting", style="cyan", no_wrap=True)
            table.add_column("Value", style="green")
            table.add_column("Status", justify="center")

            # Add key configuration items
            self._add_config_rows(table, config)

            self.console.print(table)
        else:
            print("\nConfiguration Summary:")
            print("-" * 40)
            self._print_config_text(config)

    def _add_config_rows(self, table: "Table", config: dict[str, Any]) -> None:
        """Add configuration rows to rich table."""
        # Analysis mode
        analysis_mode = config.get("analysis_settings", {}).get("static_mode", False)
        mode_str = "Static" if analysis_mode else "Laminar Flow"
        table.add_row("Analysis Mode", mode_str, "✓")

        # Frame range
        analyzer = config.get("analyzer_parameters", {})
        start_frame = analyzer.get("start_frame", "N/A")
        end_frame = analyzer.get("end_frame", "N/A")
        frame_count = (
            end_frame - start_frame
            if isinstance(start_frame, int) and isinstance(end_frame, int)
            else "N/A"
        )
        table.add_row(
            "Frame Range", f"{start_frame} - {end_frame} ({frame_count} frames)", "✓"
        )

        # Data paths
        exp_data = config.get("experimental_data", {})
        data_path = exp_data.get("data_folder_path", "N/A")
        data_exists = Path(data_path).exists() if data_path != "N/A" else False
        status = "✓" if data_exists else "⚠"
        table.add_row("Data Path", str(data_path), status)

        # Angle filtering
        angle_config = config.get("optimization_config", {}).get("angle_filtering", {})
        angle_enabled = angle_config.get("enabled", True)
        angle_status = "✓" if angle_enabled else "✗"
        table.add_row(
            "Angle Filtering", "Enabled" if angle_enabled else "Disabled", angle_status
        )

    def _print_config_text(self, config: dict[str, Any]) -> None:
        """Print configuration in plain text format."""
        # Analysis mode
        analysis_mode = config.get("analysis_settings", {}).get("static_mode", False)
        mode_str = "Static" if analysis_mode else "Laminar Flow"
        print(f"Analysis Mode: {mode_str}")

        # Frame range
        analyzer = config.get("analyzer_parameters", {})
        start_frame = analyzer.get("start_frame", "N/A")
        end_frame = analyzer.get("end_frame", "N/A")
        print(f"Frame Range: {start_frame} - {end_frame}")

        # Data paths
        exp_data = config.get("experimental_data", {})
        data_path = exp_data.get("data_folder_path", "N/A")
        print(f"Data Path: {data_path}")

        print()

    def print_method_selection(self, methods: list[str], selected_method: str) -> None:
        """Print method selection information."""
        if self.verbosity < 1:
            return

        if self.enable_rich and self.console:
            table = Table(
                title="Analysis Methods", show_header=True, header_style="bold magenta"
            )
            table.add_column("Method", style="cyan")
            table.add_column("Type", style="yellow")
            table.add_column("Status", justify="center")

            method_types = {
                "nelder_mead": "Classical",
                "gurobi": "Classical",
                "wasserstein": "Robust",
                "scenario": "Robust",
                "ellipsoidal": "Robust",
            }

            for method in methods:
                method_type = method_types.get(method.lower(), "Unknown")
                status = (
                    "✓ Selected"
                    if method.lower() in selected_method.lower()
                    else "○ Available"
                )
                table.add_row(method.title().replace("_", " "), method_type, status)

            self.console.print(table)
        else:
            print(f"\nSelected Analysis Methods: {selected_method}")
            print(f"Available Methods: {', '.join(methods)}\n")

    def show_progress_summary(self, tracker: ProgressTracker) -> None:
        """Display progress summary and performance metrics."""
        if self.verbosity < 1:
            return

        summary = tracker.get_summary()

        if self.enable_rich and self.console:
            # Create progress summary table
            table = Table(
                title="Analysis Progress Summary",
                show_header=True,
                header_style="bold green",
            )
            table.add_column("Task", style="cyan")
            table.add_column("Progress", justify="center")
            table.add_column("Status", justify="center")
            table.add_column("Time", style="yellow")

            for _task_id, task_info in summary["tasks"].items():
                progress_pct = task_info["progress"] * 100
                progress_str = f"{progress_pct:.1f}%"

                if task_info["completed"]:
                    status = "✓ Done" if not task_info["failed"] else "✗ Failed"
                    status_style = "green" if not task_info["failed"] else "red"
                else:
                    status = "⏳ Running"
                    status_style = "yellow"

                elapsed = task_info["elapsed"]
                time_str = f"{elapsed:.1f}s"

                table.add_row(
                    task_info["description"],
                    progress_str,
                    f"[{status_style}]{status}[/{status_style}]",
                    time_str,
                )

            self.console.print(table)

            # Summary stats
            stats_text = f"Total: {summary['total_tasks']} | Completed: {summary['completed_tasks']} | Failed: {summary['failed_tasks']}"
            self.console.print(f"\n[bold]{stats_text}[/bold]")
        else:
            print("\nProgress Summary:")
            print("-" * 40)
            for _task_id, task_info in summary["tasks"].items():
                progress_pct = task_info["progress"] * 100
                status = "DONE" if task_info["completed"] else "RUNNING"
                elapsed = task_info["elapsed"]
                print(
                    f"{task_info['description']}: {progress_pct:.1f}% [{status}] ({elapsed:.1f}s)"
                )

            print(
                f"\nTotal: {summary['total_tasks']} | Completed: {summary['completed_tasks']} | Failed: {summary['failed_tasks']}"
            )

    def print_error_with_context(
        self, error: Exception, context: str = "", suggestions: list[str] | None = None
    ) -> None:
        """Print error with contextual information and suggestions."""
        if suggestions is None:
            suggestions = []

        if self.enable_rich and self.console:
            # Error panel
            error_text = f"[bold red]Error:[/bold red] {error!s}"
            if context:
                error_text += f"\n[dim]Context: {context}[/dim]"

            self.console.print(
                Panel(
                    error_text,
                    title="[red]Error Occurred[/red]",
                    border_style="red",
                    padding=(1, 2),
                )
            )

            # Suggestions
            if suggestions:
                self.console.print("\n[bold yellow]Suggestions:[/bold yellow]")
                for i, suggestion in enumerate(suggestions, 1):
                    self.console.print(f"  {i}. {suggestion}")
        else:
            print(f"\nERROR: {error!s}")
            if context:
                print(f"Context: {context}")
            if suggestions:
                print("\nSuggestions:")
                for i, suggestion in enumerate(suggestions, 1):
                    print(f"  {i}. {suggestion}")
            print()

    def prompt_user_choice(
        self, question: str, choices: list[str], default: str | None = None
    ) -> str:
        """Prompt user for a choice with validation."""
        if self.enable_rich and self.console:
            choice_text = " / ".join(f"[cyan]{choice}[/cyan]" for choice in choices)
            if default:
                choice_text += f" (default: [green]{default}[/green])"

            while True:
                answer = Prompt.ask(f"{question} [{choice_text}]")
                if not answer and default:
                    return default
                if answer.lower() in [c.lower() for c in choices]:
                    return answer.lower()
                self.console.print(
                    f"[red]Invalid choice. Please select from: {', '.join(choices)}[/red]"
                )
        else:
            choice_text = " / ".join(choices)
            if default:
                choice_text += f" (default: {default})"

            while True:
                answer = input(f"{question} [{choice_text}]: ").strip()
                if not answer and default:
                    return default
                if answer.lower() in [c.lower() for c in choices]:
                    return answer.lower()
                print(f"Invalid choice. Please select from: {', '.join(choices)}")

    def confirm_action(self, message: str, default: bool = True) -> bool:
        """Confirm user action with yes/no prompt."""
        if self.enable_rich and self.console:
            return Confirm.ask(message, default=default)
        default_text = "Y/n" if default else "y/N"
        while True:
            answer = input(f"{message} [{default_text}]: ").strip().lower()
            if not answer:
                return default
            if answer in ["y", "yes"]:
                return True
            if answer in ["n", "no"]:
                return False
            print("Please answer with 'y' or 'n'")

    def print_performance_summary(
        self, execution_time: float, memory_usage: float | None = None
    ) -> None:
        """Print performance summary at the end of execution."""
        if self.verbosity < 1:
            return

        if self.enable_rich and self.console:
            perf_table = Table(
                title="Performance Summary", show_header=True, header_style="bold cyan"
            )
            perf_table.add_column("Metric", style="cyan")
            perf_table.add_column("Value", style="green")

            perf_table.add_row("Total Execution Time", f"{execution_time:.2f} seconds")

            if memory_usage:
                perf_table.add_row("Peak Memory Usage", f"{memory_usage:.1f} MB")

            # Add optimization suggestions
            suggestions = self._get_performance_suggestions(
                execution_time, memory_usage
            )
            if suggestions:
                perf_table.add_row("Optimization Hints", "\n".join(suggestions))

            self.console.print(perf_table)
        else:
            print("\nPerformance Summary:")
            print("-" * 30)
            print(f"Total Execution Time: {execution_time:.2f} seconds")
            if memory_usage:
                print(f"Peak Memory Usage: {memory_usage:.1f} MB")

            suggestions = self._get_performance_suggestions(
                execution_time, memory_usage
            )
            if suggestions:
                print("\nOptimization Hints:")
                for suggestion in suggestions:
                    print(f"  - {suggestion}")
            print()

    def _get_performance_suggestions(
        self, execution_time: float, memory_usage: float | None
    ) -> list[str]:
        """Generate performance optimization suggestions."""
        suggestions = []

        if execution_time > 300:  # 5 minutes
            suggestions.append(
                "Consider using angle filtering to reduce computation time"
            )
            suggestions.append("Try static mode if your sample is not flowing")

        if memory_usage and memory_usage > 2000:  # 2GB
            suggestions.append("Consider processing smaller frame ranges")
            suggestions.append("Enable data caching to reduce memory usage")

        if execution_time > 60:  # 1 minute
            suggestions.append("Consider enabling parallel processing if available")

        return suggestions

    def show_results_preview(self, results: dict[str, Any]) -> None:
        """Show a preview of analysis results."""
        if self.verbosity < 1:
            return

        if self.enable_rich and self.console:
            # Results summary table
            table = Table(
                title="Analysis Results Summary",
                show_header=True,
                header_style="bold green",
            )
            table.add_column("Method", style="cyan")
            table.add_column("Chi-squared", style="yellow")
            table.add_column("Status", justify="center")

            # Extract method results
            method_results = results.get("method_results", {})
            for method_name, method_data in method_results.items():
                chi2 = method_data.get("chi_squared", "N/A")
                chi2_str = (
                    f"{chi2:.2e}" if isinstance(chi2, (int, float)) else str(chi2)
                )

                success = method_data.get("success", True)
                status = "✓ Success" if success else "✗ Failed"
                status_style = "green" if success else "red"

                table.add_row(
                    method_name.replace("_", " ").title(),
                    chi2_str,
                    f"[{status_style}]{status}[/{status_style}]",
                )

            self.console.print(table)

            # Output information
            output_dir = results.get("output_directory", "N/A")
            if output_dir != "N/A":
                self.console.print(f"\n[bold]Results saved to:[/bold] {output_dir}")
        else:
            print("\nAnalysis Results Summary:")
            print("-" * 40)

            method_results = results.get("method_results", {})
            for method_name, method_data in method_results.items():
                chi2 = method_data.get("chi_squared", "N/A")
                success = method_data.get("success", True)
                status = "SUCCESS" if success else "FAILED"

                print(f"{method_name}: Chi² = {chi2}, Status = {status}")

            output_dir = results.get("output_directory", "N/A")
            if output_dir != "N/A":
                print(f"\nResults saved to: {output_dir}")

    def interactive_configuration_assistant(self) -> dict[str, Any]:
        """Interactive assistant for configuration setup."""
        if not self.enable_rich:
            print("Interactive assistant requires 'rich' package. Using defaults.")
            return {}

        self.console.print(
            Panel(
                "[bold blue]Configuration Assistant[/bold blue]\n"
                "Let's set up your analysis configuration interactively.",
                border_style="blue",
                padding=(1, 2),
            )
        )

        config = {}

        # Analysis mode selection
        mode_choice = self.prompt_user_choice(
            "Select analysis mode", ["static", "laminar_flow"], "static"
        )

        config["analysis_settings"] = {"static_mode": mode_choice == "static"}

        if mode_choice == "static":
            submode = self.prompt_user_choice(
                "Select static submode", ["isotropic", "anisotropic"], "anisotropic"
            )
            config["analysis_settings"]["static_submode"] = submode

        # Method selection
        method_choice = self.prompt_user_choice(
            "Select optimization methods", ["classical", "robust", "all"], "classical"
        )

        config["selected_method"] = method_choice

        # Data path
        data_path = Prompt.ask("Enter data folder path", default="./data/")

        config["experimental_data"] = {"data_folder_path": data_path}

        self.console.print("\n[green]Configuration complete![/green]")
        return config

    def show_help_topics(self) -> None:
        """Show available help topics and usage examples."""
        if self.enable_rich and self.console:
            help_panel = Panel(
                "[bold]Common Usage Examples:[/bold]\n\n"
                "[cyan]Basic analysis:[/cyan]\n"
                "  homodyne --method classical --config my_config.json\n\n"
                "[cyan]Robust optimization:[/cyan]\n"
                "  homodyne --method robust --config my_config.json\n\n"
                "[cyan]Compare all methods:[/cyan]\n"
                "  homodyne --method all --config my_config.json\n\n"
                "[cyan]Interactive mode:[/cyan]\n"
                "  homodyne --interactive\n\n"
                "[cyan]Enable shell completion:[/cyan]\n"
                "  homodyne --install-completion bash",
                title="[green]Help & Examples[/green]",
                border_style="green",
                padding=(1, 2),
            )
            self.console.print(help_panel)
        else:
            print("\nCommon Usage Examples:")
            print("=" * 40)
            print("Basic analysis:")
            print("  homodyne --method classical --config my_config.json")
            print("\nRobust optimization:")
            print("  homodyne --method robust --config my_config.json")
            print("\nCompare all methods:")
            print("  homodyne --method all --config my_config.json")
            print("\nEnable shell completion:")
            print("  homodyne --install-completion bash")
            print()


def enhance_argument_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Enhance argument parser with improved help and validation."""

    # Add interactive mode
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Launch interactive configuration assistant",
    )

    # Add verbosity control
    verbosity_group = parser.add_mutually_exclusive_group()
    verbosity_group.add_argument(
        "--verbose",
        "-v",
        action="count",
        default=1,
        help="Increase verbosity (use -vv for more verbose output)",
    )
    verbosity_group.add_argument(
        "--quiet", "-q", action="store_true", help="Suppress non-essential output"
    )

    # Add performance monitoring
    parser.add_argument(
        "--monitor-performance",
        action="store_true",
        help="Enable detailed performance monitoring",
    )

    # Add validation flags
    parser.add_argument(
        "--validate-config", action="store_true", help="Validate configuration and exit"
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate configuration and show what would be executed",
    )

    return parser


def create_enhanced_cli() -> tuple[argparse.ArgumentParser, CLIEnhancer]:
    """Create enhanced CLI parser and interface."""
    parser = argparse.ArgumentParser(
        description="Enhanced Homodyne Scattering Analysis with Real-time Feedback",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --method classical --config my_config.json
  %(prog)s --method all --verbose --monitor-performance
  %(prog)s --interactive
  %(prog)s --validate-config --config my_config.json

For shell completion:
  %(prog)s --install-completion bash
  %(prog)s --install-completion zsh
""",
    )

    # Core arguments (keep existing ones)
    parser.add_argument(
        "--method",
        "-m",
        choices=["classical", "robust", "all"],
        default="classical",
        help="Analysis method to use (default: classical)",
    )

    parser.add_argument(
        "--config",
        "-c",
        default="./homodyne_config.json",
        help="Path to configuration file (default: ./homodyne_config.json)",
    )

    parser.add_argument(
        "--output-dir",
        "-o",
        default="./homodyne_results",
        help="Output directory for results (default: ./homodyne_results)",
    )

    # Enhance with new features
    parser = enhance_argument_parser(parser)

    # Create CLI enhancer
    cli_enhancer = CLIEnhancer(enable_rich=RICH_AVAILABLE)

    return parser, cli_enhancer


if __name__ == "__main__":
    # Example usage and testing
    parser, cli = create_enhanced_cli()

    # Test header
    cli.print_header("Homodyne Scattering Analysis", "Enhanced CLI Interface Test")

    # Test configuration summary
    test_config = {
        "analysis_settings": {"static_mode": True},
        "analyzer_parameters": {"start_frame": 1001, "end_frame": 2000},
        "experimental_data": {"data_folder_path": "./data/test/"},
        "optimization_config": {"angle_filtering": {"enabled": True}},
    }

    cli.print_configuration_summary(test_config)

    # Test method selection
    cli.print_method_selection(["nelder_mead", "gurobi", "wasserstein"], "classical")

    # Test error reporting
    cli.print_error_with_context(
        ValueError("Test error"),
        "Testing error display",
        ["Check your configuration file", "Verify data paths exist"],
    )

    print("\nCLI Enhancement test completed!")
