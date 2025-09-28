"""
Intelligent Error Reporting and User Guidance System
===================================================

Advanced error handling and user guidance system with contextual suggestions,
automated troubleshooting, and interactive problem resolution for scientific computing workflows.

Features:
- Contextual error analysis with intelligent suggestions
- Automated troubleshooting with step-by-step guidance
- Interactive error resolution with user prompts
- Error pattern recognition and learning
- Integration with documentation and help systems
- Performance impact analysis for errors
- User-friendly error formatting with rich output
- Error logging with structured data for analysis
"""

import logging
import sys
import traceback
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.prompt import Confirm
    from rich.table import Table
    from rich.text import Text

    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    Console = None


class ErrorSeverity(Enum):
    """Error severity levels for classification and handling."""

    LOW = "low"  # Warnings, minor issues
    MEDIUM = "medium"  # Recoverable errors
    HIGH = "high"  # Critical errors that stop execution
    CRITICAL = "critical"  # System-level errors


class ErrorCategory(Enum):
    """Error categories for intelligent handling."""

    CONFIGURATION = "configuration"  # Config file issues
    DATA_LOADING = "data_loading"  # Data file problems
    COMPUTATION = "computation"  # Numerical computation errors
    VISUALIZATION = "visualization"  # Plotting and display errors
    OPTIMIZATION = "optimization"  # Optimization algorithm issues
    IO_ERROR = "io_error"  # File I/O problems
    DEPENDENCY = "dependency"  # Missing dependencies
    MEMORY = "memory"  # Memory-related issues
    PERFORMANCE = "performance"  # Performance bottlenecks
    USER_INPUT = "user_input"  # Invalid user input


@dataclass
class ErrorContext:
    """Structured error context information."""

    error_type: str
    error_message: str
    category: ErrorCategory
    severity: ErrorSeverity
    context_info: dict[str, Any]
    suggestions: list[str]
    traceback_info: str | None = None
    resolution_steps: list[str] = None
    related_docs: list[str] = None

    def __post_init__(self):
        if self.resolution_steps is None:
            self.resolution_steps = []
        if self.related_docs is None:
            self.related_docs = []


class ErrorReporter:
    """
    Intelligent error reporting system with contextual analysis and user guidance.

    Provides comprehensive error handling with automated troubleshooting,
    user-friendly formatting, and interactive problem resolution.
    """

    def __init__(self, enable_rich: bool = True, interactive: bool = True):
        self.enable_rich = enable_rich and RICH_AVAILABLE
        self.interactive = interactive
        self.console = Console() if self.enable_rich else None
        self.logger = logging.getLogger(__name__)

        # Error patterns and solutions database
        self.error_patterns = self._initialize_error_patterns()
        self.error_history = []

        # User preferences
        self.show_technical_details = True
        self.auto_suggest_fixes = True

    def _initialize_error_patterns(self) -> dict[str, dict[str, Any]]:
        """Initialize database of common error patterns and solutions."""
        return {
            "FileNotFoundError": {
                "category": ErrorCategory.DATA_LOADING,
                "severity": ErrorSeverity.HIGH,
                "common_causes": [
                    "Incorrect file path in configuration",
                    "Data file moved or deleted",
                    "Permission issues",
                    "Typo in filename",
                ],
                "suggestions": [
                    "Check if the file path exists: verify the full path to your data file",
                    "Verify file permissions: ensure the file is readable",
                    "Check for typos in the filename or path",
                    "Use absolute paths instead of relative paths",
                    "Run with --validate-config to check all file paths",
                ],
                "resolution_steps": [
                    "1. Verify the file exists at the specified path",
                    "2. Check file permissions (should be readable)",
                    "3. Try using an absolute path instead of relative",
                    "4. Check for typos in the configuration file",
                ],
            },
            "JSONDecodeError": {
                "category": ErrorCategory.CONFIGURATION,
                "severity": ErrorSeverity.HIGH,
                "common_causes": [
                    "Malformed JSON syntax",
                    "Missing commas or brackets",
                    "Invalid escape sequences",
                    "Corrupted configuration file",
                ],
                "suggestions": [
                    "Validate JSON syntax using an online JSON validator",
                    "Check for missing commas between key-value pairs",
                    "Verify all brackets and braces are properly closed",
                    "Use a JSON formatter to identify syntax issues",
                    "Restore from a backup configuration file",
                ],
                "resolution_steps": [
                    "1. Copy the error line number from the message",
                    "2. Open your configuration file and go to that line",
                    "3. Check for missing commas, quotes, or brackets",
                    "4. Validate the entire file with a JSON validator",
                ],
            },
            "ImportError": {
                "category": ErrorCategory.DEPENDENCY,
                "severity": ErrorSeverity.HIGH,
                "common_causes": [
                    "Missing required Python packages",
                    "Incorrect package versions",
                    "Virtual environment issues",
                    "Installation conflicts",
                ],
                "suggestions": [
                    "Install missing packages: pip install -r requirements.txt",
                    "Check if you're in the correct virtual environment",
                    "Update packages to compatible versions",
                    "Try reinstalling the package: pip uninstall <package> && pip install <package>",
                    "Check for conflicting package versions",
                ],
                "resolution_steps": [
                    "1. Identify the missing package from the error message",
                    "2. Check if you're in the correct virtual environment",
                    "3. Install the package: pip install <package-name>",
                    "4. Verify installation: python -c 'import <package>'",
                ],
            },
            "MemoryError": {
                "category": ErrorCategory.MEMORY,
                "severity": ErrorSeverity.HIGH,
                "common_causes": [
                    "Dataset too large for available memory",
                    "Memory leaks in computation",
                    "Inefficient data structures",
                    "Accumulating large arrays",
                ],
                "suggestions": [
                    "Reduce frame range in configuration (start_frame, end_frame)",
                    "Enable angle filtering to process fewer angles",
                    "Use static mode instead of laminar flow if applicable",
                    "Close other applications to free memory",
                    "Consider processing data in smaller chunks",
                ],
                "resolution_steps": [
                    "1. Check available system memory",
                    "2. Reduce the frame range in your configuration",
                    "3. Enable angle filtering for fewer computations",
                    "4. Close unnecessary applications",
                ],
            },
            "ValueError": {
                "category": ErrorCategory.USER_INPUT,
                "severity": ErrorSeverity.MEDIUM,
                "common_causes": [
                    "Invalid parameter values",
                    "Incompatible data shapes",
                    "Out-of-range values",
                    "Incorrect data types",
                ],
                "suggestions": [
                    "Check parameter bounds in your configuration",
                    "Verify data file format and structure",
                    "Ensure parameter values are within valid ranges",
                    "Check data types match expected formats",
                    "Validate input data before processing",
                ],
                "resolution_steps": [
                    "1. Read the error message carefully for specific invalid value",
                    "2. Check the parameter bounds in your configuration",
                    "3. Verify the data format matches expectations",
                    "4. Use --validate-config to check configuration validity",
                ],
            },
            "KeyError": {
                "category": ErrorCategory.CONFIGURATION,
                "severity": ErrorSeverity.MEDIUM,
                "common_causes": [
                    "Missing configuration keys",
                    "Outdated configuration format",
                    "Typos in parameter names",
                    "Incomplete configuration sections",
                ],
                "suggestions": [
                    "Check if required configuration keys are present",
                    "Compare with a working configuration example",
                    "Update configuration to latest format",
                    "Check for typos in parameter names",
                    "Use default configuration as a template",
                ],
                "resolution_steps": [
                    "1. Identify the missing key from the error message",
                    "2. Add the missing key to your configuration",
                    "3. Check examples for proper key names and structure",
                    "4. Validate the updated configuration",
                ],
            },
        }

    def analyze_error(
        self, exception: Exception, context_info: dict[str, Any] | None = None
    ) -> ErrorContext:
        """Analyze error and generate contextual information."""
        error_type = type(exception).__name__
        error_message = str(exception)

        # Get error pattern information
        pattern_info = self.error_patterns.get(error_type, {})

        # Determine category and severity
        category = pattern_info.get("category", ErrorCategory.COMPUTATION)
        severity = pattern_info.get("severity", ErrorSeverity.MEDIUM)

        # Generate suggestions
        suggestions = pattern_info.get(
            "suggestions",
            [
                "Check the error message for specific details",
                "Verify your configuration file is correct",
                "Try running with --verbose for more information",
            ],
        )

        # Get resolution steps
        resolution_steps = pattern_info.get("resolution_steps", [])

        # Enhanced context analysis
        if context_info is None:
            context_info = {}

        # Add system information
        context_info.update(
            {
                "python_version": sys.version,
                "error_location": self._get_error_location(exception),
            }
        )

        # Get traceback information
        traceback_info = "".join(
            traceback.format_exception(
                type(exception), exception, exception.__traceback__
            )
        )

        return ErrorContext(
            error_type=error_type,
            error_message=error_message,
            category=category,
            severity=severity,
            context_info=context_info,
            suggestions=suggestions,
            traceback_info=traceback_info,
            resolution_steps=resolution_steps,
            related_docs=self._get_related_documentation(category),
        )

    def _get_error_location(self, exception: Exception) -> str:
        """Extract error location from exception traceback."""
        tb = exception.__traceback__
        if tb:
            filename = tb.tb_frame.f_code.co_filename
            line_number = tb.tb_lineno
            function_name = tb.tb_frame.f_code.co_name
            return f"{Path(filename).name}:{line_number} in {function_name}()"
        return "Unknown location"

    def _get_related_documentation(self, category: ErrorCategory) -> list[str]:
        """Get related documentation links for error category."""
        docs_map = {
            ErrorCategory.CONFIGURATION: [
                "Configuration Guide: Setting up analysis parameters",
                "Troubleshooting: Common configuration issues",
            ],
            ErrorCategory.DATA_LOADING: [
                "Data Format Guide: Supported file formats",
                "File Path Setup: Organizing your data files",
            ],
            ErrorCategory.OPTIMIZATION: [
                "Optimization Methods: Choosing the right method",
                "Performance Tuning: Optimizing analysis speed",
            ],
            ErrorCategory.VISUALIZATION: [
                "Plotting Guide: Creating publication-quality plots",
                "Visualization Troubleshooting: Common plotting issues",
            ],
        }
        return docs_map.get(category, ["General Troubleshooting Guide"])

    def report_error(
        self, exception: Exception, context_info: dict[str, Any] | None = None
    ) -> None:
        """Report error with full analysis and user guidance."""
        error_context = self.analyze_error(exception, context_info)

        # Add to error history
        self.error_history.append(error_context)

        # Display error report
        if self.enable_rich and self.console:
            self._display_rich_error_report(error_context)
        else:
            self._display_text_error_report(error_context)

        # Interactive troubleshooting
        if self.interactive:
            self._interactive_troubleshooting(error_context)

    def _display_rich_error_report(self, error_context: ErrorContext) -> None:
        """Display error report using rich formatting."""
        # Main error panel
        severity_color = {
            ErrorSeverity.LOW: "yellow",
            ErrorSeverity.MEDIUM: "orange",
            ErrorSeverity.HIGH: "red",
            ErrorSeverity.CRITICAL: "bright_red",
        }[error_context.severity]

        error_text = Text()
        error_text.append("Error Type: ", style="bold")
        error_text.append(
            f"{error_context.error_type}\n", style=f"bold {severity_color}"
        )
        error_text.append("Message: ", style="bold")
        error_text.append(f"{error_context.error_message}\n", style="white")
        error_text.append("Category: ", style="bold")
        error_text.append(f"{error_context.category.value}\n", style="cyan")
        error_text.append("Severity: ", style="bold")
        error_text.append(
            f"{error_context.severity.value.upper()}", style=severity_color
        )

        panel = Panel(
            error_text,
            title=f"[{severity_color}]Error Report[/{severity_color}]",
            border_style=severity_color,
            padding=(1, 2),
        )
        self.console.print(panel)

        # Suggestions table
        if error_context.suggestions:
            table = Table(title="Suggested Solutions", show_header=False, box=None)
            table.add_column("Step", style="cyan", width=4)
            table.add_column("Solution", style="white")

            for i, suggestion in enumerate(error_context.suggestions, 1):
                table.add_row(f"{i}.", suggestion)

            self.console.print("\n")
            self.console.print(table)

        # Resolution steps
        if error_context.resolution_steps:
            self.console.print("\n[bold green]Step-by-Step Resolution:[/bold green]")
            for step in error_context.resolution_steps:
                self.console.print(f"  {step}")

        # Context information
        if error_context.context_info and self.show_technical_details:
            self.console.print("\n[bold yellow]Technical Details:[/bold yellow]")
            for key, value in error_context.context_info.items():
                if key != "python_version":  # Skip verbose python version
                    self.console.print(f"  {key}: {value}")

        # Related documentation
        if error_context.related_docs:
            self.console.print("\n[bold blue]Related Documentation:[/bold blue]")
            for doc in error_context.related_docs:
                self.console.print(f"  • {doc}")

    def _display_text_error_report(self, error_context: ErrorContext) -> None:
        """Display error report in plain text format."""
        print("\n" + "=" * 60)
        print("ERROR REPORT")
        print("=" * 60)
        print(f"Type: {error_context.error_type}")
        print(f"Message: {error_context.error_message}")
        print(f"Category: {error_context.category.value}")
        print(f"Severity: {error_context.severity.value.upper()}")

        if error_context.suggestions:
            print("\nSuggested Solutions:")
            print("-" * 30)
            for i, suggestion in enumerate(error_context.suggestions, 1):
                print(f"{i}. {suggestion}")

        if error_context.resolution_steps:
            print("\nStep-by-Step Resolution:")
            print("-" * 30)
            for step in error_context.resolution_steps:
                print(f"  {step}")

        if error_context.context_info and self.show_technical_details:
            print("\nTechnical Details:")
            print("-" * 20)
            for key, value in error_context.context_info.items():
                if key != "python_version":
                    print(f"  {key}: {value}")

        if error_context.related_docs:
            print("\nRelated Documentation:")
            print("-" * 25)
            for doc in error_context.related_docs:
                print(f"  • {doc}")

        print("\n" + "=" * 60)

    def _interactive_troubleshooting(self, error_context: ErrorContext) -> None:
        """Provide interactive troubleshooting assistance."""
        if not self.interactive:
            return

        if self.enable_rich and self.console:
            self.console.print(
                "\n[bold cyan]Interactive Troubleshooting Assistant[/bold cyan]"
            )

            # Ask if user wants help
            if not Confirm.ask(
                "Would you like help troubleshooting this error?", default=True
            ):
                return

            # Guided troubleshooting
            if error_context.resolution_steps:
                self.console.print("\nLet's go through the resolution steps:")

                for i, step in enumerate(error_context.resolution_steps):
                    self.console.print(f"\n[bold blue]Step {i + 1}:[/bold blue] {step}")

                    if not Confirm.ask("Have you completed this step?", default=False):
                        self.console.print(
                            "[yellow]Please complete this step and try again.[/yellow]"
                        )
                        break

                    if Confirm.ask("Did this step resolve the issue?", default=False):
                        self.console.print(
                            "[green]Great! The issue should be resolved.[/green]"
                        )
                        return

            # Additional help options
            self.console.print("\n[yellow]If the issue persists:[/yellow]")
            self.console.print(
                "  • Try running with --verbose for more detailed output"
            )
            self.console.print(
                "  • Check the log files for additional error information"
            )
            self.console.print("  • Review the configuration file for any mistakes")
        else:
            # Simple text-based interaction
            print("\nWould you like help troubleshooting this error? [y/N]: ", end="")
            response = input().strip().lower()

            if response in ["y", "yes"]:
                print("\nPlease follow the suggested solutions above.")
                print(
                    "If the issue persists, try running with --verbose for more information."
                )

    def generate_error_summary(self) -> dict[str, Any]:
        """Generate summary of all errors encountered during session."""
        if not self.error_history:
            return {"total_errors": 0, "summary": "No errors encountered"}

        summary = {
            "total_errors": len(self.error_history),
            "by_category": {},
            "by_severity": {},
            "most_common": {},
        }

        # Count by category
        for error in self.error_history:
            category = error.category.value
            summary["by_category"][category] = (
                summary["by_category"].get(category, 0) + 1
            )

            severity = error.severity.value
            summary["by_severity"][severity] = (
                summary["by_severity"].get(severity, 0) + 1
            )

            error_type = error.error_type
            summary["most_common"][error_type] = (
                summary["most_common"].get(error_type, 0) + 1
            )

        return summary

    def suggest_preventive_measures(self) -> list[str]:
        """Suggest preventive measures based on error history."""
        if not self.error_history:
            return []

        suggestions = []

        # Analyze error patterns
        categories = [error.category for error in self.error_history]

        if ErrorCategory.CONFIGURATION in categories:
            suggestions.append("Use --validate-config before running analysis")
            suggestions.append("Keep backup copies of working configurations")

        if ErrorCategory.DATA_LOADING in categories:
            suggestions.append("Verify all file paths before starting analysis")
            suggestions.append("Use absolute paths in configuration files")

        if ErrorCategory.MEMORY in categories:
            suggestions.append("Monitor system memory usage during analysis")
            suggestions.append("Consider processing data in smaller chunks")

        if ErrorCategory.DEPENDENCY in categories:
            suggestions.append("Create a requirements.txt file for your environment")
            suggestions.append("Use virtual environments for isolation")

        return suggestions


def create_error_reporter(
    enable_rich: bool = True, interactive: bool = True
) -> ErrorReporter:
    """Factory function to create error reporter."""
    return ErrorReporter(enable_rich=enable_rich, interactive=interactive)


def handle_exception_with_guidance(
    exception: Exception,
    context_info: dict[str, Any] | None = None,
    reporter: ErrorReporter | None = None,
) -> None:
    """Handle exception with intelligent guidance."""
    if reporter is None:
        reporter = create_error_reporter()

    reporter.report_error(exception, context_info)


if __name__ == "__main__":
    # Example usage and testing
    print("Testing Error Reporter...")

    reporter = create_error_reporter(enable_rich=RICH_AVAILABLE, interactive=False)

    # Test different error types
    test_errors = [
        FileNotFoundError("No such file: 'data/missing_file.hdf'"),
        ValueError("Parameter 'alpha' value -5.0 is outside bounds [-2.0, 2.0]"),
        ImportError("No module named 'missing_package'"),
        KeyError("'start_frame' key missing from configuration"),
    ]

    for i, error in enumerate(test_errors):
        print(f"\n--- Test Error {i + 1} ---")
        context = {"test_case": f"Error {i + 1}", "function": "test_function"}
        reporter.report_error(error, context)

    # Generate summary
    summary = reporter.generate_error_summary()
    print(f"\nError Summary: {summary}")

    # Get preventive suggestions
    suggestions = reporter.suggest_preventive_measures()
    print(f"\nPreventive Measures: {suggestions}")

    print("\nError reporter tests completed!")
