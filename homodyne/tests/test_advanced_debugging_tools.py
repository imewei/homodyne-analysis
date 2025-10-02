"""
Advanced Debugging and Diagnostic Tools
=======================================

Comprehensive debugging and diagnostic framework for Task 5.3.
Provides advanced debugging capabilities, error tracing, and diagnostic analysis.

Authors: Wei Chen, Hongrui He
Institution: Argonne National Laboratory
"""

import gc
import json
import os
import sys
import time
import traceback
import warnings
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import asdict
from dataclasses import dataclass
from functools import wraps
from pathlib import Path
from typing import Any

import numpy as np
import psutil

warnings.filterwarnings("ignore")


@dataclass
class DebugEvent:
    """Debug event data structure."""

    timestamp: float
    event_type: str
    function_name: str
    line_number: int
    filename: str
    variables: dict[str, Any]
    memory_usage: float
    execution_time: float | None = None
    error_info: str | None = None


@dataclass
class DiagnosticReport:
    """Diagnostic analysis report."""

    report_type: str
    severity: str
    description: str
    affected_components: list[str]
    recommendations: list[str]
    debugging_info: dict[str, Any]


class AdvancedDebugger:
    """Advanced debugging and tracing system."""

    def __init__(self):
        self.debug_events = []
        self.trace_enabled = False
        self.memory_profiling = False
        self.performance_profiling = False
        self.error_handlers = {}
        self.custom_breakpoints = set()

    def enable_tracing(self):
        """Enable function call tracing."""
        self.trace_enabled = True
        sys.settrace(self._trace_calls)

    def disable_tracing(self):
        """Disable function call tracing."""
        self.trace_enabled = False
        sys.settrace(None)

    def _trace_calls(self, frame, event, arg):
        """Trace function calls and returns."""
        if not self.trace_enabled:
            return None

        if event in ["call", "return", "exception"]:
            filename = frame.f_code.co_filename
            function_name = frame.f_code.co_name
            line_number = frame.f_lineno

            # Skip internal Python files
            if "/lib/python" in filename or "/site-packages/" in filename:
                return None

            # Get current memory usage
            process = psutil.Process()
            memory_usage = process.memory_info().rss / 1024 / 1024  # MB

            # Extract local variables (limited to avoid overflow)
            variables = {}
            try:
                local_vars = frame.f_locals
                for key, value in list(local_vars.items())[
                    :10
                ]:  # Limit to 10 variables
                    try:
                        # Convert to JSON-serializable format
                        if isinstance(value, (int, float, str, bool, type(None))):
                            variables[key] = value
                        elif isinstance(value, (list, tuple)) and len(value) < 10:
                            variables[key] = str(value)[
                                :100
                            ]  # Truncate long representations
                        elif hasattr(value, "__class__"):
                            variables[key] = f"<{value.__class__.__name__}>"
                        else:
                            variables[key] = str(type(value))
                    except:
                        variables[key] = "<unprintable>"
            except:
                variables = {}

            error_info = None
            if event == "exception" and arg:
                error_info = f"{arg[0].__name__}: {arg[1]}"

            debug_event = DebugEvent(
                timestamp=time.time(),
                event_type=event,
                function_name=function_name,
                line_number=line_number,
                filename=filename,
                variables=variables,
                memory_usage=memory_usage,
                error_info=error_info,
            )

            self.debug_events.append(debug_event)

            # Limit debug events to prevent memory overflow
            if len(self.debug_events) > 1000:
                self.debug_events = self.debug_events[-500:]

        return self._trace_calls

    def add_breakpoint(self, filename: str, line_number: int):
        """Add custom breakpoint."""
        self.custom_breakpoints.add((filename, line_number))

    def debug_decorator(self, func_name: str | None = None):
        """Decorator for debugging specific functions."""

        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                start_time = time.perf_counter()
                function_name = func_name or func.__name__

                # Record function entry
                entry_event = DebugEvent(
                    timestamp=time.time(),
                    event_type="function_entry",
                    function_name=function_name,
                    line_number=func.__code__.co_firstlineno,
                    filename=func.__code__.co_filename,
                    variables={"args": str(args)[:100], "kwargs": str(kwargs)[:100]},
                    memory_usage=psutil.Process().memory_info().rss / 1024 / 1024,
                )
                self.debug_events.append(entry_event)

                try:
                    result = func(*args, **kwargs)
                    execution_time = time.perf_counter() - start_time

                    # Record successful exit
                    exit_event = DebugEvent(
                        timestamp=time.time(),
                        event_type="function_exit",
                        function_name=function_name,
                        line_number=func.__code__.co_firstlineno,
                        filename=func.__code__.co_filename,
                        variables={"result_type": str(type(result))},
                        memory_usage=psutil.Process().memory_info().rss / 1024 / 1024,
                        execution_time=execution_time,
                    )
                    self.debug_events.append(exit_event)

                    return result

                except Exception as e:
                    execution_time = time.perf_counter() - start_time

                    # Record exception
                    error_event = DebugEvent(
                        timestamp=time.time(),
                        event_type="function_exception",
                        function_name=function_name,
                        line_number=func.__code__.co_firstlineno,
                        filename=func.__code__.co_filename,
                        variables={"exception": str(e)},
                        memory_usage=psutil.Process().memory_info().rss / 1024 / 1024,
                        execution_time=execution_time,
                        error_info=str(e),
                    )
                    self.debug_events.append(error_event)

                    raise

            return wrapper

        return decorator

    def get_debug_summary(self) -> dict[str, Any]:
        """Get debugging session summary."""
        if not self.debug_events:
            return {"message": "No debug events recorded"}

        # Analyze events
        function_calls = [
            e for e in self.debug_events if e.event_type in ["call", "function_entry"]
        ]
        exceptions = [
            e
            for e in self.debug_events
            if e.event_type in ["exception", "function_exception"]
        ]

        # Memory usage analysis
        memory_usage = [e.memory_usage for e in self.debug_events]
        max_memory = max(memory_usage) if memory_usage else 0
        min_memory = min(memory_usage) if memory_usage else 0

        # Function performance analysis
        function_stats = {}
        for event in self.debug_events:
            if event.execution_time is not None:
                func_name = event.function_name
                if func_name not in function_stats:
                    function_stats[func_name] = {
                        "call_count": 0,
                        "total_time": 0,
                        "avg_time": 0,
                        "max_time": 0,
                    }

                function_stats[func_name]["call_count"] += 1
                function_stats[func_name]["total_time"] += event.execution_time
                function_stats[func_name]["max_time"] = max(
                    function_stats[func_name]["max_time"], event.execution_time
                )
                function_stats[func_name]["avg_time"] = (
                    function_stats[func_name]["total_time"]
                    / function_stats[func_name]["call_count"]
                )

        return {
            "total_events": len(self.debug_events),
            "function_calls": len(function_calls),
            "exceptions": len(exceptions),
            "memory_usage": {
                "max_mb": max_memory,
                "min_mb": min_memory,
                "range_mb": max_memory - min_memory,
            },
            "function_performance": function_stats,
            "recent_exceptions": [
                {
                    "function": e.function_name,
                    "error": e.error_info,
                    "timestamp": e.timestamp,
                }
                for e in exceptions[-5:]  # Last 5 exceptions
            ],
        }


class DiagnosticAnalyzer:
    """Advanced diagnostic analysis system."""

    def __init__(self):
        self.diagnostic_reports = []
        self.system_metrics = {}

    def analyze_performance_bottlenecks(self) -> list[DiagnosticReport]:
        """Analyze performance bottlenecks."""
        reports = []

        # Test various performance scenarios
        bottleneck_tests = [
            self._test_cpu_intensive_operations,
            self._test_memory_intensive_operations,
            self._test_io_operations,
            self._test_algorithm_complexity,
        ]

        for test_func in bottleneck_tests:
            try:
                test_result = test_func()
                if test_result["issues"]:
                    report = DiagnosticReport(
                        report_type="performance_bottleneck",
                        severity=test_result["severity"],
                        description=test_result["description"],
                        affected_components=test_result["components"],
                        recommendations=test_result["recommendations"],
                        debugging_info=test_result["debug_info"],
                    )
                    reports.append(report)
            except Exception as e:
                error_report = DiagnosticReport(
                    report_type="diagnostic_error",
                    severity="high",
                    description=f"Failed to run diagnostic test: {e}",
                    affected_components=["diagnostic_system"],
                    recommendations=["Review diagnostic test implementation"],
                    debugging_info={"error": str(e)},
                )
                reports.append(error_report)

        return reports

    def _test_cpu_intensive_operations(self) -> dict[str, Any]:
        """Test CPU-intensive operations."""
        import numpy as np

        # Test matrix operations
        size = 300
        A = np.random.rand(size, size)
        B = np.random.rand(size, size)

        start_time = time.perf_counter()
        A @ B
        execution_time = time.perf_counter() - start_time

        # Performance threshold
        max_time = 1.0  # 1 second for 300x300 matrix multiplication

        issues = execution_time > max_time
        severity = (
            "high" if execution_time > max_time * 2 else "medium" if issues else "low"
        )

        return {
            "issues": issues,
            "severity": severity,
            "description": f"Matrix multiplication took {execution_time:.3f}s (threshold: {max_time}s)",
            "components": ["numpy", "linear_algebra"],
            "recommendations": (
                [
                    "Use optimized BLAS libraries",
                    "Consider using smaller matrix sizes",
                    "Implement chunked processing for large matrices",
                ]
                if issues
                else []
            ),
            "debug_info": {
                "execution_time": execution_time,
                "matrix_size": size,
                "threshold": max_time,
            },
        }

    def _test_memory_intensive_operations(self) -> dict[str, Any]:
        """Test memory-intensive operations."""
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024

        # Create large data structures
        large_arrays = []
        for i in range(5):
            large_arrays.append(np.random.rand(1000, 1000))

        peak_memory = process.memory_info().rss / 1024 / 1024
        memory_increase = peak_memory - initial_memory

        # Clean up
        del large_arrays
        gc.collect()

        final_memory = process.memory_info().rss / 1024 / 1024
        memory_released = peak_memory - final_memory

        # Memory efficiency threshold
        efficiency_threshold = 0.7
        memory_efficiency = (
            memory_released / memory_increase if memory_increase > 0 else 1.0
        )

        issues = memory_efficiency < efficiency_threshold
        severity = "high" if memory_efficiency < 0.5 else "medium" if issues else "low"

        return {
            "issues": issues,
            "severity": severity,
            "description": f"Memory efficiency: {memory_efficiency:.2f} (threshold: {efficiency_threshold})",
            "components": ["memory_management", "garbage_collection"],
            "recommendations": (
                [
                    "Review memory allocation patterns",
                    "Implement explicit memory cleanup",
                    "Use memory pooling for large objects",
                ]
                if issues
                else []
            ),
            "debug_info": {
                "initial_memory_mb": initial_memory,
                "peak_memory_mb": peak_memory,
                "final_memory_mb": final_memory,
                "memory_efficiency": memory_efficiency,
            },
        }

    def _test_io_operations(self) -> dict[str, Any]:
        """Test I/O operations."""
        import tempfile

        # Test file I/O performance
        data = "test data " * 10000  # ~90KB of data

        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            temp_filename = f.name

        try:
            # Write test
            start_time = time.perf_counter()
            with open(temp_filename, "w") as f:
                f.write(data)
            write_time = time.perf_counter() - start_time

            # Read test
            start_time = time.perf_counter()
            with open(temp_filename) as f:
                f.read()
            read_time = time.perf_counter() - start_time

            # Performance thresholds
            max_write_time = 0.1  # 100ms
            max_read_time = 0.1  # 100ms

            write_issues = write_time > max_write_time
            read_issues = read_time > max_read_time
            issues = write_issues or read_issues

            severity = (
                "high"
                if (write_time > max_write_time * 2 or read_time > max_read_time * 2)
                else "medium" if issues else "low"
            )

            return {
                "issues": issues,
                "severity": severity,
                "description": f"I/O performance - Write: {write_time:.3f}s, Read: {read_time:.3f}s",
                "components": ["file_io", "disk_operations"],
                "recommendations": (
                    [
                        "Use buffered I/O for large files",
                        "Consider asynchronous I/O operations",
                        "Implement file caching strategies",
                    ]
                    if issues
                    else []
                ),
                "debug_info": {
                    "write_time": write_time,
                    "read_time": read_time,
                    "data_size": len(data),
                    "write_threshold": max_write_time,
                    "read_threshold": max_read_time,
                },
            }

        finally:
            # Cleanup
            try:
                os.unlink(temp_filename)
            except:
                pass

    def _test_algorithm_complexity(self) -> dict[str, Any]:
        """Test algorithm complexity."""
        # Test sorting algorithm performance
        import random

        sizes = [1000, 2000, 4000]
        times = []

        for size in sizes:
            data = [random.random() for _ in range(size)]

            start_time = time.perf_counter()
            sorted(data)
            execution_time = time.perf_counter() - start_time
            times.append(execution_time)

        # Check scaling (should be roughly O(n log n) for good sorting)
        scaling_factor = times[-1] / times[0] if times[0] > 0 else float("inf")
        expected_scaling = (sizes[-1] / sizes[0]) * np.log(sizes[-1] / sizes[0])

        complexity_efficiency = (
            scaling_factor / expected_scaling if expected_scaling > 0 else float("inf")
        )

        # Complexity threshold
        max_complexity_ratio = 2.0
        issues = complexity_efficiency > max_complexity_ratio

        severity = (
            "high"
            if complexity_efficiency > max_complexity_ratio * 2
            else "medium" if issues else "low"
        )

        return {
            "issues": issues,
            "severity": severity,
            "description": f"Algorithm complexity ratio: {complexity_efficiency:.2f} (threshold: {max_complexity_ratio})",
            "components": ["sorting_algorithms", "algorithm_complexity"],
            "recommendations": (
                [
                    "Review algorithm choices for better complexity",
                    "Consider using specialized data structures",
                    "Implement algorithmic optimizations",
                ]
                if issues
                else []
            ),
            "debug_info": {
                "scaling_factor": scaling_factor,
                "expected_scaling": expected_scaling,
                "complexity_efficiency": complexity_efficiency,
                "execution_times": times,
                "data_sizes": sizes,
            },
        }

    def analyze_error_patterns(
        self, debug_events: list[DebugEvent]
    ) -> list[DiagnosticReport]:
        """Analyze error patterns from debug events."""
        reports = []

        # Find exceptions
        exceptions = [e for e in debug_events if e.error_info]

        if not exceptions:
            return reports

        # Group exceptions by type
        exception_groups = {}
        for event in exceptions:
            error_type = (
                event.error_info.split(":")[0]
                if ":" in event.error_info
                else event.error_info
            )
            if error_type not in exception_groups:
                exception_groups[error_type] = []
            exception_groups[error_type].append(event)

        # Analyze each exception group
        for error_type, events in exception_groups.items():
            if len(events) > 1:  # Multiple occurrences indicate a pattern
                affected_functions = list({e.function_name for e in events})

                report = DiagnosticReport(
                    report_type="error_pattern",
                    severity="high" if len(events) > 5 else "medium",
                    description=f"Recurring {error_type} error ({len(events)} occurrences)",
                    affected_components=affected_functions,
                    recommendations=[
                        f"Review error handling for {error_type}",
                        "Add input validation",
                        "Implement graceful error recovery",
                    ],
                    debugging_info={
                        "error_type": error_type,
                        "occurrence_count": len(events),
                        "affected_functions": affected_functions,
                        "timestamps": [e.timestamp for e in events],
                    },
                )
                reports.append(report)

        return reports

    def analyze_memory_leaks(
        self, debug_events: list[DebugEvent]
    ) -> list[DiagnosticReport]:
        """Analyze potential memory leaks."""
        reports = []

        if len(debug_events) < 10:
            return reports

        # Analyze memory usage trend
        memory_usage = [e.memory_usage for e in debug_events]

        # Simple linear trend analysis
        x = list(range(len(memory_usage)))
        n = len(x)
        sum_x = sum(x)
        sum_y = sum(memory_usage)
        sum_xy = sum(x[i] * memory_usage[i] for i in range(n))
        sum_x2 = sum(x[i] ** 2 for i in range(n))

        # Calculate slope (memory growth rate)
        slope = (
            (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x**2)
            if (n * sum_x2 - sum_x**2) != 0
            else 0
        )

        # Memory leak threshold (MB per event)
        leak_threshold = 0.1

        if slope > leak_threshold:
            report = DiagnosticReport(
                report_type="memory_leak",
                severity="high" if slope > leak_threshold * 2 else "medium",
                description=f"Potential memory leak detected (growth rate: {slope:.3f} MB/event)",
                affected_components=["memory_management"],
                recommendations=[
                    "Review object lifecycle management",
                    "Check for circular references",
                    "Implement explicit cleanup procedures",
                    "Use memory profiling tools for detailed analysis",
                ],
                debugging_info={
                    "memory_growth_rate": slope,
                    "initial_memory": memory_usage[0],
                    "final_memory": memory_usage[-1],
                    "total_growth": memory_usage[-1] - memory_usage[0],
                    "event_count": len(debug_events),
                },
            )
            reports.append(report)

        return reports


class ErrorTracker:
    """Advanced error tracking and analysis."""

    def __init__(self):
        self.error_log = []
        self.error_handlers = {}

    def register_error_handler(self, error_type: type, handler: Callable):
        """Register custom error handler."""
        self.error_handlers[error_type] = handler

    @contextmanager
    def error_context(self, context_name: str):
        """Context manager for error tracking."""
        start_time = time.time()
        try:
            yield
        except Exception as e:
            self._log_error(e, context_name, start_time)

            # Call custom handler if available
            error_type = type(e)
            if error_type in self.error_handlers:
                try:
                    self.error_handlers[error_type](e)
                except:
                    pass  # Don't let handler errors break the original flow

            raise

    def _log_error(self, error: Exception, context: str, start_time: float):
        """Log error with context information."""
        error_info = {
            "timestamp": time.time(),
            "context": context,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "traceback": traceback.format_exc(),
            "execution_time": time.time() - start_time,
            "memory_usage": psutil.Process().memory_info().rss / 1024 / 1024,
        }

        self.error_log.append(error_info)

    def get_error_analysis(self) -> dict[str, Any]:
        """Get comprehensive error analysis."""
        if not self.error_log:
            return {"message": "No errors logged"}

        # Group errors by type
        error_types = {}
        for error in self.error_log:
            error_type = error["error_type"]
            if error_type not in error_types:
                error_types[error_type] = []
            error_types[error_type].append(error)

        # Calculate statistics
        total_errors = len(self.error_log)
        unique_error_types = len(error_types)

        # Find most common errors
        common_errors = sorted(
            error_types.items(), key=lambda x: len(x[1]), reverse=True
        )[:5]

        # Recent errors
        recent_errors = sorted(
            self.error_log, key=lambda x: x["timestamp"], reverse=True
        )[:10]

        return {
            "total_errors": total_errors,
            "unique_error_types": unique_error_types,
            "most_common_errors": [
                {
                    "error_type": error_type,
                    "count": len(errors),
                    "percentage": (len(errors) / total_errors) * 100,
                }
                for error_type, errors in common_errors
            ],
            "recent_errors": [
                {
                    "timestamp": error["timestamp"],
                    "context": error["context"],
                    "error_type": error["error_type"],
                    "message": error["error_message"][:100],  # Truncate long messages
                }
                for error in recent_errors
            ],
            "error_contexts": list({error["context"] for error in self.error_log}),
        }


def run_advanced_debugging_tools():
    """Main function to run advanced debugging tools."""
    print("Advanced Debugging and Diagnostic Tools - Task 5.3")
    print("=" * 65)

    # Create debugging and diagnostic systems
    debugger = AdvancedDebugger()
    analyzer = DiagnosticAnalyzer()
    error_tracker = ErrorTracker()

    print("Setting up debugging environment...")

    # Test debugging with sample functions
    @debugger.debug_decorator("sample_function")
    def sample_function(x, y):
        """Sample function for debugging."""
        if x < 0:
            raise ValueError("x must be non-negative")
        return x**2 + y**2

    @debugger.debug_decorator("matrix_operation")
    def matrix_operation():
        """Sample matrix operation."""
        import numpy as np

        A = np.random.rand(100, 100)
        B = np.random.rand(100, 100)
        return A @ B

    # Enable tracing for a short period
    print("Testing debugging capabilities...")

    # Test normal execution
    try:
        sample_function(3, 4)
        matrix_operation()
    except Exception as e:
        print(f"Unexpected error: {e}")

    # Test error handling
    with error_tracker.error_context("error_test"):
        try:
            sample_function(-1, 2)  # This should raise an error
        except ValueError:
            pass  # Expected error

    # Run diagnostic analysis
    print("Running diagnostic analysis...")
    performance_reports = analyzer.analyze_performance_bottlenecks()
    error_pattern_reports = analyzer.analyze_error_patterns(debugger.debug_events)
    memory_leak_reports = analyzer.analyze_memory_leaks(debugger.debug_events)

    # Get summaries
    debug_summary = debugger.get_debug_summary()
    error_analysis = error_tracker.get_error_analysis()

    # Compile comprehensive report
    comprehensive_report = {
        "debugging_summary": debug_summary,
        "error_analysis": error_analysis,
        "diagnostic_reports": {
            "performance_bottlenecks": [
                asdict(report) for report in performance_reports
            ],
            "error_patterns": [asdict(report) for report in error_pattern_reports],
            "memory_leaks": [asdict(report) for report in memory_leak_reports],
        },
        "summary_statistics": {
            "total_debug_events": len(debugger.debug_events),
            "total_diagnostic_reports": len(performance_reports)
            + len(error_pattern_reports)
            + len(memory_leak_reports),
            "high_severity_issues": len(
                [
                    r
                    for r in performance_reports
                    + error_pattern_reports
                    + memory_leak_reports
                    if r.severity == "high"
                ]
            ),
            "total_errors_tracked": len(error_tracker.error_log),
        },
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    # Display summary
    stats = comprehensive_report["summary_statistics"]
    print("\nDEBUGGING AND DIAGNOSTIC SUMMARY:")
    print(f"  Debug Events Captured: {stats['total_debug_events']}")
    print(f"  Diagnostic Reports Generated: {stats['total_diagnostic_reports']}")
    print(f"  High Severity Issues: {stats['high_severity_issues']}")
    print(f"  Errors Tracked: {stats['total_errors_tracked']}")

    # Display diagnostic results
    if performance_reports:
        print("\nPERFORMANCE BOTTLENECKS:")
        for report in performance_reports:
            print(f"  • {report.description} (Severity: {report.severity})")

    if error_pattern_reports:
        print("\nERROR PATTERNS:")
        for report in error_pattern_reports:
            print(f"  • {report.description} (Severity: {report.severity})")

    if memory_leak_reports:
        print("\nMEMORY ISSUES:")
        for report in memory_leak_reports:
            print(f"  • {report.description} (Severity: {report.severity})")

    # Save results
    results_dir = Path("debugging_tools_results")
    results_dir.mkdir(exist_ok=True)

    results_file = results_dir / "task_5_3_advanced_debugging_report.json"
    with open(results_file, "w") as f:
        json.dump(comprehensive_report, f, indent=2)

    print(f"\n📄 Debugging report saved to: {results_file}")
    print("✅ Task 5.3 Advanced Debugging Tools Complete!")
    print(f"🔍 {stats['total_debug_events']} debug events captured")
    print(f"📋 {stats['total_diagnostic_reports']} diagnostic reports generated")
    print(f"⚠️  {stats['high_severity_issues']} high severity issues identified")

    return comprehensive_report


if __name__ == "__main__":
    run_advanced_debugging_tools()
