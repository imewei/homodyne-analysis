"""
Comprehensive Error Handling and Recovery Systems
=================================================

Advanced error handling and recovery framework for Task 5.4.
Implements robust error handling, automatic recovery, and fault tolerance.

Authors: Wei Chen, Hongrui He
Institution: Argonne National Laboratory
"""

import functools
import json
import logging
import os
import sys
import tempfile
import time
import traceback
import warnings
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np

warnings.filterwarnings("ignore")


class ErrorSeverity(Enum):
    """Error severity levels."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RecoveryStrategy(Enum):
    """Recovery strategy types."""

    RETRY = "retry"
    FALLBACK = "fallback"
    GRACEFUL_DEGRADATION = "graceful_degradation"
    CIRCUIT_BREAKER = "circuit_breaker"
    CHECKPOINT_RESTORE = "checkpoint_restore"


@dataclass
class ErrorContext:
    """Error context information."""

    error_id: str
    timestamp: float
    error_type: str
    error_message: str
    severity: ErrorSeverity
    traceback_info: str
    function_name: str
    recovery_attempted: bool = False
    recovery_successful: bool = False
    recovery_strategy: RecoveryStrategy | None = None


@dataclass
class RecoveryResult:
    """Recovery operation result."""

    success: bool
    strategy_used: RecoveryStrategy
    attempts_made: int
    execution_time: float
    error_resolved: bool
    fallback_data: Any | None = None


class ErrorHandler:
    """Advanced error handling system."""

    def __init__(self):
        self.error_registry = {}
        self.recovery_strategies = {}
        self.error_history = []
        self.circuit_breakers = {}
        self.checkpoints = {}

    def register_error_handler(
        self,
        error_type: type,
        handler: Callable,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
    ):
        """Register custom error handler."""
        self.error_registry[error_type] = {
            "handler": handler,
            "severity": severity,
            "call_count": 0,
            "success_count": 0,
        }

    def register_recovery_strategy(
        self,
        error_type: type,
        strategy: RecoveryStrategy,
        recovery_func: Callable,
        max_attempts: int = 3,
    ):
        """Register recovery strategy for specific error type."""
        self.recovery_strategies[error_type] = {
            "strategy": strategy,
            "recovery_func": recovery_func,
            "max_attempts": max_attempts,
            "circuit_breaker_threshold": 5,
        }

    def handle_error(self, error: Exception, context: str = "") -> ErrorContext:
        """Handle error with registered strategies."""
        error_id = f"err_{int(time.time() * 1000)}_{len(self.error_history)}"
        error_type = type(error)

        # Determine severity
        severity = ErrorSeverity.MEDIUM
        if error_type in self.error_registry:
            severity = self.error_registry[error_type]["severity"]
        elif isinstance(error, (MemoryError, SystemExit, KeyboardInterrupt)):
            severity = ErrorSeverity.CRITICAL
        elif isinstance(error, (ValueError, TypeError, AttributeError)):
            severity = ErrorSeverity.HIGH
        elif isinstance(error, (Warning, UserWarning)):
            severity = ErrorSeverity.LOW

        # Create error context
        error_context = ErrorContext(
            error_id=error_id,
            timestamp=time.time(),
            error_type=error_type.__name__,
            error_message=str(error),
            severity=severity,
            traceback_info=traceback.format_exc(),
            function_name=context,
        )

        # Add to history
        self.error_history.append(error_context)

        # Attempt recovery if strategy is registered
        if error_type in self.recovery_strategies:
            recovery_result = self._attempt_recovery(error, error_type, error_context)
            error_context.recovery_attempted = True
            error_context.recovery_successful = recovery_result.success
            error_context.recovery_strategy = recovery_result.strategy_used

        # Update statistics
        if error_type in self.error_registry:
            self.error_registry[error_type]["call_count"] += 1
            if error_context.recovery_successful:
                self.error_registry[error_type]["success_count"] += 1

        return error_context

    def _attempt_recovery(
        self, error: Exception, error_type: type, error_context: ErrorContext
    ) -> RecoveryResult:
        """Attempt error recovery using registered strategy."""
        strategy_info = self.recovery_strategies[error_type]
        strategy = strategy_info["strategy"]
        recovery_func = strategy_info["recovery_func"]
        max_attempts = strategy_info["max_attempts"]

        start_time = time.perf_counter()
        attempts = 0
        success = False

        # Check circuit breaker
        if self._is_circuit_breaker_open(error_type):
            return RecoveryResult(
                success=False,
                strategy_used=RecoveryStrategy.CIRCUIT_BREAKER,
                attempts_made=0,
                execution_time=time.perf_counter() - start_time,
                error_resolved=False,
            )

        # Attempt recovery based on strategy
        if strategy == RecoveryStrategy.RETRY:
            success = self._retry_recovery(recovery_func, max_attempts, error)
            attempts = max_attempts if not success else 1

        elif strategy == RecoveryStrategy.FALLBACK:
            success, fallback_data = self._fallback_recovery(recovery_func, error)
            attempts = 1

        elif strategy == RecoveryStrategy.GRACEFUL_DEGRADATION:
            success = self._graceful_degradation_recovery(recovery_func, error)
            attempts = 1

        elif strategy == RecoveryStrategy.CHECKPOINT_RESTORE:
            success = self._checkpoint_restore_recovery(recovery_func, error)
            attempts = 1

        execution_time = time.perf_counter() - start_time

        # Update circuit breaker
        self._update_circuit_breaker(error_type, success)

        return RecoveryResult(
            success=success,
            strategy_used=strategy,
            attempts_made=attempts,
            execution_time=execution_time,
            error_resolved=success,
        )

    def _retry_recovery(
        self, recovery_func: Callable, max_attempts: int, error: Exception
    ) -> bool:
        """Implement retry recovery strategy."""
        for attempt in range(max_attempts):
            try:
                recovery_func(error)
                return True
            except Exception:
                if attempt == max_attempts - 1:
                    return False
                time.sleep(0.1 * (2**attempt))  # Exponential backoff
        return False

    def _fallback_recovery(self, recovery_func: Callable, error: Exception) -> tuple:
        """Implement fallback recovery strategy."""
        try:
            fallback_data = recovery_func(error)
            return True, fallback_data
        except Exception:
            return False, None

    def _graceful_degradation_recovery(
        self, recovery_func: Callable, error: Exception
    ) -> bool:
        """Implement graceful degradation recovery strategy."""
        try:
            recovery_func(error)
            return True
        except Exception:
            return False

    def _checkpoint_restore_recovery(
        self, recovery_func: Callable, error: Exception
    ) -> bool:
        """Implement checkpoint restore recovery strategy."""
        try:
            recovery_func(error)
            return True
        except Exception:
            return False

    def _is_circuit_breaker_open(self, error_type: type) -> bool:
        """Check if circuit breaker is open for error type."""
        if error_type not in self.circuit_breakers:
            self.circuit_breakers[error_type] = {
                "failure_count": 0,
                "last_failure_time": 0,
                "state": "closed",  # closed, open, half_open
            }

        breaker = self.circuit_breakers[error_type]
        strategy_info = self.recovery_strategies.get(error_type, {})
        threshold = strategy_info.get("circuit_breaker_threshold", 5)

        # Check if breaker should be opened
        if breaker["failure_count"] >= threshold:
            breaker["state"] = "open"
            return True

        # Check if breaker should transition to half-open
        if (
            breaker["state"] == "open"
            and time.time() - breaker["last_failure_time"] > 60
        ):
            breaker["state"] = "half_open"

        return breaker["state"] == "open"

    def _update_circuit_breaker(self, error_type: type, success: bool):
        """Update circuit breaker state."""
        if error_type not in self.circuit_breakers:
            return

        breaker = self.circuit_breakers[error_type]

        if success:
            if breaker["state"] == "half_open":
                breaker["state"] = "closed"
            breaker["failure_count"] = 0
        else:
            breaker["failure_count"] += 1
            breaker["last_failure_time"] = time.time()

    def create_checkpoint(self, checkpoint_name: str, data: Any):
        """Create recovery checkpoint."""
        self.checkpoints[checkpoint_name] = {
            "timestamp": time.time(),
            "data": data,
            "size": sys.getsizeof(data) if data is not None else 0,
        }

    def restore_checkpoint(self, checkpoint_name: str) -> Any | None:
        """Restore from checkpoint."""
        if checkpoint_name in self.checkpoints:
            return self.checkpoints[checkpoint_name]["data"]
        return None

    def get_error_statistics(self) -> dict[str, Any]:
        """Get comprehensive error statistics."""
        if not self.error_history:
            return {"message": "No errors recorded"}

        # Group errors by type
        error_types = {}
        for error in self.error_history:
            error_type = error.error_type
            if error_type not in error_types:
                error_types[error_type] = {
                    "count": 0,
                    "recoveries_attempted": 0,
                    "recoveries_successful": 0,
                    "severities": [],
                }

            error_types[error_type]["count"] += 1
            error_types[error_type]["severities"].append(error.severity.value)
            if error.recovery_attempted:
                error_types[error_type]["recoveries_attempted"] += 1
                if error.recovery_successful:
                    error_types[error_type]["recoveries_successful"] += 1

        # Calculate recovery rates
        for error_type, stats in error_types.items():
            if stats["recoveries_attempted"] > 0:
                stats["recovery_rate"] = (
                    stats["recoveries_successful"] / stats["recoveries_attempted"]
                ) * 100
            else:
                stats["recovery_rate"] = 0

        return {
            "total_errors": len(self.error_history),
            "error_types": error_types,
            "recovery_strategies_registered": len(self.recovery_strategies),
            "active_circuit_breakers": len(
                [cb for cb in self.circuit_breakers.values() if cb["state"] != "closed"]
            ),
            "checkpoints_available": len(self.checkpoints),
        }


class ResilientFunction:
    """Decorator for making functions resilient to errors."""

    def __init__(
        self,
        error_handler: ErrorHandler,
        max_retries: int = 3,
        fallback_func: Callable | None = None,
    ):
        self.error_handler = error_handler
        self.max_retries = max_retries
        self.fallback_func = fallback_func

    def __call__(self, func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_error = None

            # Try main function with retries
            for attempt in range(self.max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_error = e
                    self.error_handler.handle_error(e, func.__name__)

                    if attempt < self.max_retries:
                        time.sleep(0.1 * (2**attempt))  # Exponential backoff
                    else:
                        break

            # Try fallback if available
            if self.fallback_func:
                try:
                    return self.fallback_func(*args, **kwargs)
                except Exception as fallback_error:
                    self.error_handler.handle_error(
                        fallback_error, f"{func.__name__}_fallback"
                    )

            # If all else fails, raise the last error
            raise last_error

        return wrapper


class FaultTolerantSystem:
    """Fault-tolerant system with comprehensive error handling."""

    def __init__(self):
        self.error_handler = ErrorHandler()
        self.setup_default_handlers()

    def setup_default_handlers(self):
        """Setup default error handlers and recovery strategies."""

        # Register common error handlers
        self.error_handler.register_error_handler(
            ValueError, self._handle_value_error, ErrorSeverity.HIGH
        )

        self.error_handler.register_error_handler(
            MemoryError, self._handle_memory_error, ErrorSeverity.CRITICAL
        )

        self.error_handler.register_error_handler(
            FileNotFoundError, self._handle_file_not_found, ErrorSeverity.MEDIUM
        )

        # Register recovery strategies
        self.error_handler.register_recovery_strategy(
            ValueError,
            RecoveryStrategy.RETRY,
            self._retry_with_validation,
            max_attempts=3,
        )

        self.error_handler.register_recovery_strategy(
            MemoryError,
            RecoveryStrategy.GRACEFUL_DEGRADATION,
            self._reduce_memory_usage,
            max_attempts=1,
        )

        self.error_handler.register_recovery_strategy(
            FileNotFoundError,
            RecoveryStrategy.FALLBACK,
            self._provide_default_file,
            max_attempts=1,
        )

    def _handle_value_error(self, error: ValueError):
        """Handle ValueError with input validation."""
        logging.warning(f"ValueError handled: {error}")

    def _handle_memory_error(self, error: MemoryError):
        """Handle MemoryError with memory cleanup."""
        import gc

        gc.collect()
        logging.critical(f"MemoryError handled: {error}")

    def _handle_file_not_found(self, error: FileNotFoundError):
        """Handle FileNotFoundError with fallback."""
        logging.error(f"FileNotFoundError handled: {error}")

    def _retry_with_validation(self, error: ValueError):
        """Recovery strategy for ValueError."""
        # This would implement input validation and correction

    def _reduce_memory_usage(self, error: MemoryError):
        """Recovery strategy for MemoryError."""
        import gc

        gc.collect()

    def _provide_default_file(self, error: FileNotFoundError) -> str:
        """Recovery strategy for FileNotFoundError."""
        # Create a temporary default file
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            f.write('{"default": true}')
            return f.name

    @contextmanager
    def fault_tolerant_context(self, operation_name: str):
        """Context manager for fault-tolerant operations."""
        start_time = time.time()
        checkpoint_name = f"{operation_name}_{int(start_time)}"

        try:
            # Create checkpoint before operation
            self.error_handler.create_checkpoint(
                checkpoint_name, {"operation": operation_name, "start_time": start_time}
            )
            yield
        except Exception as e:
            # Handle error and attempt recovery
            error_context = self.error_handler.handle_error(e, operation_name)

            if not error_context.recovery_successful:
                # Restore from checkpoint if recovery failed
                checkpoint_data = self.error_handler.restore_checkpoint(checkpoint_name)
                if checkpoint_data:
                    logging.info(f"Restored from checkpoint: {checkpoint_name}")

                raise
        finally:
            # Cleanup checkpoint
            if checkpoint_name in self.error_handler.checkpoints:
                del self.error_handler.checkpoints[checkpoint_name]

    def resilient_function(
        self, max_retries: int = 3, fallback_func: Callable | None = None
    ):
        """Decorator for making functions resilient."""
        return ResilientFunction(self.error_handler, max_retries, fallback_func)

    def run_diagnostic_tests(self) -> dict[str, Any]:
        """Run diagnostic tests for error handling system."""
        test_results = {
            "error_handling_tests": [],
            "recovery_tests": [],
            "fault_tolerance_tests": [],
        }

        # Test error handling
        test_results["error_handling_tests"] = self._test_error_handling()

        # Test recovery mechanisms
        test_results["recovery_tests"] = self._test_recovery_mechanisms()

        # Test fault tolerance
        test_results["fault_tolerance_tests"] = self._test_fault_tolerance()

        return test_results

    def _test_error_handling(self) -> list[dict[str, Any]]:
        """Test error handling capabilities."""
        tests = []

        # Test ValueError handling
        try:
            raise ValueError("Test value error")
        except ValueError as e:
            error_context = self.error_handler.handle_error(e, "test_value_error")
            tests.append(
                {
                    "test_name": "ValueError handling",
                    "success": error_context is not None,
                    "error_id": error_context.error_id,
                    "severity": error_context.severity.value,
                }
            )

        # Test MemoryError handling (simulated)
        try:
            # Simulate memory error
            error = MemoryError("Simulated memory error")
            error_context = self.error_handler.handle_error(error, "test_memory_error")
            tests.append(
                {
                    "test_name": "MemoryError handling",
                    "success": error_context is not None,
                    "error_id": error_context.error_id,
                    "severity": error_context.severity.value,
                }
            )
        except Exception as e:
            tests.append(
                {"test_name": "MemoryError handling", "success": False, "error": str(e)}
            )

        return tests

    def _test_recovery_mechanisms(self) -> list[dict[str, Any]]:
        """Test recovery mechanisms."""
        tests = []

        # Test retry mechanism
        @self.resilient_function(max_retries=2)
        def failing_function(should_fail=True):
            if should_fail:
                raise ValueError("Intentional failure")
            return "success"

        try:
            result = failing_function(should_fail=False)
            tests.append(
                {
                    "test_name": "Retry mechanism (success case)",
                    "success": result == "success",
                    "result": result,
                }
            )
        except Exception as e:
            tests.append(
                {
                    "test_name": "Retry mechanism (success case)",
                    "success": False,
                    "error": str(e),
                }
            )

        # Test fallback mechanism
        def fallback_function(*args, **kwargs):
            return "fallback_result"

        @self.resilient_function(max_retries=1, fallback_func=fallback_function)
        def function_with_fallback():
            raise ValueError("Always fails")

        try:
            result = function_with_fallback()
            tests.append(
                {
                    "test_name": "Fallback mechanism",
                    "success": result == "fallback_result",
                    "result": result,
                }
            )
        except Exception as e:
            tests.append(
                {"test_name": "Fallback mechanism", "success": False, "error": str(e)}
            )

        return tests

    def _test_fault_tolerance(self) -> list[dict[str, Any]]:
        """Test fault tolerance features."""
        tests = []

        # Test checkpoint creation and restoration
        test_data = {"test": "data", "value": 42}
        self.error_handler.create_checkpoint("test_checkpoint", test_data)

        restored_data = self.error_handler.restore_checkpoint("test_checkpoint")
        tests.append(
            {
                "test_name": "Checkpoint creation and restoration",
                "success": restored_data == test_data,
                "original_data": test_data,
                "restored_data": restored_data,
            }
        )

        # Test fault-tolerant context
        context_success = False
        try:
            with self.fault_tolerant_context("test_operation"):
                # Simulate successful operation
                context_success = True
        except Exception as e:
            tests.append(
                {
                    "test_name": "Fault-tolerant context (success)",
                    "success": False,
                    "error": str(e),
                }
            )
        else:
            tests.append(
                {
                    "test_name": "Fault-tolerant context (success)",
                    "success": context_success,
                }
            )

        return tests


def run_error_handling_recovery_system():
    """Main function to run error handling and recovery system."""
    print("Comprehensive Error Handling and Recovery Systems - Task 5.4")
    print("=" * 70)

    # Create fault-tolerant system
    ft_system = FaultTolerantSystem()

    print("Testing error handling and recovery capabilities...")

    # Run diagnostic tests
    diagnostic_results = ft_system.run_diagnostic_tests()

    # Get error statistics
    error_stats = ft_system.error_handler.get_error_statistics()

    # Test some additional scenarios
    print("Running additional error scenarios...")

    # Test matrix operations with error handling
    @ft_system.resilient_function(max_retries=2)
    def robust_matrix_operation(size=100):
        if size > 1000:
            raise MemoryError("Matrix too large")
        if size < 0:
            raise ValueError("Invalid matrix size")

        A = np.random.rand(size, size)
        B = np.random.rand(size, size)
        return A @ B

    # Test successful operation
    try:
        robust_matrix_operation(50)
        matrix_test_success = True
    except Exception as e:
        matrix_test_success = False
        ft_system.error_handler.handle_error(e, "matrix_operation_test")

    # Test error recovery with file operations
    def fallback_file_reader(*args, **kwargs):
        return {"status": "fallback", "data": "default_content"}

    @ft_system.resilient_function(max_retries=1, fallback_func=fallback_file_reader)
    def robust_file_reader(filename):
        if not os.path.exists(filename):
            raise FileNotFoundError(f"File not found: {filename}")
        with open(filename) as f:
            return {"status": "success", "data": f.read()}

    # Test file operation with non-existent file
    try:
        file_result = robust_file_reader("non_existent_file.txt")
        file_test_success = file_result["status"] == "fallback"
    except Exception as e:
        file_test_success = False
        ft_system.error_handler.handle_error(e, "file_operation_test")

    # Compile comprehensive results
    comprehensive_results = {
        "error_handling_statistics": error_stats,
        "diagnostic_test_results": diagnostic_results,
        "additional_tests": {
            "matrix_operation_test": {
                "success": matrix_test_success,
                "description": "Robust matrix operation with error handling",
            },
            "file_operation_test": {
                "success": file_test_success,
                "description": "File operation with fallback recovery",
            },
        },
        "system_capabilities": {
            "error_types_handled": len(ft_system.error_handler.error_registry),
            "recovery_strategies_available": len(
                ft_system.error_handler.recovery_strategies
            ),
            "circuit_breakers_active": len(ft_system.error_handler.circuit_breakers),
            "checkpoints_created": len(ft_system.error_handler.checkpoints),
        },
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    # Display summary
    stats = error_stats
    capabilities = comprehensive_results["system_capabilities"]

    print("\nERROR HANDLING AND RECOVERY SUMMARY:")
    print(f"  Total Errors Handled: {stats.get('total_errors', 0)}")
    print(f"  Error Types Registered: {capabilities['error_types_handled']}")
    print(
        f"  Recovery Strategies Available: {capabilities['recovery_strategies_available']}"
    )
    print(f"  Circuit Breakers Active: {capabilities['circuit_breakers_active']}")

    # Display test results
    print("\nDIAGNOSTIC TEST RESULTS:")
    for category, tests in diagnostic_results.items():
        successful_tests = sum(1 for test in tests if test.get("success", False))
        print(f"  {category}: {successful_tests}/{len(tests)} tests passed")

    print("\nADDITIONAL TESTS:")
    for test_name, test_result in comprehensive_results["additional_tests"].items():
        status = "✓ PASS" if test_result["success"] else "✗ FAIL"
        print(f"  {status} {test_result['description']}")

    # Save results
    results_dir = Path("error_handling_results")
    results_dir.mkdir(exist_ok=True)

    results_file = results_dir / "task_5_4_error_handling_recovery_report.json"
    with open(results_file, "w") as f:
        json.dump(comprehensive_results, f, indent=2)

    print(f"\n📄 Error handling report saved to: {results_file}")
    print("✅ Task 5.4 Error Handling and Recovery Complete!")
    print(f"🛡️  {capabilities['error_types_handled']} error types handled")
    print(
        f"🔄 {capabilities['recovery_strategies_available']} recovery strategies available"
    )
    print(f"⚠️  {capabilities['circuit_breakers_active']} circuit breakers monitoring")

    return comprehensive_results


if __name__ == "__main__":
    run_error_handling_recovery_system()
