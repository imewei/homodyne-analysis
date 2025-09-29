"""
Real-time Progress Tracking and User Feedback
=============================================

Advanced progress tracking system with real-time updates, ETA calculation,
and multi-level progress reporting for complex scientific computing workflows.

Features:
- Thread-safe progress tracking with atomic updates
- Real-time ETA calculation with moving averages
- Multi-level progress tracking (tasks, subtasks, operations)
- Adaptive refresh rates for optimal performance
- Memory-efficient circular buffers for timing data
- Rich console output with color coding and animations
- Progress persistence for long-running computations
"""

import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from dataclasses import field
from enum import Enum
from typing import Any

try:
    from rich.console import Console
    from rich.live import Live
    from rich.progress import BarColumn
    from rich.progress import MofNCompleteColumn
    from rich.progress import Progress
    from rich.progress import SpinnerColumn
    from rich.progress import TaskID
    from rich.progress import TextColumn
    from rich.progress import TimeElapsedColumn
    from rich.progress import TimeRemainingColumn
    from rich.progress import TransferSpeedColumn

    # from rich.text import Text  # Unused currently

    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    Console = None
    Progress = None
    TaskID = None
    Live = None


class ProgressLevel(Enum):
    """Progress tracking levels for hierarchical progress reporting."""

    ANALYSIS = "analysis"  # Top-level analysis progress
    METHOD = "method"  # Individual optimization method
    ITERATION = "iteration"  # Optimization iterations
    COMPUTATION = "computation"  # Core computations (correlations, etc.)
    IO = "io"  # File I/O operations


@dataclass
class ProgressState:
    """Thread-safe progress state with timing information."""

    current: int = 0
    total: int = 0
    description: str = "Processing..."
    start_time: float = field(default_factory=time.time)
    last_update: float = field(default_factory=time.time)
    timing_buffer: deque = field(default_factory=lambda: deque(maxlen=20))
    completed: bool = False
    failed: bool = False
    error_message: str = ""

    def __post_init__(self):
        self._lock = threading.Lock()

    def update(
        self, current: int | None = None, description: str | None = None
    ) -> None:
        """Thread-safe progress update."""
        with self._lock:
            now = time.time()
            if current is not None:
                # Calculate rate for timing buffer
                if self.current > 0:  # Avoid division by zero
                    delta_items = current - self.current
                    delta_time = now - self.last_update
                    if delta_time > 0:
                        rate = delta_items / delta_time
                        self.timing_buffer.append(rate)

                self.current = current

            if description is not None:
                self.description = description

            self.last_update = now

    @property
    def progress_ratio(self) -> float:
        """Get progress ratio (0.0 to 1.0)."""
        if self.total <= 0:
            return 0.0
        return min(1.0, self.current / self.total)

    @property
    def eta_seconds(self) -> float | None:
        """Calculate ETA based on recent timing data."""
        if not self.timing_buffer or self.total <= self.current:
            return None

        # Use median rate from recent measurements for stability
        rates = list(self.timing_buffer)
        if not rates:
            return None

        rates.sort()
        median_rate = rates[len(rates) // 2]

        if median_rate <= 0:
            return None

        remaining_items = self.total - self.current
        return remaining_items / median_rate

    @property
    def elapsed_seconds(self) -> float:
        """Get elapsed time since start."""
        return time.time() - self.start_time


class ProgressTracker:
    """
    Advanced progress tracking system for scientific computing workflows.

    Provides hierarchical progress tracking with real-time updates,
    ETA calculation, and rich console output.
    """

    def __init__(self, enable_rich: bool = True, refresh_rate: float = 0.1):
        self.enable_rich = enable_rich and RICH_AVAILABLE
        self.refresh_rate = refresh_rate
        self.states: dict[str, ProgressState] = {}
        self.active_tasks: dict[str, TaskID] = {}
        self._lock = threading.Lock()

        # Initialize console and progress display
        if self.enable_rich:
            self.console = Console()
            self.progress = Progress(
                SpinnerColumn(),
                TextColumn("[bold blue]{task.description}"),
                BarColumn(complete_style="green", finished_style="green"),
                MofNCompleteColumn(),
                "[",
                TimeElapsedColumn(),
                "<",
                TimeRemainingColumn(),
                "]",
                TransferSpeedColumn(),
                console=self.console,
                expand=True,
            )
            self.live = None
        else:
            # Fallback to simple text progress
            self.console = None
            self.progress = None
            self.live = None
            self._last_print_time = 0
            self._print_interval = 2.0  # Print every 2 seconds for fallback

    def create_task(
        self,
        task_id: str,
        description: str,
        total: int,
        level: ProgressLevel = ProgressLevel.COMPUTATION,
    ) -> str:
        """Create a new progress tracking task."""
        with self._lock:
            state = ProgressState(total=total, description=description)
            self.states[task_id] = state

            if self.enable_rich and self.progress is not None:
                rich_task_id = self.progress.add_task(
                    description, total=total, start=True
                )
                self.active_tasks[task_id] = rich_task_id

        return task_id

    def update_task(
        self,
        task_id: str,
        current: int | None = None,
        description: str | None = None,
        increment: int | None = None,
    ) -> None:
        """Update progress for a specific task."""
        if task_id not in self.states:
            return

        state = self.states[task_id]

        # Handle increment vs absolute update
        if increment is not None:
            current = state.current + increment

        # Update state
        state.update(current=current, description=description)

        # Update rich display if available
        if self.enable_rich and task_id in self.active_tasks:
            rich_task_id = self.active_tasks[task_id]

            update_kwargs = {}
            if current is not None:
                update_kwargs["completed"] = current
            if description is not None:
                update_kwargs["description"] = description

            if update_kwargs and self.progress is not None:
                self.progress.update(rich_task_id, **update_kwargs)
        else:
            # Fallback text progress
            self._print_fallback_progress(task_id)

    def complete_task(
        self, task_id: str, success: bool = True, message: str = ""
    ) -> None:
        """Mark a task as completed."""
        if task_id not in self.states:
            return

        state = self.states[task_id]
        state.completed = True
        state.failed = not success
        state.error_message = message

        if self.enable_rich and task_id in self.active_tasks:
            rich_task_id = self.active_tasks[task_id]
            if self.progress is not None:
                if success:
                    self.progress.update(rich_task_id, completed=state.total)
                    final_desc = f"✓ {state.description}"
                else:
                    final_desc = f"✗ {state.description} - {message}"

                self.progress.update(rich_task_id, description=final_desc)

                # Remove from active tasks after a short delay
                threading.Timer(1.0, lambda: self._cleanup_task(task_id)).start()
        else:
            # Fallback completion message
            status = "✓ COMPLETED" if success else f"✗ FAILED: {message}"
            print(f"[{status}] {state.description}", file=sys.stderr)

    def _cleanup_task(self, task_id: str) -> None:
        """Clean up completed task from active tracking."""
        with self._lock:
            if task_id in self.active_tasks:
                del self.active_tasks[task_id]

    def _print_fallback_progress(self, task_id: str) -> None:
        """Print simple text progress for fallback mode."""
        current_time = time.time()
        if current_time - self._last_print_time < self._print_interval:
            return

        state = self.states[task_id]
        progress_pct = state.progress_ratio * 100

        eta_str = ""
        if state.eta_seconds is not None:
            eta_min = int(state.eta_seconds // 60)
            eta_sec = int(state.eta_seconds % 60)
            eta_str = f" ETA: {eta_min:02d}:{eta_sec:02d}"

        print(
            f"\r[{progress_pct:5.1f}%] {state.description} ({state.current}/{state.total}){eta_str}",
            end="",
            file=sys.stderr,
        )

        self._last_print_time = current_time

    def __enter__(self):
        """Context manager entry."""
        if self.enable_rich and self.progress is not None:
            self.live = Live(
                self.progress,
                console=self.console,
                refresh_per_second=1 / self.refresh_rate,
            )
            self.live.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        if self.live is not None:
            self.live.__exit__(exc_type, exc_val, exc_tb)

        # Print final newline for fallback mode
        if not self.enable_rich:
            print(file=sys.stderr)

    def get_summary(self) -> dict[str, Any]:
        """Get summary of all tracked tasks."""
        summary = {
            "total_tasks": len(self.states),
            "completed_tasks": sum(1 for s in self.states.values() if s.completed),
            "failed_tasks": sum(1 for s in self.states.values() if s.failed),
            "active_tasks": len(self.active_tasks),
            "tasks": {},
        }

        for task_id, state in self.states.items():
            summary["tasks"][task_id] = {
                "progress": state.progress_ratio,
                "current": state.current,
                "total": state.total,
                "description": state.description,
                "elapsed": state.elapsed_seconds,
                "eta": state.eta_seconds,
                "completed": state.completed,
                "failed": state.failed,
            }

        return summary


# Global progress tracker instance
_global_tracker: ProgressTracker | None = None
_tracker_lock = threading.Lock()


def get_progress_tracker(
    enable_rich: bool = True, refresh_rate: float = 0.1
) -> ProgressTracker:
    """Get global progress tracker instance (singleton pattern)."""
    global _global_tracker

    with _tracker_lock:
        if _global_tracker is None:
            _global_tracker = ProgressTracker(
                enable_rich=enable_rich, refresh_rate=refresh_rate
            )

    return _global_tracker


def reset_progress_tracker() -> None:
    """Reset global progress tracker (useful for testing)."""
    global _global_tracker

    with _tracker_lock:
        _global_tracker = None


# Convenience functions for common progress tracking patterns
def track_analysis_progress(method_name: str, total_steps: int) -> str:
    """Start tracking analysis progress for a specific method."""
    tracker = get_progress_tracker()
    task_id = f"analysis_{method_name}"
    tracker.create_task(
        task_id=task_id,
        description=f"Running {method_name} optimization",
        total=total_steps,
        level=ProgressLevel.ANALYSIS,
    )
    return task_id


def track_computation_progress(computation_name: str, total_items: int) -> str:
    """Start tracking computation progress (correlations, etc.)."""
    tracker = get_progress_tracker()
    task_id = f"compute_{computation_name}"
    tracker.create_task(
        task_id=task_id,
        description=f"Computing {computation_name}",
        total=total_items,
        level=ProgressLevel.COMPUTATION,
    )
    return task_id


def track_io_progress(operation_name: str, total_files: int) -> str:
    """Start tracking I/O progress."""
    tracker = get_progress_tracker()
    task_id = f"io_{operation_name}"
    tracker.create_task(
        task_id=task_id,
        description=f"I/O: {operation_name}",
        total=total_files,
        level=ProgressLevel.IO,
    )
    return task_id


class ProgressContext:
    """Context manager for automatic progress tracking."""

    def __init__(self, description: str, total: int, task_id: str | None = None):
        self.description = description
        self.total = total
        self.task_id = task_id or f"task_{int(time.time() * 1000)}"
        self.tracker = get_progress_tracker()

    def __enter__(self):
        self.tracker.create_task(
            task_id=self.task_id, description=self.description, total=self.total
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        success = exc_type is None
        message = str(exc_val) if exc_val else ""
        self.tracker.complete_task(self.task_id, success=success, message=message)

    def update(
        self,
        current: int | None = None,
        increment: int | None = None,
        description: str | None = None,
    ) -> None:
        """Update progress within the context."""
        self.tracker.update_task(
            task_id=self.task_id,
            current=current,
            increment=increment,
            description=description,
        )


if __name__ == "__main__":
    # Example usage and testing

    print("Testing Progress Tracker...")

    with ProgressTracker() as tracker:
        # Test multiple concurrent tasks
        task1 = tracker.create_task(
            "analysis_nelder_mead", "Nelder-Mead Optimization", 100
        )
        task2 = tracker.create_task(
            "compute_correlations", "Computing C2 Correlations", 50
        )

        # Simulate work with different progress rates
        for i in range(100):
            time.sleep(0.05)  # Simulate work

            tracker.update_task(task1, current=i + 1)

            if i < 50:
                tracker.update_task(task2, current=i + 1)
            elif i == 50:
                tracker.complete_task(task2, success=True)

        tracker.complete_task(task1, success=True)

    print("\nTesting context manager...")

    with ProgressContext("Testing context manager", 20) as ctx:
        for i in range(20):
            time.sleep(0.1)
            ctx.update(current=i + 1)

    print("Progress tracking tests completed!")
