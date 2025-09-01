"""
Pytest Configuration for Rheo-SAXS-XPCS Tests
=============================================

Configuration and shared fixtures for the test suite.
"""

import os
import shutil
import sys
import time
import warnings
from pathlib import Path

import matplotlib
import numpy as np
import pytest

# Import specific fixtures needed by tests - avoid star import to prevent circular imports

# CRITICAL: Set threading environment variables BEFORE any imports that might use Numba
# This must happen before importing pytest, numpy, matplotlib, etc.
os.environ["PYTHONWARNINGS"] = "ignore"
# Conservative threading for test stability (consistent with performance
# settings)
os.environ["OMP_NUM_THREADS"] = "1"  # Use single thread to avoid conflicts
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMBA_NUM_THREADS"] = "1"  # Single thread for test stability
os.environ["NUMBA_DISABLE_INTEL_SVML"] = "1"
os.environ["NUMBA_FASTMATH"] = "0"  # Conservative JIT for stability
os.environ["NUMBA_THREADING_LAYER"] = "safe"  # Use safe threading layer

# Now safe to import everything else

# Use non-GUI matplotlib backend for testing
matplotlib.use("Agg")

# Add the project root directory to Python path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def mark_directory_as_test_artifact(directory_path: Path) -> None:
    """Mark a directory as a test artifact for safe cleanup.

    Creates a hidden .test-artifact file in the directory to indicate
    that it was created by tests and can be safely removed during cleanup.

    Parameters
    ----------
    directory_path : Path
        The directory to mark as a test artifact
    """
    try:
        directory_path.mkdir(parents=True, exist_ok=True)
        marker_file = directory_path / ".test-artifact"
        marker_file.write_text(
            f"Test artifact created at {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
        )
    except Exception:
        # Don't fail tests if marker creation fails
        pass


@pytest.fixture(autouse=True, scope="function")
def cleanup_test_artifacts(request):
    """Automatically clean up test artifacts after each test.

    This fixture ensures that temporary directories created during testing
    (especially homodyne_results folders) are cleaned up automatically.

    SAFETY: Only removes homodyne_results directories that were explicitly
    marked as test artifacts, NOT any pre-existing directories.
    """
    # Store initial working directory and check for pre-existing
    # homodyne_results
    initial_cwd = Path.cwd()

    # Track which homodyne_results directories existed before the test
    pre_existing_results_dirs = set()
    potential_paths = [
        initial_cwd / "homodyne_results",
        initial_cwd
        / "homodyne"
        / "homodyne_results",  # Include ./homodyne/homodyne_results
    ]

    for path in potential_paths:
        if path.exists() and path.is_dir():
            pre_existing_results_dirs.add(path)

    yield  # Run the test

    # Cleanup after test - ONLY remove directories that are marked as test
    # artifacts
    try:
        current_cwd = Path.cwd()
        cleanup_candidates = [
            initial_cwd / "homodyne_results",
            current_cwd / "homodyne_results",  # In case cwd changed during test
            initial_cwd
            / "homodyne"
            / "homodyne_results",  # Include ./homodyne/homodyne_results
            current_cwd
            / "homodyne"
            / "homodyne_results",  # In case cwd changed during test
        ]

        for cleanup_path in cleanup_candidates:
            if cleanup_path.exists() and cleanup_path.is_dir():
                # CONSERVATIVE SAFETY CHECK: Only remove if:
                # 1. This directory wasn't there before the test AND
                # 2. The directory contains a test artifact marker
                test_marker = cleanup_path / ".test-artifact"

                # Special case: Always clean up ./homodyne/homodyne_results if it was created during tests
                # This directory should never exist in a clean project
                # structure
                is_nested_homodyne_results = "homodyne/homodyne_results" in str(
                    cleanup_path
                )

                if (
                    cleanup_path not in pre_existing_results_dirs
                    and test_marker.exists()
                ) or (
                    is_nested_homodyne_results
                    and cleanup_path not in pre_existing_results_dirs
                ):
                    try:
                        shutil.rmtree(cleanup_path)
                        # Only print in verbose mode to avoid cluttering output
                        if getattr(request.config.option, "verbose", 0) > 1:
                            print(
                                f"\n✓ Cleaned up test-created artifact: {cleanup_path}"
                            )
                    except (OSError, PermissionError):
                        # Silently continue if cleanup fails
                        pass
                else:
                    # This directory existed before the test - preserve it
                    if getattr(request.config.option, "verbose", 0) > 1:
                        if cleanup_path in pre_existing_results_dirs:
                            print(
                                f"\n⚠ Preserved pre-existing directory: {cleanup_path}"
                            )
                        elif not is_nested_homodyne_results:
                            print(
                                f"\n⚠ Preserved directory without test marker: {cleanup_path}"
                            )

    except Exception:
        # Don't fail tests due to cleanup issues
        pass


@pytest.fixture(autouse=True, scope="session")
def cleanup_session_artifacts():
    """Clean up any remaining test artifacts after the entire test session.

    SAFETY: Only removes homodyne_results directories that are explicitly
    marked as test artifacts. Never removes user analysis results.
    """
    yield  # Run all tests

    # Final cleanup after all tests - only clean up marked test artifacts
    try:
        project_root = Path(__file__).parent.parent.parent
        current_cwd = Path.cwd()

        # Check for test artifacts in both project root and current directory
        cleanup_candidates = [
            project_root / "homodyne_results",
            current_cwd / "homodyne_results",
            project_root
            / "homodyne"
            / "homodyne_results",  # Include ./homodyne/homodyne_results
            current_cwd
            / "homodyne"
            / "homodyne_results",  # Include ./homodyne/homodyne_results
        ]

        for homodyne_results_path in cleanup_candidates:
            if homodyne_results_path.exists() and homodyne_results_path.is_dir():
                # Special case: Always clean up ./homodyne/homodyne_results
                # This directory should never exist in a clean project
                # structure
                is_nested_homodyne_results = "homodyne/homodyne_results" in str(
                    homodyne_results_path
                )

                # CONSERVATIVE SAFETY: Only remove if explicitly marked as test artifact
                # OR if it's the nested homodyne/homodyne_results directory
                test_marker = homodyne_results_path / ".test-artifact"
                if test_marker.exists() or is_nested_homodyne_results:
                    try:
                        shutil.rmtree(homodyne_results_path)
                        print(
                            f"\n✓ Final cleanup: Removed test artifact {homodyne_results_path}"
                        )
                    except (OSError, PermissionError):
                        print(
                            f"\n⚠ Could not remove test artifact {homodyne_results_path}"
                        )
                else:
                    # No test marker and not nested - this could be user data,
                    # preserve it
                    print(
                        f"\n⚠ Preserved user directory (no test marker): {homodyne_results_path}"
                    )

    except Exception:
        pass


def pytest_sessionfinish(session, exitstatus):
    """Hook that runs after the entire test session is finished.

    SAFETY: Only performs cleanup for directories explicitly marked as test artifacts.
    Never removes user analysis results.
    """
    try:
        # Final cleanup - only remove directories marked as test artifacts
        current_dir = Path.cwd()
        project_root = Path(__file__).parent.parent.parent

        # Check for test artifacts in both project root and current directory
        cleanup_candidates = [
            project_root / "homodyne_results",
            current_dir / "homodyne_results",
            project_root
            / "homodyne"
            / "homodyne_results",  # Include ./homodyne/homodyne_results
            current_dir
            / "homodyne"
            / "homodyne_results",  # Include ./homodyne/homodyne_results
        ]

        for path in cleanup_candidates:
            if path.exists() and path.is_dir():
                # Special case: Always clean up ./homodyne/homodyne_results
                is_nested_homodyne_results = "homodyne/homodyne_results" in str(path)

                # CONSERVATIVE SAFETY: Only remove if explicitly marked as test artifact
                # OR if it's the nested homodyne/homodyne_results directory
                test_marker = path / ".test-artifact"
                if test_marker.exists() or is_nested_homodyne_results:
                    try:
                        shutil.rmtree(path)
                        if session.config.option.verbose > 0:
                            print(f"\n✓ Test cleanup: Removed test artifact {path}")
                    except (OSError, PermissionError):
                        if session.config.option.verbose > 0:
                            print(f"\n⚠ Could not clean up test artifact {path}")
                else:
                    # No test marker and not nested - preserve this directory
                    if (
                        session.config.option.verbose > 0
                        and not is_nested_homodyne_results
                    ):
                        print(f"\n⚠ Preserved user directory (no test marker): {path}")
    except Exception:
        # Don't break test reporting due to cleanup issues
        pass


# Configure numpy to raise on errors (helps catch numerical issues)
np.seterr(
    all="raise", under="ignore"
)  # Ignore underflow which is common in exp calculations

# Filter common warnings during testing
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=PendingDeprecationWarning)
warnings.filterwarnings(
    "ignore", category=RuntimeWarning, message=".*invalid value encountered.*"
)

# Comprehensive matplotlib font warning suppression
warnings.filterwarnings(
    "ignore", category=UserWarning, message=".*Glyph.*missing from font.*"
)
warnings.filterwarnings(
    "ignore", category=UserWarning, message=".*SUBSCRIPT.*missing from font.*"
)
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message=".*SUPERSCRIPT.*missing from font.*",
)

# Suppress all matplotlib UserWarnings to catch font issues

warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib.*")

# Suppress font warnings from homodyne modules that use matplotlib
warnings.filterwarnings("ignore", category=UserWarning, module="homodyne.core.io_utils")
warnings.filterwarnings("ignore", category=UserWarning, module="homodyne.plotting")


def pytest_configure(config):
    """Configure pytest with custom markers and settings."""
    config.addinivalue_line(
        "markers",
        "slow: marks tests as slow (deselect with '-m \"not slow\"')",
    )
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line(
        "markers", "plotting: marks tests that require plotting functionality"
    )
    config.addinivalue_line(
        "markers", "io: marks tests that involve file I/O operations"
    )
    config.addinivalue_line(
        "markers", "mcmc_integration: marks tests as MCMC integration tests"
    )

    # Performance testing markers
    config.addinivalue_line(
        "markers",
        "performance: mark test as a performance test that should be fast",
    )
    config.addinivalue_line("markers", "memory: mark test as a memory usage test")
    config.addinivalue_line(
        "markers", "regression: mark test as a performance regression test"
    )
    config.addinivalue_line(
        "markers",
        "benchmark: mark test for benchmarking (requires pytest-benchmark)",
    )

    # Phase 4 optimization markers
    config.addinivalue_line("markers", "fast: marks tests that run quickly (< 1s)")
    config.addinivalue_line(
        "markers", "unit: marks unit tests (isolated, no external dependencies)"
    )
    config.addinivalue_line(
        "markers", "system: marks system-level tests (require environment setup)"
    )
    config.addinivalue_line(
        "markers", "ci_skip: marks tests to skip in CI environments"
    )
    config.addinivalue_line(
        "markers", "gpu: marks tests that can utilize GPU acceleration"
    )
    config.addinivalue_line("markers", "jax: marks tests requiring JAX dependencies")
    config.addinivalue_line("markers", "mcmc: marks tests requiring MCMC dependencies")

    # Configure warnings filters
    config.addinivalue_line("filterwarnings", "ignore::UserWarning:matplotlib.*")
    config.addinivalue_line("filterwarnings", "ignore::UserWarning:homodyne.*")
    config.addinivalue_line(
        "filterwarnings", "ignore:.*Glyph.*missing from font.*:UserWarning"
    )


def pytest_collection_modifyitems(config, items):
    """Automatically mark tests based on their location and content.
    
    This runs AFTER local conftest.py hooks, so we check for existing markers
    to avoid conflicts with subdirectory marking rules.
    """
    for item in items:
        # Check what markers are already present from local conftest.py files
        existing_markers = {marker.name for marker in item.iter_markers()}
        
        # Initialize marker flags based on existing markers
        has_integration = "integration" in existing_markers
        has_mcmc = "mcmc" in existing_markers
        has_slow = "slow" in existing_markers
        
        # Only add markers if they're not already present
        # Mark tests based on directory structure
        if "/integration/" in item.nodeid and not has_integration:
            item.add_marker(pytest.mark.integration)
            has_integration = True
        
        if "/mcmc/" in item.nodeid and not has_mcmc:
            item.add_marker(pytest.mark.mcmc)
            has_mcmc = True
            
        if "/performance/" in item.nodeid:
            if "performance" not in existing_markers:
                item.add_marker(pytest.mark.performance)
            if not has_slow:
                item.add_marker(pytest.mark.slow)
                has_slow = True
                
        if "/system/" in item.nodeid:
            if "system" not in existing_markers:
                item.add_marker(pytest.mark.system)
            if not has_slow:
                item.add_marker(pytest.mark.slow)
                has_slow = True

        # Mark tests based on filename patterns
        if "test_integration" in item.nodeid and not has_integration:
            item.add_marker(pytest.mark.integration)
            has_integration = True

        # Mark plotting tests
        if ("test_plotting" in item.nodeid or "plot" in item.name.lower()) and "plotting" not in existing_markers:
            item.add_marker(pytest.mark.plotting)

        # Mark I/O tests
        if ("test_io" in item.nodeid or "io_utils" in item.nodeid) and "io" not in existing_markers:
            item.add_marker(pytest.mark.io)

        # Mark slow tests (integration, large data handling, etc.)
        if not has_slow and (
            any(
                keyword in item.name.lower()
                for keyword in ["integration", "large", "concurrent", "memory"]
            ) or has_integration or has_mcmc
        ):
            item.add_marker(pytest.mark.slow)
            has_slow = True
            
        # IMPORTANT: Explicitly mark tests as fast if they are NOT slow/integration/mcmc
        # This ensures the marker filter "not slow and not integration and not mcmc" works
        if not has_slow and not has_integration and not has_mcmc and "fast" not in existing_markers:
            item.add_marker(pytest.mark.fast)
            
        # Mark unit tests explicitly if not already marked
        if "/unit/" in item.nodeid and "unit" not in existing_markers:
            item.add_marker(pytest.mark.unit)


@pytest.fixture(autouse=True)
def reset_matplotlib():
    """Reset matplotlib state between tests to avoid interference."""
    import matplotlib.pyplot as plt

    yield
    plt.close("all")  # Close all figures after each test
    plt.rcdefaults()  # Reset rcParams to defaults


@pytest.fixture(autouse=True)
def suppress_dlascl_warnings():
    """Suppress DLASCL warnings from LAPACK while preserving other stderr output."""

    # For now, let's try a simpler approach - just yield without capturing
    # since stderr redirection can be tricky with multiprocessing
    yield


@pytest.fixture(autouse=True)
def numpy_random_seed():
    """Set a consistent random seed for reproducible tests."""
    np.random.seed(42)
    yield
    # Reset to a different seed to ensure independence
    np.random.seed()


@pytest.fixture(autouse=True)
def track_test_performance_auto(request):
    """Automatically track test performance and apply markers."""
    start_time = time.perf_counter()
    yield
    duration = time.perf_counter() - start_time

    # Automatically mark tests based on execution time
    if duration > 5.0:  # 5 seconds threshold for slow tests
        if not any(marker.name == "slow" for marker in request.node.iter_markers()):
            request.node.add_marker(pytest.mark.slow)
    elif duration < 1.0:  # Fast test threshold
        if not any(marker.name == "fast" for marker in request.node.iter_markers()):
            request.node.add_marker(pytest.mark.fast)

    # Store performance data for potential CI optimization
    setattr(request.node, "_test_duration", duration)


@pytest.fixture(scope="session")
def test_data_dir():
    """Path to test data directory."""
    return Path(__file__).parent / "data"


@pytest.fixture(scope="session")
def sample_config_path():
    """Path to the sample configuration file."""
    return Path(__file__).parent.parent.parent / "homodyne_config.json"


# Import all fixtures from fixtures module to make them available
from homodyne.tests.fixtures.data import (
    dummy_analysis_results,
    dummy_config,
    dummy_correlation_data,
    dummy_hdf5_data,
    dummy_phi_angles,
    dummy_theoretical_data,
    dummy_time_arrays,
    mock_optimization_result,
    test_output_directory,
)


def pytest_report_header(config):
    """Add custom header information to pytest report."""
    import matplotlib
    import numpy as np

    # Check for optional dependencies
    optional_deps = {}

    try:
        import numba

        optional_deps["numba"] = numba.__version__
    except ImportError:
        optional_deps["numba"] = "Not available"

    try:
        import pymc

        optional_deps["pymc"] = pymc.__version__
    except ImportError:
        optional_deps["pymc"] = "Not available"

    try:
        import arviz

        optional_deps["arviz"] = arviz.__version__
    except ImportError:
        optional_deps["arviz"] = "Not available"

    try:
        import corner

        optional_deps["corner"] = corner.__version__
    except ImportError:
        optional_deps["corner"] = "Not available"

    header_lines = [
        f"numpy: {np.__version__}",
        f"matplotlib: {matplotlib.__version__}",
        f"matplotlib backend: {matplotlib.get_backend()}",
        "Optional dependencies:",
    ]

    for name, version in optional_deps.items():
        header_lines.append(f"  {name}: {version}")

    return "\n".join(header_lines)


def pytest_runtest_setup(item):
    """Setup actions before each test."""
    # Clear any potential memory issues
    import gc

    gc.collect()


def pytest_runtest_teardown(item, nextitem):
    """Teardown actions after each test."""
    import gc

    import matplotlib.pyplot as plt

    # Ensure all matplotlib figures are closed
    plt.close("all")

    # Force garbage collection to free memory
    gc.collect()


def pytest_addoption(parser):
    """Add command line options."""
    parser.addoption(
        "--update-baselines",
        action="store_true",
        default=False,
        help="Update performance baselines with current results",
    )
    parser.addoption(
        "--performance-threshold",
        type=float,
        default=1.5,
        help="Performance regression threshold (default: 1.5x)",
    )
    parser.addoption(
        "--skip-slow-performance",
        action="store_true",
        default=False,
        help="Skip slow performance tests",
    )
