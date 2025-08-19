"""
Pytest Configuration for Rheo-SAXS-XPCS Tests
=============================================

Configuration and shared fixtures for the test suite.
"""

import pytest
import numpy as np
import matplotlib
import warnings
import sys
import os
import shutil
from pathlib import Path

# Use non-GUI matplotlib backend for testing
matplotlib.use("Agg")

# Suppress LAPACK/DLASCL warnings by setting environment variables
os.environ["PYTHONWARNINGS"] = "ignore"
# Some BLAS/LAPACK libraries respect these
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMBA_DISABLE_INTEL_SVML"] = "1"

# Add the project root directory to Python path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


@pytest.fixture(autouse=True, scope="function")
def cleanup_test_artifacts(request):
    """Automatically clean up test artifacts after each test.
    
    This fixture ensures that temporary directories created during testing
    (especially homodyne_results folders) are cleaned up automatically.
    
    SAFETY: Only removes homodyne_results directories that were created
    during the test execution, NOT pre-existing directories with user data.
    """
    # Store initial working directory and check for pre-existing homodyne_results
    initial_cwd = Path.cwd()
    
    # Track which homodyne_results directories existed before the test
    pre_existing_results_dirs = set()
    potential_paths = [
        initial_cwd / "homodyne_results",
    ]
    
    for path in potential_paths:
        if path.exists() and path.is_dir():
            pre_existing_results_dirs.add(path)
    
    yield  # Run the test
    
    # Cleanup after test - ONLY remove directories that didn't exist before
    try:
        current_cwd = Path.cwd()
        cleanup_candidates = [
            initial_cwd / "homodyne_results",
            current_cwd / "homodyne_results",  # In case cwd changed during test
        ]
        
        for cleanup_path in cleanup_candidates:
            if cleanup_path.exists() and cleanup_path.is_dir():
                # SAFETY CHECK: Only remove if this directory wasn't there before the test
                if cleanup_path not in pre_existing_results_dirs:
                    try:
                        shutil.rmtree(cleanup_path)
                        # Only print in verbose mode to avoid cluttering output
                        if getattr(request.config.option, 'verbose', 0) > 1:
                            print(f"\n✓ Cleaned up test-created artifact: {cleanup_path}")
                    except (OSError, PermissionError):
                        # Silently continue if cleanup fails
                        pass
                else:
                    # This directory existed before the test - preserve it
                    if getattr(request.config.option, 'verbose', 0) > 1:
                        print(f"\n⚠ Preserved pre-existing directory: {cleanup_path}")
                    
    except Exception:
        # Don't fail tests due to cleanup issues
        pass


@pytest.fixture(autouse=True, scope="session")
def cleanup_session_artifacts():
    """Clean up any remaining test artifacts after the entire test session.
    
    SAFETY: Only removes homodyne_results in the project root that were
    likely created during testing, not in user working directories.
    """
    yield  # Run all tests
    
    # Final cleanup after all tests - only clean up in the project directory
    try:
        project_root = Path(__file__).parent.parent.parent
        
        # Only clean up homodyne_results in the project root directory
        # This is safe because users typically don't run analysis from the project root
        current_cwd = Path.cwd()
        
        # Only clean up if we're running from the project directory itself
        if current_cwd == project_root:
            homodyne_results_path = project_root / "homodyne_results"
            
            if homodyne_results_path.exists() and homodyne_results_path.is_dir():
                try:
                    shutil.rmtree(homodyne_results_path)
                    print(f"\n✓ Final cleanup: Removed {homodyne_results_path}")
                except (OSError, PermissionError):
                    print(f"\n⚠ Could not remove {homodyne_results_path} - manual cleanup may be needed")
        else:
            # Don't clean up homodyne_results when running from user directories
            pass
    except Exception:
        pass


def pytest_sessionfinish(session, exitstatus):
    """Hook that runs after the entire test session is finished.
    
    SAFETY: Only performs cleanup in safe contexts (project directory)
    to avoid removing user analysis results.
    """
    try:
        # Final cleanup - only in safe contexts
        current_dir = Path.cwd()
        project_root = Path(__file__).parent.parent.parent
        
        # Only clean up homodyne_results if we're in the project directory
        # This prevents accidental deletion of user analysis results
        if current_dir == project_root:
            cleanup_candidates = [
                project_root / "homodyne_results",
            ]
            
            for path in cleanup_candidates:
                if path.exists() and path.is_dir():
                    try:
                        shutil.rmtree(path)
                        if session.config.option.verbose > 0:
                            print(f"\n✓ Test cleanup: Removed {path}")
                    except (OSError, PermissionError):
                        if session.config.option.verbose > 0:
                            print(f"\n⚠ Could not clean up {path}")
        else:
            # Running from user directory - don't clean up homodyne_results
            if session.config.option.verbose > 0:
                print(f"\n⚠ Skipping cleanup in user directory: {current_dir}")
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
    "ignore", category=UserWarning, message=".*SUPERSCRIPT.*missing from font.*"
)

# Suppress all matplotlib UserWarnings to catch font issues
import matplotlib

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

    # Configure warnings filters
    config.addinivalue_line("filterwarnings", "ignore::UserWarning:matplotlib.*")
    config.addinivalue_line("filterwarnings", "ignore::UserWarning:homodyne.*")
    config.addinivalue_line(
        "filterwarnings", "ignore:.*Glyph.*missing from font.*:UserWarning"
    )


def pytest_collection_modifyitems(config, items):
    """Automatically mark tests based on their location and content."""
    for item in items:
        # Mark integration tests
        if "test_integration" in item.nodeid:
            item.add_marker(pytest.mark.integration)

        # Mark plotting tests
        if "test_plotting" in item.nodeid or "plot" in item.name.lower():
            item.add_marker(pytest.mark.plotting)

        # Mark I/O tests
        if "test_io" in item.nodeid or "io_utils" in item.nodeid:
            item.add_marker(pytest.mark.io)

        # Mark slow tests (integration, large data handling, etc.)
        if any(
            keyword in item.name.lower()
            for keyword in ["integration", "large", "concurrent", "memory"]
        ):
            item.add_marker(pytest.mark.slow)


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
    import contextlib
    from io import StringIO
    import threading

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


@pytest.fixture(scope="session")
def test_data_dir():
    """Path to test data directory."""
    return Path(__file__).parent / "data"


@pytest.fixture(scope="session")
def sample_config_path():
    """Path to the sample configuration file."""
    return Path(__file__).parent.parent.parent / "homodyne_config.json"


# Import all fixtures from fixtures.py to make them available
from .fixtures import *


def pytest_report_header(config):
    """Add custom header information to pytest report."""
    import numpy as np
    import matplotlib

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
