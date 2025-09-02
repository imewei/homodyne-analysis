"""
Type Safety Helpers for Test Modules
====================================

This module provides standardized patterns for handling optional dependencies
and mock objects in tests, improving type safety while maintaining compatibility
with missing scientific computing dependencies.
"""

from typing import Any, cast, TYPE_CHECKING
from unittest.mock import Mock

def mock_analysis_core() -> Any:
    """Create properly typed mock analysis core."""
    return cast(Any, Mock())

def mock_mcmc_sampler() -> Any:
    """Create properly typed mock MCMC sampler.""" 
    return cast(Any, Mock())

def mock_classical_optimizer() -> Any:
    """Create properly typed mock classical optimizer."""
    return cast(Any, Mock())

def mock_robust_optimizer() -> Any:
    """Create properly typed mock robust optimizer."""
    return cast(Any, Mock())

def check_dependencies_available() -> bool:
    """Check if optional dependencies are available for testing."""
    try:
        import pymc
        import arviz
        return True
    except ImportError:
        return False

def check_performance_deps_available() -> bool:
    """Check if performance testing dependencies are available."""
    try:
        import numba
        import scipy.optimize
        return True
    except ImportError:
        return False

# Conditional imports for TYPE_CHECKING
if TYPE_CHECKING:
    try:
        from homodyne.analysis.core import HomodyneAnalysisCore
        from homodyne.optimization.mcmc import MCMCSampler
        from homodyne.optimization.classical import ClassicalOptimizer
        from homodyne.optimization.robust import RobustHomodyneOptimizer
    except ImportError:
        # Fallback types for type checking when imports fail
        HomodyneAnalysisCore = Any  # type: ignore
        MCMCSampler = Any  # type: ignore  
        ClassicalOptimizer = Any  # type: ignore
        RobustHomodyneOptimizer = Any  # type: ignore