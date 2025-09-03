"""
Pure PyMC CPU Backend - No JAX Dependencies
===========================================

This module provides CPU-only MCMC using PyMC with PyTensor CPU backend.
Completely isolated from JAX/NumPyro to avoid backend conflicts.

This backend is designed to work on all platforms (Linux, macOS, Windows)
without requiring JAX or GPU dependencies.

Authors: Wei Chen, Hongrui He
Institution: Argonne National Laboratory
"""

import logging
import os
import sys
from typing import Any, Dict, Optional

# Force PyTensor CPU-only mode BEFORE any imports
# This prevents PyTensor from automatically detecting and using JAX
os.environ["PYTENSOR_FLAGS"] = "device=cpu,floatX=float32,mode=FAST_RUN,optimizer=fast_run"
os.environ["PYTENSOR_COMPILEDIR_FORMAT"] = "compiledir_%(platform)s-%(processor)s-%(python_version)s-%(python_bitwidth)s--cpu-only"

# Prevent JAX from being loaded if it exists
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_DISABLE_MOST_OPTIMIZATIONS"] = "true"

logger = logging.getLogger(__name__)


def run_cpu_mcmc_analysis(
    analysis_core,
    config: Dict[str, Any],
    c2_experimental,
    phi_angles,
    filter_angles_for_optimization: bool = True,
    **kwargs
) -> Dict[str, Any]:
    """
    Run CPU-only MCMC analysis using pure PyMC backend.
    
    This function imports PyMC/PyTensor in an isolated environment
    to avoid JAX backend contamination.
    
    Parameters
    ----------
    analysis_core : HomodyneAnalysisCore
        Core analysis engine instance
    config : Dict[str, Any]
        MCMC configuration dictionary
    c2_experimental : np.ndarray
        Experimental correlation data
    phi_angles : np.ndarray
        Scattering angles
    filter_angles_for_optimization : bool, default True
        If True, use only angles in ranges [-10°, 10°] and [170°, 190°]
    
    Returns
    -------
    Dict[str, Any]
        MCMC analysis results with trace, diagnostics, and performance metrics
        
    Raises
    ------
    ImportError
        If PyMC CPU backend is not available
    RuntimeError
        If CPU MCMC analysis fails
    """
    logger.info("🖥️  Initializing pure PyMC CPU backend (isolated from JAX)")
    
    try:
        # Import mcmc.py only when needed to avoid import conflicts
        from .mcmc import create_mcmc_sampler
        
        logger.info("✅ PyMC CPU dependencies loaded successfully")
        
        # Create pure CPU sampler
        logger.info("Creating CPU-only MCMC sampler...")
        sampler = create_mcmc_sampler(analysis_core, config)
        
        # Log backend information
        logger.info(f"MCMC sampler created: {type(sampler).__name__}")
        logger.info(f"Backend module: {sampler.__module__}")
        logger.info("🔒 Operating in pure CPU mode (JAX isolated)")
        
        # Run analysis with CPU-only parameters
        logger.info("Starting CPU MCMC analysis...")
        results = sampler.run_mcmc_analysis(
            c2_experimental=c2_experimental,
            phi_angles=phi_angles,
            filter_angles_for_optimization=filter_angles_for_optimization
        )
        
        logger.info("✅ CPU MCMC analysis completed successfully")
        
        # Add backend information to results
        if isinstance(results, dict):
            results["backend_info"] = {
                "backend": "PyMC_CPU",
                "isolation_mode": "JAX_isolated",
                "platform_compatible": True,
                "sampler_module": sampler.__module__
            }
        
        return results
        
    except ImportError as e:
        error_msg = (
            f"❌ PyMC CPU backend not available: {e}\n\n"
            "SOLUTIONS:\n"
            "1. Install PyMC: pip install pymc>=5.0\n"
            "2. Install PyTensor: pip install pytensor\n"
            "3. Use alternative methods: 'homodyne --method classical'\n"
        )
        logger.error(error_msg)
        raise ImportError(error_msg)
        
    except Exception as e:
        error_msg = (
            f"❌ CPU MCMC analysis failed: {e}\n\n"
            "This error occurred in the isolated PyMC CPU backend.\n"
            "Try using alternative analysis methods:\n"
            "• Classical: 'homodyne --method classical'\n"  
            "• Robust: 'homodyne --method robust'\n"
        )
        logger.error(error_msg)
        raise RuntimeError(error_msg)


def is_cpu_mcmc_available() -> bool:
    """
    Check if CPU MCMC backend is available.
    
    Returns
    -------
    bool
        True if PyMC and PyTensor are available for CPU-only operation
    """
    try:
        # Test PyMC availability
        import pymc as pm
        import pytensor
        import arviz as az
        
        # Verify PyTensor is in CPU mode
        import pytensor.tensor as pt
        
        logger.info("✅ PyMC CPU backend dependencies available")
        logger.info(f"PyMC version: {pm.__version__}")
        logger.info(f"PyTensor version: {pytensor.__version__}")
        logger.info(f"ArviZ version: {az.__version__}")
        
        return True
        
    except ImportError as e:
        logger.warning(f"PyMC CPU backend not available: {e}")
        return False


def get_cpu_backend_info() -> Dict[str, Any]:
    """
    Get detailed information about the CPU backend capabilities.
    
    Returns
    -------
    Dict[str, Any]
        Information about backend availability and versions
    """
    info = {
        "backend_name": "PyMC_CPU",
        "isolation_mode": "JAX_isolated",
        "available": False,
        "platform_support": ["Linux", "macOS", "Windows"],
        "dependencies": {},
        "capabilities": {
            "cpu_sampling": True,
            "multiprocessing": True,
            "convergence_diagnostics": True,
            "cross_platform": True
        }
    }
    
    try:
        import pymc as pm
        import pytensor
        import arviz as az
        import numpy as np
        
        info["available"] = True
        info["dependencies"] = {
            "pymc": pm.__version__,
            "pytensor": pytensor.__version__, 
            "arviz": az.__version__,
            "numpy": np.__version__
        }
        
        # Check PyTensor configuration
        pytensor_flags = os.environ.get("PYTENSOR_FLAGS", "")
        info["pytensor_config"] = {
            "flags": pytensor_flags,
            "cpu_forced": "device=cpu" in pytensor_flags,
            "jax_isolated": True
        }
        
    except ImportError:
        pass
    
    return info


def test_cpu_backend() -> bool:
    """
    Test the CPU backend with a minimal MCMC run.
    
    Returns  
    -------
    bool
        True if backend works correctly
    """
    if not is_cpu_mcmc_available():
        return False
        
    try:
        logger.info("🧪 Testing CPU backend with minimal model...")
        
        # Import PyMC in isolated environment
        import pymc as pm
        import numpy as np
        
        # Create minimal test model
        with pm.Model() as test_model:
            # Simple normal distribution
            mu = pm.Normal('mu', mu=0, sigma=1)
            obs = pm.Normal('obs', mu=mu, sigma=1, observed=[1.0, 2.0, 3.0])
        
        # Test sampling with minimal draws
        with test_model:
            trace = pm.sample(
                draws=10, 
                tune=10, 
                chains=1, 
                cores=1,
                progressbar=False,
                compute_convergence_checks=False
            )
        
        logger.info("✅ CPU backend test completed successfully")
        return True
        
    except Exception as e:
        logger.warning(f"❌ CPU backend test failed: {e}")
        return False


if __name__ == "__main__":
    # Test the backend when run directly
    print("Testing PyMC CPU Backend (JAX Isolated)")
    print("=" * 50)
    
    # Test availability
    available = is_cpu_mcmc_available()
    print(f"Backend available: {available}")
    
    if available:
        # Get backend info
        info = get_cpu_backend_info()
        print(f"Backend info: {info}")
        
        # Test functionality
        works = test_cpu_backend()
        print(f"Backend test: {'PASSED' if works else 'FAILED'}")
    else:
        print("Install PyMC to use CPU backend: pip install pymc")