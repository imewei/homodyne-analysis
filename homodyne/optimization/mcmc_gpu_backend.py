"""
Pure NumPyro GPU/JAX Backend - No PyMC Dependencies
===================================================

This module provides GPU/JAX MCMC using NumPyro.
Can fall back to JAX CPU mode. Completely isolated from PyMC.

This backend is optimized for high-performance computing with GPU acceleration
and can gracefully fall back to JAX CPU mode when GPU is not available.

Authors: Wei Chen, Hongrui He
Institution: Argonne National Laboratory
"""

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)


def configure_jax_backend(
    force_cpu: bool = False, gpu_memory_fraction: float = 0.8
) -> dict[str, Any]:
    """
    Configure JAX backend for GPU or CPU operation.

    Parameters
    ----------
    force_cpu : bool, default False
        If True, force JAX to use CPU mode only
    gpu_memory_fraction : float, default 0.8
        Fraction of GPU memory to allocate (0.1 to 1.0)

    Returns
    -------
    Dict[str, Any]
        Configuration information and detected hardware
    """
    config_info = {
        "requested_mode": "CPU" if force_cpu else "GPU_with_CPU_fallback",
        "jax_configured": False,
        "gpu_available": False,
        "gpu_count": 0,
        "backend": "unknown",
    }

    if force_cpu:
        logger.info("🖥️  Configuring JAX for CPU-only operation")
        os.environ["JAX_PLATFORMS"] = "cpu"
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        config_info["backend"] = "JAX_CPU"
    else:
        logger.info("🚀 Configuring JAX for GPU operation with CPU fallback")
        # Remove CPU-only restrictions
        os.environ.pop("JAX_PLATFORMS", None)
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)

        # Set GPU memory fraction
        os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = str(gpu_memory_fraction)

    try:
        import jax

        config_info["jax_configured"] = True

        # Detect available devices
        devices = jax.devices()
        gpu_devices = jax.devices("gpu")
        cpu_devices = jax.devices("cpu")

        config_info.update(
            {
                "gpu_available": len(gpu_devices) > 0,
                "gpu_count": len(gpu_devices),
                "cpu_count": len(cpu_devices),
                "total_devices": len(devices),
                "default_device": str(jax.devices()[0]),
            }
        )

        if not force_cpu and len(gpu_devices) > 0:
            config_info["backend"] = "JAX_GPU"
            logger.info(
                f"✅ JAX configured for GPU: {len(gpu_devices)} GPU(s) available"
            )
            for i, device in enumerate(gpu_devices):
                logger.info(f"   GPU {i}: {device}")
        else:
            config_info["backend"] = "JAX_CPU"
            logger.info(f"🖥️  JAX configured for CPU: {len(cpu_devices)} CPU device(s)")

    except ImportError:
        logger.warning("JAX not available for configuration")
    except Exception as e:
        logger.warning(f"JAX configuration warning: {e}")

    return config_info


def run_gpu_mcmc_analysis(
    analysis_core,
    config: dict[str, Any],
    c2_experimental,
    phi_angles,
    filter_angles_for_optimization: bool = True,
    is_static_mode: bool = False,
    effective_param_count: int = 7,
    force_cpu: bool = False,
    **kwargs,
) -> dict[str, Any]:
    """
    Run GPU/JAX MCMC analysis using pure NumPyro backend.

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
    is_static_mode : bool, default False
        Whether to use static analysis mode
    effective_param_count : int, default 7
        Number of effective parameters
    force_cpu : bool, default False
        If True, force JAX to use CPU mode

    Returns
    -------
    Dict[str, Any]
        MCMC analysis results with trace, diagnostics, and performance metrics

    Raises
    ------
    ImportError
        If NumPyro GPU/JAX backend is not available
    RuntimeError
        If GPU/JAX MCMC analysis fails
    """
    # Configure JAX backend
    jax_config = configure_jax_backend(force_cpu)

    logger.info(
        f"🚀 Initializing NumPyro {jax_config['backend']} backend (PyMC isolated)"
    )

    try:
        # Import mcmc_gpu.py only when needed
        from .mcmc_gpu import create_gpu_mcmc_sampler

        logger.info("✅ NumPyro/JAX dependencies loaded successfully")

        # Create NumPyro sampler
        logger.info("Creating GPU/JAX MCMC sampler...")
        sampler = create_gpu_mcmc_sampler(analysis_core, config)

        # Log backend information
        logger.info(f"MCMC sampler created: {type(sampler).__name__}")
        logger.info(f"Backend module: {sampler.__module__}")
        logger.info(f"🔒 Operating in {jax_config['backend']} mode (PyMC isolated)")

        if jax_config["gpu_available"] and not force_cpu:
            logger.info(f"🎮 Using {jax_config['gpu_count']} GPU(s) for acceleration")
        else:
            logger.info("🖥️  Using JAX CPU mode for computation")

        # Run analysis with GPU/JAX parameters (including static mode info)
        logger.info("Starting NumPyro MCMC analysis...")
        results = sampler.run_mcmc_analysis(
            c2_experimental=c2_experimental,
            phi_angles=phi_angles,
            filter_angles_for_optimization=filter_angles_for_optimization,
            is_static_mode=is_static_mode,
            effective_param_count=effective_param_count,
        )

        logger.info("✅ NumPyro MCMC analysis completed successfully")

        # Add backend information to results
        if isinstance(results, dict):
            results["backend_info"] = {
                "backend": jax_config["backend"],
                "isolation_mode": "PyMC_isolated",
                "gpu_used": jax_config["gpu_available"] and not force_cpu,
                "gpu_count": jax_config["gpu_count"],
                "sampler_module": sampler.__module__,
                "jax_config": jax_config,
            }

        return results

    except ImportError as e:
        error_msg = (
            f"❌ NumPyro GPU/JAX backend not available: {e}\n\n"
            "SOLUTIONS:\n"
            "1. Install NumPyro: pip install numpyro\n"
            "2. Install JAX: pip install jax jaxlib\n"
            "3. For GPU: pip install jax[cuda] (Linux only)\n"
            "4. Use CPU MCMC: 'homodyne --method mcmc'\n"
            "5. Use classical methods: 'homodyne --method classical'\n"
        )
        logger.error(error_msg)
        raise ImportError(error_msg)

    except Exception as e:
        if force_cpu:
            error_msg = (
                f"❌ JAX CPU MCMC failed: {e}\n\n"
                "This error occurred in JAX CPU fallback mode.\n"
                "Try alternative analysis methods:\n"
                "• CPU MCMC: 'homodyne --method mcmc'\n"
                "• Classical: 'homodyne --method classical'\n"
            )
            logger.error(error_msg)
            raise RuntimeError(error_msg)
        else:
            # Try falling back to CPU mode
            logger.warning(f"GPU MCMC failed ({e}), trying JAX CPU fallback...")
            try:
                return run_gpu_mcmc_analysis(
                    analysis_core=analysis_core,
                    config=config,
                    c2_experimental=c2_experimental,
                    phi_angles=phi_angles,
                    filter_angles_for_optimization=filter_angles_for_optimization,
                    is_static_mode=is_static_mode,
                    effective_param_count=effective_param_count,
                    force_cpu=True,
                    **kwargs,
                )
            except Exception as fallback_e:
                error_msg = (
                    f"❌ Both GPU and JAX CPU MCMC failed.\n\n"
                    f"GPU error: {e}\n"
                    f"CPU fallback error: {fallback_e}\n\n"
                    "SOLUTIONS:\n"
                    "• Use CPU MCMC: 'homodyne --method mcmc'\n"
                    "• Use classical methods: 'homodyne --method classical'\n"
                    "• Check JAX installation: python -c 'import jax; print(jax.devices())'\n"
                )
                logger.error(error_msg)
                raise RuntimeError(error_msg)


def is_gpu_mcmc_available() -> bool:
    """
    Check if GPU/JAX MCMC backend is available.

    Returns
    -------
    bool
        True if NumPyro and JAX are available
    """
    try:
        import jax
        import numpyro

        logger.info("✅ NumPyro/JAX backend dependencies available")
        logger.info(f"JAX version: {jax.__version__}")
        logger.info(f"NumPyro version: {numpyro.__version__}")
        return True
    except ImportError as e:
        logger.warning(f"NumPyro/JAX backend not available: {e}")
        return False


def is_gpu_hardware_available() -> bool:
    """
    Check if GPU hardware is actually available for computation.

    Returns
    -------
    bool
        True if GPU hardware is detected by JAX
    """
    try:
        import jax

        gpu_devices = jax.devices("gpu")
        gpu_available = len(gpu_devices) > 0

        if gpu_available:
            logger.info(f"🎮 GPU hardware detected: {len(gpu_devices)} device(s)")
            for i, device in enumerate(gpu_devices):
                logger.info(f"   GPU {i}: {device}")
        else:
            logger.info("🖥️  No GPU hardware detected, JAX will use CPU")

        return gpu_available
    except:
        return False


def get_gpu_backend_info() -> dict[str, Any]:
    """
    Get detailed information about the GPU/JAX backend capabilities.

    Returns
    -------
    Dict[str, Any]
        Information about backend availability, hardware, and versions
    """
    info = {
        "backend_name": "NumPyro_GPU_JAX",
        "isolation_mode": "PyMC_isolated",
        "available": False,
        "gpu_hardware": False,
        "platform_support": ["Linux", "macOS", "Windows"],
        "dependencies": {},
        "capabilities": {
            "gpu_acceleration": False,
            "jax_cpu_fallback": False,
            "high_performance": False,
            "automatic_differentiation": True,
        },
    }

    try:
        import jax
        import numpy as np
        import numpyro

        info["available"] = True
        info["dependencies"] = {
            "jax": jax.__version__,
            "numpyro": numpyro.__version__,
            "numpy": np.__version__,
        }

        # Check hardware capabilities
        devices = jax.devices()
        gpu_devices = jax.devices("gpu")

        info.update(
            {
                "gpu_hardware": len(gpu_devices) > 0,
                "gpu_count": len(gpu_devices),
                "total_devices": len(devices),
                "device_list": [str(d) for d in devices],
            }
        )

        info["capabilities"].update(
            {
                "gpu_acceleration": len(gpu_devices) > 0,
                "jax_cpu_fallback": True,
                "high_performance": True,
            }
        )

    except ImportError:
        pass

    return info


def test_gpu_backend(force_cpu: bool = False) -> bool:
    """
    Test the GPU/JAX backend with a minimal MCMC run.

    Parameters
    ----------
    force_cpu : bool, default False
        If True, test JAX CPU mode specifically

    Returns
    -------
    bool
        True if backend works correctly
    """
    if not is_gpu_mcmc_available():
        return False

    try:
        mode = "JAX CPU" if force_cpu else "GPU/JAX"
        logger.info(f"🧪 Testing {mode} backend with minimal model...")

        # Configure JAX
        configure_jax_backend(force_cpu)

        # Import NumPyro in isolated environment
        import jax
        import jax.numpy as jnp
        import numpyro
        import numpyro.distributions as dist
        from numpyro.infer import MCMC, NUTS

        # Create minimal test model
        def test_model():
            mu = numpyro.sample("mu", dist.Normal(0, 1))
            numpyro.sample("obs", dist.Normal(mu, 1), obs=jnp.array([1.0, 2.0, 3.0]))

        # Test sampling with minimal draws
        nuts_kernel = NUTS(test_model)
        mcmc = MCMC(nuts_kernel, num_warmup=10, num_samples=10)
        mcmc.run(jax.random.PRNGKey(0))

        logger.info(f"✅ {mode} backend test completed successfully")
        return True

    except Exception as e:
        logger.warning(f"❌ {mode} backend test failed: {e}")
        return False


if __name__ == "__main__":
    # Test the backend when run directly
    print("Testing NumPyro GPU/JAX Backend (PyMC Isolated)")
    print("=" * 50)

    # Test availability
    available = is_gpu_mcmc_available()
    print(f"Backend available: {available}")

    if available:
        # Get backend info
        info = get_gpu_backend_info()
        print(f"Backend info: {info}")

        # Test GPU functionality
        if info["gpu_hardware"]:
            print("Testing GPU mode...")
            gpu_works = test_gpu_backend(force_cpu=False)
            print(f"GPU backend test: {'PASSED' if gpu_works else 'FAILED'}")

        # Test CPU fallback
        print("Testing JAX CPU mode...")
        cpu_works = test_gpu_backend(force_cpu=True)
        print(f"JAX CPU backend test: {'PASSED' if cpu_works else 'FAILED'}")

    else:
        print("Install NumPyro/JAX to use GPU backend: pip install numpyro jax")
