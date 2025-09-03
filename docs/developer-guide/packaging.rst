Packaging and Distribution
==========================

This document describes the packaging architecture and distribution options for the homodyne package, with a focus on the isolated MCMC backend system.

Overview
--------

The homodyne package features an **isolated MCMC backend architecture** that completely separates PyMC CPU and NumPyro/JAX GPU implementations to prevent PyTensor/JAX conflicts while maintaining optimal performance across all platforms.

Isolated MCMC Backend Architecture
-----------------------------------

**Revolutionary Backend Separation**

The package now includes completely isolated MCMC backends:

- **CPU Backend**: Pure PyMC implementation (``mcmc_cpu_backend.py``) - completely isolated from JAX
- **GPU Backend**: Pure NumPyro/JAX implementation (``mcmc_gpu_backend.py``) - completely isolated from PyMC
- **Unified Interface**: Seamless switching between backends based on environment and user intent

**Backend Isolation Benefits**

1. **Complete Isolation**: No more PyTensor/JAX namespace conflicts
2. **Platform Compatibility**: 
   - Linux: Full GPU support for GPU backend
   - macOS/Windows: CPU-only for both backends (JAX CPU fallback for GPU backend)
3. **Dependency Management**: Users can install only what they need
4. **Backward Compatibility**: Existing installations continue to work
5. **Development Flexibility**: Each backend can evolve independently

Architecture Components
-----------------------

**Backend Wrapper Files**

.. code-block:: text

   homodyne/optimization/
   ├── mcmc.py                    # Legacy unified MCMC (deprecated)
   ├── mcmc_gpu.py               # Legacy GPU MCMC (deprecated)
   ├── mcmc_cpu_backend.py       # Isolated PyMC CPU backend ✅
   ├── mcmc_gpu_backend.py       # Isolated NumPyro/JAX GPU backend ✅
   └── __init__.py               # Updated imports

**Backend Selection Logic**

The ``get_mcmc_backend()`` function in ``run_homodyne.py`` automatically selects the appropriate backend:

.. code-block:: python

   def get_mcmc_backend():
       """Get appropriate MCMC backend based on environment and intent."""
       gpu_intent = os.environ.get("HOMODYNE_GPU_INTENT", "false").lower() == "true"
       
       if gpu_intent:
           # GPU mode requested - use isolated NumPyro backend
           from .optimization.mcmc_gpu_backend import run_gpu_mcmc_analysis
           return run_gpu_mcmc_analysis, "NumPyro_GPU_JAX", True
       else:
           # CPU mode - use isolated PyMC backend  
           from .optimization.mcmc_cpu_backend import run_cpu_mcmc_analysis
           return run_cpu_mcmc_analysis, "PyMC_CPU", False

**Command Routing**

+------------------+----------------+-------------------+--------------------+
| Command          | Backend        | Implementation    | Environment        |
+==================+================+===================+====================+
| ``homodyne``     | **PyMC CPU**   | mcmc_cpu_backend  | All platforms      |
+------------------+----------------+-------------------+--------------------+
| ``homodyne-gpu`` | **NumPyro**    | mcmc_gpu_backend  | Linux (GPU), CPU   |
+------------------+----------------+-------------------+--------------------+

Installation Options
--------------------

**Package Extras Structure**

The package now provides granular installation options for isolated backends:

.. code-block:: toml

   [project.optional-dependencies]
   # Isolated MCMC backends
   mcmc = [
       "pymc>=5.10.0",
       "arviz>=0.15.1", 
       "pytensor>=2.18.0",
       "corner>=2.2.0"
   ]
   mcmc-gpu = [
       "numpyro>=0.13.0",
       "jax[cuda12]>=0.4.20",
       "jaxlib>=0.4.20"
   ]
   mcmc-all = [
       "homodyne-analysis[mcmc]",
       "homodyne-analysis[mcmc-gpu]"
   ]
   all = [
       "homodyne-analysis[mcmc-all]",
       # ... other dependencies
   ]

**Installation Commands**

Basic Installation
~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   pip install homodyne-analysis

CPU-only MCMC (Pure PyMC)
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   pip install homodyne-analysis[mcmc]
   # or
   pip install -r requirements-mcmc-cpu.txt

GPU MCMC (Pure NumPyro/JAX with CPU fallback)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   pip install homodyne-analysis[mcmc-gpu]  
   # or
   pip install -r requirements-mcmc-gpu.txt

Both Backends
~~~~~~~~~~~~~

.. code-block:: bash

   pip install homodyne-analysis[mcmc-all]
   # or  
   pip install -r requirements-mcmc-all.txt

Everything
~~~~~~~~~~

.. code-block:: bash

   pip install homodyne-analysis[all]
   # or
   pip install -r requirements-all.txt

Requirements Files
------------------

**New Isolated Backend Requirements**

The package now includes specific requirements files for each backend:

- ``requirements-mcmc-cpu.txt``: Pure PyMC CPU backend dependencies
- ``requirements-mcmc-gpu.txt``: Pure NumPyro/JAX GPU backend dependencies  
- ``requirements-mcmc-all.txt``: Combined backend dependencies

**Updated Files**

- ``requirements-optional.txt``: Updated with isolated architecture notes
- ``requirements-all.txt``: References new MCMC backend files
- ``requirements-dev.txt``: Development dependencies for both backends

Usage Examples
--------------

**CPU Backend (Pure PyMC)**

.. code-block:: bash

   # Uses isolated CPU backend
   homodyne --method mcmc --config config.json

   # Environment variable (optional)
   HOMODYNE_GPU_INTENT=false homodyne --method mcmc

**GPU Backend (Pure NumPyro/JAX)**

.. code-block:: bash

   # Uses isolated GPU backend with CPU fallback
   homodyne-gpu --method mcmc --config config.json

   # Environment variable (optional)  
   HOMODYNE_GPU_INTENT=true homodyne --method mcmc

**Python API Usage**

.. code-block:: python

   import os
   from homodyne.run_homodyne import get_mcmc_backend
   
   # Force specific backend
   os.environ["HOMODYNE_GPU_INTENT"] = "true"  # or "false"
   
   mcmc_function, backend_name, has_gpu = get_mcmc_backend()
   print(f"Using backend: {backend_name}")
   
   # Use the isolated backend function
   results = mcmc_function(
       analysis_core=analyzer,
       config=config,
       c2_experimental=data,
       phi_angles=angles,
       filter_angles_for_optimization=True
   )

Packaging Files
---------------

**pyproject.toml Updates**

.. code-block:: toml

   [project]
   keywords = [
       "xpcs", "correlation", "scattering", "mcmc", 
       "bayesian", "optimization", "isolated-backends"
   ]
   
   [project.scripts]
   homodyne = "homodyne.run_homodyne:main"
   homodyne-gpu = "homodyne.runtime.gpu.gpu_wrapper:main"

   [tool.setuptools.package-data]
   homodyne = [
       "optimization/mcmc_cpu_backend.py",
       "optimization/mcmc_gpu_backend.py",
       # ... other files
   ]

**MANIFEST.in Updates**

.. code-block:: text

   # Isolated MCMC Backend Architecture - Complete PyMC/NumPyro separation
   include homodyne/optimization/mcmc_cpu_backend.py
   include homodyne/optimization/mcmc_gpu_backend.py
   
   # Isolated backend requirements
   include requirements-mcmc-cpu.txt
   include requirements-mcmc-gpu.txt
   include requirements-mcmc-all.txt

**setup.py Updates**

Updated installation messages and command descriptions to reflect backend separation:

.. code-block:: python

   setup(
       # ... other parameters
       long_description="""
       Advanced XPCS analysis with isolated MCMC backends:
       - Pure PyMC CPU backend (cross-platform)
       - Pure NumPyro/JAX GPU backend (Linux with CPU fallback)
       - Complete PyTensor/JAX conflict resolution
       """,
       entry_points={
           'console_scripts': [
               'homodyne=homodyne.run_homodyne:main',
               'homodyne-gpu=homodyne.runtime.gpu.gpu_wrapper:main',
           ],
       },
   )

Testing and Validation
-----------------------

**Backend Isolation Testing**

The package includes comprehensive tests for backend isolation:

.. code-block:: bash

   # Test isolated CPU backend only
   pytest homodyne/tests/unit/optimization/test_mcmc.py -v
   
   # Test isolated GPU backend only  
   pytest homodyne/tests/unit/optimization/test_mcmc_gpu.py -v
   
   # Test backend isolation validation
   pytest homodyne/tests/unit/optimization/test_mcmc_cross_validation.py -v

**Installation Testing**

.. code-block:: bash

   # Test CPU-only installation
   pip install homodyne-analysis[mcmc]
   python -c "from homodyne.optimization.mcmc_cpu_backend import is_cpu_mcmc_available; print(is_cpu_mcmc_available())"
   
   # Test GPU installation  
   pip install homodyne-analysis[mcmc-gpu]
   python -c "from homodyne.optimization.mcmc_gpu_backend import is_gpu_mcmc_available; print(is_gpu_mcmc_available())"

**Compatibility Testing**

The isolated backend architecture maintains backward compatibility:

- All existing ``pip install homodyne-analysis[all]`` installations get both backends
- Legacy code continues to work without modification
- Gradual migration path to isolated backends

Distribution Strategy
---------------------

**PyPI Distribution**

The package is distributed on PyPI with isolated backend support:

- **Main package**: ``homodyne-analysis`` (includes both backends with ``[all]``)
- **Granular extras**: ``[mcmc]``, ``[mcmc-gpu]``, ``[mcmc-all]``
- **Platform-specific**: Automatic JAX CUDA selection on Linux

**Conda Distribution** (Future)

Planned conda-forge distribution with environment separation:

.. code-block:: bash

   # CPU-only environment
   conda create -n homodyne-cpu -c conda-forge homodyne-analysis-cpu
   
   # GPU environment  
   conda create -n homodyne-gpu -c conda-forge homodyne-analysis-gpu

**Docker Images** (Future)

Planned Docker images for each backend:

.. code-block:: bash

   # CPU-only image
   docker pull homodyne/homodyne-cpu:latest
   
   # GPU-enabled image
   docker pull homodyne/homodyne-gpu:latest

Security and Quality
--------------------

**Dependency Isolation**

The isolated backend architecture enhances security:

- **Reduced attack surface**: Install only needed dependencies
- **Conflict prevention**: No more PyTensor/JAX namespace issues  
- **Version management**: Independent backend versioning

**Quality Assurance**

.. code-block:: bash

   # Security scanning with Bandit
   bandit -r homodyne/ --exclude homodyne/tests/

   # Dependency auditing
   pip-audit --desc

   # Code quality checks
   ruff check homodyne/
   mypy homodyne/

Migration Guide
---------------

**For Existing Users**

Existing installations continue to work without changes:

.. code-block:: bash

   # Existing installations - no action needed
   pip install homodyne-analysis[all]  # Gets both backends

**For New Installations**

Choose the appropriate backend for your needs:

.. code-block:: bash

   # CPU-only (recommended for most users)
   pip install homodyne-analysis[mcmc]
   
   # GPU acceleration (Linux with NVIDIA GPU)
   pip install homodyne-analysis[mcmc-gpu]
   
   # Both backends (maximum flexibility)
   pip install homodyne-analysis[mcmc-all]

**Code Migration**

No code changes required - the isolated backends provide the same interface:

.. code-block:: python

   # This continues to work unchanged
   from homodyne.optimization.mcmc import MCMCSampler
   sampler = MCMCSampler(analyzer, config)
   results = sampler.run_mcmc_analysis()

Performance Impact
------------------

**Startup Time**

- **Faster imports**: Only load needed backend dependencies
- **Reduced memory**: Smaller initial footprint
- **Quick initialization**: No more PyTensor/JAX conflicts

**Runtime Performance**

- **CPU Backend**: Consistent PyMC performance across platforms
- **GPU Backend**: Full NumPyro/JAX acceleration on Linux
- **No conflicts**: Eliminated PyTensor/JAX interference

**Memory Usage**

- **Isolated backends**: Reduced memory overlap
- **Platform optimization**: JAX GPU memory management on Linux
- **Fallback efficiency**: Smooth CPU fallback when needed

Future Enhancements
-------------------

**Planned Improvements**

1. **Auto-detection**: Automatic backend selection based on hardware
2. **Hybrid mode**: Combine backends for optimal performance
3. **Cloud backends**: Remote computation support
4. **Custom backends**: Plugin architecture for user backends

**Community Contributions**

The isolated backend architecture enables community contributions:

- **Backend plugins**: Add new sampling methods
- **Platform support**: Extend to new hardware/OS combinations  
- **Performance optimization**: Backend-specific improvements