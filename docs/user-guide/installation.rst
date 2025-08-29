Installation Guide
==================

System Requirements
-------------------

- **Python**: 3.12 or higher
- **Operating System**: Windows, macOS, or Linux
- **Memory**: Minimum 4GB RAM (8GB+ recommended for MCMC)
- **Storage**: ~500MB for full installation with dependencies
- **GPU** (optional): NVIDIA GPU with CUDA 12.x support for accelerated MCMC and JAX computations

Quick Installation (Recommended)
--------------------------------

The easiest way to install the Homodyne Analysis package is from PyPI using pip:

**Basic Installation**

.. code-block:: bash

   pip install homodyne-analysis

This installs the core dependencies (numpy, scipy, matplotlib) along with the main package.

**Full Installation with All Features**

.. code-block:: bash

   pip install homodyne-analysis[all]

This includes all optional dependencies: performance acceleration (numba, jax), robust optimization (cvxpy), MCMC analysis (pymc, arviz, pytensor), documentation tools, and development utilities.

Optional Installation Extras
-----------------------------

You can install specific feature sets using pip extras:

**For Enhanced Performance (Numba JIT + JAX acceleration):**

.. code-block:: bash

   pip install homodyne-analysis[performance]
   # OR for JAX-specific features:
   pip install homodyne-analysis[jax]

.. note::
   On Linux systems with NVIDIA GPUs, JAX will automatically install with CUDA 12 support for GPU acceleration.

**For MCMC Bayesian Analysis:**

.. code-block:: bash

   pip install homodyne-analysis[mcmc]

.. note::
   This now includes NumPyro for GPU-accelerated MCMC sampling. On Linux systems with NVIDIA GPUs, JAX with CUDA 12 support is automatically installed.

**For Robust Optimization (Noise-Resistant Methods):**

.. code-block:: bash

   pip install homodyne-analysis[robust]
   # Includes CVXPY for distributionally robust optimization

**For XPCS Data Handling:**

.. code-block:: bash

   pip install homodyne-analysis[data]

**For Documentation Building:**

.. code-block:: bash

   pip install homodyne-analysis[docs]

**For Development:**

.. code-block:: bash

   pip install homodyne-analysis[dev]

**For Gurobi Optimization (Requires License):**

.. code-block:: bash

   pip install homodyne-analysis[gurobi]
   # or manually: pip install gurobipy

**For Shell Tab Completion:**

.. code-block:: bash

   pip install homodyne-analysis[completion]
   # Then install completion for your shell:
   homodyne --install-completion bash  # or zsh, fish, powershell

   # To remove completion later:
   homodyne --uninstall-completion bash  # or zsh, fish, powershell

**For Security and Code Quality Tools:**

.. code-block:: bash

   pip install homodyne-analysis[quality]
   # Includes black, isort, flake8, mypy, ruff, bandit, pip-audit

**Enhanced Shell Experience:**

The completion system provides multiple interaction methods:

- **Tab completion**: ``homodyne --method <TAB>`` shows available options
- **Command shortcuts**: ``hc`` (classical), ``hm`` (mcmc), ``hr`` (robust), ``ha`` (all)
- **Help reference**: ``homodyne_help`` shows all available options and current config files

.. code-block:: bash

   # After installation, restart shell or reload config
   source ~/.zshrc  # or ~/.bashrc for bash

   # Test shortcuts (always work even if tab completion fails)
   hc --verbose     # homodyne --method classical --verbose
   homodyne_help    # Show all options and current config files

**All Dependencies:**

.. code-block:: bash

   pip install homodyne-analysis[all]

Development Installation
------------------------

For development, contributing, or accessing the latest unreleased features:

**Step 1: Clone the Repository**

.. code-block:: bash

   git clone https://github.com/imewei/homodyne.git
   cd homodyne

**Step 2: Install in Development Mode**

.. code-block:: bash

   # Install with all development dependencies
   pip install -e .[all]

   # Or install minimal development setup
   pip install -e .[dev]

Verification
------------

Test your installation:

.. code-block:: python

   import homodyne
   print(f"Homodyne version: {homodyne.__version__}")

   # Test basic functionality
   from homodyne import ConfigManager
   config = ConfigManager()
   print("✅ Installation successful!")

Common Issues
-------------

**Import Errors:**

If you encounter import errors, try reinstalling the package:

.. code-block:: bash

   pip install --upgrade homodyne-analysis

   # Or with all dependencies
   pip install --upgrade homodyne-analysis[all]

**MCMC Issues:**

For MCMC functionality, ensure the mcmc extras are installed:

.. code-block:: bash

   pip install homodyne-analysis[mcmc]

   # Test MCMC availability
   python -c "import pymc; print('PyMC available')"

**Performance Issues:**

For optimal performance, install the performance extras:

.. code-block:: bash

   pip install homodyne-analysis[performance]
   python -c "import numba; print(f'Numba version: {numba.__version__}')"
   python -c "import jax; print(f'JAX devices: {jax.devices()}')"  # Should show GPU if available

**Gurobi License Issues:**

Gurobi optimization requires a valid license. For academic users, free licenses are available:

.. code-block:: bash

   # Install Gurobi
   pip install gurobipy

   # Verify license (should not raise errors)
   python -c "import gurobipy as gp; m = gp.Model(); print('✅ Gurobi license valid')"

For licensing help, visit `Gurobi Academic Licenses <https://www.gurobi.com/academia/academic-program-and-licenses/>`_.

**Package Not Found:**

If pip cannot find the package, ensure you're using the correct name:

.. code-block:: bash

   pip install homodyne-analysis  # Correct package name
   # NOT: pip install homodyne    # This won't work

GPU Acceleration
----------------

The package now supports GPU acceleration for MCMC sampling and JAX computations on Linux systems with NVIDIA GPUs.

**Automatic GPU Support**

When you install with ``[jax]``, ``[mcmc]``, or ``[performance]`` options on a Linux system, JAX will automatically be installed with CUDA 12 support:

.. code-block:: bash

   # Any of these will include GPU support on Linux:
   pip install homodyne-analysis[jax]
   pip install homodyne-analysis[mcmc]        # Includes NumPyro for GPU MCMC
   pip install homodyne-analysis[performance]

   # IMPORTANT: Activate GPU support after installation
   source activate_gpu.sh

**Activate and Verify GPU**

For pip-installed JAX, you must activate GPU support:

.. code-block:: bash

   # First, activate GPU support (required for pip-installed NVIDIA libraries)
   source activate_gpu.sh

   # Then verify GPU detection
   python -c "import jax; print(f'JAX devices: {jax.devices()}')"
   # Should show: [CudaDevice(id=0)]

.. code-block:: python

   # In Python (after activation):
   import jax
   print(f"JAX devices: {jax.devices()}")
   # Output should show: [CudaDevice(id=0), ...] for GPU

   print(f"JAX backend: {jax.default_backend()}")
   # Should show 'gpu' if GPU is available

**Enable GPU for MCMC**

The MCMC module automatically uses GPU acceleration when available:

.. code-block:: python

   from homodyne.optimization.mcmc import HodomyneMCMC

   # GPU acceleration is automatic when use_jax_backend=True (default)
   mcmc = HodomyneMCMC(mode="laminar_flow", use_jax_backend=True)

   # The module will log:
   # "Using JAX backend with NumPyro NUTS for GPU acceleration"

**Manual GPU Configuration**

For different CUDA versions or manual control:

.. code-block:: bash

   # For CUDA 11.x instead of 12.x:
   pip install --upgrade "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

   # Force CPU-only installation (override automatic GPU):
   pip install --upgrade "jax[cpu]"

**GPU Requirements**

- **Operating System**: Linux (GPU acceleration not available on Windows/macOS via pip)
- **Hardware**: NVIDIA GPU with CUDA capability
- **Software**: NVIDIA drivers and CUDA toolkit (12.x for default installation)
- **Memory**: GPU memory requirements depend on problem size

**Troubleshooting GPU Issues**

If GPU is not detected:

1. Check NVIDIA drivers: ``nvidia-smi``
2. Verify CUDA installation: ``nvcc --version``
3. Reinstall JAX with explicit CUDA version
4. Check JAX GPU guide: https://jax.readthedocs.io/en/latest/installation.html#gpu
5. See ``GPU_SETUP.md`` for detailed troubleshooting and setup instructions

Getting Help
------------

If you encounter installation issues:

1. Check the `troubleshooting guide <../developer-guide/troubleshooting.html>`_
2. Search existing `GitHub issues <https://github.com/imewei/homodyne/issues>`_
3. Create a new issue with your system details and error messages
