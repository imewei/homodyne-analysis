Installation Guide
==================

System Requirements
-------------------

- **Python**: 3.12 or higher
- **Operating System**: Windows, macOS, or Linux
- **Memory**: Minimum 4GB RAM (8GB+ recommended for MCMC)
- **Storage**: ~500MB for full installation with dependencies
- **GPU** (optional): NVIDIA GPU with system CUDA 12.6+ and cuDNN 9.12+ support for accelerated MCMC and JAX computations

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
   On Linux systems with NVIDIA GPUs, JAX will automatically install with system CUDA 12.6+ support for GPU acceleration.

**For MCMC Bayesian Analysis:**

.. code-block:: bash

   pip install homodyne-analysis[mcmc]

.. note::
   This now includes NumPyro for GPU-accelerated MCMC sampling. On Linux systems with NVIDIA GPUs, JAX with system CUDA 12.6+ support is automatically installed.

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

.. note::
   **Conda Environment Integration**: When installed in a conda environment, completion scripts are automatically integrated during installation. Use ``homodyne --install-completion`` to check status or ``homodyne-cleanup`` to remove all environment scripts during uninstallation.

**For Security and Code Quality Tools:**

.. code-block:: bash

   pip install homodyne-analysis[quality]
   # Includes black, isort, flake8, mypy, ruff, bandit, pip-audit

**Enhanced Shell Experience:**

The completion system provides multiple interaction methods:

- **Tab completion**: ``homodyne --method <TAB>`` shows available methods (classical, mcmc, robust, all)
- **Command shortcuts**: ``hc`` (classical), ``hm`` (mcmc), ``hr`` (robust), ``ha`` (all)
- **GPU shortcuts**: ``hgm`` (GPU mcmc), ``hga`` (GPU all) - Linux only
- **Config shortcuts**: ``hconfig``, ``hgconfig`` for configuration files
- **Help reference**: ``homodyne_help`` shows all available options and current config files

.. code-block:: bash

   # After installation, restart shell or reload config
   source ~/.zshrc  # or ~/.bashrc for bash

   # Test shortcuts (always work even if tab completion fails)
   hc --verbose     # homodyne --method classical --verbose
   hgm --config my.json  # GPU-accelerated MCMC (Linux only)
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

**Shell Completion Not Working:**

If tab completion or command shortcuts don't work after installation:

.. code-block:: bash

   # Check completion status (conda environments)
   homodyne --install-completion zsh    # Shows current status
   
   # Manual cleanup if needed
   homodyne-cleanup                     # Remove all conda environment scripts
   
   # Reinstall and restart shell
   pip install --upgrade homodyne-analysis[completion]
   source ~/.zshrc    # or ~/.bashrc for bash

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

System CUDA GPU Acceleration
-----------------------------

The package now supports system CUDA GPU acceleration for MCMC sampling and JAX computations on Linux systems with NVIDIA GPUs.

**System Requirements**

- Linux operating system
- System CUDA 12.6+ installed at ``/usr/local/cuda``
- cuDNN 9.12+ installed in system libraries
- NVIDIA GPU with driver 560.28+

**Automatic System CUDA Support**

When you install with ``[jax]``, ``[mcmc]``, or ``[performance]`` options on a Linux system, JAX will automatically be installed with system CUDA 12.6+ support:

.. code-block:: bash

   # Any of these will include system CUDA GPU support on Linux:
   pip install homodyne-analysis[jax]
   pip install homodyne-analysis[mcmc]        # Includes NumPyro for GPU MCMC
   pip install homodyne-analysis[performance]

   # IMPORTANT: Activate system CUDA GPU support after installation
   source activate_gpu.sh

**Activate and Verify System CUDA GPU**

For system CUDA JAX, you must activate GPU support:

.. code-block:: bash

   # First, activate system CUDA GPU support
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
   # "Using JAX backend with NumPyro NUTS for system CUDA GPU acceleration"

**Command Usage**

.. code-block:: bash

   # CPU-only analysis (reliable, all platforms)
   homodyne --config config.json --method mcmc

   # System CUDA GPU-accelerated analysis (Linux only)
   homodyne-gpu --config config.json --method mcmc

**System CUDA Requirements**

- **Operating System**: Linux (system CUDA GPU acceleration not available on Windows/macOS)
- **Hardware**: NVIDIA GPU with CUDA capability
- **Software**: System CUDA 12.6+ and cuDNN 9.12+ installed
- **Drivers**: NVIDIA driver version 560.28+
- **Memory**: GPU memory requirements depend on problem size

**Troubleshooting System CUDA GPU Issues**

If GPU is not detected:

1. Check NVIDIA drivers: ``nvidia-smi``
2. Verify system CUDA installation: ``nvcc --version`` (should show 12.6+)
3. Check cuDNN installation: ``ls /usr/lib/x86_64-linux-gnu/libcudnn.so.9*``
4. Run GPU activation: ``source activate_gpu.sh``
5. See ``GPU_SETUP.md`` for detailed system CUDA setup and troubleshooting instructions

Getting Help
------------

If you encounter installation issues:

1. Check the `troubleshooting guide <../developer-guide/troubleshooting.html>`_
2. Search existing `GitHub issues <https://github.com/imewei/homodyne/issues>`_
3. Create a new issue with your system details and error messages
