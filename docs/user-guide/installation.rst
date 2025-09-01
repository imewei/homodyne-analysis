Installation Guide
==================

System Requirements
-------------------

- **Python**: 3.12 or higher
- **Operating System**: Windows, macOS, or Linux
- **Memory**: Minimum 4GB RAM (8GB+ recommended for MCMC)
- **Storage**: ~500MB for full installation with dependencies
- **GPU** (optional, Linux only): NVIDIA GPU with system CUDA 12.6+ and cuDNN 9.12+ support for JAX backend GPU acceleration with PyTensor environment variable auto-configuration

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
   On Linux systems with NVIDIA GPUs, JAX will automatically install with system CUDA 12.6+ support for GPU acceleration. PyTensor environment variables (``PYTENSOR_FLAGS``) are automatically configured to use CPU mode and avoid C compilation issues.

**For MCMC Bayesian Analysis:**

.. code-block:: bash

   pip install homodyne-analysis[mcmc]

.. note::
   This now includes NumPyro for JAX backend GPU-accelerated MCMC sampling. On Linux systems with NVIDIA GPUs, JAX with system CUDA 12.6+ support is automatically installed, and PyTensor environment variables are auto-configured for optimal performance (JAX handles GPU operations, PyTensor uses CPU mode).

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

**Unified Post-Installation Setup:**

.. code-block:: bash

   pip install homodyne-analysis[all]
   
   # Run the unified post-install setup (recommended)
   homodyne-post-install --shell zsh --gpu --advanced
   
   # Or interactive setup
   homodyne-post-install

.. note::
   **Unified Post-Install System**: The package now includes a unified post-installation system that consolidates shell completion, GPU acceleration, and advanced tools into a single streamlined setup. This replaces the previous separate installation steps.

**For Security and Code Quality Tools:**

.. code-block:: bash

   pip install homodyne-analysis[quality]
   # Includes black, isort, flake8, mypy, ruff, bandit, pip-audit

**Unified Shell Experience:**

After running ``homodyne-post-install``, the system provides:

- **Advanced CLI tools**: ``homodyne-gpu-optimize``, ``homodyne-validate``
- **Unified completion**: Smart tab completion across shells (zsh, bash, fish)
- **Convenient aliases**: ``hm`` (mcmc), ``hc`` (classical), ``hr`` (robust), ``ha`` (all)
- **GPU utilities**: ``gpu-status`` for hardware monitoring
- **System validation**: ``homodyne-validate --quick`` for health checks

.. code-block:: bash

   # After post-install setup, restart shell
   source ~/.zshrc  # or ~/.bashrc for bash

   # Use convenient aliases and tools
   hm config.json           # homodyne --method mcmc config.json
   gpu-status              # Check GPU status
   homodyne-validate       # Comprehensive system validation

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

**Post-Install Issues:**

If shell features don't work after installation:

.. code-block:: bash

   # Re-run the unified post-install setup
   homodyne-post-install --shell zsh --gpu --advanced
   
   # Restart your shell
   source ~/.zshrc    # or ~/.bashrc for bash
   
   # For cleanup and fresh start:
   homodyne-cleanup                     # Interactive cleanup
   homodyne-cleanup --all              # Remove all installed features
   
   # Validate system after setup
   homodyne-validate --quick

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

JAX Backend GPU Acceleration with PyTensor Environment Variable Auto-Configuration
----------------------------------------------------------------------------------

The package supports JAX backend GPU acceleration with automatic PyTensor environment variable configuration for optimal MCMC performance on Linux systems with NVIDIA GPUs.

**System Requirements (Linux Only)**

- Linux operating system (GPU acceleration not available on Windows/macOS)
- System CUDA 12.6+ installed at ``/usr/local/cuda``
- cuDNN 9.12+ installed in system libraries
- NVIDIA GPU with driver 560.28+
- Virtual environment (conda/mamba/venv/virtualenv) for automatic environment variable configuration

**Automatic JAX Backend GPU + PyTensor Environment Variable Setup**

When you install with ``[jax]``, ``[mcmc]``, or ``[performance]`` options on a Linux system:

1. **JAX backend**: Automatically installs with system CUDA 12.6+ support for GPU operations
2. **PyTensor environment variables**: Automatically configured for CPU mode to avoid C compilation issues
3. **Environment integration**: Automatic activation/deactivation scripts for conda/mamba environments

.. code-block:: bash

   # Any of these will include JAX backend GPU support + PyTensor auto-configuration on Linux:
   pip install homodyne-analysis[jax]         # JAX GPU backend + PyTensor CPU mode
   pip install homodyne-analysis[mcmc]        # Includes NumPyro for JAX backend GPU MCMC
   pip install homodyne-analysis[performance] # Full performance optimization

   # PyTensor environment variables automatically configured:
   # PYTENSOR_FLAGS="device=cpu,floatX=float64,mode=FAST_COMPILE,optimizer=fast_compile,cxx="

**Verify JAX Backend GPU + PyTensor Configuration**

After installation, verify the setup:

.. code-block:: bash

   # Check JAX backend GPU status and PyTensor configuration (conda/mamba environments)
   homodyne_gpu_status

   # Manual verification - check PyTensor environment variables
   echo $PYTENSOR_FLAGS
   # Should show: device=cpu,floatX=float64,mode=FAST_COMPILE,optimizer=fast_compile,cxx=

   # Verify JAX GPU detection
   python -c "import jax; print(f'JAX devices: {jax.devices()}')"
   # Should show: [CudaDevice(id=0)]

.. code-block:: python

   # In Python - verify JAX backend GPU + PyTensor CPU configuration:
   import os
   print(f"PyTensor flags: {os.environ.get('PYTENSOR_FLAGS')}")
   # Should show: device=cpu,floatX=float64,mode=FAST_COMPILE,optimizer=fast_compile,cxx=
   
   import jax
   print(f"JAX devices: {jax.devices()}")
   # Output should show: [CudaDevice(id=0), ...] for GPU
   
   # Verify PyTensor CPU configuration
   try:
       import pytensor
       from pytensor import config
       print(f"PyTensor device: {config.device}")  # Should show: cpu
       print(f"PyTensor C++ compiler: '{config.cxx}'")  # Should show: ''
   except ImportError:
       print("PyTensor not installed")

**JAX Backend GPU MCMC with PyTensor CPU Mode**

The MCMC module automatically uses JAX backend for GPU operations while PyTensor runs on CPU:

.. code-block:: python

   from homodyne.optimization.mcmc import HodomyneMCMC

   # JAX backend GPU + PyTensor CPU configuration is automatic
   # - JAX handles GPU operations (MCMC sampling, numerical computations)
   # - PyTensor uses CPU mode (avoids C compilation issues)
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
5. Run ``homodyne-post-install --gpu`` for automated GPU setup and troubleshooting

Getting Help
------------

If you encounter installation issues:

1. Check the `troubleshooting guide <../developer-guide/troubleshooting.html>`_
2. Search existing `GitHub issues <https://github.com/imewei/homodyne/issues>`_
3. Create a new issue with your system details and error messages
