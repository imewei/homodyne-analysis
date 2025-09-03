Quick Reference
===============

.. index:: commands, configuration, troubleshooting, parameters, backends, API

Essential commands, code snippets, and configuration examples for homodyne analysis.

.. contents:: Quick Navigation
   :local:
   :depth: 2

Installation Quick Start
------------------------

.. code-block:: bash

   # Choose your backend
   pip install homodyne-analysis[mcmc]      # CPU backend (PyMC) - Recommended
   pip install homodyne-analysis[mcmc-gpu]  # GPU backend (NumPyro/JAX) - Linux
   pip install homodyne-analysis[all]       # Both backends + all features
   
   # Post-installation setup
   homodyne-post-install --shell zsh --gpu --advanced
   source ~/.zshrc  # or ~/.bashrc
   homodyne-validate --quick

Commands Reference
------------------

Essential Commands
~~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 30 20 50
   :header-rows: 1

   * - Command
     - Alias
     - Description
   * - ``homodyne-config --mode <mode> --sample <name>``
     - ``hconfig``
     - Generate configuration template
   * - ``homodyne --method classical``
     - ``hc``
     - Classical optimization (fastest)
   * - ``homodyne --method mcmc``
     - ``hm``
     - CPU MCMC sampling (PyMC)
   * - ``homodyne-gpu --method mcmc``
     - ``hgm``
     - GPU MCMC sampling (NumPyro/JAX)
   * - ``homodyne --method all``
     - ``ha``
     - All methods
   * - ``homodyne-validate``
     - ``hval``
     - System health check

Backend Selection
~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # CPU Backend (Pure PyMC - cross-platform)
   homodyne --method mcmc                          # Default
   HOMODYNE_GPU_INTENT=false homodyne --method mcmc  # Explicit

   # GPU Backend (Pure NumPyro/JAX - Linux with fallback)
   homodyne-gpu --method mcmc                      # Recommended  
   HOMODYNE_GPU_INTENT=true homodyne --method mcmc   # Explicit

Analysis Modes
--------------

Quick Mode Selection
~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 20 15 15 20 30
   :header-rows: 1

   * - Mode
     - Params
     - Speed
     - Command
     - When to Use
   * - **Static Isotropic**
     - 3
     - ⚡⚡⚡
     - ``--mode static_isotropic``
     - Isotropic systems, fastest analysis
   * - **Static Anisotropic**
     - 3
     - ⚡⚡
     - ``--mode static_anisotropic``
     - Angular dependencies, no flow
   * - **Laminar Flow**
     - 7
     - ⚡
     - ``--mode laminar_flow``
     - Flow systems, complete analysis

Configuration Templates
-----------------------

Minimal Configuration
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: json

   {
     "analysis_settings": {
       "static_mode": true,
       "static_submode": "isotropic"
     },
     "file_paths": {
       "c2_data_file": "data.h5",
       "phi_angles_file": "angles.txt"
     },
     "initial_parameters": {
       "values": [1000, -0.5, 100]
     }
   }

Performance Optimized
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: json

   {
     "analysis_settings": {
       "static_mode": true,
       "static_submode": "isotropic",
       "enable_angle_filtering": true,
       "angle_filter_ranges": [[-5, 5], [175, 185]]
     },
     "performance_settings": {
       "num_threads": 4,
       "data_type": "float32"
     },
     "initial_parameters": {
       "values": [1000, -0.5, 100]
     }
   }

MCMC Configuration
~~~~~~~~~~~~~~~~~~

.. code-block:: json

   {
     "optimization_config": {
       "mcmc_sampling": {
         "enabled": true,
         "draws": 2000,
         "tune": 1000,
         "chains": 4,
         "target_accept": 0.95
       }
     }
   }

Python API Quick Start
-----------------------

Essential Imports
~~~~~~~~~~~~~~~~~

.. code-block:: python

   from homodyne import HomodyneAnalysisCore, ConfigManager
   from homodyne.run_homodyne import get_mcmc_backend
   import os

Basic Workflow
~~~~~~~~~~~~~~

.. code-block:: python

   # Load configuration
   config = ConfigManager("config.json")
   
   # Initialize analysis
   analysis = HomodyneAnalysisCore(config)
   analysis.load_experimental_data()
   
   # Run analysis
   results = analysis.optimize_classical()  # Classical
   results = analysis.optimize_all()        # All methods

Isolated Backend Usage
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Automatic backend selection
   mcmc_function, backend_name, has_gpu = get_mcmc_backend()
   print(f"Using: {backend_name}")
   
   # Force specific backend
   os.environ["HOMODYNE_GPU_INTENT"] = "false"  # CPU
   # os.environ["HOMODYNE_GPU_INTENT"] = "true"   # GPU
   
   # Run with isolated backend
   results = mcmc_function(
       analysis_core=analyzer,
       config=config.config,
       c2_experimental=data,
       phi_angles=angles,
       filter_angles_for_optimization=True
   )

Configuration Management
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Load and validate
   config = ConfigManager("my_config.json")
   if config.validate():
       print("✓ Configuration valid")
   
   # Check features
   if config.is_mcmc_enabled():
       print("✓ MCMC enabled")
   
   # Access settings
   settings = config.get_analysis_settings()
   print(f"Mode: {settings['static_mode']}")

Error Handling
~~~~~~~~~~~~~~

.. code-block:: python

   try:
       results = analysis.optimize_classical()
   except HomodyneError as e:
       print(f"Analysis error: {e}")
   except Exception as e:
       print(f"Unexpected error: {e}")

Parameters Reference
--------------------

Static Isotropic (3 Parameters)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 15 20 15 50
   :header-rows: 1

   * - Parameter
     - Symbol
     - Unit
     - Description
   * - ``D0``
     - D₀
     - [Å²/s]
     - Reference diffusion coefficient
   * - ``alpha``
     - α
     - [-]
     - Time dependence exponent
   * - ``D_offset``
     - D_offset
     - [Å²/s]
     - Baseline diffusion component

Laminar Flow (7 Parameters)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 15 20 15 50
   :header-rows: 1

   * - Parameter
     - Symbol
     - Unit
     - Description
   * - ``D0``
     - D₀
     - [Å²/s]
     - Reference diffusion coefficient
   * - ``alpha``
     - α
     - [-]
     - Time dependence exponent
   * - ``D_offset``
     - D_offset
     - [Å²/s]
     - Baseline diffusion component
   * - ``gamma_dot_t0``
     - γ̇₀
     - [s⁻¹]
     - Reference shear rate
   * - ``beta``
     - β
     - [-]
     - Shear exponent
   * - ``gamma_dot_t_offset``
     - γ̇_offset
     - [s⁻¹]
     - Baseline shear component
   * - ``phi0``
     - φ₀
     - [deg]
     - Angular offset parameter

MCMC Priors
~~~~~~~~~~~~

.. list-table::
   :widths: 20 30 15 35
   :header-rows: 1

   * - Parameter
     - Prior Distribution
     - Unit
     - Typical Values
   * - ``D0``
     - TruncatedNormal(μ=1e4, σ=1000, >1)
     - [Å²/s]
     - 1000-10000
   * - ``alpha``
     - Normal(μ=-1.5, σ=0.1)
     - [-]
     - -2.0 to 0.0
   * - ``contrast``
     - TruncatedNormal(μ=0.3, σ=0.1)
     - [-]
     - 0.05-0.5
   * - ``offset``
     - TruncatedNormal(μ=1.0, σ=0.2)
     - [-]
     - 0.05-1.95

Troubleshooting Quick Fixes
----------------------------

Common Issues
~~~~~~~~~~~~~

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Problem
     - Solution
   * - **"File not found"**
     - Check paths in config: ``homodyne-config --validate config.json``
   * - **"Optimization failed"**
     - Try different initial params or simpler mode
   * - **Slow performance**
     - Enable angle filtering: ``"enable_angle_filtering": true``
   * - **MCMC won't converge**
     - Use classical results as initial: ``homodyne --method all``
   * - **Import errors**
     - Check installation: ``homodyne-validate --quick``
   * - **GPU not detected**
     - Run: ``gpu-status`` and ``homodyne-gpu-optimize --report``

Backend Issues
~~~~~~~~~~~~~~

.. code-block:: bash

   # Test CPU backend
   python -c "from homodyne.optimization.mcmc_cpu_backend import is_cpu_mcmc_available; print(is_cpu_mcmc_available())"
   
   # Test GPU backend
   python -c "from homodyne.optimization.mcmc_gpu_backend import is_gpu_mcmc_available; print(is_gpu_mcmc_available())"
   
   # Force specific backend
   HOMODYNE_GPU_INTENT=false homodyne --method mcmc  # CPU
   HOMODYNE_GPU_INTENT=true homodyne --method mcmc   # GPU

Performance Optimization
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: json

   {
     "analysis_settings": {
       "enable_angle_filtering": true,
       "angle_filter_ranges": [[-5, 5], [175, 185]]
     },
     "performance_settings": {
       "num_threads": 4,
       "data_type": "float32"
     },
     "optimization_config": {
       "mcmc_sampling": {
         "chains": 4,
         "cores": 4
       }
     }
   }

File Formats
------------

Data File Formats
~~~~~~~~~~~~~~~~~~

**Supported formats:**

* **HDF5** (recommended): ``data.h5``, ``data.hdf5``
* **NumPy**: ``data.npz``
* **PyXPCS**: HDF5 with specific structure

**Required data structure:**

.. code-block:: text

   c2_data:     shape (n_angles, n_tau_times, n_delay_times)
   tau_values:  shape (n_tau_times,) - correlation lag times
   delay_values: shape (n_delay_times,) - delay times (optional)

Angles File Format
~~~~~~~~~~~~~~~~~~

.. code-block:: text

   # Simple text file with angles in degrees
   0.0
   15.0
   30.0
   45.0
   ...

Output Files
~~~~~~~~~~~~

.. code-block:: text

   homodyne_results/
   ├── homodyne_analysis_results.json    # Main results
   ├── run.log                           # Analysis log
   ├── classical/                       # Classical results
   │   └── nelder_mead/
   │       ├── parameters.json
   │       └── fitted_data.npz
   ├── mcmc/                            # MCMC results  
   │   ├── parameters.json
   │   ├── fitted_data.npz
   │   ├── trace_data.npz
   │   └── diagnostics.json
   └── plots/                          # Visualization plots

System Information
------------------

Installation Verification
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Quick system check
   homodyne-validate --quick
   
   # Full system validation
   homodyne-validate --verbose
   
   # Check specific components
   homodyne-validate --test gpu
   homodyne-validate --test completion

Backend Information
~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Check backend status
   python -c "from homodyne.run_homodyne import get_mcmc_backend; print(get_mcmc_backend())"
   
   # GPU hardware info
   gpu-status
   homodyne-gpu-optimize --report

Environment Variables
~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Variable
     - Description
   * - ``HOMODYNE_GPU_INTENT``
     - ``"true"`` for GPU backend, ``"false"`` for CPU backend
   * - ``OMP_NUM_THREADS``
     - OpenMP thread count (set to 4 for stability)
   * - ``JAX_ENABLE_X64``
     - ``"0"`` for float32 (GPU performance), ``"1"`` for float64

See Also
--------

.. seealso::

   * :doc:`installation` - Detailed installation guide
   * :doc:`quickstart` - Step-by-step tutorial
   * :doc:`configuration` - Complete configuration reference
   * :doc:`../api-reference/mcmc` - MCMC API documentation
   * :doc:`../developer-guide/packaging` - Backend architecture details