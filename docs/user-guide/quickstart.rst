Quick Start Guide
=================

This guide will get you analyzing homodyne scattering data in minutes.

Installation
------------

.. code-block:: bash

   pip install homodyne-analysis[all]

5-Minute Tutorial
-----------------

**Step 0: Optional Shell Enhancement (Recommended)**

.. code-block:: bash

   # Enable shell completion and shortcuts for faster workflow
   pip install homodyne-analysis[completion]
   homodyne --install-completion zsh  # or bash, fish, powershell
   source ~/.zshrc                    # Restart shell or reload config

   # To remove completion later:
   # homodyne --uninstall-completion zsh

   # Test shortcuts (work immediately)
   homodyne_help                      # Show all available options

**Step 1: Create a Configuration**

.. code-block:: bash

   # Create a configuration for isotropic analysis (fastest)
   homodyne-config --mode static_isotropic --sample my_sample

   # Or using shortcuts after shell enhancement:
   # Tab completion: homodyne-config --mode <TAB>  (shows modes)
   # Fast reference: homodyne_help                (shows all options)

**Step 2: Prepare Your Data**

Ensure your experimental data is in the correct format:

- **C2 data file**: Correlation function data (HDF5 or NPZ format)
- **Angle file**: Scattering angles (text file with angles in degrees)

**Step 3: Run Analysis**

.. code-block:: bash

   # Data validation first (optional, saves plots to ./homodyne_results/exp_data/)
   homodyne --config my_sample_config.json --plot-experimental-data
   # Or with shortcuts: hplot (if config file is homodyne_config.json)

   # Basic analysis (fastest, saves results to ./homodyne_results/)
   homodyne --config my_sample_config.json --method classical
   # Or with shortcuts: hc --config my_sample_config.json

   # Run all methods with verbose output
   homodyne --config my_sample_config.json --method all --verbose
   # Or with shortcuts: ha --config my_sample_config.json --verbose

   # Quick analysis using different methods:
   # hc        # homodyne --method classical
   # hr        # homodyne --method robust
   # ha        # homodyne --method all

**Step 4: View Results**

Results are saved to the ``homodyne_results/`` directory with organized subdirectories:

- **Main results**: ``homodyne_analysis_results.json`` with parameter estimates and fit quality
- **Classical output**: ``./classical/`` subdirectory with method-specific directories (``nelder_mead/``, ``gurobi/``)
- **Robust output**: ``./robust/`` subdirectory with method-specific directories (``wasserstein/``, ``scenario/``, ``ellipsoidal/``)
- **Experimental plots**: ``./exp_data/`` subdirectory with validation plots (if using ``--plot-experimental-data``)

**Method-Specific Outputs**:

- **Classical** (``./classical/``): Method-specific directories with fast point estimates, consolidated ``fitted_data.npz`` files
- **Robust** (``./robust/``): Noise-resistant optimization with method-specific directories (wasserstein, scenario, ellipsoidal)
- **All methods**: Save experimental, fitted, and residuals data in consolidated ``fitted_data.npz`` files per method

Python API Example
-------------------

.. code-block:: python

   from homodyne import HomodyneAnalysisCore, ConfigManager

   # Load configuration
   config = ConfigManager("my_experiment.json")

   # Initialize analysis
   analysis = HomodyneAnalysisCore(config)

   # Load experimental data
   analysis.load_experimental_data()

   # Run classical optimization
   classical_results = analysis.optimize_classical()
   print(f"Classical chi-squared: {classical_results.fun:.3f}")


Analysis Modes Quick Reference
------------------------------

Choose the appropriate mode for your system:

**Static Isotropic (Fastest)**

- Use when: System is isotropic, no angular dependencies
- Parameters: 3 (D₀, α, D_offset)
- Speed: ⭐⭐⭐
- Command: ``--static-isotropic``

**Static Anisotropic**

- Use when: System has angular dependencies but no flow
- Parameters: 3 (D₀, α, D_offset)
- Speed: ⭐⭐
- Command: ``--static-anisotropic``

**Laminar Flow (Most Complete)**

- Use when: System under flow conditions
- Parameters: 7 (D₀, α, D_offset, γ̇₀, β, γ̇_offset, φ₀)
- Speed: ⭐
- Command: ``--laminar-flow``

Configuration Tips
------------------

**Quick Configuration:**

.. code-block:: javascript

   {
     "analysis_settings": {
       "static_mode": true,
       "static_submode": "isotropic"
     },
     "file_paths": {
       "c2_data_file": "path/to/your/data.h5",
       "phi_angles_file": "path/to/angles.txt"
     },
     "initial_parameters": {
       "values": [1000, -0.5, 100]
     }
   }

**Performance Optimization:**

.. code-block:: javascript

   {
     "analysis_settings": {
       "enable_angle_filtering": true,
       "angle_filter_ranges": [[-5, 5], [175, 185]]
     },
     "performance_settings": {
       "num_threads": 4,
       "data_type": "float32"
     }
   }

Logging Control Options
-----------------------

The homodyne package provides flexible logging control for different use cases:

.. list-table:: Logging Options
   :widths: 20 25 25 30
   :header-rows: 1

   * - Option
     - Console Output
     - File Output
     - Use Case
   * - **Default**
     - INFO level
     - INFO level
     - Normal interactive analysis
   * - ``--verbose``
     - DEBUG level
     - DEBUG level
     - Detailed troubleshooting
   * - ``--quiet``
     - None
     - INFO level
     - Batch processing, clean output

**Examples:**

.. code-block:: bash

   # Normal mode with INFO-level logging
   homodyne --config my_config.json --method classical

   # Verbose mode with detailed debugging
   homodyne --config my_config.json --method all --verbose

   # Quiet mode for batch processing (logs only to file)
   homodyne --config my_config.json --method classical --quiet

   # Error: Cannot combine conflicting options
   homodyne --verbose --quiet  # ERROR

**Important:** File logging is always enabled and saves to ``output_dir/run.log`` regardless of console settings.

Performance Features
--------------------

The homodyne package includes advanced performance optimization and stability features:

**JIT Compilation Warmup**

Automatic Numba kernel pre-compilation eliminates JIT overhead:

.. code-block:: python

   from homodyne.core.kernels import warmup_numba_kernels

   # Warmup all computational kernels
   warmup_results = warmup_numba_kernels()
   print(f"Kernels warmed up in {warmup_results['total_warmup_time']:.3f}s")

**Performance Monitoring**

Built-in performance monitoring with automatic optimization:

.. code-block:: python

   from homodyne.core.config import performance_monitor

   # Monitor function performance
   def my_analysis():
       with performance_monitor.time_function("my_analysis"):
           # Your analysis code here
           pass

   # Access performance statistics
   stats = performance_monitor.get_timing_summary()
   print(f"Performance stats: {stats}")

**Benchmarking Tools**

Stable and adaptive benchmarking for research:

.. code-block:: python

   from homodyne.core.profiler import stable_benchmark, adaptive_stable_benchmark

   # Standard benchmarking with outlier filtering
   results = stable_benchmark(my_function, warmup_runs=5, measurement_runs=15)
   cv = results['std'] / results['mean']
   print(f"Performance: {results['mean']:.4f}s ± {cv:.3f} CV")

   # Adaptive benchmarking (finds optimal measurement count)
   results = adaptive_stable_benchmark(my_function, target_cv=0.10)
   print(f"Achieved {results['cv']:.3f} CV in {results['total_runs']} runs")

**Performance Stability Achievements**

The homodyne package has been optimized for excellent performance stability:

- **97% reduction** in chi-squared calculation variability (CV < 0.31)
- **Balanced optimization** settings for numerical stability
- **Conservative threading** (max 4 cores) for consistent results
- **Production-ready** benchmarking with reliable measurements

**Configuration Options**

Enable advanced performance features in your config:

.. code-block:: json

   {
     "performance_settings": {
       "numba_optimization": {
         "stability_enhancements": {
           "enable_kernel_warmup": true,
           "optimize_memory_layout": true,
           "environment_optimization": {
             "auto_configure": true,
             "max_threads": 4
           }
         },
         "performance_monitoring": {
           "smart_caching": {
             "enabled": true,
             "max_memory_mb": 500.0
           }
         }
       }
     }
   }

Next Steps
----------

- Learn about :doc:`analysis-modes` in detail
- Explore :doc:`configuration` options
- See :doc:`examples` for real-world use cases
- Review the :doc:`../api-reference/core` for advanced usage

Common First-Time Issues
-------------------------

**"File not found" errors:**
   Check that file paths in your configuration are correct and files exist.

**"Optimization failed" warnings:**
   Try different initial parameter values or switch to a simpler analysis mode.

**Slow performance:**
   Enable angle filtering and ensure Numba is installed for JIT compilation.


------------------------


.. list-table:: Parameter Prior Distributions
   :widths: 20 30 15 35
   :header-rows: 1

   * - Parameter
     - Distribution
     - Unit
     - Physical Meaning
   * - ``D0``
     - TruncatedNormal(μ=1e4, σ=1000.0, lower=1.0)
     - [Å²/s]
     - Reference diffusion coefficient
   * - ``alpha``
     - Normal(μ=-1.5, σ=0.1)
     - [dimensionless]
     - Time dependence exponent
   * - ``D_offset``
     - Normal(μ=0.0, σ=10.0)
     - [Å²/s]
     - Baseline diffusion component
   * - ``gamma_dot_t0``
     - TruncatedNormal(μ=1e-3, σ=1e-2, lower=1e-6)
     - [s⁻¹]
     - Reference shear rate
   * - ``beta``
     - Normal(μ=0.0, σ=0.1)
     - [dimensionless]
     - Shear exponent
   * - ``gamma_dot_t_offset``
     - Normal(μ=0.0, σ=1e-3)
     - [s⁻¹]
     - Baseline shear component
   * - ``phi0``
     - Normal(μ=0.0, σ=5.0)
     - [degrees]
     - Angular offset parameter

Scaling Parameters for Physical Constraints
--------------------------------------------


.. list-table:: Scaling Parameter Constraints
   :widths: 20 30 15 35
   :header-rows: 1

   * - Parameter
     - Distribution
     - Range
     - Physical Meaning
   * - ``contrast``
     - TruncatedNormal(μ=0.3, σ=0.1)
     - (0.05, 0.5]
     - Scaling factor for correlation strength
   * - ``offset``
     - TruncatedNormal(μ=1.0, σ=0.2)
     - (0.05, 1.95)
     - Baseline correlation level
   * - ``c2_fitted``
     - -
     - [1.0, 2.0]
     - Final correlation function range
   * - ``c2_theory``
     - -
     - [0.0, 1.0]
     - Theoretical correlation function range

The relationship is: **c2_fitted = c2_theory × contrast + offset**

**Configuration Format:**

.. code-block:: json

   {
     "parameter_space": {
       "bounds": [
         {"name": "D0", "min": 1.0, "max": 1000000, "type": "Normal"},
         {"name": "alpha", "min": -2.0, "max": 2.0, "type": "Normal"},
         {"name": "D_offset", "min": -100, "max": 100, "type": "Normal"}
       ]
     }
   }
