Quick Start Guide
=================

This guide will get you analyzing homodyne scattering data in minutes.

Installation
------------

.. code-block:: bash

   # Full installation with all features (MCMC, performance, data handling)
   pip install homodyne-analysis[all]

5-Minute Tutorial
-----------------

**Step 1: Create a Configuration**

.. code-block:: bash

   # Create a configuration for isotropic analysis (fastest)
   homodyne-config --mode static_isotropic --sample my_sample

**Step 2: Prepare Your Data**

Ensure your experimental data is in the correct format:

- **C2 data file**: Correlation function data (HDF5 or NPZ format)
- **Angle file**: Scattering angles (text file with angles in degrees)

**Step 3: Run Analysis**

.. code-block:: bash

   # Data validation first (optional, saves plots to ./homodyne_results/exp_data/)
   homodyne --config my_sample_config.json --plot-experimental-data
   
   # Basic analysis (fastest, saves results to ./homodyne_results/)
   homodyne --config my_sample_config.json --method classical
   
   # Run in quiet mode (file logging only, no console output)
   homodyne --config my_sample_config.json --method classical --quiet

**Step 4: View Results**

Results are saved to the ``homodyne_results/`` directory with organized subdirectories:

- **Main results**: ``homodyne_analysis_results.json`` with parameter estimates and fit quality
- **Classical output**: ``./classical/`` subdirectory with ``.npz`` data files and C2 heatmaps
- **MCMC output**: ``./mcmc/`` subdirectory with posterior distributions, trace data, diagnostics, and 3D visualizations
- **Experimental plots**: ``./exp_data/`` subdirectory with validation plots (if using ``--plot-experimental-data``)

**Method-Specific Outputs**:

- **Classical** (``./classical/``): Fast point estimates, fitted data files, residuals analysis
- **MCMC** (``./mcmc/``): Full posterior distributions, convergence diagnostics, trace plots, corner plots, 3D surface plots  
- **Both methods**: Save experimental, fitted, and residuals data as compressed ``.npz`` files for further analysis

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
   
   # Optional: Run MCMC for uncertainty quantification
   if config.is_mcmc_enabled():
       mcmc_results = analysis.run_mcmc_sampling()
       print(f"MCMC converged: {mcmc_results['converged']}")

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

**MCMC convergence issues:**
   Start with classical optimization, then use those results to initialize MCMC.

MCMC Prior Distributions
------------------------

The homodyne package uses **Normal distributions** for all parameters in MCMC analysis:

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

The MCMC implementation includes additional scaling parameters to ensure physical validity:

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