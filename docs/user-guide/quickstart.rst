Quick Start Guide
=================

This guide will get you analyzing homodyne scattering data in minutes.

5-Minute Tutorial
-----------------

**Step 1: Create a Configuration**

.. code-block:: bash

   # Create a configuration for isotropic analysis (fastest)
   python create_config.py --mode static_isotropic --sample my_sample

**Step 2: Prepare Your Data**

Ensure your experimental data is in the correct format:

- **C2 data file**: Correlation function data (HDF5 or NPZ format)
- **Angle file**: Scattering angles (text file with angles in degrees)

**Step 3: Run Analysis**

.. code-block:: bash

   # Data validation first (optional, saves plots to ./homodyne_results/exp_data/)
   python run_homodyne.py --config my_sample_config.json --plot-experimental-data
   
   # Basic analysis (fastest, saves results to ./homodyne_results/)
   python run_homodyne.py --config my_sample_config.json --method classical

**Step 4: View Results**

Results are saved to the ``homodyne_results/`` directory with organized subdirectories:

- **Main results**: ``homodyne_analysis_results.json`` with parameter estimates and fit quality
- **Classical output**: ``./classical/`` subdirectory with ``.npz`` data files and C2 heatmaps
- **Experimental plots**: ``./exp_data/`` subdirectory with validation plots (if using ``--plot-experimental-data``)

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