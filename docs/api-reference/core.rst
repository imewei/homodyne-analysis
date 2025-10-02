Core API
========

The core API provides the main classes and functions for homodyne analysis.

HomodyneAnalysisCore
--------------------

The main analysis class that orchestrates the entire homodyne analysis workflow.

**Key Methods:**

* ``__init__(config)`` - Initialize with configuration
* ``load_experimental_data()`` - Load experimental correlation data
* ``run_analysis()`` - Run the complete analysis workflow
* ``get_results()`` - Extract analysis results
* ``save_results(output_dir)`` - Save results to disk

**Properties:**

* ``config`` - Configuration manager instance
* ``experimental_data`` - Loaded experimental data
* ``results`` - Analysis results dictionary

ConfigManager
-------------

Manages configuration loading, validation, and access.

**Key Methods:**

* ``__init__(config_file)`` - Load configuration from file
* ``validate_config()`` - Validate configuration settings
* ``get_analysis_mode()`` - Get the analysis mode (static_isotropic, etc.)
* ``get_active_parameters()`` - Get list of active parameters
* ``is_angle_filtering_enabled()`` - Check if angle filtering is enabled

Core Kernels
------------

High-performance computational kernels for correlation analysis.

**JIT-Compiled Functions:**

* ``compute_g1_correlation_numba()`` - Compute g1 correlation function
* ``create_time_integral_matrix_numba()`` - Create time integral matrices
* ``calculate_diffusion_coefficient_numba()`` - Calculate diffusion coefficients
* ``compute_sinc_squared_numba()`` - Compute sinc² functions

I/O Utilities
-------------

Data input/output utilities for loading experimental data and saving results.

**File Operations:**

* ``save_json()`` - Save data as JSON with NumPy support
* ``save_numpy()`` - Save as NumPy compressed files
* ``ensure_dir()`` - Create directories with proper permissions
* ``timestamped_filename()`` - Generate timestamped filenames

Example Usage
-------------

**Basic Classical Optimization**:

.. code-block:: python

   import numpy as np
   import json
   from homodyne.analysis.core import HomodyneAnalysisCore
   from homodyne.optimization.classical import ClassicalOptimizer
   from homodyne.data.xpcs_loader import load_xpcs_data

   # Load configuration
   with open("my_experiment.json", 'r') as f:
       config = json.load(f)

   # Initialize analysis core
   core = HomodyneAnalysisCore(config)

   # Load experimental data
   phi_angles = np.array([0, 36, 72, 108, 144])
   c2_data = load_xpcs_data(
       data_path=config['experimental_data']['data_folder_path'],
       phi_angles=phi_angles,
       n_angles=len(phi_angles)
   )

   # Run optimization
   optimizer = ClassicalOptimizer(core, config)
   params, results = optimizer.run_classical_optimization_optimized(
       phi_angles=phi_angles,
       c2_experimental=c2_data
   )

   print(f"Optimal parameters: {params}")
   print(f"Chi-squared: {results.chi_squared:.6e}")
   print(f"Best method: {results.best_method}")

**Robust Optimization for Noisy Data**:

.. code-block:: python

   from homodyne.optimization.robust import RobustHomodyneOptimizer

   # Initialize robust optimizer
   robust = RobustHomodyneOptimizer(core, config)

   # Run Wasserstein DRO optimization
   result_dict = robust.optimize(
       phi_angles=phi_angles,
       c2_experimental=c2_data,
       method="wasserstein",  # Options: wasserstein, scenario, ellipsoidal
       epsilon=0.1  # Uncertainty radius
   )

   print(f"Optimal parameters: {result_dict['optimal_params']}")
   print(f"Chi-squared: {result_dict['chi_squared']:.6e}")

**Configuration Access**:

.. code-block:: python

   # Access configuration values
   analysis_mode = config.get('analysis_settings', {}).get('static_mode', False)
   print(f"Static mode: {analysis_mode}")

   # Check subsampling settings
   subsampling = config.get('subsampling', {})
   n_angles = subsampling.get('n_angles', 4)
   print(f"Angle subsampling: {n_angles} angles")
   if config.is_angle_filtering_enabled():
       ranges = config.get_target_angle_ranges()
       print(f"Target angle ranges: {ranges}")

**High-Performance Computing**:

.. code-block:: python

   from homodyne import (
       compute_g1_correlation_numba,
       create_time_integral_matrix_numba,
       performance_monitor
   )

   # Use performance monitoring
   with performance_monitor() as monitor:
       # Compute correlation with JIT compilation
       g1_values = compute_g1_correlation_numba(
           diffusion_coeff, shear_rate, time_points, angles
       )

   print(f"Computation time: {monitor.elapsed_time:.4f}s")
