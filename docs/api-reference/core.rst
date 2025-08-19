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

**Basic Analysis**:

.. code-block:: python

   from homodyne import HomodyneAnalysisCore, ConfigManager
   
   # Initialize configuration
   config = ConfigManager("my_experiment.json")
   
   # Create analysis instance
   analysis = HomodyneAnalysisCore(config)
   
   # Load data and run analysis
   analysis.load_experimental_data()
   results = analysis.run_analysis()
   
   print(f"Analysis completed: {len(results)} angle analyses")
   print(f"Best chi-squared: {min(r['chi_squared'] for r in results):.4f}")

**Advanced Configuration**:

.. code-block:: python

   from homodyne import ConfigManager
   
   config = ConfigManager("advanced_config.json")
   
   # Check analysis mode
   mode = config.get_analysis_mode()
   print(f"Analysis mode: {mode}")
   
   # Get active parameters
   params = config.get_active_parameters()
   print(f"Active parameters: {params}")
   
   # Check if angle filtering is enabled
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
