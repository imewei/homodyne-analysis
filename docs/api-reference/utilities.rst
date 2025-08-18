Utilities
=========

Utility functions for data handling, validation, and common operations.

Data Handling
-------------

.. autofunction:: homodyne.core.io_utils.save_json

   Save data in JSON format with NumPy array support.

.. autofunction:: homodyne.core.io_utils.save_numpy

   Save analysis results as NumPy compressed files.

.. autofunction:: homodyne.core.io_utils.save_analysis_results

   Save complete analysis results in multiple formats.

Angle Processing
----------------

.. autofunction:: homodyne.core.io_utils.ensure_dir

   Create directories with proper permissions.

.. autofunction:: homodyne.core.io_utils.timestamped_filename

   Generate timestamped filenames for results.

Configuration Utilities
------------------------

.. autoclass:: homodyne.core.config.ConfigManager

   Configuration management with validation and template support.

.. autofunction:: homodyne.core.config.configure_logging

   Configure logging for analysis sessions.

Performance Utilities
----------------------

.. autofunction:: homodyne.core.kernels.memory_efficient_cache

   Memory-efficient caching decorator for expensive computations.

.. autofunction:: homodyne.core.kernels.exp_negative_vectorized

   Optimized vectorized exponential operations.

Plotting Utilities
------------------

.. autofunction:: homodyne.plotting.plot_c2_heatmaps

   Plot experimental correlation data as heatmaps.

.. autofunction:: homodyne.plotting.plot_mcmc_corner

   Create corner plots for MCMC parameter distributions.

.. autofunction:: homodyne.plotting.plot_mcmc_trace

   Create trace plots for MCMC convergence diagnostics.

.. autofunction:: homodyne.plotting.plot_3d_surface

   Create 3D surface plots of correlation data.

Usage Examples
--------------

**Data Loading and Validation**:

.. code-block:: python

   from homodyne.utils import load_data_file, validate_data_format
   
   # Load experimental data
   data = load_data_file("correlation_data.h5")
   
   # Validate format
   is_valid, issues = validate_data_format(data)
   if not is_valid:
       for issue in issues:
           print(f"⚠️ {issue}")

**Angle Filtering**:

.. code-block:: python

   from homodyne.utils import load_angles, apply_angle_filtering
   
   # Load angles
   phi_angles = load_angles("scattering_angles.txt")
   
   # Apply filtering
   filtered_data, filtered_angles = apply_angle_filtering(
       correlation_data, 
       phi_angles,
       ranges=[[-5, 5], [175, 185]]
   )
   
   print(f"Filtered from {len(phi_angles)} to {len(filtered_angles)} angles")

**Configuration Management**:

.. code-block:: python

   from homodyne.utils import validate_config, expand_env_vars
   
   # Load and validate configuration
   with open("config.json") as f:
       config_dict = json.load(f)
   
   # Expand environment variables
   config_dict = expand_env_vars(config_dict)
   
   # Validate
   is_valid, errors = validate_config(config_dict)
   if not is_valid:
       for error in errors:
           print(f"❌ {error}")

**Performance Optimization**:

.. code-block:: python

   from homodyne.utils import estimate_memory_usage, optimize_data_types
   
   # Estimate memory requirements
   memory_gb = estimate_memory_usage(
       data_shape=(1000, 500),
       num_angles=360,
       analysis_mode="laminar_flow"
   )
   print(f"Estimated memory usage: {memory_gb:.1f} GB")
   
   # Optimize data types
   optimized_data = optimize_data_types(
       correlation_data, 
       target_precision="float32"
   )

**Results Visualization**:

.. code-block:: python

   from homodyne.utils import plot_fit_results, plot_mcmc_diagnostics
   
   # Plot optimization results
   fig1 = plot_fit_results(
       experimental_data,
       fitted_data,
       parameters=result.x,
       chi_squared=result.fun
   )
   fig1.savefig("fit_results.png", dpi=300)
   
   # Plot MCMC diagnostics (if available)
   if mcmc_trace is not None:
       fig2 = plot_mcmc_diagnostics(mcmc_trace)
       fig2.savefig("mcmc_diagnostics.png", dpi=300)

File I/O Functions
------------------

.. autofunction:: homodyne.core.io_utils.get_output_directory

   Get organized output directory structure.

.. autofunction:: homodyne.core.io_utils.save_fig

   Save matplotlib figures with proper formatting.

**Error Handling Example**:

.. code-block:: python

   from homodyne.utils import ConfigurationError, DataFormatError
   
   try:
       config = ConfigManager("config.json")
       analysis = HomodyneAnalysisCore(config)
       result = analysis.optimize_classical()
       
   except ConfigurationError as e:
       print(f"Configuration issue: {e}")
   except DataFormatError as e:
       print(f"Data format problem: {e}")
   except Exception as e:
       print(f"Unexpected error: {e}")

