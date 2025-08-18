Utilities
=========

Utility functions for data handling, validation, and common operations.

Data Handling
-------------

.. autofunction:: homodyne.utils.load_data_file

   Load correlation data from HDF5 or NPZ files.

.. autofunction:: homodyne.utils.save_results

   Save analysis results in multiple formats.

.. autofunction:: homodyne.utils.validate_data_format

   Validate experimental data structure and content.

Angle Processing
----------------

.. autofunction:: homodyne.utils.load_angles

   Load scattering angles from text files.

.. autofunction:: homodyne.utils.validate_angles

   Validate angle data and ranges.

.. autofunction:: homodyne.utils.apply_angle_filtering

   Apply angle filtering for performance optimization.

.. autofunction:: homodyne.utils.convert_angle_units

   Convert between degrees and radians.

Configuration Utilities
------------------------

.. autofunction:: homodyne.utils.validate_config

   Comprehensive configuration validation.

.. autofunction:: homodyne.utils.merge_configs

   Merge multiple configuration dictionaries.

.. autofunction:: homodyne.utils.expand_env_vars

   Expand environment variables in file paths.

Performance Utilities
----------------------

.. autofunction:: homodyne.utils.estimate_memory_usage

   Estimate memory requirements for analysis.

.. autofunction:: homodyne.utils.optimize_data_types

   Convert data to optimal types for performance.

.. autofunction:: homodyne.utils.setup_parallel_processing

   Configure parallel processing settings.

Plotting Utilities
------------------

.. autofunction:: homodyne.utils.plot_correlation_data

   Plot experimental correlation data.

.. autofunction:: homodyne.utils.plot_fit_results

   Visualize optimization results and fits.

.. autofunction:: homodyne.utils.plot_mcmc_diagnostics

   Create MCMC convergence diagnostic plots.

.. autofunction:: homodyne.utils.plot_parameter_correlations

   Plot parameter correlation matrices.

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

.. autofunction:: homodyne.utils.create_output_directory

   Create organized output directory structure.

.. autofunction:: homodyne.utils.save_analysis_report

   Generate comprehensive analysis report.

.. autofunction:: homodyne.utils.export_parameters

   Export parameters in various formats (JSON, CSV, etc.).

Error Handling
--------------

.. autofunction:: homodyne.utils.HomodyneError

   Base exception class for homodyne-specific errors.

.. autofunction:: homodyne.utils.ConfigurationError

   Raised for configuration-related issues.

.. autofunction:: homodyne.utils.DataFormatError

   Raised for data format problems.

.. autofunction:: homodyne.utils.ConvergenceError

   Raised for optimization convergence failures.

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

Constants
---------

.. autodata:: homodyne.utils.DEFAULT_ANGLE_FILTER_RANGES

   Default angle filtering ranges for optimization.

.. autodata:: homodyne.utils.SUPPORTED_DATA_FORMATS

   List of supported data file formats.

.. autodata:: homodyne.utils.PHYSICAL_PARAMETER_BOUNDS

   Default bounds for physical parameters.