Usage Guide
===========

This comprehensive guide covers all aspects of using the homodyne package for X-ray Photon Correlation Spectroscopy (XPCS) analysis.

Command Line Interface
----------------------

The main entry point is ``run_homodyne.py``, which provides a comprehensive command-line interface:

Basic Analysis Commands
~~~~~~~~~~~~~~~~~~~~~~~

**Static Isotropic Analysis** (fastest, 3 parameters):

.. code-block:: bash

   python run_homodyne.py --static-isotropic --method classical

**Static Anisotropic Analysis** (3 parameters with angle filtering):

.. code-block:: bash

   python run_homodyne.py --static-anisotropic --method classical

**Laminar Flow Analysis** (full 7-parameter model):

.. code-block:: bash

   python run_homodyne.py --laminar-flow --method classical

Custom Configuration
~~~~~~~~~~~~~~~~~~~~~

Use custom configuration files:

.. code-block:: bash

   python run_homodyne.py --config my_experiment.json --output-dir ./results

Data Validation
~~~~~~~~~~~~~~~

Generate experimental data validation plots:

.. code-block:: bash

   # Basic data validation
   python run_homodyne.py --plot-experimental-data

   # Verbose validation with debug logging
   python run_homodyne.py --plot-experimental-data --verbose

   # Combined validation with analysis
   python run_homodyne.py --plot-experimental-data --method all --verbose

Optimization Methods
~~~~~~~~~~~~~~~~~~~~~

**Classical Optimization** (fast point estimates):

.. code-block:: bash

   python run_homodyne.py --static-isotropic --method classical

**Bayesian MCMC Sampling** (uncertainty quantification):

.. code-block:: bash

   python run_homodyne.py --static-isotropic --method mcmc

**Combined Analysis** (classical followed by MCMC):

.. code-block:: bash

   python run_homodyne.py --static-isotropic --method all

Python API
-----------

Basic Analysis Workflow
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from homodyne import HomodyneAnalysisCore, ConfigManager
   
   # Load configuration
   config = ConfigManager("config_static_isotropic_sample.json")
   
   # Initialize analysis engine
   analysis = HomodyneAnalysisCore(config)
   
   # Run classical optimization
   classical_results = analysis.optimize_classical()
   
   # Run MCMC analysis (optional)
   mcmc_results = analysis.optimize_mcmc()

Advanced Configuration Loading
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from homodyne.core import ConfigManager
   
   # Load with parameter overrides
   config = ConfigManager("base_config.json")
   config.override_parameters({
       "D0": 1.5e-12,
       "alpha": 0.8,
       "max_iterations": 2000
   })

Accessing Results
~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Classical optimization results
   optimized_params = classical_results.x
   chi_squared = classical_results.fun
   
   # MCMC results (when available)
   posterior_samples = mcmc_results.posterior
   parameter_means = mcmc_results.posterior.mean()
   parameter_stds = mcmc_results.posterior.std()

Configuration Management
------------------------

Creating Configurations
~~~~~~~~~~~~~~~~~~~~~~~

Use the enhanced configuration generator:

.. code-block:: bash

   # Create mode-specific configurations
   python create_config.py --mode static_isotropic --sample protein_01
   python create_config.py --mode static_anisotropic --sample collagen --author "Your Name"
   python create_config.py --mode laminar_flow --sample microgel

   # Custom output location
   python create_config.py --mode static_isotropic --output my_custom_config.json

Configuration Templates
~~~~~~~~~~~~~~~~~~~~~~~

The package provides several pre-configured templates:

- ``config_static_isotropic.json``: Optimized for isotropic analysis
- ``config_static_anisotropic.json``: Static analysis with angle filtering
- ``config_laminar_flow.json``: Full flow analysis with all 7 parameters
- ``config_template.json``: Master template with comprehensive documentation

Mode Selection in Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Specify analysis mode through configuration settings:

.. code-block:: json

   {
     "analysis_settings": {
       "static_mode": true,
       "static_submode": "isotropic"
     }
   }

**Mode Selection Rules**:

- ``static_mode: false`` → Laminar Flow Mode (7 parameters)
- ``static_mode: true, static_submode: "isotropic"`` → Static Isotropic Mode (3 parameters)
- ``static_mode: true, static_submode: "anisotropic"`` → Static Anisotropic Mode (3 parameters)
- ``static_mode: true, static_submode: null`` → Static Anisotropic Mode (default)

Active Parameters
~~~~~~~~~~~~~~~~~

Specify which parameters to optimize:

.. code-block:: json

   {
     "initial_parameters": {
       "active_parameters": ["D0", "alpha", "D_offset"]
     }
   }

**Default Active Parameters**:

- **Static Modes**: ``["D0", "alpha", "D_offset"]``
- **Laminar Flow**: ``["D0", "alpha", "D_offset", "gamma_dot_t0", "beta", "gamma_dot_t_offset", "phi0"]``

Data Input and Output
---------------------

Supported Data Formats
~~~~~~~~~~~~~~~~~~~~~~~

The package supports multiple data formats:

- **HDF5**: Recommended for large datasets
- **NPZ**: NumPy archive format
- **MAT**: MATLAB format
- **JSON**: For metadata and configuration

Data Loading Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: json

   {
     "file_paths": {
       "c2_data_file": "path/to/correlation_data.h5",
       "phi_angles_file": "path/to/angles.npz",
       "cache_directory": "./cache"
     }
   }

Output Management
~~~~~~~~~~~~~~~~~

.. code-block:: json

   {
     "output": {
       "results_directory": "./results",
       "plot_directory": "./plots",
       "save_intermediate_results": true,
       "output_format": "json"
     }
   }

Performance Optimization
-------------------------

Environment Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~

Set optimal environment variables:

.. code-block:: bash

   # Threading optimization
   export OMP_NUM_THREADS=8
   export OPENBLAS_NUM_THREADS=8
   export MKL_NUM_THREADS=8

   # Numba compatibility
   export NUMBA_DISABLE_INTEL_SVML=1

Configuration Tuning
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: json

   {
     "performance": {
       "num_threads": 8,
       "memory_limit_gb": 16,
       "use_numba_jit": true,
       "enable_angle_filtering": true
     }
   }

Angle Filtering
~~~~~~~~~~~~~~~

Enable angle filtering to focus on specific angular ranges:

.. code-block:: json

   {
     "analysis_settings": {
       "enable_angle_filtering": true,
       "angle_filter_ranges": [[-10, 10], [170, 190]]
     }
   }

**Benefits**:
- 3-5x speedup for large datasets
- Focuses optimization on relevant angular ranges
- Minimal accuracy loss

Workflow Integration
--------------------

Experimental Data Validation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Enable automatic data validation:

.. code-block:: json

   {
     "workflow_integration": {
       "analysis_workflow": {
         "plot_experimental_data_on_load": true
       }
     }
   }

Or use command line:

.. code-block:: bash

   python run_homodyne.py --plot-experimental-data --config my_config.json

Quality Control Workflow
~~~~~~~~~~~~~~~~~~~~~~~~

1. **Data Validation**: Always validate experimental data first
2. **Mode Selection**: Choose appropriate analysis mode
3. **Classical Analysis**: Get initial parameter estimates
4. **MCMC Analysis**: Obtain uncertainty quantification
5. **Results Validation**: Check convergence and quality metrics

.. code-block:: bash

   # Complete workflow example
   python run_homodyne.py --plot-experimental-data --static-anisotropic --method all --verbose

Testing and Benchmarking
-------------------------

Test Suite Execution
~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Standard test run
   python homodyne/run_tests.py

   # Quick tests (exclude slow integration tests)
   python homodyne/run_tests.py --fast

   # Parallel execution
   python homodyne/run_tests.py --parallel 4

   # Coverage reporting
   python homodyne/run_tests.py --coverage

Performance Benchmarking
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Comprehensive performance analysis
   python benchmark_performance.py --iterations 50 --size 1000

   # Quick performance check
   python benchmark_performance.py --fast

Best Practices
--------------

Data Preparation
~~~~~~~~~~~~~~~~

1. **Validate Data Quality**: Use ``--plot-experimental-data`` to check data quality
2. **Check Statistics**: Ensure mean values around 1.0 for g₂ functions
3. **Verify Contrast**: Confirm sufficient dynamic contrast (>0.001)

Analysis Strategy
~~~~~~~~~~~~~~~~

1. **Start Simple**: Begin with isotropic mode for initial exploration
2. **Progressive Complexity**: Move to anisotropic or flow modes as needed
3. **Validate Results**: Always check fit quality and parameter reasonableness
4. **Uncertainty Quantification**: Use MCMC for publication-quality results

Performance Strategy
~~~~~~~~~~~~~~~~~~~~

1. **Enable Numba**: Install numba for 3-5x performance boost
2. **Use Angle Filtering**: Enable for large datasets
3. **Memory Management**: Adjust limits based on available RAM
4. **Parallel Processing**: Set appropriate thread counts

Error Handling
--------------

Common Issues and Solutions
~~~~~~~~~~~~~~~~~~~~~~~~~~

**Configuration Errors**:

.. code-block:: bash

   # Validate JSON syntax
   python -m json.tool config.json

**Memory Issues**:

.. code-block:: json

   {
     "performance": {
       "memory_limit_gb": 8,
       "data_type": "float32"
     }
   }

**Convergence Problems**:

.. code-block:: json

   {
     "optimization": {
       "max_iterations": 5000,
       "tolerance": 1e-8
     }
   }

This comprehensive usage guide should help you effectively use all features of the homodyne package. For specific issues, consult the :doc:`troubleshooting` section.
