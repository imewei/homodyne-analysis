Configuration System
====================

The homodyne package uses a comprehensive JSON-based configuration system that provides mode-specific templates, intelligent defaults, and extensive customization options.

Configuration Overview
----------------------

The configuration system provides several key features:

- **Mode-specific templates** optimized for different analysis scenarios
- **Intelligent defaults** with metadata injection
- **Parameter validation** and bounds checking
- **Runtime parameter overrides** for flexibility
- **Comprehensive documentation** within configuration files

Available Templates
-------------------

The package provides several pre-configured templates:

.. list-table:: Configuration Templates
   :widths: 30 20 50
   :header-rows: 1

   * - Template File
     - Analysis Mode
     - Description
   * - ``config_static_isotropic.json``
     - Static Isotropic
     - Optimized for isotropic analysis with single dummy angle
   * - ``config_static_anisotropic.json``
     - Static Anisotropic
     - Static analysis with angle filtering enabled
   * - ``config_laminar_flow.json``
     - Laminar Flow
     - Full flow analysis with all 7 parameters
   * - ``config_template.json``
     - Master Template
     - Comprehensive template with full documentation

Creating Configurations
-----------------------

Using the Configuration Generator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The enhanced ``create_config.py`` script provides an easy way to generate analysis-specific configurations:

.. code-block:: bash

   # Create isotropic configuration (fastest)
   python create_config.py --mode static_isotropic --sample protein_01

   # Create anisotropic configuration with metadata
   python create_config.py --mode static_anisotropic --sample collagen \\
                           --author "Your Name" --experiment "Static analysis"

   # Create flow analysis configuration
   python create_config.py --mode laminar_flow --sample microgel \\
                           --experiment "Microgel dynamics under shear"

   # Custom output location
   python create_config.py --mode static_isotropic --output my_config.json

Command Line Options
~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   python create_config.py --help

Available options:
- ``--mode``: Analysis mode (static_isotropic, static_anisotropic, laminar_flow)
- ``--sample``: Sample identifier for the configuration
- ``--author``: Author name for metadata
- ``--experiment``: Experiment description
- ``--output``: Custom output filename

Manual Configuration Creation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can also copy and modify existing templates:

.. code-block:: bash

   # Copy a template
   cp homodyne/config_static_isotropic.json my_experiment.json
   
   # Edit as needed
   nano my_experiment.json

Configuration Structure
-----------------------

Core Sections
~~~~~~~~~~~~~

A complete configuration file contains several key sections:

.. code-block:: json

   {
     "metadata": { /* Experiment information */ },
     "analysis_settings": { /* Mode and analysis parameters */ },
     "file_paths": { /* Input and output file locations */ },
     "initial_parameters": { /* Parameter values and bounds */ },
     "optimization": { /* Optimization settings */ },
     "chi_squared_calculation": { /* Fitting parameters */ },
     "performance": { /* Performance tuning */ },
     "output": { /* Output configuration */ },
     "workflow_integration": { /* Workflow settings */ }
   }

Metadata Section
~~~~~~~~~~~~~~~~

.. code-block:: json

   {
     "metadata": {
       "config_version": "6.0",
       "creation_date": "2024-01-15T10:30:00Z",
       "author": "Your Name",
       "experiment": "Protein dynamics study",
       "sample": "protein_sample_01",
       "notes": "Static isotropic analysis of protein solutions"
     }
   }

Analysis Settings Section
~~~~~~~~~~~~~~~~~~~~~~~~~

**Mode Selection**:

.. code-block:: json

   {
     "analysis_settings": {
       "static_mode": true,
       "static_submode": "isotropic",
       "enable_angle_filtering": false,
       "angle_filter_ranges": [[-10, 10], [170, 190]]
     }
   }

**Mode Selection Rules**:

- ``static_mode: false`` → **Laminar Flow Mode** (7 parameters)
- ``static_mode: true, static_submode: "isotropic"`` → **Static Isotropic Mode** (3 parameters)
- ``static_mode: true, static_submode: "anisotropic"`` → **Static Anisotropic Mode** (3 parameters)
- ``static_mode: true, static_submode: null`` → **Static Anisotropic Mode** (default)

File Paths Section
~~~~~~~~~~~~~~~~~~

.. code-block:: json

   {
     "file_paths": {
       "c2_data_file": "data/correlation_data.h5",
       "phi_angles_file": "data/angles.npz",
       "cache_directory": "./cache",
       "results_directory": "./results"
     }
   }

**Supported Data Formats**:
- **HDF5** (``.h5``, ``.hdf5``): Recommended for large datasets
- **NPZ** (``.npz``): NumPy archive format
- **MAT** (``.mat``): MATLAB format
- **JSON** (``.json``): For metadata and small datasets

Parameter Configuration
-----------------------

Active Parameters System
~~~~~~~~~~~~~~~~~~~~~~~~

Specify which parameters to optimize and display:

.. code-block:: json

   {
     "initial_parameters": {
       "active_parameters": ["D0", "alpha", "D_offset"]
     }
   }

**Mode-Specific Defaults**:

- **Static Modes**: ``["D0", "alpha", "D_offset"]`` (3 parameters)
- **Laminar Flow**: ``["D0", "alpha", "D_offset", "gamma_dot_t0", "beta", "gamma_dot_t_offset", "phi0"]`` (7 parameters)

Parameter Values and Bounds
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: json

   {
     "initial_parameters": {
       "D0": 1e-12,
       "alpha": 1.0,
       "D_offset": 0.0,
       "gamma_dot_t0": 1e-3,
       "beta": 1.0,
       "gamma_dot_t_offset": 0.0,
       "phi0": 0.0
     },
     "parameter_bounds": {
       "D0": [1e-15, 1e-9],
       "alpha": [0.1, 2.0],
       "D_offset": [0.0, 1e-11],
       "gamma_dot_t0": [1e-6, 1e-1],
       "beta": [0.1, 2.0],
       "gamma_dot_t_offset": [0.0, 1e-2],
       "phi0": [-3.14159, 3.14159]
     }
   }

Physical Parameter Descriptions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Diffusion Parameters**:
- **D0**: Effective diffusion coefficient (m²/s)
- **alpha**: Time scaling exponent (dimensionless)
- **D_offset**: Baseline diffusion component (m²/s)

**Flow Parameters** (Laminar Flow mode only):
- **gamma_dot_t0**: Characteristic shear rate (s⁻¹)
- **beta**: Shear rate scaling exponent (dimensionless)
- **gamma_dot_t_offset**: Baseline shear rate (s⁻¹)
- **phi0**: Angular offset for flow geometry (radians)

Optimization Configuration
--------------------------

Classical Optimization Settings
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: json

   {
     "optimization": {
       "method": "Nelder-Mead",
       "max_iterations": 10000,
       "tolerance": 1e-8,
       "initial_simplex_size": 0.1
     }
   }

MCMC Settings
~~~~~~~~~~~~~

.. code-block:: json

   {
     "mcmc": {
       "n_samples": 2000,
       "tune": 1000,
       "chains": 4,
       "target_accept": 0.8,
       "random_seed": 42
     }
   }

**MCMC Parameters**:
- **n_samples**: Number of posterior samples per chain
- **tune**: Number of tuning steps for sampler adaptation
- **chains**: Number of parallel MCMC chains
- **target_accept**: Target acceptance rate (0.8-0.95 recommended)
- **random_seed**: Random seed for reproducibility

Performance Configuration
-------------------------

Computational Settings
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: json

   {
     "performance": {
       "num_threads": 8,
       "memory_limit_gb": 16,
       "use_numba_jit": true,
       "data_type": "float64",
       "enable_caching": true
     }
   }

**Performance Parameters**:
- **num_threads**: Number of parallel threads for computation
- **memory_limit_gb**: Maximum memory usage limit
- **use_numba_jit**: Enable Numba JIT compilation for speedup
- **data_type**: Numerical precision (``float64`` or ``float32``)
- **enable_caching**: Enable result caching for repeated runs

Angle Filtering
~~~~~~~~~~~~~~~

.. code-block:: json

   {
     "analysis_settings": {
       "enable_angle_filtering": true,
       "angle_filter_ranges": [[-10, 10], [170, 190]],
       "angle_units": "degrees"
     }
   }

**Benefits of Angle Filtering**:
- 3-5x speedup for large datasets
- Focuses optimization on relevant angular ranges
- Minimal accuracy loss for most systems

Output Configuration
--------------------

Results Output
~~~~~~~~~~~~~~

.. code-block:: json

   {
     "output": {
       "results_directory": "./results",
       "save_intermediate_results": true,
       "output_format": "json",
       "include_metadata": true
     }
   }

Plotting Configuration
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: json

   {
     "plotting": {
       "plot_directory": "./plots",
       "figure_format": "png",
       "figure_dpi": 300,
       "figure_size": [10, 8],
       "color_scheme": "viridis"
     }
   }

Data Validation Settings
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: json

   {
     "workflow_integration": {
       "analysis_workflow": {
         "plot_experimental_data_on_load": true,
         "validate_data_quality": true,
         "save_validation_plots": true
       }
     }
   }

Advanced Configuration
----------------------

Runtime Parameter Override
~~~~~~~~~~~~~~~~~~~~~~~~~

You can override configuration parameters at runtime using the Python API:

.. code-block:: python

   from homodyne.core import ConfigManager
   
   # Load base configuration
   config = ConfigManager("base_config.json")
   
   # Override specific parameters
   config.override_parameters({
       "D0": 2e-12,
       "max_iterations": 5000,
       "num_threads": 16
   })

Environment Variable Integration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Configuration values can reference environment variables:

.. code-block:: json

   {
     "file_paths": {
       "c2_data_file": "${DATA_DIR}/correlation_data.h5",
       "results_directory": "${RESULTS_DIR}"
     },
     "performance": {
       "num_threads": "${OMP_NUM_THREADS}"
     }
   }

Configuration Validation
------------------------

Built-in Validation
~~~~~~~~~~~~~~~~~~~

The ``ConfigManager`` automatically validates:

- JSON syntax and structure
- Required fields and sections
- Parameter bounds and types
- File path existence
- Mode-specific requirements

Manual Validation
~~~~~~~~~~~~~~~~~

You can validate configuration files manually:

.. code-block:: bash

   # Check JSON syntax
   python -m json.tool my_config.json

   # Validate with homodyne
   python -c "from homodyne.core import ConfigManager; ConfigManager('my_config.json')"

Example Configurations
----------------------

Static Isotropic Example
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: json

   {
     "metadata": {
       "config_version": "6.0",
       "analysis_mode": "static_isotropic",
       "sample": "protein_sample"
     },
     "analysis_settings": {
       "static_mode": true,
       "static_submode": "isotropic"
     },
     "initial_parameters": {
       "active_parameters": ["D0", "alpha", "D_offset"],
       "D0": 1e-12,
       "alpha": 1.0,
       "D_offset": 0.0
     },
     "file_paths": {
       "c2_data_file": "data/correlation_data.h5"
     }
   }

Laminar Flow Example
~~~~~~~~~~~~~~~~~~~

.. code-block:: json

   {
     "metadata": {
       "config_version": "6.0",
       "analysis_mode": "laminar_flow",
       "sample": "microgel_flow"
     },
     "analysis_settings": {
       "static_mode": false,
       "enable_angle_filtering": true
     },
     "initial_parameters": {
       "active_parameters": [
         "D0", "alpha", "D_offset", 
         "gamma_dot_t0", "beta", "gamma_dot_t_offset", "phi0"
       ],
       "D0": 1e-12,
       "alpha": 1.0,
       "D_offset": 0.0,
       "gamma_dot_t0": 1e-3,
       "beta": 1.0,
       "gamma_dot_t_offset": 0.0,
       "phi0": 0.0
     },
     "file_paths": {
       "c2_data_file": "data/correlation_data.h5",
       "phi_angles_file": "data/angles.npz"
     }
   }

Migration from Legacy Configurations
------------------------------------

From Legacy Static Mode
~~~~~~~~~~~~~~~~~~~~~~~

**Before** (legacy):

.. code-block:: json

   {
     "analysis_settings": {
       "static_mode": true
     }
   }

**After** (explicit):

.. code-block:: json

   {
     "analysis_settings": {
       "static_mode": true,
       "static_submode": "anisotropic"
     }
   }

**Note**: Legacy configurations automatically default to ``"anisotropic"`` mode for backward compatibility.

Scaling Optimization Migration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Remove** (now always enabled):

.. code-block:: json

   {
     "chi_squared_calculation": {
       "scaling_optimization": true  // Remove this line
     }
   }

This comprehensive configuration guide should help you effectively set up and customize analyses for your specific experimental needs.
