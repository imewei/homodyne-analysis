Configuration Guide
===================

The homodyne package uses JSON configuration files to specify analysis parameters, file paths, and options.

Quick Configuration
-------------------

**Generate a Template**:

.. code-block:: bash

   # Create configuration for specific mode
   python create_config.py --mode static_isotropic --sample my_experiment

**Basic Structure**:

.. code-block:: javascript

   {
     "analysis_settings": {
       "static_mode": true,
       "static_submode": "isotropic"
     },
     "file_paths": {
       "c2_data_file": "data/correlation_data.h5",
       "phi_angles_file": "data/phi_angles.txt"
     },
     "initial_parameters": {
       "values": [1000, -0.5, 100]
     }
   }

Configuration Sections
----------------------

Analysis Settings
~~~~~~~~~~~~~~~~~

Controls the analysis mode and behavior:

.. code-block:: javascript

   {
     "analysis_settings": {
       "static_mode": true,                    // true for static, false for flow
       "static_submode": "isotropic",          // "isotropic" or "anisotropic"
       "enable_angle_filtering": true,         // Enable angle filtering optimization
       "angle_filter_ranges": [[-5, 5], [175, 185]]  // Angle ranges to analyze
     }
   }

File Paths
~~~~~~~~~~

Specify input data locations:

.. code-block:: javascript

   {
     "file_paths": {
       "c2_data_file": "data/my_correlation_data.h5",  // Main data file
       "phi_angles_file": "data/scattering_angles.txt", // Angle file
       "output_directory": "results/"                   // Output location
     }
   }

Initial Parameters
~~~~~~~~~~~~~~~~~~

Starting values for optimization:

.. code-block:: javascript

   {
     "initial_parameters": {
       "parameter_names": ["D0", "alpha", "D_offset"],
       "values": [1000, -0.5, 100],
       "active_parameters": ["D0", "alpha", "D_offset"]  // Parameters to optimize
     }
   }

Parameter Bounds
~~~~~~~~~~~~~~~~

Optimization constraints:

.. code-block:: javascript

   {
     "parameter_space": {
       "bounds": [
         {"name": "D0", "min": 100, "max": 10000, "type": "log-uniform"},
         {"name": "alpha", "min": -2.0, "max": 0.0, "type": "uniform"},
         {"name": "D_offset", "min": 0, "max": 1000, "type": "uniform"}
       ]
     }
   }

Optimization Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~

**Classical Optimization**:

.. code-block:: javascript

   {
     "optimization_config": {
       "classical": {
         "method": "Nelder-Mead",
         "max_iterations": 1000,
         "tolerance": 1e-6
       }
     }
   }

**MCMC Configuration**:

.. code-block:: javascript

   {
     "optimization_config": {
       "mcmc_sampling": {
         "enabled": true,
         "draws": 2000,
         "tune": 1000,
         "chains": 4,
         "cores": 4,
         "target_accept": 0.9
       }
     }
   }

Performance Settings
~~~~~~~~~~~~~~~~~~~~

Optimize computation:

.. code-block:: javascript

   {
     "performance_settings": {
       "num_threads": 4,
       "data_type": "float64",
       "memory_limit_gb": 8,
       "enable_jit": true
     }
   }

Configuration Templates
-----------------------

**Static Isotropic Template**:

.. code-block:: javascript

   {
     "metadata": {
       "config_version": "6.0",
       "analysis_mode": "static_isotropic"
     },
     "analysis_settings": {
       "static_mode": true,
       "static_submode": "isotropic"
     },
     "file_paths": {
       "c2_data_file": "data/correlation_data.h5"
     },
     "initial_parameters": {
       "parameter_names": ["D0", "alpha", "D_offset"],
       "values": [1000, -0.5, 100],
       "active_parameters": ["D0", "alpha", "D_offset"]
     },
     "parameter_space": {
       "bounds": [
         {"name": "D0", "min": 100, "max": 10000, "type": "log-uniform"},
         {"name": "alpha", "min": -2.0, "max": 0.0, "type": "uniform"},
         {"name": "D_offset", "min": 0, "max": 1000, "type": "uniform"}
       ]
     }
   }

**Laminar Flow Template**:

.. code-block:: javascript

   {
     "metadata": {
       "config_version": "6.0", 
       "analysis_mode": "laminar_flow"
     },
     "analysis_settings": {
       "static_mode": false,
       "enable_angle_filtering": true,
       "angle_filter_ranges": [[-5, 5], [175, 185]]
     },
     "file_paths": {
       "c2_data_file": "data/correlation_data.h5",
       "phi_angles_file": "data/phi_angles.txt"
     },
     "initial_parameters": {
       "parameter_names": ["D0", "alpha", "D_offset", "gamma_dot_t0", "beta", "gamma_dot_t_offset", "phi0"],
       "values": [1000, -0.5, 100, 10, 0.5, 1, 0],
       "active_parameters": ["D0", "alpha", "D_offset", "gamma_dot_t0"]
     },
     "optimization_config": {
       "mcmc_sampling": {
         "enabled": true,
         "draws": 2000,
         "tune": 1000,
         "chains": 4
       }
     }
   }

Configuration Validation
-------------------------

**Check Configuration Syntax**:

.. code-block:: bash

   # Validate JSON syntax
   python -m json.tool my_config.json

**Test Configuration**:

.. code-block:: python

   from homodyne import ConfigManager
   
   # Load and validate configuration
   config = ConfigManager("my_config.json")
   config.validate()
   print("✅ Configuration is valid")

Common Configuration Patterns
------------------------------

**High-Performance Setup**:

.. code-block:: javascript

   {
     "analysis_settings": {
       "enable_angle_filtering": true,
       "angle_filter_ranges": [[-10, 10], [170, 190]]
     },
     "performance_settings": {
       "num_threads": 8,
       "data_type": "float32",
       "enable_jit": true
     }
   }

**MCMC with Convergence Diagnostics**:

.. code-block:: javascript

   {
     "optimization_config": {
       "mcmc_sampling": {
         "draws": 4000,
         "tune": 2000,
         "chains": 6,
         "target_accept": 0.95
       }
     },
     "validation_rules": {
       "mcmc_convergence": {
         "rhat_thresholds": {
           "excellent_threshold": 1.01,
           "good_threshold": 1.05,
           "acceptable_threshold": 1.1
         }
       }
     }
   }

Environment Variables
---------------------

You can use environment variables in configurations:

.. code-block:: javascript

   {
     "file_paths": {
       "c2_data_file": "${DATA_DIR}/correlation_data.h5",
       "output_directory": "${HOME}/homodyne_results"
     }
   }

Set environment variables:

.. code-block:: bash

   export DATA_DIR=/path/to/data
   export HOME=/home/username

Troubleshooting
---------------

**Configuration Errors**:

- **Invalid JSON**: Check syntax with ``python -m json.tool config.json``
- **Missing files**: Verify all file paths exist
- **Parameter bounds**: Ensure min < max for all parameters
- **Mode mismatch**: Check that parameters match the selected analysis mode

**Performance Issues**:

- Enable angle filtering for faster computation
- Use ``float32`` data type to reduce memory usage
- Increase ``num_threads`` to match your CPU cores
- Set appropriate ``memory_limit_gb`` based on available RAM