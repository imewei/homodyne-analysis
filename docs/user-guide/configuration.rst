Configuration Guide
===================

The homodyne package uses JSON configuration files to specify analysis parameters, file paths, and options.

Quick Configuration
-------------------

**Generate a Template**:

.. code-block:: bash

   # Create configuration for specific mode
   homodyne-config --mode static_isotropic --sample my_experiment

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

Parameter Bounds and Prior Distributions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Optimization constraints and MCMC prior distributions:

.. code-block:: javascript

   {
     "parameter_space": {
       "bounds": [
         {"name": "D0", "min": 100, "max": 10000, "type": "Normal"},
         {"name": "alpha", "min": -2.0, "max": 2.0, "type": "Normal"},
         {"name": "D_offset", "min": 0, "max": 1000, "type": "Normal"}
       ]
     }
   }

.. note::
   **MCMC Prior Distributions**: All parameters use **Normal distributions** in MCMC sampling. The ``type`` field in configuration files affects both classical optimization bounds and MCMC prior specification. All seven parameters (D0, alpha, D_offset, gamma_dot_t0, beta, gamma_dot_t_offset, phi0) use Normal priors in the Bayesian analysis.

Parameter Constraints and Ranges
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The homodyne package implements comprehensive physical constraints to ensure scientifically meaningful results:

**Core Model Parameters**

.. list-table::
   :header-rows: 1
   :widths: 25 25 35 15

   * - Parameter
     - Range
     - Distribution
     - Physical Constraint
   * - ``D0``
     - [1.0, 1000000.0] Å²/s
     - TruncatedNormal(μ=10000.0, σ=1000.0)
     - Must be positive
   * - ``alpha``
     - [-2.0, 2.0]
     - Normal(μ=-1.5, σ=0.1)
     - none
   * - ``D_offset``
     - [-100, 100] Å²/s
     - Normal(μ=0.0, σ=10.0)
     - none
   * - ``gamma_dot_t0``
     - [1e-06, 1.0] s⁻¹
     - TruncatedNormal(μ=0.001, σ=0.01)
     - Must be positive
   * - ``beta``
     - [-2.0, 2.0]
     - Normal(μ=0.0, σ=0.1)
     - none
   * - ``gamma_dot_t_offset``
     - [-0.01, 0.01] s⁻¹
     - Normal(μ=0.0, σ=0.001)
     - none
   * - ``phi0``
     - [-10, 10] degrees
     - Normal(μ=0.0, σ=5.0)
     - angular

**Physical Function Constraints**

The package automatically enforces positivity for time-dependent functions:

- **D(t) = D₀(t)^α + D_offset** → **max(D(t), 1×10⁻¹⁰)**

  - Prevents negative diffusion coefficients from any parameter combination
  - Maintains numerical stability with minimal threshold

- **γ̇(t) = γ̇₀(t)^β + γ̇_offset** → **max(γ̇(t), 1×10⁻¹⁰)**

  - Prevents negative shear rates from any parameter combination
  - Ensures physical validity in all optimization scenarios

**Scaling Parameters for Correlation Functions**

The relationship **c2_fitted = c2_theory × contrast + offset** uses bounded parameters:

.. list-table::
   :header-rows: 1
   :widths: 20 20 40 30

   * - Parameter
     - Range
     - Distribution
     - Physical Meaning
   * - ``contrast``
     - (0.05, 0.5]
     - TruncatedNormal(μ=0.3, σ=0.1)
     - Correlation strength scaling
   * - ``offset``
     - (0.05, 1.95)
     - TruncatedNormal(μ=1.0, σ=0.2)
     - Baseline correlation level
   * - ``c2_fitted``
     - [1.0, 2.0]
     - *derived*
     - Final correlation function
   * - ``c2_theory``
     - [0.0, 1.0]
     - *derived*
     - Theoretical correlation bounds

Optimization Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~

**Classical Optimization**:

.. code-block:: javascript

   {
     "optimization_config": {
       "classical_optimization": {
         "methods": ["Nelder-Mead"],
         "method_options": {
           "Nelder-Mead": {
             "maxiter": 1000,
             "xatol": 1e-6,
             "fatol": 1e-6
           },
           "Gurobi": {
             "max_iterations": 1000,
             "tolerance": 1e-6,
             "output_flag": 0,
             "method": 2,
             "time_limit": 300
           }
         }
       }
     }
   }

**Available Optimization Methods**:

- **Nelder-Mead**: Derivative-free simplex method, robust for noisy functions
- **Gurobi**: Quadratic programming solver (requires license), good for smooth functions with bounds

.. note::
   Gurobi is automatically detected if installed and licensed. It uses quadratic approximation
   via finite differences and excels with smooth objective functions and bounds constraints.

**Robust Optimization Configuration**:

.. code-block:: javascript

   {
     "optimization_config": {
       "robust_optimization": {
         "enabled": true,
         "uncertainty_model": "wasserstein",
         "method_options": {
           "wasserstein": {
             "uncertainty_radius": 0.02,
             "regularization_alpha": 0.005
           },
           "scenario": {
             "n_scenarios": 30,
             "bootstrap_method": "residual",
             "parallel_scenarios": true
           },
           "ellipsoidal": {
             "gamma": 0.08,
             "l1_regularization": 0.0005,
             "l2_regularization": 0.005
           }
         },
         "solver_settings": {
           "preferred_solver": "CLARABEL",
           "timeout": 300,
           "enable_caching": true
         }
       }
     }
   }

**Robust Methods Available**:

- **Wasserstein DRO**: Distributionally robust optimization using Wasserstein uncertainty sets
- **Scenario-based**: Multi-scenario optimization using bootstrap resampling for outlier resistance
- **Ellipsoidal**: Robust least squares with bounded uncertainty in correlation functions

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
         {"name": "D0", "min": 100, "max": 10000, "type": "Normal"},
         {"name": "alpha", "min": -2.0, "max": 2.0, "type": "Normal"},
         {"name": "D_offset", "min": 0, "max": 1000, "type": "Normal"}
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
