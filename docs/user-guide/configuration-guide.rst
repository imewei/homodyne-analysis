Configuration Guide
====================

This comprehensive guide covers configuration file structure, optimization settings, solver configurations, and dataset-specific templates for homodyne analysis.

.. currentmodule:: homodyne.optimization

Overview
--------

The homodyne package uses JSON configuration files to specify analysis parameters, file paths, and optimization options. It provides extensively optimized configuration templates designed for different dataset sizes and analysis requirements. Proper configuration is critical for achieving reliable convergence and optimal performance in XPCS data analysis.

**Key Optimization Principles:**

* **Scientific Data Reality**: Settings appropriate for experimental noise levels
* **Convergence over Precision**: Better to converge at 1e-6 than fail at 1e-10
* **Dataset-Adaptive Scaling**: Larger datasets enable looser tolerances
* **Problem-Specific Tuning**: Different settings for different analysis modes
* **Computational Efficiency**: Eliminate wasteful over-iteration

Quick Start
-----------

**Generate a Template:**

.. code-block:: bash

   # Create configuration for specific mode
   homodyne-config --mode static_isotropic --sample my_experiment

**Basic Configuration Structure:**

.. code-block:: json

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

Configuration File Structure
----------------------------

Analysis Settings
~~~~~~~~~~~~~~~~~

Controls the analysis mode and behavior:

.. code-block:: json

   {
     "analysis_settings": {
       "static_mode": true,
       "static_submode": "isotropic",
       "enable_angle_filtering": true,
       "angle_filter_ranges": [[-5, 5], [175, 185]]
     }
   }

File Paths
~~~~~~~~~~

Specify input data locations:

.. code-block:: json

   {
     "file_paths": {
       "c2_data_file": "data/my_correlation_data.h5",
       "phi_angles_file": "data/scattering_angles.txt",
       "output_directory": "results/"
     }
   }

Initial Parameters
~~~~~~~~~~~~~~~~~~

Starting values for optimization:

.. code-block:: json

   {
     "initial_parameters": {
       "parameter_names": ["D0", "alpha", "D_offset"],
       "values": [1000, -0.5, 100],
       "active_parameters": ["D0", "alpha", "D_offset"]
     }
   }

Parameter Constraints and Ranges
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The homodyne package implements comprehensive physical constraints:

**Core Model Parameters:**

.. list-table::
   :header-rows: 1
   :widths: 25 25 35 15

   * - Parameter
     - Range
     - Distribution (MCMC)
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

**Physical Function Constraints:**

The package automatically enforces positivity for time-dependent functions:

- **D(t) = D₀(t)^α + D_offset** → **max(D(t), 1×10⁻¹⁰)**
- **γ̇(t) = γ̇₀(t)^β + γ̇_offset** → **max(γ̇(t), 1×10⁻¹⁰)**

**Scaling Parameters:**

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

Configuration Templates
-----------------------

Dataset Size Categories
~~~~~~~~~~~~~~~~~~~~~~~

The package provides three categories of optimized templates:

.. list-table:: Configuration Template Categories
   :widths: 20 20 30 30
   :header-rows: 1

   * - Category
     - Data Points
     - Optimization Focus
     - Performance Impact
   * - **Small Dataset**
     - <50K points
     - Maximum precision & validation
     - 2-3x longer runtime
   * - **Standard**
     - 50K-1M points
     - Balanced accuracy & efficiency
     - Baseline performance
   * - **Large Dataset**
     - >1M points
     - Aggressive optimization
     - 35-65% faster

Template Overview
~~~~~~~~~~~~~~~~~

.. list-table:: Available Configuration Templates
   :widths: 35 25 20 20
   :header-rows: 1

   * - Template File
     - Analysis Mode
     - Dataset Category
     - Parameters
   * - ``config_template.json``
     - Universal
     - Standard
     - Mode-dependent
   * - ``config_static_isotropic.json``
     - Static Isotropic
     - Standard
     - 3 params
   * - ``config_static_anisotropic.json``
     - Static Anisotropic
     - Standard
     - 3 params + angles
   * - ``config_laminar_flow.json``
     - Laminar Flow
     - Standard
     - 7 params
   * - ``config_small_dataset_*.json``
     - All modes
     - Small Dataset
     - Mode-dependent
   * - ``config_large_dataset_*.json``
     - All modes
     - Large Dataset
     - Mode-dependent

Classical Optimization Settings
-------------------------------

The default configurations have been extensively optimized based on scientific data fitting requirements and benchmarking.

Nelder-Mead Method
~~~~~~~~~~~~~~~~~~

**Problem-Specific Optimized Settings:**

.. code-block:: json

   {
     "classical": {
       "nelder_mead": {
         "static_isotropic": {
           "maxiter": 1500,
           "xatol": 1e-6,
           "fatol": 1e-6
         },
         "static_anisotropic": {
           "maxiter": 2000,
           "xatol": 1e-6,
           "fatol": 1e-6
         },
         "laminar_flow": {
           "maxiter": 4000,
           "xatol": 1e-5,
           "fatol": 1e-5
         }
       }
     }
   }

**Dataset Size Adaptations:**

.. code-block:: json

   {
     "dataset_scaling": {
       "small_dataset": {
         "maxiter_multiplier": 2.5,
         "tolerance_tightening": "1e-8 to 1e-9"
       },
       "large_dataset": {
         "maxiter_reduction": "25-50%",
         "tolerance_relaxation": "5e-6 to 1e-5"
       }
     }
   }

Gurobi Trust Region Method
~~~~~~~~~~~~~~~~~~~~~~~~~~

**Optimized Trust Region Configuration:**

.. code-block:: json

   {
     "classical": {
       "gurobi_trust_region": {
         "max_iterations": 300,
         "tolerance": 1e-5,
         "initial_radius": 5.0,
         "max_radius": 50.0,
         "eta1": 0.1,
         "eta2": 0.8,
         "gamma1": 0.5,
         "gamma2": 3.0
       }
     }
   }

**Why These Changes Work Better:**

* **Larger initial radius**: Faster exploration of parameter space
* **More aggressive expansion**: Quicker convergence when making progress
* **Looser tolerances**: Appropriate for experimental data precision

MCMC Settings
-------------

MCMC NUTS Optimization
~~~~~~~~~~~~~~~~~~~~~~

**Problem-Specific MCMC Settings:**

.. code-block:: json

   {
     "mcmc": {
       "static_isotropic": {
         "draws": 2500,
         "tune": 700,
         "thin": 1,
         "target_accept": 0.85,
         "max_treedepth": 8
       },
       "static_anisotropic": {
         "draws": 3000,
         "tune": 800,
         "thin": 1,
         "target_accept": 0.83,
         "max_treedepth": 9
       },
       "laminar_flow": {
         "draws": 4000,
         "tune": 1200,
         "thin": 2,
         "target_accept": 0.78,
         "max_treedepth": 10
       }
     }
   }

**Dataset Size Scaling:**

.. code-block:: json

   {
     "mcmc_scaling": {
       "small_dataset": {
         "draws_multiplier": 1.4,
         "tune_multiplier": 1.8,
         "target_accept_increase": 0.05,
         "max_treedepth_increase": 1
       },
       "large_dataset": {
         "draws_reduction": "20-40%",
         "tune_reduction": "15-30%",
         "thin_increase": "2-4",
         "target_accept_decrease": "0.05-0.10"
       }
     }
   }

Robust Optimization Solver Configuration
----------------------------------------

Solver Hierarchy
~~~~~~~~~~~~~~~~

The optimization follows this solver preference order:

1. **CLARABEL** (Primary) - Modern interior-point solver
2. **SCS** (Fallback) - Splitting conic solver for robustness  
3. **CVXOPT** (Compatibility) - Python-based solver
4. **GUROBI** (Commercial) - When license available

CLARABEL (Primary Solver)
~~~~~~~~~~~~~~~~~~~~~~~~~~

**Optimized Settings for Scientific Data:**

.. code-block:: json

   {
     "robust_optimization": {
       "solver_settings": {
         "CLARABEL": {
           "max_iter": 400,
           "tol_gap_abs": 1e-5,
           "tol_gap_rel": 1e-5,
           "tol_feas": 1e-6,
           "equilibrate_enable": true,
           "equilibrate_max_iter": 30,
           "static_regularization_enable": true,
           "dynamic_regularization_enable": true,
           "presolve_enable": true
         }
       }
     }
   }

**Problem-Specific Adaptations:**

.. code-block:: json

   {
     "static_isotropic": {"max_iter": 400, "equilibrate_max_iter": 25},
     "static_anisotropic": {"max_iter": 450, "equilibrate_max_iter": 25}, 
     "laminar_flow": {"max_iter": 600, "equilibrate_max_iter": 40, "tol_gap_abs": 1e-4}
   }

SCS (Fallback Solver)  
~~~~~~~~~~~~~~~~~~~~~

**Optimized Fallback Configuration:**

.. code-block:: json

   {
     "SCS": {
       "max_iters": 8000,
       "eps": 1e-4,
       "alpha": 1.8,
       "scale": 3.0,
       "normalize": true,
       "adaptive_scale": true,
       "acceleration_lookback": 20,
       "time_limit_secs": 300
     }
   }

**Problem-Specific Scaling:**

.. code-block:: json

   {
     "static_isotropic": {"max_iters": 8000, "scale": 3.0},
     "static_anisotropic": {"max_iters": 10000, "scale": 3.5},
     "laminar_flow": {"max_iters": 15000, "scale": 5.0, "eps": 1e-3}
   }

Dataset-Specific Optimization
-----------------------------

Small Dataset Templates (<50K points)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Key Features:**

* **Enhanced Precision**: 10-100x tighter tolerances (1e-8 to 1e-9)
* **Conservative Sampling**: 150-300% more iterations/draws
* **Comprehensive Validation**: Small sample corrections and bootstrap methods
* **Full Precision Output**: No compression, complete diagnostics

**Performance Characteristics:**

.. list-table:: Small Dataset Performance Trade-offs
   :widths: 30 35 35
   :header-rows: 1

   * - Method
     - Standard Config
     - Small Dataset Config
   * - **Static Isotropic Classical**
     - 25-60s
     - 90-180s (3-5x longer)
   * - **Static Isotropic MCMC**
     - 45-90 minutes  
     - 2-4 hours (2-3x longer)
   * - **Laminar Flow MCMC**
     - 4-7 hours
     - 12-20 hours (2-3x longer)

Large Dataset Templates (>1M points)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Key Features:**

* **Aggressive Optimization**: 40-60% performance improvements
* **Statistical Precision Scaling**: Looser tolerances with high SNR data
* **Memory Optimization**: Enhanced caching and float32 precision
* **Computational Efficiency**: Reduced iterations due to smoother objectives

**Performance Improvements:**

.. list-table:: Large Dataset Performance Gains  
   :widths: 30 25 25 20
   :header-rows: 1

   * - Method
     - Standard Config
     - Large Dataset Config
     - Improvement
   * - **Static Isotropic Classical**
     - 45-90s
     - 18-35s
     - 60% faster
   * - **Static Isotropic MCMC**
     - 2-4 hours
     - 45-90 minutes
     - 65% faster
   * - **Static Anisotropic Classical**
     - 60-120s  
     - 30-60s
     - 50% faster
   * - **Laminar Flow MCMC**
     - 6-10 hours
     - 4-7 hours
     - 35% faster

**Memory Optimization Results:**

.. list-table:: Memory Usage Reduction
   :widths: 25 25 25 25
   :header-rows: 1

   * - Dataset Size
     - Standard Config  
     - Large Dataset Config
     - Memory Savings
   * - **1M data points**
     - 4-6 GB
     - 2-3 GB
     - 40% reduction
   * - **5M data points**
     - 12-18 GB
     - 6-9 GB
     - 50% reduction
   * - **20M data points**
     - 35-50 GB
     - 15-25 GB
     - 55% reduction

Configuration Selection Guide
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Decision Flow
~~~~~~~~~~~~~

.. code-block:: text

   Dataset Analysis
   ├── <50K points → Small Dataset Templates
   │   ├── <5K points → Most conservative settings
   │   ├── 5-15K points → Standard small dataset templates  
   │   └── 15-50K points → Moderate small dataset settings
   ├── 50K-1M points → Standard Templates
   └── >1M points → Large Dataset Templates
       ├── 1-2M points → Standard large dataset values
       ├── 5-10M points → Optimal template range
       └── 10-20M points → Additional 15-20% reduction

Template Selection Criteria
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: json

   {
     "selection_criteria": {
       "small_dataset": {
         "priority": "Maximum precision & validation",
         "trade_off": "Longer runtime for reliability",
         "use_case": "Critical analysis with limited data"
       },
       "standard": {
         "priority": "Balanced accuracy & efficiency", 
         "trade_off": "Optimal for typical XPCS datasets",
         "use_case": "General-purpose analysis"
       },
       "large_dataset": {
         "priority": "Computational efficiency",
         "trade_off": "Faster runtime with maintained accuracy",
         "use_case": "High-throughput analysis workflows"
       }
     }
   }

Usage Examples
--------------

Command Line Interface
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Small dataset analysis (high precision)
   homodyne --config config_small_dataset_static_isotropic.json --method all
   
   # Standard dataset analysis 
   homodyne --config config_static_isotropic.json --method classical
   
   # Large dataset analysis (optimized performance)
   homodyne --config config_large_dataset_laminar_flow.json --method mcmc
   
   # GPU acceleration for large datasets
   homodyne-gpu --config config_large_dataset_template.json --method mcmc

Python API
~~~~~~~~~~

.. code-block:: python

   from homodyne import HomodyneAnalysisCore, ConfigManager
   
   # Load appropriate configuration for dataset size
   config = ConfigManager("config_large_dataset_static_isotropic.json")
   analysis = HomodyneAnalysisCore(config)
   
   # Run optimized analysis
   results = analysis.optimize_all()
   
   # Save results with appropriate compression
   results.save_compressed("dataset_results.npz")

Troubleshooting
---------------

Common Issues
~~~~~~~~~~~~~

**Solver Convergence Failures:**

* **Symptoms**: "Numerical difficulties" or "Maximum iterations reached"
* **Solutions**: 
  - Increase iteration limits (``max_iter`` to 600-1000)
  - Relax tolerances (``tol_gap_abs`` to 1e-4 or 1e-5)
  - Enable equilibration (``equilibrate_enable: true``)

**Performance Issues:**

* **Symptoms**: Analysis takes too long or runs out of memory
* **Solutions**:
  - Use appropriate dataset-size template
  - Enable memory optimization (``use_float32: true``)
  - Increase chunk size for large datasets
  - Consider data subsampling for initial exploration

**Validation Failures:**

* **Symptoms**: Parameter estimates seem unrealistic
* **Solutions**:
  - Use smaller dataset template for higher precision
  - Enable enhanced validation features
  - Check data quality and preprocessing
  - Review parameter bounds and priors

Best Practices
--------------

Configuration Management
~~~~~~~~~~~~~~~~~~~~~~~~

1. **Start with Templates**: Use provided optimized templates as starting points
2. **Monitor Performance**: Track convergence rates and solution times
3. **Document Changes**: Note successful modifications for your datasets
4. **Version Control**: Save working configurations for reproducibility
5. **Validate Results**: Compare different configuration results for consistency

Parameter Tuning Workflow
~~~~~~~~~~~~~~~~~~~~~~~~~~

1. **Assess Dataset**: Determine size category and analysis requirements
2. **Select Template**: Choose appropriate template based on dataset and precision needs
3. **Monitor Convergence**: Target >95% success rate for automated workflows
4. **Optimize Performance**: Balance accuracy vs. computational resources
5. **Validate Results**: Use cross-validation and bootstrap methods when appropriate

Configuration Template Examples
-------------------------------

Static Isotropic Template
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: json

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

Laminar Flow Template
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: json

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

Configuration Validation and Environment Setup
----------------------------------------------

Validation
~~~~~~~~~~

**Check Configuration Syntax:**

.. code-block:: bash

   # Validate JSON syntax
   python -m json.tool my_config.json

**Test Configuration:**

.. code-block:: python

   from homodyne import ConfigManager

   # Load and validate configuration
   config = ConfigManager("my_config.json")
   config.validate()
   print("✅ Configuration is valid")

Environment Variables
~~~~~~~~~~~~~~~~~~~~~

You can use environment variables in configurations:

.. code-block:: json

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

Common Configuration Patterns
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**High-Performance Setup:**

.. code-block:: json

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

**MCMC with Convergence Diagnostics:**

.. code-block:: json

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

Summary
-------

The optimized configuration templates provide:

**Small Dataset Templates:**

* Maximum precision with 10-100x tighter tolerances
* Enhanced validation with small sample corrections
* Comprehensive diagnostics for thorough analysis
* Trade-off: 2-3x longer runtime for highest reliability

**Standard Templates:**

* Balanced accuracy and computational efficiency
* Scientifically appropriate tolerances for typical XPCS data
* Robust convergence with >95% success rates
* Optimal for general-purpose analysis workflows

**Large Dataset Templates:**

* Significant performance gains (35-65% speedup)
* Memory efficiency with 40-55% reduction
* Maintained scientific accuracy with high SNR data
* Scalable to datasets up to 20M data points

These configurations represent the optimal balance between **scientific accuracy**, **computational efficiency**, and **robust convergence** for XPCS data analysis across all dataset sizes and complexity levels.