Migration Guide
===============

This guide helps users migrate configurations and workflows from legacy versions of the homodyne package to the current version 6.0, ensuring smooth transitions and taking advantage of new features.

Overview of Changes
-------------------

Version 6.0 introduces several important changes:

- **Always-On Scaling Optimization**: Scaling is now automatically enabled for all analyses
- **Enhanced Mode System**: Explicit static isotropic, static anisotropic, and laminar flow modes
- **Improved Configuration Templates**: Mode-specific templates with intelligent defaults
- **Enhanced Data Validation**: Integrated experimental data validation workflows
- **Performance Optimizations**: Numba JIT compilation and smart angle filtering

Breaking Changes
----------------

Scaling Optimization
~~~~~~~~~~~~~~~~~~~~

**Legacy Behavior**: Scaling optimization was optional and controlled by configuration

**Current Behavior**: **Scaling optimization is always enabled** for scientific accuracy

**Migration Required**:

.. code-block:: json

   # REMOVE this from configuration files
   {
     "chi_squared_calculation": {
       "scaling_optimization": true  // Remove this entire section or line
     }
   }

**Impact**: 
- All analyses now automatically apply proper g₁ to g₂ scaling
- Chi-squared statistics are always scientifically meaningful
- No computational penalty for this change

Mode Configuration Changes
~~~~~~~~~~~~~~~~~~~~~~~~~

**Legacy Mode Settings**:

.. code-block:: json

   # Legacy configuration (still supported)
   {
     "analysis_settings": {
       "static_mode": true
     }
   }

**Current Mode Settings**:

.. code-block:: json

   # Explicit mode specification (recommended)
   {
     "analysis_settings": {
       "static_mode": true,
       "static_submode": "anisotropic"  // New field
     }
   }

**Migration Path**:

1. **Automatic Migration**: Legacy configurations automatically default to anisotropic mode
2. **Explicit Migration**: Update configurations to specify exact mode

.. code-block:: json

   # For isotropic analysis (new, fastest)
   {
     "analysis_settings": {
       "static_mode": true,
       "static_submode": "isotropic"
     }
   }

   # For anisotropic analysis (default for legacy configs)
   {
     "analysis_settings": {
       "static_mode": true,
       "static_submode": "anisotropic"
     }
   }

   # For flow analysis (unchanged)
   {
     "analysis_settings": {
       "static_mode": false
     }
   }

Command Line Interface Changes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**New Command Line Flags**:

.. code-block:: bash

   # New explicit mode flags
   python run_homodyne.py --static-isotropic     # New fastest mode
   python run_homodyne.py --static-anisotropic   # Explicit anisotropic
   python run_homodyne.py --laminar-flow         # Explicit flow mode

**Legacy Compatibility**:

.. code-block:: bash

   # Legacy flag (still works but deprecated)
   python run_homodyne.py --static              # Maps to --static-anisotropic

Configuration Migration
-----------------------

Automated Migration Tools
~~~~~~~~~~~~~~~~~~~~~~~~~

Use the enhanced configuration generator for new-style configurations:

.. code-block:: bash

   # Create mode-specific configurations
   python create_config.py --mode static_isotropic --sample my_sample
   python create_config.py --mode static_anisotropic --sample my_sample
   python create_config.py --mode laminar_flow --sample my_sample

Manual Migration Steps
~~~~~~~~~~~~~~~~~~~~~~

**Step 1: Remove Obsolete Settings**

Remove scaling optimization settings (now always enabled):

.. code-block:: json

   {
     "chi_squared_calculation": {
       // Remove this entire section or just the scaling_optimization line
       "scaling_optimization": true  // Remove this
     }
   }

**Step 2: Specify Analysis Mode Explicitly**

Add explicit mode specification:

.. code-block:: json

   {
     "analysis_settings": {
       "static_mode": true,
       "static_submode": "anisotropic"  // Add this line
     }
   }

**Step 3: Update Active Parameters (Optional)**

Ensure active parameters are explicitly specified:

.. code-block:: json

   {
     "initial_parameters": {
       "active_parameters": ["D0", "alpha", "D_offset"],  // Explicit list
       "D0": 1e-12,
       "alpha": 1.0,
       "D_offset": 0.0
     }
   }

**Step 4: Add Performance Settings (Recommended)**

Take advantage of new performance features:

.. code-block:: json

   {
     "performance": {
       "use_numba_jit": true,         // Enable JIT acceleration
       "num_threads": 8,              // Optimize threading
       "enable_angle_filtering": true  // For large datasets
     }
   }

Mode-Specific Migration
-----------------------

Migrating to Static Isotropic Mode
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**When to Use**: Systems with no angular dependence, need for maximum speed

**Configuration Changes**:

.. code-block:: json

   {
     "analysis_settings": {
       "static_mode": true,
       "static_submode": "isotropic",
       // angle filtering automatically disabled
     },
     "initial_parameters": {
       "active_parameters": ["D0", "alpha", "D_offset"]  // 3 parameters
     }
   }

**Expected Behavior Changes**:
- No phi_angles_file loading (uses dummy angle)
- Angle filtering automatically disabled
- Significant speed improvement
- Single angle computation

Migrating to Static Anisotropic Mode
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**When to Use**: Default for legacy static configurations, systems with angular dependence

**Configuration Changes**:

.. code-block:: json

   {
     "analysis_settings": {
       "static_mode": true,
       "static_submode": "anisotropic",
       "enable_angle_filtering": true  // Recommended for performance
     },
     "initial_parameters": {
       "active_parameters": ["D0", "alpha", "D_offset"]  // 3 parameters
     }
   }

**Expected Behavior Changes**:
- phi_angles_file loaded for angle information
- Angle filtering can be enabled for performance
- Per-angle scaling optimization

Migrating to Laminar Flow Mode
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**When to Use**: Systems under flow or shear

**Configuration Changes**:

.. code-block:: json

   {
     "analysis_settings": {
       "static_mode": false,
       "enable_angle_filtering": true  // Highly recommended for performance
     },
     "initial_parameters": {
       "active_parameters": [
         "D0", "alpha", "D_offset",
         "gamma_dot_t0", "beta", "gamma_dot_t_offset", "phi0"
       ]  // All 7 parameters
     }
   }

**Expected Behavior Changes**:
- All flow parameters active
- Angle filtering recommended for performance
- Complex parameter space optimization

Performance Migration
---------------------

Numba JIT Acceleration
~~~~~~~~~~~~~~~~~~~~~~

**New Feature**: Numba JIT compilation for 3-5x speedup

**Migration**:

.. code-block:: json

   {
     "performance": {
       "use_numba_jit": true,  // Enable JIT compilation
       "numba_cache": true     // Cache compiled functions
     }
   }

**Environment Setup**:

.. code-block:: bash

   # Install Numba if not already installed
   pip install numba
   
   # Set environment variable for compatibility
   export NUMBA_DISABLE_INTEL_SVML=1

Smart Angle Filtering
~~~~~~~~~~~~~~~~~~~~~

**New Feature**: Intelligent angle filtering for performance without accuracy loss

**Migration**:

.. code-block:: json

   {
     "analysis_settings": {
       "enable_angle_filtering": true,
       "angle_filter_ranges": [[-10, 10], [170, 190]]  // Default ranges
     }
   }

**Benefits**:
- 3-5x speedup for large datasets
- Minimal accuracy impact
- Automatic for isotropic mode

Data Validation Integration
---------------------------

New Data Validation Features
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Enhanced Workflow**: Integrated experimental data validation

**Migration Options**:

1. **Command Line Integration**:

.. code-block:: bash

   # Add data validation to existing workflows
   python run_homodyne.py --plot-experimental-data --config my_config.json

2. **Configuration Integration**:

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

**Benefits**:
- Early problem detection
- Comprehensive quality assessment
- Visual diagnostics

Workflow Migration
------------------

Analysis Workflow Updates
~~~~~~~~~~~~~~~~~~~~~~~~~

**Legacy Workflow**:

.. code-block:: bash

   # Legacy approach
   python run_homodyne.py --static --method classical

**Current Recommended Workflow**:

.. code-block:: bash

   # 1. Data validation (new)
   python run_homodyne.py --plot-experimental-data --config my_config.json
   
   # 2. Choose appropriate mode
   python run_homodyne.py --static-isotropic --method classical     # Fastest
   python run_homodyne.py --static-anisotropic --method classical   # Default
   python run_homodyne.py --laminar-flow --method classical         # Full physics
   
   # 3. Comprehensive analysis (unchanged)
   python run_homodyne.py --static-anisotropic --method all         # Classical + MCMC

Progressive Analysis Strategy
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**New Recommendation**: Use progressive complexity approach

.. code-block:: bash

   # Start simple for exploration
   python run_homodyne.py --static-isotropic --method classical
   
   # Add complexity if needed
   python run_homodyne.py --static-anisotropic --method classical
   
   # Full analysis for nonequilibrium systems
   python run_homodyne.py --laminar-flow --method all

API Migration
-------------

Python API Changes
~~~~~~~~~~~~~~~~~~

**No Breaking Changes**: The Python API remains fully compatible

**New Recommendations**:

.. code-block:: python

   from homodyne import HomodyneAnalysisCore, ConfigManager
   
   # Enhanced configuration loading
   config = ConfigManager("my_config.json")
   
   # Runtime parameter override (enhanced)
   config.override_parameters({
       "D0": 1.5e-12,
       "use_numba_jit": True,
       "enable_angle_filtering": True
   })
   
   # Analysis execution (unchanged)
   analysis = HomodyneAnalysisCore(config)
   results = analysis.optimize_classical()

Migration Validation
--------------------

Testing Migration Success
~~~~~~~~~~~~~~~~~~~~~~~~~

**Verify Configuration Loading**:

.. code-block:: python

   from homodyne.core import ConfigManager
   
   try:
       config = ConfigManager("migrated_config.json")
       print("Migration successful")
       print(f"Analysis mode: {config.get_analysis_mode()}")
   except Exception as e:
       print(f"Migration issue: {e}")

**Compare Results**:

.. code-block:: python

   # Run with legacy and new configurations
   legacy_results = run_analysis("legacy_config.json")
   new_results = run_analysis("migrated_config.json")
   
   # Results should be similar (scaling now always enabled)
   parameter_diff = np.abs(legacy_results.x - new_results.x)
   print(f"Parameter differences: {parameter_diff}")

**Performance Validation**:

.. code-block:: bash

   # Test performance improvements
   python benchmark_performance.py --config migrated_config.json

Migration Checklist
-------------------

Pre-Migration Checklist
~~~~~~~~~~~~~~~~~~~~~~~

- [ ] **Backup existing configurations** and results
- [ ] **Install updated dependencies** (especially Numba)
- [ ] **Review analysis modes** needed for your systems
- [ ] **Test migration on non-critical data** first

Configuration Migration Checklist
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- [ ] **Remove scaling_optimization settings** (now always enabled)
- [ ] **Add explicit static_submode specification** 
- [ ] **Add performance settings** (Numba, threading, angle filtering)
- [ ] **Add data validation settings** (optional but recommended)
- [ ] **Update active_parameters lists** (explicitly specify)

Post-Migration Checklist
~~~~~~~~~~~~~~~~~~~~~~~~

- [ ] **Validate configuration loading** without errors
- [ ] **Test analysis execution** on known datasets
- [ ] **Compare results** with legacy version (should be similar)
- [ ] **Verify performance improvements** (should be faster)
- [ ] **Test new features** (data validation, mode selection)

Troubleshooting Migration Issues
--------------------------------

Common Migration Problems
~~~~~~~~~~~~~~~~~~~~~~~~

**Configuration Loading Errors**:

.. code-block:: python

   # Validate migrated configuration
   python -m json.tool migrated_config.json

**Mode Detection Issues**:

.. code-block:: python

   config = ConfigManager("migrated_config.json")
   print(f"Detected mode: {config.get_analysis_mode()}")

**Performance Issues**:

.. code-block:: bash

   # Verify Numba installation
   python -c "import numba; print(numba.__version__)"
   
   # Set environment variables
   export NUMBA_DISABLE_INTEL_SVML=1

**Result Differences**:
- Small differences expected due to always-on scaling optimization
- Large differences may indicate configuration issues
- Use comparison tools to validate migrations

Migration Support
-----------------

Getting Help
~~~~~~~~~~~

**Documentation**: Refer to mode-specific documentation for detailed configuration options

**Examples**: Use the enhanced create_config.py tool to generate reference configurations

**Validation**: Use built-in validation tools to verify migrations

**Support**: File issues with specific migration problems and include:
- Legacy configuration (sanitized)
- Attempted migration configuration
- Error messages or unexpected behavior
- System information

This comprehensive migration guide ensures smooth transitions to the enhanced homodyne package while taking advantage of significant performance and usability improvements.
