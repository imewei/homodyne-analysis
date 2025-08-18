Troubleshooting Guide
====================

This comprehensive troubleshooting guide addresses common issues encountered when using the homodyne package, organized by category with specific solutions and preventive measures.

Installation Issues
-------------------

Dependency Problems
~~~~~~~~~~~~~~~~~~~

**Missing Core Dependencies**

.. code-block:: text

   ImportError: No module named 'scipy' / 'numpy' / 'matplotlib'

**Solution**:

.. code-block:: bash

   # Install core dependencies
   pip install numpy scipy matplotlib

**Numba Installation Issues**

.. code-block:: text

   ImportError: No module named 'numba'
   Or: Numba compilation errors

**Solutions**:

.. code-block:: bash

   # Install Numba
   pip install numba
   
   # If compilation issues persist, disable Intel SVML
   export NUMBA_DISABLE_INTEL_SVML=1

**MCMC Dependencies Missing**

.. code-block:: text

   ImportError: No module named 'pymc' / 'arviz' / 'pytensor'

**Solution**:

.. code-block:: bash

   # Install MCMC capabilities
   pip install pymc arviz pytensor

**Version Compatibility Issues**

.. code-block:: text

   Package version conflicts or incompatibilities

**Solution**:

.. code-block:: bash

   # Create fresh environment
   python -m venv fresh_env
   source fresh_env/bin/activate  # On Windows: fresh_env\\Scripts\\activate
   
   # Install in recommended order
   pip install numpy scipy matplotlib
   pip install numba
   pip install pymc arviz pytensor

Configuration Problems
----------------------

JSON Syntax Errors
~~~~~~~~~~~~~~~~~~

**Invalid JSON Format**

.. code-block:: text

   JSONDecodeError: Expecting ',' delimiter: line X column Y

**Solution**:

.. code-block:: bash

   # Validate JSON syntax
   python -m json.tool my_config.json

**Common JSON Issues**:
- Missing commas between elements
- Trailing commas (not allowed in JSON)
- Unquoted strings or incorrect quotes
- Missing closing brackets/braces

**Example Fix**:

.. code-block:: json

   # Incorrect
   {
     "param1": value1,  // Comment not allowed
     "param2": value2,  // Trailing comma error
   }
   
   # Correct
   {
     "param1": "value1",
     "param2": "value2"
   }

Mode Configuration Issues
~~~~~~~~~~~~~~~~~~~~~~~~

**Conflicting Mode Settings**

.. code-block:: text

   Warning: Conflicting analysis mode settings detected

**Solution**: Ensure consistent mode specification:

.. code-block:: json

   # For isotropic mode
   {
     "analysis_settings": {
       "static_mode": true,
       "static_submode": "isotropic"
     }
   }
   
   # For flow mode
   {
     "analysis_settings": {
       "static_mode": false
     }
   }

**Missing Required Parameters**

.. code-block:: text

   KeyError: Required parameter 'D0' not found in configuration

**Solution**: Verify all required parameters are specified:

.. code-block:: json

   {
     "initial_parameters": {
       "active_parameters": ["D0", "alpha", "D_offset"],
       "D0": 1e-12,
       "alpha": 1.0,
       "D_offset": 0.0
     }
   }

File Path Problems
~~~~~~~~~~~~~~~~~

**File Not Found Errors**

.. code-block:: text

   FileNotFoundError: [Errno 2] No such file or directory: 'data/correlation_data.h5'

**Solutions**:

1. **Check file paths**:

.. code-block:: bash

   # Verify file exists
   ls -la data/correlation_data.h5

2. **Use absolute paths** in configuration:

.. code-block:: json

   {
     "file_paths": {
       "c2_data_file": "/full/path/to/correlation_data.h5"
     }
   }

3. **Verify working directory**:

.. code-block:: python

   import os
   print(f"Current directory: {os.getcwd()}")

**Permission Issues**

.. code-block:: text

   PermissionError: [Errno 13] Permission denied

**Solutions**:

.. code-block:: bash

   # Check file permissions
   ls -la data/
   
   # Fix permissions if needed
   chmod 644 data/correlation_data.h5
   chmod 755 data/  # For directory

Analysis Problems
-----------------

Optimization Convergence Issues
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Poor Convergence in Classical Optimization**

.. code-block:: text

   Optimization terminated without convergence
   Maximum iterations reached

**Solutions**:

1. **Adjust optimization settings**:

.. code-block:: json

   {
     "optimization": {
       "max_iterations": 20000,
       "tolerance": 1e-10,
       "initial_simplex_size": 0.05
     }
   }

2. **Check initial parameter values**:

.. code-block:: json

   {
     "initial_parameters": {
       "D0": 1e-12,     # Should be order-of-magnitude correct
       "alpha": 1.0,    # Usually close to 1.0
       "D_offset": 0.0  # Often small or zero
     }
   }

3. **Verify parameter bounds**:

.. code-block:: json

   {
     "parameter_bounds": {
       "D0": [1e-15, 1e-9],      # Reasonable physical range
       "alpha": [0.1, 2.0],      # Positive, typically 0.5-2.0
       "D_offset": [0.0, 1e-11]  # Non-negative, usually small
     }
   }

**Local Minima Problems**

.. code-block:: text

   Optimization converged to unrealistic parameters

**Solutions**:

1. **Try multiple starting points**:

.. code-block:: python

   # Run optimization with different initial conditions
   for i in range(5):
       config.override_parameters({
           "D0": np.random.uniform(1e-13, 1e-11),
           "alpha": np.random.uniform(0.8, 1.2)
       })
       results = analysis.optimize_classical()

2. **Use combined classical + MCMC approach**:

.. code-block:: bash

   python run_homodyne.py --method all  # Classical followed by MCMC

MCMC Convergence Problems
~~~~~~~~~~~~~~~~~~~~~~~~

**Poor R-hat Values (> 1.1)**

.. code-block:: text

   Warning: R-hat values indicate poor convergence

**Solutions**:

1. **Increase tuning steps**:

.. code-block:: json

   {
     "mcmc": {
       "tune": 2000,  # Increased from 1000
       "n_samples": 2000
     }
   }

2. **Use better initial estimates**:

.. code-block:: bash

   # Run classical optimization first
   python run_homodyne.py --method classical
   # Then use results to initialize MCMC
   python run_homodyne.py --method mcmc

3. **Increase target acceptance rate**:

.. code-block:: json

   {
     "mcmc": {
       "target_accept": 0.95,  # Increased from 0.8
       "chains": 6             # More chains
     }
   }

**Low Effective Sample Size**

.. code-block:: text

   Warning: Effective sample size below 400

**Solutions**:

1. **Increase sample count**:

.. code-block:: javascript

   {
     "mcmc": {
       "n_samples": 4000,  // Increased sample size
       "tune": 2000
     }
   }

2. **Check for parameter correlations**:

.. code-block:: python

   import arviz as az
   
   # Examine parameter correlations
   az.plot_pair(mcmc_results.idata, var_names=active_parameters)

Mode-Specific Issues
~~~~~~~~~~~~~~~~~~~

**Static Isotropic Mode Issues**

.. code-block:: text

   Warning: Angle filtering enabled but static_isotropic mode detected

**Solution**: This is expected behavior. Angle filtering is automatically disabled in isotropic mode.

.. code-block:: text

   Error: phi_angles_file not found in isotropic mode

**Solution**: This is normal. phi_angles_file is not used in isotropic mode.

**Static Anisotropic Mode Issues**

.. code-block:: text

   Results similar to isotropic mode despite using anisotropic analysis

**Solution**: Your system may actually be isotropic. Compare chi-squared values:

.. code-block:: python

   # Compare analysis results
   iso_results = analyze_isotropic(config)
   aniso_results = analyze_anisotropic(config)
   
   print(f"Isotropic chi-squared: {iso_results.fun}")
   print(f"Anisotropic chi-squared: {aniso_results.fun}")
   
   # If values are similar, system is likely isotropic

**Laminar Flow Mode Issues**

.. code-block:: text

   Slow convergence or poor parameter estimates in 7-parameter mode

**Solutions**:

1. **Enable angle filtering**:

.. code-block:: json

   {
     "analysis_settings": {
       "enable_angle_filtering": true,
       "angle_filter_ranges": [[-10, 10], [170, 190]]
     }
   }

2. **Use static analysis for initial estimates**:

.. code-block:: bash

   # First run static analysis
   python run_homodyne.py --static-anisotropic --method classical
   # Use results to initialize flow analysis
   python run_homodyne.py --laminar-flow --method classical

3. **Check parameter bounds for flow parameters**:

.. code-block:: json

   {
     "parameter_bounds": {
       "gamma_dot_t0": [1e-6, 1e-1],
       "beta": [0.1, 2.0],
       "gamma_dot_t_offset": [0.0, 1e-2],
       "phi0": [-3.14159, 3.14159]
     }
   }

Data Quality Issues
-------------------

Poor Experimental Data
~~~~~~~~~~~~~~~~~~~~~~

**Low Signal Quality**

.. code-block:: text

   Warning: Low diagonal enhancement detected (< 0.001)

**Diagnostics**:

.. code-block:: bash

   # Generate data validation plots
   python run_homodyne.py --plot-experimental-data --verbose

**Solutions**:

1. **Check measurement conditions**:
   - Increase measurement time
   - Improve sample stability
   - Verify instrumental setup

2. **Adjust analysis parameters**:

.. code-block:: javascript

   {
     "analysis_settings": {
       "enable_angle_filtering": true,
       "angle_filter_ranges": [[-5, 5], [175, 185]]  // Narrower ranges
     }
   }

**Systematic Artifacts**

.. code-block:: text

   Regular patterns or unexpected correlations in data

**Diagnostics**:

1. **Visual inspection**:

.. code-block:: bash

   python run_homodyne.py --plot-experimental-data

2. **Check for instrumental issues**:
   - Detector artifacts
   - Beam instability
   - Sample environment problems

**Solutions**:

1. **Data preprocessing**:

.. code-block:: json

   {
     "data_preprocessing": {
       "apply_smoothing": true,
       "smoothing_window": 3,
       "outlier_removal": true,
       "outlier_threshold": 3.0
     }
   }

2. **Consider simpler analysis modes**:

.. code-block:: bash

   # Try isotropic mode for problematic data
   python run_homodyne.py --static-isotropic

Memory and Performance Issues
-----------------------------

Memory Exhaustion
~~~~~~~~~~~~~~~~~

**Out of Memory Errors**

.. code-block:: text

   MemoryError: Unable to allocate array

**Solutions**:

1. **Reduce memory usage**:

.. code-block:: javascript

   {
     "performance": {
       "data_type": "float32",    // 50% memory reduction
       "memory_limit_gb": 8,      // Set appropriate limit
       "chunked_processing": true // Process in chunks
     }
   }

2. **Enable angle filtering**:

.. code-block:: json

   {
     "analysis_settings": {
       "enable_angle_filtering": true,
       "angle_filter_ranges": [[-10, 10], [170, 190]]
     }
   }

3. **Monitor memory usage**:

.. code-block:: python

   import psutil
   import os
   
   process = psutil.Process(os.getpid())
   print(f"Memory usage: {process.memory_info().rss / 1024 / 1024:.1f} MB")

Performance Problems
~~~~~~~~~~~~~~~~~~~

**Slow Execution**

.. code-block:: text

   Analysis taking much longer than expected

**Solutions**:

1. **Verify Numba is working**:

.. code-block:: python

   import numba
   print(f"Numba version: {numba.__version__}")
   
   # Set environment variable if needed
   import os
   os.environ['NUMBA_DISABLE_INTEL_SVML'] = '1'

2. **Check threading configuration**:

.. code-block:: bash

   export OMP_NUM_THREADS=8
   export OPENBLAS_NUM_THREADS=8

3. **Use appropriate analysis mode**:

.. code-block:: bash

   # Start with fastest mode
   python run_homodyne.py --static-isotropic

**First Run Slowness**

.. code-block:: text

   First analysis much slower than subsequent runs

**Explanation**: This is expected due to Numba JIT compilation. Subsequent runs will be much faster.

**Solution**: Allow extra time for first run, or pre-compile:

.. code-block:: python

   # Pre-compile kernels (if needed)
   from homodyne.core.kernels import precompile_kernels
   precompile_kernels()

Environment Issues
------------------

Python Environment Problems
~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Version Compatibility**

.. code-block:: text

   Python version compatibility issues

**Solution**:

.. code-block:: bash

   # Check Python version (3.8+ required)
   python --version
   
   # Create compatible environment
   python -m venv homodyne_env
   source homodyne_env/bin/activate

**Package Conflicts**

.. code-block:: text

   Package version conflicts

**Solution**:

.. code-block:: bash

   # Clean installation
   pip uninstall numpy scipy matplotlib numba pymc arviz pytensor
   pip install numpy scipy matplotlib numba
   pip install pymc arviz pytensor

Operating System Issues
~~~~~~~~~~~~~~~~~~~~~~

**Windows-Specific Issues**

.. code-block:: text

   Issues with file paths or environment variables

**Solutions**:

1. **Use forward slashes** in file paths:

.. code-block:: javascript

   {
     "file_paths": {
       "c2_data_file": "C:/data/correlation_data.h5"  // Not C:\\data\\...
     }
   }

2. **Set environment variables**:

.. code-block:: batch

   set OMP_NUM_THREADS=8
   set NUMBA_DISABLE_INTEL_SVML=1

**macOS-Specific Issues**

.. code-block:: text

   Issues with BLAS/LAPACK libraries

**Solutions**:

.. code-block:: bash

   # Install optimized BLAS
   pip install numpy[blas]
   
   # Or use conda for better BLAS support
   conda install numpy scipy matplotlib -c conda-forge

Diagnostic Tools
---------------

Built-in Diagnostics
~~~~~~~~~~~~~~~~~~~~

**Configuration Validation**:

.. code-block:: python

   from homodyne.core import ConfigManager
   
   try:
       config = ConfigManager("my_config.json")
       print("Configuration loaded successfully")
   except Exception as e:
       print(f"Configuration error: {e}")

**Data Validation**:

.. code-block:: bash

   # Comprehensive data validation
   python run_homodyne.py --plot-experimental-data --verbose

**Performance Benchmarking**:

.. code-block:: bash

   # Quick performance check
   python benchmark_performance.py --fast

External Diagnostic Tools
~~~~~~~~~~~~~~~~~~~~~~~~~

**Memory Monitoring**:

.. code-block:: bash

   # Install memory profiler
   pip install memory-profiler
   
   # Monitor memory usage
   mprof run python run_homodyne.py --config my_config.json
   mprof plot

**CPU Profiling**:

.. code-block:: python

   import cProfile
   
   profiler = cProfile.Profile()
   profiler.enable()
   
   # Run analysis
   results = analysis.optimize_classical()
   
   profiler.disable()
   profiler.print_stats(sort='tottime')

Getting Help
-----------

Information Collection
~~~~~~~~~~~~~~~~~~~~~

When reporting issues, please provide:

1. **System Information**:

.. code-block:: python

   import sys
   import numpy as np
   import scipy
   import matplotlib
   
   print(f"Python version: {sys.version}")
   print(f"NumPy version: {np.__version__}")
   print(f"SciPy version: {scipy.__version__}")
   print(f"Matplotlib version: {matplotlib.__version__}")
   
   try:
       import numba
       print(f"Numba version: {numba.__version__}")
   except ImportError:
       print("Numba not installed")

2. **Configuration File**: Include your configuration (with sensitive data removed)

3. **Error Messages**: Complete error traceback

4. **Expected vs. Actual Behavior**: Clear description of the issue

Support Resources
~~~~~~~~~~~~~~~~~

- **Documentation**: This comprehensive documentation
- **Issues**: GitHub Issues for bug reports and questions
- **Examples**: Reference configurations and workflows in the package

Best Practices for Avoiding Issues
----------------------------------

Preventive Measures
~~~~~~~~~~~~~~~~~~~

**Regular Validation**:
- Always validate experimental data first
- Use appropriate analysis modes for your system
- Monitor computational resources

**Configuration Management**:
- Use version control for configuration files
- Validate JSON syntax before running analysis
- Keep backup configurations

**Environment Management**:
- Use virtual environments for isolation
- Document exact package versions
- Test installations with benchmark data

**Performance Monitoring**:
- Benchmark performance regularly
- Monitor memory usage trends
- Profile computational bottlenecks

This comprehensive troubleshooting guide should resolve most issues encountered while using the homodyne package. For persistent problems, please consult the support resources or file an issue with detailed diagnostic information.
