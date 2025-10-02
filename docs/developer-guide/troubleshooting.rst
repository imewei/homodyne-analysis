Troubleshooting Guide
=====================

This guide helps diagnose and resolve common issues encountered when using or developing the homodyne package.

Installation Issues
-------------------

**Import Errors**

*Problem*: ``ModuleNotFoundError`` when importing homodyne

.. code-block:: python

   >>> import homodyne
   ModuleNotFoundError: No module named 'homodyne'

*Solution*:

.. code-block:: bash

   # Check installation
   pip list | grep homodyne

   # Reinstall if missing
   pip install -e .

   # For development install
   pip install -e .[dev]

**Dependency Conflicts**

*Problem*: Version conflicts with NumPy, SciPy, or other dependencies

*Solution*:

.. code-block:: bash

   # Create fresh environment
   conda create -n homodyne-env python=3.12
   conda activate homodyne-env

   # Install dependencies step by step
   pip install numpy scipy matplotlib
   pip install numba
   pip install -e .



Configuration Issues
--------------------

**Invalid JSON Syntax**

*Problem*: JSON parsing errors in configuration files

.. code-block:: text

   json.decoder.JSONDecodeError: Expecting ',' delimiter

*Solution*:

.. code-block:: bash

   # Validate JSON syntax
   python -m json.tool my_config.json

   # Check for common issues:
   # - Missing commas
   # - Trailing commas
   # - Unquoted strings
   # - Comments (not allowed in JSON)

**Missing Required Fields**

*Problem*: Configuration validation errors

.. code-block:: text

   ConfigurationError: Required field 'analysis_settings' not found

*Solution*:

.. code-block:: python

   # Use configuration validation
   from homodyne import ConfigManager
   from homodyne.utils import ConfigurationError

   try:
       config = ConfigManager("my_config.json")
       config.validate()
   except ConfigurationError as e:
       print(f"Configuration error: {e}")
       # Fix the configuration file based on error message

**File Path Issues**

*Problem*: File not found errors

.. code-block:: text

   FileNotFoundError: [Errno 2] No such file or directory: 'data/my_data.h5'

*Solution*:

.. code-block:: python

   import os

   # Check if file exists
   data_file = "data/my_data.h5"
   if not os.path.exists(data_file):
       print(f"File not found: {data_file}")
       print(f"Current directory: {os.getcwd()}")
       print(f"Available files: {os.listdir('.')}")

   # Use absolute paths when possible
   data_file = os.path.abspath("data/my_data.h5")

Data Loading Issues
-------------------

**HDF5 Format Problems**

*Problem*: HDF5 file loading errors

.. code-block:: text

   OSError: Unable to open file (file signature not found)

*Solution*:

.. code-block:: python

   import h5py

   # Check file integrity
   try:
       with h5py.File("data.h5", 'r') as f:
           print("Available datasets:", list(f.keys()))
   except OSError as e:
       print(f"HDF5 error: {e}")
       # File may be corrupted or not HDF5 format

**Data Shape Mismatches**

*Problem*: Unexpected data dimensions

.. code-block:: text

   ValueError: Expected 2D array, got 1D array

*Solution*:

.. code-block:: python

   import numpy as np

   # Check data shape
   data = np.load("my_data.npz")
   print("Data shape:", data['correlation_data'].shape)
   print("Expected shape: (n_time_points, n_angles)")

   # Reshape if needed
   if data.ndim == 1:
       data = data.reshape(-1, 1)  # Single angle

**Missing Dataset Keys**

*Problem*: Required datasets not found in file

.. code-block:: text

   KeyError: 'tau' not found in data file

*Solution*:

.. code-block:: python

   # Check available keys
   with np.load("data.npz") as data:
       print("Available keys:", list(data.keys()))
       # Expected keys: 'tau', 'g1', 'q', 'phi_angles'

Optimization Issues
-------------------

**Convergence Failures**

*Problem*: Optimization doesn't converge

.. code-block:: text

   OptimizationWarning: Optimization terminated unsuccessfully

*Diagnosis*:

.. code-block:: python

   # Check optimization result details
   from homodyne.optimization.classical import ClassicalOptimizer
   optimizer = ClassicalOptimizer(core, config)
   params, result = optimizer.run_classical_optimization_optimized(
       phi_angles=phi_angles, c2_experimental=c2_data)
   print(f"Success: {result.success}")
   print(f"Message: {result.message}")
   print(f"Function evaluations: {result.nfev}")
   print(f"Final chi-squared: {result.chi_squared}")

*Solutions*:

1. **Better initial parameters**:

.. code-block:: python

   # Try different starting points
   config["initial_parameters"]["values"] = [800, -0.3, 150]

2. **Check optimization method options**:

.. code-block:: python

   # Nelder-Mead is the only supported classical optimization method
   config["optimization_config"]["classical_optimization"]["methods"] = ["Nelder-Mead"]
   config["optimization_config"]["classical_optimization"]["method_options"]["Nelder-Mead"]["maxiter"] = 1000

3. **Looser tolerances**:

.. code-block:: python

   config["optimization_config"]["classical"]["tolerance"] = 1e-4

**Poor Fit Quality**

*Problem*: High chi-squared values indicating poor fits

*Diagnosis*:

.. code-block:: python

   # Plot fit to visualize issues
   from homodyne.utils import plot_fit_results

   fig = plot_fit_results(
       experimental_data,
       fitted_data,
       parameters=result.x,
       chi_squared=result.fun
   )
   fig.show()

*Solutions*:

1. **Check data quality**: Ensure experimental data is clean
2. **Verify model choice**: Try different analysis modes
3. **Parameter bounds**: Ensure bounds are reasonable
4. **Data preprocessing**: Apply filtering or smoothing if appropriate

**Parameter Bounds Violations**

*Problem*: Parameters hitting optimization bounds

.. code-block:: text

   Warning: Parameter D0 at upper bound (10000)

*Solution*:

.. code-block:: python

   # Adjust parameter bounds
   config["parameter_space"]["bounds"] = [
       {"name": "D0", "min": 100, "max": 50000},  # Increased upper bound
       {"name": "alpha", "min": -2.0, "max": 0.0},
       {"name": "D_offset", "min": 0, "max": 2000}
   ]

-----------

**Convergence Diagnostics**


.. code-block:: text

   Warning: R-hat values > 1.1 detected

*Diagnosis*:

.. code-block:: python

   # Check convergence diagnostics

       if rhat > 1.1:
           print(f"⚠️ {param}: R̂ = {rhat:.3f} (poor convergence)")
       else:
           print(f"✅ {param}: R̂ = {rhat:.3f} (good convergence)")

*Solutions*:

1. **Increase tuning steps**:

.. code-block:: python


2. **More chains**:

.. code-block:: python


3. **Better initialization**:

.. code-block:: python

   # Run classical optimization first to get good starting point
   from homodyne.optimization.classical import ClassicalOptimizer
   optimizer = ClassicalOptimizer(core, config)
   params, classical_result = optimizer.run_classical_optimization_optimized(
       phi_angles=phi_angles, c2_experimental=c2_data)

**Divergences**


.. code-block:: text

   Warning: 150 divergences encountered

*Solutions*:

1. **Increase target acceptance**:

.. code-block:: python


2. **Increase max tree depth**:

.. code-block:: python


3. **Better parameterization**: Check if model is well-conditioned



*Solutions*:

1. **Reduce sample size**:

.. code-block:: python


2. **Fewer chains**:

.. code-block:: python


3. **Enable thinning**:

.. code-block:: python


Performance Issues
------------------

**Slow Analysis**

*Problem*: Analysis taking too long

*Solutions*:

1. **Enable angle filtering**:

.. code-block:: python

   config["analysis_settings"]["enable_angle_filtering"] = True
   config["analysis_settings"]["angle_filter_ranges"] = [[-5, 5], [175, 185]]

2. **Use float32**:

.. code-block:: python

   config["performance_settings"]["data_type"] = "float32"

3. **Optimize thread usage**:

.. code-block:: python

   config["performance_settings"]["num_threads"] = 4  # Match CPU cores

4. **Enable JIT compilation**:

.. code-block:: python

   config["performance_settings"]["enable_jit"] = True

**Memory Usage**

*Problem*: Excessive memory consumption

*Diagnosis*:

.. code-block:: python

   import psutil

   process = psutil.Process()
   memory_mb = process.memory_info().rss / 1024**2
   print(f"Current memory usage: {memory_mb:.1f} MB")

*Solutions*:

1. **Use chunked processing**:

.. code-block:: python

   config["performance_settings"]["chunked_processing"] = True
   config["performance_settings"]["chunk_size"] = 1000

2. **Reduce precision**:

.. code-block:: python

   config["performance_settings"]["data_type"] = "float32"

3. **Set memory limit**:

.. code-block:: python

   config["performance_settings"]["memory_limit_gb"] = 8

Model-Specific Issues
---------------------

**Isotropic Mode Issues**

*Problem*: Warnings about angle filtering in isotropic mode

.. code-block:: text

   Warning: Angle filtering enabled but static_isotropic mode detected

*Solution*: This is expected behavior - angle filtering is automatically disabled in isotropic mode.

**Flow Mode Parameter Issues**

*Problem*: Flow parameters giving unrealistic values

*Solutions*:

1. **Check if flow is actually present** in your system
2. **Start with static anisotropic** to get baseline parameters
3. **Use realistic bounds** for flow parameters:

.. code-block:: python

   flow_bounds = [
       {"name": "gamma_dot_t0", "min": 0.1, "max": 100},
       {"name": "beta", "min": -1.0, "max": 1.0},
       {"name": "gamma_dot_t_offset", "min": 0, "max": 10}
   ]

Development Issues
------------------

**Test Failures**

*Problem*: Tests failing during development

*Diagnosis*:

.. code-block:: bash

   # Run specific test with verbose output
   pytest homodyne/tests/test_specific.py::test_function -v -s

   # Run with debugging
   pytest homodyne/tests/test_specific.py --pdb

*Common Solutions*:

1. **Update test data** if model changes
2. **Check numerical tolerances** in assertions
3. **Verify fixtures** are properly set up
4. **Update dependencies** if needed

**Documentation Build Issues**

*Problem*: Sphinx documentation build failures

.. code-block:: bash

   # Build with verbose output
   cd docs/
   make clean
   make html SPHINXOPTS="-v"

*Common fixes*:

1. **Install doc dependencies**: ``pip install -e .[docs]``
2. **Check RST syntax** in documentation files
3. **Verify import paths** in API documentation
4. **Update Sphinx configuration** if needed

Getting Help
------------

**Information to Provide**

When seeking help, include:

1. **Version information**:

.. code-block:: python

   import homodyne
   import numpy
   import scipy
   print(f"Homodyne: {homodyne.__version__}")
   print(f"NumPy: {numpy.__version__}")
   print(f"SciPy: {scipy.__version__}")

2. **System information**:

.. code-block:: python

   import platform
   print(f"Python: {platform.python_version()}")
   print(f"System: {platform.system()} {platform.release()}")

3. **Error messages**: Full traceback
4. **Configuration file**: Minimal example that reproduces issue
5. **Data characteristics**: Size, format, analysis mode

**Resources**

- **GitHub Issues**: https://github.com/imewei/homodyne/issues
- **Documentation**: This documentation site
- **Examples**: Check the examples directory in the repository
