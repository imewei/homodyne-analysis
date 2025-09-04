Troubleshooting Flowchart
=========================

.. index:: troubleshooting, errors, diagnosis, GPU issues, MCMC problems, installation problems

Follow this step-by-step flowchart to diagnose and fix common issues.

.. contents:: Troubleshooting Categories
   :local:
   :depth: 2

🔍 Start Here: Quick Diagnosis
-------------------------------

**First, run the system health check:**

.. code-block:: bash

   homodyne-validate --quick

**Results interpretation:**

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Result
     - Next Action
   * - ✅ **All tests passed**
     - Your issue is likely configuration or data-related → Skip to `📊 Analysis Issues`_
   * - ⚠️ **Some tests failed**
     - Follow the specific error messages → Continue with flowchart below
   * - ❌ **Command not found**
     - Installation problem → Go to `📦 Installation Issues`_

📦 Installation Issues
----------------------

Command not found: ``homodyne``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Diagnosis:**

.. code-block:: bash

   which python
   pip list | grep homodyne

**Solutions:**

1. **Package not installed:**

   .. code-block:: bash

      pip install homodyne-analysis[mcmc]  # or [mcmc-gpu] or [all]

2. **Wrong environment:**

   .. code-block:: bash

      conda activate your-homodyne-env    # or appropriate environment
      pip install homodyne-analysis[mcmc]

3. **PATH issues:**

   .. code-block:: bash

      python -m pip install --user homodyne-analysis[mcmc]
      # Then restart terminal

Import errors in Python
~~~~~~~~~~~~~~~~~~~~~~~~

**Diagnosis:**

.. code-block:: python

   try:
       import homodyne
       print(f"✅ Homodyne version: {homodyne.__version__}")
   except ImportError as e:
       print(f"❌ Import error: {e}")

**Solutions:**

1. **Package corrupted:**

   .. code-block:: bash

      pip uninstall homodyne-analysis
      pip install homodyne-analysis[mcmc]

2. **Dependency conflicts:**

   .. code-block:: bash

      pip install homodyne-analysis[mcmc] --force-reinstall

3. **Environment issues:**

   .. code-block:: bash

      conda create -n homodyne python=3.12
      conda activate homodyne
      pip install homodyne-analysis[mcmc]

🖥️ Backend Issues
------------------

MCMC backend not available
~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Error message:** ``"MCMC module not available"`` or ``"Backend not found"``

**Diagnosis flowchart:**

.. code-block:: bash

   # Step 1: Check CPU backend
   python -c "from homodyne.optimization.mcmc_cpu_backend import is_cpu_mcmc_available; print(f'CPU: {is_cpu_mcmc_available()}')"
   
   # Step 2: Check GPU backend  
   python -c "from homodyne.optimization.mcmc_gpu_backend import is_gpu_mcmc_available; print(f'GPU: {is_gpu_mcmc_available()}')"

**Results and solutions:**

.. list-table::
   :widths: 25 25 50
   :header-rows: 1

   * - CPU Backend
     - GPU Backend
     - Action
   * - ✅ True
     - ✅ True
     - Backends available → Check `Environment Variables`_
   * - ✅ True
     - ❌ False
     - Install GPU: ``pip install homodyne-analysis[mcmc-gpu]``
   * - ❌ False
     - ❌ False
     - Install CPU: ``pip install homodyne-analysis[mcmc]``
   * - ❌ False
     - ✅ True
     - Unusual → Reinstall: ``pip install homodyne-analysis[mcmc-all]``

Environment Variables
~~~~~~~~~~~~~~~~~~~~~

**Check current backend selection:**

.. code-block:: bash

   python -c "
   import os
   from homodyne.run_homodyne import get_mcmc_backend
   
   gpu_intent = os.environ.get('HOMODYNE_GPU_INTENT', 'not set')
   print(f'GPU Intent: {gpu_intent}')
   
   try:
       func, backend, has_gpu = get_mcmc_backend()
       print(f'Selected Backend: {backend}')
       print(f'Has GPU Hardware: {has_gpu}')
   except Exception as e:
       print(f'Backend Error: {e}')
   "

**Force specific backend:**

.. code-block:: bash

   # Force CPU backend (pure PyMC)
   export HOMODYNE_GPU_INTENT=false
   homodyne --method mcmc
   
   # Force GPU backend (pure NumPyro/JAX)  
   export HOMODYNE_GPU_INTENT=true
   homodyne --method mcmc
   
   # Or use dedicated command
   homodyne-gpu --method mcmc

GPU Detection Issues
~~~~~~~~~~~~~~~~~~~~

**Error:** ``"GPU not detected"`` or ``"JAX not using GPU"``

**Diagnosis sequence:**

.. code-block:: bash

   # Step 1: Basic GPU check
   nvidia-smi
   
   # Step 2: CUDA version
   nvcc --version
   
   # Step 3: JAX GPU detection
   python -c "import jax; print(f'JAX devices: {jax.devices()}')"
   
   # Step 4: Homodyne GPU tools
   gpu-status
   homodyne-gpu-optimize --report

**Common solutions:**

1. **NVIDIA drivers missing:**

   .. code-block:: bash

      # Ubuntu/Debian
      sudo apt update && sudo apt install nvidia-driver-545
      
      # Check installation
      nvidia-smi

2. **CUDA not found:**

   .. code-block:: bash

      # Install CUDA 12.6+
      # See NVIDIA CUDA installation guide
      
      # Verify
      nvcc --version

3. **JAX not using GPU:**

   .. code-block:: bash

      pip uninstall jax jaxlib
      pip install "jax[cuda12]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

4. **Windows/macOS (no GPU support):**

   .. code-block:: bash

      # Use CPU backend instead
      export HOMODYNE_GPU_INTENT=false
      homodyne --method mcmc

📁 Configuration Issues
------------------------

Configuration file errors
~~~~~~~~~~~~~~~~~~~~~~~~~~

**Error patterns and solutions:**

.. list-table::
   :widths: 40 60
   :header-rows: 1

   * - Error Message
     - Solution
   * - ``"FileNotFoundError: config.json"``
     - Generate config: ``homodyne-config --mode static_isotropic --sample test``
   * - ``"JSON decode error"``
     - Check JSON syntax: Use online JSON validator
   * - ``"Invalid configuration"``
     - Regenerate: ``homodyne-config --mode <your_mode> --sample <name>``
   * - ``"Missing required field"``
     - Compare with template in :doc:`configuration-guide`

File path issues
~~~~~~~~~~~~~~~~

**Error:** ``"FileNotFoundError"`` for data files

**Diagnosis:**

.. code-block:: bash

   # Check if files exist
   ls -la /path/to/your/data.h5
   ls -la /path/to/your/angles.txt
   
   # Check config file paths
   grep -E "file|path" your_config.json

**Solutions:**

1. **Relative vs absolute paths:**

   .. code-block:: json

      {
        "file_paths": {
          "c2_data_file": "./data/correlation_data.h5",     // Relative to config
          "phi_angles_file": "/absolute/path/to/angles.txt" // Absolute path
        }
      }

2. **Working directory:**

   .. code-block:: bash

      # Run from config directory
      cd /path/to/config/directory
      homodyne --config my_config.json

3. **File permissions:**

   .. code-block:: bash

      chmod 644 /path/to/data/files/*

📊 Analysis Issues
------------------

Optimization fails to converge
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Error patterns:**

- ``"Optimization failed"``
- ``"Maximum iterations reached"``
- ``"NaN values in results"``

**Diagnostic sequence:**

1. **Check initial parameters:**

   .. code-block:: bash

      # Try different starting values
      homodyne-config --mode static_isotropic --sample test_init
      # Edit initial_parameters.values in config

2. **Simplify analysis mode:**

   .. code-block:: json

      {
        "analysis_settings": {
          "static_mode": true,
          "static_submode": "isotropic"  // Start with simplest mode
        }
      }

3. **Check data quality:**

   .. code-block:: bash

      # Plot experimental data
      homodyne --config config.json --plot-experimental-data

4. **Enable angle filtering:**

   .. code-block:: json

      {
        "analysis_settings": {
          "enable_angle_filtering": true,
          "angle_filter_ranges": [[-5, 5], [175, 185]]
        }
      }

MCMC convergence issues
~~~~~~~~~~~~~~~~~~~~~~~

**Error:** ``"MCMC chains did not converge"`` or ``"R-hat > 1.1"``

**Solutions by severity:**

.. list-table::
   :widths: 20 30 50
   :header-rows: 1

   * - R-hat Value
     - Severity
     - Action
   * - 1.01 - 1.05
     - ✅ Excellent
     - Continue with analysis
   * - 1.05 - 1.1
     - ⚠️ Good
     - Acceptable, consider more samples
   * - 1.1 - 1.3
     - ❌ Poor
     - Increase samples or tune parameters
   * - > 1.3
     - 🔴 Failed
     - Use classical initialization

**Step-by-step fixes:**

1. **Use classical initialization:**

   .. code-block:: bash

      # Run classical first
      homodyne --method classical --config config.json
      
      # Then MCMC (uses classical results as starting point)
      homodyne --method mcmc --config config.json

2. **Increase MCMC parameters:**

   .. code-block:: json

      {
        "optimization_config": {
          "mcmc_sampling": {
            "draws": 4000,    // Increase from 2000
            "tune": 2000,     // Increase from 1000  
            "chains": 6,      // Increase from 4
            "target_accept": 0.95
          }
        }
      }

3. **Check parameter bounds:**

   .. code-block:: json

      {
        "parameter_space": {
          "bounds": [
            {"name": "D0", "min": 100, "max": 50000},      // Wider bounds
            {"name": "alpha", "min": -3.0, "max": 1.0}     // More flexibility
          ]
        }
      }

Slow performance
~~~~~~~~~~~~~~~~

**Diagnosis:**

.. code-block:: bash

   # Check system resources during analysis
   htop                    # CPU usage
   nvidia-smi dmon         # GPU usage (if applicable)
   
   # Time your analysis
   time homodyne --method classical --config config.json

**Optimization strategies:**

1. **Enable angle filtering:**

   .. code-block:: json

      {
        "analysis_settings": {
          "enable_angle_filtering": true,
          "angle_filter_ranges": [[-5, 5], [175, 185]]
        }
      }

2. **Reduce data precision:**

   .. code-block:: json

      {
        "performance_settings": {
          "data_type": "float32"
        }
      }

3. **Optimize threading:**

   .. code-block:: bash

      export OMP_NUM_THREADS=4
      export OPENBLAS_NUM_THREADS=4

4. **Use GPU backend (Linux):**

   .. code-block:: bash

      homodyne-gpu --method mcmc --config config.json

⚡ Performance Issues
---------------------

GPU not being utilized
~~~~~~~~~~~~~~~~~~~~~~

**Check GPU usage:**

.. code-block:: bash

   # During analysis, run in another terminal:
   watch -n 1 nvidia-smi

**If GPU usage is 0%:**

1. **Check backend selection:**

   .. code-block:: bash

      # Ensure GPU backend is selected
      export HOMODYNE_GPU_INTENT=true
      homodyne --method mcmc --config config.json

2. **Use dedicated GPU command:**

   .. code-block:: bash

      homodyne-gpu --method mcmc --config config.json

3. **Check JAX GPU detection:**

   .. code-block:: python

      import jax
      print(f"JAX devices: {jax.devices()}")
      print(f"Default device: {jax.devices()[0]}")

Memory issues
~~~~~~~~~~~~~

**Error:** ``"Out of memory"`` or ``"CUDA out of memory"``

**Solutions:**

1. **Reduce batch size:**

   .. code-block:: json

      {
        "performance_settings": {
          "batch_size": 1000
        }
      }

2. **Limit GPU memory:**

   .. code-block:: bash

      export XLA_PYTHON_CLIENT_MEM_FRACTION=0.7
      homodyne-gpu --method mcmc

3. **Use CPU backend:**

   .. code-block:: bash

      export HOMODYNE_GPU_INTENT=false
      homodyne --method mcmc

4. **Enable angle filtering:**

   Reduces memory by analyzing fewer angles.

🔧 Advanced Diagnostics
------------------------

Complete system report
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Generate comprehensive report
   homodyne-validate --verbose > system_report.txt
   
   # Include GPU diagnostics
   gpu-status >> system_report.txt
   
   # Include environment info
   python -c "
   import sys, os, platform
   print(f'Python: {sys.version}')
   print(f'Platform: {platform.platform()}')
   print(f'Environment variables:')
   for k, v in os.environ.items():
       if 'HOMODYNE' in k or 'JAX' in k or 'OMP' in k:
           print(f'  {k}={v}')
   " >> system_report.txt

Debug logging
~~~~~~~~~~~~~

.. code-block:: bash

   # Enable maximum verbosity
   homodyne --config config.json --method mcmc --verbose --log-level DEBUG

Create minimal test case
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Generate minimal config
   homodyne-config --mode static_isotropic --sample minimal_test
   
   # Create synthetic data for testing
   python -c "
   import numpy as np
   
   # Minimal test data
   angles = np.array([0.0, 90.0, 180.0])
   tau = np.logspace(-6, 1, 50)
   delays = np.logspace(-3, 2, 30)
   
   # Synthetic correlation data
   c2_data = np.random.rand(len(angles), len(tau), len(delays)) * 0.5 + 1.0
   
   # Save data
   np.savez('minimal_test_data.npz', 
            c2_data=c2_data, tau_values=tau, delay_values=delays)
   np.savetxt('minimal_angles.txt', angles)
   
   print('✅ Minimal test data created')
   "
   
   # Test with minimal data
   homodyne --config minimal_test_config.json --method classical

🆘 Getting Help
----------------

When to seek help
~~~~~~~~~~~~~~~~~

**Seek help if:**

- System diagnostics show persistent failures
- You've tried solutions for your specific error
- Analysis produces unreasonable physical results
- Performance is significantly degraded

**Before asking for help, gather:**

1. **System report:** ``homodyne-validate --verbose``
2. **Error messages:** Copy exact error text
3. **Configuration file:** Your JSON config
4. **Command used:** Exact command that failed
5. **Environment:** OS, Python version, installation method

Where to get help
~~~~~~~~~~~~~~~~~

1. **Documentation:**
   
   - :doc:`../developer-guide/troubleshooting` - Detailed technical issues
   - :doc:`../api-reference/index` - API documentation
   - :doc:`configuration-guide` - Configuration reference

2. **GitHub Issues:**
   
   - `Report bugs <https://github.com/imewei/homodyne/issues/new?template=bug_report.md>`_
   - `Request features <https://github.com/imewei/homodyne/issues/new?template=feature_request.md>`_
   - `Search existing issues <https://github.com/imewei/homodyne/issues>`_

3. **Community:**
   
   - Check existing discussions
   - Share your use cases
   - Contribute improvements

.. tip::
   **Pro tip:** Most issues can be resolved by regenerating configuration files and checking file paths. When in doubt, start with ``homodyne-validate --quick``.