Installation
============

System Requirements
-------------------

* Python 3.8 or higher
* Operating System: Linux, macOS, or Windows

Core Dependencies
-----------------

The homodyne package requires the following core dependencies:

.. code-block:: bash

   pip install numpy scipy matplotlib

Performance Enhancement (Recommended)
--------------------------------------

For significant performance improvements (3-5x speedup via JIT compilation):

.. code-block:: bash

   pip install numba

Bayesian Analysis (Optional)
-----------------------------

For MCMC sampling and uncertainty quantification:

.. code-block:: bash

   pip install pymc arviz pytensor

Quick Start Installation
------------------------

Set up a Python virtual environment and install all dependencies:

.. code-block:: bash

   # Create and activate virtual environment
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate

   # Install core dependencies
   pip install numpy scipy matplotlib

   # Install performance enhancement (recommended)
   pip install numba

   # Install Bayesian analysis capabilities (optional)
   pip install pymc arviz pytensor

Specialized Data Loading (Optional)
------------------------------------

For specialized XPCS data loading capabilities:

.. code-block:: bash

   pip install pyxpcsviewer

Environment Configuration
--------------------------

For optimal performance, set the following environment variables:

.. code-block:: bash

   # Optimize BLAS/threading for performance
   export OMP_NUM_THREADS=8
   export OPENBLAS_NUM_THREADS=8
   export MKL_NUM_THREADS=8

   # Disable Intel SVML for Numba compatibility
   export NUMBA_DISABLE_INTEL_SVML=1

Verification
------------

Verify your installation by running the test suite:

.. code-block:: bash

   # Basic test run
   python homodyne/run_tests.py

   # Quick test (exclude slow integration tests)
   python homodyne/run_tests.py --fast

   # Run with coverage reporting
   python homodyne/run_tests.py --coverage

Performance Validation
-----------------------

Test the performance optimizations:

.. code-block:: bash

   # Comprehensive performance benchmark
   python benchmark_performance.py --iterations 50 --size 1000

   # Quick performance validation
   python benchmark_performance.py --fast

Troubleshooting Installation Issues
-----------------------------------

**Missing Dependencies**

If you encounter import errors, ensure all required packages are installed:

.. code-block:: bash

   # For classical optimization
   pip install scipy numpy matplotlib

   # For MCMC analysis
   pip install pymc arviz pytensor

   # For performance acceleration
   pip install numba

**Numba Compilation Issues**

If Numba fails to compile, try setting:

.. code-block:: bash

   export NUMBA_DISABLE_INTEL_SVML=1

**Memory Issues**

For large datasets, ensure sufficient RAM is available and consider:

- Reducing array sizes in configuration
- Using ``float32`` instead of ``float64`` for data type
- Adjusting ``memory_limit_gb`` setting in configuration
