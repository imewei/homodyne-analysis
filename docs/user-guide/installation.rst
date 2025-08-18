Installation Guide
==================

System Requirements
-------------------

- **Python**: 3.12 or higher
- **Operating System**: Windows, macOS, or Linux
- **Memory**: Minimum 4GB RAM (8GB+ recommended for MCMC)
- **Storage**: ~500MB for full installation with dependencies

Quick Installation (Recommended)
--------------------------------

The easiest way to install the Homodyne Analysis package is from PyPI using pip:

**Basic Installation**

.. code-block:: bash

   pip install homodyne-analysis

This installs the core dependencies (numpy, scipy, matplotlib) along with the main package.

**Full Installation with All Features**

.. code-block:: bash

   pip install homodyne-analysis[all]

This includes all optional dependencies: performance acceleration (numba), MCMC analysis (pymc, arviz, pytensor), documentation tools, and development utilities.

Optional Installation Extras
-----------------------------

You can install specific feature sets using pip extras:

**For Enhanced Performance (Numba JIT acceleration):**

.. code-block:: bash

   pip install homodyne-analysis[performance]

**For MCMC Bayesian Analysis:**

.. code-block:: bash

   pip install homodyne-analysis[mcmc]

**For XPCS Data Handling:**

.. code-block:: bash

   pip install homodyne-analysis[data]

**For Documentation Building:**

.. code-block:: bash

   pip install homodyne-analysis[docs]

**For Development:**

.. code-block:: bash

   pip install homodyne-analysis[dev]

**All Dependencies:**

.. code-block:: bash

   pip install homodyne-analysis[all]

Development Installation
------------------------

For development, contributing, or accessing the latest unreleased features:

**Step 1: Clone the Repository**

.. code-block:: bash

   git clone https://github.com/imewei/homodyne.git
   cd homodyne

**Step 2: Install in Development Mode**

.. code-block:: bash

   # Install with all development dependencies
   pip install -e .[all]
   
   # Or install minimal development setup
   pip install -e .[dev]

Verification
------------

Test your installation:

.. code-block:: python

   import homodyne
   print(f"Homodyne version: {homodyne.__version__}")
   
   # Test basic functionality
   from homodyne import ConfigManager
   config = ConfigManager()
   print("✅ Installation successful!")

Common Issues
-------------

**Import Errors:**

If you encounter import errors, try reinstalling the package:

.. code-block:: bash

   pip install --upgrade homodyne-analysis
   
   # Or with all dependencies
   pip install --upgrade homodyne-analysis[all]

**MCMC Issues:**

For MCMC functionality, ensure the mcmc extras are installed:

.. code-block:: bash

   pip install homodyne-analysis[mcmc]
   
   # Test MCMC availability
   python -c "import pymc; print('PyMC available')"

**Performance Issues:**

For optimal performance, install the performance extras:

.. code-block:: bash

   pip install homodyne-analysis[performance]
   python -c "import numba; print(f'Numba version: {numba.__version__}')"

**Package Not Found:**

If pip cannot find the package, ensure you're using the correct name:

.. code-block:: bash

   pip install homodyne-analysis  # Correct package name
   # NOT: pip install homodyne    # This won't work

Getting Help
------------

If you encounter installation issues:

1. Check the `troubleshooting guide <../developer-guide/troubleshooting.html>`_
2. Search existing `GitHub issues <https://github.com/imewei/homodyne/issues>`_
3. Create a new issue with your system details and error messages
