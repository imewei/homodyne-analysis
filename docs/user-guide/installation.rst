Installation Guide
==================

System Requirements
-------------------

- **Python**: 3.12 or higher
- **Operating System**: Windows, macOS, or Linux
- **Memory**: Minimum 4GB RAM (8GB+ recommended for MCMC)
- **Storage**: ~500MB for full installation with dependencies

Core Installation
------------------

**Step 1: Install Core Dependencies**

.. code-block:: bash

   pip install numpy scipy matplotlib

**Step 2: Install Performance Dependencies** 

.. code-block:: bash

   pip install numba

**Step 3: Install the Package**

.. code-block:: bash

   # From source (recommended for development)
   git clone https://github.com/imewei/homodyne.git
   cd homodyne
   pip install -e .

Optional Dependencies
---------------------

**For MCMC Bayesian Analysis:**

.. code-block:: bash

   pip install pymc arviz pytensor

**For Enhanced Performance:**

.. code-block:: bash

   pip install -e .[performance]

**For Development:**

.. code-block:: bash

   pip install -e .[dev]

**For Documentation:**

.. code-block:: bash

   pip install -e .[docs]

**All Dependencies:**

.. code-block:: bash

   pip install -e .[all]

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

If you encounter import errors, ensure all dependencies are installed:

.. code-block:: bash

   pip install --upgrade numpy scipy matplotlib numba

**MCMC Issues:**

For MCMC functionality, ensure PyMC is properly installed:

.. code-block:: bash

   pip install pymc arviz pytensor
   
   # Test MCMC availability
   python -c "import pymc; print('PyMC available')"

**Performance Issues:**

For optimal performance, ensure Numba is working:

.. code-block:: bash

   python -c "import numba; print(f'Numba version: {numba.__version__}')"

Getting Help
------------

If you encounter installation issues:

1. Check the `troubleshooting guide <../developer-guide/troubleshooting.html>`_
2. Search existing `GitHub issues <https://github.com/imewei/homodyne/issues>`_
3. Create a new issue with your system details and error messages