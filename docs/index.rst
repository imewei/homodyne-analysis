Homodyne Analysis Documentation
================================

.. image:: https://img.shields.io/badge/License-MIT-yellow.svg
   :target: https://opensource.org/licenses/MIT
   :alt: License: MIT

.. image:: https://img.shields.io/badge/Python-3.8%2B-blue
   :target: https://www.python.org/
   :alt: Python

.. image:: https://img.shields.io/badge/Numba-JIT%20Accelerated-green
   :target: https://numba.pydata.org/
   :alt: Numba

A comprehensive Python package for analyzing homodyne scattering in X-ray Photon Correlation Spectroscopy (XPCS) under nonequilibrium conditions. This package implements the theoretical framework described in `He et al. PNAS 2024 <https://doi.org/10.1073/pnas.2401162121>`_ for characterizing nonequilibrium dynamics in soft matter systems.

Quick Start
-----------

**Installation:**

.. code-block:: bash

   pip install numpy scipy matplotlib numba
   # For MCMC capabilities:
   pip install pymc arviz pytensor

**Basic Usage:**

.. code-block:: python

   from homodyne import HomodyneAnalysisCore, ConfigManager
   
   # Load configuration
   config = ConfigManager("my_experiment.json")
   
   # Initialize analysis
   analysis = HomodyneAnalysisCore(config)
   
   # Run analysis
   results = analysis.optimize_classical()

**Command Line:**

.. code-block:: bash

   # Basic analysis with isotropic mode (fastest)
   python run_homodyne.py --static-isotropic --method classical
   
   # Full flow analysis with uncertainty quantification
   python run_homodyne.py --laminar-flow --method mcmc

Analysis Modes
--------------

.. list-table:: 
   :widths: 20 15 25 25 15
   :header-rows: 1

   * - Mode
     - Parameters
     - Use Case
     - Speed
     - Command
   * - **Static Isotropic**
     - 3
     - Fastest, isotropic systems
     - ⭐⭐⭐
     - ``--static-isotropic``
   * - **Static Anisotropic** 
     - 3
     - Static with angular dependencies
     - ⭐⭐
     - ``--static-anisotropic``
   * - **Laminar Flow**
     - 7
     - Flow & shear analysis
     - ⭐
     - ``--laminar-flow``

Key Features
------------

🎯 **Multiple Analysis Modes**
   Static Isotropic (3 parameters), Static Anisotropic (3 parameters), and Laminar Flow (7 parameters)

⚡ **High Performance**
   Numba JIT compilation, smart angle filtering, and optimized computational kernels

🔬 **Scientific Accuracy**
   Automatic g₂ = offset + contrast × g₁ fitting for accurate chi-squared calculations

📊 **Dual Optimization**
   Fast classical optimization (Nelder-Mead) and robust Bayesian MCMC (NUTS)

🔍 **Comprehensive Validation**
   Experimental data validation plots and quality control

📈 **Visualization Tools**
   Parameter evolution tracking, MCMC diagnostics, and corner plots

User Guide
----------

.. toctree::
   :maxdepth: 2
   
   user-guide/installation
   user-guide/quickstart
   user-guide/analysis-modes
   user-guide/configuration
   user-guide/examples

API Reference
-------------

.. toctree::
   :maxdepth: 2
   
   api-reference/core
   api-reference/optimization
   api-reference/utilities

Developer Guide
---------------

.. toctree::
   :maxdepth: 2
   
   developer-guide/contributing
   developer-guide/testing
   developer-guide/performance

Citation
--------

If you use this package in your research, please cite:

.. code-block:: bibtex

   @article{he2024transport,
     title={Transport coefficient approach for characterizing nonequilibrium dynamics in soft matter},
     author={He, Hongrui and Liang, Hao and Chu, Miaoqi and Jiang, Zhang and de Pablo, Juan J and Tirrell, Matthew V and Narayanan, Suresh and Chen, Wei},
     journal={Proceedings of the National Academy of Sciences},
     volume={121},
     number={31},
     pages={e2401162121},
     year={2024},
     publisher={National Academy of Sciences},
     doi={10.1073/pnas.2401162121}
   }

Support
-------

- **Documentation**: https://imewei.github.io/homodyne/
- **Issues**: https://github.com/imewei/homodyne/issues
- **Source Code**: https://github.com/imewei/homodyne
- **License**: MIT License

Indices and Tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`