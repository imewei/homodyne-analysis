Homodyne Scattering Analysis Package
====================================

.. image:: https://img.shields.io/badge/License-MIT-yellow.svg
   :target: https://opensource.org/licenses/MIT
   :alt: License: MIT

.. image:: https://img.shields.io/badge/Python-3.8%2B-blue
   :target: https://www.python.org/
   :alt: Python

.. image:: https://img.shields.io/badge/Numba-JIT%20Accelerated-green
   :target: https://numba.pydata.org/
   :alt: Numba

A comprehensive Python package for analyzing homodyne scattering in X-ray Photon Correlation Spectroscopy (XPCS) under nonequilibrium conditions. This package implements the theoretical framework described in `He et al. PNAS 2024 <https://doi.org/10.1073/pnas.2401162121>`_ for characterizing nonequilibrium dynamics in soft matter systems through detailed transport coefficient analysis.

.. toctree::
   :maxdepth: 2
   :caption: User Guide
   :hidden:

   installation
   quickstart
   usage
   analysis_modes
   configuration

.. toctree::
   :maxdepth: 2
   :caption: Advanced Topics
   :hidden:

   optimization_methods
   scaling_optimization
   performance
   data_validation

.. toctree::
   :maxdepth: 2
   :caption: API Reference
   :hidden:

   api/modules

.. toctree::
   :maxdepth: 2
   :caption: Developer Guide
   :hidden:

   testing
   troubleshooting
   migration_guide
   contributing

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

Key Features
------------

📊 **Triple Analysis Modes**
   Static Isotropic (3 parameters), Static Anisotropic (3 parameters), and Laminar Flow (7 parameters) for comprehensive experimental coverage

⚡ **Always-On Scaling Optimization**
   Automatic g₂ = offset + contrast × g₁ fitting for scientifically accurate chi-squared calculations

🔍 **Comprehensive Data Validation**
   Experimental C2 data validation plots with standalone plotting capabilities

⚙️ **Enhanced Configuration System**
   Mode-specific templates with intelligent defaults and metadata injection

🎯 **Multiple Optimization Approaches**
   Fast classical optimization (Nelder-Mead) for point estimates and robust Bayesian MCMC (NUTS) for full posterior distributions

🚀 **Performance Optimizations**
   Numba JIT compilation for computational kernels, smart angle filtering, and memory-efficient data handling

📈 **Integrated Visualization**
   Experimental data validation plots, parameter evolution tracking, MCMC convergence diagnostics, and corner plots

✅ **Quality Assurance**
   Extensive test coverage with pytest framework and performance benchmarking tools

Analysis Modes Overview
-----------------------

.. list-table:: Analysis Mode Comparison
   :widths: 15 10 15 30 10 15
   :header-rows: 1

   * - Mode
     - Parameters
     - Angle Handling
     - Use Case
     - Speed
     - Command
   * - **Static Isotropic**
     - 3
     - Single dummy
     - Fastest, isotropic systems
     - ⭐⭐⭐
     - ``--static-isotropic``
   * - **Static Anisotropic**
     - 3
     - Filtering enabled
     - Static with angular deps
     - ⭐⭐
     - ``--static-anisotropic``
   * - **Laminar Flow**
     - 7
     - Full coverage
     - Flow & shear analysis
     - ⭐
     - ``--laminar-flow``

Physical Context
----------------

The package analyzes time-dependent intensity correlation functions g₂(φ,t₁,t₂) for complex fluids under nonequilibrium laminar flow conditions. It captures the interplay between Brownian diffusion and advective shear flow, enabling quantitative characterization of transport properties in flowing soft matter systems.

The theoretical framework implements the transport coefficient approach for characterizing nonequilibrium dynamics, providing insights into:

- Diffusion coefficients and their time-dependence
- Shear rate effects in flowing systems
- Angular dependencies in scattering patterns
- Scaling relationships in dynamic processes

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

Support and Contributing
------------------------

- **Issues**: `GitHub Issues <https://github.com/AdvancedPhotonSource/homodyne-analysis/issues>`_
- **Source Code**: `GitHub Repository <https://github.com/AdvancedPhotonSource/homodyne-analysis>`_
- **License**: MIT License

We welcome contributions! Please see our :doc:`contributing` guide for details.

.. toctree::
   :maxdepth: 2
   :caption: Contents:


Indices and Tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
