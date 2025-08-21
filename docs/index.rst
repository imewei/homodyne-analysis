Homodyne Scattering Analysis Package
====================================

.. image:: https://img.shields.io/badge/License-MIT-yellow.svg
   :target: https://opensource.org/licenses/MIT
   :alt: License: MIT

.. image:: https://img.shields.io/badge/Python-3.12%2B-blue
   :target: https://www.python.org/
   :alt: Python

.. image:: https://img.shields.io/badge/Numba-JIT%20Accelerated-green
   :target: https://numba.pydata.org/
   :alt: Numba

A high-performance Python package for analyzing homodyne scattering in X-ray Photon Correlation Spectroscopy (XPCS) under nonequilibrium conditions. Implements the theoretical framework from `He et al. PNAS 2024 <https://doi.org/10.1073/pnas.2401162121>`_ for characterizing transport properties in flowing soft matter systems.

Overview
--------

This package analyzes time-dependent intensity correlation functions c₂(φ,t₁,t₂) in complex fluids under nonequilibrium conditions, capturing the interplay between Brownian diffusion and advective shear flow. The implementation provides:

- **Three analysis modes**: Static Isotropic (3 params), Static Anisotropic (3 params), Laminar Flow (7 params)
- **Dual optimization**: Fast classical (Nelder-Mead) and robust Bayesian MCMC (NUTS)
- **High performance**: Numba JIT compilation with 3-5x speedup and smart angle filtering
- **Scientific accuracy**: Automatic g₂ = offset + contrast × g₁ fitting for proper chi-squared calculations

Quick Start
-----------

**Installation:**

.. code-block:: bash

   pip install homodyne-analysis[all]

**Python API:**

.. code-block:: python

   from homodyne import HomodyneAnalysisCore, ConfigManager
   
   config = ConfigManager("config.json")
   analysis = HomodyneAnalysisCore(config)
   results = analysis.optimize_classical()

**Command Line:**

.. code-block:: bash

   # Basic analysis
   homodyne --static-isotropic --method classical
   homodyne --static-anisotropic --method mcmc
   homodyne --laminar-flow --method all

   # Data validation only
   homodyne --plot-experimental-data --config my_config.json

   # Custom configuration and output
   homodyne --config my_experiment.json --output-dir ./results

   # Generate C2 heatmaps
   homodyne --method classical --plot-c2-heatmaps

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

**Multiple Analysis Modes**
   Static Isotropic (3 parameters), Static Anisotropic (3 parameters), and Laminar Flow (7 parameters)

**High Performance**
   Numba JIT compilation, smart angle filtering, and optimized computational kernels

**Scientific Accuracy**
   Automatic g₂ = offset + contrast × g₁ fitting for accurate chi-squared calculations

**Dual Optimization**
   Fast classical optimization (Nelder-Mead) and robust Bayesian MCMC (NUTS)

**Comprehensive Validation**
   Experimental data validation plots and quality control

**Visualization Tools**
   Parameter evolution tracking, MCMC diagnostics, and corner plots

**Performance Monitoring**
   Comprehensive performance testing, regression detection, and automated benchmarking

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
   api-reference/mcmc
   api-reference/utilities

Developer Guide
---------------

.. toctree::
   :maxdepth: 2
   
   developer-guide/contributing
   developer-guide/testing
   developer-guide/performance
   developer-guide/architecture
   developer-guide/troubleshooting

Theoretical Background
----------------------

The package implements three key equations describing correlation functions in nonequilibrium laminar flow systems:

**Equation 13 - Full Nonequilibrium Laminar Flow:**
   c₂(q⃗, t₁, t₂) = 1 + β[e^(-q²∫J(t)dt)] × sinc²[1/(2π) qh ∫γ̇(t)cos(φ(t))dt]

**Equation S-75 - Equilibrium Under Constant Shear:**
   c₂(q⃗, t₁, t₂) = 1 + β[e^(-6q²D(t₂-t₁))] sinc²[1/(2π) qh cos(φ)γ̇(t₂-t₁)]

**Equation S-76 - One-time Correlation (Siegert Relation):**
   g₂(q⃗, τ) = 1 + β[e^(-6q²Dτ)] sinc²[1/(2π) qh cos(φ)γ̇τ]

**Key Parameters:**

- q⃗: scattering wavevector [Å⁻¹]
- h: gap between stator and rotor [Å]  
- φ(t): angle between shear/flow direction and q⃗ [degrees]
- γ̇(t): time-dependent shear rate [s⁻¹]
- D(t): time-dependent diffusion coefficient [Å²/s]
- β: contrast parameter [dimensionless]

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

- **Documentation**: https://homodyne.readthedocs.io/
- **Issues**: https://github.com/imewei/homodyne/issues
- **Source Code**: https://github.com/imewei/homodyne
- **License**: MIT License

Indices and Tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`