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

This package analyzes time-dependent intensity correlation functions $c_2(\phi,t_1,t_2)$ in complex fluids under nonequilibrium conditions, capturing the interplay between Brownian diffusion and advective shear flow. The implementation provides:

- **Three analysis modes**: Static Isotropic (3 params), Static Anisotropic (3 params), Laminar Flow (7 params)
- **Dual optimization**: Fast classical (Nelder-Mead) and robust Bayesian MCMC (NUTS)
- **High performance**: Numba JIT compilation with 3-5x speedup and smart angle filtering
- **Scientific accuracy**: Automatic $g_2 = \text{offset} + \text{contrast} \times g_1$ fitting for proper $\chi^2$ calculations

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

   # Fast analysis (3 parameters)
   python run_homodyne.py --static-isotropic --method classical
   
   # Full analysis (7 parameters + uncertainty)
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

Theoretical Background
----------------------

The package implements three key equations describing correlation functions in nonequilibrium laminar flow systems:

**Equation 13 - Full Nonequilibrium Laminar Flow:**
   $c_2(\vec{q}, t_1, t_2) = 1 + \beta[e^{-q^2\int J(t)dt}] \times \text{sinc}^2[\frac{1}{2\pi} qh \int\dot{\gamma}(t)\cos(\phi(t))dt]$

**Equation S-75 - Equilibrium Under Constant Shear:**
   $c_2(\vec{q}, t_1, t_2) = 1 + \beta[e^{-6q^2D(t_2-t_1)}] \text{sinc}^2[\frac{1}{2\pi} qh \cos(\phi)\dot{\gamma}(t_2-t_1)]$

**Equation S-76 - One-time Correlation (Siegert Relation):**
   $g_2(\vec{q}, \tau) = 1 + \beta[e^{-6q^2D\tau}] \text{sinc}^2[\frac{1}{2\pi} qh \cos(\phi)\dot{\gamma}\tau]$

**Key Parameters:**

- $\vec{q}$: scattering wavevector [Å⁻¹]
- $h$: gap between stator and rotor [Å]  
- $\phi(t)$: angle between shear/flow direction and $\vec{q}$ [degrees]
- $\dot{\gamma}(t)$: time-dependent shear rate [s⁻¹]
- $D(t)$: time-dependent diffusion coefficient [Å²/s]
- $\beta$: contrast parameter [dimensionless]

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