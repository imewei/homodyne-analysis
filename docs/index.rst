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
- **Multiple optimization methods**: Classical (Nelder-Mead, Iterative Gurobi with Trust Regions), Robust (Wasserstein DRO, Scenario-based, Ellipsoidal)
- **High performance**: Numba JIT compilation with 3-5x speedup, vectorized NumPy operations, comprehensive performance monitoring
- **Scientific accuracy**: Automatic c₂ = offset + contrast × c₁ fitting for proper chi-squared calculations

Quick Start
-----------

**Installation:**

.. code-block:: bash

   pip install homodyne-analysis[all]

**Python API:**

.. code-block:: python

   import numpy as np
   import json
   from homodyne.analysis.core import HomodyneAnalysisCore
   from homodyne.optimization.classical import ClassicalOptimizer
   from homodyne.data.xpcs_loader import load_xpcs_data

   # Load configuration
   with open("config.json", 'r') as f:
       config = json.load(f)

   # Initialize analysis core
   core = HomodyneAnalysisCore(config)

   # Load experimental data
   phi_angles = np.array([0, 36, 72, 108, 144])
   c2_data = load_xpcs_data(
       data_path=config['experimental_data']['data_folder_path'],
       phi_angles=phi_angles,
       n_angles=len(phi_angles)
   )

   # Run optimization
   optimizer = ClassicalOptimizer(core, config)
   params, results = optimizer.run_classical_optimization_optimized(
       phi_angles=phi_angles,
       c2_experimental=c2_data
   )

   print(f"D₀ = {params[0]:.3e} Å²/s")
   print(f"χ² = {results.chi_squared:.6e}")

**Command Line Interface:**

.. code-block:: bash

   # Configuration generator
   homodyne-config --mode static_isotropic --sample protein_01
   homodyne-config --mode laminar_flow --sample microgel

   # Main analysis command
   homodyne                                    # Default classical method
   homodyne --method robust                    # Robust optimization only
   homodyne --method all --verbose             # All methods with debug logging

   # Analysis mode control
   homodyne --static-isotropic                 # Force 3-parameter isotropic mode
   homodyne --static-anisotropic               # Force 3-parameter anisotropic mode
   homodyne --laminar-flow                     # Force 7-parameter flow mode

   # Data visualization
   homodyne --plot-experimental-data           # Validate experimental data
   homodyne --plot-simulated-data              # Plot theoretical correlations
   homodyne --plot-simulated-data --contrast 1.5 --offset 0.1 --phi-angles "0,45,90,135"

   # Configuration and output
   homodyne --config my_config.json --output-dir ./results --verbose
   homodyne --quiet                            # File logging only, no console output

What's New in v1.0.0
--------------------

**Critical Bug Fixes:**

* **Frame Counting Convention** - Fixed 1-based inclusive counting to 0-based Python slicing conversion

  - Formula: ``time_length = end_frame - start_frame + 1`` (inclusive)
  - Resolves NaN chi-squared values and cache dimension mismatches
  - Applied across all 9 modules: analysis core, data loader, optimizers, CLI

* **Conditional Angle Subsampling** - Preserve angular information when ``n_angles < 4``

  - Prevents loss of critical angular information (e.g., 2 angles → 1 angle)
  - Time subsampling still applied for performance (~16x reduction)
  - Implemented in both classical and robust optimizers

* **Memory Optimization** - Increased ellipsoidal optimization memory limit to 90%

  - Handles large datasets with 8M+ data points without overflow
  - Fixed stacked decorator issue in robust optimization

**Performance Improvements:**

* **Numba JIT Compilation** - 3-5x speedup for core calculations
* **Vectorized Operations** - Optimized NumPy array processing throughout
* **Smart Caching** - Intelligent data caching with automatic dimension validation

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
   Automatic c₂ = offset + contrast × c₁ fitting for accurate chi-squared calculations

**Multiple Optimization Methods**

**Security and Code Quality**
   Comprehensive security scanning with Bandit, dependency vulnerability checking with pip-audit, and automated code quality tools

**Comprehensive Validation**
   Experimental data validation plots and quality control

**Visualization Tools**

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
   user-guide/plotting
   user-guide/ml-acceleration
   user-guide/examples

API Reference
-------------

.. toctree::
   :maxdepth: 2

   api-reference/core
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
   c₂(q⃗, τ) = 1 + β[e^(-6q²Dτ)] sinc²[1/(2π) qh cos(φ)γ̇τ]

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
