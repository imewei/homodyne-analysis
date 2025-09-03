API Reference
=============

This section provides comprehensive documentation for all public classes, functions, and modules in the Homodyne Analysis package.

.. contents:: Table of Contents
   :local:
   :depth: 2

Overview
--------

The Homodyne Analysis package is organized into several key modules for analyzing X-ray Photon Correlation Spectroscopy (XPCS) data under nonequilibrium conditions.

**Quick Start Example:**

.. code-block:: python

   from homodyne import HomodyneAnalysisCore, ConfigManager

   # Load configuration and run analysis
   config = ConfigManager("config.json")
   analysis = HomodyneAnalysisCore(config)
   results = analysis.optimize_all()  # Run all optimization methods

Core Modules
------------

.. toctree::
   :maxdepth: 2
   :caption: Core Functionality

   core
   analysis
   config

Optimization Methods
--------------------

.. toctree::
   :maxdepth: 2
   :caption: Optimization Engines

   optimization

Visualization and Output
------------------------

.. toctree::
   :maxdepth: 2
   :caption: Plotting and Results

   plotting

System Tools
-------------

.. toctree::
   :maxdepth: 2
   :caption: Runtime Utilities

   runtime
   cli

Complete API Reference
----------------------

.. autosummary::
   :toctree: _autosummary
   :recursive:
   :template: module.rst

   homodyne

Key Classes and Functions
-------------------------

Main Analysis Interface
~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: _autosummary

   homodyne.analysis.core.HomodyneAnalysisCore
   homodyne.core.config.ConfigManager

Optimization Functions
~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: _autosummary

   homodyne.optimization.classical.ClassicalOptimizer
   homodyne.optimization.robust.RobustHomodyneOptimizer
   homodyne.optimization.mcmc.MCMCSampler

Core Computational Functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: _autosummary

   homodyne.core.kernels.compute_g1_correlation_numba
   homodyne.core.kernels.compute_chi_squared_batch_numba
   homodyne.core.io_utils.save_analysis_results

Command Line Interfaces
~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: _autosummary

   homodyne.run_homodyne.main
   homodyne.create_config.main
   homodyne.post_install.main

Usage Patterns
--------------

**Basic Analysis Workflow:**

.. code-block:: python

   # Configuration management
   from homodyne.core.config import ConfigManager
   config = ConfigManager("my_experiment.json")

   # Main analysis
   from homodyne.analysis.core import HomodyneAnalysisCore
   analysis = HomodyneAnalysisCore(config)
   
   # Run specific optimization method
   classical_results = analysis.optimize_classical()
   robust_results = analysis.optimize_robust()
   mcmc_results = analysis.optimize_mcmc()

**Direct Function Access:**

.. code-block:: python

   # Direct optimization calls
   from homodyne.optimization.classical import optimize_nelder_mead
   from homodyne.core.io_utils import load_experimental_data
   
   data = load_experimental_data("data.npz")
   result = optimize_nelder_mead(config, data)

**Visualization:**

.. code-block:: python

   # Plotting functions
   from homodyne.plotting import plot_correlation_heatmap, plot_mcmc_diagnostics
   
   plot_correlation_heatmap(results["fitted_data"])
   plot_mcmc_diagnostics(mcmc_trace)