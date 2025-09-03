Analysis Module
===============

The analysis module contains the main analysis engine that orchestrates all optimization methods for homodyne XPCS data.

.. contents:: Contents
   :local:
   :depth: 2

Overview
--------

The :mod:`homodyne.analysis` module provides the core analysis functionality for homodyne X-ray photon correlation spectroscopy data. The main class :class:`~homodyne.analysis.core.HomodyneAnalysisCore` serves as the central orchestrator for all analysis methods.

Analysis Core
-------------

.. automodule:: homodyne.analysis.core
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

Usage Examples
--------------

Basic Analysis
~~~~~~~~~~~~~~

.. code-block:: python

   from homodyne import HomodyneAnalysisCore, ConfigManager

   # Load configuration
   config = ConfigManager("config.json")
   
   # Initialize and run analysis
   analysis = HomodyneAnalysisCore(config)
   results = analysis.optimize_all()

Method-Specific Analysis
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Classical optimization
   classical_results = analysis.optimize_classical()
   
   # Robust optimization for noisy data
   robust_results = analysis.optimize_robust()
   
   # MCMC sampling for uncertainty quantification
   mcmc_results = analysis.optimize_mcmc()

See Also
--------

- :doc:`core` - Core configuration and data handling
- :doc:`optimization` - Optimization methods
- :doc:`../user-guide/quickstart` - Getting started guide