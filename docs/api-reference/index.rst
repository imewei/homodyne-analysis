API Reference
=============

Complete API documentation for the homodyne analysis package.

.. toctree::
   :maxdepth: 2
   :caption: API Documentation

   overview
   analysis-core
   core
   robust
   utilities

Core Classes
------------

* :class:`~homodyne.analysis.core.HomodyneAnalysisCore` - Main analysis orchestrator
* :class:`~homodyne.core.config.ConfigManager` - Configuration management
* :class:`~homodyne.optimization.classical.ClassicalOptimizer` - Classical optimization
* :class:`~homodyne.optimization.robust.RobustHomodyneOptimizer` - Robust optimization

Quick Reference
---------------

**Essential Imports**:

.. code-block:: python

   from homodyne import HomodyneAnalysisCore, ConfigManager
   from homodyne.optimization.classical import ClassicalOptimizer
   from homodyne.optimization.robust import RobustHomodyneOptimizer

**Basic Workflow**:

.. code-block:: python

   import json
   from homodyne.analysis.core import HomodyneAnalysisCore
   from homodyne.optimization.classical import ClassicalOptimizer
   from homodyne.optimization.robust import RobustHomodyneOptimizer

   # 1. Load configuration
   with open("config.json") as f:
       config = json.load(f)

   # 2. Initialize analysis core
   core = HomodyneAnalysisCore(config)
   core.load_experimental_data()

   # 3. Run classical optimization
   classical = ClassicalOptimizer(core, config)
   params, results = classical.run_classical_optimization_optimized(
       phi_angles=phi_angles,
       c2_experimental=c2_data
   )

   # 4. Or run robust optimization for noisy data
   robust = RobustHomodyneOptimizer(core, config)
   robust_results = robust.optimize(
       phi_angles=phi_angles,
       c2_experimental=c2_data,
       method="wasserstein"
   )

Module Index
------------

The package includes the following key modules:

* **homodyne.core** - Core functionality and configuration
* **homodyne.analysis.core** - Main analysis engine
* **homodyne.optimization.classical** - Classical optimization (Nelder-Mead, Gurobi)
* **homodyne.optimization.robust** - Robust optimization (Wasserstein DRO, Scenario-based, Ellipsoidal)

.. note::
   For detailed API documentation, see the individual module pages in the navigation.

..
   Temporarily disabled autosummary due to import issues

   .. autosummary::
      :toctree: _autosummary
      :template: module.rst

      homodyne.core
      homodyne.core.config
      homodyne.core.kernels
      homodyne.core.io_utils
      homodyne.analysis.core
      homodyne.optimization.classical
