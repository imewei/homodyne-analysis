API Reference
=============

Complete API documentation for the homodyne analysis package.

.. contents:: Quick Navigation
   :local:
   :depth: 2

.. toctree::
   :maxdepth: 2
   :caption: Core Modules

   core
   mcmc
   robust
   utilities
   post-install
   runtime

Core Classes
------------

* :class:`~homodyne.analysis.core.HomodyneAnalysisCore` - Main analysis orchestrator
* :class:`~homodyne.core.config.ConfigManager` - Configuration management
* :class:`~homodyne.optimization.classical.ClassicalOptimizer` - Classical optimization
* :class:`~homodyne.optimization.robust.RobustHomodyneOptimizer` - Robust optimization
* :class:`~homodyne.optimization.mcmc.MCMCSampler` - Bayesian analysis
* :func:`~homodyne.post_install.install_shell_completion` - Shell completion setup
* :func:`~homodyne.post_install.install_gpu_acceleration` - GPU acceleration setup
* :func:`~homodyne.post_install.install_advanced_features` - Advanced CLI tools

Quick Reference
---------------

**Essential Imports**:

.. code-block:: python

   from homodyne import HomodyneAnalysisCore, ConfigManager
   from homodyne.optimization.classical import ClassicalOptimizer
   from homodyne.optimization.robust import RobustHomodyneOptimizer
   from homodyne.optimization.mcmc import MCMCSampler

**Basic Workflow**:

.. code-block:: python

   # 1. Configuration
   config = ConfigManager("config.json")

   # 2. Analysis setup
   analysis = HomodyneAnalysisCore(config)
   analysis.load_experimental_data()

   # 3. Run optimization methods
   classical_result = analysis.optimize_classical()    # Classical methods
   robust_result = analysis.optimize_robust()         # Robust methods
   mcmc_result = analysis.run_mcmc_sampling()         # MCMC sampling

   # Or run all methods
   all_results = analysis.optimize_all()

**Isolated MCMC Backend Usage**:

.. code-block:: python

   from homodyne.run_homodyne import get_mcmc_backend
   import os
   
   # Automatic backend selection
   mcmc_function, backend_name, has_gpu = get_mcmc_backend()
   print(f"Using backend: {backend_name}")
   
   # Force specific backend
   os.environ["HOMODYNE_GPU_INTENT"] = "false"  # CPU (PyMC)
   os.environ["HOMODYNE_GPU_INTENT"] = "true"   # GPU (NumPyro/JAX)
   
   # Run with isolated backend
   results = mcmc_function(
       analysis_core=analyzer,
       config=config.config,
       c2_experimental=data,
       phi_angles=angles
   )

Module Index
------------

The package includes the following key modules:

* **homodyne.core** - Core functionality and configuration
* **homodyne.analysis.core** - Main analysis engine
* **homodyne.optimization.classical** - Classical optimization (Nelder-Mead, Gurobi)
* **homodyne.optimization.robust** - Robust optimization (Wasserstein DRO, Scenario-based, Ellipsoidal)
* **homodyne.optimization.mcmc** - Bayesian MCMC sampling (NUTS)
* **homodyne.runtime** - Runtime utilities and system tools
* **homodyne.runtime.gpu** - GPU optimization and benchmarking
* **homodyne.runtime.utils** - System validation and health checks
* **homodyne.post_install** - Unified post-installation system
* **homodyne.uninstall_scripts** - Cleanup and removal utilities
* **homodyne.plotting** - Visualization utilities

.. note::
   For detailed API documentation, see the individual module pages in the navigation.

Cross-References
----------------

**User Guides:**

* :doc:`../user-guide/getting-started` - Choose your path based on experience
* :doc:`../user-guide/quickstart` - 5-minute tutorial
* :doc:`../user-guide/quick-reference` - Essential commands and code snippets
* :doc:`../user-guide/configuration` - Complete configuration reference

**Isolated MCMC Backend Documentation:**

* :doc:`mcmc` - Complete MCMC API with isolated backend details
* :doc:`../developer-guide/packaging` - Backend architecture and installation
* :doc:`../user-guide/installation` - Backend installation options

**Troubleshooting:**

* :doc:`../user-guide/troubleshooting-flowchart` - Step-by-step problem solving
* :doc:`../developer-guide/troubleshooting` - Technical diagnostics

**Advanced Topics:**

* :doc:`../developer-guide/architecture` - System design and patterns
* :doc:`../developer-guide/performance` - Optimization and benchmarking

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
      homodyne.optimization.mcmc
      homodyne.optimization.classical
      homodyne.plotting
