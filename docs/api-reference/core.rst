Core API
========

The core API provides the main classes and functions for homodyne analysis.

HomodyneAnalysisCore
--------------------

.. autoclass:: homodyne.core.HomodyneAnalysisCore
   :members:
   :show-inheritance:

   The main analysis class that orchestrates the entire homodyne analysis workflow.

   .. automethod:: __init__
   
   **Key Methods:**
   
   .. automethod:: load_experimental_data
   .. automethod:: optimize_classical
   .. automethod:: run_mcmc_sampling
   .. automethod:: validate_configuration

ConfigManager
-------------

.. autoclass:: homodyne.config.ConfigManager
   :members:
   :show-inheritance:

   Manages configuration loading, validation, and access.

   .. automethod:: __init__
   .. automethod:: validate
   .. automethod:: get_analysis_settings
   .. automethod:: get_file_paths
   .. automethod:: is_mcmc_enabled

ModelFunctions
--------------

.. automodule:: homodyne.models
   :members:
   :show-inheritance:

   Physical model functions for correlation analysis.

Core Functions
--------------

.. autofunction:: homodyne.utils.load_data_file

.. autofunction:: homodyne.utils.validate_angles

.. autofunction:: homodyne.utils.apply_angle_filtering

Example Usage
-------------

**Basic Analysis**:

.. code-block:: python

   from homodyne import HomodyneAnalysisCore, ConfigManager
   
   # Initialize configuration
   config = ConfigManager("my_experiment.json")
   
   # Create analysis instance
   analysis = HomodyneAnalysisCore(config)
   
   # Load data and run analysis
   analysis.load_experimental_data()
   result = analysis.optimize_classical()
   
   print(f"Optimized parameters: {result.x}")
   print(f"Chi-squared: {result.fun:.4f}")

**MCMC Analysis**:

.. code-block:: python

   from homodyne import HomodyneAnalysisCore, ConfigManager
   
   config = ConfigManager("mcmc_config.json")
   analysis = HomodyneAnalysisCore(config)
   
   # Classical optimization first
   classical_result = analysis.optimize_classical()
   
   # MCMC sampling for uncertainty quantification
   mcmc_result = analysis.run_mcmc_sampling()
   
   print(f"MCMC converged: {mcmc_result['converged']}")
   print(f"R-hat values: {mcmc_result['rhat']}")

**Configuration Validation**:

.. code-block:: python

   from homodyne import ConfigManager
   
   try:
       config = ConfigManager("my_config.json")
       config.validate()
       print("✅ Configuration is valid")
   except ValueError as e:
       print(f"❌ Configuration error: {e}")