Robust Optimization Module
==========================

.. currentmodule:: homodyne.optimization.robust

The robust optimization module provides distributionally robust optimization methods for parameter estimation under measurement uncertainty and outliers.

Key Features
------------

- **Wasserstein DRO**: Distributionally robust optimization using Wasserstein uncertainty sets
- **Scenario-based Robust**: Multi-scenario optimization using bootstrap resampling
- **Ellipsoidal Uncertainty**: Robust least squares with bounded uncertainty
- **CVXPY Integration**: High-performance convex optimization with multiple solvers
- **Automatic Scaling**: Proper ``fitted = contrast × theory + offset`` relationship
- **Reduced Chi-squared**: Proper statistical objective functions

Classes
-------

**RobustHomodyneOptimizer**

Main class for distributionally robust optimization methods.

Key Methods
-----------

**optimize(phi_angles, c2_experimental, method="wasserstein", epsilon=0.1, **kwargs)**

Run robust optimization using the specified method. This is the main public API for robust optimization.

**Parameters:**
- ``phi_angles``: Array of scattering angles [degrees]
- ``c2_experimental``: Experimental correlation data
- ``method``: Optimization method ("wasserstein", "scenario", or "ellipsoidal")
- ``epsilon``: Uncertainty radius (for Wasserstein) or related parameter
- ``**kwargs``: Method-specific options

**Returns:** Dictionary with optimal parameters, chi-squared value, and method-specific results

Configuration
-------------

Robust optimization is configured through the ``optimization_config.robust_optimization`` section:

.. code-block:: python

   config = {
       "optimization_config": {
           "robust_optimization": {
               "enabled": True,
               "uncertainty_model": "wasserstein",  # or "scenario", "ellipsoidal"
               "method_options": {
                   "wasserstein": {
                       "uncertainty_radius": 0.02,
                       "regularization_alpha": 0.005
                   },
                   "scenario": {
                       "n_scenarios": 30,
                       "bootstrap_method": "residual"
                   },
                   "ellipsoidal": {
                       "gamma": 0.08,
                       "l1_regularization": 0.0005,
                       "l2_regularization": 0.005
                   }
               }
           }
       }
   }

Usage Examples
--------------

**Basic Robust Optimization:**

.. code-block:: python

   import numpy as np
   import json
   from homodyne.analysis.core import HomodyneAnalysisCore
   from homodyne.optimization.robust import RobustHomodyneOptimizer
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

   # Create robust optimizer
   robust = RobustHomodyneOptimizer(core, config)

   # Run Wasserstein DRO
   result = robust.optimize(
       phi_angles=phi_angles,
       c2_experimental=c2_data,
       method="wasserstein",
       epsilon=0.1
   )

   print(f"Optimal parameters: {result['optimal_params']}")
   print(f"Chi-squared: {result['chi_squared']:.6e}")

**Different Robust Methods:**

.. code-block:: python

   # Wasserstein distributionally robust optimization
   wasserstein_result = robust.optimize(
       phi_angles=phi_angles,
       c2_experimental=c2_data,
       method="wasserstein",
       epsilon=0.1  # Uncertainty radius
   )

   # Scenario-based robust optimization
   scenario_result = robust.optimize(
       phi_angles=phi_angles,
       c2_experimental=c2_data,
       method="scenario",
       n_scenarios=30  # Number of bootstrap scenarios
   )

   # Ellipsoidal uncertainty robust optimization
   ellipsoidal_result = robust.optimize(
       phi_angles=phi_angles,
       c2_experimental=c2_data,
       method="ellipsoidal",
       gamma=0.08  # Uncertainty bound
   )

Performance Notes
-----------------

- **CVXPY Solvers**: Prefers CLARABEL > SCS > CVXOPT for performance
- **Caching**: Enables Jacobian and correlation caching for repeated evaluations
- **Problem Scaling**: Automatic scaling for numerical stability
- **Progressive Optimization**: Two-stage coarse-to-fine optimization

The robust optimization methods provide **noise-resistant parameter estimation** at the cost of ~2-5x longer computation time compared to classical methods.
