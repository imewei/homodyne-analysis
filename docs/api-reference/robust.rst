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

.. autoclass:: RobustHomodyneOptimizer
   :members:
   :show-inheritance:

Key Methods
-----------

.. automethod:: RobustHomodyneOptimizer.optimize_robust

.. automethod:: RobustHomodyneOptimizer._solve_wasserstein_dro

.. automethod:: RobustHomodyneOptimizer._solve_scenario_robust

.. automethod:: RobustHomodyneOptimizer._solve_ellipsoidal_robust

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

   from homodyne.optimization.robust import RobustHomodyneOptimizer
   
   optimizer = RobustHomodyneOptimizer(analysis_core, config)
   
   # Run all robust methods
   results = optimizer.optimize_robust(
       theta_init=np.array([1000, -0.5, 100]),
       phi_angles=phi_angles,
       c2_experimental=c2_data
   )

**Individual Methods:**

.. code-block:: python

   # Wasserstein DRO only
   result = optimizer._solve_wasserstein_dro(
       theta_init, phi_angles, c2_experimental,
       uncertainty_radius=0.02
   )
   
   # Scenario-based robust optimization
   result = optimizer._solve_scenario_robust(
       theta_init, phi_angles, c2_experimental,
       n_scenarios=30
   )
   
   # Ellipsoidal uncertainty sets
   result = optimizer._solve_ellipsoidal_robust(
       theta_init, phi_angles, c2_experimental,
       gamma=0.08
   )

Performance Notes
-----------------

- **CVXPY Solvers**: Prefers CLARABEL > SCS > CVXOPT for performance
- **Caching**: Enables Jacobian and correlation caching for repeated evaluations
- **Problem Scaling**: Automatic scaling for numerical stability
- **Progressive Optimization**: Two-stage coarse-to-fine optimization

The robust optimization methods provide **noise-resistant parameter estimation** at the cost of ~2-5x longer computation time compared to classical methods.