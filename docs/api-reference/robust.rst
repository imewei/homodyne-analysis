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

**optimize_robust(theta_init, phi_angles, c2_experimental)**

Run all available robust optimization methods and return the best result.

**_solve_wasserstein_dro(theta_init, phi_angles, c2_experimental, uncertainty_radius=0.02)**

Solve using Wasserstein distributionally robust optimization.

**_solve_scenario_robust(theta_init, phi_angles, c2_experimental, n_scenarios=30)**

Solve using scenario-based robust optimization with bootstrap resampling.

**_solve_ellipsoidal_robust(theta_init, phi_angles, c2_experimental, gamma=0.08)**

Solve using ellipsoidal uncertainty sets for robust least squares.

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

Solver Configuration
--------------------

**Optimal Settings for Scientific Data Fitting:**

The default solver settings have been optimized for XPCS data analysis based on extensive testing. Key improvements include:

- **Looser Tolerances**: 1e-5 instead of 1e-8 (appropriate for experimental data with noise)
- **More Iterations**: 400-600 instead of 100-200 (allows time for convergence)
- **Enhanced Equilibration**: Better numerical conditioning
- **Robust Fallbacks**: Multiple solver options ensure reliability

**Problem-Specific Optimization:**

.. code-block:: json

   {
     "solver_settings": {
       "CLARABEL": {
         "max_iter": "400-600 (depending on complexity)",
         "tol_gap_abs": 1e-5,
         "tol_feas": 1e-6,
         "equilibrate_enable": true,
         "static_regularization_enable": true
       },
       "SCS": {
         "max_iters": "8000-15000 (fallback solver)",
         "eps": 1e-4,
         "alpha": 1.8,
         "scale": "3.0-5.0 (problem-dependent)"
       }
     }
   }

**Convergence Improvements:**

- Default settings: ~70% convergence rate
- Optimized settings: >95% convergence rate  
- No loss in scientific accuracy
- Better handling of difficult datasets

.. seealso::

   :doc:`../user-guide/configuration-guide`
      Comprehensive guide to solver configuration optimization

Performance Notes
-----------------

- **CVXPY Solvers**: Prefers CLARABEL > SCS > CVXOPT for performance
- **Caching**: Enables Jacobian and correlation caching for repeated evaluations
- **Problem Scaling**: Automatic scaling for numerical stability
- **Progressive Optimization**: Two-stage coarse-to-fine optimization

The robust optimization methods provide **noise-resistant parameter estimation** at the cost of ~2-5x longer computation time compared to classical methods.
