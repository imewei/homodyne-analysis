MCMC Module
===========

The MCMC module provides Bayesian analysis capabilities using PyMC for uncertainty quantification.

MCMCSampler
-----------

.. autoclass:: homodyne.optimization.mcmc.MCMCSampler
   :members:
   :show-inheritance:

   Main class for MCMC-based parameter estimation with Bayesian inference.

   .. automethod:: __init__
   
   **Core Methods:**
   
   .. automethod:: setup_model
   .. automethod:: run_sampling
   .. automethod:: diagnose_convergence
   .. automethod:: extract_results

Convergence Diagnostics
-----------------------

.. autofunction:: homodyne.optimization.mcmc.compute_rhat

.. autofunction:: homodyne.optimization.mcmc.effective_sample_size

.. autofunction:: homodyne.optimization.mcmc.assess_convergence

Prior Distributions
-------------------

.. autofunction:: homodyne.optimization.mcmc.setup_priors

.. autofunction:: homodyne.optimization.mcmc.create_parameter_priors

Model Building
--------------

.. autofunction:: homodyne.optimization.mcmc.build_pymc_model

.. autofunction:: homodyne.optimization.mcmc.setup_likelihood

Usage Examples
--------------

**Basic MCMC Sampling**:

.. code-block:: python

   from homodyne.optimization.mcmc import MCMCSampler
   from homodyne import ConfigManager
   
   config = ConfigManager("mcmc_config.json")
   sampler = MCMCSampler(config)
   
   # Setup the Bayesian model
   sampler.setup_model(experimental_data, angles)
   
   # Run MCMC sampling
   trace = sampler.run_sampling(
       draws=2000,
       tune=1000,
       chains=4,
       cores=4
   )
   
   # Check convergence
   diagnostics = sampler.diagnose_convergence(trace)
   print(f"All parameters converged: {diagnostics['converged']}")

**Advanced Convergence Checking**:

.. code-block:: python

   from homodyne.optimization.mcmc import compute_rhat, effective_sample_size
   
   # Compute R-hat for each parameter
   rhat_values = compute_rhat(trace)
   for param, rhat in rhat_values.items():
       if rhat > 1.1:
           print(f"⚠️ {param}: R̂ = {rhat:.3f} (poor convergence)")
       else:
           print(f"✅ {param}: R̂ = {rhat:.3f} (good convergence)")
   
   # Check effective sample sizes
   ess_values = effective_sample_size(trace)
   for param, ess in ess_values.items():
       print(f"{param}: ESS = {ess:.0f}")

**Custom Prior Setup**:

.. code-block:: python

   from homodyne.optimization.mcmc import setup_priors
   import pymc as pm
   
   # Define custom priors for parameters
   with pm.Model() as model:
       priors = setup_priors(
           D0_range=(100, 5000),
           alpha_range=(-2.0, 0.0),
           D_offset_range=(0, 500)
       )
       
       # Use priors in likelihood
       likelihood = setup_likelihood(priors, experimental_data)

Convergence Thresholds
----------------------

The package uses the following convergence criteria:

.. list-table:: Convergence Quality Thresholds
   :widths: 20 15 15 50
   :header-rows: 1

   * - Metric
     - Excellent
     - Good
     - Acceptable
   * - **R̂ (R-hat)**
     - < 1.01
     - < 1.05
     - < 1.1
   * - **ESS**
     - > 1000
     - > 400
     - > 100
   * - **MCSE/SD**
     - < 0.01
     - < 0.05
     - < 0.1

Configuration
-------------

**MCMC Configuration Example**:

.. code-block:: javascript

   {
     "optimization_config": {
       "mcmc_sampling": {
         "enabled": true,
         "draws": 3000,
         "tune": 1500,
         "chains": 4,
         "cores": 4,
         "target_accept": 0.9,
         "max_treedepth": 10
       }
     },
     "validation_rules": {
       "mcmc_convergence": {
         "rhat_thresholds": {
           "excellent_threshold": 1.01,
           "good_threshold": 1.05,
           "acceptable_threshold": 1.1
         },
         "ess_thresholds": {
           "excellent_threshold": 1000,
           "good_threshold": 400,
           "acceptable_threshold": 100
         }
       }
     }
   }

Performance Tips
----------------

1. **Initialization**: Use classical optimization results to initialize MCMC
2. **Tuning**: Use adequate tuning steps (≥1000) for complex models
3. **Chains**: Run multiple chains (4-6) to assess convergence
4. **Acceptance Rate**: Target 0.8-0.9 acceptance rate
5. **Tree Depth**: Increase max_treedepth if you see divergences