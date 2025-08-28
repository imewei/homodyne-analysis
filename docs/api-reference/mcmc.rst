MCMC Module
===========

The MCMC module provides Bayesian analysis capabilities using PyMC for uncertainty quantification.

MCMCSampler
-----------

Main class for MCMC-based parameter estimation with Bayesian inference using PyMC.

**Initialization:**

* ``MCMCSampler(analysis_core, config)`` - Initialize with analysis core and configuration

**Core Methods:**

* ``run_mcmc_analysis()`` - Run full MCMC analysis with convergence checking
* ``compute_convergence_diagnostics()`` - Compute R-hat, ESS, and other diagnostics
* ``extract_posterior_statistics()`` - Extract mean, std, credible intervals
* ``get_best_params()`` - Get best-fit parameters from posterior
* ``get_parameter_uncertainties()`` - Get parameter uncertainties

**Model and Setup:**

* ``validate_model_setup()`` - Validate Bayesian model configuration
* ``get_model_summary()`` - Get summary of model structure
* ``assess_chain_mixing()`` - Assess MCMC chain mixing quality

**Results Management:**

* ``save_results(filepath)`` - Save MCMC results to file
* ``load_results(filepath)`` - Load previously saved results
* ``generate_posterior_samples()`` - Generate posterior samples for analysis

Utility Functions
-----------------

**create_mcmc_sampler(analysis_core, config)**

Helper function to create a properly configured MCMC sampler instance.

**Parameters:**
* ``analysis_core`` - HomodyneAnalysisCore instance
* ``config`` - Configuration dictionary

**Returns:**
* ``MCMCSampler`` - Configured sampler instance

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

**Prior Distributions**:

All parameters use **Normal distributions** in the MCMC implementation:

.. code-block:: python

   import pymc as pm

   # Standard prior distributions used in homodyne MCMC
   with pm.Model() as model:
       # Positive parameters use TruncatedNormal, others use Normal
       D0 = pm.TruncatedNormal("D0", mu=1e4, sigma=1000.0, lower=1.0)  # Diffusion coefficient (positive)
       alpha = pm.Normal("alpha", mu=-1.5, sigma=0.1)                 # Time exponent
       D_offset = pm.Normal("D_offset", mu=0.0, sigma=10.0)            # Baseline diffusion
       gamma_dot_t0 = pm.TruncatedNormal("gamma_dot_t0", mu=1e-3, sigma=1e-2, lower=1e-6)  # Reference shear rate (positive)
       beta = pm.Normal("beta", mu=0.0, sigma=0.1)                     # Shear exponent
       gamma_dot_t_offset = pm.Normal("gamma_dot_t_offset", mu=0.0, sigma=1e-3)  # Baseline shear
       phi0 = pm.Normal("phi0", mu=0.0, sigma=5.0)                     # Angular offset

**Scaling Parameters for Physical Constraints**:

The MCMC implementation includes physical scaling constraints to ensure valid correlation function values:

.. code-block:: python

   # Scaling optimization: c2_fitted = c2_theory * contrast + offset
   # Physical constraints: c2_fitted ∈ [1,2], c2_theory ∈ [0,1]

   with pm.Model() as model:
       # Bounded priors for scaling parameters
       contrast = pm.TruncatedNormal("contrast", mu=0.3, sigma=0.1, lower=0.05, upper=0.5)
       offset = pm.TruncatedNormal("offset", mu=1.0, sigma=0.2, lower=0.05, upper=1.95)

       # Apply scaling transformation
       c2_fitted = c2_theory * contrast + offset

       # Physical constraint enforcement
       pm.Potential("physical_constraint",
                   pt.switch(pt.and_(pt.ge(pt.min(c2_fitted), 1.0),
                                   pt.le(pt.max(c2_fitted), 2.0)),
                           0.0, -np.inf))

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
         "target_accept": 0.95,
         "max_treedepth": 10
       },
       "scaling_parameters": {
         "fitted_range": {
           "min": 1.0,
           "max": 2.0,
           "_description": "c2_fitted = c2_theory * contrast + offset, must be in [1,2]"
         },
         "theory_range": {
           "min": 0.0,
           "max": 1.0,
           "_description": "c2_theory normalized correlation function, must be in [0,1]"
         },
         "contrast": {
           "min": 0.05,
           "max": 0.5,
           "prior_mu": 0.3,
           "prior_sigma": 0.1,
           "type": "TruncatedNormal",
           "_description": "Scaling factor for correlation strength, typically ∈ (0, 0.5]"
         },
         "offset": {
           "min": 0.05,
           "max": 1.95,
           "prior_mu": 1.0,
           "prior_sigma": 0.2,
           "type": "TruncatedNormal",
           "_description": "Baseline correlation level, typically ∈ (0, 2.0), μ ≈ 1.0"
         }
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
4. **Acceptance Rate**: Target 0.95 acceptance rate for better constraint handling
5. **Tree Depth**: Increase max_treedepth if you see divergences