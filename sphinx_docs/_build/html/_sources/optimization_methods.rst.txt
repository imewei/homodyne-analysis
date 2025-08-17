Optimization Methods
====================

The homodyne package provides two complementary optimization approaches: fast classical optimization for point estimates and robust Bayesian MCMC sampling for full posterior distributions with uncertainty quantification.

Method Overview
---------------

.. list-table:: Optimization Method Comparison
   :widths: 20 20 20 40
   :header-rows: 1

   * - Method
     - Performance
     - Output Type
     - Best Use Case
   * - **Classical**
     - Fast (~minutes)
     - Point estimates
     - Exploratory analysis, parameter screening
   * - **MCMC**
     - Comprehensive (~hours)
     - Full posteriors
     - Uncertainty quantification, robust estimates
   * - **Combined**
     - Sequential
     - Both
     - Recommended workflow

Classical Optimization
----------------------

Algorithm Details
~~~~~~~~~~~~~~~~~

The classical optimization uses the **Nelder-Mead simplex method** via SciPy, which is well-suited for the complex, multi-modal optimization landscapes typical in correlation function fitting.

**Key Features**:
- Derivative-free optimization
- Robust to local minima
- Handles parameter bounds effectively
- Fast execution for most parameter spaces

Mathematical Formulation
~~~~~~~~~~~~~~~~~~~~~~~~

The objective function minimizes the chi-squared statistic:

.. math::

   \chi^2(\mathbf{p}) = \sum_i \frac{[g_{2,\text{exp}}(i) - g_{2,\text{theory}}(i, \mathbf{p})]^2}{\sigma_i^2}

where:
- :math:`\mathbf{p}` is the parameter vector
- :math:`g_{2,\text{exp}}` is the experimental correlation function
- :math:`g_{2,\text{theory}}` is the theoretical model
- :math:`\sigma_i` are the experimental uncertainties

Always-On Scaling Optimization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Scaling optimization is automatically enabled** for scientifically accurate results. The relationship between experimental and theoretical correlation functions is:

.. math::

   g_2 = \text{offset} + \text{contrast} \times g_1

The optimal scaling parameters are determined using least squares:

.. code-block:: python

   A = np.vstack([theory, np.ones(len(theory))]).T
   scaling, residuals, _, _ = np.linalg.lstsq(A, experimental, rcond=None)
   contrast, offset = scaling

Configuration
~~~~~~~~~~~~~

.. code-block:: json

   {
     "optimization": {
       "method": "Nelder-Mead",
       "max_iterations": 10000,
       "tolerance": 1e-8,
       "initial_simplex_size": 0.1
     }
   }

**Parameters**:
- **method**: Optimization algorithm (Nelder-Mead recommended)
- **max_iterations**: Maximum number of optimization steps
- **tolerance**: Convergence tolerance for parameter changes
- **initial_simplex_size**: Initial simplex size relative to parameter values

Usage Examples
~~~~~~~~~~~~~~

**Command Line**:

.. code-block:: bash

   # Basic classical optimization
   python run_homodyne.py --static-isotropic --method classical

   # With specific configuration
   python run_homodyne.py --config my_config.json --method classical

**Python API**:

.. code-block:: python

   from homodyne import HomodyneAnalysisCore, ConfigManager
   
   config = ConfigManager("my_config.json")
   analysis = HomodyneAnalysisCore(config)
   
   # Run classical optimization
   results = analysis.optimize_classical()
   
   # Access results
   optimized_params = results.x
   chi_squared = results.fun
   success = results.success
   iterations = results.nit

Performance Characteristics
~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Speed**: Typically completes in minutes for most datasets
**Memory**: Low memory footprint
**Scalability**: Performance scales well with parameter count
**Reliability**: Robust convergence for well-posed problems

Bayesian MCMC Sampling
-----------------------

Algorithm Details
~~~~~~~~~~~~~~~~~

The package uses **PyMC** with the **NUTS (No-U-Turn Sampler)** algorithm, which provides:

- Efficient exploration of posterior distributions
- Automatic step size adaptation
- Geometric convergence properties
- Robust handling of parameter correlations

**Key Features**:
- Full posterior distributions
- Uncertainty quantification
- Convergence diagnostics
- Parameter correlation analysis

Bayesian Formulation
~~~~~~~~~~~~~~~~~~~~~

The Bayesian approach treats parameters as random variables with prior distributions and computes the posterior:

.. math::

   P(\mathbf{p}|\mathbf{d}) \propto P(\mathbf{d}|\mathbf{p}) \times P(\mathbf{p})

where:
- :math:`P(\mathbf{p}|\mathbf{d})` is the posterior distribution
- :math:`P(\mathbf{d}|\mathbf{p})` is the likelihood function
- :math:`P(\mathbf{p})` is the prior distribution

Prior Distributions
~~~~~~~~~~~~~~~~~~~

The package uses weakly informative priors based on physical constraints:

.. code-block:: python

   # Example prior setup
   D0 = pm.Uniform("D0", lower=1e-15, upper=1e-9)
   alpha = pm.Uniform("alpha", lower=0.1, upper=2.0)
   D_offset = pm.Uniform("D_offset", lower=0.0, upper=1e-11)

Configuration
~~~~~~~~~~~~~

.. code-block:: json

   {
     "mcmc": {
       "n_samples": 2000,
       "tune": 1000,
       "chains": 4,
       "target_accept": 0.8,
       "random_seed": 42,
       "cores": 4
     }
   }

**Parameters**:
- **n_samples**: Number of posterior samples per chain
- **tune**: Number of tuning steps for sampler adaptation
- **chains**: Number of parallel MCMC chains
- **target_accept**: Target acceptance rate (0.8-0.95 recommended)
- **random_seed**: Random seed for reproducibility
- **cores**: Number of CPU cores for parallel sampling

Usage Examples
~~~~~~~~~~~~~~

**Command Line**:

.. code-block:: bash

   # Basic MCMC sampling
   python run_homodyne.py --static-isotropic --method mcmc

   # Flow analysis with MCMC
   python run_homodyne.py --laminar-flow --method mcmc

**Python API**:

.. code-block:: python

   from homodyne import HomodyneAnalysisCore, ConfigManager
   
   config = ConfigManager("my_config.json")
   analysis = HomodyneAnalysisCore(config)
   
   # Run MCMC sampling
   mcmc_results = analysis.optimize_mcmc()
   
   # Access posterior samples
   posterior = mcmc_results.posterior
   parameter_means = posterior.mean()
   parameter_stds = posterior.std()
   
   # Convergence diagnostics
   r_hat = mcmc_results.r_hat  # Should be < 1.1
   ess = mcmc_results.effective_sample_size

Output Analysis
~~~~~~~~~~~~~~~

**Posterior Summary**:

.. code-block:: python

   import arviz as az
   
   # Summary statistics
   summary = az.summary(mcmc_results.idata)
   print(summary)
   
   # Corner plot for parameter correlations
   az.plot_pair(mcmc_results.idata, var_names=active_parameters)

**Convergence Diagnostics**:

.. code-block:: python

   # R-hat convergence diagnostic (should be < 1.1)
   r_hat = az.rhat(mcmc_results.idata)
   
   # Effective sample size
   ess = az.ess(mcmc_results.idata)
   
   # Trace plots
   az.plot_trace(mcmc_results.idata)

Performance Characteristics
~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Speed**: Typically requires hours for comprehensive analysis
**Memory**: Higher memory requirements for large datasets
**Scalability**: Performance scales with parameter count and complexity
**Quality**: Provides robust uncertainty quantification

Combined Analysis Workflow
--------------------------

Recommended Strategy
~~~~~~~~~~~~~~~~~~~~

The most effective approach combines both methods sequentially:

1. **Classical optimization** for initial parameter estimates
2. **MCMC sampling** using classical results as starting points

.. code-block:: bash

   # Combined analysis
   python run_homodyne.py --static-anisotropic --method all

Benefits of Combined Approach
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Efficiency**:
- Classical optimization provides good starting points
- Reduces MCMC burn-in time
- Improves sampling efficiency

**Robustness**:
- Classical optimization explores parameter space quickly
- MCMC provides thorough uncertainty analysis
- Combined results validate each other

**Scientific Rigor**:
- Point estimates for parameter interpretation
- Full posteriors for uncertainty quantification
- Comprehensive characterization of parameter space

Implementation Example
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from homodyne import HomodyneAnalysisCore, ConfigManager
   
   config = ConfigManager("my_config.json")
   analysis = HomodyneAnalysisCore(config)
   
   # Step 1: Classical optimization
   classical_results = analysis.optimize_classical()
   print(f"Classical chi-squared: {classical_results.fun}")
   
   # Step 2: Use classical results to initialize MCMC
   config.override_parameters(dict(zip(
       config.get_active_parameters(),
       classical_results.x
   )))
   
   # Step 3: MCMC sampling
   mcmc_results = analysis.optimize_mcmc()
   
   # Step 4: Compare results
   classical_params = classical_results.x
   mcmc_means = mcmc_results.posterior.mean()
   mcmc_stds = mcmc_results.posterior.std()

Method Selection Guidelines
---------------------------

Choose Classical When
~~~~~~~~~~~~~~~~~~~~~

- **Exploratory analysis**: Quick parameter estimation
- **Parameter screening**: Testing multiple configurations
- **Computational constraints**: Limited time or resources
- **Initial validation**: Checking model feasibility

Choose MCMC When
~~~~~~~~~~~~~~~~

- **Publication-quality results**: Need uncertainty quantification
- **Parameter correlations**: Understanding parameter relationships
- **Robust estimation**: Want comprehensive parameter characterization
- **Scientific rigor**: Full Bayesian inference required

Choose Combined When
~~~~~~~~~~~~~~~~~~~~

- **Complete analysis**: Both point estimates and uncertainties needed
- **Complex parameter spaces**: Potential multimodality or correlations
- **Research projects**: Comprehensive characterization required
- **Validation studies**: Cross-validation of methods

Troubleshooting Optimization
-----------------------------

Classical Optimization Issues
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Convergence Problems**:

.. code-block:: json

   {
     "optimization": {
       "max_iterations": 20000,
       "tolerance": 1e-10,
       "initial_simplex_size": 0.05
     }
   }

**Poor Parameter Estimates**:
- Check initial parameter values
- Verify parameter bounds
- Ensure data quality is adequate

**Local Minima**:
- Try multiple random starting points
- Use different initial simplex sizes
- Consider parameter scaling

MCMC Issues
~~~~~~~~~~~

**Poor Convergence** (R-hat > 1.1):

.. code-block:: json

   {
     "mcmc": {
       "tune": 2000,
       "target_accept": 0.95,
       "chains": 6
     }
   }

**Low Effective Sample Size**:
- Increase number of samples
- Improve initial parameter estimates
- Check for parameter correlations

**Sampling Errors**:
- Verify parameter bounds are reasonable
- Check for numerical stability issues
- Ensure adequate computational resources

Performance Optimization
------------------------

Classical Optimization
~~~~~~~~~~~~~~~~~~~~~~

**Speed Improvements**:
- Enable Numba JIT compilation
- Use angle filtering for large datasets
- Optimize initial parameter estimates

**Memory Efficiency**:
- Use ``float32`` for large datasets
- Enable result caching
- Manage intermediate results storage

MCMC Sampling
~~~~~~~~~~~~~

**Sampling Efficiency**:
- Use good initial estimates from classical optimization
- Tune target acceptance rate (0.8-0.95)
- Adjust step size and adaptation parameters

**Computational Resources**:
- Use multiple cores for parallel chains
- Monitor memory usage for large parameter spaces
- Consider distributed computing for very large problems

Validation and Quality Assurance
---------------------------------

Result Validation
~~~~~~~~~~~~~~~~~

**Classical Results**:
- Check convergence status (``results.success``)
- Verify reasonable chi-squared values
- Compare with expected parameter ranges

**MCMC Results**:
- Monitor R-hat values (should be < 1.1)
- Check effective sample sizes (> 400 recommended)
- Examine trace plots for mixing

Cross-Validation
~~~~~~~~~~~~~~~~

.. code-block:: python

   # Compare classical and MCMC results
   classical_params = classical_results.x
   mcmc_means = mcmc_results.posterior.mean()
   
   relative_diff = np.abs(classical_params - mcmc_means) / mcmc_means
   print(f"Relative differences: {relative_diff}")
   
   # Results should generally agree within uncertainties

This comprehensive guide provides the foundation for effectively using both optimization methods in the homodyne package.
