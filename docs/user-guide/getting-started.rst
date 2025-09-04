Getting Started
===============

Choose your path based on your experience and needs.

.. contents:: Quick Navigation
   :local:
   :depth: 1

🚀 I'm New to Homodyne Analysis
--------------------------------

**Perfect starting point for beginners**

**What you'll learn:** Basic concepts, installation, and your first analysis

**Time needed:** 15-30 minutes

**Path:**

1. **Start here:** :doc:`installation` - Install with isolated backend support
2. **Learn basics:** :doc:`quickstart` - 5-minute tutorial
3. **Understand modes:** :doc:`analysis-modes` - Choose the right approach
4. **Try examples:** :doc:`examples` - Real-world use cases

**Quick commands to get started:**

.. code-block:: bash

   pip install homodyne-analysis[mcmc]              # CPU backend (recommended)
   homodyne-post-install --shell zsh               # Setup shell features
   homodyne-config --mode static_isotropic --sample my_first_analysis
   homodyne --method classical --config my_first_analysis_config.json

.. tip::
   Start with **Static Isotropic mode** (3 parameters) for the fastest and simplest analysis.

⚡ I Want to Analyze Data Quickly
----------------------------------

**Skip theory, focus on results**

**What you need:** Your data files and 5 minutes

**Time needed:** 5-10 minutes

**Express path:**

1. **Quick setup:** Run ``pip install homodyne-analysis[all]``
2. **Generate config:** ``homodyne-config --mode static_isotropic --sample quick_analysis``
3. **Edit paths:** Update file paths in ``quick_analysis_config.json``
4. **Run analysis:** ``homodyne --method classical --config quick_analysis_config.json``
5. **Check results:** Open ``homodyne_results/homodyne_analysis_results.json``

**Essential files you need:**

- **c2 data file:** HDF5 or NPZ with correlation functions
- **angles file:** Text file with scattering angles in degrees

.. admonition:: Quick Reference
   :class: tip

   See :doc:`quick-reference` for commands, parameters, and troubleshooting.

🔬 I'm a Researcher Needing Advanced Features
----------------------------------------------

**Comprehensive analysis with uncertainty quantification**

**What you get:** MCMC sampling, robust optimization, GPU acceleration

**Time needed:** 30-60 minutes for setup, then production use

**Research-focused path:**

1. **Full installation:** ``pip install homodyne-analysis[mcmc-all]`` (both CPU and GPU backends)
2. **Post-install setup:** ``homodyne-post-install --shell zsh --gpu --advanced``
3. **System validation:** ``homodyne-validate --verbose``
4. **Configuration:** :doc:`configuration-guide` - Detailed parameter setup
5. **Backend selection:** :doc:`../api-reference/mcmc` - CPU vs GPU backends

**Recommended workflow for research:**

.. code-block:: bash

   # 1. Start with classical optimization (fast)
   homodyne --method classical --config my_config.json
   
   # 2. Use results for MCMC initialization (uncertainty quantification)
   homodyne --method mcmc --config my_config.json
   
   # 3. For noisy data, try robust methods
   homodyne --method robust --config my_config.json
   
   # 4. GPU acceleration for large datasets (Linux)
   homodyne-gpu --method mcmc --config my_config.json

**Key research features:**

- **Isolated MCMC backends:** CPU (PyMC) and GPU (NumPyro/JAX) completely separated
- **Uncertainty quantification:** Posterior distributions, credible intervals
- **Robust optimization:** Methods resistant to noise and outliers  
- **Performance optimization:** GPU acceleration, JIT compilation
- **Comprehensive diagnostics:** Convergence checking, model validation

🛠️ I'm a Developer/Power User
-------------------------------

**API access, customization, and integration**

**What you'll explore:** Python API, architecture, extensibility

**Time needed:** Variable based on integration needs

**Developer path:**

1. **Development install:** ``git clone`` and ``pip install -e .[dev]``
2. **Architecture:** :doc:`../developer-guide/architecture` - System design
3. **API reference:** :doc:`../api-reference/index` - Complete API
4. **Packaging:** :doc:`../developer-guide/packaging` - Backend system details
5. **Testing:** :doc:`../developer-guide/testing` - Quality assurance

**Python API quick start:**

.. code-block:: python

   from homodyne import HomodyneAnalysisCore, ConfigManager
   from homodyne.run_homodyne import get_mcmc_backend
   
   # Configuration management
   config = ConfigManager("config.json")
   
   # Backend selection
   mcmc_function, backend_name, has_gpu = get_mcmc_backend()
   
   # Analysis orchestration
   analysis = HomodyneAnalysisCore(config)
   results = analysis.optimize_all()

**Advanced integration:**

.. code-block:: python

   # Direct backend usage
   import os
   os.environ["HOMODYNE_GPU_INTENT"] = "true"  # Force GPU backend
   
   results = mcmc_function(
       analysis_core=analyzer,
       config=config.config,
       c2_experimental=data,
       phi_angles=angles
   )

**Development tools:**

.. code-block:: bash

   # Code quality
   pre-commit install
   pytest homodyne/tests/ -v
   
   # Performance testing
   homodyne-gpu-optimize --benchmark
   
   # System validation
   homodyne-validate --test all --verbose

🔧 I Need to Troubleshoot Issues
---------------------------------

**Diagnose and fix problems**

**Common solutions at your fingertips**

**Diagnostic tools:**

.. code-block:: bash

   # System health check
   homodyne-validate --quick
   
   # Detailed system analysis
   homodyne-validate --verbose
   
   # GPU-specific diagnostics
   gpu-status
   homodyne-gpu-optimize --report
   
   # Backend testing
   python -c "from homodyne.optimization.mcmc_cpu_backend import is_cpu_mcmc_available; print(f'CPU MCMC: {is_cpu_mcmc_available()}')"

**Quick fixes:**

.. list-table::
   :widths: 40 60
   :header-rows: 1

   * - Problem
     - Solution
   * - Installation issues
     - See :doc:`installation` common issues section
   * - Configuration errors
     - Use ``homodyne-config --mode <mode>`` to regenerate
   * - Backend problems
     - See :doc:`../api-reference/mcmc` backend troubleshooting
   * - Performance issues
     - Check :doc:`../developer-guide/performance` optimization guide
   * - Analysis failures
     - Enable verbose logging: ``--verbose``

**Detailed troubleshooting:** :doc:`../developer-guide/troubleshooting`

📚 I Want to Understand the Science
------------------------------------

**Theoretical background and implementation details**

**Learn the physics and mathematics behind homodyne analysis**

**Science-focused resources:**

1. **Main paper:** `He et al. PNAS 2024 <https://doi.org/10.1073/pnas.2401162121>`_
2. **Theory overview:** See main :doc:`../index` for key equations
3. **Analysis modes:** :doc:`analysis-modes` - Physical interpretation
4. **MCMC priors:** :doc:`../api-reference/mcmc` - Bayesian approach
5. **Performance:** :doc:`../developer-guide/performance` - Computational methods

**Key equations implemented:**

- **Full nonequilibrium:** c₂(q⃗, t₁, t₂) = 1 + β[e^(-q²∫J(t)dt)] × sinc²[qh ∫γ̇(t)cos(φ(t))dt]
- **Constant shear:** c₂(q⃗, t₁, t₂) = 1 + β[e^(-6q²D(t₂-t₁))] sinc²[qh cos(φ)γ̇(t₂-t₁)]
- **One-time correlation:** g₂(q⃗, τ) = 1 + β[e^(-6q²Dτ)] sinc²[qh cos(φ)γ̇τ]

**Physical parameters:**

- **Transport coefficients:** Diffusion D(t), shear rate γ̇(t)
- **System geometry:** Gap h, scattering vector q⃗  
- **Angular dependencies:** Flow angle φ(t)

Next Steps
----------

**After choosing your path:**

1. **Install the package** using your preferred backend option
2. **Run the post-install setup** for enhanced features
3. **Validate your installation** with ``homodyne-validate``
4. **Try the quickstart** with your own data or examples
5. **Join the community** - report issues, contribute, ask questions

**Need help?**

- **Documentation:** Comprehensive guides and API reference
- **Issues:** `GitHub Issues <https://github.com/imewei/homodyne/issues>`_
- **Examples:** :doc:`examples` with real datasets
- **Troubleshooting:** :doc:`../developer-guide/troubleshooting`

.. tip::
   Bookmark :doc:`quick-reference` for essential commands and troubleshooting during your analysis work.