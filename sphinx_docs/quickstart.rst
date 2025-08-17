Quick Start Guide
=================

This guide will get you up and running with the homodyne package in minutes.

Basic Setup
-----------

1. **Install the package** (see :doc:`installation` for details):

.. code-block:: bash

   pip install numpy scipy matplotlib numba

2. **Create your first configuration**:

.. code-block:: bash

   # Create a configuration for isotropic analysis (fastest)
   python create_config.py --mode static_isotropic --sample my_sample

3. **Run your first analysis**:

.. code-block:: bash

   python run_homodyne.py --static-isotropic --method classical

First Analysis Example
-----------------------

Here's a complete example using the Python API:

.. code-block:: python

   from homodyne import HomodyneAnalysisCore, ConfigManager
   
   # Load configuration
   config = ConfigManager("config_static_isotropic_my_sample.json")
   
   # Initialize analysis
   analysis = HomodyneAnalysisCore(config)
   
   # Run classical optimization
   results = analysis.optimize_classical()
   
   # Print results
   print(f"Optimized parameters: {results.x}")
   print(f"Chi-squared value: {results.fun}")

Command Line Examples
---------------------

**Static Isotropic Mode** (fastest, 3 parameters):

.. code-block:: bash

   python run_homodyne.py --static-isotropic --method classical

**Static Anisotropic Mode** (3 parameters with angle filtering):

.. code-block:: bash

   python run_homodyne.py --static-anisotropic --method classical

**Laminar Flow Mode** (full 7-parameter analysis):

.. code-block:: bash

   python run_homodyne.py --laminar-flow --method classical

**With Data Validation**:

.. code-block:: bash

   python run_homodyne.py --plot-experimental-data --static-isotropic --method classical

**Bayesian Analysis with Uncertainty Quantification**:

.. code-block:: bash

   python run_homodyne.py --static-isotropic --method mcmc

Understanding the Analysis Modes
---------------------------------

Choose the right mode for your experimental conditions:

.. list-table:: Mode Selection Guide
   :widths: 20 15 15 50
   :header-rows: 1

   * - Experimental Condition
     - Recommended Mode
     - Parameters
     - Command
   * - Isotropic scattering, no flow
     - Static Isotropic
     - 3
     - ``--static-isotropic``
   * - Angular dependence, no flow
     - Static Anisotropic
     - 3
     - ``--static-anisotropic``
   * - System under shear/flow
     - Laminar Flow
     - 7
     - ``--laminar-flow``

Configuration Templates
-----------------------

Generate analysis-specific configurations:

.. code-block:: bash

   # Isotropic analysis
   python create_config.py --mode static_isotropic --sample protein_01
   
   # Anisotropic analysis with metadata
   python create_config.py --mode static_anisotropic --sample collagen \\
                           --author "Your Name" --experiment "Static analysis"
   
   # Flow analysis
   python create_config.py --mode laminar_flow --sample microgel \\
                           --experiment "Microgel dynamics under shear"

Data Validation
---------------

Always validate your experimental data before analysis:

.. code-block:: bash

   # Generate data validation plots
   python run_homodyne.py --plot-experimental-data --verbose

This creates validation plots in ``./plots/data_validation/`` showing:
- Full 2D correlation function heatmaps g₂(t₁,t₂)
- Diagonal slices g₂(t,t) showing temporal decay
- Cross-sectional profiles at different time points
- Statistical summaries with data quality metrics

Next Steps
----------

* **Understand Analysis Modes**: Read :doc:`analysis_modes` for detailed mode descriptions
* **Configuration Options**: See :doc:`configuration` for all available settings
* **Optimization Methods**: Learn about :doc:`optimization_methods` for classical vs. MCMC approaches
* **Performance Tuning**: Check :doc:`performance` for optimization tips
* **Troubleshooting**: See :doc:`troubleshooting` for common issues and solutions

Common Workflows
----------------

**1. Quick Parameter Estimation**:

.. code-block:: bash

   # Fast analysis for initial parameter estimates
   python run_homodyne.py --static-isotropic --method classical

**2. Comprehensive Analysis with Uncertainty**:

.. code-block:: bash

   # First get point estimates
   python run_homodyne.py --static-anisotropic --method classical
   # Then run MCMC for uncertainties
   python run_homodyne.py --static-anisotropic --method mcmc

**3. Flow System Analysis**:

.. code-block:: bash

   # Validate data first
   python run_homodyne.py --plot-experimental-data --laminar-flow
   # Then analyze
   python run_homodyne.py --laminar-flow --method all

Performance Tips
----------------

* **First Analysis**: Allow extra time for Numba JIT compilation warmup
* **Large Datasets**: Start with isotropic mode when applicable (fastest)
* **Memory Constraints**: Enable angle filtering in configuration
* **Parallel Processing**: Set appropriate ``num_threads`` in configuration

That's it! You're now ready to analyze homodyne scattering data. For more detailed information, explore the other sections of this documentation.
