Complete Workflow Tutorial
==========================

This tutorial demonstrates a complete analysis workflow from data preparation to final results interpretation using the Homodyne Analysis package.

Overview
--------

We'll walk through:

1. **Data Preparation**: Setting up configuration and input data
2. **Quality Control**: Validating experimental data
3. **Analysis Execution**: Running different optimization methods
4. **Results Interpretation**: Understanding outputs and diagnostics
5. **Visualization**: Creating publication-ready plots

Step 1: Data Preparation
------------------------

**Create a Configuration File**

Start by generating a configuration template:

.. code-block:: bash

   # Generate config for laminar flow analysis
   homodyne-config --mode laminar_flow --sample protein_solution \
                   --author "Your Name" --experiment "Flow dynamics study"

This creates a ``config.json`` file with appropriate defaults for 7-parameter laminar flow analysis.

**Prepare Your Data**

Ensure your data is in one of the supported formats:

* **HDF5 files**: From XPCS experiments (requires ``xpcs-viewer``)
* **NPZ files**: Pre-processed correlation data with structure ``(n_phi, n_t1, n_t2)``

The data should contain correlation functions with at least 4 angles (typically 0°, 45°, 90°, 135°).

Step 2: Quality Control
-----------------------

**Validate Experimental Data**

Before running analysis, check your data quality:

.. code-block:: bash

   homodyne --plot-experimental-data --config config.json --verbose

This generates validation plots in ``./homodyne_results/exp_data/``:

* 2D correlation function heatmaps for each angle
* Statistical summaries and quality metrics
* Data range and contrast analysis

**Quality Indicators to Check**:

* Mean values should be around 1.0 for g₂ correlation functions
* Enhanced diagonal values indicating proper correlation structure
* Sufficient contrast (> 0.001) for reliable fitting
* No obvious artifacts or noise patterns

Step 3: Analysis Execution
--------------------------

**Start with Classical Methods (Fast)**

For initial parameter estimation:

.. code-block:: bash

   # Fast classical optimization
   homodyne --method classical --config config.json --verbose

This runs Nelder-Mead and Gurobi (if available) optimizers, typically completing in minutes.

**Add Robust Methods (Noise-Resistant)**

For noisy experimental data:

.. code-block:: bash

   # Robust optimization methods
   homodyne --method robust --config config.json

This includes:
* Wasserstein distributionally robust optimization
* Scenario-based robust optimization  
* Ellipsoidal uncertainty set optimization

**Complete with MCMC (Comprehensive)**

For uncertainty quantification:

.. code-block:: bash

   # Bayesian MCMC sampling
   homodyne --method mcmc --config config.json --verbose

This provides parameter uncertainties and convergence diagnostics.

**Run All Methods Together**

For comprehensive analysis:

.. code-block:: bash

   # Complete analysis with all methods
   homodyne --method all --config config.json --verbose

Step 4: Results Interpretation
------------------------------

**Understanding Output Structure**

Results are organized by method:

.. code-block:: text

   ./homodyne_results/
   ├── homodyne_analysis_results.json    # Summary of all methods
   ├── classical/
   │   ├── nelder_mead/
   │   │   ├── parameters.json           # Human-readable parameters
   │   │   ├── fitted_data.npz          # Complete numerical data
   │   │   └── c2_heatmaps_*.png        # Correlation visualizations
   │   └── gurobi/                      # (if available)
   ├── robust/
   │   ├── wasserstein/
   │   ├── scenario/
   │   └── ellipsoidal/
   ├── mcmc/
   │   ├── mcmc_summary.json
   │   ├── trace_plot.png               # MCMC diagnostics
   │   └── corner_plot.png              # Parameter posteriors
   └── diagnostic_summary.png           # Cross-method comparison

**Key Parameters to Examine**

For **laminar flow mode** (7 parameters):

.. list-table::
   :header-rows: 1
   :widths: 20 25 55

   * - Parameter
     - Physical Meaning  
     - Typical Values
   * - ``D0``
     - Diffusion coefficient
     - 10-100,000 Ų/s
   * - ``alpha``
     - Time scaling exponent
     - -2.0 to 2.0
   * - ``D_offset``
     - Baseline diffusion
     - Small compared to D0
   * - ``gamma_dot_0``
     - Shear rate amplitude
     - 0.001-1.0 s⁻¹
   * - ``beta``
     - Shear scaling exponent
     - -2.0 to 2.0
   * - ``gamma_dot_offset``
     - Baseline shear
     - Small compared to γ̇₀
   * - ``phi0``
     - Angular offset
     - -10 to 10 degrees

**Quality Metrics**

* **Chi-squared (χ²)**: Lower values indicate better fits
* **Reduced χ²**: Should be close to 1.0 for good fits
* **R̂ values** (MCMC): Should be < 1.1 for convergence
* **Parameter uncertainties**: Smaller uncertainties indicate more reliable estimates

Step 5: Visualization and Publication
-------------------------------------

**Generate Correlation Heatmaps**

High-quality visualizations are automatically created:

.. code-block:: bash

   # Force regeneration of heatmaps
   homodyne --method classical --plot-c2-heatmaps --config config.json

**Create Custom Visualizations**

Use the fitted data for custom plots:

.. code-block:: python

   import numpy as np
   import matplotlib.pyplot as plt

   # Load fitted data from best method
   data = np.load("./homodyne_results/classical/nelder_mead/fitted_data.npz")
   
   c2_experimental = data["c2_experimental"] 
   c2_fitted = data["c2_fitted"]
   phi_angles = data["phi_angles"]
   t1, t2 = data["t1"], data["t2"]

   # Create publication-quality plots
   fig, axes = plt.subplots(2, 2, figsize=(12, 10))
   
   for i, phi in enumerate(phi_angles):
       ax = axes.flat[i]
       im = ax.imshow(c2_fitted[i], extent=[t1[0], t1[-1], t2[0], t2[-1]],
                     origin='lower', aspect='auto', cmap='viridis')
       ax.set_title(f'φ = {phi:.0f}°')
       ax.set_xlabel('t₁ (s)')
       ax.set_ylabel('t₂ (s)')
       plt.colorbar(im, ax=ax)

   plt.tight_layout()
   plt.savefig('correlation_functions_publication.pdf', dpi=300)

**Parameter Evolution Analysis**

For time-resolved studies:

.. code-block:: python

   # Extract time-dependent functions
   parameters = data["parameters"]
   D0, alpha, D_offset = parameters[:3]  # First three parameters
   
   # Calculate D(t) evolution  
   t = np.linspace(0, 1, 100)
   D_t = D0 * t**alpha + D_offset
   
   plt.figure(figsize=(8, 6))
   plt.plot(t, D_t, 'b-', linewidth=2, label='D(t)')
   plt.xlabel('Time (s)')
   plt.ylabel('Diffusion Coefficient (Ų/s)')
   plt.title('Time-Dependent Diffusion')
   plt.legend()
   plt.grid(True, alpha=0.3)
   plt.savefig('diffusion_evolution.pdf', dpi=300)

Best Practices and Tips
-----------------------

**Method Selection Strategy**

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - Data Quality
     - Recommended Method
     - Reasoning
   * - High SNR, clean data
     - Classical → MCMC
     - Fast + uncertainty quantification
   * - Moderate noise
     - Robust → MCMC  
     - Noise resistance + uncertainties
   * - High noise, outliers
     - Robust only
     - Maximum noise resistance
   * - Exploratory analysis
     - Classical
     - Speed for parameter screening

**Performance Optimization**

.. code-block:: bash

   # Set threading for reproducible performance
   export OMP_NUM_THREADS=4
   export NUMBA_DISABLE_INTEL_SVML=1

   # Enable GPU acceleration (Linux + NVIDIA GPU)
   homodyne-post-install --gpu
   
   # Run with GPU acceleration
   homodyne --method mcmc --config config.json

**Troubleshooting Common Issues**

* **Convergence Problems**: Try different initial parameter guesses or reduce parameter bounds
* **Poor Fits**: Check data quality, consider different analysis modes
* **Slow MCMC**: Reduce number of samples or enable GPU acceleration
* **Memory Issues**: Use smaller datasets or increase system memory

**Publication Checklist**

- [ ] Data quality validation completed
- [ ] Appropriate analysis mode selected (static vs laminar flow)
- [ ] Multiple optimization methods compared
- [ ] Parameter uncertainties quantified (MCMC)
- [ ] Convergence diagnostics checked (R̂ < 1.1)
- [ ] Physical parameter ranges verified
- [ ] High-resolution figures generated (300+ DPI)
- [ ] Method details documented for reproducibility

Conclusion
----------

This complete workflow provides a systematic approach to homodyne scattering analysis:

1. **Quality Control**: Always validate data before analysis
2. **Method Comparison**: Use multiple optimization approaches
3. **Uncertainty Quantification**: Include MCMC for rigorous error analysis  
4. **Physical Validation**: Ensure parameters have reasonable values
5. **Visualization**: Create clear, publication-ready figures

Following these steps ensures robust, reproducible analysis of nonequilibrium transport phenomena in soft matter systems.

For more detailed information on specific topics, see:

* :doc:`installation` - Complete installation guide
* :doc:`configuration-guide` - Configuration file details
* :doc:`analysis-modes` - Analysis mode selection
* :doc:`plotting` - Advanced visualization options
* :doc:`../developer-guide/troubleshooting` - Troubleshooting guide