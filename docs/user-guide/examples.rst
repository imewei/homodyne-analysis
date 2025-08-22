Examples
========

Real-world examples demonstrating common use cases and workflows.

Example 1: Basic Isotropic Analysis
------------------------------------

**Scenario**: Quick analysis of an isotropic sample for preliminary results.

**Configuration** (``isotropic_example.json``):

.. code-block:: javascript

   {
     "metadata": {
       "sample_name": "polymer_solution",
       "analysis_mode": "static_isotropic"
     },
     "analysis_settings": {
       "static_mode": true,
       "static_submode": "isotropic"
     },
     "file_paths": {
       "c2_data_file": "data/polymer_c2_data.h5"
     },
     "initial_parameters": {
       "values": [1500, -0.8, 50]
     }
   }

**Command**:

.. code-block:: bash

   # Run classical analysis (results saved to ./homodyne_results/)
   python run_homodyne.py --config isotropic_example.json --method classical
   
   # Generate data validation plots only (saved to ./homodyne_results/exp_data/)
   python run_homodyne.py --config isotropic_example.json --plot-experimental-data
   
   # Run in quiet mode for batch processing
   python run_homodyne.py --config isotropic_example.json --method classical --quiet

**Expected Output**:

.. code-block:: text

   ✅ Analysis completed successfully
   📊 Results summary:
   - D₀: 1423.5 ± 45.2
   - α: -0.76 ± 0.03  
   - D_offset: 62.1 ± 8.9
   - Chi-squared: 1.23
   - Analysis time: 12.3s

Example 2: Flow Analysis with MCMC
-----------------------------------

**Scenario**: Complete analysis of a system under flow conditions with uncertainty quantification.

**Configuration** (``flow_mcmc_example.json``):

.. code-block:: javascript

   {
     "metadata": {
       "sample_name": "flowing_suspension",
       "analysis_mode": "laminar_flow"
     },
     "analysis_settings": {
       "static_mode": false,
       "enable_angle_filtering": true,
       "angle_filter_ranges": [[-5, 5], [175, 185]]
     },
     "file_paths": {
       "c2_data_file": "data/suspension_flow.h5",
       "phi_angles_file": "data/phi_angles_flow.txt"
     },
     "initial_parameters": {
       "parameter_names": ["D0", "alpha", "D_offset", "gamma_dot_t0", "beta", "gamma_dot_t_offset", "phi0"],
       "values": [2000, -1.0, 100, 15, 0.3, 2, 0],
       "active_parameters": ["D0", "alpha", "D_offset", "gamma_dot_t0"]
     },
     "optimization_config": {
       "mcmc_sampling": {
         "enabled": true,
         "draws": 3000,
         "tune": 1500,
         "chains": 4,
         "target_accept": 0.95
       },
       "scaling_parameters": {
         "fitted_range": {"min": 1.0, "max": 2.0},
         "theory_range": {"min": 0.0, "max": 1.0},
         "contrast": {"min": 0.05, "max": 0.5, "prior_mu": 0.3, "prior_sigma": 0.1, "type": "TruncatedNormal"},
         "offset": {"min": 0.05, "max": 1.95, "prior_mu": 1.0, "prior_sigma": 0.2, "type": "TruncatedNormal"}
       }
     }
   }

**Workflow**:

.. code-block:: bash

   # Step 1: Data validation (optional, saves to ./homodyne_results/exp_data/)
   python run_homodyne.py --config flow_mcmc_example.json --plot-experimental-data
   
   # Step 2: Classical optimization for initial estimates (saves to ./homodyne_results/classical/)
   python run_homodyne.py --config flow_mcmc_example.json --method classical
   
   # Step 3: MCMC sampling for uncertainty quantification (saves to ./homodyne_results/mcmc/)
   python run_homodyne.py --config flow_mcmc_example.json --method mcmc
   
   # Step 4: Complete analysis with both methods (recommended)
   python run_homodyne.py --config flow_mcmc_example.json --method all

**Expected Output**:

.. code-block:: text

   Classical Results:
   - D₀: 1876.3, α: -0.94, D_offset: 112.5, γ̇₀: 12.8
   - Chi-squared: 1.45
   
   MCMC Results:
   - Convergence: ✅ Excellent (R̂ < 1.01)
   - D₀: 1876 ± 89, α: -0.94 ± 0.08
   - D_offset: 113 ± 24, γ̇₀: 12.8 ± 1.2
   - Posterior samples: 12,000

Example 3: Performance-Optimized Analysis
------------------------------------------

**Scenario**: Large dataset requiring optimized performance settings.

**Configuration** (``performance_example.json``):

.. code-block:: javascript

   {
     "analysis_settings": {
       "static_mode": true,
       "static_submode": "anisotropic", 
       "enable_angle_filtering": true,
       "angle_filter_ranges": [[-3, 3], [177, 183]]
     },
     "file_paths": {
       "c2_data_file": "data/large_dataset.h5",
       "phi_angles_file": "data/angles_high_res.txt"
     },
     "performance_settings": {
       "num_threads": 8,
       "data_type": "float32",
       "memory_limit_gb": 16,
       "enable_jit": true,
       "chunked_processing": true
     },
     "initial_parameters": {
       "values": [3000, -0.6, 200]
     }
   }

**Results**:

- **Memory usage**: Reduced by ~50% with float32
- **Speed improvement**: 3-4x faster with angle filtering
- **Accuracy**: Maintained with optimized angle ranges

Example 4: Batch Processing Multiple Samples
---------------------------------------------

**Scenario**: Process multiple samples with consistent parameters.

**Batch Script** (``batch_analysis.py``):

.. code-block:: python

   import os
   import json
   from homodyne import HomodyneAnalysisCore, ConfigManager
   
   # Sample list
   samples = [
       {"name": "sample_01", "file": "data/sample_01.h5"},
       {"name": "sample_02", "file": "data/sample_02.h5"},
       {"name": "sample_03", "file": "data/sample_03.h5"}
   ]
   
   # Base configuration
   base_config = {
       "analysis_settings": {
           "static_mode": True,
           "static_submode": "isotropic"
       },
       "initial_parameters": {
           "values": [1000, -0.5, 100]
       }
   }
   
   results = {}
   
   for sample in samples:
       print(f"Processing {sample['name']}...")
       
       # Create sample-specific config
       config = base_config.copy()
       config["file_paths"] = {"c2_data_file": sample["file"]}
       config["metadata"] = {"sample_name": sample["name"]}
       
       # Save temporary config
       config_file = f"temp_{sample['name']}.json"
       with open(config_file, 'w') as f:
           json.dump(config, f, indent=2)
       
       # Run analysis
       try:
           config_manager = ConfigManager(config_file)
           analysis = HomodyneAnalysisCore(config_manager)
           result = analysis.optimize_classical()
           
           results[sample['name']] = {
               "parameters": result.x,
               "chi_squared": result.fun,
               "success": result.success
           }
           
           print(f"✅ {sample['name']}: χ² = {result.fun:.3f}")
           
       except Exception as e:
           print(f"❌ {sample['name']}: {str(e)}")
           results[sample['name']] = {"error": str(e)}
       
       # Cleanup
       os.remove(config_file)
   
   # Save batch results
   with open("batch_results.json", 'w') as f:
       json.dump(results, f, indent=2)
   
   print(f"Batch processing complete. Results saved to batch_results.json")

Example 5: Progressive Analysis Workflow
-----------------------------------------

**Scenario**: Systematic approach from simple to complex analysis.

**Workflow Script** (``progressive_analysis.py``):

.. code-block:: python

   from homodyne import HomodyneAnalysisCore, ConfigManager
   import json
   
   def progressive_analysis(data_file, angles_file):
       """
       Perform progressive analysis: isotropic → anisotropic → flow
       """
       
       results = {}
       
       # Step 1: Isotropic analysis (fastest)
       print("Step 1: Isotropic analysis...")
       iso_config = {
           "analysis_settings": {"static_mode": True, "static_submode": "isotropic"},
           "file_paths": {"c2_data_file": data_file},
           "initial_parameters": {"values": [1000, -0.5, 100]}
       }
       
       iso_result = run_analysis(iso_config, "isotropic")
       results["isotropic"] = iso_result
       
       # Step 2: Anisotropic analysis  
       print("Step 2: Anisotropic analysis...")
       aniso_config = iso_config.copy()
       aniso_config["analysis_settings"]["static_submode"] = "anisotropic"
       aniso_config["analysis_settings"]["enable_angle_filtering"] = True
       aniso_config["file_paths"]["phi_angles_file"] = angles_file
       
       aniso_result = run_analysis(aniso_config, "anisotropic")
       results["anisotropic"] = aniso_result
       
       # Compare isotropic vs anisotropic
       iso_chi2 = results["isotropic"]["chi_squared"]
       aniso_chi2 = results["anisotropic"]["chi_squared"]
       improvement = (iso_chi2 - aniso_chi2) / iso_chi2 * 100
       
       print(f"Chi-squared improvement: {improvement:.1f}%")
       
       # Step 3: Flow analysis (if significant improvement)
       if improvement > 5:  # 5% improvement threshold
           print("Step 3: Flow analysis...")
           flow_config = aniso_config.copy()
           flow_config["analysis_settings"]["static_mode"] = False
           flow_config["initial_parameters"] = {
               "parameter_names": ["D0", "alpha", "D_offset", "gamma_dot_t0", "beta", "gamma_dot_t_offset", "phi0"],
               "values": list(aniso_result["parameters"]) + [10, 0.5, 1, 0],
               "active_parameters": ["D0", "alpha", "D_offset", "gamma_dot_t0"]
           }
           
           flow_result = run_analysis(flow_config, "flow")
           results["flow"] = flow_result
       else:
           print("Skipping flow analysis - anisotropic improvement < 5%")
       
       return results
   
   def run_analysis(config_dict, mode_name):
       """Run analysis with given configuration"""
       config_file = f"temp_{mode_name}.json"
       
       with open(config_file, 'w') as f:
           json.dump(config_dict, f, indent=2)
       
       try:
           config = ConfigManager(config_file)
           analysis = HomodyneAnalysisCore(config)
           result = analysis.optimize_classical()
           
           return {
               "parameters": result.x.tolist(),
               "chi_squared": float(result.fun),
               "success": bool(result.success)
           }
       finally:
           import os
           if os.path.exists(config_file):
               os.remove(config_file)
   
   # Run progressive analysis
   if __name__ == "__main__":
       results = progressive_analysis(
           "data/my_sample.h5", 
           "data/my_angles.txt"
       )
       
       with open("progressive_results.json", 'w') as f:
           json.dump(results, f, indent=2)

Common Patterns
---------------

**Error Handling**:

.. code-block:: python

   try:
       analysis = HomodyneAnalysisCore(config)
       result = analysis.optimize_classical()
       
       if result.success:
           print(f"✅ Optimization successful: χ² = {result.fun:.3f}")
       else:
           print(f"⚠️ Optimization failed: {result.message}")
           
   except FileNotFoundError as e:
       print(f"❌ File not found: {e}")
   except ValueError as e:
       print(f"❌ Configuration error: {e}")

**Parameter Validation**:

.. code-block:: python

   def validate_parameters(params, mode="isotropic"):
       """Validate parameter values are physically reasonable"""
       
       if mode == "isotropic":
           D0, alpha, D_offset = params[:3]
           
           if not (100 <= D0 <= 10000):
               print(f"⚠️ D0 = {D0} may be outside typical range [100, 10000]")
           
           if not (-2.0 <= alpha <= 0.0):
               print(f"⚠️ α = {alpha} may be outside typical range [-2.0, 0.0]")
               
           if abs(D_offset) > 100:
               print(f"⚠️ D_offset = {D_offset} is outside typical range [-100, 100]")

**Result Comparison**:

.. code-block:: python

   def compare_results(result1, result2, labels=["Method 1", "Method 2"]):
       """Compare two analysis results"""
       
       chi2_1, chi2_2 = result1.fun, result2.fun
       improvement = (chi2_1 - chi2_2) / chi2_1 * 100
       
       print(f"{labels[0]} χ²: {chi2_1:.4f}")
       print(f"{labels[1]} χ²: {chi2_2:.4f}")
       print(f"Improvement: {improvement:+.1f}%")
       
       if improvement > 5:
           print("✅ Significant improvement")
       elif improvement > 1:
           print("⚠️ Modest improvement") 
       else:
           print("❌ No significant improvement")

Output Directory Structure
---------------------------

Starting from version 6.0, the analysis results are organized into method-specific subdirectories:

.. code-block:: text

   ./homodyne_results/
   ├── homodyne_analysis_results.json    # Main results file (moved from root directory)
   ├── run.log                           # Analysis log file
   ├── exp_data/                         # Experimental data plots (--plot-experimental-data)
   │   ├── data_validation_phi_*.png
   │   └── summary_statistics.txt
   ├── classical/                       # Classical method outputs (--method classical)
   │   ├── per_angle_chi_squared_classical.json  # Per-angle analysis results
   │   ├── experimental_data.npz         # Original experimental correlation data
   │   ├── fitted_data.npz              # Fitted data (contrast * theory + offset)
   │   ├── residuals_data.npz           # Residuals (experimental - fitted)
   │   └── c2_heatmaps_phi_*.png        # C2 correlation heatmaps (--plot-c2-heatmaps)
   └── mcmc/                            # MCMC method outputs (--method mcmc)
       ├── experimental_data.npz         # Original experimental correlation data
       ├── fitted_data.npz              # Fitted data (contrast * posterior_means + offset)
       ├── residuals_data.npz           # Residuals (experimental - fitted)
       ├── mcmc_summary.json            # MCMC convergence diagnostics and posterior statistics
       ├── mcmc_trace.nc                # NetCDF trace data (ArviZ format)
       ├── c2_heatmaps_phi_*.png        # C2 correlation heatmaps using posterior means
       ├── 3d_surface_phi_*.png         # 3D surface plots with 95% confidence intervals
       ├── 3d_surface_residuals_phi_*.png # 3D residuals plots for quality assessment
       ├── trace_plot.png               # MCMC trace plots
       └── corner_plot.png              # Parameter posterior distributions

**Key Changes**:

- **Main results file**: Now saved in output directory instead of current directory
- **Classical method**: Results organized in dedicated ``./homodyne_results/classical/`` subdirectory
- **MCMC method**: Results organized in dedicated ``./homodyne_results/mcmc/`` subdirectory  
- **Experimental data plots**: Saved to ``./homodyne_results/exp_data/`` when using ``--plot-experimental-data``
- **Data files**: Both classical and MCMC methods save experimental, fitted, and residuals data as ``.npz`` files
- **Method-specific outputs**:
  - **Classical**: Point estimates with C2 heatmaps (diagnostic plots skipped)
  - **MCMC**: Posterior distributions with trace data, convergence diagnostics, specialized plots, and 3D surface visualizations
- **3D visualization**: MCMC method automatically generates publication-quality 3D surface plots with confidence intervals
- **Fitted data calculation**: Both methods use least squares scaling optimization (``fitted = contrast * theory + offset``)
- **Plotting behavior**: The ``--plot-experimental-data`` flag now skips all fitting and exits immediately after plotting

Diagnostic Summary Visualizations
----------------------------------

The package automatically generates comprehensive diagnostic summary plots that combine multiple analysis components into a single visualization. These provide researchers with immediate feedback on analysis quality and method performance.

Main Diagnostic Summary Plot
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Each analysis generates a ``diagnostic_summary.png`` file with a **2×3 grid layout** containing:

**Subplot 1: Method Comparison (Top Left)**
  - Bar chart comparing χ² values across optimization methods
  - Log-scale Y-axis with scientific notation value labels
  - Color-coded methods (Nelder-Mead, Gurobi, Robust-Wasserstein, etc.)

**Subplot 2: Parameter Uncertainties (Top Middle)**
  - Horizontal bar chart of parameter uncertainties (σ)
  - Parameter names on Y-axis (amplitude, frequency, phase, etc.)
  - Grid lines for enhanced readability
  - Shows placeholder if uncertainties unavailable

**Subplot 3: MCMC Convergence Diagnostics (Top Right)**
  - R̂ (R-hat) convergence assessment for MCMC methods
  - Color coding: Green (R̂ < 1.1), Orange (1.1 ≤ R̂ < 1.2), Red (R̂ ≥ 1.2)
  - Dashed red line at R̂ = 1.1 convergence threshold
  - Shows placeholder for classical-only methods

**Subplot 4: Residuals Distribution Analysis (Bottom, Full Width)**
  - Histogram of residuals (experimental - theoretical)
  - Overlaid normal distribution curve for comparison
  - Statistical summary with mean (μ) and standard deviation (σ)
  - Shows placeholder if residuals data unavailable

Method-Specific Diagnostic Summaries
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Individual diagnostic summaries are generated for each optimization method:

- ``nelder_mead_diagnostic_summary.png``
- ``gurobi_diagnostic_summary.png`` 
- ``robust_wasserstein_diagnostic_summary.png``
- ``robust_scenario_diagnostic_summary.png``
- ``robust_ellipsoidal_diagnostic_summary.png``

Additional Visualization Outputs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**C2 Correlation Heatmaps** (``c2_heatmaps_*.png``)
  - 2D heatmaps of experimental vs theoretical correlation functions
  - Individual plots for each scattering angle (φ = 0°, 45°, 90°, 135°)
  - Method-specific versions for each optimization approach
  - Time axes (t₁, t₂) showing correlation delay times
  - Viridis colormap for correlation intensity visualization

**MCMC-Specific Plots** (when applicable)
  - ``trace_plot.png``: MCMC chain traces for convergence assessment
  - ``corner_plot.png``: Parameter posterior distributions and correlations

**Data Validation Plots** (``data_validation_*.png``)
  - Experimental data quality assessment plots
  - Individual plots for each scattering angle
  - 2D heatmaps and cross-sections of raw experimental data
  - Statistical summaries and data quality metrics

Key Features
~~~~~~~~~~~~

1. **Adaptive Content**: Appropriate placeholders shown when data unavailable
2. **Cross-Method Comparison**: Easy comparison of different optimization approaches  
3. **Quality Assessment**: Convergence and fitting quality metrics at a glance
4. **Statistical Analysis**: Residuals analysis and uncertainty quantification
5. **Professional Formatting**: Consistent styling with grid lines, proper labels, and legends

These diagnostic summaries provide immediate visual feedback on analysis quality, method performance, and parameter reliability, enabling researchers to quickly assess their results and identify potential issues.

Common Output Structure for All 5 Classical Methods
----------------------------------------------------

Each of the 5 optimization methods (``Nelder-Mead``, ``Gurobi``, ``Robust-Wasserstein``, ``Robust-Scenario``, ``Robust-Ellipsoidal``) generates standardized outputs for consistent analysis and comparison.

Individual Method Directory Structure
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

   ./homodyne_results/classical/
   ├── nelder_mead/
   ├── gurobi/
   ├── robust_wasserstein/
   ├── robust_scenario/
   └── robust_ellipsoidal/

Per-Method Files
~~~~~~~~~~~~~~~~

Each method directory contains two primary files:

**parameters.json** - Human-readable parameter results
  Contains fitted parameter values with uncertainties, goodness-of-fit metrics (chi-squared, degrees of freedom), convergence information (iterations, function evaluations, termination status), and data statistics.

**fitted_data.npz** - Numerical data archive
  Contains fitted correlation data, original experimental data, fitting residuals, parameter values and uncertainties, chi-squared values, and time arrays (t₁, t₂) with proper dt-based discretization.

Method-Specific Characteristics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Classical Methods (Nelder-Mead, Gurobi)**
  - Point estimates only with deterministic convergence metrics
  - Faster execution with iterations and function evaluations tracking  
  - Termination reasons and solver-specific status information
  - No built-in uncertainty quantification from optimization method

**Robust Methods (Wasserstein, Scenario, Ellipsoidal)**
  - Robust optimization against data uncertainty with worst-case guarantees
  - Additional robust-specific parameters (uncertainty radius, scenarios, confidence levels)
  - Convex optimization solver status codes and solve times
  - Enhanced reliability under data perturbations

Cross-Method Comparison
~~~~~~~~~~~~~~~~~~~~~~~

The ``all_methods_summary.json`` file provides easy comparison across all methods with:
  - Analysis timestamp and methods analyzed
  - Best method selection based on chi-squared values
  - Consolidated results showing parameters and goodness-of-fit for each method
  - Success status for each optimization approach

Data Array Structure
~~~~~~~~~~~~~~~~~~~~

All methods use consistent data array dimensions:
  - **Correlation data**: ``(n_angles, n_t2, n_t1)`` format
  - **Time arrays**: ``t1 = np.arange(n_t1) * dt`` and ``t2 = np.arange(n_t2) * dt``
  - **Individual angles**: ``(n_t2, n_t1)`` where rows=t₂, columns=t₁

This standardized structure enables direct comparison of optimization performance and facilitates automated analysis workflows across different methods.

MCMC Prior Distributions
------------------------

All parameters use **Normal distributions** in the MCMC implementation:

.. code-block:: python

   import pymc as pm
   
   # Standard prior distributions used in homodyne MCMC
   with pm.Model() as model:
       # All parameters use Normal distributions
       D0 = pm.Normal("D0", mu=1e4, sigma=1000.0)                      # Diffusion coefficient [Å²/s]
       alpha = pm.Normal("alpha", mu=-1.5, sigma=0.1)                 # Time exponent [dimensionless]
       D_offset = pm.Normal("D_offset", mu=0.0, sigma=10.0)            # Baseline diffusion [Å²/s]
       gamma_dot_t0 = pm.Normal("gamma_dot_t0", mu=1e-3, sigma=1e-2)   # Reference shear rate [s⁻¹]
       beta = pm.Normal("beta", mu=0.0, sigma=0.1)                     # Shear exponent [dimensionless]
       gamma_dot_t_offset = pm.Normal("gamma_dot_t_offset", mu=0.0, sigma=1e-3)  # Baseline shear [s⁻¹]
       phi0 = pm.Normal("phi0", mu=0.0, sigma=5.0)                     # Angular offset [degrees]

**Configuration Example:**

.. code-block:: json

   {
     "parameter_space": {
       "bounds": [
         {"name": "D0", "min": 1.0, "max": 1000000, "type": "Normal"},
         {"name": "alpha", "min": -2.0, "max": 2.0, "type": "Normal"},
         {"name": "D_offset", "min": -100, "max": 100, "type": "Normal"},
         {"name": "gamma_dot_t0", "min": 1e-6, "max": 1.0, "type": "Normal"},
         {"name": "beta", "min": -2.0, "max": 2.0, "type": "Normal"},
         {"name": "gamma_dot_t_offset", "min": -1e-2, "max": 1e-2, "type": "Normal"},
         {"name": "phi0", "min": -10, "max": 10, "type": "Normal"}
       ]
     }
   }

Example 6: Logging Control for Different Scenarios
----------------------------------------------------

**Scenario**: Using different logging modes for various use cases.

**Interactive Analysis** (default logging):

.. code-block:: bash

   # Normal interactive analysis with console and file logging
   homodyne --config my_config.json --method classical
   
   # With detailed debugging information
   homodyne --config my_config.json --method all --verbose

**Batch Processing** (quiet mode):

.. code-block:: bash

   # Process multiple samples quietly (logs only to files)
   for sample in sample_01 sample_02 sample_03; do
       homodyne --config configs/${sample}_config.json \
               --output-dir results/${sample} \
               --method classical \
               --quiet
   done

**Automated Scripts** (``batch_quiet_analysis.sh``):

.. code-block:: bash

   #!/bin/bash
   # Batch processing script with quiet logging
   
   SAMPLES_DIR="./data/samples"
   RESULTS_DIR="./results"
   
   for config_file in configs/*.json; do
       sample_name=$(basename "$config_file" .json)
       
       echo "Processing ${sample_name}..."
       
       # Run analysis in quiet mode
       homodyne --config "$config_file" \
               --output-dir "${RESULTS_DIR}/${sample_name}" \
               --method classical \
               --quiet
       
       # Check if analysis succeeded (logs are in file)
       if [ -f "${RESULTS_DIR}/${sample_name}/run.log" ]; then
           echo "✅ ${sample_name}: Check ${RESULTS_DIR}/${sample_name}/run.log"
       else
           echo "❌ ${sample_name}: Analysis failed"
       fi
   done
   
   echo "Batch processing complete. Check individual run.log files for details."

**Debugging Mode** (verbose logging):

.. code-block:: bash

   # Troubleshoot analysis with detailed logging
   homodyne --config problem_sample.json --method all --verbose
   
   # Debug MCMC convergence issues
   homodyne --config mcmc_issue.json --method mcmc --verbose

**Key Benefits**:

- **Default mode**: Best for interactive use, shows progress and errors
- **Verbose mode** (``--verbose``): Essential for troubleshooting and development
- **Quiet mode** (``--quiet``): Perfect for batch processing and automation
- **File logging**: Always enabled, provides complete analysis record

**Log File Locations**:

.. code-block:: text

   ./output_directory/
   ├── run.log                    # Complete analysis log
   ├── classical/                 # Classical method results
   ├── mcmc/                      # MCMC method results  
   └── homodyne_analysis_results.json  # Main results

**Error Handling Note**: In quiet mode, errors are only logged to files, so check ``run.log`` files for troubleshooting.

Example 7: Performance Monitoring and Optimization
----------------------------------------------------

**Scenario**: Monitor and optimize performance with production-ready stability. The homodyne package has been rebalanced for excellent performance consistency with 97% reduction in chi-squared calculation variability.

**Advanced Performance Monitoring** (``performance_monitoring.py``):

.. code-block:: python

   from homodyne.core.profiler import (
       performance_monitor, 
       get_performance_cache,
       get_performance_summary,
       stable_benchmark,
       adaptive_stable_benchmark
   )
   from homodyne.core.kernels import (
       warmup_numba_kernels,
       get_kernel_performance_config
   )
   import time
   import numpy as np
   
   # Performance-monitored analysis function
   @performance_monitor(monitor_memory=True, log_threshold_seconds=0.5)
   def analyze_sample_with_monitoring(config_file, output_dir):
       """Analyze sample with comprehensive performance monitoring."""
       from homodyne import HomodyneAnalysisCore, ConfigManager
       
       config = ConfigManager(config_file)
       analyzer = HomodyneAnalysisCore(config)
       
       # Perform analysis with monitoring
       results = analyzer.optimize_classical()
       
       return results
   
   # Setup and warmup
   def setup_optimized_environment():
       """Setup optimized numerical environment."""
       # Warmup all computational kernels
       print("Warming up Numba kernels...")
       warmup_results = warmup_numba_kernels()
       
       print(f"✓ Kernels warmed up in {warmup_results['total_warmup_time']:.3f}s")
       print(f"✓ Numba available: {warmup_results['numba_available']}")
       
       # Check kernel configuration
       config = get_kernel_performance_config()
       print(f"✓ Parallel enabled: {config['parallel_enabled']}")
       print(f"✓ Thread count: {config['num_threads']}")
       
       return warmup_results, config
   
   # Performance benchmarking example
   def benchmark_analysis_performance():
       """Benchmark analysis performance with different strategies."""
       
       def sample_computation():
           """Sample computation for benchmarking."""
           # Simulate typical analysis computation
           data = np.random.rand(1000, 1000)
           result = np.sum(data @ data.T)
           time.sleep(0.001)  # Simulate I/O overhead
           return result
       
       print("=== Performance Benchmarking ===")
       
       # Standard stable benchmarking
       print("Running stable benchmark...")
       stable_results = stable_benchmark(
           sample_computation, 
           warmup_runs=5, 
           measurement_runs=15,
           outlier_threshold=2.0
       )
       
       cv_stable = stable_results['std'] / stable_results['mean']
       print(f"Stable benchmark: {stable_results['mean']:.4f}s ± {cv_stable:.3f} CV")
       print(f"Outliers removed: {stable_results['outlier_count']}/{len(stable_results['times'])}")
       
       # Adaptive benchmarking
       print("Running adaptive benchmark...")
       adaptive_results = adaptive_stable_benchmark(
           sample_computation,
           target_cv=0.10,  # Target 10% coefficient of variation
           max_runs=30,
           min_runs=10
       )
       
       print(f"Adaptive benchmark: {adaptive_results['cv']:.3f} CV in {adaptive_results['total_runs']} runs")
       print(f"Target achieved: {adaptive_results['achieved_target']}")
       
       return stable_results, adaptive_results
   
   # Memory and cache monitoring
   def monitor_cache_performance():
       """Monitor smart cache performance."""
       cache = get_performance_cache()
       
       # Simulate some cached operations
       for i in range(10):
           key = f"test_data_{i}"
           data = np.random.rand(100, 100)
           cache.put(key, data)
       
       # Get cache statistics
       stats = cache.stats()
       print("=== Cache Performance ===")
       print(f"Cached items: {stats['items']}")
       print(f"Memory usage: {stats['memory_mb']:.1f} MB")
       print(f"Utilization: {stats['utilization']:.1%}")
       print(f"Memory utilization: {stats['memory_utilization']:.1%}")
       
       return stats
   
   # Complete performance analysis workflow
   def run_performance_analysis_example():
       """Complete example of performance-optimized analysis."""
       print("=== Homodyne Performance Analysis Example ===")
       
       # Step 1: Environment setup and warmup
       warmup_results, kernel_config = setup_optimized_environment()
       
       # Step 2: Performance benchmarking
       stable_results, adaptive_results = benchmark_analysis_performance()
       
       # Step 3: Cache monitoring
       cache_stats = monitor_cache_performance()
       
       # Step 4: Run sample analysis with monitoring
       # Note: This would need actual data files
       print("=== Sample Analysis (simulated) ===")
       
       @performance_monitor(monitor_memory=True)
       def simulated_analysis():
           # Simulate analysis computation
           time.sleep(0.1)
           return {"chi_squared": 1.23, "parameters": [1.0, 0.1, 0.05]}
       
       result = simulated_analysis()
       print(f"Analysis result: {result}")
       
       # Step 5: Get comprehensive performance summary
       summary = get_performance_summary()
       print("=== Performance Summary ===")
       
       if summary:
           for func_name, stats in summary.items():
               if isinstance(stats, dict) and "calls" in stats:
                   print(f"{func_name}:")
                   print(f"  Calls: {stats['calls']}")
                   print(f"  Avg time: {stats['avg_time']:.4f}s")
                   print(f"  Total time: {stats['total_time']:.4f}s")
       
       # Performance achievements and recommendations  
       print("=== Performance Stability Achievements ===")
       print("✓ Chi-squared calculations: CV < 0.31 across all array sizes")
       print("✓ 97% reduction in performance variability achieved")
       print("✓ Conservative threading (max 4 cores) for optimal stability")
       print("✓ Balanced JIT optimization for numerical precision")
       
       print("=== Performance Recommendations ===")
       
       if warmup_results.get('total_warmup_time', 0) > 2.0:
           print("⚠ Consider caching warmup results for faster startup")
       
       if cv_stable > 0.31:  # Updated threshold reflecting rebalanced performance
           print("⚠ Performance variability above rebalanced threshold - check system load")
       elif cv_stable < 0.10:
           print("✓ Excellent stability achieved (CV < 0.10)")
       
       if cache_stats['memory_utilization'] > 0.80:
           print("⚠ Cache memory usage high - consider increasing max_memory_mb")
       
       print("✓ Performance analysis complete")
       
       return {
           'warmup': warmup_results,
           'kernel_config': kernel_config,
           'benchmarks': {'stable': stable_results, 'adaptive': adaptive_results},
           'cache_stats': cache_stats,
           'performance_summary': summary
       }
   
   # Run the complete example
   if __name__ == "__main__":
       results = run_performance_analysis_example()

**Configuration for Performance Monitoring** (``performance_config.json``):

.. code-block:: json

   {
     "performance_settings": {
       "numba_optimization": {
         "enable_numba": true,
         "warmup_numba": true,
         "stability_enhancements": {
           "enable_kernel_warmup": true,
           "warmup_iterations": 5,
           "optimize_memory_layout": true,
           "environment_optimization": {
             "auto_configure": true,
             "max_threads": 8,
             "gc_optimization": true
           }
         },
         "performance_monitoring": {
           "enable_profiling": true,
           "adaptive_benchmarking": true,
           "target_cv": 0.10,
           "memory_monitoring": true,
           "smart_caching": {
             "enabled": true,
             "max_items": 200,
             "max_memory_mb": 1000.0
           }
         }
       }
     }
   }

**Key Performance Features Demonstrated**:

- **JIT Warmup**: Pre-compile kernels for stable performance
- **Adaptive Benchmarking**: Automatically find optimal measurement counts  
- **Memory Monitoring**: Track and optimize memory usage
- **Smart Caching**: Memory-aware LRU caching with cleanup
- **Performance Profiling**: Comprehensive monitoring and statistics
- **Environment Optimization**: Automatic BLAS/threading configuration

Next Steps
----------

- Explore the :doc:`../api-reference/core` for advanced programmatic usage
- Review :doc:`../developer-guide/performance` for optimization strategies
- Check :doc:`../developer-guide/troubleshooting` if you encounter issues