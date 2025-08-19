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
         "target_accept": 0.9
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
               
           if D_offset < 0:
               print(f"⚠️ D_offset = {D_offset} should be non-negative")

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

Next Steps
----------

- Explore the :doc:`../api-reference/core` for advanced programmatic usage
- Review :doc:`../developer-guide/performance` for optimization strategies
- Check :doc:`../developer-guide/troubleshooting` if you encounter issues