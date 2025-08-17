Data Validation
===============

Comprehensive data validation is crucial for reliable XPCS analysis. The homodyne package provides integrated experimental data validation tools with standalone plotting capabilities.

Overview
--------

Data validation ensures:

- **Data Quality Assessment**: Verify experimental data meets analysis requirements
- **Early Problem Detection**: Identify issues before time-intensive analysis
- **Visual Diagnostics**: Generate comprehensive validation plots
- **Quality Metrics**: Quantitative measures of data suitability

Validation Features
-------------------

Comprehensive Validation Plots
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The validation system generates multiple diagnostic visualizations:

- **Full 2D correlation function heatmaps** g₂(t₁,t₂) for each angle
- **Diagonal slices** g₂(t,t) showing temporal decay
- **Cross-sectional profiles** at different time points
- **Statistical summaries** with data quality metrics

Quality Indicators
~~~~~~~~~~~~~~~~~~

**Expected Characteristics for Good Data**:

- **Mean values** around 1.0 (expected for g₂ correlation functions)
- **Enhanced diagonal values** (should be higher than off-diagonal)
- **Sufficient contrast** (> 0.001) indicating dynamic signal
- **Consistent structure** across different angles
- **Appropriate noise levels** for meaningful analysis

Usage Methods
--------------

Command Line Integration
~~~~~~~~~~~~~~~~~~~~~~~~

**Basic Data Validation**:

.. code-block:: bash

   # Generate data validation plots only
   python run_homodyne.py --plot-experimental-data --config my_config.json

   # Verbose validation with debug logging
   python run_homodyne.py --plot-experimental-data --config my_config.json --verbose

**Combined Validation and Analysis**:

.. code-block:: bash

   # Validate data and perform analysis
   python run_homodyne.py --plot-experimental-data --method all --verbose

   # Mode-specific validation
   python run_homodyne.py --plot-experimental-data --static-isotropic --method classical

Configuration-Based Validation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Enable automatic validation through configuration:

.. code-block:: json

   {
     "workflow_integration": {
       "analysis_workflow": {
         "plot_experimental_data_on_load": true,
         "validate_data_quality": true,
         "save_validation_plots": true
       }
     }
   }

**Configuration Options**:
- **plot_experimental_data_on_load**: Automatically generate plots when loading data
- **validate_data_quality**: Perform quantitative quality checks
- **save_validation_plots**: Save plots to disk for later review

Python API Usage
~~~~~~~~~~~~~~~~~

.. code-block:: python

   from homodyne import HomodyneAnalysisCore, ConfigManager
   from homodyne.plotting import plot_experimental_data_validation
   
   # Load configuration and data
   config = ConfigManager("my_config.json")
   analysis = HomodyneAnalysisCore(config)
   
   # Generate validation plots
   plot_experimental_data_validation(
       config, 
       output_directory="./plots/validation",
       show_plots=True,
       save_plots=True
   )

Output Organization
-------------------

Directory Structure
~~~~~~~~~~~~~~~~~~~

Validation plots are organized in a clear directory structure:

.. code-block:: text

   plots/
   └── data_validation/
       ├── correlation_heatmaps/
       │   ├── angle_000_heatmap.png
       │   ├── angle_001_heatmap.png
       │   └── ...
       ├── diagonal_slices/
       │   ├── diagonal_comparison.png
       │   └── decay_profiles.png
       ├── cross_sections/
       │   ├── t1_sections.png
       │   └── t2_sections.png
       └── summary_statistics.png

Plot Types and Interpretation
-----------------------------

2D Correlation Heatmaps
~~~~~~~~~~~~~~~~~~~~~~~

**Purpose**: Visualize the full g₂(t₁,t₂) correlation matrix for each angle

.. code-block:: python

   # Typical heatmap characteristics
   # - Enhanced diagonal (g₂(t,t) > g₂(t₁,t₂) for t₁ ≠ t₂)
   # - Smooth decay away from diagonal
   # - Consistent structure across angles

**Quality Indicators**:
- **Enhanced diagonal**: Clear enhancement along t₁ = t₂
- **Smooth structure**: No abrupt discontinuities or artifacts
- **Appropriate contrast**: Visible difference between diagonal and off-diagonal
- **Angular consistency**: Similar overall structure across different angles

**Common Issues**:
- **Flat correlation**: No diagonal enhancement (poor dynamics)
- **Excessive noise**: Random fluctuations dominating structure
- **Systematic artifacts**: Regular patterns indicating instrumental issues

Diagonal Slices g₂(t,t)
~~~~~~~~~~~~~~~~~~~~~~~

**Purpose**: Show temporal decay characteristics along the diagonal

.. code-block:: python

   # Expected diagonal behavior
   # g₂(0) > 1 (intercept enhancement)
   # Smooth decay with increasing lag time
   # Plateau at long times near unity

**Quality Indicators**:
- **Intercept value**: g₂(0) should be > 1 for dynamic systems
- **Smooth decay**: Monotonic decrease without oscillations
- **Appropriate time range**: Decay spans available time window
- **Long-time plateau**: Approaches baseline value at long times

**Problematic Patterns**:
- **No intercept enhancement**: g₂(0) ≈ 1 indicates no dynamics
- **Oscillatory decay**: May indicate aliasing or systematic errors
- **No plateau**: Insufficient time range or poor statistics

Cross-Sectional Profiles
~~~~~~~~~~~~~~~~~~~~~~~~

**Purpose**: Examine correlation behavior at fixed time points

.. code-block:: python

   # Cross-sections at different t₁ values
   # Show how correlation evolves with t₂
   # Reveal time-dependent behavior

**Analysis Points**:
- **Peak positions**: Should align with diagonal
- **Peak widths**: Related to correlation time scales
- **Baseline levels**: Off-diagonal correlation values
- **Symmetry**: g₂(t₁,t₂) = g₂(t₂,t₁) for stationary processes

Statistical Summary Plots
~~~~~~~~~~~~~~~~~~~~~~~~~

**Purpose**: Quantitative quality metrics and statistics

.. code-block:: python

   # Key statistics displayed:
   # - Mean correlation values
   # - Standard deviations
   # - Dynamic range and contrast
   # - Signal-to-noise estimates

**Quality Metrics**:
- **Mean values**: Should be near 1.0 for normalized data
- **Contrast**: (max - min) / mean should be > 0.001
- **Signal-to-noise**: Ratio of signal to statistical fluctuations
- **Coverage**: Fraction of data points with adequate statistics

Data Quality Assessment
-----------------------

Automated Quality Checks
~~~~~~~~~~~~~~~~~~~~~~~~

The validation system performs automated quality assessment:

.. code-block:: python

   def assess_data_quality(correlation_data):
       """Automated data quality assessment."""
       
       # Calculate key metrics
       mean_value = np.mean(correlation_data)
       diagonal_enhancement = np.mean(np.diag(correlation_data)) - mean_value
       contrast = (np.max(correlation_data) - np.min(correlation_data)) / mean_value
       
       # Quality flags
       quality_flags = {
           'mean_near_unity': 0.8 <= mean_value <= 1.2,
           'diagonal_enhanced': diagonal_enhancement > 0.001,
           'sufficient_contrast': contrast > 0.001,
           'no_negative_values': np.min(correlation_data) >= 0
       }
       
       return quality_flags

**Quality Criteria**:
- **Mean near unity**: 0.8 ≤ mean(g₂) ≤ 1.2
- **Diagonal enhancement**: mean(diag(g₂)) - mean(g₂) > 0.001
- **Sufficient contrast**: (max - min)/mean > 0.001
- **Non-negative values**: All correlation values ≥ 0

Quality Reporting
~~~~~~~~~~~~~~~~~

Generate comprehensive quality reports:

.. code-block:: text

   ========================================
   EXPERIMENTAL DATA VALIDATION REPORT
   ========================================
   
   Dataset: protein_sample_01.h5
   Analysis Date: 2024-01-15 10:30:00
   
   OVERALL QUALITY: GOOD ✓
   
   Quality Metrics:
   ├── Mean g₂ value: 1.003 ✓
   ├── Diagonal enhancement: 0.045 ✓
   ├── Contrast ratio: 0.043 ✓
   ├── Signal-to-noise: 15.2 ✓
   └── Data coverage: 98.5% ✓
   
   Angular Analysis:
   ├── Number of angles: 128
   ├── Angle range: 0° - 180°
   ├── Consistent quality: 96% of angles pass QC ✓
   └── Recommended angles: 0°-15°, 165°-180°

Troubleshooting Data Issues
---------------------------

Common Data Problems
~~~~~~~~~~~~~~~~~~~

**Poor Signal Quality**:

.. code-block:: text

   Symptoms:
   - Low diagonal enhancement (< 0.001)
   - High noise levels
   - Inconsistent structure across angles
   
   Possible Causes:
   - Insufficient measurement time
   - Poor sample preparation
   - Instrumental instability
   - Inadequate signal-to-noise ratio

**Systematic Artifacts**:

.. code-block:: text

   Symptoms:
   - Regular patterns in heatmaps
   - Unexpected correlations
   - Angular inconsistencies
   
   Possible Causes:
   - Detector artifacts
   - Beam instability
   - Sample environment issues
   - Data processing errors

**Normalization Issues**:

.. code-block:: text

   Symptoms:
   - Mean values far from unity
   - Negative correlation values
   - Unrealistic scaling
   
   Possible Causes:
   - Incorrect dark current subtraction
   - Improper flat field correction
   - Background subtraction errors
   - Calibration problems

Resolution Strategies
~~~~~~~~~~~~~~~~~~~~

**Data Quality Improvement**:

1. **Increase measurement time** for better statistics
2. **Improve sample stability** and environmental control
3. **Check instrumental calibration** and alignment
4. **Verify data processing pipeline** for systematic errors

**Analysis Adaptations**:

1. **Use angle filtering** to focus on high-quality angular ranges
2. **Adjust time windows** to exclude problematic regions
3. **Apply additional smoothing** for noisy data
4. **Consider simpler analysis modes** for marginal data quality

**Configuration Adjustments**:

.. code-block:: json

   {
     "data_preprocessing": {
       "apply_smoothing": true,
       "smoothing_window": 3,
       "outlier_removal": true,
       "outlier_threshold": 3.0
     },
     "analysis_settings": {
       "enable_angle_filtering": true,
       "angle_filter_ranges": [[-5, 5], [175, 185]]
     }
   }

Integration with Analysis Workflow
----------------------------------

Workflow Recommendations
~~~~~~~~~~~~~~~~~~~~~~~~

**Standard Workflow**:

1. **Data Validation**: Always validate data before analysis
2. **Quality Assessment**: Review validation plots and metrics
3. **Parameter Adjustment**: Modify analysis parameters based on data quality
4. **Analysis Execution**: Proceed with appropriate analysis mode
5. **Result Validation**: Cross-check results with data quality assessment

**Quality-Based Mode Selection**:

.. code-block:: python

   def select_analysis_mode_based_on_quality(quality_metrics):
       """Recommend analysis mode based on data quality."""
       
       if quality_metrics['contrast'] < 0.005:
           return "static_isotropic"  # Fastest for marginal data
       elif quality_metrics['angular_consistency'] < 0.8:
           return "static_isotropic"  # Avoid angular complications
       elif quality_metrics['signal_to_noise'] > 10:
           return "laminar_flow"      # Full analysis for high-quality data
       else:
           return "static_anisotropic"  # Balanced approach

Automated Quality Gating
~~~~~~~~~~~~~~~~~~~~~~~~

Implement quality gates to prevent analysis of poor data:

.. code-block:: python

   def quality_gate_check(data_validation_results):
       """Check if data meets minimum quality standards."""
       
       min_requirements = {
           'mean_value': (0.9, 1.1),
           'contrast': 0.001,
           'diagonal_enhancement': 0.001,
           'coverage': 0.95
       }
       
       for metric, requirement in min_requirements.items():
           if not meets_requirement(data_validation_results[metric], requirement):
               raise DataQualityError(f"Data fails {metric} requirement")
       
       return True

Best Practices
--------------

Validation Workflow
~~~~~~~~~~~~~~~~~~~

**Always validate first**:
- Generate validation plots before investing time in analysis
- Use validation results to guide analysis parameter selection
- Document data quality issues for reproducibility

**Regular monitoring**:
- Compare validation results across different datasets
- Track data quality trends over time
- Maintain quality standards for consistent results

**Integration with analysis**:
- Use validation results to optimize analysis parameters
- Adjust computational resources based on data complexity
- Select appropriate analysis modes based on data quality

Documentation and Reporting
~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Include validation in reports**:
- Document data quality assessment results
- Include key validation plots in publications
- Report any data quality limitations

**Maintain validation records**:
- Save validation plots and reports
- Track data quality metrics over time
- Use for method development and troubleshooting

This comprehensive data validation framework ensures reliable and robust XPCS analysis results.
