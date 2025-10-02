=======================
Data Visualization
=======================

The homodyne package provides comprehensive plotting capabilities for both experimental and simulated C₂ correlation function data. This guide covers the visualization tools available for data validation, analysis results, and theoretical predictions.

.. contents::
   :local:
   :depth: 2

Overview
========

The plotting functionality in homodyne supports two main visualization workflows:

1. **Experimental Data Visualization**: Validate and inspect experimental correlation data
2. **Simulated Data Visualization**: Display theoretical predictions and fitted results with scaling transformations

Both workflows generate publication-quality heatmaps with detailed statistical analysis and customizable parameters.

Experimental Data Plotting
===========================

The ``--plot-experimental-data`` functionality allows you to visualize experimental correlation data without performing fitting, making it ideal for data quality assessment and validation.

Basic Usage
-----------

.. code-block:: bash

   # Basic experimental data plotting
   homodyne --plot-experimental-data --config experiment.json

   # With verbose output for debugging
   homodyne --plot-experimental-data --config experiment.json --verbose

   # Quiet mode for batch processing
   homodyne --plot-experimental-data --config experiment.json --quiet

Supported Data Formats
----------------------

HDF5 Files
^^^^^^^^^^

The primary format for experimental XPCS data using the PyXPCS viewer library:

.. code-block:: json

   {
     "experimental_data": {
       "data_file_name": "your_experiment.hdf",
       "exchange_key": "exchange",
       "data_folder_path": "./data/experiment/"
     }
   }

**Features:**
- Automatic loading via PyXPCS viewer
- Exchange key specification for HDF5 structure navigation
- Support for multi-angle experimental datasets

NPZ Files
^^^^^^^^^

Pre-processed correlation data in NumPy compressed format:

**File Structure:**
- ``c2_data``: 3D array with shape ``(n_phi, n_t1, n_t2)``
- ``t1``, ``t2``: Time coordinate arrays
- ``phi_angles``: Angular coordinates in degrees

.. code-block:: python

   import numpy as np

   # Example NPZ file structure
   data = np.load("experimental_c2_data.npz")
   c2_data = data["c2_data"]          # Shape: (4, 60, 60) for 4 angles
   t1 = data["t1"]                    # Time array for first correlation axis
   t2 = data["t2"]                    # Time array for second correlation axis
   phi_angles = data["phi_angles"]    # Angular coordinates [0, 45, 90, 135]

Output Structure
----------------

Experimental data plots are saved to ``./homodyne_results/exp_data/``:

.. code-block:: text

   ./homodyne_results/exp_data/
   ├── data_validation_phi_0.png      # Phi = 0° analysis
   ├── data_validation_phi_45.png     # Phi = 45° analysis
   ├── data_validation_phi_90.png     # Phi = 90° analysis
   └── data_validation_phi_135.png    # Phi = 135° analysis

**Plot Layout:**
- **2-column layout**: Streamlined visualization
- **Left column**: C₂ heatmap showing ``c₂(t₁, t₂)``
- **Right column**: Statistical summaries and quality metrics
- **Removed elements**: Diagonal and cross-section plots for cleaner presentation

Multi-Angle Handling
--------------------

For experiments with multiple phi angles, each angle is plotted individually:

.. code-block:: bash

   # Automatically handles multiple phi angles
   homodyne --plot-experimental-data --config multi_angle_experiment.json

**Benefits:**
- Individual optimization of color scaling per angle
- Clear visualization of angle-dependent scattering patterns
- Statistical analysis specific to each scattering geometry

Simulated Data Plotting
========================

The ``--plot-simulated-data`` functionality displays theoretical predictions and fitted correlation functions with customizable scaling transformations.

Basic Usage
-----------

.. code-block:: bash

   # Basic simulated data plotting
   homodyne --plot-simulated-data --config theory.json

   # With custom scaling parameters
   homodyne --plot-simulated-data --config theory.json --contrast 0.3 --offset 1.2

   # Override phi angles from command line
   homodyne --plot-simulated-data --config theory.json --phi-angles 0,45,90,135

Scaling Transformation
----------------------

The core feature of simulated data plotting is the scaling transformation:

.. math::

   c_{2,\text{fitted}} = \text{contrast} \times c_{2,\text{theoretical}} + \text{offset}

**Parameters:**
- ``contrast``: Scaling factor (default: 1.0, no scaling)
- ``offset``: Baseline offset (default: 0.0, no offset)
- ``phi-angles``: Comma-separated angles in degrees

**Physical Interpretation:**
- **Contrast**: Accounts for experimental signal strength and instrumental factors
- **Offset**: Baseline correlation level from incoherent scattering and background

Command-Line Arguments
----------------------

Scaling Parameters
^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   # Custom contrast and offset
   --contrast 0.25 --offset 1.1

   # Default values (no scaling)
   --contrast 1.0 --offset 0.0

Phi Angles Override
^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   # Standard angles
   --phi-angles 0,45,90,135

   # High-resolution angular sampling
   --phi-angles 0,15,30,45,60,75,90,105,120,135,150,165

   # Specific angles of interest
   --phi-angles 0,90,180

**Note:** Command-line ``--phi-angles`` overrides config file specifications

Data File Requirements
----------------------

Required NPZ Files
^^^^^^^^^^^^^^^^^^

**Theoretical Data** (``theoretical_c2_data.npz``):
- Contains pure theoretical predictions from model
- Used as input for scaling transformation

**Fitted Data** (``fitted_c2_data.npz``):
- Contains scaled theoretical data matching experimental conditions
- Result of applying scaling transformation

**Array Structure:**

.. code-block:: python

   # Both files should contain:
   {
     "c2_data": array,        # Shape: (n_phi, n_t1, n_t2)
     "t1": array,             # Time coordinate array
     "t2": array,             # Time coordinate array
     "phi_angles": array      # Angular coordinates in degrees
   }

Visualization Features
----------------------

Color Scaling
^^^^^^^^^^^^^

**Individual Angle Optimization:**
- ``vmin = min(c2_data)`` calculated separately for each phi angle
- Maximizes contrast and visibility for each scattering geometry
- Prevents saturation from dominant angular contributions

**Clean Presentation:**
- No grid lines on heatmaps for professional appearance
- Consistent colormap (viridis) across all visualizations
- Publication-quality formatting and labeling

Configuration Integration
=========================

Plotting Configuration
----------------------

The configuration file supports comprehensive plotting settings:

.. code-block:: json

   {
     "output_settings": {
       "plotting": {
         "experimental_data_plotting": {
           "enabled": true,
           "data_sources": {
             "hdf_files": {
               "supported": true,
               "loader": "pyxpcs.viewer",
               "exchange_key": "exchange"
             },
             "npz_files": {
               "supported": true,
               "format": "theoretical_c2_data.npz or fitted_c2_data.npz"
             }
           },
           "plot_layout": {
             "columns": 2,
             "include_diagonal_plots": false,
             "include_crosssection_plots": false
           },
           "multiple_phi_handling": {
             "plot_individually": true
           }
         },
         "simulated_data_plotting": {
           "enabled": true,
           "scaling_transformation": {
             "formula": "c2_fitted = contrast * c2_theoretical + offset",
             "default_contrast": 1.0,
             "default_offset": 0.0
           },
           "phi_angles_input": {
             "command_line_override": true,
             "format": "--phi-angles 0,45,90,135",
             "fallback_to_config": true
           },
           "heatmap_settings": {
             "colormap": "viridis",
             "color_scaling": {
               "per_angle_vmin": true
             },
             "remove_grid": true
           }
         }
       }
     }
   }

Usage Examples
==============

Data Quality Assessment
-----------------------

.. code-block:: bash

   # Quick validation of experimental data
   homodyne --plot-experimental-data --config experiment.json

   # Detailed debugging output
   homodyne --plot-experimental-data --config experiment.json --verbose

**Look for:**
- Correlation values around 1.0-2.0 for proper c₂ functions
- Clear time-dependent decay patterns
- Consistent behavior across different phi angles
- Absence of artifacts or discontinuities

Theoretical Prediction Visualization
------------------------------------

.. code-block:: bash

   # Display raw theoretical predictions
   homodyne --plot-simulated-data --config theory.json

   # Scale predictions to match experimental conditions
   homodyne --plot-simulated-data --config theory.json --contrast 0.3 --offset 1.1

   # Custom angular sampling for detailed analysis
   homodyne --plot-simulated-data --config theory.json --phi-angles 0,30,60,90,120,150

Model Validation Workflow
--------------------------

.. code-block:: bash

   # Step 1: Validate experimental data quality
   homodyne --plot-experimental-data --config experiment.json --verbose

   # Step 2: Generate theoretical predictions
   homodyne --plot-simulated-data --config model.json --contrast 0.25

   # Step 3: Run full analysis for comparison
   homodyne --config combined.json --method classical --plot-c2-heatmaps

**Workflow Benefits:**
- Systematic quality control before analysis
- Visual validation of model predictions
- Direct comparison between theory and experiment
- Publication-ready visualization output

Best Practices
==============

Data Preparation
----------------

1. **Experimental Data:**
   - Ensure proper HDF5 structure with exchange keys
   - Verify time arrays are properly spaced with consistent ``dt``
   - Check phi angle coverage matches experimental geometry

2. **Simulated Data:**
   - Generate theoretical predictions with same time/angle sampling as experiment
   - Use appropriate scaling parameters based on experimental conditions
   - Validate array dimensions match expected structure

Visualization Quality
---------------------

1. **Color Scaling:**
   - Use per-angle vmin optimization for multi-angle datasets
   - Maintain consistent colormap (viridis) across all plots
   - Avoid saturated or unclear visualization ranges

2. **Layout and Presentation:**
   - Utilize 2-column layout for efficient space usage
   - Remove unnecessary grid lines for clean appearance
   - Include proper axis labels and units

3. **Statistical Analysis:**
   - Review quality metrics in statistical summary panels
   - Check for outliers or unexpected patterns
   - Validate correlation function ranges and decay behavior

Error Handling and Troubleshooting
==================================

Common Issues
-------------

**Missing Data Files:**

.. code-block:: bash

   # Error: NPZ files not found
   # Solution: Verify file paths in configuration
   ls -la theoretical_c2_data.npz fitted_c2_data.npz

**Dimension Mismatches:**

.. code-block:: python

   # Check array dimensions
   data = np.load("experimental_data.npz")
   print(f"c2_data shape: {data['c2_data'].shape}")  # Expected: (n_phi, n_t1, n_t2)
   print(f"phi_angles: {data['phi_angles']}")        # Expected: [0, 45, 90, 135]

**Invalid Phi Angles:**

.. code-block:: bash

   # Error: Cannot parse phi angles
   # Solution: Use proper comma-separated format
   --phi-angles 0,45,90,135  # Correct
   --phi-angles "0 45 90 135"  # Incorrect

Performance Considerations
---------------------------

**Large Datasets:**
- Use chunked loading for memory-efficient processing
- Consider subsampling time arrays for initial visualization
- Enable quiet mode for batch processing multiple files

**Multi-Angle Analysis:**
- Parallel processing automatically handles multiple phi angles
- Individual plots allow focused analysis of specific scattering geometries
- Color scaling optimization improves visualization quality

This comprehensive plotting functionality provides researchers with powerful tools for data validation, theoretical prediction visualization, and publication-quality figure generation in XPCS analysis workflows.
