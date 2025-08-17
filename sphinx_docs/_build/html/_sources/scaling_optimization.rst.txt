Scaling Optimization
====================

**Scaling optimization is now always enabled** across all analysis modes for scientifically accurate results. This critical feature ensures meaningful chi-squared statistics and proper fitting between experimental and theoretical correlation functions.

Mathematical Foundation
-----------------------

Relationship Between Functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The scaling optimization determines the optimal relationship between experimental and theoretical correlation functions:

.. math::

   g_2 = \text{offset} + \text{contrast} \times g_1

where:
- **g₁**: Theoretical first-order correlation function
- **g₂**: Experimental second-order correlation function  
- **contrast**: Fitted scaling parameter (multiplicative factor)
- **offset**: Fitted baseline parameter (additive factor)

Physical Significance
~~~~~~~~~~~~~~~~~~~~~

Scaling optimization accounts for systematic factors present in experimental data that are not captured by the theoretical model:

**Instrumental Response Functions**:
- Detector response characteristics
- Optical system transfer functions
- Beam profile effects

**Background Signals**:
- Electronic noise and dark current
- Scattered light from sample environment
- Systematic baseline offsets

**Detector Variations**:
- Pixel-to-pixel sensitivity differences
- Gain variations across detector area
- Non-uniform quantum efficiency

**Normalization Differences**:
- Data processing and reduction variations
- Calibration factor differences
- Systematic scaling in measurement chain

Implementation Details
----------------------

Least Squares Solution
~~~~~~~~~~~~~~~~~~~~~~~

The optimal scaling parameters are determined using a least squares approach:

.. code-block:: python

   # Mathematical implementation
   A = np.vstack([theory, np.ones(len(theory))]).T
   scaling, residuals, rank, singular_values = np.linalg.lstsq(A, experimental, rcond=None)
   contrast, offset = scaling
   
   # Apply scaling to get fitted function
   fitted = theory * contrast + offset

This provides the optimal linear transformation that minimizes:

.. math::

   \chi^2 = \sum_i \frac{(\text{experimental}_i - \text{fitted}_i)^2}{\sigma_i^2}

Matrix Formulation
~~~~~~~~~~~~~~~~~~

The scaling optimization can be expressed in matrix form:

.. math::

   \begin{bmatrix}
   g_{1}(t_1) & 1 \\
   g_{1}(t_2) & 1 \\
   \vdots & \vdots \\
   g_{1}(t_n) & 1
   \end{bmatrix}
   \begin{bmatrix}
   \text{contrast} \\
   \text{offset}
   \end{bmatrix}
   =
   \begin{bmatrix}
   g_{2}(t_1) \\
   g_{2}(t_2) \\
   \vdots \\
   g_{2}(t_n)
   \end{bmatrix}

Scientific Benefits
-------------------

Meaningful Chi-Squared Statistics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Without scaling optimization**, chi-squared values are dominated by systematic offsets and scaling differences, making it impossible to assess the quality of the physics model:

.. math::

   \chi^2_{\text{raw}} = \sum_i \frac{(g_{2,\text{exp}}(i) - g_{1,\text{theory}}(i))^2}{\sigma_i^2}

This typically yields artificially large chi-squared values that don't reflect model adequacy.

**With scaling optimization**, chi-squared reflects the actual quality of the physics model:

.. math::

   \chi^2_{\text{scaled}} = \sum_i \frac{(g_{2,\text{exp}}(i) - [\text{offset} + \text{contrast} \times g_{1,\text{theory}}(i)])^2}{\sigma_i^2}

This provides scientifically meaningful goodness-of-fit statistics.

Model Comparison
~~~~~~~~~~~~~~~~

Scaling optimization enables proper model comparison by ensuring that:
- Different models are compared on equal footing
- Systematic experimental factors don't bias model selection
- Chi-squared differences reflect true model performance
- Parameter uncertainties are properly estimated

Physical Parameter Extraction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The scaling parameters themselves can provide physical insights:

**Contrast Parameter**:
- Related to the coherence of the scattering
- Indicates signal-to-noise ratio
- Reflects experimental conditions quality

**Offset Parameter**:
- Baseline correlation level
- Background contribution
- Systematic experimental offsets

Integration with Analysis Modes
-------------------------------

Static Isotropic Mode
~~~~~~~~~~~~~~~~~~~~~

In isotropic mode, scaling optimization:
- Uses the single theoretical g₁ function
- Applies uniform scaling across all data points
- Provides rapid, accurate parameter estimation

.. code-block:: python

   # Isotropic scaling implementation
   theory_g1 = compute_g1_isotropic(params, time_points)
   contrast, offset = optimize_scaling(experimental_g2, theory_g1)
   fitted_g2 = offset + contrast * theory_g1

Static Anisotropic Mode
~~~~~~~~~~~~~~~~~~~~~~~

In anisotropic mode, scaling optimization:
- Applies per-angle scaling when enabled
- Accounts for angular variations in detector response
- Maintains physics model accuracy across angles

.. code-block:: python

   # Anisotropic scaling with angle filtering
   for angle_idx in filtered_angles:
       theory_g1_angle = compute_g1_anisotropic(params, time_points, angles[angle_idx])
       contrast, offset = optimize_scaling(experimental_g2[angle_idx], theory_g1_angle)
       fitted_g2[angle_idx] = offset + contrast * theory_g1_angle

Laminar Flow Mode
~~~~~~~~~~~~~~~~~

In flow mode, scaling optimization:
- Handles the complex g₁(t₁,t₂) = g₁,diff × g₁,shear product
- Accounts for flow-dependent experimental factors
- Maintains accuracy across the full parameter space

.. code-block:: python

   # Flow mode scaling
   theory_g1_diff = compute_g1_diffusion(params, time_points)
   theory_g1_shear = compute_g1_shear(params, time_points, angles)
   theory_g1_total = theory_g1_diff * theory_g1_shear
   contrast, offset = optimize_scaling(experimental_g2, theory_g1_total)

Performance Considerations
--------------------------

Computational Efficiency
~~~~~~~~~~~~~~~~~~~~~~~~

The scaling optimization adds minimal computational overhead:
- **Matrix operation**: Single least squares solution per optimization step
- **Vectorized implementation**: Efficient NumPy operations
- **Memory efficient**: Uses existing data arrays

**Timing Impact**: < 1% additional computation time for typical datasets

Numerical Stability
~~~~~~~~~~~~~~~~~~~

The implementation ensures numerical stability through:

.. code-block:: python

   # Robust implementation with conditioning check
   A = np.vstack([theory, np.ones(len(theory))]).T
   
   # Check matrix conditioning
   condition_number = np.linalg.cond(A)
   if condition_number > 1e12:
       # Use regularized solution
       scaling = np.linalg.lstsq(A, experimental, rcond=1e-15)[0]
   else:
       scaling = np.linalg.lstsq(A, experimental, rcond=None)[0]

Integration with Optimization
-----------------------------

Classical Optimization
~~~~~~~~~~~~~~~~~~~~~~

During classical optimization, scaling is computed at each iteration:

.. code-block:: python

   def objective_function(params):
       # Compute theoretical g1
       theory_g1 = compute_g1(params)
       
       # Apply scaling optimization
       contrast, offset = optimize_scaling(experimental_g2, theory_g1)
       fitted_g2 = offset + contrast * theory_g1
       
       # Return chi-squared
       return np.sum((experimental_g2 - fitted_g2)**2 / uncertainties**2)

MCMC Sampling
~~~~~~~~~~~~~

In Bayesian analysis, scaling parameters can be treated as:

1. **Fixed scaling** (default): Optimized at each MCMC step
2. **Sampled scaling**: Include contrast and offset as parameters

.. code-block:: python

   # MCMC with fixed scaling (recommended)
   def log_likelihood(params):
       theory_g1 = compute_g1(params)
       contrast, offset = optimize_scaling(experimental_g2, theory_g1)
       fitted_g2 = offset + contrast * theory_g1
       return -0.5 * np.sum((experimental_g2 - fitted_g2)**2 / uncertainties**2)

Historical Context and Migration
-------------------------------

Legacy Behavior
~~~~~~~~~~~~~~~

In earlier versions, scaling optimization was optional and controlled by:

.. code-block:: json

   {
     "chi_squared_calculation": {
       "scaling_optimization": true  // This setting is now obsolete
     }
   }

Current Implementation
~~~~~~~~~~~~~~~~~~~~~

**Scaling optimization is now always enabled** because:
- Scientific accuracy requires proper g₁ to g₂ transformation
- Chi-squared statistics are meaningless without scaling
- All published results depend on correct scaling
- No computational penalty for always enabling

Migration Guide
~~~~~~~~~~~~~~~

**Remove obsolete settings** from configuration files:

.. code-block:: json

   {
     "chi_squared_calculation": {
       // Remove this entire section or just the scaling_optimization line
       "scaling_optimization": true  // Remove this line
     }
   }

**No code changes required** - scaling is automatically applied.

Validation and Quality Control
------------------------------

Scaling Parameter Diagnostics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Monitor scaling parameters for data quality assessment:

.. code-block:: python

   # Typical ranges for scaling parameters
   typical_contrast_range = (0.001, 0.1)  # Depends on experimental setup
   typical_offset_range = (0.98, 1.02)    # Near unity for good data

   if not (typical_contrast_range[0] <= contrast <= typical_contrast_range[1]):
       warnings.warn(f"Unusual contrast value: {contrast}")
   
   if not (typical_offset_range[0] <= offset <= typical_offset_range[1]):
       warnings.warn(f"Unusual offset value: {offset}")

Visual Validation
~~~~~~~~~~~~~~~~

Create diagnostic plots to assess scaling quality:

.. code-block:: python

   import matplotlib.pyplot as plt
   
   # Plot experimental vs fitted data
   plt.figure(figsize=(10, 6))
   plt.plot(time_points, experimental_g2, 'o', label='Experimental')
   plt.plot(time_points, fitted_g2, '-', label='Fitted (scaled)')
   plt.plot(time_points, theory_g1, '--', label='Theory (unscaled)')
   plt.xlabel('Time')
   plt.ylabel('Correlation Function')
   plt.legend()
   plt.title(f'Scaling: contrast={contrast:.4f}, offset={offset:.4f}')

Best Practices
--------------

Data Quality Requirements
~~~~~~~~~~~~~~~~~~~~~~~~

For optimal scaling optimization:
- **Sufficient contrast**: Experimental g₂ should show clear dynamic behavior
- **Good statistics**: Adequate signal-to-noise ratio
- **Proper normalization**: Data should be pre-normalized when possible
- **Consistent units**: Ensure experimental and theoretical functions use same units

Interpretation Guidelines
~~~~~~~~~~~~~~~~~~~~~~~~

**Contrast Parameter**:
- Should be positive for physical systems
- Magnitude indicates measurement quality
- Very small values (< 0.001) may indicate poor dynamics or high noise

**Offset Parameter**:
- Should be near unity for well-normalized data
- Large deviations may indicate systematic problems
- Negative values are generally unphysical

Troubleshooting
~~~~~~~~~~~~~~~

**Poor scaling results** may indicate:
- Inadequate experimental data quality
- Inappropriate physics model for the system
- Systematic experimental problems
- Numerical instability in computation

**Solutions**:
- Validate experimental data quality
- Check model appropriateness for system
- Review experimental setup and procedures
- Verify numerical stability of computations

This comprehensive scaling optimization ensures that the homodyne package provides scientifically accurate and meaningful results across all analysis modes.
