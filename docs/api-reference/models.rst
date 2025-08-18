Physical Models
===============

This module contains the physical model functions used in homodyne analysis.

Correlation Models
------------------

.. autofunction:: homodyne.core.kernels.compute_g1_correlation_numba

   Computes the normalized first-order correlation function g₁(τ).

.. autofunction:: homodyne.analysis.core.static_isotropic_analysis

   Model for isotropic systems at equilibrium.

.. autofunction:: homodyne.analysis.core.static_anisotropic_analysis

   Model for anisotropic systems at equilibrium.

.. autofunction:: homodyne.analysis.core.laminar_flow_analysis

   Complete model for systems under laminar flow conditions.

Diffusion Functions
-------------------

.. autofunction:: homodyne.core.kernels.calculate_diffusion_coefficient_numba

   Computes the diffusive contribution to correlation decay.

.. autofunction:: homodyne.core.kernels.calculate_diffusion_coefficient_numba

   Time-dependent diffusion coefficient D(t).

Shear Flow Functions
--------------------

.. autofunction:: homodyne.core.kernels.compute_sinc_squared_numba

   Computes shear contribution using sinc² functions.

.. autofunction:: homodyne.core.kernels.compute_sinc_squared_numba

   Angular phase factor for flow direction.

.. autofunction:: homodyne.core.kernels.calculate_shear_rate_numba

   Time-dependent shear rate γ̇(t).

Utility Functions
-----------------

.. autofunction:: homodyne.analysis.core.validate_parameters

   Validates parameter values are physically reasonable.

.. autofunction:: homodyne.analysis.core.compute_chi_squared_with_validation

   Computes chi-squared goodness of fit.

Model Equations
---------------

**Static Isotropic Model**:

.. math::

   g_1(τ) = \exp\left(-q^2 \int_0^τ D(t) dt\right)

where:

.. math::

   D(t) = D_0 t^α + D_{\text{offset}}

**Static Anisotropic Model**:

Same as isotropic but with angle-dependent scaling factors.

**Laminar Flow Model**:

.. math::

   g_{1,\text{total}}(τ) = g_{1,\text{diffusion}}(τ) \times g_{1,\text{shear}}(τ)

.. math::

   g_{1,\text{shear}}(τ) = \text{sinc}^2(\Phi)

where:

.. math::

   \Phi = \frac{q \sin(\phi - \phi_0)}{2} \int_0^τ \gammȧ(t) dt

.. math::

   \gammȧ(t) = \gammȧ_0 t^β + \gammȧ_{\text{offset}}

Usage Examples
--------------

**Computing Correlations**:

.. code-block:: python

   from homodyne.models import static_isotropic_model
   import numpy as np
   
   # Time points
   tau = np.logspace(-6, 1, 100)
   
   # Parameters: [D0, alpha, D_offset]
   params = [1500, -0.8, 50]
   
   # Scattering vector
   q = 0.001  # μm⁻¹
   
   # Compute correlation
   g1 = static_isotropic_model(tau, params, q)
   
   print(f"g1(τ=1μs) = {g1[50]:.4f}")

**Flow Model**:

.. code-block:: python

   from homodyne.models import laminar_flow_model
   
   # Flow parameters: [D0, alpha, D_offset, gamma_dot_t0, beta, gamma_dot_t_offset, phi0]
   flow_params = [1200, -0.9, 80, 15, 0.3, 2, 0]
   
   # Angle array
   phi_angles = np.array([0, 45, 90, 135, 180])  # degrees
   
   # Compute correlation for each angle
   correlations = []
   for phi in phi_angles:
       g1_phi = laminar_flow_model(tau, flow_params, q, phi)
       correlations.append(g1_phi)
   
   correlations = np.array(correlations)

**Parameter Validation**:

.. code-block:: python

   from homodyne.models import validate_parameters
   
   # Check if parameters are reasonable
   params = [1500, -0.8, 50]
   is_valid, messages = validate_parameters(params, mode="isotropic")
   
   if is_valid:
       print("✅ Parameters are valid")
   else:
       for msg in messages:
           print(f"⚠️ {msg}")

**Chi-squared Calculation**:

.. code-block:: python

   from homodyne.models import compute_chi_squared
   
   # Theoretical and experimental data
   theory = static_isotropic_model(tau, params, q)
   experimental = np.load("experimental_data.npy")
   
   # Calculate goodness of fit
   chi2 = compute_chi_squared(theory, experimental)
   print(f"Chi-squared: {chi2:.4f}")

Parameter Ranges
----------------

**Typical Physical Ranges**:

.. list-table:: Parameter Guidelines
   :widths: 15 20 20 45
   :header-rows: 1

   * - Parameter
     - Symbol
     - Typical Range
     - Physical Meaning
   * - **D₀**
     - D₀
     - 100 - 10,000 μm²/s
     - Effective diffusion coefficient
   * - **α**
     - α
     - -2.0 to 0.0
     - Time scaling exponent
   * - **D_offset**
     - D_offset
     - 0 - 1,000 μm²/s
     - Baseline diffusion
   * - **γ̇₀**
     - γ̇₀
     - 1 - 100 s⁻¹
     - Shear rate amplitude
   * - **β**
     - β
     - -1.0 to 1.0
     - Shear time exponent
   * - **γ̇_offset**
     - γ̇_offset
     - 0 - 10 s⁻¹
     - Baseline shear rate
   * - **φ₀**
     - φ₀
     - 0 - 360°
     - Flow direction angle

Model Selection
---------------

**Choose models based on your system**:

1. **Static Isotropic**: Fastest, for truly isotropic systems
2. **Static Anisotropic**: For systems with angular dependence
3. **Laminar Flow**: Most complete, for flow conditions