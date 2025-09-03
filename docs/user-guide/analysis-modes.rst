Analysis Modes
==============

The homodyne package supports three analysis modes optimized for different experimental scenarios.

Mode Overview
-------------

.. list-table:: Analysis Mode Comparison
   :widths: 15 10 15 30 10 15
   :header-rows: 1

   * - Mode
     - Parameters
     - Angle Handling
     - Use Case
     - Speed
     - Command
   * - **Static Isotropic**
     - 3
     - Single dummy
     - Fastest, isotropic systems
     - ⭐⭐⭐
     - ``--static-isotropic``
   * - **Static Anisotropic**
     - 3
     - Filtering enabled
     - Static with angular deps
     - ⭐⭐
     - ``--static-anisotropic``
   * - **Laminar Flow**
     - 7
     - Full coverage
     - Flow & shear analysis
     - ⭐
     - ``--laminar-flow``

Static Isotropic Mode
---------------------

**Physical Context**: Analysis of systems at equilibrium with isotropic scattering where results don't depend on scattering angle.

**Model Equation**:

.. math::

   g_1(t_1,t_2) = \exp(-q^2 \int_{t_1}^{t_2} D(t) dt)

where there is no angular dependence in the correlation function.

**Parameters (3 total)**:

- **D₀**: Effective diffusion coefficient
- **α**: Time exponent characterizing dynamic scaling
- **D_offset**: Baseline diffusion component

**Key Features**:

- **No angle filtering**: Automatically disabled regardless of configuration
- **No phi_angles_file loading**: Uses single dummy angle
- **Fastest analysis mode**: Minimal computational overhead

**When to Use**:

- Isotropic samples
- Quick validation runs
- Preliminary analysis
- Systems where angular effects are negligible

**Example Configuration**:

.. code-block:: javascript

   {
     "analysis_settings": {
       "static_mode": true,
       "static_submode": "isotropic"
     },
     "initial_parameters": {
       "parameter_names": ["D0", "alpha", "D_offset"],
       "values": [1000, -0.5, 100]
     }
   }

Static Anisotropic Mode
-----------------------

**Physical Context**: Analysis of systems at equilibrium with angular dependence but no flow effects.

**Parameters**: D₀, α, D_offset (same as isotropic mode)

**Key Features**:

- **Angle filtering enabled**: For optimization efficiency
- **phi_angles_file loaded**: For angle information
- **Per-angle scaling optimization**: Accounts for angular variations

**When to Use**:

- Systems with angular dependence
- Static samples with anisotropic properties
- When isotropic mode gives poor fits
- Intermediate complexity analysis

**Example Configuration**:

.. code-block:: javascript

   {
     "analysis_settings": {
       "static_mode": true,
       "static_submode": "anisotropic",
       "enable_angle_filtering": true,
       "angle_filter_ranges": [[-5, 5], [175, 185]]
     },
     "file_paths": {
       "phi_angles_file": "data/phi_angles.txt"
     }
   }

Laminar Flow Mode
-----------------

**Physical Context**: Complete analysis of systems under flow conditions with both diffusion and shear contributions.

**Model Equations**:

The full expression combines diffusive and shear contributions:

.. math::

   g_{1,\text{total}}(t_1,t_2) = g_{1,\text{diffusion}}(t_1,t_2) \times g_{1,\text{shear}}(t_1,t_2)

where the diffusion contribution follows:

.. math::

   g_{1,\text{diffusion}}(t_1,t_2) = \exp\left(-q^2 \int_{t_1}^{t_2} D(t) dt\right)

and the shear contribution is:

.. math::

   g_{1,\text{shear}}(t_1,t_2) = \text{sinc}^2\left(\frac{qh}{2\pi} \cos(\phi) \int_{t_1}^{t_2} \dot{\gamma}(t) dt\right)

with the time-dependent transport coefficients:

.. math::

   D(t) &= D_0 t^\alpha + D_{\text{offset}} \\
   \dot{\gamma}(t) &= \dot{\gamma}_0 t^\beta + \dot{\gamma}_{\text{offset}}

**Parameters (7 total)**:

.. list-table::
   :widths: 20 30 15 35
   :header-rows: 1

   * - Parameter
     - Symbol
     - Units
     - Physical Meaning
   * - **D₀**
     - :math:`D_0`
     - Å\ :sup:`2`\ /s\ :sup:`1+α`
     - Effective diffusion coefficient
   * - **α**
     - :math:`\alpha`
     - dimensionless
     - Time exponent for diffusion scaling
   * - **D_offset**
     - :math:`D_{\text{offset}}`
     - Å\ :sup:`2`\ /s
     - Baseline diffusion component
   * - **γ̇₀**
     - :math:`\dot{\gamma}_0`
     - s\ :sup:`-1-β`
     - Shear rate amplitude
   * - **β**
     - :math:`\beta`
     - dimensionless
     - Shear rate time exponent
   * - **γ̇_offset**
     - :math:`\dot{\gamma}_{\text{offset}}`
     - s\ :sup:`-1`
     - Baseline shear rate
   * - **φ₀**
     - :math:`\phi_0`
     - degrees
     - Phase angle for shear/flow direction

**Physical Interpretation**:

The laminar flow mode captures:

- **Brownian diffusion**: Random thermal motion characterized by D₀, α, D_offset
- **Advective shear flow**: Systematic flow characterized by γ̇₀, β, γ̇_offset, φ₀
- **Angular dependencies**: Full angular coverage with flow direction effects

**Example Configuration**:

.. code-block:: javascript

   {
     "analysis_settings": {
       "static_mode": false,
       "enable_angle_filtering": true
     },
     "initial_parameters": {
       "parameter_names": ["D0", "alpha", "D_offset", "gamma_dot_t0", "beta", "gamma_dot_t_offset", "phi0"],
       "values": [1000, -0.5, 100, 10, 0.5, 1, 0],
       "active_parameters": ["D0", "alpha", "D_offset", "gamma_dot_t0"]
     }
   }

**When to Use**:

- Systems under flow conditions
- Nonequilibrium conditions are present
- Complete transport analysis is required
- You have sufficient computational resources

Progressive Analysis Strategy
-----------------------------

A recommended approach is to use progressive complexity:

1. **Exploration**: Start with isotropic mode for initial parameter estimates
2. **Validation**: Compare with anisotropic mode to check for angular effects
3. **Full Analysis**: Use laminar flow mode for complete characterization

**Example Workflow**:

.. code-block:: bash

   # Step 1: Quick isotropic analysis
   python run_homodyne.py --static-isotropic --method classical

   # Step 2: Check for angular effects
   python run_homodyne.py --static-anisotropic --method classical

   # Step 3: Full flow analysis (if needed)
   python run_homodyne.py --laminar-flow --method mcmc

Mode Selection Guidelines
-------------------------

**Choose Static Isotropic when**:
- System is known to be isotropic
- You need quick results
- Doing preliminary data validation
- Angular effects are negligible

**Choose Static Anisotropic when**:
- System shows angular dependence
- No flow conditions present
- Isotropic results are unsatisfactory
- Need moderate complexity analysis

**Choose Laminar Flow when**:
- System is under flow conditions
- Nonequilibrium conditions are present
- Complete transport analysis is required
- You have sufficient computational resources

Troubleshooting
---------------

**"Angle filtering enabled but static_isotropic mode detected"**:
   This is expected behavior - angle filtering is automatically disabled in isotropic mode.

**"phi_angles_file not found" in isotropic mode**:
   This is normal - phi_angles_file is not used in isotropic mode.

**Poor convergence with angle filtering**:
   Try adjusting ``angle_filter_ranges`` or disabling filtering temporarily.

**Results similar to isotropic mode**:
   Your system may indeed be isotropic - compare chi-squared values.

**Slow optimization**:
   Enable angle filtering for 3-5x speedup with minimal accuracy loss.
