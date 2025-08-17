Analysis Modes
==============

The homodyne analysis package supports three distinct analysis modes, each optimized for different experimental scenarios and computational requirements.

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

Physical Context
~~~~~~~~~~~~~~~~

Analysis of systems at equilibrium with isotropic scattering where results don't depend on scattering angle. This is the fastest analysis mode and is ideal for preliminary analysis and systems with no angular dependence.

**Mathematical Model**:

.. math::

   g_1(t_1,t_2) = \exp(-q^2 \int_{t_1}^{t_2} D(t) dt)

where there is no angular dependence in the correlation function.

Parameters (3 total)
~~~~~~~~~~~~~~~~~~~

- **D₀**: Effective diffusion coefficient
- **α**: Time exponent characterizing dynamic scaling
- **D_offset**: Baseline diffusion component

Key Features
~~~~~~~~~~~~

- **No angle filtering**: Automatically disabled regardless of configuration
- **No phi_angles_file loading**: Uses single dummy angle internally
- **Fastest execution**: Minimal computational overhead
- **Simplified physics**: No angular dependencies considered

When to Use
~~~~~~~~~~~

- Isotropic samples with no preferred orientations
- Quick validation runs and preliminary analysis
- Systems where angular effects are negligible
- Computational efficiency is critical

Example Configuration
~~~~~~~~~~~~~~~~~~~~

.. code-block:: json

   {
     "analysis_settings": {
       "static_mode": true,
       "static_submode": "isotropic"
     },
     "initial_parameters": {
       "active_parameters": ["D0", "alpha", "D_offset"]
     }
   }

Command Line Usage
~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Basic isotropic analysis
   python run_homodyne.py --static-isotropic --method classical

   # Create isotropic configuration
   python create_config.py --mode static_isotropic --sample my_sample

Static Anisotropic Mode
-----------------------

Physical Context
~~~~~~~~~~~~~~~~

Analysis of systems at equilibrium with angular dependence but no flow effects. This mode considers the directional properties of scattering while maintaining the three-parameter model.

**Mathematical Model**:

.. math::

   g_1(t_1,t_2) = \exp(-q^2 \int_{t_1}^{t_2} D(t) dt)

Same as isotropic mode but with angle filtering to focus optimization on specific angular ranges.

Parameters (3 total)
~~~~~~~~~~~~~~~~~~~

- **D₀**: Effective diffusion coefficient (same as isotropic)
- **α**: Time exponent characterizing dynamic scaling  
- **D_offset**: Baseline diffusion component

Key Features
~~~~~~~~~~~~

- **Angle filtering enabled**: Focuses on specific angular ranges for optimization efficiency
- **phi_angles_file loaded**: Reads angle information from data files
- **Per-angle scaling optimization**: Accounts for angular variations
- **Moderate computational cost**: Balance between speed and accuracy

When to Use
~~~~~~~~~~~

- Static samples with measurable angular variations
- Systems with preferred orientations or anisotropic structures
- When you need more accuracy than isotropic mode
- Moderate computational resources available

Angular Range Optimization
~~~~~~~~~~~~~~~~~~~~~~~~~~

The anisotropic mode typically focuses on two angular ranges:

- **Forward scattering**: [-10°, 10°] around 0°
- **Backward scattering**: [170°, 190°] around 180°

This provides 3-5x speedup while maintaining high accuracy.

Example Configuration
~~~~~~~~~~~~~~~~~~~~

.. code-block:: json

   {
     "analysis_settings": {
       "static_mode": true,
       "static_submode": "anisotropic",
       "enable_angle_filtering": true,
       "angle_filter_ranges": [[-10, 10], [170, 190]]
     },
     "initial_parameters": {
       "active_parameters": ["D0", "alpha", "D_offset"]
     }
   }

Command Line Usage
~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Basic anisotropic analysis
   python run_homodyne.py --static-anisotropic --method classical

   # Create anisotropic configuration
   python create_config.py --mode static_anisotropic --sample my_sample

Laminar Flow Mode
-----------------

Physical Context
~~~~~~~~~~~~~~~~

Analysis of systems under controlled shear flow conditions with the complete physics model. This mode captures the interplay between Brownian diffusion and advective shear flow.

**Mathematical Model**:

.. math::

   g_1(t_1,t_2) = g_{1,\text{diff}}(t_1,t_2) \times g_{1,\text{shear}}(t_1,t_2)

where the shear effects are modeled as:

.. math::

   g_{1,\text{shear}}(t_1,t_2) = \text{sinc}^2(\Phi)

Parameters (7 total)
~~~~~~~~~~~~~~~~~~~

**Diffusion Parameters**:
- **D₀**: Effective diffusion coefficient
- **α**: Time exponent for diffusion scaling
- **D_offset**: Baseline diffusion component

**Flow Parameters**:
- **γ̇₀**: Characteristic shear rate
- **β**: Shear rate exponent for flow scaling
- **γ̇_offset**: Baseline shear component
- **φ₀**: Angular offset parameter for flow geometry

Key Features
~~~~~~~~~~~~

- **Complete physics model**: Includes all flow and diffusion effects
- **phi_angles_file required**: Angle information essential for flow effects
- **Complex parameter space**: Potential correlations between parameters
- **Comprehensive analysis**: Full transport coefficient characterization

When to Use
~~~~~~~~~~~

- Systems under controlled shear or flow
- Nonequilibrium conditions
- Transport coefficient analysis required
- Publication-quality results with full uncertainty quantification

Physical Interpretation
~~~~~~~~~~~~~~~~~~~~~~

The laminar flow mode captures:

- **Brownian diffusion**: Random thermal motion characterized by D₀, α, D_offset
- **Advective flow**: Systematic motion due to applied shear (γ̇₀, β, γ̇_offset)
- **Geometric effects**: Angular dependence of flow effects (φ₀)
- **Scaling relationships**: Time-dependent behavior in both diffusion and flow

Example Configuration
~~~~~~~~~~~~~~~~~~~~

.. code-block:: json

   {
     "analysis_settings": {
       "static_mode": false
     },
     "initial_parameters": {
       "active_parameters": [
         "D0", "alpha", "D_offset", 
         "gamma_dot_t0", "beta", "gamma_dot_t_offset", "phi0"
       ]
     }
   }

Command Line Usage
~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Full flow analysis
   python run_homodyne.py --laminar-flow --method classical

   # Create flow configuration
   python create_config.py --mode laminar_flow --sample my_sample

Mode Selection Guidelines
-------------------------

Choosing the Right Mode
~~~~~~~~~~~~~~~~~~~~~~~

**Start with Static Isotropic** if:

- Your system is likely isotropic
- You need quick results for exploration
- Computational resources are limited
- Angular effects are expected to be minimal

**Use Static Anisotropic** if:

- Angular dependence is expected or observed
- You have moderate computational resources
- Better accuracy than isotropic mode is needed
- System is at equilibrium but shows directional properties

**Use Laminar Flow** if:

- System is under applied shear or flow
- Nonequilibrium conditions are present
- Complete transport analysis is required
- You have sufficient computational resources

Progressive Analysis Strategy
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A recommended approach is to use progressive complexity:

1. **Exploration**: Start with isotropic mode for initial parameter estimates
2. **Validation**: Use anisotropic mode to check for angular effects
3. **Complete Analysis**: Apply flow mode if nonequilibrium effects are present

.. code-block:: bash

   # Progressive analysis workflow
   python run_homodyne.py --static-isotropic --method classical
   python run_homodyne.py --static-anisotropic --method classical  
   python run_homodyne.py --laminar-flow --method all

Performance Considerations
--------------------------

Computational Scaling
~~~~~~~~~~~~~~~~~~~~~

.. list-table:: Relative Performance
   :widths: 20 15 15 50
   :header-rows: 1

   * - Mode
     - Speed
     - Memory
     - Notes
   * - Static Isotropic
     - 1x
     - Low
     - Single angle, minimal overhead
   * - Static Anisotropic
     - 3x
     - Medium
     - Angle filtering provides speedup
   * - Laminar Flow
     - 10x
     - High
     - Full parameter space, complex model

Optimization Strategies
~~~~~~~~~~~~~~~~~~~~~~~

**For Large Datasets**:

- Enable angle filtering in anisotropic and flow modes
- Use ``float32`` data type to reduce memory usage
- Increase ``memory_limit_gb`` setting appropriately

**For Flow Mode**:

- Start with good initial parameter estimates from static analysis
- Use classical optimization first, then MCMC for uncertainties
- Consider parameter bounds to constrain search space

Mode-Specific Troubleshooting
------------------------------

Static Isotropic Issues
~~~~~~~~~~~~~~~~~~~~~~

**"Angle filtering enabled but static_isotropic mode detected"**:
This is expected behavior - angle filtering is automatically disabled in isotropic mode.

**"phi_angles_file not found" in isotropic mode**:
This is normal - phi_angles_file is not used in isotropic mode.

Static Anisotropic Issues
~~~~~~~~~~~~~~~~~~~~~~~~

**Poor convergence with angle filtering**:
Try adjusting ``angle_filter_ranges`` or disabling filtering temporarily.

**Results similar to isotropic mode**:
Your system may indeed be isotropic - compare chi-squared values.

Laminar Flow Issues
~~~~~~~~~~~~~~~~~~

**Slow optimization**:
Enable angle filtering for 3-5x speedup with minimal accuracy loss.

**Parameter correlation problems**:
Use good initial estimates from static analysis first.

**MCMC convergence issues**:
- Increase tuning steps (``tune: 2000+``)
- Increase target acceptance rate (``target_accept: 0.95``)
- Use better initial parameter estimates

This comprehensive guide should help you select and effectively use the appropriate analysis mode for your experimental conditions.
