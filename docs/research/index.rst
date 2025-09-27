Research Documentation
======================

This section provides comprehensive research-grade documentation for the homodyne-analysis package,
including theoretical foundations, computational methods, and experimental validation.

.. toctree::
   :maxdepth: 2
   :caption: Research Components

   theoretical_framework
   computational_methods
   experimental_validation
   performance_analysis
   publications
   collaboration

Overview
--------

The homodyne-analysis package implements the theoretical framework for analyzing X-ray Photon
Correlation Spectroscopy (XPCS) data under nonequilibrium conditions, as described in:

**He, H., Liang, H., Chu, M., Jiang, Z., de Pablo, J.J., Tirrell, M.V., Narayanan, S., & Chen, W.**
*Transport coefficient approach for characterizing nonequilibrium dynamics in soft matter.*
Proceedings of the National Academy of Sciences, 121(31), e2401162121 (2024).
DOI: `10.1073/pnas.2401162121 <https://doi.org/10.1073/pnas.2401162121>`_

Research Scope
--------------

Scientific Applications
~~~~~~~~~~~~~~~~~~~~~~~

* **Soft Matter Physics**: Colloidal dynamics, active matter, biological systems
* **Flow Rheology**: Shear thinning/thickening, microfluidics, complex fluids
* **Materials Science**: Phase transitions, glass transition, crystallization

Computational Innovation
~~~~~~~~~~~~~~~~~~~~~~~~

* **High-Performance Computing**: Numba JIT compilation with 3-5x speedup
* **Robust Optimization**: Distributionally robust optimization with uncertainty quantification
* **Memory Efficiency**: Vectorized operations with smart caching strategies

Experimental Integration
~~~~~~~~~~~~~~~~~~~~~~~~

* **Synchrotron Facilities**: Advanced Photon Source (APS) integration
* **Data Standards**: HDF5 and standardized XPCS data formats
* **Validation Protocols**: Statistical validation and uncertainty analysis

Mathematical Framework
----------------------

The package analyzes time-dependent intensity correlation functions:

.. math::

   c_2(\vec{q}, t_1, t_2) = 1 + \beta\left[e^{-q^2\int J(t)dt}\right] \times
   \text{sinc}^2\left[\frac{1}{2\pi} qh \int\dot{\gamma}(t)\cos(\phi(t))dt\right]

where:

* :math:`\vec{q}`: scattering wavevector [Å⁻¹]
* :math:`h`: gap between stator and rotor [Å]
* :math:`\phi(t)`: angle between shear/flow direction and :math:`\vec{q}` [degrees]
* :math:`\dot{\gamma}(t)`: time-dependent shear rate [s⁻¹]
* :math:`D(t)`: time-dependent diffusion coefficient [Å²/s]
* :math:`\beta`: instrumental contrast parameter

Analysis Modes
--------------

.. list-table:: Analysis Capabilities
   :header-rows: 1
   :widths: 20 10 40 15 15

   * - Mode
     - Parameters
     - Physical Description
     - Complexity
     - Applications
   * - Static Isotropic
     - 3
     - Brownian motion only
     - O(N)
     - Equilibrium systems
   * - Static Anisotropic
     - 3
     - Angular dependence
     - O(N log N)
     - Structured materials
   * - Laminar Flow
     - 7
     - Full nonequilibrium
     - O(N²)
     - Flowing systems

Research Validation
-------------------

Statistical Methods
~~~~~~~~~~~~~~~~~~~

* **Cross-validation**: K-fold validation for parameter estimation reliability
* **Bootstrap Analysis**: Statistical uncertainty quantification
* **Residual Analysis**: Goodness-of-fit assessment and outlier detection

Experimental Validation
~~~~~~~~~~~~~~~~~~~~~~~

* **Synchrotron Data**: Validation against APS Sector 8-ID-I measurements
* **Sample Systems**: Colloidal suspensions, protein solutions, active matter
* **Flow Conditions**: Laminar shear, pressure-driven flow, microfluidic geometries

Performance Benchmarks
~~~~~~~~~~~~~~~~~~~~~~

.. list-table:: Performance Comparison (Intel Xeon, 8 cores)
   :header-rows: 1
   :widths: 20 20 20 20

   * - Data Size
     - Pure Python
     - Numba JIT
     - Speedup
   * - 100 points
     - 2.3 s
     - 0.7 s
     - 3.3×
   * - 500 points
     - 12.1 s
     - 3.2 s
     - 3.8×
   * - 1000 points
     - 45.2 s
     - 8.9 s
     - 5.1×
   * - 5000 points
     - 892 s
     - 178 s
     - 5.0×

Citation Guidelines
-------------------

Primary Research Citation
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bibtex

   @article{he2024transport,
     title={Transport coefficient approach for characterizing nonequilibrium dynamics in soft matter},
     author={He, Hongrui and Liang, Hao and Chu, Miaoqi and Jiang, Zhang and de Pablo, Juan J and Tirrell, Matthew V and Narayanan, Suresh and Chen, Wei},
     journal={Proceedings of the National Academy of Sciences},
     volume={121},
     number={31},
     pages={e2401162121},
     year={2024},
     publisher={National Academy of Sciences},
     doi={10.1073/pnas.2401162121}
   }

Software Package Citation
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bibtex

   @software{homodyne_analysis,
     title={homodyne-analysis: High-performance XPCS analysis with robust optimization},
     author={Chen, Wei and He, Hongrui},
     year={2024},
     url={https://github.com/imewei/homodyne},
     version={0.7.1},
     institution={Argonne National Laboratory}
   }

Research Collaboration
----------------------

The development of this research software is supported by:

**Funding Agencies**
  * U.S. Department of Energy, Office of Science, Basic Energy Sciences
  * National Science Foundation (NSF) Division of Materials Research
  * Advanced Photon Source User Facility

**Collaborating Institutions**
  * Argonne National Laboratory - X-ray Science Division
  * University of Chicago - Institute for Molecular Engineering
  * Northwestern University - Department of Materials Science

**Contact Information**
  * **Principal Investigator**: Wei Chen (wchen@anl.gov)
  * **Lead Developer**: Hongrui He
  * **Technical Support**: GitHub Issues
  * **Research Collaboration**: Argonne National Laboratory, X-ray Science Division
