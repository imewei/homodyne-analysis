Publications and Citations
==========================

This section provides comprehensive citation information and publication guidelines for the
homodyne-analysis package and related research.

Primary Research Publication
----------------------------

**Transport coefficient approach for characterizing nonequilibrium dynamics in soft matter**

.. admonition:: Primary Citation
   :class: important

   He, H., Liang, H., Chu, M., Jiang, Z., de Pablo, J.J., Tirrell, M.V., Narayanan, S., & Chen, W.
   *Transport coefficient approach for characterizing nonequilibrium dynamics in soft matter.*
   **Proceedings of the National Academy of Sciences**, 121(31), e2401162121 (2024).
   DOI: `10.1073/pnas.2401162121 <https://doi.org/10.1073/pnas.2401162121>`_

Abstract
~~~~~~~~

We present a transport coefficient approach for characterizing nonequilibrium dynamics in soft
matter systems using X-ray photon correlation spectroscopy (XPCS). This method enables direct
extraction of time-dependent diffusion coefficients and shear rates from intensity correlation
functions, providing unprecedented insight into the interplay between Brownian motion and
advective flow in complex fluids.

Key Contributions
~~~~~~~~~~~~~~~~~

1. **Theoretical Framework**: Development of time-dependent correlation functions for nonequilibrium systems
2. **Experimental Validation**: Demonstration using synchrotron XPCS measurements
3. **Computational Implementation**: High-performance algorithms for parameter extraction
4. **Robust Analysis**: Statistical methods for handling experimental uncertainty

Software Package Citation
-------------------------

**homodyne-analysis: High-performance XPCS analysis with robust optimization**

.. code-block:: bibtex

   @software{homodyne_analysis_2024,
     title={homodyne-analysis: High-performance XPCS analysis with robust optimization},
     author={Chen, Wei and He, Hongrui},
     year={2024},
     url={https://github.com/imewei/homodyne},
     version={0.7.1},
     institution={Argonne National Laboratory},
     license={MIT},
     keywords={XPCS, homodyne scattering, nonequilibrium dynamics, robust optimization}
   }

Software Features Citation
~~~~~~~~~~~~~~~~~~~~~~~~~~

When citing specific features of the software, use the following format:

**For Robust Optimization Methods:**

.. code-block:: bibtex

   @misc{homodyne_robust_optimization,
     title={Distributionally robust optimization for XPCS data analysis},
     author={Chen, Wei and He, Hongrui},
     howpublished={homodyne-analysis software package},
     year={2024},
     note={Implementation of Wasserstein distributionally robust optimization}
   }

**For Performance Optimization:**

.. code-block:: bibtex

   @misc{homodyne_performance,
     title={High-performance computing methods for XPCS analysis},
     author={Chen, Wei and He, Hongrui},
     howpublished={homodyne-analysis software package},
     year={2024},
     note={Numba JIT compilation achieving 3-5x speedup}
   }

Related Publications
--------------------

Foundational Theory
~~~~~~~~~~~~~~~~~~~

.. [Berne1976] Berne, B.J. & Pecora, R. *Dynamic Light Scattering: With Applications to Chemistry,
               Biology, and Physics.* Wiley, New York (1976).

.. [Brown1993] Brown, W. *Dynamic Light Scattering: The Method and Some Applications.*
               Oxford University Press, Oxford (1993).

.. [Pusey1989] Pusey, P.N. & van Megen, W. "Dynamic light scattering by non-ergodic media."
               *Physica A*, 157(2), 705-741 (1989).

X-ray Photon Correlation Spectroscopy
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. [Grubel2008] Grübel, G., Madsen, A. & Robert, A. "X-ray photon correlation spectroscopy."
                *X-Ray Photon Correlation Spectroscopy*, pp. 953-995. Springer (2008).

.. [Sutton2002] Sutton, M. "A review of X-ray intensity fluctuation spectroscopy."
                *Comptes Rendus Physique*, 9(5-6), 657-667 (2008).

.. [Sandy2010] Sandy, A.R., Narayanan, S., Sprung, M., Su, J.H., Evans-Lutterodt, K.,
               Isakovic, A.F. & Stein, A. "Instrumentation developments for surface and bulk
               scattering and spectroscopy studies at the Advanced Photon Source."
               *Journal of Synchrotron Radiation*, 17(6), 711-716 (2010).

Nonequilibrium Soft Matter
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. [Cates2012] Cates, M.E. & Tailleur, J. "Motility-induced phase separation."
               *Annual Review of Condensed Matter Physics*, 6, 219-244 (2015).

.. [Marchetti2013] Marchetti, M.C., Joanny, J.F., Ramaswamy, S., Liverpool, T.B., Prost, J.,
                   Rao, M. & Simha, R.A. "Hydrodynamics of soft active matter."
                   *Reviews of Modern Physics*, 85(3), 1143 (2013).

.. [Bechinger2016] Bechinger, C., Di Leonardo, R., Löwen, H., Reichhardt, C., Volpe, G. &
                   Volpe, G. "Active particles in complex and crowded environments."
                   *Reviews of Modern Physics*, 88(4), 045006 (2016).

Robust Optimization Theory
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. [BenTal2009] Ben-Tal, A., El Ghaoui, L. & Nemirovski, A. *Robust Optimization.*
                Princeton University Press (2009).

.. [Delage2010] Delage, E. & Ye, Y. "Distributionally robust optimization under moment
                uncertainty with application to data-driven problems."
                *Operations Research*, 58(3), 595-612 (2010).

.. [Mohajerin2018] Mohajerin Esfahani, P. & Kuhn, D. "Data-driven distributionally robust
                   optimization using the Wasserstein metric: Performance guarantees and
                   tractable reformulations." *Mathematical Programming*, 171(1-2), 115-166 (2018).

Publication Guidelines
----------------------

Academic Use Citation Requirements
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When using the homodyne-analysis package in academic research, please include:

1. **Primary Citation**: The PNAS 2024 paper for the theoretical framework
2. **Software Citation**: The software package citation for computational methods
3. **Acknowledgment**: Recognition of Argonne National Laboratory and funding sources

Example Citation in Paper
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

   Data analysis was performed using the homodyne-analysis package (Chen & He, 2024),
   which implements the theoretical framework for nonequilibrium XPCS analysis
   developed by He et al. (2024). The robust optimization methods employed
   distributionally robust optimization with Wasserstein uncertainty sets to
   account for experimental noise and measurement uncertainty.

   References:
   - Chen, W. & He, H. homodyne-analysis: High-performance XPCS analysis with
     robust optimization. https://github.com/imewei/homodyne (2024).
   - He, H. et al. Transport coefficient approach for characterizing nonequilibrium
     dynamics in soft matter. Proc. Natl. Acad. Sci. U.S.A. 121, e2401162121 (2024).

Acknowledgment Template
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

   We acknowledge the use of the homodyne-analysis software package developed at
   Argonne National Laboratory. This research used resources of the Advanced Photon
   Source, a U.S. Department of Energy (DOE) Office of Science User Facility
   operated for the DOE Office of Science by Argonne National Laboratory under
   Contract No. DE-AC02-06CH11357.

Research Data and Reproducibility
---------------------------------

Data Availability Statement
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The homodyne-analysis package enables full reproducibility of research results through:

**Code Availability**
  Open-source software available at: https://github.com/imewei/homodyne

**Documentation**
  Complete documentation at: https://homodyne.readthedocs.io

**Test Data**
  Example datasets and validation cases included in the repository

**Configuration Files**
  Standardized configuration templates for different analysis modes

Reproducibility Guidelines
~~~~~~~~~~~~~~~~~~~~~~~~~~

To ensure reproducible research with the homodyne-analysis package:

1. **Version Control**: Always specify the exact version used
2. **Configuration**: Include complete configuration files
3. **Environment**: Document the computational environment (Python version, dependencies)
4. **Parameters**: Report all optimization parameters and bounds
5. **Validation**: Include residual analysis and goodness-of-fit metrics

Example Reproducibility Statement
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

   Analysis was performed using homodyne-analysis v0.7.1 with Python 3.12 and
   NumPy 1.24.0. All optimization used the default parameter bounds with robust
   Wasserstein optimization (ε=0.1). Configuration files and analysis scripts
   are available in the supplementary material. The analysis can be reproduced
   using the command: `homodyne --config experiment_config.json --method robust`

Conference Presentations
------------------------

Recommended Presentation Templates
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Title Slide Template**

.. code-block:: text

   Transport Coefficient Analysis of Nonequilibrium Soft Matter
   Using High-Performance XPCS with Robust Optimization

   [Your Name]¹, [Collaborators]
   ¹[Your Institution]

   Based on: He et al., PNAS 121, e2401162121 (2024)
   Software: homodyne-analysis v0.7.1 (Chen & He, 2024)

**Methods Slide Template**

.. code-block:: text

   Computational Methods

   • High-performance XPCS analysis with homodyne-analysis package
   • Numba JIT compilation for 3-5x speedup
   • Distributionally robust optimization for noise resilience
   • Three analysis modes: Static Isotropic (3 params), Static Anisotropic (3 params),
     Laminar Flow (7 params)

   Software: github.com/imewei/homodyne
   Documentation: homodyne.readthedocs.io

Journal-Specific Guidelines
---------------------------

Physical Review Letters
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bibtex

   @article{he2024transport,
     title={Transport coefficient approach for characterizing nonequilibrium dynamics in soft matter},
     author={He, Hongrui and Liang, Hao and Chu, Miaoqi and Jiang, Zhang and de Pablo, Juan J and Tirrell, Matthew V and Narayanan, Suresh and Chen, Wei},
     journal={Proceedings of the National Academy of Sciences},
     volume={121},
     number={31},
     pages={e2401162121},
     year={2024}
   }

Nature Physics
~~~~~~~~~~~~~~

.. code-block:: text

   He, H. et al. Transport coefficient approach for characterizing nonequilibrium
   dynamics in soft matter. Proc. Natl. Acad. Sci. USA 121, e2401162121 (2024).

Science
~~~~~~~

.. code-block:: text

   H. He et al., Proc. Natl. Acad. Sci. USA 121, e2401162121 (2024).

Funding Acknowledgment
----------------------

Required Funding Acknowledgments
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When using this software in federally funded research, include:

.. code-block:: text

   This research was supported by the U.S. Department of Energy, Office of Science,
   Basic Energy Sciences under Contract No. DE-AC02-06CH11357. Use of the Advanced
   Photon Source was supported by the U.S. Department of Energy, Office of Science,
   Office of Basic Energy Sciences.

Optional Extended Acknowledgments
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

   The authors thank Wei Chen and Hongrui He at Argonne National Laboratory for
   developing the homodyne-analysis software package. We acknowledge productive
   discussions with the X-ray Science Division at Argonne National Laboratory
   and the scientific user community of the Advanced Photon Source.

Impact and Metrics
------------------

Citation Tracking
~~~~~~~~~~~~~~~~~

The primary publication and software package can be tracked through:

- **Google Scholar**: Direct citation tracking
- **DOI Metrics**: Crossref citation counts
- **GitHub Metrics**: Stars, forks, and usage statistics
- **ORCID Integration**: Author citation profiles

Research Impact
~~~~~~~~~~~~~~~

The homodyne-analysis package enables research in:

**Scientific Domains**
  - Soft matter physics
  - Nonequilibrium statistical mechanics
  - Active matter systems
  - Complex fluids and rheology

**Methodological Advances**
  - Robust optimization in experimental physics
  - High-performance scientific computing
  - Uncertainty quantification in scattering data

**Community Benefits**
  - Open-source scientific software
  - Reproducible research practices
  - Educational resources for XPCS analysis

For questions about citations, publications, or research collaboration,
contact: Wei Chen (wchen@anl.gov)
