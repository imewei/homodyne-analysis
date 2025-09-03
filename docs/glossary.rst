Glossary
========

.. index:: glossary, terminology, definitions

Comprehensive glossary of terms used in homodyne analysis and this package.

.. contents:: Quick Navigation
   :local:
   :depth: 1

Analysis Terms
--------------

.. glossary::

   Homodyne Analysis
      Method for analyzing X-ray Photon Correlation Spectroscopy (XPCS) data in flowing systems to characterize transport properties like diffusion and shear.

   Static Isotropic Mode
      Fastest analysis mode (3 parameters) for systems with no angular dependencies or flow. Analyzes: D₀, α, D_offset.

   Static Anisotropic Mode  
      Analysis mode (3 parameters) for systems with angular dependencies but no flow. Includes angle filtering.

   Laminar Flow Mode
      Complete analysis mode (7 parameters) for flowing systems. Analyzes: D₀, α, D_offset, γ̇₀, β, γ̇_offset, φ₀.

   Angle Filtering
      Optimization technique that analyzes only specific angular ranges to improve performance and focus on physically relevant orientations.

   Time-dependent Diffusion
      Diffusion coefficient D(t) that varies over time, often following power-law: D(t) = D₀(t/t₀)^α.

   Shear Rate
      Rate of deformation γ̇(t) in flowing fluids, measured in reciprocal seconds [s⁻¹].

   Correlation Function
      Mathematical function c₂(φ,t₁,t₂) describing temporal correlations in scattered intensity as a function of angle φ and time delays.

Technical Terms
---------------

.. glossary::

   Isolated Backend Architecture
      Revolutionary system design that completely separates PyMC CPU and NumPyro/JAX GPU implementations to prevent PyTensor/JAX namespace conflicts.

   CPU Backend
      Pure PyMC implementation (``mcmc_cpu_backend.py``) for Bayesian analysis, completely isolated from JAX dependencies. Cross-platform compatible.

   GPU Backend
      Pure NumPyro/JAX implementation (``mcmc_gpu_backend.py``) for high-performance GPU-accelerated Bayesian analysis, completely isolated from PyMC dependencies.

   Backend Selection
      Process of automatically choosing between CPU and GPU backends based on environment variables (``HOMODYNE_GPU_INTENT``) and hardware availability.

   PyTensor/JAX Conflicts
      Namespace and compilation conflicts that occurred when PyMC (PyTensor) and JAX were used in the same process. Resolved by isolated backend architecture.

   MCMC
      Markov Chain Monte Carlo - Bayesian sampling method for uncertainty quantification using NUTS (No-U-Turn Sampler).

   NUTS Sampler
      No-U-Turn Sampler - advanced MCMC algorithm used for efficient Bayesian parameter estimation.

   R-hat (R̂)
      Convergence diagnostic for MCMC chains. Values < 1.1 indicate good convergence, < 1.05 excellent.

   Effective Sample Size (ESS)
      Number of effectively independent samples in MCMC chains. ESS > 400 recommended for reliable inference.

Parameters and Physics
----------------------

.. glossary::

   D₀ (D0)
      Reference diffusion coefficient at reference time, units: [Å²/s]. Typical values: 1000-10000.

   α (alpha)
      Time dependence exponent for diffusion D(t) = D₀(t/t₀)^α. Dimensionless. Typical values: -2.0 to 0.0.

   D_offset
      Baseline diffusion component, units: [Å²/s]. Accounts for instrumental or systematic effects.

   γ̇₀ (gamma_dot_t0)
      Reference shear rate at reference time, units: [s⁻¹]. Typical values: 1e-6 to 1e-2.

   β (beta)
      Shear exponent for time-dependent shear rate γ̇(t) = γ̇₀(t/t₀)^β. Dimensionless.

   γ̇_offset (gamma_dot_t_offset)
      Baseline shear component, units: [s⁻¹]. Accounts for residual flow effects.

   φ₀ (phi0)
      Angular offset parameter, units: [degrees]. Accounts for alignment between flow and scattering geometry.

   Contrast
      Scaling parameter in c₂_fitted = c₂_theory × contrast + offset. Dimensionless, range: (0.05, 0.5].

   Offset
      Baseline parameter in scaling transformation. Dimensionless, range: (0.05, 1.95).

   q-vector
      Scattering wave vector, units: [Å⁻¹]. Determines length scale probed by scattering experiment.

   Scattering Angle (φ)
      Angle between flow direction and scattering wave vector, units: [degrees].

Optimization Methods
--------------------

.. glossary::

   Classical Optimization
      Deterministic optimization methods: Nelder-Mead simplex and Gurobi trust-region methods. Fastest, provides point estimates.

   Nelder-Mead
      Gradient-free optimization algorithm using simplex method. Robust for noisy objective functions.

   Gurobi Optimization
      Commercial optimization solver using iterative trust-region approach. Requires license.

   Robust Optimization
      Optimization methods designed to handle noise and outliers: Wasserstein DRO, Scenario-based, Ellipsoidal.

   Wasserstein DRO
      Distributionally Robust Optimization using Wasserstein distance to handle data uncertainty.

   Scenario-based Optimization
      Robust method considering multiple data scenarios to find solutions robust to variability.

   Ellipsoidal Uncertainty
      Robust optimization approach modeling parameter uncertainty as ellipsoidal sets.

   Bayesian Analysis
      Statistical approach using MCMC sampling to quantify parameter uncertainties via posterior distributions.

   Posterior Distribution
      Probability distribution of parameters given data and priors, obtained through Bayesian inference.

   Prior Distribution
      Initial probability distribution of parameters before observing data. Uses Normal and TruncatedNormal distributions.

   Credible Interval
      Bayesian equivalent of confidence interval. Probability range containing true parameter with specified probability.

Software Architecture
---------------------

.. glossary::

   Configuration Manager
      Class responsible for loading, validating, and managing JSON configuration files.

   Analysis Core
      Main orchestrator class (``HomodyneAnalysisCore``) that coordinates data loading, optimization, and results generation.

   JIT Compilation
      Just-In-Time compilation using Numba for 3-5x performance acceleration of computational kernels.

   Numba
      Python compiler that translates functions to optimized machine code for significant performance improvements.

   JAX
      Google's library for high-performance computing with automatic differentiation and GPU support.

   PyMC
      Probabilistic programming framework for Bayesian statistical modeling using advanced MCMC methods.

   NumPyro
      Probabilistic programming library built on JAX, providing GPU-accelerated Bayesian inference.

   Shell Completion
      Advanced command-line completion system with context awareness, caching, and cross-shell compatibility.

   Post-installation System
      Unified setup system for shell completion, GPU acceleration, and advanced CLI tools.

   GPU Optimization
      Hardware-specific optimization including CUDA detection, memory tuning, and performance benchmarking.

Data and File Formats
----------------------

.. glossary::

   HDF5 Format
      Hierarchical Data Format version 5. Recommended format for c₂ correlation data. Extension: .h5, .hdf5.

   NPZ Format
      NumPy compressed archive format. Alternative format for correlation data. Extension: .npz.

   PyXPCS Format
      Specific HDF5 structure used by PyXPCS software for XPCS data analysis.

   Correlation Data
      Experimental c₂(φ,t₁,t₂) data with shape (n_angles, n_tau_times, n_delay_times).

   Tau Values
      Correlation lag times, shape (n_tau_times,), typically logarithmically spaced from microseconds to seconds.

   Delay Values
      Time delays between correlation measurements, shape (n_delay_times,), optional dimension.

   Angle File
      Simple text file containing scattering angles in degrees, one angle per line.

   Configuration File
      JSON file specifying analysis parameters, file paths, and optimization settings.

   Results File
      JSON output containing fitted parameters, uncertainties, fit quality metrics, and metadata.

Performance and System
-----------------------

.. glossary::

   CUDA
      NVIDIA's parallel computing platform enabling GPU acceleration. Required for GPU backend on Linux.

   cuDNN
      NVIDIA's Deep Neural Network library. Required for optimal JAX performance on GPU.

   System Validation
      Comprehensive testing of installation, backends, shell completion, and hardware capabilities.

   Performance Monitoring
      Built-in system for tracking execution times, memory usage, and optimization convergence.

   Benchmarking
      Systematic performance testing with statistical analysis, outlier filtering, and stability assessment.

   Warm-up
      Pre-compilation of JIT kernels to eliminate first-run overhead and ensure consistent performance.

   Memory Management
      Optimization strategies for large datasets including chunking, lazy loading, and garbage collection.

   Threading
      Parallel execution using multiple CPU cores. Controlled by environment variables like ``OMP_NUM_THREADS``.

Environment Variables
---------------------

.. glossary::

   HOMODYNE_GPU_INTENT
      Environment variable controlling backend selection. "true" for GPU backend, "false" for CPU backend.

   JAX_ENABLE_X64
      JAX precision control. "0" for float32 (GPU performance), "1" for float64 (CPU precision).

   XLA_PYTHON_CLIENT_MEM_FRACTION
      Controls GPU memory allocation for JAX. Values: 0.1-0.9, default depends on GPU memory.

   OMP_NUM_THREADS
      OpenMP thread count for CPU parallelization. Recommended: 4 for stability.

   PYTENSOR_FLAGS
      PyTensor configuration flags. Automatically configured for CPU mode in isolated architecture.

   XLA_FLAGS
      Advanced XLA compiler flags for GPU optimization. Used for performance tuning.

Error Messages and Diagnostics
-------------------------------

.. glossary::

   Import Error
      Python error when required modules cannot be loaded. Usually indicates missing dependencies.

   Backend Not Available
      Error when requested MCMC backend (CPU or GPU) dependencies are not installed.

   GPU Not Detected
      Warning when CUDA-capable GPU is not found or drivers are missing. Falls back to CPU.

   Configuration Error
      Error in JSON configuration file syntax, missing required fields, or invalid parameter values.

   Convergence Error
      Error when optimization algorithms fail to find solution or MCMC chains fail to converge.

   File Not Found Error
      Error when data files specified in configuration cannot be located or accessed.

   Memory Error
      Error when system runs out of RAM or GPU memory during analysis. Requires optimization.

   Namespace Conflict
      Historical error when PyTensor and JAX conflicted. Resolved by isolated backend architecture.

See Also
--------

* :doc:`user-guide/quick-reference` - Commands and code examples
* :doc:`user-guide/analysis-modes` - Detailed mode descriptions  
* :doc:`api-reference/mcmc` - MCMC API and backend details
* :doc:`developer-guide/architecture` - System design and patterns
* :doc:`user-guide/troubleshooting-flowchart` - Problem diagnosis