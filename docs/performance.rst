Performance Guide
=================

This comprehensive guide covers performance optimization, monitoring, and best practices for the homodyne package.

.. contents:: Contents
   :depth: 3
   :local:

Performance Overview (v0.6.5+)
===============================

The homodyne package includes comprehensive performance optimizations across all analysis methods: classical optimization, robust optimization, and MCMC sampling. Key features include JIT compilation, JAX backend GPU acceleration with PyTensor environment variable auto-configuration, performance monitoring, and automated benchmarking.

Key Performance Features
------------------------

**JIT Compilation (Numba)**
   - 3-5x speedup for core computational kernels
   - Automatic warmup and caching
   - Optimized for chi-squared calculations and correlation functions

**JAX Backend GPU Acceleration with PyTensor Environment Variable Auto-Configuration**
   - JAX backend handles GPU operations for MCMC sampling via NumPyro (Linux only)
   - PyTensor environment variables automatically configured for CPU mode (avoids C compilation issues)
   - High-performance numerical computations with automatic differentiation
   - Automatic system CUDA GPU detection on Linux systems with NVIDIA GPUs
   - Seamless fallback to CPU when GPU unavailable or on non-Linux platforms

**Performance Monitoring**
   - Built-in profiling decorators
   - Memory usage tracking
   - Performance regression detection
   - Automated benchmarking with statistical analysis

**Optimization-Specific Performance**
   - **Classical**: Optimized angle filtering, vectorized operations
   - **Robust**: CVXPY solver optimization, caching, progressive optimization
   - **MCMC**: JAX backend GPU acceleration with NumPyro + PyTensor CPU mode (Linux only), thinning support, convergence diagnostics

Method Performance Comparison
=============================

**Speed Ranking (fastest to slowest):**

1. **Classical Optimization** (Nelder-Mead, Gurobi) - ~seconds to minutes
   - Best for: Exploratory analysis, parameter screening
   - Trade-offs: No uncertainty quantification, sensitive to local minima

2. **Robust Optimization** (Wasserstein DRO, Scenario-based, Ellipsoidal) - ~2-5x classical
   - Best for: Noisy data, outlier resistance, measurement uncertainty
   - Trade-offs: Slower than classical, requires CVXPY

3. **MCMC Sampling** (NUTS) - ~hours
   - Best for: Full uncertainty quantification, publication-quality results
   - Trade-offs: Slowest method, requires careful convergence assessment

Performance Optimization Strategies
===================================

Classical Optimization
-----------------------

**Angle Filtering Optimization:**

.. code-block:: python

   # Enable smart angle filtering for faster optimization
   config = {
       "optimization_config": {
           "angle_filtering": {
               "enabled": True,
               "target_ranges": [[-10, 10], [170, 190]]
           }
       }
   }

**Gurobi Trust Region Optimization:**

.. code-block:: python

   # Iterative Gurobi with trust region for improved convergence
   config = {
       "optimization_config": {
           "classical_optimization": {
               "methods": ["Gurobi", "Nelder-Mead"],  # Gurobi with trust regions tried first
               "method_options": {
                   "Gurobi": {
                       "max_iterations": 50,  # Outer trust region iterations
                       "tolerance": 1e-6,
                       "trust_region_initial": 0.1,
                       "trust_region_min": 1e-8,
                       "trust_region_max": 1.0
                   }
               }
           }
       }
   }

Robust Optimization
-------------------

**Solver Optimization:**

.. code-block:: python

   # CLARABEL is typically fastest, followed by SCS
   config = {
       "optimization_config": {
           "robust_optimization": {
               "solver_settings": {
                   "preferred_solver": "CLARABEL",
                   "enable_caching": True,
                   "enable_progressive_optimization": True
               }
           }
       }
   }

**Method Selection by Speed:**

1. **Ellipsoidal** - Fastest robust method
2. **Wasserstein DRO** - Moderate speed, good uncertainty modeling
3. **Scenario-based** - Slowest, most robust to outliers

MCMC Optimization
-----------------

**JAX/NumPyro System CUDA GPU Acceleration:**

.. code-block:: python

   # Enable JAX backend for system CUDA GPU acceleration (automatic on Linux with NVIDIA GPU)
   config = {
       "optimization_config": {
           "mcmc_sampling": {
               "use_jax": True,  # Automatically detects system CUDA GPU availability
               "cores": 4        # Multi-core CPU if JAX unavailable
           }
       }
   }

   # Or programmatically:
   from homodyne.optimization.mcmc import HodomyneMCMC

   # System CUDA GPU acceleration is automatic when available
   mcmc = HodomyneMCMC(mode="laminar_flow", use_jax_backend=True)

   # Verify system CUDA GPU detection:
   import jax
   print(f"JAX devices: {jax.devices()}")  # Shows GPU devices if available

**Sampling Efficiency:**

.. code-block:: python

   # Optimized MCMC settings for different problem sizes

   # Static mode (3 parameters)
   static_config = {
       "draws": 8000,
       "tune": 1000,
       "thin": 2,        # Effective samples: 4000
       "chains": 4,
       "target_accept": 0.95
   }

   # Laminar flow (7 parameters)
   flow_config = {
       "draws": 10000,
       "tune": 2000,
       "thin": 1,        # All samples needed for complex posterior
       "chains": 6,
       "target_accept": 0.95
   }

**Memory Optimization:**

.. code-block:: python

   # For memory-constrained systems
   memory_config = {
       "draws": 5000,
       "tune": 1000,
       "thin": 5,        # Effective samples: 1000, lower memory usage
       "chains": 2
   }

Performance Monitoring
======================

Built-in Profiling
-------------------

**Function-level Monitoring:**

.. code-block:: python

   from homodyne.core.profiler import performance_monitor

   @performance_monitor(monitor_memory=True, log_threshold_seconds=0.5)
   def my_analysis_function(data):
       return process_data(data)

   # Get performance statistics
   from homodyne.core.profiler import get_performance_summary
   summary = get_performance_summary()
   print(f"Function called {summary['my_analysis_function']['calls']} times")
   print(f"Average time: {summary['my_analysis_function']['avg_time']:.3f}s")

**Benchmarking Utilities:**

.. code-block:: python

   from homodyne.core.profiler import stable_benchmark

   # Reliable performance measurement with statistical analysis
   results = stable_benchmark(my_function, warmup_runs=5, measurement_runs=15)
   print(f"Mean time: {results['mean']:.4f}s, CV: {results['std']/results['mean']:.3f}")

Performance Testing
===================

**Automated Performance Tests:**

.. code-block:: bash

   # Run performance validation
   python -m pytest -m performance

   # Run regression detection
   python -m pytest -m regression

   # Benchmark with statistical analysis
   python -m pytest -m benchmark --benchmark-only

**Performance Baselines:**

The package maintains performance baselines with excellent stability:

- **Chi-squared calculation**: ~0.8-1.2ms (CV ≤ 0.09)
- **Correlation calculation**: ~0.26-0.28ms (CV ≤ 0.16)
- **Memory efficiency**: Automatic cleanup prevents >50MB accumulation
- **Stability**: 95%+ improvement in coefficient of variation

Environment Optimization
========================

**Threading Configuration:**

.. code-block:: bash

   # Conservative threading for numerical stability (automatically set)
   export NUMBA_NUM_THREADS=4
   export OPENBLAS_NUM_THREADS=4

**JIT Optimization:**

.. code-block:: bash

   # Balanced optimization (automatically configured)
   export NUMBA_FASTMATH=0      # Disabled for numerical stability
   export NUMBA_LOOP_VECTORIZE=1
   export NUMBA_OPT=2           # Moderate optimization level

**Memory Management:**

.. code-block:: bash

   # Numba caching for faster startup
   export NUMBA_CACHE_DIR=~/.numba_cache

Troubleshooting Performance Issues
==================================

**Common Issues and Solutions:**

1. **Slow MCMC Sampling**
   - Enable JAX backend: ``pip install homodyne-analysis[mcmc]``  # Includes JAX with system CUDA GPU support on Linux
   - Remember to run ``source activate_gpu.sh`` before use
   - Reduce problem size: Use angle filtering
   - Optimize MCMC settings: Increase ``thin`` parameter

2. **High Memory Usage**
   - Enable thinning in MCMC: ``"thin": 2`` or higher
   - Use progressive optimization: ``"enable_progressive_optimization": true``
   - Monitor with: ``@performance_monitor(monitor_memory=True)``

3. **Classical Optimization Convergence**
   - Try improved Gurobi solver: ``pip install gurobipy`` (requires license, uses iterative trust region)
   - Adjust tolerances: Lower ``xatol`` and ``fatol`` in config
   - Enable angle filtering: Reduces parameter space complexity
   - Configure trust region: Adjust ``trust_region_initial`` in Gurobi options

4. **Robust Optimization Solver Issues**
   - Install preferred solvers: ``pip install clarabel``
   - Enable fallback: ``"fallback_to_classical": true``
   - Adjust regularization: Lower ``regularization_alpha``

**Performance Profiling:**

.. code-block:: python

   # Profile a complete analysis
   from homodyne.core.profiler import performance_monitor

   @performance_monitor(monitor_memory=True)
   def full_analysis():
       analysis = HomodyneAnalysisCore(config)
       return analysis.optimize_all()

   result = full_analysis()
   # Check logs for performance breakdown

JAX Backend GPU Acceleration with PyTensor Environment Variable Auto-Configuration
====================================================================================

The package provides comprehensive JAX backend GPU acceleration for MCMC sampling with automatic PyTensor environment variable configuration on Linux systems.

**System Requirements (Linux Only)**

- Linux operating system (GPU acceleration not available on Windows/macOS)
- System CUDA 12.6+ installed at ``/usr/local/cuda``
- cuDNN 9.12+ installed in system libraries
- NVIDIA GPU with driver 560.28+
- Virtual environment (conda/mamba/venv/virtualenv) for automatic environment variable configuration

**Unified Post-Install GPU Setup**

The package now includes a unified post-installation system for GPU acceleration setup:

.. code-block:: bash

   # Install with GPU support
   pip install homodyne-analysis[all]
   
   # Run unified post-install setup
   homodyne-post-install --shell zsh --gpu --advanced
   
   # Validate GPU setup
   homodyne-validate --test gpu
   gpu-status  # Check GPU status

**Automatic JAX Backend GPU + PyTensor Environment Variable Configuration**

The unified system automatically configures:

1. **JAX backend**: Installs with system CUDA 12.6+ support for GPU operations
2. **PyTensor environment variables**: Auto-configured for CPU mode (avoids C compilation issues)
3. **Environment integration**: Smart activation/deactivation scripts for all virtual environments
4. **Advanced tools**: homodyne-gpu-optimize for hardware benchmarking

.. code-block:: bash

   # PyTensor environment variables automatically configured:
   # PYTENSOR_FLAGS="device=cpu,floatX=float64,mode=FAST_COMPILE,optimizer=fast_compile,cxx="

**JAX Backend GPU Performance Benefits**

- **MCMC Sampling**: 5-10x speedup with NumPyro/JAX backend GPU acceleration
- **PyTensor Stability**: No C compilation issues (CPU mode with auto-configured environment variables)
- **Vectorized Operations**: Massive parallelization on GPU through JAX backend
- **Multi-chain Sampling**: Efficient parallel chain execution on GPU
- **Large Dataset Processing**: GPU memory enables bigger problems

**Verifying JAX Backend GPU + PyTensor Configuration**

.. code-block:: bash

   # Unified system validation
   homodyne-validate --quick           # Quick system check
   homodyne-validate --test gpu        # GPU-specific tests
   
   # GPU status and benchmarking
   gpu-status                          # Check GPU hardware status
   homodyne-gpu-optimize --benchmark   # GPU performance testing

   # Manual verification - check PyTensor environment variables
   echo $PYTENSOR_FLAGS
   # Should show: device=cpu,floatX=float64,mode=FAST_COMPILE,optimizer=fast_compile,cxx=

.. code-block:: python

   # Then in Python:
   import jax

   # Check available devices
   print(f"JAX devices: {jax.devices()}")
   # Should show: [CudaDevice(id=0), ...] for GPU

   # Check default backend
   print(f"Backend: {jax.default_backend()}")
   # Should show: 'gpu' if GPU is being used

   # Test system CUDA GPU performance
   import jax.numpy as jnp
   x = jnp.ones((1000, 1000))
   y = x @ x  # Matrix multiplication on GPU

**MCMC System CUDA GPU Acceleration**

The MCMC module automatically detects and uses system CUDA GPU when available:

.. code-block:: python

   from homodyne.optimization.mcmc import HodomyneMCMC

   # System CUDA GPU acceleration is automatic
   mcmc = HodomyneMCMC(
       mode="laminar_flow",
       use_jax_backend=True  # Default: True
   )

   # The module will log:
   # INFO - Using JAX backend with NumPyro NUTS for system CUDA GPU acceleration

   # Run sampling (will use system CUDA GPU if available)
   result = mcmc.run_mcmc(
       data=data,
       draws=4000,
       tune=1000,
       chains=4  # Parallel chains on GPU
   )

**GPU Memory Management**

.. code-block:: python

   # Monitor GPU memory usage
   from jax import devices

   # Get GPU memory info
   gpu = devices('gpu')[0]
   memory_stats = gpu.memory_stats()
   print(f"GPU memory used: {memory_stats['bytes_in_use'] / 1e9:.2f} GB")

   # Clear GPU memory if needed
   import gc
   gc.collect()

**Troubleshooting System CUDA GPU Issues**

1. **System CUDA GPU Not Detected**:

   .. code-block:: bash

      # Make sure you activated system CUDA GPU support
      source activate_gpu.sh

      # Check NVIDIA driver
      nvidia-smi

      # Check system CUDA version (should be 12.6+)
      nvcc --version

      # Check cuDNN installation
      ls /usr/lib/x86_64-linux-gnu/libcudnn.so.9*

      # For detailed setup: see GPU_SETUP.md

2. **Out of Memory Errors**:

   - Reduce batch size or number of chains
   - Enable memory-efficient sampling
   - Use CPU for very large problems

3. **Performance Not Improved**:

   - Check if problem size is large enough for system CUDA GPU benefit
   - Verify JAX is using GPU backend
   - Profile to identify bottlenecks

Best Practices
==============

**Development Workflow:**

1. **Start with classical** methods for rapid prototyping
2. **Use angle filtering** to reduce computational complexity
3. **Enable robust methods** for noisy/uncertain data
4. **Run MCMC last** for full uncertainty quantification
5. **Monitor performance** with built-in profiling tools

**Production Deployment:**

1. **Install performance extras**: ``pip install homodyne-analysis[performance,jax]``  # System CUDA GPU support included on Linux
2. **Configure environment variables** for optimal threading
3. **Enable caching** in robust optimization settings
4. **Use appropriate hardware** (NVIDIA GPU with system CUDA 12.6+ for MCMC on Linux, multi-core CPU for classical/robust)
5. **Validate with benchmarks** before deployment

Code Quality and Maintenance
============================

**Code Quality Standards (v0.6.5+):**

The homodyne package maintains high code quality standards with comprehensive tooling:

**Formatting and Style:**

.. code-block:: bash

   # All code formatted with Black (88-character line length)
   black homodyne --line-length 88

   # Import sorting with isort
   isort homodyne --profile black

   # Linting with flake8
   flake8 homodyne --max-line-length 88

   # Type checking with mypy
   mypy homodyne --ignore-missing-imports

**Quality Improvements (Recent):**

- ✅ **Black formatting**: 100% compliant across all files
- ✅ **Import organization**: Consistent import sorting with isort
- ✅ **Code reduction**: Removed 308 lines of unused fallback implementations
- ✅ **Type annotations**: Improved import patterns to resolve mypy warnings
- ✅ **Critical fixes**: Resolved comparison operators and missing function definitions

**Code Statistics:**

.. list-table:: Code Quality Metrics
   :widths: 25 25 25 25
   :header-rows: 1

   * - Tool
     - Status
     - Issues
     - Notes
   * - **Black**
     - ✅ 100%
     - 0
     - 88-char line length
   * - **isort**
     - ✅ 100%
     - 0
     - Sorted and optimized
   * - **flake8**
     - ⚠️ ~400
     - E501, F401
     - Mostly line length and data scripts
   * - **mypy**
     - ⚠️ ~285
     - Various
     - Missing library stubs, annotations

**Development Workflow:**

1. **Pre-commit hooks**: Automatic formatting and linting
2. **Continuous integration**: Code quality checks on all PRs
3. **Performance regression detection**: Automated benchmarking
4. **Test coverage**: Comprehensive test suite with 95%+ coverage
5. **Documentation**: Sphinx-based documentation with examples

**Performance and Quality Balance:**

The package achieves both high performance and maintainable code through:

- **Optimized algorithms**: Trust region Gurobi, vectorized operations
- **Clean architecture**: Modular design with clear separation of concerns
- **Comprehensive testing**: Unit, integration, and performance tests
- **Documentation**: Detailed API documentation and user guides

The homodyne package is designed for **high-performance scientific computing** with comprehensive optimization strategies and maintainable, high-quality code.
