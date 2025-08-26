Performance Guide
=================

This comprehensive guide covers performance optimization, monitoring, and best practices for the homodyne package.

.. contents:: Contents
   :depth: 3
   :local:

Performance Overview (v0.6.5+)
===============================

The homodyne package includes comprehensive performance optimizations across all analysis methods: classical optimization, robust optimization, and MCMC sampling. Key features include JIT compilation, JAX acceleration, performance monitoring, and automated benchmarking.

Key Performance Features
------------------------

**JIT Compilation (Numba)**
   - 3-5x speedup for core computational kernels
   - Automatic warmup and caching
   - Optimized for chi-squared calculations and correlation functions

**JAX Backend Integration**
   - GPU/TPU acceleration for MCMC sampling
   - High-performance numerical computations
   - Automatic fallback to CPU when needed

**Performance Monitoring**
   - Built-in profiling decorators
   - Memory usage tracking
   - Performance regression detection
   - Automated benchmarking with statistical analysis

**Optimization-Specific Performance**
   - **Classical**: Optimized angle filtering, vectorized operations
   - **Robust**: CVXPY solver optimization, caching, progressive optimization
   - **MCMC**: JAX/NumPyro acceleration, thinning support, convergence diagnostics

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

**Solver Selection:**

.. code-block:: python

   # Gurobi is faster than Nelder-Mead for smooth problems (requires license)
   config = {
       "optimization_config": {
           "classical_optimization": {
               "methods": ["Gurobi", "Nelder-Mead"]  # Gurobi tried first
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

**JAX Acceleration:**

.. code-block:: python

   # Enable JAX backend for GPU acceleration
   config = {
       "optimization_config": {
           "mcmc_sampling": {
               "use_jax": True,  # Automatically detects GPU availability
               "cores": 4        # Multi-core CPU if JAX unavailable
           }
       }
   }

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
   - Enable JAX backend: ``pip install jax jaxlib``
   - Reduce problem size: Use angle filtering
   - Optimize MCMC settings: Increase ``thin`` parameter

2. **High Memory Usage**
   - Enable thinning in MCMC: ``"thin": 2`` or higher
   - Use progressive optimization: ``"enable_progressive_optimization": true``
   - Monitor with: ``@performance_monitor(monitor_memory=True)``

3. **Classical Optimization Convergence**
   - Try Gurobi solver: ``pip install gurobipy`` (requires license)
   - Adjust tolerances: Lower ``xatol`` and ``fatol`` in config
   - Enable angle filtering: Reduces parameter space complexity

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

Best Practices
==============

**Development Workflow:**

1. **Start with classical** methods for rapid prototyping
2. **Use angle filtering** to reduce computational complexity  
3. **Enable robust methods** for noisy/uncertain data
4. **Run MCMC last** for full uncertainty quantification
5. **Monitor performance** with built-in profiling tools

**Production Deployment:**

1. **Install performance extras**: ``pip install homodyne-analysis[performance,jax]``
2. **Configure environment variables** for optimal threading
3. **Enable caching** in robust optimization settings
4. **Use appropriate hardware** (GPU for MCMC, multi-core for classical/robust)
5. **Validate with benchmarks** before deployment

The homodyne package is designed for **high-performance scientific computing** with comprehensive optimization strategies across all analysis methods.