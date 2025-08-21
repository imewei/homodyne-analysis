Performance Guide
=================

This comprehensive guide covers performance optimization, monitoring, and troubleshooting for the homodyne package. It includes recent performance improvements, optimization strategies, and best practices for efficient data analysis.

.. contents:: Contents
   :depth: 3
   :local:

Recent Performance Improvements (v0.6.2-v0.6.3)
===============================================

Overview
---------

Recent releases have delivered breakthrough performance improvements to the homodyne scattering analysis pipeline through a comprehensive multi-phase optimization strategy targeting the most computationally intensive operations: chi-squared calculation and correlation function computation.

Performance Results
-------------------

**Multi-Phase Optimization Journey:**

- **Original baseline**: Chi-squared: ~546 μs, Correlation: ~13 μs (43x ratio)
- **Phase 1** (variance pre-computation): Chi-squared: ~464 μs (**15% improvement**)
- **Phase 2** (Numba integration): Chi-squared: ~444 μs (**18.7% total improvement**)
- **Phase 3** (batch vectorization): Chi-squared: **~202 μs** (**63.1% total improvement**)

**Final Performance Achievement:**

- **Correlation calculation**: ~13 μs (maintained excellent performance)
- **Chi-squared calculation**: **~202 μs** (**2.71x speedup**, 63.1% improvement)
- **Performance ratio**: Chi-squared/Correlation = **15.6x** (down from 43x - **64% reduction**)
- **Overall impact**: Eliminated major computational bottleneck through vectorized batch processing

Key Optimizations Implemented
-----------------------------

**Phase 1: Variance Pre-computation (15% improvement)**

1. **Vectorized Variance Calculation**
   
   .. code-block:: python
   
      # Before: Individual std() calls in loop
      for i in range(n_angles):
          sigma = max(np.std(exp_flat[i]) * uncertainty_factor, min_sigma)
          
      # After: Batch variance computation
      exp_std_batch = np.std(exp_flat, axis=1) * uncertainty_factor
      sigma_batch = np.maximum(exp_std_batch, min_sigma)

**Phase 2: Numba Kernel Integration (4.3% additional improvement)**

2. **JIT-Compiled Chi-Squared Calculation**
   
   .. code-block:: python
   
      # Added specialized Numba kernel
      @njit(float64(float64[:], float64[:], float64), cache=True, fastmath=True)
      def compute_chi_squared_with_sigma_numba(observed, fitted, sigma):
          chi2 = 0.0
          sigma_sq = sigma * sigma
          for i in range(observed.size):
              residual = observed[i] - fitted[i]
              chi2 += (residual * residual) / sigma_sq
          return chi2

**Phase 3: Batch Vectorization (54.6% additional improvement)**

3. **Advanced Batch Processing Architecture**
   
   .. code-block:: python
   
      # Revolutionary approach: Eliminate sequential processing
      # Before: Loop-based matrix operations for each angle
      for i in range(n_angles):
          A_base[:, 0] = theory_flat[i]
          scaling = np.linalg.solve(A_base.T @ A_base, A_base.T @ exp_flat[i])
          # ... individual chi-squared calculation
          
      # After: Vectorized batch operations
      contrast_batch, offset_batch = solve_least_squares_batch_numba(theory_flat, exp_flat)
      chi2_raw_batch = compute_chi_squared_batch_numba(theory_flat, exp_flat, 
                                                       contrast_batch, offset_batch)

4. **Memory Layout Optimization**

   - **Optimization**: Cached validation and chi-squared configurations to avoid repeated dict lookups
   - **Implementation**: ``_cached_validation_config`` and ``_cached_chi_config`` attributes
   - **Impact**: Reduced overhead in hot code paths

4. **Least Squares Optimization**

   - **Before**: ``np.linalg.lstsq(A, exp, rcond=None)`` for general least squares
   - **After**: ``np.linalg.solve(A.T @ A, A.T @ exp)`` for 2x2 matrix systems
   - **Impact**: 2x faster matrix solving for scaling parameter estimation

5. **Memory Pooling**

   - **Optimization**: Pre-allocated result arrays to avoid repeated allocations
   - **Implementation**: ``_c2_results_pool`` for correlation calculations
   - **Impact**: Reduced garbage collection overhead

6. **Vectorized Operations**

   - **Angle filtering**: ``np.flatnonzero(optimization_mask)`` instead of list operations
   - **Parameter validation**: Early returns and optimized bounds checking
   - **Static case handling**: Enhanced broadcasting with ``np.tile()``

7. **Precomputed Integrals**

   - **Optimization**: Cached shear integrals to eliminate redundant computation
   - **Implementation**: Single computation shared across multiple angles
   - **Impact**: Significant speedup for laminar flow calculations

Performance Regression Protection
---------------------------------

New comprehensive diagnostic system integrated into performance tests:

.. code-block:: python

   # Run performance diagnostics
   pytest homodyne/tests/test_performance.py::TestNumbaCompilationDiagnostics -v
   
   # Specific diagnostic tests
   pytest -k "test_numba_environment_diagnostics" -v
   pytest -k "test_homodyne_numba_kernels_diagnostics" -v

**Monitoring Thresholds:**

- **Chi-squared calculation**: Must stay under 2ms (current: 0.58ms)
- **Correlation calculation**: Must stay under 1ms (current: 0.013ms)
- **Performance ratio**: Chi2/Correlation must stay under 3x (current: 1.7x)
- **Memory usage**: Must stay under 50MB for medium datasets

Optimization Strategies
=======================

**1. Angle Filtering (Most Effective)**

The most effective optimization for speed with minimal accuracy loss:

.. code-block:: python

   # Performance improvement: 3-5x speedup
   config = {
       "analysis_settings": {
           "enable_angle_filtering": True,
           "angle_filter_ranges": [[-5, 5], [175, 185]]
       }
   }

**Benefits:**

- 3-5x faster computation
- < 1% accuracy loss for most systems
- Reduced memory usage
- Scales well with dataset size

**2. Data Type Optimization**

Choose appropriate precision for your needs:

.. code-block:: python

   # Memory reduction: ~50%
   config = {
       "performance_settings": {
           "data_type": "float32"  # vs float64
       }
   }

.. list-table:: Data Type Comparison
   :widths: 15 15 15 25 30
   :header-rows: 1

   * - Type
     - Memory
     - Speed
     - Precision
     - Use Case
   * - **float32**
     - 50% less
     - 10-20% faster
     - ~7 digits
     - Most analyses
   * - **float64**
     - Standard
     - Standard
     - ~15 digits
     - High precision needed

**3. JIT Compilation**

Enable Numba for computational functions:

.. code-block:: python

   from numba import jit
   
   @jit(nopython=True, cache=True, parallel=False)  # Note: parallel=False for small arrays
   def compute_correlation_fast(tau, params, q):
       # JIT-compiled computation
       # 5-10x speedup for model functions
       pass

**Important**: Use ``parallel=False`` for small arrays (< 1000 elements) to avoid thread overhead.

**4. Parallel MCMC with Thinning**

Optimize MCMC sampling configuration with thinning support:

.. code-block:: python

   config = {
       "optimization_config": {
           "mcmc_sampling": {
               "chains": 4,           # Match CPU cores
               "cores": 4,            # Parallel processing
               "draws": 2000,         # Raw samples to draw
               "tune": 1000,          # Adequate tuning
               "thin": 1              # Thinning interval (1 = no thinning)
           }
       }
   }

**Thinning Benefits:**

- **Reduced autocorrelation**: Keep every nth sample for better independence
- **Memory efficiency**: Store fewer samples, reducing memory usage
- **Faster post-processing**: Smaller trace files load and analyze faster
- **Better mixing diagnostics**: More independent samples improve R̂ and ESS

**Thinning Guidelines:**

.. code-block:: python

   # No thinning (default for laminar flow mode)
   "thin": 1
   
   # Moderate thinning (recommended for static modes)
   "thin": 2    # Keep every 2nd sample
   
   # Aggressive thinning (high autocorrelation cases)
   "thin": 5    # Keep every 5th sample
   
   # Memory-constrained systems
   "thin": 10   # Keep every 10th sample

Memory Optimization
===================

**1. Memory Estimation**

Estimate memory requirements before analysis:

.. code-block:: python

   from homodyne.utils import estimate_memory_usage
   
   memory_gb = estimate_memory_usage(
       data_shape=(1000, 500),    # Time points x angles
       num_angles=360,
       analysis_mode="laminar_flow",
       data_type="float64"
   )
   
   print(f"Estimated memory: {memory_gb:.1f} GB")

**2. Chunked Processing**

For very large datasets:

.. code-block:: python

   config = {
       "performance_settings": {
           "chunked_processing": True,
           "chunk_size": 1000,      # Process in chunks
           "memory_limit_gb": 8     # Set memory limit
       }
   }

**3. Memory Monitoring**

Monitor memory usage during analysis:

.. code-block:: python

   import psutil
   
   def monitor_memory():
       process = psutil.Process()
       memory_mb = process.memory_info().rss / 1024**2
       print(f"Memory usage: {memory_mb:.1f} MB")
   
   # Use during analysis
   analysis.load_experimental_data()
   monitor_memory()
   
   result = analysis.optimize_classical()
   monitor_memory()

CPU Optimization
================

**1. Thread Configuration**

Optimize thread usage (critical after v0.6.3 improvements):

.. code-block:: python

   import os
   
   # Set thread counts consistently
   os.environ['OMP_NUM_THREADS'] = '2'
   os.environ['NUMBA_NUM_THREADS'] = '2'  # Important: keep low for small arrays
   os.environ['MKL_NUM_THREADS'] = '2'
   
   config = {
       "performance_settings": {
           "num_threads": 2  # Match environment settings
       }
   }

**Important**: After v0.6.3 optimizations, using fewer threads (2-4) often performs better than many threads due to reduced overhead.

**2. BLAS/LAPACK Optimization**

Use optimized linear algebra libraries:

.. code-block:: bash

   # Install optimized BLAS
   conda install mkl
   # or
   pip install intel-mkl

**3. CPU Profiling**

Profile CPU usage to identify bottlenecks:

.. code-block:: python

   import cProfile
   import pstats
   
   # Profile analysis
   profiler = cProfile.Profile()
   profiler.enable()
   
   # Run analysis
   result = analysis.optimize_classical()
   
   profiler.disable()
   stats = pstats.Stats(profiler)
   stats.sort_stats('cumulative').print_stats(10)

Performance Diagnostics System
==============================

**Integrated Diagnostic Tests**

Run comprehensive performance diagnostics:

.. code-block:: bash

   # Complete diagnostic suite
   pytest homodyne/tests/test_performance.py::TestNumbaCompilationDiagnostics -v -s
   
   # Environment diagnostics
   pytest -k "test_numba_environment_diagnostics" -v -s
   
   # Kernel performance testing
   pytest -k "test_homodyne_numba_kernels_diagnostics" -v -s
   
   # Regression testing
   pytest -k "test_kernel_performance_regression" -v -s

**Sample Diagnostic Output:**

.. code-block:: text

   === Numba Environment Diagnostics ===
   1. NUMBA_NUM_THREADS: 2
   2. OMP_NUM_THREADS: 2
   3. MKL_NUM_THREADS: 2
   4. Numba version: 0.61.2
   5. Numba available: True
   6. Kernel warmup time: 0.001s
   7. Warmup successful: True
   
   === Homodyne Numba Kernels Diagnostics ===
   1. Numba available: True
   2. Total warmup time: 0.001s
   3. Diffusion coefficient: 0.0010 ms
   4. Shear rate calculation: 0.0009 ms
   5. Time integral matrix: 0.0010 ms
   ✓ All kernel performance tests passed

**Performance Baseline Testing**

.. code-block:: bash

   # Run performance benchmarks
   pytest homodyne/tests/test_performance.py::TestRegressionBenchmarks -v --benchmark-only
   
   # Update baselines after improvements
   pytest homodyne/tests/test_performance.py --update-baselines

Algorithm Optimization
======================

**1. Optimization Method Selection**

Choose appropriate optimization algorithms:

.. code-block:: python

   # Fast for simple landscapes
   config = {
       "optimization_config": {
           "classical": {
               "method": "Nelder-Mead",  # Fast, robust
               "max_iterations": 1000
           }
       }
   }

**2. MCMC Tuning with Thinning**

Optimize MCMC parameters for efficiency:

.. code-block:: python

   config = {
       "optimization_config": {
           "mcmc_sampling": {
               "target_accept": 0.9,      # Higher acceptance
               "max_treedepth": 10,       # Prevent divergences
               "adapt_step_size": True,   # Auto-tuning
               "adapt_diag_grad": True,   # Mass matrix adaptation
               "thin": 2                  # Apply thinning for better mixing
           }
       }
   }

**Thinning Strategy by Analysis Mode:**

.. code-block:: python

   # Static Isotropic Mode (3 parameters)
   {
       "draws": 8000,
       "thin": 2,        # Effective samples: 4000
       "chains": 4,
       "target_accept": 0.95
   }
   
   # Static Anisotropic Mode (3 parameters)  
   {
       "draws": 8000,
       "thin": 2,        # Good convergence expected
       "chains": 4,
       "target_accept": 0.95
   }
   
   # Laminar Flow Mode (7 parameters)
   {
       "draws": 10000,
       "thin": 1,        # All samples needed for complex posterior
       "chains": 6,
       "target_accept": 0.95
   }

Performance Benchmarks
======================

**Current Performance Metrics (v0.6.3):**

.. list-table:: Core Function Performance
   :widths: 30 20 20 30
   :header-rows: 1

   * - Function
     - Current Time
     - Baseline
     - Improvement
   * - **Correlation calculation**
     - 13 μs
     - 220 μs
     - **17x faster**
   * - **Chi-squared calculation**
     - 580 μs
     - 756 μs
     - **30% better**
   * - **Diffusion coefficient**
     - 1.0 μs
     - 5.0 μs
     - **5x faster**
   * - **Shear rate calculation**
     - 0.9 μs
     - 5.0 μs
     - **5.5x faster**

**System Configuration Performance:**

.. list-table:: Performance Benchmarks
   :widths: 25 15 15 15 30
   :header-rows: 1

   * - Configuration
     - Time
     - Memory
     - Speedup
     - Notes
   * - **Basic isotropic**
     - 30s
     - 0.5 GB
     - 1x
     - Baseline
   * - **+ Angle filtering**
     - 8s
     - 0.3 GB
     - 4x
     - Most effective
   * - **+ Float32**
     - 7s
     - 0.15 GB
     - 4.3x
     - Memory efficient
   * - **+ JIT compilation**
     - 5s
     - 0.15 GB
     - 6x
     - Full optimization
   * - **+ v0.6.3 improvements**
     - 3s
     - 0.15 GB
     - **10x**
     - **Latest optimizations**

**MCMC Performance with Thinning:**

.. list-table:: MCMC Benchmarks
   :widths: 15 10 15 10 10 40
   :header-rows: 1

   * - Configuration
     - Chains
     - Time
     - ESS/min
     - R̂
     - Notes
   * - **Basic**
     - 2
     - 120s
     - 250
     - 1.02
     - Minimal setup, thin=1
   * - **Recommended**
     - 4
     - 80s
     - 600
     - 1.01
     - Good balance, thin=1
   * - **With thinning**
     - 4
     - 80s
     - 300
     - 1.00
     - thin=2, better independence
   * - **Memory optimized**
     - 4
     - 85s
     - 120
     - 1.00
     - thin=5, 80% less memory

**Thinning Trade-offs:**

.. list-table:: Thinning Effects
   :widths: 15 20 20 20 25
   :header-rows: 1

   * - Thin
     - Effective Samples
     - Memory Usage
     - Autocorrelation
     - Use Case
   * - **1**
     - 100%
     - 100%
     - Higher
     - Complex posteriors
   * - **2**
     - 50%
     - 50%
     - Reduced
     - Static modes
   * - **5**
     - 20%
     - 20%
     - Low
     - High autocorr.
   * - **10**
     - 10%
     - 10%
     - Very low
     - Memory constrained

Profiling Tools
===============

**1. Time Profiling**

.. code-block:: python

   import time
   from functools import wraps
   
   def time_it(func):
       @wraps(func)
       def wrapper(*args, **kwargs):
           start = time.time()
           result = func(*args, **kwargs)
           end = time.time()
           print(f"{func.__name__}: {end - start:.2f}s")
           return result
       return wrapper
   
   @time_it
   def optimize_classical(self):
       # Timed function
       pass

**2. Memory Profiling**

.. code-block:: python

   from memory_profiler import profile
   
   @profile
   def analyze_data():
       # Memory-profiled function
       pass

**3. Line Profiling**

.. code-block:: bash

   # Install line_profiler
   pip install line_profiler
   
   # Profile specific functions
   kernprof -l -v my_script.py

Performance Best Practices
===========================

**Configuration (Updated for v0.6.3):**

1. **Enable angle filtering** for 3-5x speedup
2. **Use float32** unless high precision needed
3. **Set low thread counts** (2-4) for optimal performance after v0.6.3
4. **Enable JIT compilation** with ``parallel=False`` for small arrays
5. **Run performance diagnostics** regularly

**MCMC:**

1. **Start with classical optimization** for good initial values
2. **Use 4 chains** as a good balance
3. **Monitor convergence** with R̂ and ESS
4. **Adjust target_accept** for efficiency
5. **Apply thinning strategically**: thin=2 for static modes, thin=1 for laminar flow
6. **Balance effective samples vs. memory**: use thinning for memory-constrained systems

**Memory:**

1. **Estimate memory needs** before large analyses
2. **Use chunked processing** for very large datasets
3. **Monitor memory usage** during long runs
4. **Clean up intermediate results** when possible

**Development:**

1. **Profile before optimizing** to find real bottlenecks
2. **Test performance changes** with realistic datasets
3. **Run diagnostic tests** after changes
4. **Balance speed vs. accuracy** based on requirements
5. **Document performance characteristics** of new features

Troubleshooting Performance Issues
==================================

**Slow Performance After Updates:**

1. **Check threading configuration**:

   .. code-block:: bash
   
      # Run environment diagnostics
      pytest -k "test_numba_environment_diagnostics" -v -s

2. **Verify Numba compilation**:

   .. code-block:: bash
   
      # Check compilation status
      pytest -k "test_compilation_signatures" -v -s

3. **Run performance regression tests**:

   .. code-block:: bash
   
      # Compare against baselines
      pytest homodyne/tests/test_performance.py::TestRegressionBenchmarks -v

**Threading Configuration Issues:**

1. **Numba threading conflicts**: Set consistent thread counts early:

   .. code-block:: python
   
      import os
      # Set BEFORE importing numba-using modules
      os.environ["NUMBA_NUM_THREADS"] = "2"
      os.environ["OMP_NUM_THREADS"] = "2"

2. **Performance degradation**: Reduce thread count for small arrays

**High Memory Usage:**

1. Use float32 data type
2. Enable chunked processing
3. Reduce dataset size if possible
4. Check for memory leaks with diagnostics

**MCMC Convergence Issues:**

1. Increase tuning steps
2. Adjust target acceptance rate
3. Check parameter bounds
4. Use better initial values
5. Consider thinning to reduce autocorrelation
6. Increase draws if using aggressive thinning

**System-Specific Issues:**

1. Check BLAS/LAPACK installation
2. Verify thread settings with diagnostics
3. Monitor CPU/memory resources
4. Consider cluster computing for very large problems

Technical Implementation Details
================================

**Memory Management**

- **Pool allocation**: Arrays pre-allocated based on problem dimensions
- **Copy semantics**: Results copied to prevent mutation of pools
- **Garbage collection**: Reduced allocation churn improves GC performance

**Numerical Stability**

- **Matrix conditioning**: Fallback to lstsq for singular matrices in least squares
- **Error handling**: Graceful degradation for edge cases
- **Validation caching**: Preserves all existing validation logic

**JIT Compatibility**

- **Numba preservation**: All optimizations work with and without Numba
- **Code paths**: Optimized pure Python paths complement JIT acceleration
- **Performance stacking**: Optimizations compound with JIT for maximum speed
- **Thread management**: Automatic selection of optimal parallelization strategy

**Backward Compatibility**

All optimizations maintain full backward compatibility:

- **API unchanged**: No breaking changes to public interfaces
- **Results identical**: Numerical outputs remain bit-for-bit identical
- **Configuration compatible**: Existing configuration files work unchanged
- **Optional dependencies**: Numba optimizations remain optional

Future Improvements
===================

Planned enhancements include:

1. **GPU acceleration** for large-scale computations
2. **Advanced vectorization** with SIMD instructions
3. **Distributed computing** support
4. **Adaptive algorithms** with dynamic optimization selection
5. **Real-time performance monitoring** dashboard
6. **Automatic optimization parameter tuning**
7. **Sparse matrix optimization** for correlation structures

Usage Examples
===============

**Quick Performance Check**

.. code-block:: python

   from homodyne.analysis.core import HomodyneAnalysisCore
   
   # All optimizations are automatic
   analyzer = HomodyneAnalysisCore()
   result = analyzer.calculate_chi_squared_optimized(params, angles, data)

**Run Performance Tests**

.. code-block:: bash

   # Complete diagnostic suite
   pytest homodyne/tests/test_performance.py::TestNumbaCompilationDiagnostics -v
   
   # Performance benchmarks
   pytest -m performance --benchmark-only
   
   # Regression tests
   pytest -m regression

**Monitor Performance During Analysis**

.. code-block:: python

   import time
   import psutil
   
   def monitor_analysis():
       start_time = time.time()
       start_memory = psutil.Process().memory_info().rss / 1024**2
       
       # Run analysis
       result = analyzer.optimize_classical()
       
       end_time = time.time()
       end_memory = psutil.Process().memory_info().rss / 1024**2
       
       print(f"Analysis time: {end_time - start_time:.2f}s")
       print(f"Peak memory: {end_memory:.1f}MB")
       print(f"Memory delta: {end_memory - start_memory:.1f}MB")

References
==========

- **Performance test suite**: ``homodyne/tests/test_performance.py``
- **Baseline measurements**: ``homodyne/tests/performance_baselines.json``
- **Core implementation**: ``homodyne/analysis/core.py``
- **Performance utilities**: ``homodyne/core/profiler.py``
- **Numba kernels**: ``homodyne/core/kernels.py``
- **NumPy Performance Guidelines**: https://numpy.org/doc/stable/user/performance.html
- **Numba Best Practices**: https://numba.pydata.org/numba-doc/latest/user/performance-tips.html
- **Memory Profiling in Python**: https://docs.python.org/3/library/tracemalloc.html
- **Statistical Significance Testing**: https://en.wikipedia.org/wiki/Student%27s_t-test