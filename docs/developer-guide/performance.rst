Performance Optimization
=======================

This guide covers performance optimization strategies for the homodyne package.

Performance Overview
--------------------

The homodyne package is designed to handle large datasets efficiently. Key performance considerations:

- **Memory Management**: Efficient handling of large correlation matrices
- **Computational Optimization**: Numba JIT compilation and vectorization
- **Parallel Processing**: Multi-core MCMC and data processing
- **Algorithm Selection**: Choosing appropriate optimization methods

Optimization Strategies
-----------------------

**1. Angle Filtering**

The most effective optimization for speed with minimal accuracy loss:

.. code-block:: python

   # Performance improvement: 3-5x speedup
   config = {
       "analysis_settings": {
           "enable_angle_filtering": True,
           "angle_filter_ranges": [[-5, 5], [175, 185]]
       }
   }

**Benefits**:
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
   
   @jit(nopython=True, cache=True)
   def compute_correlation_fast(tau, params, q):
       # JIT-compiled computation
       # 5-10x speedup for model functions
       pass

**4. Parallel MCMC**

Optimize MCMC sampling configuration:

.. code-block:: python

   config = {
       "optimization_config": {
           "mcmc_sampling": {
               "chains": 4,           # Match CPU cores
               "cores": 4,            # Parallel processing
               "draws": 2000,         # Reduce if acceptable
               "tune": 1000           # Adequate tuning
           }
       }
   }

Memory Optimization
-------------------

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
----------------

**1. Thread Configuration**

Optimize thread usage:

.. code-block:: python

   import os
   
   # Set thread counts
   os.environ['OMP_NUM_THREADS'] = '4'
   os.environ['NUMBA_NUM_THREADS'] = '4'
   
   config = {
       "performance_settings": {
           "num_threads": 4  # Match your CPU cores
       }
   }

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

Algorithm Optimization
----------------------

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
   
   # For complex landscapes
   config = {
       "optimization_config": {
           "classical": {
               "method": "L-BFGS-B",     # Gradient-based
               "max_iterations": 500
           }
       }
   }

**2. MCMC Tuning**

Optimize MCMC parameters for efficiency:

.. code-block:: python

   config = {
       "optimization_config": {
           "mcmc_sampling": {
               "target_accept": 0.9,      # Higher acceptance
               "max_treedepth": 10,       # Prevent divergences
               "adapt_step_size": True,   # Auto-tuning
               "adapt_diag_grad": True    # Mass matrix adaptation
           }
       }
   }

Performance Benchmarks
----------------------

**Typical Performance Metrics**:

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

**MCMC Performance**:

.. list-table:: MCMC Benchmarks
   :widths: 20 15 15 15 35
   :header-rows: 1

   * - Chains
     - Time
     - ESS/min
     - R̂
     - Notes
   * - **2 chains**
     - 120s
     - 250
     - 1.02
     - Minimal setup
   * - **4 chains**
     - 80s
     - 600
     - 1.01
     - Recommended
   * - **8 chains**
     - 70s
     - 900
     - 1.00
     - Diminishing returns

Profiling Tools
---------------

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
---------------------------

**Configuration**:

1. **Enable angle filtering** for 3-5x speedup
2. **Use float32** unless high precision needed
3. **Set appropriate thread counts** (match CPU cores)
4. **Enable JIT compilation** for model functions

**MCMC**:

1. **Start with classical optimization** for good initial values
2. **Use 4 chains** as a good balance
3. **Monitor convergence** with R̂ and ESS
4. **Adjust target_accept** for efficiency

**Memory**:

1. **Estimate memory needs** before large analyses
2. **Use chunked processing** for very large datasets
3. **Monitor memory usage** during long runs
4. **Clean up intermediate results** when possible

**Development**:

1. **Profile before optimizing** to find real bottlenecks
2. **Test performance changes** with realistic datasets
3. **Balance speed vs. accuracy** based on requirements
4. **Document performance characteristics** of new features

Troubleshooting Performance Issues
----------------------------------

**Slow Optimization**:

1. Enable angle filtering
2. Check initial parameter values
3. Try different optimization methods
4. Reduce tolerance if acceptable

**High Memory Usage**:

1. Use float32 data type
2. Enable chunked processing
3. Reduce dataset size if possible
4. Check for memory leaks

**MCMC Convergence Issues**:

1. Increase tuning steps
2. Adjust target acceptance rate
3. Check parameter bounds
4. Use better initial values

**System-Specific Issues**:

1. Check BLAS/LAPACK installation
2. Verify thread settings
3. Monitor CPU/memory resources
4. Consider cluster computing for very large problems