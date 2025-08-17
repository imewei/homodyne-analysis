Performance Optimization
========================

The homodyne package is designed for high-performance analysis of XPCS data with multiple optimization strategies available for different computational scenarios.

Performance Overview
--------------------

The package achieves high performance through several key strategies:

- **Numba JIT Compilation**: 3-5x speedup for computational kernels
- **Smart Angle Filtering**: Reduces computation by focusing on relevant angles
- **Memory-Efficient Algorithms**: Optimized data structures and memory usage
- **Vectorized Operations**: NumPy-based vectorized computations
- **Intelligent Caching**: Reuse of computed results

Performance Benchmarks
-----------------------

Relative Performance by Mode
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table:: Performance Characteristics
   :widths: 20 15 15 15 35
   :header-rows: 1

   * - Analysis Mode
     - Relative Speed
     - Memory Usage
     - Scalability
     - Optimization Strategies
   * - Static Isotropic
     - 1x (baseline)
     - Low
     - Excellent
     - Single angle, minimal overhead
   * - Static Anisotropic
     - 3-5x slower
     - Medium
     - Good
     - Angle filtering provides speedup
   * - Laminar Flow
     - 10-20x slower
     - High
     - Fair
     - Complex parameter space

Numba JIT Performance Impact
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table:: Numba JIT Acceleration
   :widths: 30 20 20 30
   :header-rows: 1

   * - Computational Kernel
     - Without Numba
     - With Numba
     - Speedup Factor
   * - Correlation function calculation
     - 100% (baseline)
     - 25%
     - 4x
   * - Matrix operations
     - 100%
     - 30%
     - 3.3x
   * - Chi-squared computation
     - 100%
     - 20%
     - 5x
   * - Overall analysis
     - 100%
     - 25-35%
     - 3-4x

Environment Configuration
-------------------------

Operating System Optimization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Linux/macOS**:

.. code-block:: bash

   # Optimize BLAS threading
   export OMP_NUM_THREADS=8
   export OPENBLAS_NUM_THREADS=8
   export MKL_NUM_THREADS=8
   
   # Numba compatibility
   export NUMBA_DISABLE_INTEL_SVML=1
   
   # Memory management
   export MALLOC_TRIM_THRESHOLD_=100000

**Windows**:

.. code-block:: batch

   set OMP_NUM_THREADS=8
   set OPENBLAS_NUM_THREADS=8
   set MKL_NUM_THREADS=8
   set NUMBA_DISABLE_INTEL_SVML=1

Hardware Recommendations
~~~~~~~~~~~~~~~~~~~~~~~~

**CPU**: 
- Multi-core processor (8+ cores recommended)
- High clock speed for single-threaded performance
- Support for AVX2/AVX512 instructions

**Memory**:
- 16+ GB RAM for typical datasets
- 32+ GB for large datasets or flow analysis
- Fast memory (DDR4-3200 or better)

**Storage**:
- SSD for data files and cache directory
- High-speed storage for large datasets

Configuration Tuning
---------------------

Performance Settings
~~~~~~~~~~~~~~~~~~~~

Optimize configuration for your computational resources:

.. code-block:: json

   {
     "performance": {
       "num_threads": 8,
       "memory_limit_gb": 16,
       "use_numba_jit": true,
       "data_type": "float64",
       "enable_caching": true,
       "cache_directory": "./cache"
     }
   }

**Key Parameters**:
- **num_threads**: Number of parallel threads (match CPU cores)
- **memory_limit_gb**: Maximum memory usage
- **use_numba_jit**: Enable JIT compilation (recommended)
- **data_type**: Precision vs. memory trade-off
- **enable_caching**: Cache computed results

Angle Filtering Optimization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Enable angle filtering for significant performance improvements:

.. code-block:: json

   {
     "analysis_settings": {
       "enable_angle_filtering": true,
       "angle_filter_ranges": [[-10, 10], [170, 190]]
     }
   }

**Benefits**:
- 3-5x speedup for large datasets
- Focuses computation on relevant angular ranges
- Minimal accuracy loss for most systems

**Customization**:

.. code-block:: json

   {
     "analysis_settings": {
       "enable_angle_filtering": true,
       "angle_filter_ranges": [
         [-15, 15],     # Forward scattering
         [165, 195],    # Backward scattering
         [85, 95]       # Side scattering (optional)
       ]
     }
   }

Memory Optimization
-------------------

Memory-Efficient Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For large datasets or limited memory systems:

.. code-block:: json

   {
     "performance": {
       "data_type": "float32",
       "memory_limit_gb": 8,
       "chunked_processing": true,
       "chunk_size": 1000
     }
   }

**Trade-offs**:
- **float32 vs float64**: 50% memory reduction, minimal precision loss
- **Chunked processing**: Processes data in smaller chunks
- **Memory limits**: Prevents system memory exhaustion

Data Loading Optimization
~~~~~~~~~~~~~~~~~~~~~~~~~

Optimize data loading and caching:

.. code-block:: json

   {
     "file_paths": {
       "cache_directory": "/fast_ssd/cache",
       "enable_memory_mapping": true,
       "preload_data": true
     }
   }

**Strategies**:
- **Fast storage**: Use SSD for cache directory
- **Memory mapping**: Efficient large file access
- **Preloading**: Load frequently accessed data into memory

Computational Optimization
--------------------------

Numba JIT Compilation
~~~~~~~~~~~~~~~~~~~~~

Numba provides significant acceleration for computational kernels:

**Automatic Optimization**:
- JIT compilation occurs on first function call
- Subsequent calls use optimized machine code
- Minimal developer intervention required

**Performance Kernels**:

.. code-block:: python

   from numba import jit
   
   @jit(nopython=True, cache=True)
   def compute_chi_squared_fast(experimental, theoretical, uncertainties):
       """Optimized chi-squared calculation."""
       chi_sq = 0.0
       for i in range(len(experimental)):
           diff = experimental[i] - theoretical[i]
           chi_sq += (diff / uncertainties[i]) ** 2
       return chi_sq

**Configuration**:

.. code-block:: json

   {
     "performance": {
       "use_numba_jit": true,
       "numba_cache": true,
       "numba_parallel": true
     }
   }

Vectorized Operations
~~~~~~~~~~~~~~~~~~~~

Leverage NumPy's vectorized operations for maximum performance:

.. code-block:: python

   # Optimized vectorized implementation
   def apply_scaling_vectorized(theory, experimental):
       """Vectorized scaling optimization."""
       A = np.column_stack([theory, np.ones(len(theory))])
       scaling = np.linalg.lstsq(A, experimental, rcond=None)[0]
       return scaling[0], scaling[1]  # contrast, offset

**Performance Tips**:
- Use NumPy arrays instead of Python lists
- Leverage broadcasting for element-wise operations
- Minimize Python loops in favor of vectorized operations

Optimization Algorithm Tuning
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Fine-tune optimization algorithms for your specific use case:

**Classical Optimization**:

.. code-block:: json

   {
     "optimization": {
       "method": "Nelder-Mead",
       "max_iterations": 10000,
       "tolerance": 1e-8,
       "initial_simplex_size": 0.1,
       "adaptive": true
     }
   }

**MCMC Sampling**:

.. code-block:: json

   {
     "mcmc": {
       "n_samples": 2000,
       "tune": 1000,
       "chains": 4,
       "target_accept": 0.8,
       "cores": 4
     }
   }

Mode-Specific Optimization
--------------------------

Static Isotropic Mode
~~~~~~~~~~~~~~~~~~~~~

Optimizations for maximum speed:

.. code-block:: json

   {
     "analysis_settings": {
       "static_mode": true,
       "static_submode": "isotropic"
     },
     "performance": {
       "skip_angle_loading": true,
       "minimal_memory_usage": true,
       "fast_approximations": true
     }
   }

**Key Features**:
- Single angle computation
- Minimal memory footprint
- Fast convergence

Static Anisotropic Mode
~~~~~~~~~~~~~~~~~~~~~~

Balance between speed and accuracy:

.. code-block:: json

   {
     "analysis_settings": {
       "static_mode": true,
       "static_submode": "anisotropic",
       "enable_angle_filtering": true
     },
     "performance": {
       "angle_filter_ranges": [[-10, 10], [170, 190]],
       "parallel_angle_processing": true
     }
   }

Laminar Flow Mode
~~~~~~~~~~~~~~~~

Optimization for complex parameter spaces:

.. code-block:: json

   {
     "analysis_settings": {
       "static_mode": false,
       "enable_angle_filtering": true
     },
     "performance": {
       "parameter_caching": true,
       "gradient_approximation": true,
       "adaptive_step_size": true
     }
   }

Benchmarking Tools
------------------

Built-in Performance Benchmarking
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use the comprehensive benchmarking suite:

.. code-block:: bash

   # Complete performance analysis
   python benchmark_performance.py --iterations 50 --size 1000

   # Quick performance check
   python benchmark_performance.py --fast

   # Custom benchmarking
   python benchmark_performance.py --iterations 20 --size 500 --detailed

**Benchmark Components**:
- Computational kernels with Numba JIT
- Matrix operations and vectorized functions
- Configuration loading and caching
- Memory allocation patterns

Custom Performance Monitoring
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Monitor performance in your analysis:

.. code-block:: python

   import time
   from homodyne import HomodyneAnalysisCore, ConfigManager
   
   # Performance monitoring
   config = ConfigManager("my_config.json")
   analysis = HomodyneAnalysisCore(config)
   
   start_time = time.time()
   results = analysis.optimize_classical()
   end_time = time.time()
   
   print(f"Analysis completed in {end_time - start_time:.2f} seconds")
   print(f"Chi-squared evaluations: {results.nfev}")
   print(f"Performance: {results.nfev / (end_time - start_time):.1f} evaluations/second")

Profiling and Diagnostics
~~~~~~~~~~~~~~~~~~~~~~~~~

Profile your analysis for bottleneck identification:

.. code-block:: python

   import cProfile
   import pstats
   
   # Profile the analysis
   profiler = cProfile.Profile()
   profiler.enable()
   
   results = analysis.optimize_classical()
   
   profiler.disable()
   stats = pstats.Stats(profiler)
   stats.sort_stats('tottime').print_stats(10)

Scaling Strategies
------------------

Dataset Size Scaling
~~~~~~~~~~~~~~~~~~~~

Strategies for different dataset sizes:

**Small Datasets** (< 1GB):
- Use default settings
- Enable all features for maximum accuracy
- Memory constraints are not limiting

**Medium Datasets** (1-10GB):
- Enable angle filtering
- Consider float32 precision
- Use SSD for cache storage

**Large Datasets** (> 10GB):
- Mandatory angle filtering
- Chunked processing
- Distributed computing consideration

.. code-block:: json

   {
     "performance": {
       "data_type": "float32",
       "chunked_processing": true,
       "chunk_size": 500,
       "memory_limit_gb": 32
     },
     "analysis_settings": {
       "enable_angle_filtering": true,
       "angle_filter_ranges": [[-5, 5], [175, 185]]
     }
   }

Computational Resource Scaling
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Single Machine Optimization**:

.. code-block:: json

   {
     "performance": {
       "num_threads": 16,
       "use_all_cores": true,
       "memory_limit_gb": 64
     }
   }

**Multi-Machine Considerations**:
- Parallel MCMC chains across machines
- Distributed data processing
- Network storage optimization

Performance Troubleshooting
---------------------------

Common Performance Issues
~~~~~~~~~~~~~~~~~~~~~~~~

**Slow JIT Compilation on First Run**:
- Expected behavior for Numba
- Subsequent runs will be fast
- Consider pre-compilation for production

**Memory Exhaustion**:
- Reduce `memory_limit_gb`
- Use `float32` data type
- Enable chunked processing

**Poor MCMC Performance**:
- Use better initial estimates from classical optimization
- Increase `target_accept` rate
- Use more chains with fewer samples each

Optimization Checklist
~~~~~~~~~~~~~~~~~~~~~~

**Essential Optimizations**:
- ✅ Enable Numba JIT compilation
- ✅ Set appropriate thread count
- ✅ Use angle filtering for large datasets
- ✅ Configure adequate memory limits

**Advanced Optimizations**:
- ✅ Use SSD for cache directory
- ✅ Optimize BLAS threading
- ✅ Consider float32 for memory-constrained systems
- ✅ Enable result caching

**Monitoring and Validation**:
- ✅ Benchmark performance regularly
- ✅ Monitor memory usage
- ✅ Profile computational bottlenecks
- ✅ Validate results quality

This comprehensive performance guide ensures optimal computational efficiency while maintaining scientific accuracy across all analysis scenarios.
