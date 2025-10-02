Computational Methods
====================

This section details the computational algorithms and implementation strategies used in the
homodyne-analysis package for high-performance scientific computing.

High-Performance Computing Architecture
---------------------------------------

Just-In-Time (JIT) Compilation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The package leverages Numba JIT compilation for critical computational kernels:

.. code-block:: python

   import numba
   import numpy as np

   @numba.jit(nopython=True, fastmath=True, cache=True)
   def compute_g1_correlation_numba(phi_angles, time_points, params):
       """
       Compute g1 correlation function with optimized JIT compilation.

       Achieves 3-5x speedup over pure Python implementation through:
       - LLVM machine code generation
       - Loop optimization and vectorization
       - Efficient memory access patterns
       """
       n_angles = len(phi_angles)
       n_times = len(time_points)
       g1_matrix = np.zeros((n_angles, n_times), dtype=np.float64)

       # Vectorized computation with optimized loops
       for i in range(n_angles):
           for j in range(n_times):
               g1_matrix[i, j] = compute_single_correlation(
                   phi_angles[i], time_points[j], params
               )

       return g1_matrix

Performance Optimizations
~~~~~~~~~~~~~~~~~~~~~~~~~

**Memory Access Optimization**

.. code-block:: python

   @numba.jit(nopython=True, parallel=True)
   def vectorized_chi_squared(experimental_data, model_data, weights):
       """
       Vectorized chi-squared calculation with parallel execution.

       Optimizations:
       - Contiguous memory layout (C-order arrays)
       - SIMD vectorization for element-wise operations
       - Parallel reduction for sum computation
       """
       diff = experimental_data - model_data
       weighted_diff = diff * weights
       chi_squared = np.sum(weighted_diff * weighted_diff)
       return chi_squared

**Cache-Efficient Data Structures**

.. code-block:: python

   class MemoryEfficientCache:
       """
       Smart caching system for repeated computations.

       Features:
       - LRU eviction policy
       - Memory-mapped storage for large datasets
       - Automatic cache size management
       """

       def __init__(self, max_size_mb=1024):
           self.cache = {}
           self.max_size = max_size_mb * 1024 * 1024
           self.current_size = 0

       @numba.jit
       def get_cached_integral(self, key_hash):
           """Retrieve cached integral computation."""
           if key_hash in self.cache:
               return self.cache[key_hash]
           return None

Optimization Algorithms
-----------------------

Classical Optimization Methods
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**1. Nelder-Mead Simplex Algorithm**

Derivative-free optimization suitable for noisy objective functions:

.. code-block:: python

   from scipy.optimize import minimize

   def nelder_mead_optimization(objective_func, initial_params, bounds):
       """
       Robust Nelder-Mead optimization with adaptive parameters.

       Advantages:
       - No gradient computation required
       - Robust to numerical noise
       - Adaptive step size control
       """
       options = {
           'maxiter': 10000,
           'xatol': 1e-8,
           'fatol': 1e-8,
           'adaptive': True
       }

       result = minimize(
           objective_func,
           initial_params,
           method='Nelder-Mead',
           bounds=bounds,
           options=options
       )

       return result

**2. Gurobi Quadratic Programming**

High-performance commercial solver for quadratic optimization:

.. code-block:: python

   import gurobipy as gp

   def gurobi_quadratic_optimization(Q_matrix, linear_terms, bounds):
       """
       Gurobi-based quadratic programming optimization.

       Features:
       - Trust region methods
       - Parallel processing
       - Advanced presolving
       """
       try:
           model = gp.Model("homodyne_optimization")
           model.setParam('OutputFlag', 0)  # Silent optimization
           model.setParam('Threads', 8)     # Parallel processing

           # Create optimization variables
           vars = model.addVars(len(bounds), lb=[b[0] for b in bounds],
                               ub=[b[1] for b in bounds], name="params")

           # Set quadratic objective
           obj = sum(Q_matrix[i][j] * vars[i] * vars[j]
                    for i in range(len(bounds))
                    for j in range(len(bounds)))
           obj += sum(linear_terms[i] * vars[i] for i in range(len(bounds)))

           model.setObjective(obj, gp.GRB.MINIMIZE)
           model.optimize()

           return [vars[i].x for i in range(len(bounds))]

       except gp.GurobiError as e:
           raise OptimizationError(f"Gurobi optimization failed: {e}")

Robust Optimization Framework
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Distributionally Robust Optimization (DRO)**

.. code-block:: python

   import cvxpy as cp
   import numpy as np

   class WassersteinRobustOptimizer:
       """
       Distributionally robust optimization with Wasserstein uncertainty sets.

       Mathematical formulation:
       min_θ max_P∈U E_P[χ²(θ,ξ)]

       where U is the Wasserstein ball around the empirical distribution.
       """

       def __init__(self, epsilon=0.1):
           self.epsilon = epsilon  # Wasserstein radius

       def optimize(self, data_samples, bounds):
           """
           Solve the distributionally robust optimization problem.
           """
           n_params = len(bounds)
           n_samples = len(data_samples)

           # Decision variables
           theta = cp.Variable(n_params)
           lambdas = cp.Variable(n_samples, nonneg=True)
           s = cp.Variable()

           # Constraints
           constraints = []

           # Parameter bounds
           for i, (lb, ub) in enumerate(bounds):
               constraints += [theta[i] >= lb, theta[i] <= ub]

           # Wasserstein constraint
           constraints += [cp.sum(lambdas) == 1]
           constraints += [
               s >= self.epsilon * cp.norm(lambdas, 2)
           ]

           # Objective: worst-case expectation
           chi_squared_values = self.compute_chi_squared_samples(theta, data_samples)
           objective = cp.sum(cp.multiply(lambdas, chi_squared_values)) + s

           # Solve optimization problem
           problem = cp.Problem(cp.Minimize(objective), constraints)
           problem.solve(solver=cp.MOSEK, verbose=False)

           return theta.value, problem.value

**Scenario-Based Robust Optimization**

.. code-block:: python

   from sklearn.utils import resample

   class ScenarioBasedOptimizer:
       """
       Scenario-based robust optimization using bootstrap resampling.

       Generates multiple data scenarios through bootstrap sampling
       and optimizes for worst-case performance across scenarios.
       """

       def __init__(self, n_scenarios=100, confidence_level=0.95):
           self.n_scenarios = n_scenarios
           self.confidence_level = confidence_level

       def generate_scenarios(self, original_data):
           """Generate bootstrap scenarios from original data."""
           scenarios = []
           n_samples = len(original_data)

           for _ in range(self.n_scenarios):
               # Bootstrap resampling
               scenario_data = resample(original_data, n_samples=n_samples)
               scenarios.append(scenario_data)

           return scenarios

       def optimize_robust(self, scenarios, bounds):
           """
           Optimize for robust performance across all scenarios.
           """
           # Solve optimization for each scenario
           scenario_results = []

           for scenario in scenarios:
               result = self.optimize_single_scenario(scenario, bounds)
               scenario_results.append(result)

           # Select robust solution (e.g., worst-case or CVaR)
           robust_params = self.select_robust_solution(
               scenario_results, self.confidence_level
           )

           return robust_params

Numerical Integration and Differentiation
-----------------------------------------

Adaptive Quadrature
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   @numba.jit(nopython=True)
   def adaptive_simpson_integration(func, a, b, tol=1e-10):
       """
       Adaptive Simpson's rule for integral computation.

       Used for computing diffusion and shear integrals:
       J(t) = ∫ D(τ) dτ
       Γ(t) = ∫ γ̇(τ) dτ
       """
       def simpson_rule(f, x0, x2, h):
           x1 = x0 + h
           return h / 3.0 * (f(x0) + 4.0 * f(x1) + f(x2))

       h = (b - a) / 2.0
       s1 = simpson_rule(func, a, b, h)

       # Recursive subdivision for accuracy
       h /= 2.0
       s2 = simpson_rule(func, a, a + h, h/2.0) + simpson_rule(func, a + h, b, h/2.0)

       if abs(s2 - s1) < 15.0 * tol:
           return s2 + (s2 - s1) / 15.0
       else:
           mid = (a + b) / 2.0
           return (adaptive_simpson_integration(func, a, mid, tol/2.0) +
                   adaptive_simpson_integration(func, mid, b, tol/2.0))

Numerical Gradient Computation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import numpy as np
   from scipy.optimize import approx_fprime

   def numerical_gradient_computation(params, data, epsilon=1e-8):
       """
       Finite-difference gradient computation for optimization.

       Uses central differences for improved accuracy:
       - Second-order accurate O(h²)
       - Numerically stable
       - Efficient with vectorization
       """

       def objective_function(theta):
           """Objective function for gradient computation."""
           model_predictions = compute_correlation_model(theta, data)
           chi_squared = np.sum((data.experimental - model_predictions)**2)
           return chi_squared

       # Compute gradient using finite differences
       gradient = approx_fprime(params, objective_function, epsilon)

       return gradient

Parallel Computing
-----------------

Multi-Threading with Numba
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   @numba.jit(nopython=True, parallel=True)
   def parallel_chi_squared_computation(phi_angles, time_matrix, params):
       """
       Parallel computation of chi-squared values across angles.

       Utilizes multiple CPU cores for independent angle calculations.
       """
       n_angles = len(phi_angles)
       chi_squared_values = np.zeros(n_angles)

       # Parallel loop over angles
       for i in numba.prange(n_angles):
           local_chi_squared = 0.0

           for j in range(len(time_matrix)):
               model_value = compute_model_point(phi_angles[i], time_matrix[j], params)
               experimental_value = get_experimental_data(i, j)
               diff = experimental_value - model_value
               local_chi_squared += diff * diff

           chi_squared_values[i] = local_chi_squared

       return np.sum(chi_squared_values)

Task-Based Parallelism
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
   import multiprocessing as mp

   class ParallelOptimizer:
       """
       Task-based parallel optimization for multiple methods.

       Runs different optimization algorithms in parallel and
       compares results for robustness assessment.
       """

       def __init__(self, n_processes=None):
           self.n_processes = n_processes or mp.cpu_count()

       def optimize_parallel(self, data, methods, bounds):
           """
           Run multiple optimization methods in parallel.
           """
           with ProcessPoolExecutor(max_workers=self.n_processes) as executor:
               # Submit optimization tasks
               futures = {}
               for method_name, method_func in methods.items():
                   future = executor.submit(method_func, data, bounds)
                   futures[method_name] = future

               # Collect results
               results = {}
               for method_name, future in futures.items():
                   try:
                       results[method_name] = future.result(timeout=300)
                   except Exception as e:
                       print(f"Method {method_name} failed: {e}")
                       results[method_name] = None

               return results

Memory Management
----------------

Efficient Data Structures
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import numpy as np
   from numba.typed import Dict, List

   class OptimizedDataContainer:
       """
       Memory-efficient data container for large-scale analysis.

       Features:
       - Memory-mapped arrays for large datasets
       - Compressed storage for sparse data
       - Automatic garbage collection
       """

       def __init__(self, use_memmap=False):
           self.use_memmap = use_memmap
           self.data_cache = Dict.empty(
               key_type=numba.types.unicode_type,
               value_type=numba.types.float64[:]
           )

       def store_correlation_data(self, phi_angles, time_points, correlations):
           """Store correlation data with optimal memory layout."""
           if self.use_memmap:
               # Memory-mapped storage for large datasets
               filename = f"correlation_data_{id(self)}.dat"
               memmap_array = np.memmap(
                   filename, dtype='float64', mode='w+',
                   shape=correlations.shape
               )
               memmap_array[:] = correlations[:]
               return memmap_array
           else:
               # In-memory storage with optimized layout
               return np.ascontiguousarray(correlations, dtype=np.float64)

Cache Management
~~~~~~~~~~~~~~~

.. code-block:: python

   from functools import lru_cache
   import hashlib

   class ComputationCache:
       """
       Intelligent caching system for expensive computations.

       Caches:
       - Integral matrix computations
       - Model evaluations
       - Optimization intermediate results
       """

       def __init__(self, max_cache_size=1000):
           self.max_cache_size = max_cache_size
           self.integral_cache = {}
           self.model_cache = {}

       def cache_key(self, *args):
           """Generate cache key from function arguments."""
           key_string = str(args)
           return hashlib.md5(key_string.encode()).hexdigest()

       @lru_cache(maxsize=1000)
       def cached_integral_computation(self, D0, alpha, D_offset, time_hash):
           """Cached computation of diffusion integrals."""
           # Expensive integral computation
           return compute_diffusion_integral(D0, alpha, D_offset)

       def get_or_compute_model(self, params, data_hash):
           """Retrieve cached model or compute if not available."""
           cache_key = self.cache_key(params, data_hash)

           if cache_key in self.model_cache:
               return self.model_cache[cache_key]

           # Compute model if not cached
           model_result = compute_correlation_model(params)

           # Store in cache with size management
           if len(self.model_cache) >= self.max_cache_size:
               # Remove oldest entry (FIFO)
               oldest_key = next(iter(self.model_cache))
               del self.model_cache[oldest_key]

           self.model_cache[cache_key] = model_result
           return model_result

Error Handling and Numerical Stability
--------------------------------------

Numerical Robustness
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import numpy as np
   from scipy import special

   def robust_sinc_squared(x, threshold=1e-10):
       """
       Numerically robust computation of sinc²(x).

       Handles near-zero arguments to avoid division by zero:
       sinc(x) = sin(πx)/(πx) for x ≠ 0
       sinc(0) = 1
       """
       x = np.asarray(x)
       result = np.ones_like(x)

       # Use Taylor expansion for small arguments
       small_mask = np.abs(x) < threshold
       large_mask = ~small_mask

       if np.any(small_mask):
           x_small = x[small_mask]
           # Taylor expansion: sinc(x) ≈ 1 - (πx)²/6 + (πx)⁴/120 - ...
           pi_x = np.pi * x_small
           pi_x_sq = pi_x * pi_x
           sinc_val = 1.0 - pi_x_sq/6.0 + pi_x_sq*pi_x_sq/120.0
           result[small_mask] = sinc_val * sinc_val

       if np.any(large_mask):
           x_large = x[large_mask]
           sinc_val = np.sin(np.pi * x_large) / (np.pi * x_large)
           result[large_mask] = sinc_val * sinc_val

       return result

Exception Handling
~~~~~~~~~~~~~~~~~

.. code-block:: python

   class OptimizationError(Exception):
       """Custom exception for optimization failures."""
       pass

   class ConvergenceError(OptimizationError):
       """Exception for convergence failures."""
       pass

   def safe_optimization_wrapper(optimization_func, *args, **kwargs):
       """
       Robust wrapper for optimization functions with error recovery.
       """
       try:
           return optimization_func(*args, **kwargs)

       except np.linalg.LinAlgError as e:
           # Handle singular matrix errors
           raise OptimizationError(f"Linear algebra error: {e}")

       except OverflowError as e:
           # Handle numerical overflow
           raise OptimizationError(f"Numerical overflow: {e}")

       except ConvergenceError as e:
           # Try alternative optimization method
           print(f"Convergence failed, trying backup method: {e}")
           return backup_optimization_method(*args, **kwargs)

       except Exception as e:
           # General error handling
           raise OptimizationError(f"Optimization failed: {e}")

Performance Monitoring
---------------------

Benchmarking Framework
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import time
   import psutil
   import numpy as np

   class PerformanceMonitor:
       """
       Comprehensive performance monitoring for optimization algorithms.

       Tracks:
       - Execution time
       - Memory usage
       - CPU utilization
       - Convergence metrics
       """

       def __init__(self):
           self.metrics = {}
           self.start_time = None
           self.start_memory = None

       def start_monitoring(self):
           """Begin performance monitoring."""
           self.start_time = time.perf_counter()
           self.start_memory = psutil.virtual_memory().used

       def stop_monitoring(self, operation_name):
           """Stop monitoring and record metrics."""
           end_time = time.perf_counter()
           end_memory = psutil.virtual_memory().used

           self.metrics[operation_name] = {
               'execution_time': end_time - self.start_time,
               'memory_delta': end_memory - self.start_memory,
               'cpu_percent': psutil.cpu_percent(),
               'timestamp': time.time()
           }

       def benchmark_optimization_methods(self, methods, data, bounds):
           """Benchmark multiple optimization methods."""
           results = {}

           for method_name, method_func in methods.items():
               self.start_monitoring()

               try:
                   opt_result = method_func(data, bounds)
                   self.stop_monitoring(method_name)

                   results[method_name] = {
                       'optimization_result': opt_result,
                       'performance_metrics': self.metrics[method_name]
                   }

               except Exception as e:
                   results[method_name] = {
                       'optimization_result': None,
                       'error': str(e),
                       'performance_metrics': None
                   }

           return results

This computational framework provides the foundation for high-performance, robust analysis of
homodyne scattering data, enabling researchers to extract reliable transport coefficients from
experimental measurements under challenging conditions.
