#!/usr/bin/env python3
"""
Performance Benchmark Script for Homodyne Package Optimizations
===============================================================

This script benchmarks the performance improvements from optimizations
including Numba acceleration, memory optimizations, and configuration caching.

Usage:
    python benchmark_performance.py [--iterations N] [--config CONFIG]
"""

import time
import numpy as np
import argparse
from pathlib import Path
import json

# Import homodyne components
from homodyne import (
    ConfigManager,
    create_time_integral_matrix_numba,
    calculate_diffusion_coefficient_numba,
    calculate_shear_rate_numba,
    compute_g1_correlation_numba,
    compute_sinc_squared_numba,
    # New optimized kernels
    create_symmetric_matrix_optimized,
    matrix_vector_multiply_optimized,
    apply_scaling_vectorized,
    compute_chi_squared_fast,
    exp_negative_vectorized
)
from homodyne.core.config import performance_monitor


def benchmark_kernel_performance(iterations=10, array_size=1000):
    """Benchmark computational kernels."""
    print(f"=== Benchmarking Computational Kernels (size={array_size}, iterations={iterations}) ===")
    
    # Generate test data
    time_array = np.linspace(0.1, 10.0, array_size)
    time_dependent_data = np.random.random(array_size) + 1.0
    matrix_data = np.random.random((array_size, array_size))
    
    results = {}
    
    # Benchmark time integral matrix creation
    print("Testing time integral matrix creation...")
    start = time.perf_counter()
    for _ in range(iterations):
        result = create_time_integral_matrix_numba(time_dependent_data)
    elapsed = time.perf_counter() - start
    results['time_integral_matrix'] = elapsed / iterations
    print(f"  Average time: {results['time_integral_matrix']:.4f}s")
    
    # Benchmark diffusion coefficient calculation
    print("Testing diffusion coefficient calculation...")
    start = time.perf_counter()
    for _ in range(iterations):
        result = calculate_diffusion_coefficient_numba(time_array, 1000.0, -0.5, 100.0)
    elapsed = time.perf_counter() - start
    results['diffusion_coefficient'] = elapsed / iterations
    print(f"  Average time: {results['diffusion_coefficient']:.4f}s")
    
    # Benchmark shear rate calculation
    print("Testing shear rate calculation...")
    start = time.perf_counter()
    for _ in range(iterations):
        result = calculate_shear_rate_numba(time_array, 0.01, -0.8, 0.001)
    elapsed = time.perf_counter() - start
    results['shear_rate'] = elapsed / iterations
    print(f"  Average time: {results['shear_rate']:.4f}s")
    
    # Benchmark g1 correlation
    print("Testing g1 correlation computation...")
    start = time.perf_counter()
    for _ in range(iterations):
        result = compute_g1_correlation_numba(matrix_data, 0.001)
    elapsed = time.perf_counter() - start
    results['g1_correlation'] = elapsed / iterations
    print(f"  Average time: {results['g1_correlation']:.4f}s")
    
    # Benchmark sinc squared computation
    print("Testing sinc squared computation...")
    start = time.perf_counter()
    for _ in range(iterations):
        result = compute_sinc_squared_numba(matrix_data, 0.1)
    elapsed = time.perf_counter() - start
    results['sinc_squared'] = elapsed / iterations
    print(f"  Average time: {results['sinc_squared']:.4f}s")
    
    return results


def benchmark_optimized_kernels(iterations=10, array_size=1000):
    """Benchmark new optimized kernels."""
    print(f"\\n=== Benchmarking Optimized Kernels (size={array_size}, iterations={iterations}) ===")
    
    # Generate test data
    diagonal_values = np.random.random(array_size)
    matrix_data = np.random.random((array_size, array_size))
    vector_data = np.random.random(array_size)
    
    results = {}
    
    # Benchmark optimized matrix creation
    print("Testing optimized symmetric matrix creation...")
    start = time.perf_counter()
    for _ in range(iterations):
        result = create_symmetric_matrix_optimized(diagonal_values, array_size)
    elapsed = time.perf_counter() - start
    results['symmetric_matrix'] = elapsed / iterations
    print(f"  Average time: {results['symmetric_matrix']:.4f}s")
    
    # Benchmark optimized matrix-vector multiply
    print("Testing optimized matrix-vector multiplication...")
    start = time.perf_counter()
    for _ in range(iterations):
        result = matrix_vector_multiply_optimized(matrix_data, vector_data)
    elapsed = time.perf_counter() - start
    results['matrix_vector_multiply'] = elapsed / iterations
    print(f"  Average time: {results['matrix_vector_multiply']:.4f}s")
    
    # Benchmark vectorized scaling
    print("Testing vectorized scaling...")
    start = time.perf_counter()
    for _ in range(iterations):
        result = apply_scaling_vectorized(vector_data, 1.5, 0.1)
    elapsed = time.perf_counter() - start
    results['vectorized_scaling'] = elapsed / iterations
    print(f"  Average time: {results['vectorized_scaling']:.4f}s")
    
    # Benchmark fast chi-squared
    print("Testing fast chi-squared computation...")
    observed = np.random.poisson(100, array_size).astype(float)
    expected = np.random.gamma(2, 50, array_size)
    start = time.perf_counter()
    for _ in range(iterations):
        result = compute_chi_squared_fast(observed, expected)
    elapsed = time.perf_counter() - start
    results['chi_squared_fast'] = elapsed / iterations
    print(f"  Average time: {results['chi_squared_fast']:.4f}s")
    
    # Benchmark vectorized exponential
    print("Testing vectorized exponential...")
    start = time.perf_counter()
    for _ in range(iterations):
        result = exp_negative_vectorized(vector_data, 0.001)
    elapsed = time.perf_counter() - start
    results['exp_negative'] = elapsed / iterations
    print(f"  Average time: {results['exp_negative']:.4f}s")
    
    return results


def benchmark_config_loading(config_file, iterations=100):
    """Benchmark configuration loading performance."""
    print(f"\\n=== Benchmarking Configuration Loading (iterations={iterations}) ===")
    
    # Benchmark regular config loading
    print("Testing configuration loading...")
    start = time.perf_counter()
    for _ in range(iterations):
        config = ConfigManager(config_file)
    elapsed = time.perf_counter() - start
    avg_time = elapsed / iterations
    
    print(f"  Average config loading time: {avg_time:.4f}s")
    
    # Test cached access performance
    config = ConfigManager(config_file)
    print("Testing cached configuration access...")
    
    start = time.perf_counter()
    for _ in range(iterations * 10):  # More iterations for fast operations
        _ = config.is_static_mode_enabled()
        _ = config.get_effective_parameter_count()
    elapsed = time.perf_counter() - start
    avg_access_time = elapsed / (iterations * 10)
    
    print(f"  Average cached access time: {avg_access_time:.6f}s")
    
    # Display performance monitoring results
    performance_monitor.log_performance_summary()
    
    return {'config_loading': avg_time, 'cached_access': avg_access_time}


def compare_numpy_vs_optimized(array_size=1000):
    """Compare numpy operations vs optimized kernels."""
    print(f"\\n=== Comparing NumPy vs Optimized Operations (size={array_size}) ===")
    
    # Test data
    matrix_a = np.random.random((array_size, array_size))
    vector_b = np.random.random(array_size)
    data_array = np.random.random(array_size)
    
    iterations = 10
    
    # Compare matrix-vector multiplication
    print("Matrix-vector multiplication comparison:")
    
    # NumPy version
    start = time.perf_counter()
    for _ in range(iterations):
        result_numpy = matrix_a @ vector_b
    numpy_time = time.perf_counter() - start
    
    # Optimized version
    start = time.perf_counter()
    for _ in range(iterations):
        result_optimized = matrix_vector_multiply_optimized(matrix_a, vector_b)
    optimized_time = time.perf_counter() - start
    
    print(f"  NumPy:     {numpy_time/iterations:.4f}s")
    print(f"  Optimized: {optimized_time/iterations:.4f}s")
    print(f"  Speedup:   {numpy_time/optimized_time:.2f}x")
    
    # Compare scaling operations
    print("\\nData scaling comparison:")
    
    # NumPy version
    start = time.perf_counter()
    for _ in range(iterations):
        result_numpy = 1.5 * data_array + 0.1
    numpy_time = time.perf_counter() - start
    
    # Optimized version
    start = time.perf_counter()
    for _ in range(iterations):
        result_optimized = apply_scaling_vectorized(data_array, 1.5, 0.1)
    optimized_time = time.perf_counter() - start
    
    print(f"  NumPy:     {numpy_time/iterations:.4f}s")
    print(f"  Optimized: {optimized_time/iterations:.4f}s")
    print(f"  Speedup:   {numpy_time/optimized_time:.2f}x")


def main():
    """Main benchmark function."""
    parser = argparse.ArgumentParser(description="Benchmark homodyne package performance")
    parser.add_argument("--iterations", type=int, default=10, 
                       help="Number of iterations for benchmarking")
    parser.add_argument("--size", type=int, default=500,
                       help="Array size for benchmarking")
    parser.add_argument("--config", type=str, default="my_config_optimized.json",
                       help="Configuration file to use")
    
    args = parser.parse_args()
    
    print("🚀 Homodyne Package Performance Benchmark")
    print("=" * 50)
    
    # Check if optimized config exists, fallback to regular config
    config_file = args.config
    if not Path(config_file).exists():
        config_file = "my_config.json"
        print(f"⚠️  Optimized config not found, using {config_file}")
    
    print(f"Configuration: {config_file}")
    print(f"Array size: {args.size}")
    print(f"Iterations: {args.iterations}")
    
    # Run benchmarks
    all_results = {}
    
    try:
        # Benchmark core kernels
        kernel_results = benchmark_kernel_performance(args.iterations, args.size)
        all_results['kernels'] = kernel_results
        
        # Benchmark optimized kernels
        optimized_results = benchmark_optimized_kernels(args.iterations, args.size)
        all_results['optimized_kernels'] = optimized_results
        
        # Benchmark configuration
        config_results = benchmark_config_loading(config_file, min(args.iterations, 50))
        all_results['configuration'] = config_results
        
        # Compare operations
        compare_numpy_vs_optimized(args.size)
        
        # Summary
        print("\\n" + "=" * 50)
        print("📊 BENCHMARK SUMMARY")
        print("=" * 50)
        
        total_kernel_time = sum(kernel_results.values())
        total_optimized_time = sum(optimized_results.values())
        
        print(f"Total core kernel time:      {total_kernel_time:.4f}s")
        print(f"Total optimized kernel time: {total_optimized_time:.4f}s")
        print(f"Configuration loading:       {config_results['config_loading']:.4f}s")
        print(f"Cached config access:        {config_results['cached_access']:.6f}s")
        
        # Save results
        results_file = "benchmark_results.json"
        with open(results_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"\\n📁 Results saved to: {results_file}")
        
        print("\\n✅ Benchmark completed successfully!")
        
    except Exception as e:
        print(f"\\n❌ Benchmark failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())