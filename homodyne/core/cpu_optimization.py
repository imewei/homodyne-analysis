"""
CPU Optimization Utilities for Homodyne Analysis
================================================

High-performance CPU-only optimization utilities for replacing GPU acceleration
with advanced CPU vectorization, threading, and cache optimization strategies.

This module implements CPU-specific optimization techniques including:
- SIMD vectorization with NumPy/SciPy
- OpenMP threading through Numba
- Memory hierarchy optimization for cache efficiency
- Multiprocessing for CPU-bound scientific computations
- CPU-specific performance monitoring and tuning

Authors: Wei Chen, Hongrui He
Institution: Argonne National Laboratory
"""

import multiprocessing as mp
import platform
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any

import numpy as np
import psutil

try:
    from numba import jit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    warnings.warn("Numba not available - CPU optimizations will be limited")

try:
    SCIPY_FFT_AVAILABLE = True
except ImportError:
    SCIPY_FFT_AVAILABLE = False
    warnings.warn("SciPy FFT not available - some optimizations disabled")


class CPUOptimizer:
    """
    CPU-specific optimization utilities for scientific computing.

    Provides advanced CPU optimization techniques including SIMD vectorization,
    cache-aware algorithms, and OpenMP threading for homodyne analysis.
    """

    def __init__(self, config: dict[str, Any] | None = None):
        """
        Initialize CPU optimizer.

        Parameters
        ----------
        config : dict[str, Any], optional
            CPU optimization configuration
        """
        self.config = config or {}
        self.cpu_count = mp.cpu_count()
        self.cache_info = self._detect_cpu_cache()
        self.simd_support = self._detect_simd_support()

        # Set optimal thread counts
        self.max_threads = self.config.get("max_threads", self.cpu_count)
        self.chunk_size = self.config.get("chunk_size", "auto")

    def _detect_cpu_cache(self) -> dict[str, int]:
        """Detect CPU cache sizes for optimization."""
        cache_info = {
            "l1_cache_kb": 32,    # Default conservative estimate
            "l2_cache_kb": 256,   # Default conservative estimate
            "l3_cache_kb": 8192,  # Default conservative estimate
        }

        try:
            # Try to get actual cache sizes
            if hasattr(psutil, 'cpu_stats'):
                # Estimate based on CPU model if available
                cpu_info = platform.processor()
                if "intel" in cpu_info.lower():
                    cache_info["l3_cache_kb"] = 12288  # Intel typical
                elif "amd" in cpu_info.lower():
                    cache_info["l3_cache_kb"] = 16384  # AMD typical
        except Exception:
            pass  # Use defaults

        return cache_info

    def _detect_simd_support(self) -> dict[str, bool]:
        """Detect SIMD instruction set support."""
        simd_support = {
            "sse": False,
            "avx": False,
            "avx2": False,
            "avx512": False,
        }

        try:
            # Check NumPy build info for SIMD support
            import numpy as np
            config = np.show_config(mode='dicts')
            if 'Blas' in config:
                blas_info = str(config['Blas'])
                simd_support["sse"] = "sse" in blas_info.lower()
                simd_support["avx"] = "avx" in blas_info.lower()
                simd_support["avx2"] = "avx2" in blas_info.lower()
                simd_support["avx512"] = "avx512" in blas_info.lower()
        except Exception:
            pass  # Use defaults

        return simd_support

    def optimize_matrix_operations_cpu(
        self,
        matrix: np.ndarray,
        operation: str = "correlation",
        cache_size_kb: int | None = None
    ) -> np.ndarray:
        """
        Optimize matrix operations for CPU cache hierarchy.

        Parameters
        ----------
        matrix : np.ndarray
            Input matrix for operations
        operation : str
            Type of operation ('correlation', 'fft', 'matmul')
        cache_size_kb : int, optional
            Cache size in KB (auto-detected if None)

        Returns
        -------
        np.ndarray
            Optimized result
        """
        if cache_size_kb is None:
            cache_size_kb = self.cache_info["l3_cache_kb"]

        # Calculate optimal block size for cache efficiency
        element_size = matrix.dtype.itemsize
        optimal_block_size = int(np.sqrt(cache_size_kb * 1024 // (3 * element_size)))

        rows, cols = matrix.shape

        if operation == "correlation":
            return self._cache_optimized_correlation(matrix, optimal_block_size)
        elif operation == "fft":
            return self._cache_optimized_fft(matrix, optimal_block_size)
        elif operation == "matmul":
            return self._cache_optimized_matmul(matrix, optimal_block_size)
        else:
            raise ValueError(f"Unsupported operation: {operation}")

    def _cache_optimized_correlation(self, matrix: np.ndarray, block_size: int) -> np.ndarray:
        """Cache-friendly correlation computation."""
        rows, cols = matrix.shape
        result = np.zeros((rows, rows))

        # Block-wise processing for cache efficiency
        for i in range(0, rows, block_size):
            for j in range(0, rows, block_size):
                i_end = min(i + block_size, rows)
                j_end = min(j + block_size, rows)

                # Process cache-sized blocks
                block_i = matrix[i:i_end, :]
                block_j = matrix[j:j_end, :]

                # Vectorized correlation using NumPy
                result[i:i_end, j:j_end] = np.corrcoef(block_i, block_j)[:i_end-i, :j_end-j]

        return result

    def _cache_optimized_fft(self, matrix: np.ndarray, block_size: int) -> np.ndarray:
        """Cache-friendly FFT computation."""
        if not SCIPY_FFT_AVAILABLE:
            return np.fft.fft2(matrix)

        rows, cols = matrix.shape
        result = np.zeros_like(matrix, dtype=complex)

        # Block-wise FFT processing
        for i in range(0, rows, block_size):
            for j in range(0, cols, block_size):
                i_end = min(i + block_size, rows)
                j_end = min(j + block_size, cols)

                # Process cache-sized blocks with SciPy FFT
                block = matrix[i:i_end, j:j_end]
                result[i:i_end, j:j_end] = scipy.fft.fft2(block)

        return result

    def _cache_optimized_matmul(self, matrix: np.ndarray, block_size: int) -> np.ndarray:
        """Cache-friendly matrix multiplication."""
        # For self-multiplication (common in correlation analysis)
        n = matrix.shape[0]
        result = np.zeros((n, n))

        # Blocked matrix multiplication for cache efficiency
        for i in range(0, n, block_size):
            for j in range(0, n, block_size):
                for k in range(0, n, block_size):
                    i_end = min(i + block_size, n)
                    j_end = min(j + block_size, n)
                    k_end = min(k + block_size, n)

                    # Accumulate block results
                    result[i:i_end, j:j_end] += (
                        matrix[i:i_end, k:k_end] @ matrix[k:k_end, j:j_end].T
                    )

        return result

    def parallel_chi_squared_cpu(
        self,
        parameter_sets: list[np.ndarray],
        phi_degrees: np.ndarray,
        data: tuple[np.ndarray, np.ndarray],
        max_workers: int | None = None
    ) -> list[float]:
        """
        Parallel chi-squared computation using CPU multiprocessing.

        Parameters
        ----------
        parameter_sets : list[np.ndarray]
            Parameter sets to evaluate
        phi_degrees : np.ndarray
            Angular values in degrees
        data : tuple[np.ndarray, np.ndarray]
            (experimental_g2, experimental_errors)
        max_workers : int, optional
            Maximum worker processes

        Returns
        -------
        list[float]
            Chi-squared values for each parameter set
        """
        if max_workers is None:
            max_workers = min(self.max_threads, len(parameter_sets))

        # Determine optimal chunk size
        if self.chunk_size == "auto":
            chunk_size = max(1, len(parameter_sets) // (max_workers * 4))
        else:
            chunk_size = self.chunk_size

        # Parallel processing with optimized chunking
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit chunks of work
            futures = []
            for i in range(0, len(parameter_sets), chunk_size):
                chunk = parameter_sets[i:i + chunk_size]
                future = executor.submit(
                    self._compute_chi_squared_chunk,
                    chunk, phi_degrees, data
                )
                futures.append(future)

            # Collect results
            chi_squared_values = []
            for future in as_completed(futures):
                chunk_results = future.result()
                chi_squared_values.extend(chunk_results)

        return chi_squared_values

    def _compute_chi_squared_chunk(
        self,
        parameter_chunk: list[np.ndarray],
        phi_degrees: np.ndarray,
        data: tuple[np.ndarray, np.ndarray]
    ) -> list[float]:
        """Compute chi-squared for a chunk of parameters."""
        experimental_g2, experimental_errors = data
        results = []

        for params in parameter_chunk:
            # Vectorized chi-squared computation
            predicted_g2 = self._compute_theoretical_g2(params, phi_degrees)

            # Efficient chi-squared calculation
            residuals = (experimental_g2 - predicted_g2) / experimental_errors
            chi_squared = np.sum(residuals**2)
            results.append(chi_squared)

        return results

    def _compute_theoretical_g2(self, params: np.ndarray, phi_degrees: np.ndarray) -> np.ndarray:
        """
        Compute theoretical g2 function (placeholder implementation).

        This should be replaced with the actual homodyne g2 calculation.
        """
        # Placeholder - should use actual homodyne correlation function
        # This would typically involve complex calculations based on the
        # theoretical framework from He et al. PNAS 2024
        return np.ones_like(phi_degrees)


# Numba-optimized functions (if available)
if NUMBA_AVAILABLE:

    @jit(nopython=True, parallel=True)
    def vectorized_correlation_numba(
        matrix1: np.ndarray,
        matrix2: np.ndarray
    ) -> np.ndarray:
        """
        Numba-optimized vectorized correlation with OpenMP threading.

        Parameters
        ----------
        matrix1, matrix2 : np.ndarray
            Input matrices for correlation

        Returns
        -------
        np.ndarray
            Correlation matrix
        """
        n1, m = matrix1.shape
        n2, _ = matrix2.shape
        result = np.zeros((n1, n2))

        # Parallel loop using Numba's prange (OpenMP)
        for i in prange(n1):
            for j in range(n2):
                # Vectorized correlation calculation
                mean1 = np.mean(matrix1[i, :])
                mean2 = np.mean(matrix2[j, :])

                num = np.sum((matrix1[i, :] - mean1) * (matrix2[j, :] - mean2))
                den1 = np.sqrt(np.sum((matrix1[i, :] - mean1)**2))
                den2 = np.sqrt(np.sum((matrix2[j, :] - mean2)**2))

                if den1 * den2 > 0:
                    result[i, j] = num / (den1 * den2)

        return result

    @jit(nopython=True, parallel=True)
    def fast_chi_squared_numba(
        experimental_data: np.ndarray,
        theoretical_data: np.ndarray,
        errors: np.ndarray
    ) -> float:
        """
        Numba-optimized chi-squared calculation with OpenMP.

        Parameters
        ----------
        experimental_data, theoretical_data, errors : np.ndarray
            Data arrays for chi-squared calculation

        Returns
        -------
        float
            Chi-squared value
        """
        n = experimental_data.size
        chi_squared = 0.0

        # Parallel reduction using Numba
        for i in prange(n):
            residual = (experimental_data.flat[i] - theoretical_data.flat[i]) / errors.flat[i]
            chi_squared += residual * residual

        return chi_squared

else:
    # Fallback implementations without Numba
    def vectorized_correlation_numba(matrix1: np.ndarray, matrix2: np.ndarray) -> np.ndarray:
        """Fallback correlation without Numba optimization."""
        return np.corrcoef(matrix1, matrix2)[:matrix1.shape[0], matrix1.shape[0]:]

    def fast_chi_squared_numba(
        experimental_data: np.ndarray,
        theoretical_data: np.ndarray,
        errors: np.ndarray
    ) -> float:
        """Fallback chi-squared without Numba optimization."""
        residuals = (experimental_data - theoretical_data) / errors
        return float(np.sum(residuals**2))


def get_cpu_optimization_info() -> dict[str, Any]:
    """
    Get comprehensive CPU optimization information.

    Returns
    -------
    dict[str, Any]
        CPU optimization capabilities and configuration
    """
    optimizer = CPUOptimizer()

    return {
        "cpu_count": optimizer.cpu_count,
        "cache_info": optimizer.cache_info,
        "simd_support": optimizer.simd_support,
        "numba_available": NUMBA_AVAILABLE,
        "scipy_fft_available": SCIPY_FFT_AVAILABLE,
        "recommended_threads": optimizer.max_threads,
        "optimization_features": {
            "cache_optimization": True,
            "vectorization": True,
            "multiprocessing": True,
            "openmp_threading": NUMBA_AVAILABLE,
            "simd_acceleration": any(optimizer.simd_support.values()),
        }
    }


def create_cpu_optimizer(config: dict[str, Any] | None = None) -> CPUOptimizer:
    """
    Factory function to create CPU optimizer.

    Parameters
    ----------
    config : dict[str, Any], optional
        CPU optimization configuration

    Returns
    -------
    CPUOptimizer
        Configured CPU optimizer instance
    """
    return CPUOptimizer(config)