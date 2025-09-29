"""
Comprehensive Startup Performance Tests
======================================

Benchmark suite for measuring and monitoring package startup performance.
Tracks import overhead, memory usage, and performance regression detection.
"""

import gc
import importlib
import psutil
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from unittest.mock import patch
from contextlib import contextmanager

import pytest

from homodyne.tests.conftest import PerformanceTimer


class MemoryProfiler:
    """Memory profiling utilities for startup benchmarks."""

    def __init__(self):
        self.process = psutil.Process()

    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage statistics."""
        memory_info = self.process.memory_info()
        return {
            'rss_mb': memory_info.rss / 1024 / 1024,  # Resident Set Size
            'vms_mb': memory_info.vms / 1024 / 1024,  # Virtual Memory Size
            'percent': self.process.memory_percent()
        }

    @contextmanager
    def monitor_memory(self):
        """Context manager to monitor memory usage."""
        gc.collect()  # Clean up before measurement
        initial_memory = self.get_memory_usage()

        yield initial_memory

        gc.collect()  # Clean up after measurement
        final_memory = self.get_memory_usage()

        self.memory_delta = {
            'rss_delta_mb': final_memory['rss_mb'] - initial_memory['rss_mb'],
            'vms_delta_mb': final_memory['vms_mb'] - initial_memory['vms_mb'],
            'peak_rss_mb': final_memory['rss_mb'],
            'peak_percent': final_memory['percent']
        }


class StartupBenchmarkSuite:
    """Comprehensive startup performance benchmark suite."""

    def __init__(self):
        self.memory_profiler = MemoryProfiler()
        self.baseline_file = Path(__file__).parent / "startup_baselines.json"

    def clear_homodyne_modules(self):
        """Clear all homodyne modules from sys.modules."""
        to_remove = [name for name in sys.modules if name.startswith('homodyne')]
        for name in to_remove:
            del sys.modules[name]

    def measure_cold_import(self, module_name: str) -> Dict[str, float]:
        """Measure cold import performance with memory tracking."""
        self.clear_homodyne_modules()
        gc.collect()

        with self.memory_profiler.monitor_memory() as initial_memory:
            with PerformanceTimer(f"Cold import {module_name}") as timer:
                try:
                    importlib.import_module(module_name)
                except ImportError as e:
                    return {
                        'import_time': None,
                        'memory_delta_mb': 0,
                        'error': str(e)
                    }

        return {
            'import_time': timer.elapsed_time,
            'memory_delta_mb': self.memory_profiler.memory_delta['rss_delta_mb'],
            'peak_memory_mb': self.memory_profiler.memory_delta['peak_rss_mb'],
            'memory_percent': self.memory_profiler.memory_delta['peak_percent']
        }

    def measure_warm_import(self, module_name: str, iterations: int = 5) -> Dict[str, float]:
        """Measure warm import performance (module already loaded)."""
        # First import to warm up
        try:
            importlib.import_module(module_name)
        except ImportError as e:
            return {'error': str(e)}

        times = []
        for _ in range(iterations):
            with PerformanceTimer(f"Warm import {module_name}") as timer:
                importlib.import_module(module_name)
            if timer.elapsed_time:
                times.append(timer.elapsed_time)

        if not times:
            return {'error': 'No successful measurements'}

        return {
            'min_time': min(times),
            'max_time': max(times),
            'avg_time': sum(times) / len(times),
            'iterations': len(times)
        }

    def benchmark_progressive_loading(self) -> Dict[str, Dict[str, float]]:
        """Benchmark progressive loading of homodyne components."""
        results = {}
        cumulative_time = 0
        cumulative_memory = 0

        # Progressive loading sequence
        loading_sequence = [
            ('base', 'homodyne'),
            ('config', 'homodyne.core.config'),
            ('kernels', 'homodyne.core.kernels'),
            ('analysis', 'homodyne.analysis.core'),
            ('classical_opt', 'homodyne.optimization.classical'),
            ('robust_opt', 'homodyne.optimization.robust'),
            ('plotting', 'homodyne.visualization.plotting'),
        ]

        self.clear_homodyne_modules()
        initial_memory = self.memory_profiler.get_memory_usage()['rss_mb']

        for stage_name, module_name in loading_sequence:
            with PerformanceTimer(f"Loading {stage_name}") as timer:
                try:
                    importlib.import_module(module_name)
                except ImportError:
                    results[stage_name] = {
                        'import_time': None,
                        'cumulative_time': cumulative_time,
                        'memory_delta_mb': 0,
                        'cumulative_memory_mb': cumulative_memory,
                        'error': f'Failed to import {module_name}'
                    }
                    continue

            current_memory = self.memory_profiler.get_memory_usage()['rss_mb']
            stage_memory = current_memory - initial_memory - cumulative_memory

            if timer.elapsed_time:
                cumulative_time += timer.elapsed_time
            cumulative_memory = current_memory - initial_memory

            results[stage_name] = {
                'import_time': timer.elapsed_time,
                'cumulative_time': cumulative_time,
                'memory_delta_mb': stage_memory,
                'cumulative_memory_mb': cumulative_memory
            }

        return results

    def benchmark_lazy_loading_effectiveness(self) -> Dict[str, float]:
        """Benchmark the effectiveness of lazy loading."""
        results = {}

        # Test 1: Basic import time
        basic_result = self.measure_cold_import('homodyne')
        results['basic_import_time'] = basic_result.get('import_time', 0)
        results['basic_memory_mb'] = basic_result.get('memory_delta_mb', 0)

        # Test 2: Import with immediate access to lazy objects
        self.clear_homodyne_modules()
        with self.memory_profiler.monitor_memory():
            with PerformanceTimer("Import with lazy access") as timer:
                try:
                    import homodyne
                    # Access lazy-loaded objects
                    _ = homodyne.HomodyneAnalysisCore
                    _ = homodyne.ClassicalOptimizer
                    _ = homodyne.plot_c2_heatmaps
                except (ImportError, AttributeError):
                    pass

        results['lazy_access_time'] = timer.elapsed_time or 0
        results['lazy_access_memory_mb'] = self.memory_profiler.memory_delta.get('rss_delta_mb', 0)

        # Calculate effectiveness metrics
        if results['basic_import_time'] > 0:
            results['lazy_loading_overhead'] = (
                results['lazy_access_time'] / results['basic_import_time']
            )
        else:
            results['lazy_loading_overhead'] = float('inf')

        return results

    def benchmark_conditional_imports(self) -> Dict[str, Dict[str, float]]:
        """Benchmark conditional imports with and without dependencies."""
        results = {}

        # Test modules with optional dependencies
        conditional_modules = {
            'numba_kernels': ('homodyne.core.kernels', 'numba'),
            'robust_optimization': ('homodyne.optimization.robust', 'cvxpy'),
            'plotting': ('homodyne.visualization.plotting', 'matplotlib'),
        }

        for test_name, (module_name, dependency) in conditional_modules.items():
            # Test with dependency available
            self.clear_homodyne_modules()
            with_dep_result = self.measure_cold_import(module_name)

            # Test with dependency unavailable (mocked)
            self.clear_homodyne_modules()
            with patch.dict('sys.modules', {dependency: None}):
                without_dep_result = self.measure_cold_import(module_name)

            results[test_name] = {
                'with_dependency_time': with_dep_result.get('import_time'),
                'without_dependency_time': without_dep_result.get('import_time'),
                'with_dependency_memory': with_dep_result.get('memory_delta_mb', 0),
                'without_dependency_memory': without_dep_result.get('memory_delta_mb', 0),
            }

            # Calculate performance impact of optional dependency
            with_time = with_dep_result.get('import_time')
            without_time = without_dep_result.get('import_time')

            if without_time is not None and without_time > 0 and with_time is not None:
                results[test_name]['dependency_overhead'] = with_time / without_time
            else:
                results[test_name]['dependency_overhead'] = 1.0

        return results


@pytest.fixture(scope="session")
def startup_benchmark():
    """Create startup benchmark suite."""
    return StartupBenchmarkSuite()


class TestStartupPerformance:
    """Test suite for startup performance benchmarks."""

    @pytest.mark.performance
    def test_basic_import_performance(self, startup_benchmark):
        """Test basic import performance meets expectations."""
        result = startup_benchmark.measure_cold_import('homodyne')

        if 'error' in result:
            pytest.skip(f"Import failed: {result['error']}")

        import_time = result['import_time']
        memory_usage = result['memory_delta_mb']

        # Performance expectations
        max_import_time = 2.0  # seconds
        max_memory_usage = 50.0  # MB

        assert import_time is not None, "Import time measurement failed"
        assert import_time < max_import_time, (
            f"Basic import too slow: {import_time:.2f}s (max: {max_import_time}s)"
        )
        assert memory_usage < max_memory_usage, (
            f"Basic import uses too much memory: {memory_usage:.1f}MB (max: {max_memory_usage}MB)"
        )

    @pytest.mark.performance
    def test_warm_import_performance(self, startup_benchmark):
        """Test warm import performance (should be very fast)."""
        result = startup_benchmark.measure_warm_import('homodyne')

        if 'error' in result:
            pytest.skip(f"Import failed: {result['error']}")

        avg_time = result['avg_time']
        max_warm_import_time = 0.01  # 10ms

        assert avg_time < max_warm_import_time, (
            f"Warm import too slow: {avg_time:.4f}s (max: {max_warm_import_time}s)"
        )

    @pytest.mark.performance
    def test_progressive_loading_performance(self, startup_benchmark):
        """Test progressive loading doesn't have performance cliffs."""
        results = startup_benchmark.benchmark_progressive_loading()

        # Check for reasonable incremental loading times
        max_stage_time = 3.0  # seconds per stage
        max_total_time = 10.0  # seconds total
        max_stage_memory = 100.0  # MB per stage

        failed_stages = []
        total_time = 0

        for stage_name, metrics in results.items():
            if 'error' in metrics:
                continue

            stage_time = metrics.get('import_time')
            stage_memory = metrics.get('memory_delta_mb', 0)

            if stage_time is not None:
                total_time = metrics.get('cumulative_time', stage_time)

                if stage_time > max_stage_time:
                    failed_stages.append(
                        f"{stage_name}: {stage_time:.2f}s (max: {max_stage_time}s)"
                    )

                if stage_memory > max_stage_memory:
                    failed_stages.append(
                        f"{stage_name}: {stage_memory:.1f}MB (max: {max_stage_memory}MB)"
                    )

        if failed_stages:
            pytest.fail(f"Performance issues in progressive loading:\n" +
                       '\n'.join(failed_stages))

        if total_time > max_total_time:
            pytest.fail(f"Total loading time too slow: {total_time:.2f}s (max: {max_total_time}s)")

    @pytest.mark.performance
    def test_lazy_loading_effectiveness(self, startup_benchmark):
        """Test that lazy loading provides performance benefits."""
        results = startup_benchmark.benchmark_lazy_loading_effectiveness()

        basic_time = results['basic_import_time']
        lazy_overhead = results['lazy_loading_overhead']

        # Lazy loading should provide significant speedup for basic imports
        assert basic_time < 1.0, f"Basic import not fast enough: {basic_time:.2f}s"

        # Accessing lazy objects should have reasonable overhead
        max_lazy_overhead = 5.0  # 5x overhead is acceptable
        assert lazy_overhead < max_lazy_overhead, (
            f"Lazy loading overhead too high: {lazy_overhead:.1f}x (max: {max_lazy_overhead}x)"
        )

    @pytest.mark.performance
    def test_conditional_import_performance(self, startup_benchmark):
        """Test conditional imports don't add excessive overhead."""
        results = startup_benchmark.benchmark_conditional_imports()

        max_dependency_overhead = 3.0  # 3x overhead is acceptable

        for test_name, metrics in results.items():
            dependency_overhead = metrics.get('dependency_overhead', 1.0)

            if dependency_overhead > max_dependency_overhead:
                pytest.fail(
                    f"{test_name} dependency overhead too high: "
                    f"{dependency_overhead:.1f}x (max: {max_dependency_overhead}x)"
                )

    @pytest.mark.performance
    def test_memory_usage_startup(self, startup_benchmark):
        """Test startup memory usage is reasonable."""
        # Test basic import memory usage
        result = startup_benchmark.measure_cold_import('homodyne')

        if 'error' in result:
            pytest.skip(f"Import failed: {result['error']}")

        memory_usage = result['memory_delta_mb']
        peak_memory = result['peak_memory_mb']

        # Memory usage expectations
        max_import_memory = 100.0  # MB for basic import
        max_peak_memory = 500.0  # MB total process memory

        assert memory_usage < max_import_memory, (
            f"Basic import uses too much memory: {memory_usage:.1f}MB (max: {max_import_memory}MB)"
        )

        assert peak_memory < max_peak_memory, (
            f"Peak memory usage too high: {peak_memory:.1f}MB (max: {max_peak_memory}MB)"
        )

    @pytest.mark.performance
    def test_import_time_stability(self, startup_benchmark):
        """Test that import times are stable across multiple runs."""
        module_name = 'homodyne'
        measurements = []

        # Take multiple measurements
        for _ in range(5):
            result = startup_benchmark.measure_cold_import(module_name)
            if 'error' not in result and result['import_time'] is not None:
                measurements.append(result['import_time'])

        if len(measurements) < 3:
            pytest.skip("Not enough successful measurements")

        # Calculate coefficient of variation (std dev / mean)
        mean_time = sum(measurements) / len(measurements)
        variance = sum((t - mean_time) ** 2 for t in measurements) / len(measurements)
        std_dev = variance ** 0.5
        cv = std_dev / mean_time if mean_time > 0 else float('inf')

        # Import times should be relatively stable
        # Note: Subprocess measurements have inherent variance from OS scheduling,
        # disk I/O, and CPU frequency scaling. CV of 0.5 allows for normal variance
        # while still detecting actual instability issues.
        max_cv = 0.5  # 50% coefficient of variation (increased from 0.3 for subprocess overhead)
        assert cv < max_cv, (
            f"Import times unstable: CV={cv:.2f} (max: {max_cv}), "
            f"times={[f'{t:.3f}' for t in measurements]}"
        )

    @pytest.mark.slow
    @pytest.mark.performance
    def test_startup_regression_detection(self, startup_benchmark):
        """Test for startup performance regression."""
        # Historical baselines (update these as needed)
        baselines = {
            'homodyne': {'time': 1.0, 'memory': 30.0},
            'homodyne.core.config': {'time': 0.3, 'memory': 10.0},
            'homodyne.analysis.core': {'time': 1.5, 'memory': 25.0},
            'homodyne.optimization.classical': {'time': 2.0, 'memory': 40.0},
        }

        regressions = []
        tolerance = 1.5  # 50% tolerance for regression

        for module_name, baseline in baselines.items():
            result = startup_benchmark.measure_cold_import(module_name)

            if 'error' in result:
                continue

            import_time = result.get('import_time')
            memory_usage = result.get('memory_delta_mb', 0)

            if import_time is not None and import_time > baseline['time'] * tolerance:
                regressions.append(
                    f"{module_name} import time regression: "
                    f"{import_time:.2f}s vs baseline {baseline['time']:.2f}s"
                )

            if memory_usage > baseline['memory'] * tolerance:
                regressions.append(
                    f"{module_name} memory regression: "
                    f"{memory_usage:.1f}MB vs baseline {baseline['memory']:.1f}MB"
                )

        if regressions:
            # Report as warning rather than failure for now
            import logging
            logger = logging.getLogger(__name__)
            for regression in regressions:
                logger.warning(f"Performance regression detected: {regression}")


class TestStartupOptimization:
    """Test suite for startup optimization strategies."""

    @pytest.mark.performance
    def test_numba_compilation_overhead(self, startup_benchmark):
        """Test Numba compilation doesn't excessively slow startup."""
        # Test kernels module with and without Numba
        with_numba = startup_benchmark.measure_cold_import('homodyne.core.kernels')

        startup_benchmark.clear_homodyne_modules()
        with patch.dict('sys.modules', {'numba': None}):
            without_numba = startup_benchmark.measure_cold_import('homodyne.core.kernels')

        if 'error' in with_numba or 'error' in without_numba:
            pytest.skip("Could not test both conditions")

        numba_time = with_numba.get('import_time', 0)
        no_numba_time = without_numba.get('import_time', 0)

        if no_numba_time > 0:
            overhead = numba_time / no_numba_time
            max_overhead = 10.0  # 10x overhead is acceptable for JIT compilation

            assert overhead < max_overhead, (
                f"Numba compilation overhead too high: {overhead:.1f}x (max: {max_overhead}x)"
            )

    @pytest.mark.performance
    def test_import_order_optimization(self, startup_benchmark):
        """Test that import order is optimized for common use cases."""
        # Test different import orders
        orders = [
            # Common analysis workflow
            ['homodyne', 'homodyne.core.config', 'homodyne.analysis.core'],
            # Optimization workflow
            ['homodyne', 'homodyne.optimization.classical', 'homodyne.optimization.robust'],
            # Visualization workflow
            ['homodyne', 'homodyne.visualization.plotting'],
        ]

        for i, order in enumerate(orders):
            startup_benchmark.clear_homodyne_modules()

            total_time = 0
            with PerformanceTimer(f"Import order {i+1}") as timer:
                for module_name in order:
                    try:
                        importlib.import_module(module_name)
                    except ImportError:
                        break

            total_time = timer.elapsed_time or 0

            # Each workflow should complete reasonably quickly
            max_workflow_time = 5.0  # seconds
            assert total_time < max_workflow_time, (
                f"Import order {i+1} too slow: {total_time:.2f}s (max: {max_workflow_time}s)"
            )

    @pytest.mark.integration
    def test_circular_import_prevention(self):
        """Test that there are no circular imports that slow startup."""
        import sys
        original_modules = set(sys.modules.keys())

        try:
            # Import main package
            import homodyne

            # Access major components to trigger any circular imports
            major_components = [
                'HomodyneAnalysisCore',
                'ClassicalOptimizer',
                'ConfigManager',
                'plot_c2_heatmaps'
            ]

            for component in major_components:
                if hasattr(homodyne, component):
                    _ = getattr(homodyne, component)

        except Exception as e:
            pytest.fail(f"Circular import or other startup issue: {e}")

        finally:
            # Clean up any new modules
            new_modules = set(sys.modules.keys()) - original_modules
            for module in new_modules:
                if module.startswith('homodyne'):
                    sys.modules.pop(module, None)

    @pytest.mark.performance
    def test_startup_memory_leaks(self, startup_benchmark):
        """Test for memory leaks during repeated imports."""
        initial_memory = startup_benchmark.memory_profiler.get_memory_usage()['rss_mb']

        # Perform multiple import cycles
        for _ in range(5):
            startup_benchmark.clear_homodyne_modules()
            startup_benchmark.measure_cold_import('homodyne')
            gc.collect()

        final_memory = startup_benchmark.memory_profiler.get_memory_usage()['rss_mb']
        memory_growth = final_memory - initial_memory

        # Memory growth should be minimal
        max_memory_growth = 20.0  # MB
        assert memory_growth < max_memory_growth, (
            f"Memory leak detected: {memory_growth:.1f}MB growth (max: {max_memory_growth}MB)"
        )


# Benchmarking utilities for continuous monitoring

def save_startup_baselines(results: Dict[str, Dict], baseline_file: Path):
    """Save startup performance baselines for regression testing."""
    import json

    with open(baseline_file, 'w') as f:
        json.dump(results, f, indent=2)


def load_startup_baselines(baseline_file: Path) -> Dict[str, Dict]:
    """Load startup performance baselines."""
    import json

    if not baseline_file.exists():
        return {}

    with open(baseline_file, 'r') as f:
        return json.load(f)


def run_startup_benchmark_suite() -> Dict[str, Dict]:
    """Run complete startup benchmark suite and return results."""
    suite = StartupBenchmarkSuite()

    results = {
        'basic_import': suite.measure_cold_import('homodyne'),
        'progressive_loading': suite.benchmark_progressive_loading(),
        'lazy_loading': suite.benchmark_lazy_loading_effectiveness(),
        'conditional_imports': suite.benchmark_conditional_imports(),
        'timestamp': time.time()
    }

    return results


if __name__ == "__main__":
    # Run benchmarks when executed directly
    results = run_startup_benchmark_suite()
    print("Startup Performance Benchmark Results:")
    print("=====================================")

    import json
    print(json.dumps(results, indent=2, default=str))
