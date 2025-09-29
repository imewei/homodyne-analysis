"""
Test Suite for Startup Performance Monitoring
=============================================

Comprehensive tests for startup performance monitoring, baseline management,
and regression detection system.

Authors: Wei Chen, Hongrui He
Institution: Argonne National Laboratory
"""

import pytest
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch, Mock

from homodyne.performance.startup_monitoring import (
    StartupMetrics,
    PerformanceBaseline,
    RegressionAlert,
    StartupPerformanceMonitor,
    get_startup_monitor,
    establish_default_baselines,
    check_startup_health,
)


class TestStartupMetrics:
    """Test suite for StartupMetrics dataclass."""

    def test_metrics_creation(self):
        """Test creation of startup metrics."""
        metrics = StartupMetrics(
            timestamp="2024-01-01T00:00:00Z",
            import_time=1.5,
            memory_usage_mb=100.0,
            python_version="3.12.0",
            package_version="0.7.1",
            platform="Linux x86_64",
            cpu_count=8,
            optimization_enabled=True,
            import_errors=[],
            dependency_load_times={"numpy": 0.5, "scipy": 0.3},
            total_modules_loaded=50,
            lazy_modules_count=10,
            immediate_modules_count=40,
        )

        assert metrics.import_time == 1.5
        assert metrics.memory_usage_mb == 100.0
        assert metrics.optimization_enabled is True
        assert len(metrics.dependency_load_times) == 2


class TestPerformanceBaseline:
    """Test suite for PerformanceBaseline dataclass."""

    def test_baseline_creation(self):
        """Test creation of performance baseline."""
        baseline = PerformanceBaseline(
            name="test_baseline",
            target_import_time=2.0,
            max_memory_usage_mb=150.0,
            acceptable_variance_percent=10.0,
            measurement_count=5,
            environment_tags=["test", "local"],
            created_at="2024-01-01T00:00:00Z",
            updated_at="2024-01-01T00:00:00Z",
        )

        assert baseline.name == "test_baseline"
        assert baseline.target_import_time == 2.0
        assert baseline.acceptable_variance_percent == 10.0
        assert "test" in baseline.environment_tags


class TestRegressionAlert:
    """Test suite for RegressionAlert dataclass."""

    def test_alert_creation(self):
        """Test creation of regression alert."""
        alert = RegressionAlert(
            alert_id="test_alert_123",
            metric_name="import_time",
            current_value=3.0,
            baseline_value=2.0,
            degradation_percent=50.0,
            severity="critical",
            timestamp="2024-01-01T00:00:00Z",
            recommendations=["Check for new dependencies"],
        )

        assert alert.metric_name == "import_time"
        assert alert.degradation_percent == 50.0
        assert alert.severity == "critical"
        assert len(alert.recommendations) == 1


class TestStartupPerformanceMonitor:
    """Test suite for StartupPerformanceMonitor class."""

    def setup_method(self):
        """Setup test environment with temporary directory."""
        self.temp_dir = tempfile.mkdtemp()
        self.monitor = StartupPerformanceMonitor(
            package_name="homodyne",
            baseline_dir=Path(self.temp_dir)
        )

    def test_monitor_initialization(self):
        """Test monitor initialization."""
        assert self.monitor.package_name == "homodyne"
        assert self.monitor.baseline_dir.exists()
        assert isinstance(self.monitor.baselines, dict)
        assert isinstance(self.monitor.recent_metrics, type(self.monitor.recent_metrics))

    def test_system_info_collection(self):
        """Test system information collection."""
        system_info = self.monitor._get_system_info()

        assert "python_version" in system_info
        assert "package_version" in system_info
        assert "platform" in system_info
        assert "cpu_count" in system_info
        assert isinstance(system_info["cpu_count"], int)
        assert system_info["cpu_count"] > 0

    @pytest.mark.slow
    def test_single_startup_measurement(self):
        """Test single startup measurement."""
        # Mock subprocess to avoid actual package import
        with patch('subprocess.run') as mock_subprocess:
            mock_result = Mock()
            mock_result.returncode = 0
            mock_result.stdout = "IMPORT_TIME: 1.5\n"
            mock_result.stderr = ""
            mock_subprocess.return_value = mock_result

            with patch('psutil.Process') as mock_process:
                mock_proc = Mock()
                mock_proc.memory_info.return_value.rss = 100 * 1024 * 1024  # 100MB
                mock_process.return_value = mock_proc

                result = self.monitor._single_startup_measurement()

                assert "import_time" in result
                assert "memory_usage" in result
                assert result["import_time"] == 1.5

    def test_baseline_establishment(self):
        """Test baseline establishment."""
        # Mock the measurement process
        with patch.object(self.monitor, 'measure_startup_performance') as mock_measure:
            mock_metrics = StartupMetrics(
                timestamp=datetime.now(timezone.utc).isoformat(),
                import_time=1.8,
                memory_usage_mb=120.0,
                python_version="3.12.0",
                package_version="0.7.1",
                platform="Test Platform",
                cpu_count=4,
                optimization_enabled=True,
                import_errors=[],
                dependency_load_times={},
                total_modules_loaded=50,
                lazy_modules_count=10,
                immediate_modules_count=40,
            )
            mock_measure.return_value = mock_metrics

            baseline = self.monitor.establish_baseline(
                name="test_baseline",
                target_import_time=2.0,
                max_memory_usage_mb=150.0,
                acceptable_variance_percent=10.0,
                measurement_count=3
            )

            assert baseline.name == "test_baseline"
            assert baseline.target_import_time == 2.0
            assert "test_baseline" in self.monitor.baselines

    def test_performance_regression_detection(self):
        """Test performance regression detection."""
        # Create a baseline
        baseline = PerformanceBaseline(
            name="test_baseline",
            target_import_time=2.0,
            max_memory_usage_mb=150.0,
            acceptable_variance_percent=10.0,
            measurement_count=5,
            environment_tags=["test"],
            created_at=datetime.now(timezone.utc).isoformat(),
            updated_at=datetime.now(timezone.utc).isoformat(),
        )
        self.monitor.baselines["test_baseline"] = baseline

        # Create metrics that exceed baseline
        bad_metrics = StartupMetrics(
            timestamp=datetime.now(timezone.utc).isoformat(),
            import_time=3.0,  # Exceeds baseline
            memory_usage_mb=200.0,  # Exceeds baseline
            python_version="3.12.0",
            package_version="0.7.1",
            platform="Test Platform",
            cpu_count=4,
            optimization_enabled=True,
            import_errors=[],
            dependency_load_times={},
            total_modules_loaded=50,
            lazy_modules_count=10,
            immediate_modules_count=40,
        )

        alerts = self.monitor.check_performance_regression(
            "test_baseline",
            bad_metrics
        )

        assert len(alerts) >= 1  # Should have at least import time alert
        assert any(alert.metric_name == "import_time" for alert in alerts)

    def test_performance_trend_analysis(self):
        """Test performance trend analysis."""
        # Mock historical metrics
        with patch.object(self.monitor, '_load_historical_metrics') as mock_load:
            mock_metrics = [
                StartupMetrics(
                    timestamp=(datetime.now(timezone.utc)).isoformat(),
                    import_time=1.5 + i * 0.1,  # Degrading trend
                    memory_usage_mb=100.0 + i * 5,
                    python_version="3.12.0",
                    package_version="0.7.1",
                    platform="Test Platform",
                    cpu_count=4,
                    optimization_enabled=True,
                    import_errors=[],
                    dependency_load_times={},
                    total_modules_loaded=50,
                    lazy_modules_count=10,
                    immediate_modules_count=40,
                )
                for i in range(10)
            ]
            mock_load.return_value = mock_metrics

            trend = self.monitor.get_performance_trend(days=30)

            assert "import_time_trend" in trend
            assert "memory_usage_trend" in trend
            assert trend["total_measurements"] == 10
            assert trend["import_time_trend"]["trend_direction"] == "degrading"

    def test_comprehensive_performance_report(self):
        """Test comprehensive performance report generation."""
        # Mock the measurement and regression check
        with patch.object(self.monitor, 'measure_startup_performance') as mock_measure:
            with patch.object(self.monitor, 'check_performance_regression') as mock_check:
                with patch.object(self.monitor, 'get_performance_trend') as mock_trend:

                    # Setup mocks
                    mock_metrics = StartupMetrics(
                        timestamp=datetime.now(timezone.utc).isoformat(),
                        import_time=1.5,
                        memory_usage_mb=100.0,
                        python_version="3.12.0",
                        package_version="0.7.1",
                        platform="Test Platform",
                        cpu_count=4,
                        optimization_enabled=True,
                        import_errors=[],
                        dependency_load_times={},
                        total_modules_loaded=50,
                        lazy_modules_count=10,
                        immediate_modules_count=40,
                    )
                    mock_measure.return_value = mock_metrics
                    mock_check.return_value = []  # No alerts
                    mock_trend.return_value = {"period_days": 30, "total_measurements": 10}

                    # Add a test baseline
                    self.monitor.baselines["test"] = PerformanceBaseline(
                        name="test",
                        target_import_time=2.0,
                        max_memory_usage_mb=150.0,
                        acceptable_variance_percent=10.0,
                        measurement_count=5,
                        environment_tags=["test"],
                        created_at=datetime.now(timezone.utc).isoformat(),
                        updated_at=datetime.now(timezone.utc).isoformat(),
                    )

                    report = self.monitor.generate_performance_report()

                    assert "current_metrics" in report
                    assert "baseline_status" in report
                    assert "alerts" in report
                    assert "trend_analysis" in report
                    assert "summary" in report

    def test_baseline_persistence(self):
        """Test baseline save/load functionality."""
        # Create and save a baseline
        baseline = PerformanceBaseline(
            name="persistent_test",
            target_import_time=1.5,
            max_memory_usage_mb=120.0,
            acceptable_variance_percent=15.0,
            measurement_count=3,
            environment_tags=["persistent", "test"],
            created_at=datetime.now(timezone.utc).isoformat(),
            updated_at=datetime.now(timezone.utc).isoformat(),
        )

        self.monitor.baselines["persistent_test"] = baseline
        self.monitor.save_baselines()

        # Create new monitor and load baselines
        new_monitor = StartupPerformanceMonitor(
            package_name="homodyne",
            baseline_dir=Path(self.temp_dir)
        )

        assert "persistent_test" in new_monitor.baselines
        loaded_baseline = new_monitor.baselines["persistent_test"]
        assert loaded_baseline.name == "persistent_test"
        assert loaded_baseline.target_import_time == 1.5

    def test_metrics_storage_and_retrieval(self):
        """Test metrics storage and historical retrieval."""
        # Create test metrics
        metrics = StartupMetrics(
            timestamp=datetime.now(timezone.utc).isoformat(),
            import_time=1.5,
            memory_usage_mb=100.0,
            python_version="3.12.0",
            package_version="0.7.1",
            platform="Test Platform",
            cpu_count=4,
            optimization_enabled=True,
            import_errors=[],
            dependency_load_times={"numpy": 0.5},
            total_modules_loaded=50,
            lazy_modules_count=10,
            immediate_modules_count=40,
        )

        # Store metrics
        self.monitor._store_metrics(metrics)

        # Retrieve historical metrics
        historical = self.monitor._load_historical_metrics(days=1)

        assert len(historical) >= 1
        assert historical[0].import_time == 1.5

    def test_performance_rating_calculation(self):
        """Test performance rating calculation."""
        # Excellent performance
        excellent_metrics = StartupMetrics(
            timestamp=datetime.now(timezone.utc).isoformat(),
            import_time=0.8,
            memory_usage_mb=80.0,
            python_version="3.12.0",
            package_version="0.7.1",
            platform="Test Platform",
            cpu_count=4,
            optimization_enabled=True,
            import_errors=[],
            dependency_load_times={},
            total_modules_loaded=50,
            lazy_modules_count=10,
            immediate_modules_count=40,
        )

        rating = self.monitor._calculate_performance_rating(excellent_metrics, [])
        assert rating == "excellent"

        # Poor performance with critical alerts
        critical_alert = RegressionAlert(
            alert_id="critical_test",
            metric_name="import_time",
            current_value=5.0,
            baseline_value=2.0,
            degradation_percent=150.0,
            severity="critical",
            timestamp=datetime.now(timezone.utc).isoformat(),
            recommendations=[],
        )

        rating = self.monitor._calculate_performance_rating(excellent_metrics, [critical_alert])
        assert rating == "poor"


class TestGlobalFunctions:
    """Test global monitoring functions."""

    def test_get_startup_monitor_singleton(self):
        """Test global monitor singleton behavior."""
        monitor1 = get_startup_monitor()
        monitor2 = get_startup_monitor()

        # Should return the same instance
        assert monitor1 is monitor2

    @pytest.mark.slow
    def test_establish_default_baselines(self):
        """Test establishment of default baselines."""
        with patch('homodyne.performance.startup_monitoring.StartupPerformanceMonitor.establish_baseline') as mock_establish:
            mock_baseline = PerformanceBaseline(
                name="test",
                target_import_time=2.0,
                max_memory_usage_mb=150.0,
                acceptable_variance_percent=10.0,
                measurement_count=5,
                environment_tags=[],
                created_at=datetime.now(timezone.utc).isoformat(),
                updated_at=datetime.now(timezone.utc).isoformat(),
            )
            mock_establish.return_value = mock_baseline

            baselines = establish_default_baselines()

            assert len(baselines) >= 1
            assert mock_establish.call_count >= 1

    def test_check_startup_health(self):
        """Test startup health check function."""
        with patch('homodyne.performance.startup_monitoring.StartupPerformanceMonitor.measure_startup_performance') as mock_measure:
            mock_metrics = StartupMetrics(
                timestamp=datetime.now(timezone.utc).isoformat(),
                import_time=1.5,  # Good performance
                memory_usage_mb=100.0,  # Good memory usage
                python_version="3.12.0",
                package_version="0.7.1",
                platform="Test Platform",
                cpu_count=4,
                optimization_enabled=True,
                import_errors=[],  # No errors
                dependency_load_times={},
                total_modules_loaded=50,
                lazy_modules_count=10,
                immediate_modules_count=40,
            )
            mock_measure.return_value = mock_metrics

            health = check_startup_health()

            assert "status" in health
            assert health["status"] == "healthy"
            assert "import_time" in health
            assert health["import_time"] == 1.5

    def test_check_startup_health_unhealthy(self):
        """Test startup health check with poor performance."""
        with patch('homodyne.performance.startup_monitoring.StartupPerformanceMonitor.measure_startup_performance') as mock_measure:
            mock_metrics = StartupMetrics(
                timestamp=datetime.now(timezone.utc).isoformat(),
                import_time=5.0,  # Poor performance
                memory_usage_mb=400.0,  # High memory usage
                python_version="3.12.0",
                package_version="0.7.1",
                platform="Test Platform",
                cpu_count=4,
                optimization_enabled=True,
                import_errors=["Some import error"],  # Has errors
                dependency_load_times={},
                total_modules_loaded=50,
                lazy_modules_count=10,
                immediate_modules_count=40,
            )
            mock_measure.return_value = mock_metrics

            health = check_startup_health()

            assert health["status"] == "unhealthy"
            assert len(health["issues"]) >= 1


class TestIntegrationWithMainPackage:
    """Test integration with main package functions."""

    def test_main_package_monitoring_functions(self):
        """Test monitoring functions in main package."""
        import homodyne

        # Should have monitoring functions available
        assert hasattr(homodyne, 'establish_performance_baseline')
        assert hasattr(homodyne, 'check_performance_health')
        assert hasattr(homodyne, 'monitor_startup_performance')
        assert hasattr(homodyne, 'get_performance_trend_report')

    def test_establish_baseline_function(self):
        """Test main package baseline establishment."""
        import homodyne

        # Mock to avoid actual measurement
        with patch('homodyne.performance.startup_monitoring.StartupPerformanceMonitor.establish_baseline') as mock_establish:
            mock_baseline = PerformanceBaseline(
                name="test_main",
                target_import_time=2.0,
                max_memory_usage_mb=150.0,
                acceptable_variance_percent=15.0,
                measurement_count=5,
                environment_tags=[],
                created_at=datetime.now(timezone.utc).isoformat(),
                updated_at=datetime.now(timezone.utc).isoformat(),
            )
            mock_establish.return_value = mock_baseline

            result = homodyne.establish_performance_baseline(
                name="test_main",
                target_import_time=2.0
            )

            assert "name" in result
            assert result["name"] == "test_main"

    def test_performance_health_check_function(self):
        """Test main package health check."""
        import homodyne

        # Mock to avoid actual measurement
        with patch('homodyne.performance.simple_monitoring.SimpleStartupMonitor.check_startup_health') as mock_health:
            mock_health.return_value = {
                "status": "good",
                "import_time": 1.5,
                "package_version": "0.7.1",
                "python_version": "3.12.0",
                "optimization_enabled": True,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "assessment": "Good startup performance"
            }

            health = homodyne.check_performance_health()

            assert "status" in health
            # The simple monitoring returns "good" for times < 2.0s
            assert health["status"] in ["excellent", "good", "fair"]

    def test_monitor_startup_performance_function(self):
        """Test main package monitoring function."""
        import homodyne

        # Mock to avoid actual measurement
        with patch('homodyne.performance.simple_monitoring.SimpleStartupMonitor.measure_startup_time') as mock_measure:
            from homodyne.performance.simple_monitoring import SimpleStartupMetrics

            mock_metrics = SimpleStartupMetrics(
                timestamp=datetime.now(timezone.utc).isoformat(),
                import_time=1.3,
                package_version="0.7.1",
                python_version="3.12.0",
                optimization_enabled=True,
                measurement_iterations=3
            )
            mock_measure.return_value = mock_metrics

            result = homodyne.monitor_startup_performance(iterations=3)

            assert "import_time" in result
            # Check that the result is close to the mocked value
            assert abs(result["import_time"] - 1.3) < 0.1


@pytest.mark.benchmark
class TestPerformanceBenchmarks:
    """Performance benchmarks for monitoring system."""

    @pytest.mark.slow
    def test_monitoring_overhead(self):
        """Test that monitoring system doesn't add significant overhead."""
        import time

        # Measure overhead of monitoring system initialization
        start_time = time.perf_counter()
        monitor = StartupPerformanceMonitor(
            baseline_dir=Path(tempfile.mkdtemp())
        )
        end_time = time.perf_counter()

        initialization_time = end_time - start_time

        # Should initialize quickly (adjust threshold as needed)
        assert initialization_time < 0.1, f"Monitor initialization too slow: {initialization_time:.4f}s"

    def test_baseline_operation_performance(self):
        """Test performance of baseline operations."""
        import time

        monitor = StartupPerformanceMonitor(
            baseline_dir=Path(tempfile.mkdtemp())
        )

        # Mock metrics to avoid measurement overhead
        mock_metrics = StartupMetrics(
            timestamp=datetime.now(timezone.utc).isoformat(),
            import_time=1.5,
            memory_usage_mb=100.0,
            python_version="3.12.0",
            package_version="0.7.1",
            platform="Test Platform",
            cpu_count=4,
            optimization_enabled=True,
            import_errors=[],
            dependency_load_times={},
            total_modules_loaded=50,
            lazy_modules_count=10,
            immediate_modules_count=40,
        )

        with patch.object(monitor, 'measure_startup_performance', return_value=mock_metrics):
            start_time = time.perf_counter()

            # Test baseline establishment
            monitor.establish_baseline("perf_test", 2.0, 150.0)

            # Test regression check
            monitor.check_performance_regression("perf_test", mock_metrics)

            end_time = time.perf_counter()

        operation_time = end_time - start_time

        # Should complete quickly
        assert operation_time < 1.0, f"Baseline operations too slow: {operation_time:.4f}s"


if __name__ == "__main__":
    pytest.main([__file__])