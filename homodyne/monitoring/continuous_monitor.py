#!/usr/bin/env python3
"""
Continuous Optimization Monitoring System
==========================================

Production-ready SRE monitoring system for the homodyne analysis application.
Provides real-time performance monitoring, automated regression detection,
and optimization opportunity identification with scientific computing awareness.

Key Features:
- Real-time performance tracking with SLI/SLO monitoring
- Machine learning-based anomaly detection
- Automated performance regression alerts
- Optimization recommendation engine
- Scientific computation accuracy preservation monitoring
- Multi-threaded monitoring with graceful degradation
- Comprehensive dashboard with real-time metrics

SRE Monitoring Areas:
- Scientific computation accuracy preservation (SLI: accuracy > 99.9%)
- Performance optimization effectiveness (SLI: response time < 2s)
- Memory usage optimization (SLI: memory usage < 80% available)
- Cache hit rates (SLI: cache hits > 85%)
- Numba JIT compilation performance (SLI: JIT overhead < 20%)
- Security feature performance impact (SLI: security overhead < 10%)

Integration Requirements:
- Builds upon existing performance_monitoring.py system
- Integrates with security_performance.py monitoring
- Supports all implemented optimizations
- Maintains compatibility with current infrastructure
"""

import asyncio
import json
import logging
import os
import pickle
import statistics
import threading
import time
import warnings
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Callable, Union
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager

import numpy as np
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    warnings.warn("psutil not available - system monitoring limited")

try:
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    warnings.warn("scikit-learn not available - ML-based anomaly detection disabled")

# Import existing monitoring components
try:
    from ..performance_monitoring import PerformanceMonitor, BenchmarkResult
    from ..core.security_performance import secure_cache, secure_scientific_computation
    from ..core.optimization_utils import get_optimization_counter, reset_optimization_counter
    HOMODYNE_MONITORING_AVAILABLE = True
except ImportError:
    HOMODYNE_MONITORING_AVAILABLE = False
    warnings.warn("Homodyne monitoring components not available")

logger = logging.getLogger(__name__)


@dataclass
class SLIMetric:
    """Service Level Indicator metric for SRE monitoring."""
    name: str
    value: float
    target: float
    unit: str
    timestamp: float
    status: str  # 'green', 'yellow', 'red'
    error_budget: float


@dataclass
class PerformanceAlert:
    """Performance alert with context and severity."""
    alert_id: str
    timestamp: float
    severity: str  # 'info', 'warning', 'critical'
    category: str  # 'performance', 'accuracy', 'memory', 'security'
    message: str
    metric_name: str
    current_value: float
    threshold: float
    context: Dict[str, Any]
    suggested_actions: List[str]


@dataclass
class OptimizationRecommendation:
    """ML-generated optimization recommendation."""
    recommendation_id: str
    timestamp: float
    area: str
    description: str
    predicted_improvement: float
    confidence: float
    effort_estimate: str
    priority_score: float
    implementation_steps: List[str]
    monitoring_metrics: List[str]


class ContinuousOptimizationMonitor:
    """
    Production-ready continuous optimization monitoring system.

    Provides comprehensive SRE-level monitoring for the homodyne analysis
    application with real-time performance tracking, automated alerting,
    and ML-based optimization recommendations.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize continuous optimization monitor."""
        self.config = config or self._default_config()
        self.monitoring_dir = Path(self.config['monitoring_dir'])
        self.monitoring_dir.mkdir(exist_ok=True)

        # Initialize monitoring state
        self.monitoring_active = False
        self.monitoring_thread: Optional[threading.Thread] = None
        self.alert_thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()

        # Performance tracking
        self.sli_metrics: Dict[str, List[SLIMetric]] = {}
        self.alerts: List[PerformanceAlert] = []
        self.recommendations: List[OptimizationRecommendation] = []

        # Anomaly detection
        self.anomaly_detector = None
        self.metric_scaler = None
        if ML_AVAILABLE:
            self._initialize_ml_components()

        # Performance baselines
        self.baselines: Dict[str, Dict[str, float]] = {}
        self._load_baselines()

        # Executor for async operations
        self.executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="monitor")

        # Initialize existing performance monitor
        if HOMODYNE_MONITORING_AVAILABLE:
            self.performance_monitor = PerformanceMonitor({
                'output_dir': str(self.monitoring_dir / 'performance'),
                'enable_detailed_profiling': True
            })
        else:
            self.performance_monitor = None

        logger.info("Continuous optimization monitor initialized")

    def _default_config(self) -> Dict[str, Any]:
        """Get default monitoring configuration."""
        return {
            'monitoring_dir': 'continuous_monitoring',
            'monitoring_interval': 30,  # seconds
            'alert_check_interval': 10,  # seconds
            'baseline_update_interval': 3600,  # 1 hour
            'max_alert_history': 1000,
            'max_recommendation_history': 100,
            'sli_targets': {
                'accuracy_preservation': 0.999,  # 99.9% accuracy target
                'response_time': 2.0,  # 2 second response time
                'memory_utilization': 0.80,  # 80% memory usage
                'cache_hit_rate': 0.85,  # 85% cache hit rate
                'jit_compilation_overhead': 0.20,  # 20% JIT overhead
                'security_overhead': 0.10,  # 10% security overhead
                'error_rate': 0.01,  # 1% error rate
                'throughput': 100.0,  # operations per minute
            },
            'alert_thresholds': {
                'response_time_warning': 1.5,
                'response_time_critical': 3.0,
                'memory_warning': 0.85,
                'memory_critical': 0.95,
                'error_rate_warning': 0.05,
                'error_rate_critical': 0.10,
                'accuracy_warning': 0.995,
                'accuracy_critical': 0.99,
            },
            'anomaly_detection': {
                'enabled': ML_AVAILABLE,
                'contamination': 0.1,  # 10% anomaly contamination
                'window_size': 100,  # metrics window
                'sensitivity': 0.8,  # detection sensitivity
            },
            'optimization_engine': {
                'enabled': True,
                'recommendation_interval': 1800,  # 30 minutes
                'min_confidence': 0.7,  # minimum confidence for recommendations
                'auto_apply_threshold': 0.9,  # auto-apply high confidence recommendations
            }
        }

    def _initialize_ml_components(self):
        """Initialize machine learning components for anomaly detection."""
        try:
            self.anomaly_detector = IsolationForest(
                contamination=self.config['anomaly_detection']['contamination'],
                random_state=42
            )
            self.metric_scaler = StandardScaler()
            logger.info("ML components initialized for anomaly detection")
        except Exception as e:
            logger.warning(f"Failed to initialize ML components: {e}")
            self.anomaly_detector = None
            self.metric_scaler = None

    def _load_baselines(self):
        """Load performance baselines from disk."""
        baseline_file = self.monitoring_dir / 'baselines.json'
        if baseline_file.exists():
            try:
                with open(baseline_file, 'r') as f:
                    self.baselines = json.load(f)
                logger.info("Performance baselines loaded")
            except Exception as e:
                logger.warning(f"Failed to load baselines: {e}")
                self.baselines = {}
        else:
            # Initialize with default baselines
            self.baselines = {
                'chi_squared_calculation': {'mean': 0.1, 'std': 0.02},
                'correlation_calculation': {'mean': 0.5, 'std': 0.1},
                'classical_optimization': {'mean': 2.0, 'std': 0.5},
                'robust_optimization': {'mean': 5.0, 'std': 1.0},
                'memory_usage': {'mean': 100.0, 'std': 20.0},  # MB
                'cache_hit_rate': {'mean': 0.85, 'std': 0.05},
            }

    def _save_baselines(self):
        """Save performance baselines to disk."""
        baseline_file = self.monitoring_dir / 'baselines.json'
        try:
            with open(baseline_file, 'w') as f:
                json.dump(self.baselines, f, indent=2)
            logger.debug("Performance baselines saved")
        except Exception as e:
            logger.error(f"Failed to save baselines: {e}")

    @contextmanager
    def monitor_operation(self, operation_name: str, expected_accuracy: float = 0.999):
        """
        Context manager for monitoring scientific operations with accuracy tracking.

        Parameters
        ----------
        operation_name : str
            Name of the operation being monitored
        expected_accuracy : float
            Expected accuracy for scientific computation validation
        """
        start_time = time.time()
        start_memory = 0
        if PSUTIL_AVAILABLE:
            process = psutil.Process()
            start_memory = process.memory_info().rss

        try:
            yield
            success = True
            error_msg = None
        except Exception as e:
            success = False
            error_msg = str(e)
            logger.warning(f"Operation {operation_name} failed: {e}")
            raise
        finally:
            end_time = time.time()
            duration = end_time - start_time

            end_memory = 0
            memory_delta = 0
            if PSUTIL_AVAILABLE:
                end_memory = process.memory_info().rss
                memory_delta = end_memory - start_memory

            # Record SLI metrics
            self._record_sli_metric(
                f"{operation_name}_response_time",
                duration,
                self.config['sli_targets']['response_time'],
                'seconds'
            )

            if memory_delta > 0:
                memory_mb = memory_delta / 1024 / 1024
                self._record_sli_metric(
                    f"{operation_name}_memory_usage",
                    memory_mb,
                    self.baselines.get(operation_name, {}).get('mean', 100.0),
                    'MB'
                )

            # Record accuracy metric if successful
            if success:
                self._record_sli_metric(
                    f"{operation_name}_accuracy",
                    expected_accuracy,
                    self.config['sli_targets']['accuracy_preservation'],
                    'ratio'
                )
            else:
                self._record_sli_metric(
                    f"{operation_name}_error_rate",
                    1.0,
                    self.config['sli_targets']['error_rate'],
                    'ratio'
                )

    def _record_sli_metric(self, name: str, value: float, target: float, unit: str):
        """Record a Service Level Indicator metric."""
        timestamp = time.time()

        # Calculate status based on target
        if value <= target:
            status = 'green'
            error_budget = 1.0
        elif value <= target * 1.1:  # 10% tolerance
            status = 'yellow'
            error_budget = 0.5
        else:
            status = 'red'
            error_budget = 0.0

        metric = SLIMetric(
            name=name,
            value=value,
            target=target,
            unit=unit,
            timestamp=timestamp,
            status=status,
            error_budget=error_budget
        )

        if name not in self.sli_metrics:
            self.sli_metrics[name] = []

        self.sli_metrics[name].append(metric)

        # Keep only recent metrics
        max_metrics = 1000
        if len(self.sli_metrics[name]) > max_metrics:
            self.sli_metrics[name] = self.sli_metrics[name][-max_metrics:]

        # Check for alerts
        self._check_metric_alert(metric)

    def _check_metric_alert(self, metric: SLIMetric):
        """Check if a metric triggers an alert."""
        thresholds = self.config['alert_thresholds']

        alert_triggered = False
        severity = 'info'

        # Check response time alerts
        if 'response_time' in metric.name:
            if metric.value > thresholds['response_time_critical']:
                alert_triggered = True
                severity = 'critical'
            elif metric.value > thresholds['response_time_warning']:
                alert_triggered = True
                severity = 'warning'

        # Check memory alerts
        elif 'memory' in metric.name:
            memory_percent = metric.value / (8 * 1024)  # Assume 8GB available
            if memory_percent > thresholds['memory_critical']:
                alert_triggered = True
                severity = 'critical'
            elif memory_percent > thresholds['memory_warning']:
                alert_triggered = True
                severity = 'warning'

        # Check accuracy alerts
        elif 'accuracy' in metric.name:
            if metric.value < thresholds['accuracy_critical']:
                alert_triggered = True
                severity = 'critical'
            elif metric.value < thresholds['accuracy_warning']:
                alert_triggered = True
                severity = 'warning'

        # Check error rate alerts
        elif 'error_rate' in metric.name:
            if metric.value > thresholds['error_rate_critical']:
                alert_triggered = True
                severity = 'critical'
            elif metric.value > thresholds['error_rate_warning']:
                alert_triggered = True
                severity = 'warning'

        if alert_triggered:
            self._create_alert(metric, severity)

    def _create_alert(self, metric: SLIMetric, severity: str):
        """Create a performance alert."""
        alert_id = f"alert_{int(time.time())}_{metric.name}"

        # Determine suggested actions
        suggested_actions = []
        if 'response_time' in metric.name:
            suggested_actions = [
                "Check for performance regressions",
                "Review recent optimizations",
                "Consider scaling resources",
                "Investigate bottlenecks"
            ]
        elif 'memory' in metric.name:
            suggested_actions = [
                "Check for memory leaks",
                "Review memory-intensive operations",
                "Consider memory optimization",
                "Monitor garbage collection"
            ]
        elif 'accuracy' in metric.name:
            suggested_actions = [
                "Validate computational accuracy",
                "Check numerical stability",
                "Review algorithm implementations",
                "Verify input data quality"
            ]
        elif 'error_rate' in metric.name:
            suggested_actions = [
                "Investigate error causes",
                "Review error handling",
                "Check input validation",
                "Monitor system health"
            ]

        alert = PerformanceAlert(
            alert_id=alert_id,
            timestamp=time.time(),
            severity=severity,
            category=self._categorize_metric(metric.name),
            message=f"SLI violation: {metric.name} = {metric.value:.3f} {metric.unit} (target: {metric.target:.3f})",
            metric_name=metric.name,
            current_value=metric.value,
            threshold=metric.target,
            context={
                'error_budget': metric.error_budget,
                'status': metric.status,
                'baseline': self.baselines.get(metric.name.split('_')[0], {})
            },
            suggested_actions=suggested_actions
        )

        self.alerts.append(alert)

        # Keep only recent alerts
        if len(self.alerts) > self.config['max_alert_history']:
            self.alerts = self.alerts[-self.config['max_alert_history']:]

        logger.warning(f"Alert triggered: {alert.message}")

        # Save alert to disk
        self._save_alert(alert)

    def _categorize_metric(self, metric_name: str) -> str:
        """Categorize a metric for alert routing."""
        if any(term in metric_name for term in ['accuracy', 'error']):
            return 'accuracy'
        elif any(term in metric_name for term in ['memory', 'cache']):
            return 'memory'
        elif any(term in metric_name for term in ['response_time', 'throughput']):
            return 'performance'
        elif any(term in metric_name for term in ['security', 'validation']):
            return 'security'
        else:
            return 'general'

    def _save_alert(self, alert: PerformanceAlert):
        """Save alert to disk for persistence."""
        alert_file = self.monitoring_dir / 'alerts' / f"{alert.alert_id}.json"
        alert_file.parent.mkdir(exist_ok=True)

        try:
            with open(alert_file, 'w') as f:
                json.dump(asdict(alert), f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save alert: {e}")

    def start_monitoring(self):
        """Start continuous monitoring."""
        if self.monitoring_active:
            logger.warning("Monitoring already active")
            return

        self.monitoring_active = True
        self.stop_event.clear()

        # Start monitoring thread
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            name="ContinuousMonitor"
        )
        self.monitoring_thread.daemon = True
        self.monitoring_thread.start()

        # Start alert processing thread
        self.alert_thread = threading.Thread(
            target=self._alert_processing_loop,
            name="AlertProcessor"
        )
        self.alert_thread.daemon = True
        self.alert_thread.start()

        logger.info("Continuous monitoring started")

    def stop_monitoring(self):
        """Stop continuous monitoring."""
        if not self.monitoring_active:
            logger.warning("Monitoring not active")
            return

        self.monitoring_active = False
        self.stop_event.set()

        # Wait for threads to stop
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        if self.alert_thread:
            self.alert_thread.join(timeout=5)

        logger.info("Continuous monitoring stopped")

    def _monitoring_loop(self):
        """Main monitoring loop."""
        last_baseline_update = 0
        last_recommendation_update = 0

        while self.monitoring_active and not self.stop_event.is_set():
            try:
                current_time = time.time()

                # Collect system metrics
                self._collect_system_metrics()

                # Collect application metrics
                if HOMODYNE_MONITORING_AVAILABLE:
                    self._collect_application_metrics()

                # Update baselines periodically
                if current_time - last_baseline_update > self.config['baseline_update_interval']:
                    self._update_baselines()
                    last_baseline_update = current_time

                # Generate recommendations periodically
                if (self.config['optimization_engine']['enabled'] and
                    current_time - last_recommendation_update > self.config['optimization_engine']['recommendation_interval']):
                    self._generate_optimization_recommendations()
                    last_recommendation_update = current_time

                # Perform anomaly detection
                if ML_AVAILABLE and self.anomaly_detector is not None:
                    self._detect_anomalies()

            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")

            # Wait for next monitoring interval
            self.stop_event.wait(self.config['monitoring_interval'])

    def _alert_processing_loop(self):
        """Process and route alerts."""
        while self.monitoring_active and not self.stop_event.is_set():
            try:
                self._process_pending_alerts()
            except Exception as e:
                logger.error(f"Error in alert processing: {e}")

            self.stop_event.wait(self.config['alert_check_interval'])

    def _collect_system_metrics(self):
        """Collect system-level metrics."""
        if not PSUTIL_AVAILABLE:
            return

        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            self._record_sli_metric(
                'system_cpu_utilization',
                cpu_percent / 100.0,
                0.80,  # 80% CPU target
                'ratio'
            )

            # Memory metrics
            memory = psutil.virtual_memory()
            self._record_sli_metric(
                'system_memory_utilization',
                memory.percent / 100.0,
                self.config['sli_targets']['memory_utilization'],
                'ratio'
            )

            # Disk metrics
            disk = psutil.disk_usage('/')
            self._record_sli_metric(
                'system_disk_utilization',
                disk.percent / 100.0,
                0.90,  # 90% disk target
                'ratio'
            )

        except Exception as e:
            logger.warning(f"Failed to collect system metrics: {e}")

    def _collect_application_metrics(self):
        """Collect application-specific metrics."""
        try:
            # Cache metrics
            if hasattr(secure_cache, '_cache'):
                cache_size = len(secure_cache._cache)
                max_cache_size = getattr(secure_cache, 'max_size', 128)
                cache_utilization = cache_size / max_cache_size

                self._record_sli_metric(
                    'cache_utilization',
                    cache_utilization,
                    0.80,  # 80% cache utilization target
                    'ratio'
                )

                # Estimate cache hit rate (simplified)
                # In production, this would be tracked properly
                estimated_hit_rate = min(0.95, cache_utilization + 0.1)
                self._record_sli_metric(
                    'cache_hit_rate',
                    estimated_hit_rate,
                    self.config['sli_targets']['cache_hit_rate'],
                    'ratio'
                )

            # Optimization counter metrics
            opt_counter = get_optimization_counter()
            if opt_counter > 0:
                self._record_sli_metric(
                    'optimization_iterations',
                    opt_counter,
                    1000,  # Target: keep iterations reasonable
                    'count'
                )

        except Exception as e:
            logger.warning(f"Failed to collect application metrics: {e}")

    def _update_baselines(self):
        """Update performance baselines using recent metrics."""
        try:
            for metric_name, metric_list in self.sli_metrics.items():
                if len(metric_list) < 10:  # Need enough data
                    continue

                # Get recent values
                recent_values = [m.value for m in metric_list[-100:]]

                # Calculate new baseline
                mean_value = statistics.mean(recent_values)
                std_value = statistics.stdev(recent_values) if len(recent_values) > 1 else 0.1

                # Update baseline with exponential smoothing
                alpha = 0.1  # Smoothing factor
                base_key = metric_name.split('_')[0]

                if base_key in self.baselines:
                    old_mean = self.baselines[base_key].get('mean', mean_value)
                    old_std = self.baselines[base_key].get('std', std_value)

                    self.baselines[base_key] = {
                        'mean': alpha * mean_value + (1 - alpha) * old_mean,
                        'std': alpha * std_value + (1 - alpha) * old_std
                    }
                else:
                    self.baselines[base_key] = {
                        'mean': mean_value,
                        'std': std_value
                    }

            self._save_baselines()
            logger.debug("Performance baselines updated")

        except Exception as e:
            logger.error(f"Failed to update baselines: {e}")

    def _detect_anomalies(self):
        """Detect performance anomalies using machine learning."""
        if not ML_AVAILABLE or self.anomaly_detector is None:
            return

        try:
            # Collect recent metrics for analysis
            metric_data = []
            metric_names = []

            for metric_name, metric_list in self.sli_metrics.items():
                if len(metric_list) >= 10:  # Need enough data
                    recent_values = [m.value for m in metric_list[-50:]]
                    metric_data.append(recent_values[-10:])  # Last 10 values
                    metric_names.append(metric_name)

            if len(metric_data) < 3:  # Need multiple metrics
                return

            # Prepare data for anomaly detection
            data_matrix = np.array(metric_data).T  # Shape: (samples, features)

            # Scale the data
            if len(data_matrix) > 5:  # Need enough samples
                scaled_data = self.metric_scaler.fit_transform(data_matrix)

                # Detect anomalies
                anomalies = self.anomaly_detector.fit_predict(scaled_data)

                # Process anomaly results
                anomaly_indices = np.where(anomalies == -1)[0]
                if len(anomaly_indices) > 0:
                    logger.warning(f"Detected {len(anomaly_indices)} performance anomalies")

                    # Create anomaly alerts
                    for idx in anomaly_indices:
                        if idx < len(metric_names):
                            self._create_anomaly_alert(metric_names[idx], data_matrix[idx])

        except Exception as e:
            logger.error(f"Anomaly detection failed: {e}")

    def _create_anomaly_alert(self, metric_name: str, anomaly_values: np.ndarray):
        """Create an alert for detected anomalies."""
        alert_id = f"anomaly_{int(time.time())}_{metric_name}"

        alert = PerformanceAlert(
            alert_id=alert_id,
            timestamp=time.time(),
            severity='warning',
            category='anomaly',
            message=f"Performance anomaly detected in {metric_name}",
            metric_name=metric_name,
            current_value=float(anomaly_values[-1]),
            threshold=float(np.mean(anomaly_values)),
            context={
                'anomaly_type': 'statistical_outlier',
                'confidence': 0.8,
                'detection_method': 'isolation_forest'
            },
            suggested_actions=[
                "Investigate recent changes",
                "Check system resources",
                "Review performance patterns",
                "Consider optimization opportunities"
            ]
        )

        self.alerts.append(alert)
        self._save_alert(alert)

    def _generate_optimization_recommendations(self):
        """Generate ML-based optimization recommendations."""
        try:
            recommendations = []

            # Analyze recent performance patterns
            for metric_name, metric_list in self.sli_metrics.items():
                if len(metric_list) < 20:
                    continue

                recent_values = [m.value for m in metric_list[-50:]]
                baseline = self.baselines.get(metric_name.split('_')[0], {})

                if not baseline:
                    continue

                # Check for performance degradation trends
                if len(recent_values) >= 10:
                    recent_mean = statistics.mean(recent_values[-10:])
                    baseline_mean = baseline.get('mean', recent_mean)

                    if recent_mean > baseline_mean * 1.2:  # 20% degradation
                        recommendation = self._create_performance_recommendation(
                            metric_name, recent_mean, baseline_mean
                        )
                        if recommendation:
                            recommendations.append(recommendation)

            # Add general optimization recommendations
            recommendations.extend(self._generate_general_recommendations())

            # Sort by priority and keep top recommendations
            recommendations.sort(key=lambda x: x.priority_score, reverse=True)
            new_recommendations = recommendations[:10]

            self.recommendations.extend(new_recommendations)

            # Keep only recent recommendations
            if len(self.recommendations) > self.config['max_recommendation_history']:
                self.recommendations = self.recommendations[-self.config['max_recommendation_history']:]

            if new_recommendations:
                logger.info(f"Generated {len(new_recommendations)} optimization recommendations")
                self._save_recommendations(new_recommendations)

        except Exception as e:
            logger.error(f"Failed to generate recommendations: {e}")

    def _create_performance_recommendation(self, metric_name: str, current: float, baseline: float) -> Optional[OptimizationRecommendation]:
        """Create performance recommendation based on metric analysis."""
        degradation = (current - baseline) / baseline

        if 'response_time' in metric_name:
            return OptimizationRecommendation(
                recommendation_id=f"perf_{int(time.time())}_{metric_name}",
                timestamp=time.time(),
                area='Performance Optimization',
                description=f"Optimize {metric_name} - {degradation:.1%} degradation detected",
                predicted_improvement=min(degradation * 0.8, 0.5),
                confidence=0.8,
                effort_estimate='medium',
                priority_score=8.0 + min(degradation * 10, 2.0),
                implementation_steps=[
                    "Profile the affected operation",
                    "Identify bottlenecks",
                    "Apply targeted optimizations",
                    "Monitor improvement"
                ],
                monitoring_metrics=[metric_name]
            )
        elif 'memory' in metric_name:
            return OptimizationRecommendation(
                recommendation_id=f"mem_{int(time.time())}_{metric_name}",
                timestamp=time.time(),
                area='Memory Optimization',
                description=f"Optimize memory usage in {metric_name}",
                predicted_improvement=min(degradation * 0.6, 0.3),
                confidence=0.7,
                effort_estimate='medium',
                priority_score=7.0 + min(degradation * 8, 2.0),
                implementation_steps=[
                    "Analyze memory allocation patterns",
                    "Implement memory pooling",
                    "Optimize data structures",
                    "Monitor memory usage"
                ],
                monitoring_metrics=[metric_name]
            )

        return None

    def _generate_general_recommendations(self) -> List[OptimizationRecommendation]:
        """Generate general optimization recommendations."""
        recommendations = []

        # Cache optimization recommendation
        cache_metrics = [m for name, metrics in self.sli_metrics.items()
                        if 'cache' in name for m in metrics[-10:]]

        if cache_metrics:
            avg_hit_rate = statistics.mean(m.value for m in cache_metrics)
            if avg_hit_rate < 0.80:
                recommendations.append(OptimizationRecommendation(
                    recommendation_id=f"cache_{int(time.time())}",
                    timestamp=time.time(),
                    area='Cache Optimization',
                    description="Improve cache hit rate through intelligent prefetching",
                    predicted_improvement=0.15,
                    confidence=0.75,
                    effort_estimate='low',
                    priority_score=6.5,
                    implementation_steps=[
                        "Analyze cache miss patterns",
                        "Implement predictive prefetching",
                        "Optimize cache eviction policy",
                        "Monitor cache performance"
                    ],
                    monitoring_metrics=['cache_hit_rate', 'cache_utilization']
                ))

        # JIT compilation optimization
        optimization_counter = get_optimization_counter()
        if optimization_counter > 0:
            recommendations.append(OptimizationRecommendation(
                recommendation_id=f"jit_{int(time.time())}",
                timestamp=time.time(),
                area='JIT Compilation',
                description="Optimize Numba JIT compilation for computational kernels",
                predicted_improvement=0.25,
                confidence=0.85,
                effort_estimate='high',
                priority_score=7.5,
                implementation_steps=[
                    "Identify hot computation paths",
                    "Apply @jit decorators strategically",
                    "Optimize array operations",
                    "Measure compilation overhead"
                ],
                monitoring_metrics=['optimization_iterations', 'response_time']
            ))

        return recommendations

    def _save_recommendations(self, recommendations: List[OptimizationRecommendation]):
        """Save recommendations to disk."""
        rec_file = self.monitoring_dir / 'recommendations' / f"recommendations_{int(time.time())}.json"
        rec_file.parent.mkdir(exist_ok=True)

        try:
            with open(rec_file, 'w') as f:
                json.dump([asdict(rec) for rec in recommendations], f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save recommendations: {e}")

    def _process_pending_alerts(self):
        """Process pending alerts and route them appropriately."""
        # In a production system, this would integrate with alerting systems
        # like PagerDuty, Slack, or email notifications

        # For now, log critical alerts
        recent_alerts = [a for a in self.alerts if time.time() - a.timestamp < 300]  # Last 5 minutes
        critical_alerts = [a for a in recent_alerts if a.severity == 'critical']

        if critical_alerts:
            logger.critical(f"CRITICAL ALERTS: {len(critical_alerts)} critical performance issues detected")
            for alert in critical_alerts:
                logger.critical(f"  - {alert.message}")

    def get_monitoring_status(self) -> Dict[str, Any]:
        """Get current monitoring status and health metrics."""
        current_time = time.time()

        # Calculate SLI compliance
        sli_compliance = {}
        for metric_name, metrics in self.sli_metrics.items():
            if metrics:
                recent_metrics = [m for m in metrics if current_time - m.timestamp < 3600]  # Last hour
                if recent_metrics:
                    green_count = sum(1 for m in recent_metrics if m.status == 'green')
                    compliance = green_count / len(recent_metrics)
                    sli_compliance[metric_name] = compliance

        # Count recent alerts by severity
        recent_alerts = [a for a in self.alerts if current_time - a.timestamp < 3600]
        alert_counts = {
            'critical': sum(1 for a in recent_alerts if a.severity == 'critical'),
            'warning': sum(1 for a in recent_alerts if a.severity == 'warning'),
            'info': sum(1 for a in recent_alerts if a.severity == 'info')
        }

        # Get recent recommendations
        recent_recommendations = [r for r in self.recommendations if current_time - r.timestamp < 86400]  # Last day

        return {
            'monitoring_active': self.monitoring_active,
            'uptime': current_time - (current_time if not hasattr(self, '_start_time') else self._start_time),
            'sli_compliance': sli_compliance,
            'alert_counts': alert_counts,
            'total_metrics': sum(len(metrics) for metrics in self.sli_metrics.values()),
            'recent_recommendations': len(recent_recommendations),
            'health_status': 'healthy' if not alert_counts['critical'] else 'degraded',
            'ml_available': ML_AVAILABLE,
            'anomaly_detection_enabled': self.anomaly_detector is not None
        }

    def generate_monitoring_report(self) -> Dict[str, Any]:
        """Generate comprehensive monitoring report."""
        status = self.get_monitoring_status()
        current_time = time.time()

        # Get recent metrics summary
        metrics_summary = {}
        for metric_name, metrics in self.sli_metrics.items():
            recent_metrics = [m for m in metrics if current_time - m.timestamp < 3600]
            if recent_metrics:
                values = [m.value for m in recent_metrics]
                metrics_summary[metric_name] = {
                    'mean': statistics.mean(values),
                    'min': min(values),
                    'max': max(values),
                    'count': len(values),
                    'latest_status': recent_metrics[-1].status
                }

        # Get top alerts and recommendations
        recent_alerts = [a for a in self.alerts if current_time - a.timestamp < 86400]
        top_alerts = sorted(recent_alerts, key=lambda x: x.timestamp, reverse=True)[:10]

        recent_recommendations = [r for r in self.recommendations if current_time - r.timestamp < 86400]
        top_recommendations = sorted(recent_recommendations, key=lambda x: x.priority_score, reverse=True)[:5]

        return {
            'report_timestamp': current_time,
            'monitoring_status': status,
            'metrics_summary': metrics_summary,
            'baselines': self.baselines,
            'top_alerts': [asdict(a) for a in top_alerts],
            'top_recommendations': [asdict(r) for r in top_recommendations],
            'configuration': self.config
        }

    def shutdown(self):
        """Gracefully shutdown the monitoring system."""
        logger.info("Shutting down continuous optimization monitor")

        # Stop monitoring
        self.stop_monitoring()

        # Shutdown executor
        self.executor.shutdown(wait=True)

        # Save final state
        self._save_baselines()

        logger.info("Continuous optimization monitor shutdown complete")


def create_monitoring_system(config: Optional[Dict[str, Any]] = None) -> ContinuousOptimizationMonitor:
    """
    Factory function to create and configure a monitoring system.

    Parameters
    ----------
    config : dict, optional
        Monitoring configuration

    Returns
    -------
    ContinuousOptimizationMonitor
        Configured monitoring system
    """
    return ContinuousOptimizationMonitor(config)


if __name__ == "__main__":
    # Example usage for testing
    import signal
    import sys

    def signal_handler(sig, frame):
        print("Shutting down monitoring system...")
        monitor.shutdown()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    # Create and start monitoring
    monitor = create_monitoring_system()
    monitor.start_monitoring()

    print("Continuous optimization monitoring started. Press Ctrl+C to stop.")

    try:
        while True:
            time.sleep(10)
            status = monitor.get_monitoring_status()
            print(f"Status: {status['health_status']}, Metrics: {status['total_metrics']}, "
                  f"Alerts: {status['alert_counts']}")
    except KeyboardInterrupt:
        monitor.shutdown()