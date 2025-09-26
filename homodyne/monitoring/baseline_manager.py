#!/usr/bin/env python3
"""
Performance Baseline Management System
======================================

Advanced baseline management system for tracking, updating, and validating
performance baselines in the homodyne analysis application. Provides automated
baseline establishment, drift detection, and intelligent baseline updates.

Key Features:
- Automated baseline establishment from historical data
- Statistical baseline validation with confidence intervals
- Baseline drift detection and alerting
- Intelligent baseline update strategies
- Multi-dimensional baseline tracking (time, environment, workload)
- Baseline versioning and rollback capabilities
- Performance regression detection against baselines
- Seasonal and trend-aware baseline adjustments

Baseline Types:
- Performance Metrics: Response time, throughput, latency percentiles
- Resource Utilization: CPU, memory, disk, network usage
- Scientific Accuracy: Computation precision, numerical stability
- Quality Metrics: Error rates, success rates, availability
- Optimization Metrics: Cache hit rates, JIT compilation overhead

Statistical Methods:
- Percentile-based baselines (P50, P95, P99)
- Moving averages with trend adjustment
- Seasonal decomposition for periodic workloads
- Anomaly-resistant baseline calculation
- Confidence interval estimation
- Statistical significance testing
"""

import json
import logging
import pickle
import statistics
import time
import warnings
from collections import defaultdict, deque
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import threading

import numpy as np

# Statistical analysis
try:
    from scipy import stats
    from scipy.signal import savgol_filter
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    warnings.warn("scipy not available - advanced statistical features limited")

# Time series analysis
try:
    from statsmodels.tsa.seasonal import seasonal_decompose
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    warnings.warn("statsmodels not available - time series analysis limited")

logger = logging.getLogger(__name__)


@dataclass
class BaselineMetric:
    """Individual baseline metric with statistical properties."""
    name: str
    value: float
    confidence_interval: Tuple[float, float]
    percentiles: Dict[str, float]  # P50, P90, P95, P99
    sample_count: int
    last_updated: float
    update_frequency: int  # seconds
    drift_threshold: float
    metadata: Dict[str, Any]


@dataclass
class BaselineVersion:
    """Versioned baseline snapshot."""
    version: str
    timestamp: float
    baselines: Dict[str, BaselineMetric]
    performance_summary: Dict[str, Any]
    environment_context: Dict[str, Any]
    validation_status: str  # 'valid', 'invalid', 'pending'
    notes: str


@dataclass
class BaselineDrift:
    """Detected baseline drift event."""
    drift_id: str
    metric_name: str
    baseline_value: float
    current_value: float
    drift_magnitude: float
    drift_direction: str  # 'increase', 'decrease'
    detection_time: float
    confidence: float
    significance: float
    suggested_action: str


class BaselineCalculator:
    """Statistical baseline calculation engine."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config

    def calculate_baseline(self, data: List[float],
                         method: str = 'percentile') -> BaselineMetric:
        """
        Calculate baseline metric from historical data.

        Parameters
        ----------
        data : List[float]
            Historical metric values
        method : str
            Calculation method ('percentile', 'mean', 'median', 'robust')

        Returns
        -------
        BaselineMetric
            Calculated baseline metric
        """
        if len(data) < 10:
            raise ValueError("Insufficient data for baseline calculation")

        data_array = np.array(data)

        # Remove outliers for robust calculation
        if method == 'robust':
            q1, q3 = np.percentile(data_array, [25, 75])
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            data_array = data_array[(data_array >= lower_bound) & (data_array <= upper_bound)]

        # Calculate baseline value
        if method == 'percentile':
            baseline_value = np.percentile(data_array, 50)  # Median
        elif method == 'mean':
            baseline_value = np.mean(data_array)
        elif method == 'median':
            baseline_value = np.median(data_array)
        elif method == 'robust':
            baseline_value = np.median(data_array)  # Robust median
        else:
            baseline_value = np.median(data_array)  # Default to median

        # Calculate confidence interval
        confidence_interval = self._calculate_confidence_interval(data_array)

        # Calculate percentiles
        percentiles = {
            'p50': float(np.percentile(data_array, 50)),
            'p90': float(np.percentile(data_array, 90)),
            'p95': float(np.percentile(data_array, 95)),
            'p99': float(np.percentile(data_array, 99))
        }

        # Calculate drift threshold (based on standard deviation)
        std_dev = np.std(data_array)
        drift_threshold = std_dev * 2.0  # 2 sigma threshold

        return BaselineMetric(
            name="",  # Will be set by caller
            value=float(baseline_value),
            confidence_interval=confidence_interval,
            percentiles=percentiles,
            sample_count=len(data_array),
            last_updated=time.time(),
            update_frequency=3600,  # Default 1 hour
            drift_threshold=drift_threshold,
            metadata={
                'method': method,
                'original_samples': len(data),
                'filtered_samples': len(data_array),
                'std_dev': float(std_dev),
                'data_range': (float(np.min(data_array)), float(np.max(data_array)))
            }
        )

    def _calculate_confidence_interval(self, data: np.ndarray,
                                     confidence: float = 0.95) -> Tuple[float, float]:
        """Calculate confidence interval for baseline."""
        if SCIPY_AVAILABLE and len(data) > 2:
            # Use t-distribution for small samples
            mean = np.mean(data)
            sem = stats.sem(data)  # Standard error of mean
            interval = stats.t.interval(confidence, len(data)-1, loc=mean, scale=sem)
            return (float(interval[0]), float(interval[1]))
        else:
            # Simple percentile-based interval
            alpha = (1 - confidence) / 2
            lower = np.percentile(data, alpha * 100)
            upper = np.percentile(data, (1 - alpha) * 100)
            return (float(lower), float(upper))

    def detect_trend(self, data: List[float]) -> Dict[str, Any]:
        """Detect trend in time series data."""
        if len(data) < 20:
            return {'trend': 'insufficient_data', 'strength': 0.0}

        data_array = np.array(data)

        # Simple linear trend detection
        x = np.arange(len(data_array))
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, data_array)

        # Classify trend
        if abs(r_value) < 0.3:
            trend = 'stable'
        elif slope > 0:
            trend = 'increasing'
        else:
            trend = 'decreasing'

        # Apply smoothing if available
        smoothed_data = data_array
        if SCIPY_AVAILABLE and len(data_array) > 10:
            try:
                window_length = min(11, len(data_array) // 2 * 2 + 1)  # Odd number
                smoothed_data = savgol_filter(data_array, window_length, 3)
            except Exception:
                pass  # Use original data

        return {
            'trend': trend,
            'strength': float(abs(r_value)),
            'slope': float(slope),
            'p_value': float(p_value),
            'significant': p_value < 0.05,
            'smoothed_data': smoothed_data.tolist()
        }

    def detect_seasonality(self, data: List[float], period: int = 24) -> Dict[str, Any]:
        """Detect seasonal patterns in data."""
        if not STATSMODELS_AVAILABLE or len(data) < period * 3:
            return {'seasonal': False, 'strength': 0.0}

        try:
            # Perform seasonal decomposition
            ts = pd.Series(data)
            decomposition = seasonal_decompose(ts, model='additive', period=period)

            # Calculate seasonal strength
            seasonal_var = np.var(decomposition.seasonal)
            residual_var = np.var(decomposition.resid.dropna())
            seasonal_strength = seasonal_var / (seasonal_var + residual_var)

            return {
                'seasonal': seasonal_strength > 0.3,
                'strength': float(seasonal_strength),
                'period': period,
                'seasonal_component': decomposition.seasonal.tolist(),
                'trend_component': decomposition.trend.dropna().tolist()
            }

        except Exception as e:
            logger.debug(f"Seasonality detection failed: {e}")
            return {'seasonal': False, 'strength': 0.0}


class BaselineManager:
    """
    Advanced baseline management system for performance monitoring.

    Provides comprehensive baseline tracking, validation, and drift detection
    specifically designed for scientific computing workloads.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize baseline manager."""
        self.config = config or self._default_config()
        self.baselines_dir = Path(self.config['baselines_dir'])
        self.baselines_dir.mkdir(exist_ok=True)

        # Baseline storage
        self.current_baselines: Dict[str, BaselineMetric] = {}
        self.baseline_versions: List[BaselineVersion] = []
        self.historical_data: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=self.config['max_history_points'])
        )

        # Drift detection
        self.detected_drifts: List[BaselineDrift] = []
        self.drift_callbacks: List[callable] = []

        # Background processing
        self.update_thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()
        self.lock = threading.RLock()

        # Calculator
        self.calculator = BaselineCalculator(self.config)

        # Load existing baselines
        self._load_baselines()

        logger.info("Baseline manager initialized")

    def _default_config(self) -> Dict[str, Any]:
        """Get default baseline manager configuration."""
        return {
            'baselines_dir': 'baselines',
            'max_history_points': 10000,
            'max_baseline_versions': 50,
            'default_update_frequency': 3600,  # 1 hour
            'drift_detection_sensitivity': 2.0,  # sigma multiplier
            'confidence_level': 0.95,
            'min_samples_for_baseline': 50,
            'baseline_validation_threshold': 0.8,
            'auto_update_enabled': True,
            'drift_notification_enabled': True,
            'baseline_methods': {
                'response_time': 'percentile',
                'memory_usage': 'robust',
                'accuracy': 'mean',
                'error_rate': 'percentile',
                'default': 'percentile'
            },
            'update_frequencies': {
                'high_frequency': 1800,  # 30 minutes
                'normal_frequency': 3600,  # 1 hour
                'low_frequency': 7200,   # 2 hours
                'daily': 86400          # 24 hours
            },
            'drift_thresholds': {
                'response_time': 0.2,  # 20% change
                'memory_usage': 0.3,   # 30% change
                'accuracy': 0.01,      # 1% change
                'error_rate': 0.1,     # 10% change
                'default': 0.2
            }
        }

    def _load_baselines(self):
        """Load existing baselines from disk."""
        current_file = self.baselines_dir / 'current_baselines.json'
        versions_file = self.baselines_dir / 'baseline_versions.json'

        # Load current baselines
        if current_file.exists():
            try:
                with open(current_file, 'r') as f:
                    data = json.load(f)
                    for name, baseline_data in data.items():
                        self.current_baselines[name] = BaselineMetric(**baseline_data)
                logger.info(f"Loaded {len(self.current_baselines)} current baselines")
            except Exception as e:
                logger.warning(f"Failed to load current baselines: {e}")

        # Load baseline versions
        if versions_file.exists():
            try:
                with open(versions_file, 'r') as f:
                    versions_data = json.load(f)
                    for version_data in versions_data:
                        # Reconstruct BaselineMetric objects
                        baselines = {}
                        for name, baseline_data in version_data['baselines'].items():
                            baselines[name] = BaselineMetric(**baseline_data)

                        version = BaselineVersion(
                            version=version_data['version'],
                            timestamp=version_data['timestamp'],
                            baselines=baselines,
                            performance_summary=version_data['performance_summary'],
                            environment_context=version_data['environment_context'],
                            validation_status=version_data['validation_status'],
                            notes=version_data['notes']
                        )
                        self.baseline_versions.append(version)

                logger.info(f"Loaded {len(self.baseline_versions)} baseline versions")
            except Exception as e:
                logger.warning(f"Failed to load baseline versions: {e}")

    def _save_baselines(self):
        """Save baselines to disk."""
        current_file = self.baselines_dir / 'current_baselines.json'
        versions_file = self.baselines_dir / 'baseline_versions.json'

        try:
            # Save current baselines
            with open(current_file, 'w') as f:
                data = {name: asdict(baseline) for name, baseline in self.current_baselines.items()}
                json.dump(data, f, indent=2)

            # Save baseline versions
            with open(versions_file, 'w') as f:
                versions_data = []
                for version in self.baseline_versions:
                    version_dict = asdict(version)
                    # Convert BaselineMetric objects to dicts
                    version_dict['baselines'] = {
                        name: asdict(baseline) for name, baseline in version.baselines.items()
                    }
                    versions_data.append(version_dict)
                json.dump(versions_data, f, indent=2)

            logger.debug("Baselines saved to disk")

        except Exception as e:
            logger.error(f"Failed to save baselines: {e}")

    def add_data_point(self, metric_name: str, value: float,
                      timestamp: Optional[float] = None):
        """
        Add a data point for baseline calculation.

        Parameters
        ----------
        metric_name : str
            Name of the metric
        value : float
            Metric value
        timestamp : float, optional
            Timestamp of the measurement
        """
        if timestamp is None:
            timestamp = time.time()

        with self.lock:
            self.historical_data[metric_name].append((timestamp, value))

        # Check if baseline should be updated
        if self._should_update_baseline(metric_name):
            self.update_baseline(metric_name)

    def _should_update_baseline(self, metric_name: str) -> bool:
        """Check if baseline should be updated for a metric."""
        if not self.config['auto_update_enabled']:
            return False

        # Check if we have enough data
        if len(self.historical_data[metric_name]) < self.config['min_samples_for_baseline']:
            return False

        # Check update frequency
        if metric_name in self.current_baselines:
            last_updated = self.current_baselines[metric_name].last_updated
            update_frequency = self.current_baselines[metric_name].update_frequency
            if time.time() - last_updated < update_frequency:
                return False

        return True

    def update_baseline(self, metric_name: str,
                       method: Optional[str] = None) -> bool:
        """
        Update baseline for a specific metric.

        Parameters
        ----------
        metric_name : str
            Name of the metric to update
        method : str, optional
            Calculation method to use

        Returns
        -------
        bool
            True if baseline was updated successfully
        """
        with self.lock:
            if metric_name not in self.historical_data:
                logger.warning(f"No historical data for metric: {metric_name}")
                return False

            # Get historical values
            data_points = list(self.historical_data[metric_name])
            if len(data_points) < self.config['min_samples_for_baseline']:
                logger.warning(f"Insufficient data for baseline update: {metric_name}")
                return False

            # Extract values (ignore timestamps for now)
            values = [point[1] for point in data_points]

            # Determine calculation method
            if method is None:
                method = self._get_method_for_metric(metric_name)

            try:
                # Calculate new baseline
                new_baseline = self.calculator.calculate_baseline(values, method)
                new_baseline.name = metric_name

                # Set update frequency based on metric type
                new_baseline.update_frequency = self._get_update_frequency(metric_name)

                # Detect drift if we have an existing baseline
                if metric_name in self.current_baselines:
                    drift = self._detect_drift(metric_name, new_baseline)
                    if drift:
                        self._handle_drift(drift)

                # Update baseline
                self.current_baselines[metric_name] = new_baseline

                # Save to disk
                self._save_baselines()

                logger.info(f"Baseline updated for {metric_name}: {new_baseline.value:.3f}")
                return True

            except Exception as e:
                logger.error(f"Failed to update baseline for {metric_name}: {e}")
                return False

    def _get_method_for_metric(self, metric_name: str) -> str:
        """Get appropriate calculation method for a metric."""
        methods = self.config['baseline_methods']

        # Check for exact match
        if metric_name in methods:
            return methods[metric_name]

        # Check for pattern matches
        metric_lower = metric_name.lower()
        for pattern, method in methods.items():
            if pattern in metric_lower:
                return method

        return methods.get('default', 'percentile')

    def _get_update_frequency(self, metric_name: str) -> int:
        """Get update frequency for a metric."""
        frequencies = self.config['update_frequencies']

        # Categorize metrics by update frequency needs
        metric_lower = metric_name.lower()

        if any(term in metric_lower for term in ['error', 'accuracy', 'security']):
            return frequencies['high_frequency']  # Critical metrics
        elif any(term in metric_lower for term in ['response_time', 'latency']):
            return frequencies['normal_frequency']  # Performance metrics
        elif any(term in metric_lower for term in ['memory', 'cpu', 'disk']):
            return frequencies['low_frequency']  # Resource metrics
        else:
            return frequencies['daily']  # Other metrics

    def _detect_drift(self, metric_name: str, new_baseline: BaselineMetric) -> Optional[BaselineDrift]:
        """Detect drift between current and new baseline."""
        if metric_name not in self.current_baselines:
            return None

        current_baseline = self.current_baselines[metric_name]
        current_value = current_baseline.value
        new_value = new_baseline.value

        # Calculate drift magnitude
        if current_value != 0:
            drift_magnitude = abs(new_value - current_value) / abs(current_value)
        else:
            drift_magnitude = abs(new_value - current_value)

        # Check against threshold
        threshold = self._get_drift_threshold(metric_name)
        if drift_magnitude < threshold:
            return None

        # Calculate statistical significance
        confidence = self._calculate_drift_confidence(current_baseline, new_baseline)
        if confidence < 0.8:  # Require high confidence
            return None

        # Determine drift direction
        if new_value > current_value:
            direction = 'increase'
        else:
            direction = 'decrease'

        # Generate suggested action
        suggested_action = self._generate_drift_action(metric_name, drift_magnitude, direction)

        drift = BaselineDrift(
            drift_id=f"drift_{metric_name}_{int(time.time())}",
            metric_name=metric_name,
            baseline_value=current_value,
            current_value=new_value,
            drift_magnitude=drift_magnitude,
            drift_direction=direction,
            detection_time=time.time(),
            confidence=confidence,
            significance=drift_magnitude / threshold,
            suggested_action=suggested_action
        )

        return drift

    def _get_drift_threshold(self, metric_name: str) -> float:
        """Get drift detection threshold for a metric."""
        thresholds = self.config['drift_thresholds']

        # Check for exact match
        if metric_name in thresholds:
            return thresholds[metric_name]

        # Check for pattern matches
        metric_lower = metric_name.lower()
        for pattern, threshold in thresholds.items():
            if pattern in metric_lower:
                return threshold

        return thresholds.get('default', 0.2)

    def _calculate_drift_confidence(self, current: BaselineMetric,
                                  new: BaselineMetric) -> float:
        """Calculate confidence in drift detection."""
        # Simple confidence based on sample counts and overlap
        min_samples = min(current.sample_count, new.sample_count)
        max_samples = max(current.sample_count, new.sample_count)

        # Sample size confidence
        sample_confidence = min(1.0, min_samples / 100.0)

        # Confidence interval overlap
        current_ci = current.confidence_interval
        new_ci = new.confidence_interval

        overlap = max(0, min(current_ci[1], new_ci[1]) - max(current_ci[0], new_ci[0]))
        total_range = max(current_ci[1], new_ci[1]) - min(current_ci[0], new_ci[0])

        if total_range > 0:
            overlap_confidence = 1.0 - (overlap / total_range)
        else:
            overlap_confidence = 0.5

        # Combined confidence
        return (sample_confidence + overlap_confidence) / 2.0

    def _generate_drift_action(self, metric_name: str, magnitude: float,
                             direction: str) -> str:
        """Generate suggested action for drift."""
        metric_lower = metric_name.lower()

        if 'response_time' in metric_lower or 'latency' in metric_lower:
            if direction == 'increase':
                return "Investigate performance degradation and consider optimization"
            else:
                return "Validate performance improvements and update expectations"
        elif 'memory' in metric_lower:
            if direction == 'increase':
                return "Check for memory leaks or increased workload"
            else:
                return "Verify memory optimizations are working correctly"
        elif 'accuracy' in metric_lower:
            return "Critical: Investigate accuracy changes and validate algorithms"
        elif 'error' in metric_lower:
            if direction == 'increase':
                return "Urgent: Investigate error rate increase"
            else:
                return "Monitor continued error rate improvement"
        else:
            return f"Monitor {metric_name} for continued {direction} trend"

    def _handle_drift(self, drift: BaselineDrift):
        """Handle detected drift."""
        self.detected_drifts.append(drift)

        # Keep only recent drifts
        current_time = time.time()
        self.detected_drifts = [
            d for d in self.detected_drifts
            if current_time - d.detection_time < 86400 * 7  # Last week
        ]

        logger.warning(f"Baseline drift detected: {drift.metric_name} "
                      f"({drift.drift_direction} {drift.drift_magnitude:.1%})")

        # Notify callbacks
        for callback in self.drift_callbacks:
            try:
                callback(drift)
            except Exception as e:
                logger.error(f"Drift callback failed: {e}")

    def create_baseline_version(self, version_name: str,
                              notes: str = "") -> BaselineVersion:
        """
        Create a versioned snapshot of current baselines.

        Parameters
        ----------
        version_name : str
            Version identifier
        notes : str
            Version notes

        Returns
        -------
        BaselineVersion
            Created baseline version
        """
        with self.lock:
            # Copy current baselines
            baselines_copy = {
                name: BaselineMetric(
                    name=baseline.name,
                    value=baseline.value,
                    confidence_interval=baseline.confidence_interval,
                    percentiles=baseline.percentiles.copy(),
                    sample_count=baseline.sample_count,
                    last_updated=baseline.last_updated,
                    update_frequency=baseline.update_frequency,
                    drift_threshold=baseline.drift_threshold,
                    metadata=baseline.metadata.copy()
                )
                for name, baseline in self.current_baselines.items()
            }

            # Calculate performance summary
            performance_summary = self._calculate_performance_summary()

            # Get environment context
            environment_context = self._get_environment_context()

            # Create version
            version = BaselineVersion(
                version=version_name,
                timestamp=time.time(),
                baselines=baselines_copy,
                performance_summary=performance_summary,
                environment_context=environment_context,
                validation_status='valid',
                notes=notes
            )

            # Add to versions list
            self.baseline_versions.append(version)

            # Keep only recent versions
            max_versions = self.config['max_baseline_versions']
            if len(self.baseline_versions) > max_versions:
                self.baseline_versions = self.baseline_versions[-max_versions:]

            # Save to disk
            self._save_baselines()

            logger.info(f"Created baseline version: {version_name}")
            return version

    def _calculate_performance_summary(self) -> Dict[str, Any]:
        """Calculate performance summary for current baselines."""
        if not self.current_baselines:
            return {}

        summary = {
            'total_metrics': len(self.current_baselines),
            'response_time_p95': None,
            'memory_usage_avg': None,
            'error_rate_avg': None,
            'accuracy_avg': None
        }

        # Extract key metrics
        for name, baseline in self.current_baselines.items():
            name_lower = name.lower()
            if 'response_time' in name_lower and summary['response_time_p95'] is None:
                summary['response_time_p95'] = baseline.percentiles.get('p95', baseline.value)
            elif 'memory' in name_lower and summary['memory_usage_avg'] is None:
                summary['memory_usage_avg'] = baseline.value
            elif 'error' in name_lower and summary['error_rate_avg'] is None:
                summary['error_rate_avg'] = baseline.value
            elif 'accuracy' in name_lower and summary['accuracy_avg'] is None:
                summary['accuracy_avg'] = baseline.value

        return summary

    def _get_environment_context(self) -> Dict[str, Any]:
        """Get current environment context."""
        context = {
            'timestamp': time.time(),
            'system_info': {}
        }

        # Add system information if available
        try:
            import psutil
            context['system_info'] = {
                'cpu_count': psutil.cpu_count(),
                'memory_total': psutil.virtual_memory().total,
                'python_version': __import__('sys').version
            }
        except Exception:
            pass

        return context

    def rollback_to_version(self, version_name: str) -> bool:
        """
        Rollback baselines to a specific version.

        Parameters
        ----------
        version_name : str
            Version to rollback to

        Returns
        -------
        bool
            True if rollback successful
        """
        # Find the version
        target_version = None
        for version in self.baseline_versions:
            if version.version == version_name:
                target_version = version
                break

        if target_version is None:
            logger.error(f"Version not found: {version_name}")
            return False

        try:
            with self.lock:
                # Replace current baselines
                self.current_baselines = target_version.baselines.copy()

                # Save to disk
                self._save_baselines()

                logger.info(f"Rolled back to baseline version: {version_name}")
                return True

        except Exception as e:
            logger.error(f"Rollback failed: {e}")
            return False

    def validate_baseline(self, metric_name: str) -> Dict[str, Any]:
        """
        Validate a baseline against recent data.

        Parameters
        ----------
        metric_name : str
            Metric to validate

        Returns
        -------
        Dict[str, Any]
            Validation results
        """
        if metric_name not in self.current_baselines:
            return {'valid': False, 'reason': 'Baseline not found'}

        if metric_name not in self.historical_data:
            return {'valid': False, 'reason': 'No historical data'}

        baseline = self.current_baselines[metric_name]
        recent_data = list(self.historical_data[metric_name])[-100:]  # Last 100 points

        if len(recent_data) < 10:
            return {'valid': False, 'reason': 'Insufficient recent data'}

        # Extract recent values
        recent_values = [point[1] for point in recent_data]

        # Calculate validation metrics
        recent_mean = np.mean(recent_values)
        recent_std = np.std(recent_values)

        # Check if recent data falls within baseline confidence interval
        ci = baseline.confidence_interval
        within_ci_count = sum(1 for value in recent_values
                             if ci[0] <= value <= ci[1])
        within_ci_ratio = within_ci_count / len(recent_values)

        # Validation criteria
        valid = within_ci_ratio >= self.config['baseline_validation_threshold']

        validation_result = {
            'valid': valid,
            'within_ci_ratio': within_ci_ratio,
            'threshold': self.config['baseline_validation_threshold'],
            'recent_mean': recent_mean,
            'baseline_value': baseline.value,
            'recent_std': recent_std,
            'baseline_ci': ci,
            'sample_count': len(recent_values),
            'reason': 'Valid' if valid else 'Recent data outside baseline range'
        }

        return validation_result

    def get_baseline_comparison(self, metric_name: str,
                              time_range: int = 86400) -> Dict[str, Any]:
        """
        Compare current baseline with historical performance.

        Parameters
        ----------
        metric_name : str
            Metric to compare
        time_range : int
            Time range in seconds for comparison

        Returns
        -------
        Dict[str, Any]
            Comparison results
        """
        if metric_name not in self.current_baselines:
            return {'error': 'Baseline not found'}

        if metric_name not in self.historical_data:
            return {'error': 'No historical data'}

        current_time = time.time()
        cutoff_time = current_time - time_range

        # Get data within time range
        time_range_data = [
            point for point in self.historical_data[metric_name]
            if point[0] >= cutoff_time
        ]

        if len(time_range_data) < 10:
            return {'error': 'Insufficient data in time range'}

        values = [point[1] for point in time_range_data]
        baseline = self.current_baselines[metric_name]

        # Calculate comparison metrics
        comparison = {
            'baseline_value': baseline.value,
            'current_mean': np.mean(values),
            'current_median': np.median(values),
            'current_std': np.std(values),
            'current_min': np.min(values),
            'current_max': np.max(values),
            'sample_count': len(values),
            'time_range_hours': time_range / 3600
        }

        # Calculate performance vs baseline
        if baseline.value != 0:
            comparison['performance_vs_baseline'] = (comparison['current_mean'] - baseline.value) / baseline.value
        else:
            comparison['performance_vs_baseline'] = 0.0

        # Determine status
        if abs(comparison['performance_vs_baseline']) < 0.05:  # 5% threshold
            comparison['status'] = 'stable'
        elif comparison['performance_vs_baseline'] > 0:
            comparison['status'] = 'degraded' if 'response_time' in metric_name.lower() else 'improved'
        else:
            comparison['status'] = 'improved' if 'response_time' in metric_name.lower() else 'degraded'

        return comparison

    def add_drift_callback(self, callback: callable):
        """Add callback for drift notifications."""
        self.drift_callbacks.append(callback)

    def get_drift_history(self, days: int = 7) -> List[BaselineDrift]:
        """Get drift history for the specified number of days."""
        cutoff_time = time.time() - (days * 86400)
        return [drift for drift in self.detected_drifts if drift.detection_time >= cutoff_time]

    def get_baseline_status(self) -> Dict[str, Any]:
        """Get overall baseline system status."""
        current_time = time.time()

        # Count baselines by status
        outdated_baselines = 0
        valid_baselines = 0

        for name, baseline in self.current_baselines.items():
            age = current_time - baseline.last_updated
            if age > baseline.update_frequency * 2:  # 2x update frequency
                outdated_baselines += 1
            else:
                valid_baselines += 1

        # Recent drift count
        recent_drifts = len([d for d in self.detected_drifts
                           if current_time - d.detection_time < 86400])

        return {
            'total_baselines': len(self.current_baselines),
            'valid_baselines': valid_baselines,
            'outdated_baselines': outdated_baselines,
            'baseline_versions': len(self.baseline_versions),
            'recent_drifts': recent_drifts,
            'auto_update_enabled': self.config['auto_update_enabled'],
            'total_metrics_tracked': len(self.historical_data),
            'health_status': 'healthy' if recent_drifts == 0 and outdated_baselines == 0 else 'degraded'
        }

    def cleanup(self):
        """Clean up baseline manager resources."""
        # Stop background processing
        self.stop_event.set()
        if self.update_thread:
            self.update_thread.join(timeout=5)

        # Save final state
        self._save_baselines()

        logger.info("Baseline manager cleanup completed")


if __name__ == "__main__":
    # Example usage and testing
    import random
    import time

    # Create baseline manager
    manager = BaselineManager()

    # Simulate adding data points
    print("Simulating performance data...")
    for i in range(200):
        # Simulate response time with trend
        base_time = 1.0 + 0.001 * i  # Slight degradation trend
        response_time = base_time + random.normal(0, 0.1)
        manager.add_data_point('response_time', response_time)

        # Simulate memory usage
        memory_usage = 100 + random.normal(0, 10)
        manager.add_data_point('memory_usage', memory_usage)

        # Simulate accuracy
        accuracy = 0.999 + random.normal(0, 0.001)
        manager.add_data_point('accuracy', max(0, min(1, accuracy)))

    # Force baseline updates
    print("\nUpdating baselines...")
    for metric in ['response_time', 'memory_usage', 'accuracy']:
        success = manager.update_baseline(metric)
        print(f"Baseline update for {metric}: {'Success' if success else 'Failed'}")

    # Create a baseline version
    print("\nCreating baseline version...")
    version = manager.create_baseline_version('v1.0', 'Initial baseline')

    # Check baseline status
    print("\nBaseline status:")
    status = manager.get_baseline_status()
    for key, value in status.items():
        print(f"  {key}: {value}")

    # Validate baselines
    print("\nValidating baselines...")
    for metric in ['response_time', 'memory_usage', 'accuracy']:
        validation = manager.validate_baseline(metric)
        print(f"  {metric}: {'Valid' if validation['valid'] else 'Invalid'}")

    # Simulate some drift
    print("\nSimulating performance drift...")
    for i in range(50):
        # Introduce significant response time increase
        response_time = 2.0 + random.normal(0, 0.1)
        manager.add_data_point('response_time', response_time)

    # Force update to detect drift
    manager.update_baseline('response_time')

    # Check for drifts
    drifts = manager.get_drift_history()
    print(f"\nDetected drifts: {len(drifts)}")
    for drift in drifts:
        print(f"  {drift.metric_name}: {drift.drift_direction} {drift.drift_magnitude:.1%}")

    # Cleanup
    manager.cleanup()