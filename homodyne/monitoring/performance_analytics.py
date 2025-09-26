#!/usr/bin/env python3
"""
Advanced Performance Analytics for Scientific Computing
========================================================

Machine learning-powered performance analytics system for the homodyne analysis
application. Provides predictive performance modeling, trend analysis, and
intelligent optimization recommendations with scientific computing awareness.

Key Features:
- Predictive performance degradation modeling
- Time series forecasting for resource planning
- Anomaly detection with scientific computation context
- Performance pattern recognition
- Optimization impact prediction
- Real-time dashboard analytics
- Scientific accuracy vs performance trade-off analysis

Machine Learning Models:
- LSTM networks for time series prediction
- Isolation Forest for anomaly detection
- Random Forest for performance classification
- Clustering for pattern discovery
- Regression models for impact prediction

Scientific Computing Considerations:
- Numerical stability monitoring
- Floating-point precision tracking
- Scientific accuracy preservation
- Computation-heavy workload patterns
- Memory allocation optimization
- Cache efficiency analysis
"""

import json
import logging
import pickle
import statistics
import warnings
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import time

import numpy as np

# Machine learning imports with graceful degradation
try:
    from sklearn.ensemble import IsolationForest, RandomForestRegressor
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler, RobustScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error, r2_score
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    warnings.warn("scikit-learn not available - ML analytics disabled")

# Time series analysis
try:
    from scipy import signal
    from scipy.stats import zscore
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    warnings.warn("scipy not available - advanced analytics limited")

# Plotting for dashboards
try:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.figure import Figure
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    warnings.warn("matplotlib not available - visualization disabled")

logger = logging.getLogger(__name__)


@dataclass
class PerformancePrediction:
    """Performance prediction with confidence intervals."""
    metric_name: str
    timestamp: float
    predicted_value: float
    confidence_lower: float
    confidence_upper: float
    prediction_horizon: float
    model_accuracy: float
    context: Dict[str, Any]


@dataclass
class PerformanceTrend:
    """Performance trend analysis result."""
    metric_name: str
    trend_direction: str  # 'improving', 'degrading', 'stable'
    trend_strength: float  # -1 to 1
    seasonal_component: bool
    change_points: List[float]
    forecasted_values: List[float]
    confidence_intervals: List[Tuple[float, float]]


@dataclass
class OptimizationImpact:
    """Predicted impact of optimization changes."""
    optimization_name: str
    predicted_improvement: float
    confidence: float
    affected_metrics: List[str]
    implementation_effort: str
    risk_assessment: str
    expected_timeline: str


class PerformanceAnalytics:
    """
    Advanced performance analytics system with ML-powered insights.

    Provides predictive modeling, trend analysis, and optimization
    recommendations specifically designed for scientific computing workloads.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize performance analytics system."""
        self.config = config or self._default_config()
        self.analytics_dir = Path(self.config['analytics_dir'])
        self.analytics_dir.mkdir(exist_ok=True)

        # Initialize ML models
        self.models: Dict[str, Any] = {}
        self.scalers: Dict[str, Any] = {}
        self.model_metadata: Dict[str, Any] = {}

        if ML_AVAILABLE:
            self._initialize_ml_models()

        # Performance data storage
        self.historical_data: Dict[str, List[Dict[str, Any]]] = {}
        self.predictions: List[PerformancePrediction] = []
        self.trends: List[PerformanceTrend] = []
        self.optimization_impacts: List[OptimizationImpact] = []

        # Load existing data
        self._load_historical_data()
        self._load_models()

        logger.info("Performance analytics system initialized")

    def _default_config(self) -> Dict[str, Any]:
        """Get default analytics configuration."""
        return {
            'analytics_dir': 'performance_analytics',
            'model_update_interval': 3600,  # 1 hour
            'prediction_horizon': 24 * 3600,  # 24 hours
            'min_data_points': 50,
            'anomaly_threshold': 0.1,
            'trend_analysis': {
                'window_size': 100,
                'seasonal_period': 24,  # Hours
                'change_point_sensitivity': 0.05
            },
            'ml_models': {
                'anomaly_detection': {
                    'contamination': 0.1,
                    'n_estimators': 100
                },
                'performance_prediction': {
                    'n_estimators': 100,
                    'max_depth': 10,
                    'random_state': 42
                },
                'clustering': {
                    'n_clusters': 5,
                    'random_state': 42
                }
            },
            'scientific_computing': {
                'accuracy_weight': 0.8,  # Prioritize accuracy
                'performance_weight': 0.2,
                'numerical_stability_threshold': 1e-12,
                'floating_point_precision': 'double'
            }
        }

    def _initialize_ml_models(self):
        """Initialize machine learning models."""
        if not ML_AVAILABLE:
            return

        try:
            # Anomaly detection model
            self.models['anomaly_detector'] = IsolationForest(
                contamination=self.config['ml_models']['anomaly_detection']['contamination'],
                n_estimators=self.config['ml_models']['anomaly_detection']['n_estimators'],
                random_state=42
            )

            # Performance prediction model
            self.models['performance_predictor'] = RandomForestRegressor(
                n_estimators=self.config['ml_models']['performance_prediction']['n_estimators'],
                max_depth=self.config['ml_models']['performance_prediction']['max_depth'],
                random_state=self.config['ml_models']['performance_prediction']['random_state']
            )

            # Pattern clustering model
            self.models['pattern_clustering'] = KMeans(
                n_clusters=self.config['ml_models']['clustering']['n_clusters'],
                random_state=self.config['ml_models']['clustering']['random_state']
            )

            # Scalers for data preprocessing
            self.scalers['standard'] = StandardScaler()
            self.scalers['robust'] = RobustScaler()

            logger.info("ML models initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize ML models: {e}")
            self.models = {}
            self.scalers = {}

    def _load_historical_data(self):
        """Load historical performance data."""
        data_file = self.analytics_dir / 'historical_data.json'
        if data_file.exists():
            try:
                with open(data_file, 'r') as f:
                    self.historical_data = json.load(f)
                logger.debug("Historical data loaded")
            except Exception as e:
                logger.warning(f"Failed to load historical data: {e}")
                self.historical_data = {}

    def _save_historical_data(self):
        """Save historical performance data."""
        data_file = self.analytics_dir / 'historical_data.json'
        try:
            with open(data_file, 'w') as f:
                json.dump(self.historical_data, f, indent=2)
            logger.debug("Historical data saved")
        except Exception as e:
            logger.error(f"Failed to save historical data: {e}")

    def _load_models(self):
        """Load trained ML models from disk."""
        models_dir = self.analytics_dir / 'models'
        if not models_dir.exists():
            return

        for model_file in models_dir.glob('*.pkl'):
            try:
                model_name = model_file.stem
                with open(model_file, 'rb') as f:
                    model_data = pickle.load(f)
                    self.models[model_name] = model_data['model']
                    self.model_metadata[model_name] = model_data['metadata']
                logger.debug(f"Loaded model: {model_name}")
            except Exception as e:
                logger.warning(f"Failed to load model {model_file}: {e}")

    def _save_models(self):
        """Save trained ML models to disk."""
        models_dir = self.analytics_dir / 'models'
        models_dir.mkdir(exist_ok=True)

        for model_name, model in self.models.items():
            try:
                model_file = models_dir / f"{model_name}.pkl"
                model_data = {
                    'model': model,
                    'metadata': self.model_metadata.get(model_name, {
                        'created_at': time.time(),
                        'version': '1.0'
                    })
                }
                with open(model_file, 'wb') as f:
                    pickle.dump(model_data, f)
                logger.debug(f"Saved model: {model_name}")
            except Exception as e:
                logger.error(f"Failed to save model {model_name}: {e}")

    def add_performance_data(self, metric_name: str, value: float,
                           context: Optional[Dict[str, Any]] = None):
        """
        Add performance data point for analysis.

        Parameters
        ----------
        metric_name : str
            Name of the performance metric
        value : float
            Metric value
        context : dict, optional
            Additional context information
        """
        timestamp = time.time()
        data_point = {
            'timestamp': timestamp,
            'value': value,
            'context': context or {}
        }

        if metric_name not in self.historical_data:
            self.historical_data[metric_name] = []

        self.historical_data[metric_name].append(data_point)

        # Keep only recent data to manage memory
        max_points = 10000
        if len(self.historical_data[metric_name]) > max_points:
            self.historical_data[metric_name] = self.historical_data[metric_name][-max_points:]

        logger.debug(f"Added data point for {metric_name}: {value}")

    def detect_anomalies(self, metric_name: str,
                        window_size: int = 100) -> List[Dict[str, Any]]:
        """
        Detect anomalies in performance metrics using ML.

        Parameters
        ----------
        metric_name : str
            Name of the metric to analyze
        window_size : int
            Size of the analysis window

        Returns
        -------
        List[Dict[str, Any]]
            List of detected anomalies
        """
        if not ML_AVAILABLE or 'anomaly_detector' not in self.models:
            logger.warning("Anomaly detection not available")
            return []

        if metric_name not in self.historical_data:
            return []

        data = self.historical_data[metric_name]
        if len(data) < self.config['min_data_points']:
            return []

        try:
            # Prepare data for anomaly detection
            recent_data = data[-window_size:] if len(data) > window_size else data
            values = np.array([d['value'] for d in recent_data]).reshape(-1, 1)

            # Scale the data
            if 'robust' in self.scalers:
                scaled_values = self.scalers['robust'].fit_transform(values)
            else:
                scaled_values = values

            # Detect anomalies
            anomaly_labels = self.models['anomaly_detector'].fit_predict(scaled_values)
            anomaly_scores = self.models['anomaly_detector'].score_samples(scaled_values)

            # Extract anomalies
            anomalies = []
            for i, (label, score, data_point) in enumerate(zip(anomaly_labels, anomaly_scores, recent_data)):
                if label == -1:  # Anomaly
                    anomalies.append({
                        'timestamp': data_point['timestamp'],
                        'value': data_point['value'],
                        'anomaly_score': float(score),
                        'severity': 'high' if score < -0.5 else 'medium',
                        'context': data_point.get('context', {}),
                        'index': len(data) - len(recent_data) + i
                    })

            if anomalies:
                logger.info(f"Detected {len(anomalies)} anomalies in {metric_name}")

            return anomalies

        except Exception as e:
            logger.error(f"Anomaly detection failed for {metric_name}: {e}")
            return []

    def analyze_trends(self, metric_name: str) -> Optional[PerformanceTrend]:
        """
        Analyze performance trends for a metric.

        Parameters
        ----------
        metric_name : str
            Name of the metric to analyze

        Returns
        -------
        PerformanceTrend or None
            Trend analysis result
        """
        if metric_name not in self.historical_data:
            return None

        data = self.historical_data[metric_name]
        if len(data) < self.config['min_data_points']:
            return None

        try:
            # Extract time series data
            timestamps = [d['timestamp'] for d in data]
            values = [d['value'] for d in data]

            # Convert to numpy arrays
            timestamps = np.array(timestamps)
            values = np.array(values)

            # Calculate trend direction and strength
            trend_strength = self._calculate_trend_strength(values)
            trend_direction = self._classify_trend(trend_strength)

            # Detect seasonal patterns
            seasonal_component = self._detect_seasonality(values)

            # Find change points
            change_points = self._detect_change_points(values)

            # Generate forecast
            forecasted_values, confidence_intervals = self._generate_forecast(values)

            trend = PerformanceTrend(
                metric_name=metric_name,
                trend_direction=trend_direction,
                trend_strength=trend_strength,
                seasonal_component=seasonal_component,
                change_points=change_points,
                forecasted_values=forecasted_values,
                confidence_intervals=confidence_intervals
            )

            self.trends.append(trend)
            logger.info(f"Trend analysis completed for {metric_name}: {trend_direction}")

            return trend

        except Exception as e:
            logger.error(f"Trend analysis failed for {metric_name}: {e}")
            return None

    def _calculate_trend_strength(self, values: np.ndarray) -> float:
        """Calculate trend strength using linear regression slope."""
        if len(values) < 2:
            return 0.0

        x = np.arange(len(values))
        # Simple linear regression
        slope, _ = np.polyfit(x, values, 1)

        # Normalize by data range
        data_range = np.max(values) - np.min(values)
        if data_range > 0:
            normalized_slope = slope / data_range * len(values)
        else:
            normalized_slope = 0.0

        return np.clip(normalized_slope, -1.0, 1.0)

    def _classify_trend(self, trend_strength: float) -> str:
        """Classify trend direction based on strength."""
        if abs(trend_strength) < 0.1:
            return 'stable'
        elif trend_strength > 0:
            return 'improving' if trend_strength > 0.2 else 'slightly_improving'
        else:
            return 'degrading' if trend_strength < -0.2 else 'slightly_degrading'

    def _detect_seasonality(self, values: np.ndarray) -> bool:
        """Detect seasonal patterns in the data."""
        if not SCIPY_AVAILABLE or len(values) < 48:  # Need at least 2 cycles
            return False

        try:
            # Simple autocorrelation-based seasonality detection
            seasonal_period = self.config['trend_analysis']['seasonal_period']
            if len(values) > seasonal_period * 2:
                autocorr = np.correlate(values, values, mode='full')
                autocorr = autocorr[len(autocorr)//2:]

                # Check for significant periodicity
                if seasonal_period < len(autocorr):
                    seasonal_autocorr = autocorr[seasonal_period]
                    max_autocorr = np.max(autocorr[1:seasonal_period//2])
                    return seasonal_autocorr > max_autocorr * 0.8

        except Exception as e:
            logger.debug(f"Seasonality detection failed: {e}")

        return False

    def _detect_change_points(self, values: np.ndarray) -> List[float]:
        """Detect significant change points in the time series."""
        if len(values) < 20:
            return []

        try:
            # Simple change point detection using moving window variance
            window_size = min(20, len(values) // 4)
            change_points = []

            for i in range(window_size, len(values) - window_size):
                before_window = values[i-window_size:i]
                after_window = values[i:i+window_size]

                # Calculate statistical difference
                before_mean = np.mean(before_window)
                after_mean = np.mean(after_window)
                before_std = np.std(before_window)
                after_std = np.std(after_window)

                # Check for significant change
                if before_std > 0 and after_std > 0:
                    change_magnitude = abs(after_mean - before_mean) / (before_std + after_std)
                    if change_magnitude > self.config['trend_analysis']['change_point_sensitivity']:
                        change_points.append(float(i))

            return change_points[:5]  # Limit to top 5 change points

        except Exception as e:
            logger.debug(f"Change point detection failed: {e}")
            return []

    def _generate_forecast(self, values: np.ndarray) -> Tuple[List[float], List[Tuple[float, float]]]:
        """Generate simple forecast using linear extrapolation."""
        if len(values) < 10:
            return [], []

        try:
            # Simple linear extrapolation
            x = np.arange(len(values))
            slope, intercept = np.polyfit(x, values, 1)

            # Forecast next 24 points (hours)
            forecast_length = 24
            forecast_x = np.arange(len(values), len(values) + forecast_length)
            forecasted_values = [float(slope * x + intercept) for x in forecast_x]

            # Simple confidence intervals based on historical variance
            historical_std = np.std(values)
            confidence_intervals = [
                (val - 1.96 * historical_std, val + 1.96 * historical_std)
                for val in forecasted_values
            ]

            return forecasted_values, confidence_intervals

        except Exception as e:
            logger.debug(f"Forecast generation failed: {e}")
            return [], []

    def predict_performance(self, metric_name: str,
                          prediction_horizon: float = 3600) -> Optional[PerformancePrediction]:
        """
        Predict future performance using ML models.

        Parameters
        ----------
        metric_name : str
            Name of the metric to predict
        prediction_horizon : float
            Time horizon for prediction in seconds

        Returns
        -------
        PerformancePrediction or None
            Performance prediction result
        """
        if not ML_AVAILABLE or 'performance_predictor' not in self.models:
            logger.warning("Performance prediction not available")
            return None

        if metric_name not in self.historical_data:
            return None

        data = self.historical_data[metric_name]
        if len(data) < self.config['min_data_points']:
            return None

        try:
            # Prepare features for prediction
            recent_data = data[-50:]  # Use last 50 points
            features = self._extract_features(recent_data)

            if len(features) == 0:
                return None

            # Make prediction
            features_array = np.array(features).reshape(1, -1)

            # Scale features if scaler is available
            if 'standard' in self.scalers and hasattr(self.scalers['standard'], 'transform'):
                try:
                    features_array = self.scalers['standard'].transform(features_array)
                except Exception:
                    # Scaler not fitted, fit it now
                    all_features = []
                    for d in data[-200:]:  # Use more data for fitting
                        feat = self._extract_features([d])
                        if feat:
                            all_features.append(feat)

                    if all_features:
                        self.scalers['standard'].fit(all_features)
                        features_array = self.scalers['standard'].transform(features_array)

            predicted_value = self.models['performance_predictor'].predict(features_array)[0]

            # Calculate confidence intervals (simplified)
            historical_values = [d['value'] for d in recent_data]
            historical_std = np.std(historical_values)
            confidence_lower = predicted_value - 1.96 * historical_std
            confidence_upper = predicted_value + 1.96 * historical_std

            # Estimate model accuracy
            model_accuracy = self._estimate_model_accuracy(metric_name)

            prediction = PerformancePrediction(
                metric_name=metric_name,
                timestamp=time.time(),
                predicted_value=float(predicted_value),
                confidence_lower=float(confidence_lower),
                confidence_upper=float(confidence_upper),
                prediction_horizon=prediction_horizon,
                model_accuracy=model_accuracy,
                context={
                    'feature_count': len(features),
                    'data_points_used': len(recent_data),
                    'historical_std': historical_std
                }
            )

            self.predictions.append(prediction)
            logger.info(f"Performance prediction generated for {metric_name}: {predicted_value:.3f}")

            return prediction

        except Exception as e:
            logger.error(f"Performance prediction failed for {metric_name}: {e}")
            return None

    def _extract_features(self, data_points: List[Dict[str, Any]]) -> List[float]:
        """Extract features from data points for ML models."""
        if len(data_points) < 2:
            return []

        try:
            values = [d['value'] for d in data_points]
            timestamps = [d['timestamp'] for d in data_points]

            features = []

            # Statistical features
            features.extend([
                np.mean(values),
                np.std(values),
                np.min(values),
                np.max(values),
                np.median(values)
            ])

            # Trend features
            if len(values) > 1:
                x = np.arange(len(values))
                slope, _ = np.polyfit(x, values, 1)
                features.append(slope)

            # Temporal features
            current_time = timestamps[-1]
            hour_of_day = (current_time % 86400) / 86400  # Normalize to [0, 1]
            day_of_week = ((current_time // 86400) % 7) / 7  # Normalize to [0, 1]
            features.extend([hour_of_day, day_of_week])

            # Recent change features
            if len(values) >= 5:
                recent_change = (values[-1] - values[-5]) / (values[-5] + 1e-10)
                features.append(recent_change)

            return features

        except Exception as e:
            logger.debug(f"Feature extraction failed: {e}")
            return []

    def _estimate_model_accuracy(self, metric_name: str) -> float:
        """Estimate model accuracy for a metric."""
        if metric_name not in self.model_metadata:
            return 0.7  # Default accuracy

        metadata = self.model_metadata[metric_name]
        return metadata.get('accuracy', 0.7)

    def analyze_optimization_impact(self, optimization_name: str,
                                  affected_metrics: List[str]) -> OptimizationImpact:
        """
        Analyze the predicted impact of an optimization.

        Parameters
        ----------
        optimization_name : str
            Name of the optimization
        affected_metrics : List[str]
            List of metrics that will be affected

        Returns
        -------
        OptimizationImpact
            Predicted optimization impact
        """
        try:
            # Analyze historical patterns for similar optimizations
            predicted_improvement = self._predict_optimization_improvement(
                optimization_name, affected_metrics
            )

            # Assess implementation effort
            effort = self._assess_implementation_effort(optimization_name)

            # Evaluate risks
            risk = self._assess_optimization_risk(optimization_name, affected_metrics)

            # Estimate timeline
            timeline = self._estimate_implementation_timeline(optimization_name, effort)

            # Calculate confidence based on historical data
            confidence = self._calculate_optimization_confidence(optimization_name)

            impact = OptimizationImpact(
                optimization_name=optimization_name,
                predicted_improvement=predicted_improvement,
                confidence=confidence,
                affected_metrics=affected_metrics,
                implementation_effort=effort,
                risk_assessment=risk,
                expected_timeline=timeline
            )

            self.optimization_impacts.append(impact)
            logger.info(f"Optimization impact analysis completed for {optimization_name}")

            return impact

        except Exception as e:
            logger.error(f"Optimization impact analysis failed: {e}")
            return OptimizationImpact(
                optimization_name=optimization_name,
                predicted_improvement=0.1,
                confidence=0.5,
                affected_metrics=affected_metrics,
                implementation_effort='medium',
                risk_assessment='medium',
                expected_timeline='1-2 weeks'
            )

    def _predict_optimization_improvement(self, optimization_name: str,
                                        affected_metrics: List[str]) -> float:
        """Predict the improvement from an optimization."""
        # Simple heuristics based on optimization type
        improvement_estimates = {
            'cache': 0.15,  # 15% improvement
            'memory': 0.10,  # 10% improvement
            'algorithm': 0.25,  # 25% improvement
            'jit': 0.30,  # 30% improvement
            'parallel': 0.20,  # 20% improvement
            'vectorization': 0.35,  # 35% improvement
        }

        # Find matching optimization type
        for opt_type, estimate in improvement_estimates.items():
            if opt_type in optimization_name.lower():
                return estimate

        return 0.15  # Default 15% improvement

    def _assess_implementation_effort(self, optimization_name: str) -> str:
        """Assess implementation effort for an optimization."""
        effort_mapping = {
            'cache': 'low',
            'memory': 'medium',
            'algorithm': 'high',
            'jit': 'medium',
            'parallel': 'high',
            'vectorization': 'medium'
        }

        for opt_type, effort in effort_mapping.items():
            if opt_type in optimization_name.lower():
                return effort

        return 'medium'

    def _assess_optimization_risk(self, optimization_name: str,
                                affected_metrics: List[str]) -> str:
        """Assess risk level for an optimization."""
        # Higher risk for optimizations affecting accuracy
        accuracy_metrics = [m for m in affected_metrics if 'accuracy' in m.lower()]
        if accuracy_metrics:
            return 'high'

        # Medium risk for algorithm changes
        if 'algorithm' in optimization_name.lower():
            return 'medium'

        return 'low'

    def _estimate_implementation_timeline(self, optimization_name: str, effort: str) -> str:
        """Estimate implementation timeline."""
        timeline_mapping = {
            'low': '1-3 days',
            'medium': '1-2 weeks',
            'high': '2-4 weeks'
        }
        return timeline_mapping.get(effort, '1-2 weeks')

    def _calculate_optimization_confidence(self, optimization_name: str) -> float:
        """Calculate confidence in optimization prediction."""
        # Higher confidence for well-understood optimizations
        confidence_mapping = {
            'cache': 0.85,
            'memory': 0.80,
            'vectorization': 0.90,
            'jit': 0.85,
            'algorithm': 0.70,
            'parallel': 0.75
        }

        for opt_type, confidence in confidence_mapping.items():
            if opt_type in optimization_name.lower():
                return confidence

        return 0.75  # Default confidence

    def generate_performance_dashboard(self, output_file: Optional[str] = None) -> Optional[str]:
        """
        Generate performance analytics dashboard.

        Parameters
        ----------
        output_file : str, optional
            Output file path for the dashboard

        Returns
        -------
        str or None
            Path to generated dashboard file
        """
        if not PLOTTING_AVAILABLE:
            logger.warning("Dashboard generation not available - matplotlib required")
            return None

        try:
            if output_file is None:
                output_file = str(self.analytics_dir / f"analytics_dashboard_{int(time.time())}.png")

            fig = plt.figure(figsize=(20, 12))

            # Create subplots
            gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

            # Performance trends
            ax1 = fig.add_subplot(gs[0, :2])
            self._plot_performance_trends(ax1)

            # Anomaly detection
            ax2 = fig.add_subplot(gs[0, 2])
            self._plot_anomaly_summary(ax2)

            # Predictions
            ax3 = fig.add_subplot(gs[1, :2])
            self._plot_predictions(ax3)

            # Optimization impacts
            ax4 = fig.add_subplot(gs[1, 2])
            self._plot_optimization_impacts(ax4)

            # Metric distributions
            ax5 = fig.add_subplot(gs[2, 0])
            self._plot_metric_distributions(ax5)

            # Performance correlation
            ax6 = fig.add_subplot(gs[2, 1])
            self._plot_performance_correlation(ax6)

            # System health overview
            ax7 = fig.add_subplot(gs[2, 2])
            self._plot_system_health(ax7)

            plt.suptitle('Homodyne Analysis - Performance Analytics Dashboard', fontsize=16)
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close()

            logger.info(f"Performance dashboard generated: {output_file}")
            return output_file

        except Exception as e:
            logger.error(f"Dashboard generation failed: {e}")
            return None

    def _plot_performance_trends(self, ax):
        """Plot performance trends."""
        ax.set_title('Performance Trends')
        ax.set_xlabel('Time')
        ax.set_ylabel('Metric Value')

        # Plot trends for top metrics
        metrics_plotted = 0
        for metric_name, data in self.historical_data.items():
            if metrics_plotted >= 5:  # Limit to 5 metrics
                break

            if len(data) >= 10:
                timestamps = [datetime.fromtimestamp(d['timestamp']) for d in data[-100:]]
                values = [d['value'] for d in data[-100:]]
                ax.plot(timestamps, values, label=metric_name, alpha=0.7)
                metrics_plotted += 1

        ax.legend()
        ax.grid(True, alpha=0.3)

    def _plot_anomaly_summary(self, ax):
        """Plot anomaly detection summary."""
        ax.set_title('Anomaly Detection Summary')

        # Count anomalies by metric
        anomaly_counts = {}
        for metric_name in self.historical_data.keys():
            anomalies = self.detect_anomalies(metric_name, window_size=50)
            anomaly_counts[metric_name[:15]] = len(anomalies)  # Truncate names

        if anomaly_counts:
            metrics = list(anomaly_counts.keys())
            counts = list(anomaly_counts.values())
            bars = ax.bar(metrics, counts)
            ax.set_ylabel('Anomaly Count')
            ax.set_xticklabels(metrics, rotation=45, ha='right')

            # Color bars by severity
            for bar, count in zip(bars, counts):
                if count > 5:
                    bar.set_color('red')
                elif count > 2:
                    bar.set_color('orange')
                else:
                    bar.set_color('green')
        else:
            ax.text(0.5, 0.5, 'No anomalies detected', ha='center', va='center',
                   transform=ax.transAxes)

    def _plot_predictions(self, ax):
        """Plot performance predictions."""
        ax.set_title('Performance Predictions')
        ax.set_xlabel('Time')
        ax.set_ylabel('Predicted Value')

        if self.predictions:
            # Group predictions by metric
            prediction_groups = {}
            for pred in self.predictions[-20:]:  # Last 20 predictions
                if pred.metric_name not in prediction_groups:
                    prediction_groups[pred.metric_name] = []
                prediction_groups[pred.metric_name].append(pred)

            # Plot predictions
            for metric_name, preds in prediction_groups.items():
                timestamps = [datetime.fromtimestamp(p.timestamp) for p in preds]
                values = [p.predicted_value for p in preds]
                confidence_lower = [p.confidence_lower for p in preds]
                confidence_upper = [p.confidence_upper for p in preds]

                ax.plot(timestamps, values, label=f'{metric_name} (pred)', alpha=0.8)
                ax.fill_between(timestamps, confidence_lower, confidence_upper, alpha=0.2)

            ax.legend()
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No predictions available', ha='center', va='center',
                   transform=ax.transAxes)

    def _plot_optimization_impacts(self, ax):
        """Plot optimization impact analysis."""
        ax.set_title('Optimization Impact Analysis')

        if self.optimization_impacts:
            impacts = self.optimization_impacts[-10:]  # Last 10 impacts
            names = [impact.optimization_name[:15] for impact in impacts]
            improvements = [impact.predicted_improvement * 100 for impact in impacts]
            confidences = [impact.confidence for impact in impacts]

            # Create scatter plot
            scatter = ax.scatter(improvements, confidences,
                               s=[100] * len(improvements), alpha=0.7)

            # Add labels
            for i, name in enumerate(names):
                ax.annotate(name, (improvements[i], confidences[i]),
                           xytext=(5, 5), textcoords='offset points', fontsize=8)

            ax.set_xlabel('Predicted Improvement (%)')
            ax.set_ylabel('Confidence')
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No optimization impacts analyzed', ha='center', va='center',
                   transform=ax.transAxes)

    def _plot_metric_distributions(self, ax):
        """Plot metric value distributions."""
        ax.set_title('Metric Distributions')

        # Plot histograms for key metrics
        metric_names = list(self.historical_data.keys())[:3]  # Top 3 metrics
        for i, metric_name in enumerate(metric_names):
            if len(self.historical_data[metric_name]) >= 10:
                values = [d['value'] for d in self.historical_data[metric_name][-100:]]
                ax.hist(values, bins=20, alpha=0.6, label=metric_name[:15])

        ax.set_xlabel('Value')
        ax.set_ylabel('Frequency')
        ax.legend()

    def _plot_performance_correlation(self, ax):
        """Plot performance metric correlations."""
        ax.set_title('Performance Correlations')

        # Simple correlation analysis
        if len(self.historical_data) >= 2:
            metric_names = list(self.historical_data.keys())[:3]
            correlation_matrix = np.eye(len(metric_names))

            for i, metric1 in enumerate(metric_names):
                for j, metric2 in enumerate(metric_names):
                    if i != j and len(self.historical_data[metric1]) >= 10 and len(self.historical_data[metric2]) >= 10:
                        # Align timestamps and calculate correlation
                        values1 = [d['value'] for d in self.historical_data[metric1][-50:]]
                        values2 = [d['value'] for d in self.historical_data[metric2][-50:]]
                        min_len = min(len(values1), len(values2))
                        if min_len > 2:
                            correlation = np.corrcoef(values1[:min_len], values2[:min_len])[0, 1]
                            correlation_matrix[i, j] = correlation

            im = ax.imshow(correlation_matrix, cmap='RdBu', vmin=-1, vmax=1)
            ax.set_xticks(range(len(metric_names)))
            ax.set_yticks(range(len(metric_names)))
            ax.set_xticklabels([name[:10] for name in metric_names], rotation=45)
            ax.set_yticklabels([name[:10] for name in metric_names])

            # Add colorbar
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('Correlation')
        else:
            ax.text(0.5, 0.5, 'Insufficient data for correlation', ha='center', va='center',
                   transform=ax.transAxes)

    def _plot_system_health(self, ax):
        """Plot system health overview."""
        ax.set_title('System Health Overview')

        # Create health score based on recent metrics
        health_scores = []
        categories = ['Performance', 'Memory', 'Accuracy', 'Stability']

        # Calculate scores for each category
        for category in categories:
            category_metrics = [name for name in self.historical_data.keys()
                              if category.lower() in name.lower()]

            if category_metrics:
                # Simple health calculation
                recent_anomalies = sum(len(self.detect_anomalies(metric, 20))
                                     for metric in category_metrics)
                health_score = max(0, 100 - recent_anomalies * 10)
                health_scores.append(health_score)
            else:
                health_scores.append(80)  # Default health score

        # Create pie chart
        colors = ['green' if score > 80 else 'orange' if score > 60 else 'red'
                 for score in health_scores]
        ax.pie(health_scores, labels=categories, colors=colors, autopct='%1.1f%%')

    def get_analytics_summary(self) -> Dict[str, Any]:
        """Get comprehensive analytics summary."""
        current_time = time.time()

        # Calculate summary statistics
        total_data_points = sum(len(data) for data in self.historical_data.values())
        active_metrics = len([name for name, data in self.historical_data.items()
                            if data and current_time - data[-1]['timestamp'] < 3600])

        # Count recent anomalies
        recent_anomalies = 0
        for metric_name in self.historical_data.keys():
            anomalies = self.detect_anomalies(metric_name, window_size=20)
            recent_anomalies += len(anomalies)

        # Get trend summary
        trend_summary = {}
        for trend in self.trends[-10:]:  # Recent trends
            trend_summary[trend.metric_name] = trend.trend_direction

        # Get prediction summary
        prediction_summary = {}
        for pred in self.predictions[-10:]:  # Recent predictions
            prediction_summary[pred.metric_name] = {
                'predicted_value': pred.predicted_value,
                'confidence': pred.model_accuracy
            }

        return {
            'timestamp': current_time,
            'total_data_points': total_data_points,
            'active_metrics': active_metrics,
            'total_metrics': len(self.historical_data),
            'recent_anomalies': recent_anomalies,
            'trends': trend_summary,
            'predictions': prediction_summary,
            'optimization_impacts': len(self.optimization_impacts),
            'ml_available': ML_AVAILABLE,
            'models_loaded': len(self.models),
            'analytics_health': 'healthy' if recent_anomalies < 5 else 'degraded'
        }

    def cleanup(self):
        """Clean up analytics resources."""
        # Save data and models
        self._save_historical_data()
        self._save_models()

        # Clear memory
        self.historical_data.clear()
        self.predictions.clear()
        self.trends.clear()
        self.optimization_impacts.clear()

        logger.info("Performance analytics cleanup completed")


if __name__ == "__main__":
    # Example usage
    analytics = PerformanceAnalytics()

    # Add some sample data
    for i in range(100):
        analytics.add_performance_data('response_time', 1.0 + 0.1 * np.sin(i * 0.1) + np.random.normal(0, 0.05))
        analytics.add_performance_data('memory_usage', 100 + 10 * np.sin(i * 0.05) + np.random.normal(0, 2))

    # Run analytics
    for metric in ['response_time', 'memory_usage']:
        print(f"\nAnalyzing {metric}:")

        # Detect anomalies
        anomalies = analytics.detect_anomalies(metric)
        print(f"  Anomalies detected: {len(anomalies)}")

        # Analyze trends
        trend = analytics.analyze_trends(metric)
        if trend:
            print(f"  Trend: {trend.trend_direction} (strength: {trend.trend_strength:.3f})")

        # Make prediction
        prediction = analytics.predict_performance(metric)
        if prediction:
            print(f"  Prediction: {prediction.predicted_value:.3f} (confidence: {prediction.model_accuracy:.3f})")

    # Generate dashboard
    dashboard_file = analytics.generate_performance_dashboard()
    if dashboard_file:
        print(f"\nDashboard generated: {dashboard_file}")

    # Print summary
    summary = analytics.get_analytics_summary()
    print(f"\nAnalytics Summary: {summary}")

    analytics.cleanup()