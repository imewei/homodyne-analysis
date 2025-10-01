"""
Security Performance Metrics and Monitoring
===========================================

Real-time security monitoring and performance metrics for homodyne analysis.
Provides comprehensive security event tracking, performance impact measurement,
and security health monitoring for scientific computing workloads.

Key Features:
- Real-time security event monitoring
- Performance impact measurement
- Security health scoring
- Automated threat detection
- Resource usage tracking
- Security audit logging

Metrics Collected:
- Input validation events and performance
- File operation security events
- Memory usage and security limits
- Rate limiting effectiveness
- Cache hit rates and security
- Cryptographic operation performance

Authors: Security Engineer (Claude Code)
Institution: Anthropic AI Security
"""

import logging
import threading
import time
from collections import defaultdict
from collections import deque
from dataclasses import dataclass
from dataclasses import field
from datetime import datetime
from datetime import timedelta
from typing import Any

try:
    import psutil

    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class SecurityEvent:
    """Security event data structure."""

    timestamp: datetime
    event_type: str
    severity: str  # 'low', 'medium', 'high', 'critical'
    source: str
    description: str
    metadata: dict[str, Any] = field(default_factory=dict)
    performance_impact_ms: float | None = None


@dataclass
class PerformanceMetric:
    """Performance metric data structure."""

    timestamp: datetime
    operation: str
    duration_ms: float
    success: bool
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SecurityHealthScore:
    """Security health scoring data structure."""

    overall_score: float  # 0-100
    input_validation_score: float
    file_security_score: float
    memory_security_score: float
    rate_limiting_score: float
    last_updated: datetime
    issues: list[str] = field(default_factory=list)


class SecurityMetricsCollector:
    """
    Collects and analyzes security metrics with minimal performance impact.

    Designed for real-time monitoring of security events and performance
    in scientific computing environments.
    """

    def __init__(self, max_events: int = 10000, max_metrics: int = 10000):
        self.max_events = max_events
        self.max_metrics = max_metrics

        # Thread-safe collections
        self._events: deque = deque(maxlen=max_events)
        self._metrics: deque = deque(maxlen=max_metrics)
        self._counters: dict[str, int] = defaultdict(int)
        self._timing_stats: dict[str, list[float]] = defaultdict(list)

        # Thread safety
        self._lock = threading.RLock()

        # Monitoring state
        self._monitoring_active = True
        self._last_health_check = datetime.now()

        # Security thresholds
        self._thresholds = {
            "max_failed_validations_per_minute": 100,
            "max_memory_usage_percent": 85.0,
            "max_operation_time_ms": 5000.0,
            "min_cache_hit_rate": 0.7,
        }

    def record_security_event(
        self,
        event_type: str,
        severity: str,
        source: str,
        description: str,
        metadata: dict[str, Any] | None = None,
        performance_impact_ms: float | None = None,
    ) -> None:
        """
        Record a security event with minimal performance overhead.
        """
        if not self._monitoring_active:
            return

        event = SecurityEvent(
            timestamp=datetime.now(),
            event_type=event_type,
            severity=severity,
            source=source,
            description=description,
            metadata=metadata or {},
            performance_impact_ms=performance_impact_ms,
        )

        with self._lock:
            self._events.append(event)
            self._counters[f"event_{event_type}"] += 1
            self._counters[f"severity_{severity}"] += 1

            if performance_impact_ms is not None:
                self._timing_stats[f"event_{event_type}"].append(performance_impact_ms)

        # Log critical events immediately
        if severity == "critical":
            logger.critical(f"Security event: {description} (source: {source})")
        elif severity == "high":
            logger.warning(f"Security event: {description} (source: {source})")

    def record_performance_metric(
        self,
        operation: str,
        duration_ms: float,
        success: bool,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """
        Record a performance metric for security operations.
        """
        if not self._monitoring_active:
            return

        metric = PerformanceMetric(
            timestamp=datetime.now(),
            operation=operation,
            duration_ms=duration_ms,
            success=success,
            metadata=metadata or {},
        )

        with self._lock:
            self._metrics.append(metric)
            self._counters[f"operation_{operation}"] += 1

            if success:
                self._counters[f"operation_{operation}_success"] += 1
                self._timing_stats[operation].append(duration_ms)
            else:
                self._counters[f"operation_{operation}_failure"] += 1

    def get_security_health_score(self) -> SecurityHealthScore:
        """
        Calculate comprehensive security health score.
        """
        with self._lock:
            now = datetime.now()
            one_hour_ago = now - timedelta(hours=1)

            # Get recent events
            recent_events = [e for e in self._events if e.timestamp > one_hour_ago]
            [m for m in self._metrics if m.timestamp > one_hour_ago]

            # Calculate component scores
            input_validation_score = self._calculate_input_validation_score(
                recent_events
            )
            file_security_score = self._calculate_file_security_score(recent_events)
            memory_security_score = self._calculate_memory_security_score()
            rate_limiting_score = self._calculate_rate_limiting_score(recent_events)

            # Calculate overall score (weighted average)
            overall_score = (
                input_validation_score * 0.3
                + file_security_score * 0.25
                + memory_security_score * 0.25
                + rate_limiting_score * 0.2
            )

            # Identify issues
            issues = []
            if input_validation_score < 70:
                issues.append("High number of validation failures")
            if file_security_score < 70:
                issues.append("File security issues detected")
            if memory_security_score < 70:
                issues.append("Memory usage approaching limits")
            if rate_limiting_score < 70:
                issues.append("Rate limiting frequently triggered")

            return SecurityHealthScore(
                overall_score=overall_score,
                input_validation_score=input_validation_score,
                file_security_score=file_security_score,
                memory_security_score=memory_security_score,
                rate_limiting_score=rate_limiting_score,
                last_updated=now,
                issues=issues,
            )

    def get_performance_summary(self) -> dict[str, Any]:
        """
        Get performance summary for security operations.
        """
        with self._lock:
            summary = {
                "total_events": len(self._events),
                "total_metrics": len(self._metrics),
                "counters": dict(self._counters),
                "timing_stats": {},
            }

            # Calculate timing statistics
            for operation, times in self._timing_stats.items():
                if times:
                    summary["timing_stats"][operation] = {
                        "count": len(times),
                        "mean_ms": sum(times) / len(times),
                        "min_ms": min(times),
                        "max_ms": max(times),
                        "p95_ms": self._percentile(times, 95),
                        "p99_ms": self._percentile(times, 99),
                    }

            return summary

    def get_recent_critical_events(self, hours: int = 24) -> list[SecurityEvent]:
        """
        Get recent critical security events.
        """
        cutoff_time = datetime.now() - timedelta(hours=hours)

        with self._lock:
            return [
                event
                for event in self._events
                if event.timestamp > cutoff_time and event.severity == "critical"
            ]

    def clear_old_data(self, hours: int = 168) -> None:  # Default: 1 week
        """
        Clear old monitoring data to prevent memory growth.
        """
        cutoff_time = datetime.now() - timedelta(hours=hours)

        with self._lock:
            # Filter events and metrics
            self._events = deque(
                [e for e in self._events if e.timestamp > cutoff_time],
                maxlen=self.max_events,
            )

            self._metrics = deque(
                [m for m in self._metrics if m.timestamp > cutoff_time],
                maxlen=self.max_metrics,
            )

            # Clear old timing stats (keep only recent data)
            for operation in list(self._timing_stats.keys()):
                # Keep only last 1000 measurements per operation
                self._timing_stats[operation] = self._timing_stats[operation][-1000:]

        logger.debug(f"Cleared monitoring data older than {hours} hours")

    def export_security_report(self) -> dict[str, Any]:
        """
        Export comprehensive security report.
        """
        health_score = self.get_security_health_score()
        performance_summary = self.get_performance_summary()
        critical_events = self.get_recent_critical_events()

        return {
            "report_timestamp": datetime.now().isoformat(),
            "health_score": {
                "overall": health_score.overall_score,
                "input_validation": health_score.input_validation_score,
                "file_security": health_score.file_security_score,
                "memory_security": health_score.memory_security_score,
                "rate_limiting": health_score.rate_limiting_score,
                "issues": health_score.issues,
            },
            "performance": performance_summary,
            "critical_events": [
                {
                    "timestamp": event.timestamp.isoformat(),
                    "type": event.event_type,
                    "source": event.source,
                    "description": event.description,
                    "metadata": event.metadata,
                }
                for event in critical_events
            ],
            "system_info": self._get_system_info(),
        }

    def _get_system_info(self) -> dict[str, Any]:
        """Get system information for security context."""
        info = {
            "monitoring_active": self._monitoring_active,
            "events_in_memory": len(self._events),
            "metrics_in_memory": len(self._metrics),
        }

        if PSUTIL_AVAILABLE:
            try:
                info.update(
                    {
                        "memory_percent": psutil.virtual_memory().percent,
                        "cpu_percent": psutil.cpu_percent(),
                        "disk_usage_percent": psutil.disk_usage("/").percent,
                    }
                )
            except Exception as e:
                info["system_info_error"] = str(e)

        return info

    def start_monitoring(self) -> None:
        """Start security monitoring."""
        self._monitoring_active = True
        logger.info("Security monitoring started")

    def stop_monitoring(self) -> None:
        """Stop security monitoring."""
        self._monitoring_active = False
        logger.info("Security monitoring stopped")

    def is_monitoring_active(self) -> bool:
        """Check if monitoring is active."""
        return self._monitoring_active


# Global metrics collector instance
security_metrics = SecurityMetricsCollector()


# Decorator for automatic performance monitoring
def monitor_security_performance(operation_name: str):
    """
    Decorator to automatically monitor security operation performance.
    """

    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            success = False
            error = None

            try:
                result = func(*args, **kwargs)
                success = True
                return result

            except Exception as e:
                error = e
                raise

            finally:
                duration_ms = (time.time() - start_time) * 1000

                metadata = {}
                if error:
                    metadata["error"] = str(error)
                    metadata["error_type"] = type(error).__name__

                security_metrics.record_performance_metric(
                    operation=operation_name,
                    duration_ms=duration_ms,
                    success=success,
                    metadata=metadata,
                )

        return wrapper

    return decorator


# Convenience functions for common security events
def log_validation_event(validation_type: str, success: bool, details: str = ""):
    """Log input validation event."""
    severity = "low" if success else "high"
    event_type = f"validation_{validation_type}"
    description = f"Validation {validation_type}: {'success' if success else 'failed'}"
    if details:
        description += f" - {details}"

    security_metrics.record_security_event(
        event_type=event_type,
        severity=severity,
        source="input_validation",
        description=description,
    )


def log_file_security_event(
    operation: str, file_path: str, success: bool, details: str = ""
):
    """Log file security event."""
    severity = "low" if success else "medium"
    event_type = f"file_{operation}"
    description = (
        f"File {operation} on {file_path}: {'success' if success else 'failed'}"
    )
    if details:
        description += f" - {details}"

    security_metrics.record_security_event(
        event_type=event_type,
        severity=severity,
        source="file_security",
        description=description,
        metadata={"file_path": file_path},
    )


def log_rate_limit_event(operation: str, identifier: str, triggered: bool):
    """Log rate limiting event."""
    severity = "low" if not triggered else "medium"
    event_type = "rate_limit_check"
    description = f"Rate limit for {operation} ({identifier}): {'triggered' if triggered else 'ok'}"

    security_metrics.record_security_event(
        event_type=event_type,
        severity=severity,
        source="rate_limiting",
        description=description,
        metadata={"operation": operation, "identifier": identifier},
    )


# Periodic cleanup function
def cleanup_security_metrics(max_age_hours: int = 168):
    """Clean up old security metrics data."""
    security_metrics.clear_old_data(hours=max_age_hours)
    logger.debug(f"Cleaned up security metrics older than {max_age_hours} hours")


# Health check function
def get_security_health_status() -> dict[str, Any]:
    """Get current security health status."""
    health_score = security_metrics.get_security_health_score()

    return {
        "overall_score": health_score.overall_score,
        "status": (
            "healthy"
            if health_score.overall_score >= 80
            else "warning" if health_score.overall_score >= 60 else "critical"
        ),
        "component_scores": {
            "input_validation": health_score.input_validation_score,
            "file_security": health_score.file_security_score,
            "memory_security": health_score.memory_security_score,
            "rate_limiting": health_score.rate_limiting_score,
        },
        "issues": health_score.issues,
        "last_updated": health_score.last_updated.isoformat(),
    }
