#!/usr/bin/env python3
"""
Continuous Optimization Monitoring System - Production Deployment Script
========================================================================

Complete demonstration and deployment script for the homodyne analysis
continuous optimization monitoring system. This script shows how to:

1. Initialize all monitoring components
2. Integrate with existing homodyne analysis systems
3. Configure production-ready monitoring
4. Set up automated optimization and alerting
5. Generate real-time performance dashboards

This script can be used as a template for production deployment or
run directly for demonstration purposes.

Usage:
    python run_continuous_monitoring.py --mode demo
    python run_continuous_monitoring.py --mode production --config monitoring_config.json
    python run_continuous_monitoring.py --mode test --duration 300
"""

import argparse
import asyncio
import json
import logging
import os
import signal
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('continuous_monitoring.log')
    ]
)
logger = logging.getLogger(__name__)

# Import monitoring components
try:
    from homodyne.monitoring import (
        ContinuousOptimizationMonitor,
        PerformanceAnalytics,
        AlertSystem,
        BaselineManager,
        OptimizationEngine
    )
    MONITORING_AVAILABLE = True
except ImportError as e:
    logger.error(f"Monitoring components not available: {e}")
    MONITORING_AVAILABLE = False

# Import existing homodyne components
try:
    from homodyne.performance_monitoring import PerformanceMonitor
    from homodyne.core.security_performance import secure_cache
    HOMODYNE_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Homodyne components not fully available: {e}")
    HOMODYNE_AVAILABLE = False


class ContinuousMonitoringSystem:
    """
    Complete continuous monitoring system orchestrator.

    Manages all monitoring components and provides a unified interface
    for production deployment and demonstration.
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize the complete monitoring system."""
        self.config = config
        self.components = {}
        self.running = False

        # Initialize components
        if MONITORING_AVAILABLE:
            self._initialize_components()
            self._setup_integrations()
        else:
            logger.error("Cannot initialize - monitoring components not available")
            sys.exit(1)

    def _initialize_components(self):
        """Initialize all monitoring components."""
        logger.info("Initializing monitoring components...")

        try:
            # Initialize core components
            self.components['monitor'] = ContinuousOptimizationMonitor(self.config)
            self.components['analytics'] = PerformanceAnalytics(self.config)
            self.components['alerts'] = AlertSystem(self.config)
            self.components['baselines'] = BaselineManager(self.config)
            self.components['optimizer'] = OptimizationEngine(self.config)

            # Initialize existing performance monitor if available
            if HOMODYNE_AVAILABLE:
                self.components['legacy_monitor'] = PerformanceMonitor(
                    self.config.get('legacy_monitor', {})
                )

            logger.info("All monitoring components initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize components: {e}")
            raise

    def _setup_integrations(self):
        """Set up integrations between components."""
        logger.info("Setting up component integrations...")

        try:
            # Link analytics and baseline manager to main monitor
            self.components['monitor'].analytics = self.components['analytics']
            self.components['monitor'].baseline_manager = self.components['baselines']

            # Link analytics and baselines to optimization engine
            self.components['optimizer'].set_analytics(self.components['analytics'])
            self.components['optimizer'].set_baseline_manager(self.components['baselines'])

            # Set up alert callbacks for baseline drift
            self.components['baselines'].add_drift_callback(
                self._handle_baseline_drift
            )

            # Register auto-remediation callbacks
            self._setup_auto_remediation()

            logger.info("Component integrations completed")

        except Exception as e:
            logger.error(f"Failed to setup integrations: {e}")
            raise

    def _setup_auto_remediation(self):
        """Set up auto-remediation callbacks."""
        optimizer = self.components['optimizer']

        # Cache optimization
        optimizer.register_remediation_callback(
            'cache_tune',
            self._auto_tune_cache
        )

        # Memory optimization
        optimizer.register_remediation_callback(
            'gc_tune',
            self._auto_tune_gc
        )

        # Performance optimization
        optimizer.register_remediation_callback(
            'parallel_increase',
            self._auto_increase_parallelism
        )

        logger.info("Auto-remediation callbacks registered")

    def _handle_baseline_drift(self, drift):
        """Handle baseline drift notifications."""
        logger.warning(f"Baseline drift detected: {drift.metric_name} "
                      f"({drift.drift_direction} {drift.drift_magnitude:.1%})")

        # Create alert for significant drift
        if drift.drift_magnitude > 0.2:  # 20% drift
            self.components['alerts'].evaluate_metric(
                f"baseline_drift_{drift.metric_name}",
                drift.drift_magnitude,
                {
                    'drift_direction': drift.drift_direction,
                    'confidence': drift.confidence,
                    'suggested_action': drift.suggested_action
                }
            )

    def _auto_tune_cache(self):
        """Auto-tune cache configuration."""
        logger.info("Auto-tuning cache configuration")

        if HOMODYNE_AVAILABLE and hasattr(secure_cache, '_cache'):
            # Increase cache size if hit rate is low
            current_size = getattr(secure_cache, 'max_size', 128)
            new_size = min(current_size * 1.5, 1000)  # Cap at 1000
            secure_cache.max_size = int(new_size)
            logger.info(f"Cache size increased from {current_size} to {new_size}")

    def _auto_tune_gc(self):
        """Auto-tune garbage collection."""
        logger.info("Auto-tuning garbage collection")

        import gc
        # Force garbage collection
        gc.collect()

        # Tune GC thresholds if memory pressure is high
        current_thresholds = gc.get_threshold()
        new_thresholds = tuple(int(t * 0.8) for t in current_thresholds)
        gc.set_threshold(*new_thresholds)
        logger.info(f"GC thresholds adjusted: {current_thresholds} -> {new_thresholds}")

    def _auto_increase_parallelism(self):
        """Auto-increase parallelism for better performance."""
        logger.info("Auto-increasing parallelism")

        # This would typically adjust thread pool sizes or parallel processing parameters
        # For demonstration, we just log the action
        logger.info("Parallelism configuration optimized")

    def start(self):
        """Start the complete monitoring system."""
        if self.running:
            logger.warning("Monitoring system already running")
            return

        logger.info("Starting continuous monitoring system...")

        try:
            # Start all components
            self.components['monitor'].start_monitoring()
            self.components['optimizer'].start_engine()

            self.running = True
            logger.info("Continuous monitoring system started successfully")

            # Generate initial reports
            self._generate_initial_reports()

        except Exception as e:
            logger.error(f"Failed to start monitoring system: {e}")
            raise

    def stop(self):
        """Stop the complete monitoring system."""
        if not self.running:
            logger.warning("Monitoring system not running")
            return

        logger.info("Stopping continuous monitoring system...")

        try:
            # Stop all components
            self.components['monitor'].stop_monitoring()
            self.components['optimizer'].stop_engine()

            # Cleanup components
            for component in self.components.values():
                if hasattr(component, 'cleanup'):
                    component.cleanup()
                elif hasattr(component, 'shutdown'):
                    component.shutdown()

            self.running = False
            logger.info("Continuous monitoring system stopped successfully")

        except Exception as e:
            logger.error(f"Error stopping monitoring system: {e}")

    def _generate_initial_reports(self):
        """Generate initial monitoring reports."""
        logger.info("Generating initial monitoring reports...")

        try:
            # Generate monitoring report
            monitor_report = self.components['monitor'].generate_monitoring_report()
            self._save_report('monitoring_report', monitor_report)

            # Generate analytics dashboard
            dashboard_file = self.components['analytics'].generate_performance_dashboard()
            if dashboard_file:
                logger.info(f"Analytics dashboard generated: {dashboard_file}")

            # Get system status
            status = self.get_system_status()
            self._save_report('system_status', status)

            logger.info("Initial reports generated successfully")

        except Exception as e:
            logger.warning(f"Failed to generate initial reports: {e}")

    def _save_report(self, report_type: str, data: Dict[str, Any]):
        """Save a report to disk."""
        try:
            reports_dir = Path('monitoring_reports')
            reports_dir.mkdir(exist_ok=True)

            filename = f"{report_type}_{int(time.time())}.json"
            filepath = reports_dir / filename

            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2, default=str)

            logger.debug(f"Report saved: {filepath}")

        except Exception as e:
            logger.error(f"Failed to save report {report_type}: {e}")

    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        status = {
            'timestamp': time.time(),
            'running': self.running,
            'components': {}
        }

        try:
            # Get status from each component
            status['components']['monitor'] = self.components['monitor'].get_monitoring_status()
            status['components']['analytics'] = self.components['analytics'].get_analytics_summary()
            status['components']['alerts'] = self.components['alerts'].get_alert_statistics()
            status['components']['baselines'] = self.components['baselines'].get_baseline_status()
            status['components']['optimizer'] = self.components['optimizer'].get_optimization_status()

            # Calculate overall health
            health_statuses = [
                comp_status.get('health_status', 'unknown')
                for comp_status in status['components'].values()
            ]

            if all(h == 'healthy' for h in health_statuses):
                status['overall_health'] = 'healthy'
            elif any(h == 'degraded' for h in health_statuses):
                status['overall_health'] = 'degraded'
            else:
                status['overall_health'] = 'unknown'

        except Exception as e:
            logger.error(f"Failed to get system status: {e}")
            status['error'] = str(e)

        return status

    def simulate_workload(self, duration: int = 300):
        """Simulate scientific computing workload for demonstration."""
        logger.info(f"Simulating workload for {duration} seconds...")

        import random
        import numpy as np

        start_time = time.time()
        iteration = 0

        while time.time() - start_time < duration:
            iteration += 1

            try:
                # Simulate various operations with performance monitoring

                # Simulate chi-squared calculation
                with self.components['monitor'].monitor_operation('chi_squared_calculation', expected_accuracy=0.999):
                    # Simulate computation with some variability
                    computation_time = random.uniform(0.5, 2.0)
                    time.sleep(computation_time)

                    # Add performance data to analytics
                    self.components['analytics'].add_performance_data(
                        'chi_squared_response_time',
                        computation_time,
                        {'iteration': iteration}
                    )

                # Simulate memory usage
                memory_usage = random.uniform(50, 200)  # MB
                self.components['analytics'].add_performance_data(
                    'memory_usage',
                    memory_usage,
                    {'iteration': iteration}
                )

                # Simulate cache hit rate
                cache_hit_rate = random.uniform(0.7, 0.95)
                self.components['analytics'].add_performance_data(
                    'cache_hit_rate',
                    cache_hit_rate,
                    {'iteration': iteration}
                )

                # Simulate accuracy metric
                accuracy = 0.999 + random.normal(0, 0.001)
                accuracy = max(0.99, min(1.0, accuracy))  # Clamp to reasonable range
                self.components['analytics'].add_performance_data(
                    'computational_accuracy',
                    accuracy,
                    {'iteration': iteration}
                )

                # Add data points to baseline manager
                self.components['baselines'].add_data_point('response_time', computation_time)
                self.components['baselines'].add_data_point('memory_usage', memory_usage)
                self.components['baselines'].add_data_point('accuracy', accuracy)

                # Occasionally simulate performance issues
                if iteration % 20 == 0:
                    # Simulate slow response
                    slow_time = random.uniform(3.0, 5.0)
                    self.components['alerts'].evaluate_metric(
                        'response_time_spike',
                        slow_time,
                        {'cause': 'simulated_spike'}
                    )

                if iteration % 30 == 0:
                    # Simulate accuracy issue
                    poor_accuracy = random.uniform(0.98, 0.995)
                    self.components['alerts'].evaluate_metric(
                        'accuracy_degradation',
                        poor_accuracy,
                        {'cause': 'simulated_degradation'}
                    )

                # Wait between iterations
                time.sleep(random.uniform(5, 15))

                # Log progress
                if iteration % 10 == 0:
                    elapsed = time.time() - start_time
                    logger.info(f"Workload simulation: {iteration} iterations, "
                              f"{elapsed:.1f}s elapsed, {duration - elapsed:.1f}s remaining")

            except Exception as e:
                logger.error(f"Error in workload simulation iteration {iteration}: {e}")
                continue

        logger.info(f"Workload simulation completed: {iteration} iterations")

    def print_status_summary(self):
        """Print a summary of system status."""
        status = self.get_system_status()

        print("\n" + "="*80)
        print("CONTINUOUS MONITORING SYSTEM STATUS")
        print("="*80)
        print(f"Overall Health: {status.get('overall_health', 'unknown').upper()}")
        print(f"System Running: {status.get('running', False)}")
        print(f"Timestamp: {time.ctime(status.get('timestamp', time.time()))}")

        print("\nComponent Status:")
        print("-" * 40)
        for comp_name, comp_status in status.get('components', {}).items():
            health = comp_status.get('health_status', 'unknown')
            print(f"  {comp_name.title()}: {health}")

        # Print key metrics
        monitor_status = status.get('components', {}).get('monitor', {})
        if monitor_status:
            print(f"\nMonitoring Metrics:")
            print(f"  Active Metrics: {monitor_status.get('active_metrics', 0)}")
            print(f"  Total Metrics: {monitor_status.get('total_metrics', 0)}")
            print(f"  Alert Counts: {monitor_status.get('alert_counts', {})}")

        baseline_status = status.get('components', {}).get('baselines', {})
        if baseline_status:
            print(f"\nBaseline Status:")
            print(f"  Total Baselines: {baseline_status.get('total_baselines', 0)}")
            print(f"  Valid Baselines: {baseline_status.get('valid_baselines', 0)}")
            print(f"  Recent Drifts: {baseline_status.get('recent_drifts', 0)}")

        optimizer_status = status.get('components', {}).get('optimizer', {})
        if optimizer_status:
            print(f"\nOptimization Status:")
            print(f"  Total Opportunities: {optimizer_status.get('total_opportunities', 0)}")
            print(f"  High Priority: {optimizer_status.get('high_priority_opportunities', 0)}")
            print(f"  Recent Executions: {optimizer_status.get('recent_executions', 0)}")

        print("="*80)


def load_config(config_file: Optional[str] = None) -> Dict[str, Any]:
    """Load monitoring configuration."""
    default_config = {
        # Monitoring intervals
        'monitoring_interval': 30,
        'analysis_interval': 600,

        # Directories
        'monitoring_dir': 'continuous_monitoring',
        'analytics_dir': 'performance_analytics',
        'alerts_dir': 'alerts',
        'baselines_dir': 'baselines',
        'optimization_dir': 'optimizations',

        # SLI targets
        'sli_targets': {
            'accuracy_preservation': 0.999,
            'response_time': 2.0,
            'memory_utilization': 0.80,
            'cache_hit_rate': 0.85,
            'jit_compilation_overhead': 0.20,
            'security_overhead': 0.10,
            'error_rate': 0.01,
            'throughput': 100.0
        },

        # Alert thresholds
        'alert_thresholds': {
            'response_time_warning': 1.5,
            'response_time_critical': 3.0,
            'memory_warning': 0.85,
            'memory_critical': 0.95,
            'error_rate_warning': 0.05,
            'error_rate_critical': 0.10,
            'accuracy_warning': 0.995,
            'accuracy_critical': 0.99
        },

        # Alert channels
        'channels': {
            'console': {
                'type': 'console',
                'enabled': True,
                'config': {'colored': True},
                'rate_limit': 60
            },
            'file': {
                'type': 'file',
                'enabled': True,
                'config': {
                    'file_path': 'alerts/alerts.log',
                    'format': 'json'
                },
                'rate_limit': 1000
            }
        },

        # Alert rules
        'rules': {
            'response_time_warning': {
                'metric_pattern': '*response_time*',
                'threshold': 1.5,
                'comparison': 'gt',
                'severity': 'warning',
                'channels': ['console', 'file'],
                'cooldown': 300
            },
            'response_time_critical': {
                'metric_pattern': '*response_time*',
                'threshold': 3.0,
                'comparison': 'gt',
                'severity': 'critical',
                'channels': ['console', 'file'],
                'cooldown': 60
            },
            'accuracy_critical': {
                'metric_pattern': '*accuracy*',
                'threshold': 0.99,
                'comparison': 'lt',
                'severity': 'critical',
                'channels': ['console', 'file'],
                'cooldown': 0
            }
        },

        # Optimization settings
        'auto_remediation_enabled': True,
        'auto_execution_threshold': 0.9,
        'accuracy_loss_threshold': 0.01,

        # ML settings
        'ml_models': {
            'anomaly_detection': {
                'contamination': 0.1,
                'n_estimators': 100
            },
            'performance_prediction': {
                'n_estimators': 100,
                'max_depth': 10
            }
        },

        # Baseline settings
        'auto_update_enabled': True,
        'drift_detection_sensitivity': 2.0,
        'min_samples_for_baseline': 50
    }

    if config_file and Path(config_file).exists():
        try:
            with open(config_file, 'r') as f:
                user_config = json.load(f)

            # Merge configurations
            def merge_dicts(default, user):
                for key, value in user.items():
                    if key in default and isinstance(default[key], dict) and isinstance(value, dict):
                        merge_dicts(default[key], value)
                    else:
                        default[key] = value

            merge_dicts(default_config, user_config)
            logger.info(f"Configuration loaded from {config_file}")

        except Exception as e:
            logger.warning(f"Failed to load config file {config_file}: {e}")
            logger.info("Using default configuration")

    return default_config


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Continuous Optimization Monitoring System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --mode demo                    # Run demo mode
  %(prog)s --mode test --duration 300     # Run test for 5 minutes
  %(prog)s --mode production --config production.json  # Production mode
        """
    )

    parser.add_argument(
        '--mode',
        choices=['demo', 'production', 'test'],
        default='demo',
        help='Monitoring mode (default: demo)'
    )

    parser.add_argument(
        '--config',
        help='Configuration file path'
    )

    parser.add_argument(
        '--duration',
        type=int,
        default=300,
        help='Test duration in seconds (default: 300)'
    )

    parser.add_argument(
        '--simulate-workload',
        action='store_true',
        help='Simulate scientific computing workload'
    )

    parser.add_argument(
        '--generate-reports',
        action='store_true',
        help='Generate monitoring reports and dashboards'
    )

    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Logging level (default: INFO)'
    )

    args = parser.parse_args()

    # Set logging level
    logging.getLogger().setLevel(getattr(logging, args.log_level))

    # Load configuration
    config = load_config(args.config)

    # Create monitoring system
    try:
        monitoring_system = ContinuousMonitoringSystem(config)
    except Exception as e:
        logger.error(f"Failed to create monitoring system: {e}")
        sys.exit(1)

    # Set up signal handlers for graceful shutdown
    def signal_handler(signum, frame):
        logger.info(f"Received signal {signum}, shutting down...")
        monitoring_system.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        # Start monitoring
        monitoring_system.start()

        if args.mode == 'demo':
            print("\n" + "="*80)
            print("CONTINUOUS OPTIMIZATION MONITORING SYSTEM - DEMO MODE")
            print("="*80)
            print("The monitoring system is now running in demo mode.")
            print("This will:")
            print("  • Monitor system performance in real-time")
            print("  • Generate alerts for performance issues")
            print("  • Detect optimization opportunities")
            print("  • Create performance baselines")
            print("  • Provide automated remediation")
            print("\nPress Ctrl+C to stop the monitoring system.")
            print("="*80)

            # Run indefinitely until interrupted
            try:
                while True:
                    time.sleep(30)
                    monitoring_system.print_status_summary()
            except KeyboardInterrupt:
                pass

        elif args.mode == 'test':
            print(f"\nRunning monitoring system test for {args.duration} seconds...")

            # Simulate workload if requested
            if args.simulate_workload:
                monitoring_system.simulate_workload(args.duration)
            else:
                time.sleep(args.duration)

            # Print final status
            monitoring_system.print_status_summary()

        elif args.mode == 'production':
            logger.info("Starting production monitoring mode")
            print("Monitoring system started in production mode.")
            print("Monitor logs at: continuous_monitoring.log")
            print("Press Ctrl+C to stop the monitoring system.")

            # Run indefinitely in production mode
            try:
                while True:
                    time.sleep(60)  # Status check every minute
                    status = monitoring_system.get_system_status()
                    if status.get('overall_health') != 'healthy':
                        logger.warning(f"System health degraded: {status.get('overall_health')}")
            except KeyboardInterrupt:
                pass

        # Generate reports if requested
        if args.generate_reports:
            logger.info("Generating final reports...")
            monitoring_system._generate_initial_reports()

    except Exception as e:
        logger.error(f"Error during monitoring execution: {e}")
        monitoring_system.stop()
        sys.exit(1)

    finally:
        # Clean shutdown
        monitoring_system.stop()
        print("\nMonitoring system stopped successfully.")


if __name__ == "__main__":
    main()