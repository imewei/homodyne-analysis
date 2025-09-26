#!/usr/bin/env python3
"""
Test Script for Continuous Optimization Monitoring System
=========================================================

Simple test script to validate that all monitoring components work correctly.
This script performs basic functionality tests and integration verification.
"""

import asyncio
import logging
import random
import sys
import time
from pathlib import Path

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def test_imports():
    """Test that all monitoring components can be imported."""
    print("Testing imports...")

    try:
        from homodyne.monitoring import (
            ContinuousOptimizationMonitor,
            PerformanceAnalytics,
            AlertSystem,
            BaselineManager,
            OptimizationEngine
        )
        print("✓ All monitoring components imported successfully")
        return True
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False

def test_component_initialization():
    """Test initialization of all components."""
    print("\nTesting component initialization...")

    try:
        from homodyne.monitoring import (
            ContinuousOptimizationMonitor,
            PerformanceAnalytics,
            AlertSystem,
            BaselineManager,
            OptimizationEngine
        )

        # Test configuration
        test_config = {
            'monitoring_dir': 'test_monitoring',
            'monitoring_interval': 10,
            'sli_targets': {
                'accuracy_preservation': 0.999,
                'response_time': 2.0
            }
        }

        # Initialize components
        monitor = ContinuousOptimizationMonitor(test_config)
        analytics = PerformanceAnalytics(test_config)
        alerts = AlertSystem(test_config)
        baselines = BaselineManager(test_config)
        optimizer = OptimizationEngine(test_config)

        print("✓ All components initialized successfully")
        return True, (monitor, analytics, alerts, baselines, optimizer)

    except Exception as e:
        print(f"✗ Initialization failed: {e}")
        return False, None

def test_basic_functionality(components):
    """Test basic functionality of each component."""
    print("\nTesting basic functionality...")

    monitor, analytics, alerts, baselines, optimizer = components

    try:
        # Test analytics
        analytics.add_performance_data('test_metric', 1.5, {'test': True})
        print("✓ Analytics: Data addition successful")

        # Test baseline manager
        baselines.add_data_point('test_response_time', 1.2)
        print("✓ Baseline Manager: Data point addition successful")

        # Test alert system (without actual delivery)
        alerts.evaluate_metric('test_alert_metric', 5.0, {'test': True})
        print("✓ Alert System: Metric evaluation successful")

        # Test optimization engine status
        status = optimizer.get_optimization_status()
        print(f"✓ Optimization Engine: Status retrieved ({status['health_status']})")

        # Test monitor status
        monitor_status = monitor.get_monitoring_status()
        print(f"✓ Monitor: Status retrieved ({monitor_status['health_status']})")

        return True

    except Exception as e:
        print(f"✗ Basic functionality test failed: {e}")
        return False

def test_integration():
    """Test integration between components."""
    print("\nTesting component integration...")

    try:
        from homodyne.monitoring import (
            ContinuousOptimizationMonitor,
            PerformanceAnalytics,
            BaselineManager,
            OptimizationEngine
        )

        # Create components
        config = {'monitoring_dir': 'test_integration'}
        monitor = ContinuousOptimizationMonitor(config)
        analytics = PerformanceAnalytics(config)
        baselines = BaselineManager(config)
        optimizer = OptimizationEngine(config)

        # Set up integrations
        monitor.analytics = analytics
        monitor.baseline_manager = baselines
        optimizer.set_analytics(analytics)
        optimizer.set_baseline_manager(baselines)

        print("✓ Component integration successful")
        return True

    except Exception as e:
        print(f"✗ Integration test failed: {e}")
        return False

def test_monitoring_operation():
    """Test the monitoring operation context manager."""
    print("\nTesting monitoring operation...")

    try:
        from homodyne.monitoring import ContinuousOptimizationMonitor

        monitor = ContinuousOptimizationMonitor({'monitoring_dir': 'test_operation'})

        # Test monitoring context
        with monitor.monitor_operation('test_operation', expected_accuracy=0.999):
            # Simulate some work
            time.sleep(0.1)

        print("✓ Monitoring operation context successful")
        return True

    except Exception as e:
        print(f"✗ Monitoring operation test failed: {e}")
        return False

def test_alert_generation():
    """Test alert generation and handling."""
    print("\nTesting alert generation...")

    try:
        from homodyne.monitoring import AlertSystem

        config = {
            'alerts_dir': 'test_alerts',
            'channels': {
                'console': {
                    'type': 'console',
                    'enabled': True,
                    'config': {'colored': False},
                    'rate_limit': 10
                }
            },
            'rules': {
                'test_rule': {
                    'metric_pattern': 'test_*',
                    'threshold': 2.0,
                    'comparison': 'gt',
                    'severity': 'warning',
                    'channels': ['console'],
                    'cooldown': 0
                }
            }
        }

        alerts = AlertSystem(config)

        # Generate test alert
        alerts.evaluate_metric('test_high_value', 3.0, {'test': True})

        # Check alert statistics
        stats = alerts.get_alert_statistics()
        print(f"✓ Alert generation successful (alerts: {stats['total_alerts']})")
        return True

    except Exception as e:
        print(f"✗ Alert generation test failed: {e}")
        return False

def test_performance_analytics():
    """Test performance analytics functionality."""
    print("\nTesting performance analytics...")

    try:
        from homodyne.monitoring import PerformanceAnalytics

        analytics = PerformanceAnalytics({'analytics_dir': 'test_analytics'})

        # Add sample data
        for i in range(50):
            value = 1.0 + 0.1 * random.random()
            analytics.add_performance_data('sample_metric', value)

        # Test anomaly detection
        anomalies = analytics.detect_anomalies('sample_metric')
        print(f"✓ Anomaly detection successful (anomalies: {len(anomalies)})")

        # Test trend analysis
        trend = analytics.analyze_trends('sample_metric')
        if trend:
            print(f"✓ Trend analysis successful (direction: {trend.trend_direction})")
        else:
            print("✓ Trend analysis successful (no trend data)")

        return True

    except Exception as e:
        print(f"✗ Performance analytics test failed: {e}")
        return False

def test_baseline_management():
    """Test baseline management functionality."""
    print("\nTesting baseline management...")

    try:
        from homodyne.monitoring import BaselineManager

        manager = BaselineManager({
            'baselines_dir': 'test_baselines',
            'auto_update_enabled': False,  # Disable auto-update for testing
            'min_samples_for_baseline': 10
        })

        # Add sample data
        for i in range(20):
            value = 1.0 + 0.1 * random.random()
            manager.add_data_point('sample_baseline', value)

        # Update baseline manually
        success = manager.update_baseline('sample_baseline')
        print(f"✓ Baseline update successful: {success}")

        # Validate baseline
        validation = manager.validate_baseline('sample_baseline')
        print(f"✓ Baseline validation successful (valid: {validation.get('valid', False)})")

        return True

    except Exception as e:
        print(f"✗ Baseline management test failed: {e}")
        return False

def test_optimization_engine():
    """Test optimization engine functionality."""
    print("\nTesting optimization engine...")

    try:
        from homodyne.monitoring import OptimizationEngine

        config = {
            'optimization_dir': 'test_optimization',
            'auto_remediation_enabled': False,  # Disable for testing
            'analysis_interval': 60
        }

        engine = OptimizationEngine(config)

        # Test status
        status = engine.get_optimization_status()
        print(f"✓ Optimization engine status: {status['health_status']}")

        # Test opportunities (should be empty initially)
        opportunities = engine.get_top_opportunities(5)
        print(f"✓ Optimization opportunities retrieved: {len(opportunities)}")

        return True

    except Exception as e:
        print(f"✗ Optimization engine test failed: {e}")
        return False

async def test_async_functionality():
    """Test asynchronous functionality."""
    print("\nTesting async functionality...")

    try:
        from homodyne.monitoring import AlertSystem
        from homodyne.monitoring.alert_system import ConsoleChannel

        # Test async alert delivery
        console_channel = ConsoleChannel({'colored': False})

        # Create a mock alert for testing
        from homodyne.monitoring.alert_system import Alert
        test_alert = Alert(
            alert_id='test_async',
            rule_id='test_rule',
            timestamp=time.time(),
            severity='info',
            category='test',
            title='Test Async Alert',
            message='This is a test alert for async functionality',
            metric_name='test_metric',
            current_value=1.0,
            threshold=0.5,
            context={'test': True},
            suggested_actions=['No action required - this is a test']
        )

        # Send alert asynchronously
        success = await console_channel.send_alert(test_alert)
        print(f"✓ Async alert delivery successful: {success}")

        return True

    except Exception as e:
        print(f"✗ Async functionality test failed: {e}")
        return False

def cleanup_test_directories():
    """Clean up test directories created during testing."""
    print("\nCleaning up test directories...")

    test_dirs = [
        'test_monitoring',
        'test_integration',
        'test_operation',
        'test_alerts',
        'test_analytics',
        'test_baselines',
        'test_optimization'
    ]

    for test_dir in test_dirs:
        try:
            import shutil
            path = Path(test_dir)
            if path.exists():
                shutil.rmtree(path)
                print(f"✓ Cleaned up {test_dir}")
        except Exception as e:
            print(f"✗ Failed to cleanup {test_dir}: {e}")

async def run_all_tests():
    """Run all tests."""
    print("="*80)
    print("CONTINUOUS OPTIMIZATION MONITORING SYSTEM - TEST SUITE")
    print("="*80)

    test_results = []

    # Test imports
    test_results.append(test_imports())

    if not test_results[-1]:
        print("\n✗ Import test failed - cannot continue with other tests")
        return False

    # Test component initialization
    init_success, components = test_component_initialization()
    test_results.append(init_success)

    if init_success and components:
        # Test basic functionality
        test_results.append(test_basic_functionality(components))

    # Test integration
    test_results.append(test_integration())

    # Test monitoring operation
    test_results.append(test_monitoring_operation())

    # Test alert generation
    test_results.append(test_alert_generation())

    # Test performance analytics
    test_results.append(test_performance_analytics())

    # Test baseline management
    test_results.append(test_baseline_management())

    # Test optimization engine
    test_results.append(test_optimization_engine())

    # Test async functionality
    test_results.append(await test_async_functionality())

    # Clean up
    cleanup_test_directories()

    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)

    passed = sum(test_results)
    total = len(test_results)

    print(f"Tests passed: {passed}/{total}")
    print(f"Success rate: {passed/total*100:.1f}%")

    if passed == total:
        print("✓ All tests passed! The monitoring system is working correctly.")
        return True
    else:
        print("✗ Some tests failed. Please check the error messages above.")
        return False

def main():
    """Main test entry point."""
    try:
        # Run async tests
        success = asyncio.run(run_all_tests())

        if success:
            print("\n🎉 Monitoring system test suite completed successfully!")
            print("The continuous optimization monitoring system is ready for use.")
            sys.exit(0)
        else:
            print("\n❌ Test suite failed. Please address the issues before using the system.")
            sys.exit(1)

    except KeyboardInterrupt:
        print("\n\nTest suite interrupted by user.")
        cleanup_test_directories()
        sys.exit(1)
    except Exception as e:
        print(f"\n\nTest suite failed with error: {e}")
        cleanup_test_directories()
        sys.exit(1)

if __name__ == "__main__":
    main()