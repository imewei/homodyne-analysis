#!/bin/bash
# Performance Regression Prevention CI Script
# Automatically checks for performance regressions in pull requests

set -e

echo "🔍 Running Performance Regression Check..."

# Run regression prevention check
python -c "
from homodyne.performance.regression_prevention import PerformanceRegressionPreventor
import sys

preventor = PerformanceRegressionPreventor()
alerts, metrics = preventor.run_comprehensive_regression_check()

# Exit with error code if critical regressions detected
critical_alerts = [a for a in alerts if a.severity == 'CRITICAL']
if critical_alerts:
    print(f'❌ CRITICAL regressions detected: {len(critical_alerts)}')
    for alert in critical_alerts:
        print(f'   • {alert.metric_name}: {alert.regression_percent:.1f}% regression')
    sys.exit(1)

warning_alerts = [a for a in alerts if a.severity == 'WARNING']
if warning_alerts:
    print(f'⚠️  Warnings detected: {len(warning_alerts)}')
    for alert in warning_alerts:
        print(f'   • {alert.metric_name}: {alert.regression_percent:.1f}% regression')

print('✅ Performance regression check passed')
"

echo "✅ Performance regression check completed"
