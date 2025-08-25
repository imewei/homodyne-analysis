#!/usr/bin/env python3
"""
Test script to run tests without optional dependencies.
This mimics the CI environment that doesn't have PyMC, arviz, etc.
"""

import sys
import subprocess

# Block optional dependencies
sys.modules['pymc'] = None
sys.modules['arviz'] = None
sys.modules['corner'] = None

# Now run pytest
if __name__ == "__main__":
    # Run one MCMC test to see if it gets skipped
    result = subprocess.run([
        sys.executable, "-m", "pytest", "-v",
        "homodyne/tests/test_mcmc_config_validation.py::TestMCMCConfigurationUsage::test_mcmc_sampler_extracts_correct_config_values",
        "--tb=short"
    ], env={**dict(os.environ), 'PYMC_AVAILABLE': 'false'} if 'os' in globals() else None)
    
    sys.exit(result.returncode)