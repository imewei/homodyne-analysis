Testing Guide
=============

Comprehensive testing strategies and practices for the homodyne package.

Test Organization
-----------------

The test suite is organized hierarchically:

.. code-block:: text

   homodyne/tests/
   ├── unit/                         # Unit tests
   │   ├── test_config.py           # Configuration tests
   │   ├── test_models.py           # Model function tests
   │   ├── test_optimization.py     # Optimization tests
   │   └── test_utils.py            # Utility function tests
   ├── integration/                  # Integration tests
   │   ├── test_full_workflow.py    # End-to-end workflow
   │   └── test_performance.py      # Performance benchmarks
   ├── fixtures/                     # Test data and fixtures
   │   ├── sample_configs/          # Sample configurations
   │   ├── synthetic_data/          # Generated test data
   │   └── reference_results/       # Expected results
   └── conftest.py                   # Shared fixtures

Running Tests
-------------

**All Tests**:

.. code-block:: bash

   # Run complete test suite
   pytest homodyne/tests/ -v

   # With coverage
   pytest homodyne/tests/ --cov=homodyne --cov-report=html

**Specific Test Categories**:

.. code-block:: bash

   # Unit tests only
   pytest homodyne/tests/unit/ -v

   # Integration tests only
   pytest homodyne/tests/integration/ -v


   # Quick tests only
   pytest homodyne/tests/ -m "not slow"

**Parallel Testing**:

.. code-block:: bash

   # Install pytest-xdist
   pip install pytest-xdist

   # Run tests in parallel
   pytest homodyne/tests/ -n 4

Test Fixtures
-------------

**Common Fixtures** (in ``conftest.py``):

.. code-block:: python

   import pytest
   import numpy as np
   from homodyne import ConfigManager

   @pytest.fixture
   def basic_config():
       """Basic configuration for testing"""
       return {
           "analysis_settings": {
               "static_mode": True,
               "static_submode": "isotropic"
           },
           "initial_parameters": {
               "values": [1000, -0.5, 100]
           }
       }

   @pytest.fixture
   def synthetic_isotropic_data():
       """Synthetic data for isotropic model"""
       tau = np.logspace(-6, 1, 100)
       params = [1500, -0.8, 50]
       q = 0.001

       # Generate perfect isotropic correlation
       g1 = np.exp(-q**2 * (params[0] * tau**(-params[1]) + params[2] * tau))

       # Add realistic noise
       noise = np.random.normal(0, 0.01, size=g1.shape)
       g1_noisy = g1 + noise

       return tau, g1_noisy, params, q

   @pytest.fixture
   def config_manager(basic_config, tmp_path):
       """ConfigManager instance for testing"""
       config_file = tmp_path / "test_config.json"
       with open(config_file, 'w') as f:
           json.dump(basic_config, f)
       return ConfigManager(str(config_file))

Unit Testing
------------

**Model Function Tests**:

.. code-block:: python

   # test_models.py
   import pytest
   import numpy as np
   from homodyne.models import static_isotropic_model

   class TestStaticIsotropicModel:
       def test_basic_functionality(self):
           tau = np.logspace(-6, 1, 100)
           params = [1000, -0.5, 100]
           q = 0.001

           g1 = static_isotropic_model(tau, params, q)

           # Basic checks
           assert len(g1) == len(tau)
           assert np.all(g1 > 0)
           assert np.all(g1 <= 1)
           assert g1[0] > g1[-1]  # Decay

       def test_parameter_bounds(self):
           tau = np.logspace(-6, 1, 10)
           q = 0.001

           # Test with extreme parameters
           params_extreme = [10000, -2.0, 1000]
           g1 = static_isotropic_model(tau, params_extreme, q)
           assert np.all(np.isfinite(g1))

       @pytest.mark.parametrize("params,expected_decay", [
           ([1000, -0.5, 0], "power_law"),
           ([1000, 0, 100], "exponential"),
           ([0, -0.5, 100], "offset_only")
       ])
       def test_decay_behavior(self, params, expected_decay):
           tau = np.logspace(-6, 1, 100)
           g1 = static_isotropic_model(tau, params, 0.001)

           # Check decay characteristics
           if expected_decay == "power_law":
               assert g1[10] > g1[50]  # Power law decay
           elif expected_decay == "exponential":
               # Check exponential form
               pass

**Configuration Tests**:

.. code-block:: python

   # test_config.py
   from homodyne.config import ConfigManager
   from homodyne.utils import ConfigurationError

   class TestConfigManager:
       def test_valid_config(self, basic_config, tmp_path):
           config_file = tmp_path / "valid.json"
           with open(config_file, 'w') as f:
               json.dump(basic_config, f)

           config = ConfigManager(str(config_file))
           assert config.validate() is True

       def test_invalid_config(self, tmp_path):
           invalid_config = {"invalid": "structure"}
           config_file = tmp_path / "invalid.json"
           with open(config_file, 'w') as f:
               json.dump(invalid_config, f)

           with pytest.raises(ConfigurationError):
               ConfigManager(str(config_file))

       def test_missing_file(self):
           with pytest.raises(FileNotFoundError):
               ConfigManager("nonexistent.json")

**Optimization Tests**:

.. code-block:: python

   # test_optimization.py
   from homodyne.analysis.core import HomodyneAnalysisCore
   from homodyne.optimization.classical import ClassicalOptimizer

   class TestClassicalOptimization:
       def test_optimization_convergence(self, config_manager,
                                       synthetic_isotropic_data):
           phi_angles, c2_data, true_params, q = synthetic_isotropic_data

           core = HomodyneAnalysisCore(config_manager)

           # Run classical optimization
           optimizer = ClassicalOptimizer(core, config_manager)
           params, result = optimizer.run_classical_optimization_optimized(
               phi_angles=phi_angles,
               c2_experimental=c2_data
           )

           # Check convergence
           assert result.success
           assert result.chi_squared < 0.1  # Good fit

           # Check parameter recovery (within 10%)
           recovered_params = params
           for i, (recovered, true) in enumerate(zip(recovered_params, true_params)):
               relative_error = abs(recovered - true) / true
               assert relative_error < 0.1, f"Parameter {i} error too large"

Integration Testing
-------------------

**Full Workflow Tests**:

.. code-block:: python

   # test_full_workflow.py
   import tempfile
   import json
   from pathlib import Path

   class TestFullWorkflow:
       def test_complete_isotropic_analysis(self, synthetic_isotropic_data):
           tau, g1_data, true_params, q = synthetic_isotropic_data

           with tempfile.TemporaryDirectory() as tmp_dir:
               tmp_path = Path(tmp_dir)

               # Create test data files
               data_file = tmp_path / "test_data.npz"
               np.savez(data_file, tau=tau, g1=g1_data, q=q)

               # Create configuration
               config = {
                   "analysis_settings": {
                       "static_mode": True,
                       "static_submode": "isotropic"
                   },
                   "file_paths": {
                       "c2_data_file": str(data_file)
                   },
                   "initial_parameters": {
                       "values": [1200, -0.6, 80]  # Slightly off true values
                   }
               }

               config_file = tmp_path / "config.json"
               with open(config_file, 'w') as f:
                   json.dump(config, f)

               # Run complete analysis
               config_manager = ConfigManager(str(config_file))
               core = HomodyneAnalysisCore(config_manager)
               core.load_experimental_data()

               # Use ClassicalOptimizer for optimization
               from homodyne.optimization.classical import ClassicalOptimizer
               optimizer = ClassicalOptimizer(core, config_manager.config)
               params, result = optimizer.run_classical_optimization_optimized(
                   phi_angles=phi_angles, c2_experimental=c2_data)

               # Verify results
               assert result.success
               assert result.chi_squared < 0.05  # Excellent fit for synthetic data

               # Check parameter recovery
               for recovered, true in zip(params, true_params):
                   assert abs(recovered - true) / true < 0.05


.. code-block:: python

   @pytest.mark.slow
           tau, g1_data, true_params, q = synthetic_isotropic_data

           config_manager.config["optimization_config"] = {
                   "enabled": True,
                   "draws": 500,    # Reduced for testing
                   "tune": 200,
                   "chains": 2
               }
           }

           core = HomodyneAnalysisCore(config_manager)
           core._tau = tau
           core._g1_data = g1_data
           core._q = q

           # Run classical first
           from homodyne.optimization.classical import ClassicalOptimizer
           optimizer = ClassicalOptimizer(core, config_manager.config)
           params, result = optimizer.run_classical_optimization_optimized(
               phi_angles=phi_angles, c2_experimental=c2_data)

           # Check convergence

           # Check parameter uncertainties are reasonable

           for param_name in posterior_means.keys():
               mean_val = posterior_means[param_name]
               std_val = posterior_stds[param_name]

               # Uncertainty should be reasonable (not too large)
               cv = std_val / abs(mean_val)  # Coefficient of variation
               assert cv < 0.5, f"Parameter {param_name} uncertainty too large"

Performance Testing
-------------------

**Benchmark Tests**:

.. code-block:: python

   # test_performance.py
   import time
   import pytest

   class TestPerformance:
       @pytest.mark.benchmark
       def test_optimization_speed(self, config_manager, synthetic_isotropic_data):
           """Test that optimization completes within reasonable time"""
           tau, g1_data, true_params, q = synthetic_isotropic_data

           core = HomodyneAnalysisCore(config_manager)
           core._tau = tau
           core._g1_data = g1_data
           core._q = q

           start_time = time.time()
           # Use ClassicalOptimizer for optimization
           from homodyne.optimization.classical import ClassicalOptimizer
           optimizer = ClassicalOptimizer(core, config_manager.config)
           params, result = optimizer.run_classical_optimization_optimized(
               phi_angles=phi_angles, c2_experimental=c2_data)
           end_time = time.time()

           # Should complete within 30 seconds
           assert end_time - start_time < 30
           assert result.success

       @pytest.mark.parametrize("dataset_size", [100, 500, 1000])
       def test_scaling_performance(self, dataset_size):
           """Test performance scaling with dataset size"""
           tau = np.logspace(-6, 1, dataset_size)
           # ... generate data of specified size ...

           # Measure performance and ensure reasonable scaling

Test Data Management
--------------------

**Synthetic Data Generation**:

.. code-block:: python

   # test_data_generator.py
   def generate_test_data(model_type="isotropic", noise_level=0.01):
       """Generate synthetic test data"""
       tau = np.logspace(-6, 1, 100)

       if model_type == "isotropic":
           params = [1500, -0.8, 50]
           g1_perfect = static_isotropic_model(tau, params, 0.001)
       elif model_type == "flow":
           params = [1200, -0.9, 80, 15, 0.3, 2, 0]
           g1_perfect = laminar_flow_model(tau, params, 0.001, 0)

       # Add noise
       noise = np.random.normal(0, noise_level, size=g1_perfect.shape)
       g1_noisy = g1_perfect + noise

       return tau, g1_noisy, params

**Reference Data**:

Store reference results for regression testing:

.. code-block:: python

   # Store expected results
   reference_results = {
       "isotropic_basic": {
           "parameters": [1500.2, -0.801, 49.8],
           "chi_squared": 0.023,
           "success": True
       }
   }

   def test_regression(self):
       # Compare current results with reference
       current_result = run_analysis()
       reference = reference_results["isotropic_basic"]

       for i, (current, expected) in enumerate(
           zip(current_params, reference["parameters"])
       ):
           assert abs(current - expected) / expected < 0.01

Test Configuration
------------------

**pytest.ini**:

.. code-block:: ini

   [tool:pytest]
   testpaths = homodyne/tests
   markers =
       slow: marks tests as slow (deselect with '-m "not slow"')
       benchmark: marks performance benchmark tests
       integration: marks integration tests

   addopts =
       --strict-markers
       --strict-config
       --disable-warnings

**Test Dependencies**:

.. code-block:: text

   # test-requirements.txt
   pytest>=6.0
   pytest-cov>=2.0
   pytest-xdist>=2.0      # Parallel testing
   pytest-benchmark>=3.0   # Performance testing
   pytest-mock>=3.0       # Mocking utilities
   hypothesis>=6.0        # Property-based testing

Continuous Integration
----------------------

**GitHub Actions Example**:

.. code-block:: yaml

   name: Tests
   on: [push, pull_request]

   jobs:
     test:
       runs-on: ubuntu-latest
       strategy:
         matrix:
           python-version: ["3.12", "3.13"]

       steps:
         - uses: actions/checkout@v3
         - name: Set up Python
           uses: actions/setup-python@v3
           with:
             python-version: ${{ matrix.python-version }}

         - name: Install dependencies
           run: |
             pip install -e .[dev]
             pip install -r test-requirements.txt

         - name: Run tests
           run: |
             pytest homodyne/tests/ --cov=homodyne --cov-report=xml

         - name: Upload coverage
           uses: codecov/codecov-action@v3

Test Best Practices
-------------------

1. **Isolation**: Each test should be independent
2. **Descriptive Names**: Test names should explain what they test
3. **Arrange-Act-Assert**: Clear test structure
4. **Edge Cases**: Test boundary conditions and error cases
5. **Performance**: Include performance regression tests
6. **Documentation**: Document complex test scenarios
7. **Maintenance**: Regularly update tests as code evolves
