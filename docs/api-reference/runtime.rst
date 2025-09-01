Runtime System
==============

Runtime utilities for system validation, GPU optimization, and shell completion.

The runtime system provides command-line tools and utilities for managing the homodyne
installation environment, validating system configuration, and optimizing GPU performance.

GPU Optimization
----------------

GPU optimization and benchmarking utilities.

.. automodule:: homodyne.runtime.gpu.optimizer
   :members:
   :undoc-members:
   :show-inheritance:

System Validation  
------------------

System validation and health check utilities.

.. automodule:: homodyne.runtime.utils.system_validator
   :members:
   :undoc-members:
   :show-inheritance:

Console Scripts
---------------

The runtime system provides these console scripts:

- ``homodyne-gpu-optimize`` - GPU optimization and benchmarking tool
- ``homodyne-validate`` - System validation and health checks

Usage Examples
--------------

**Command Line Usage:**

.. code-block:: bash

   # Run GPU optimization
   homodyne-gpu-optimize --benchmark --profile
   
   # Run system validation
   homodyne-validate --verbose --comprehensive

**Python API Usage:**

.. code-block:: python

   from homodyne.runtime.gpu.optimizer import GPUOptimizer
   from homodyne.runtime.utils.system_validator import SystemValidator
   
   # GPU optimization
   optimizer = GPUOptimizer()
   results = optimizer.run_optimization()
   
   # System validation
   validator = SystemValidator(verbose=True)
   validation_results = validator.run_all_tests()

Module Structure
----------------

The runtime system is organized into the following modules:

- ``homodyne.runtime.gpu`` - GPU-related utilities and optimization
- ``homodyne.runtime.utils`` - System utilities and validation
- ``homodyne.runtime.shell`` - Shell completion and integration