Command Line Interface Reference
==================================

This comprehensive reference covers all CLI commands provided by the Homodyne Analysis package.

Core Analysis Commands
----------------------

homodyne
~~~~~~~~

Main analysis command for X-ray Photon Correlation Spectroscopy (XPCS) data under nonequilibrium conditions.

**Usage:**

.. code-block:: bash

   homodyne [OPTIONS]

**Analysis Mode Options:**

.. code-block:: bash

   --static-isotropic      Force 3-parameter isotropic mode (fastest)
   --static-anisotropic    Force 3-parameter anisotropic mode  
   --laminar-flow         Force 7-parameter laminar flow mode (default)

**Method Selection:**

.. code-block:: bash

   --method METHOD        Choose optimization method:
                         • classical (default): Nelder-Mead + Gurobi
                         • robust: Noise-resistant methods
                         • mcmc: Bayesian MCMC sampling
                         • all: Run all methods

**Configuration and Data:**

.. code-block:: bash

   --config FILE          Configuration file path (default: config.json)
   --output-dir DIR       Output directory (default: ./homodyne_results)

**Visualization Options:**

.. code-block:: bash

   --plot-experimental-data    Generate experimental data validation plots
   --plot-simulated-data      Generate theoretical correlation plots
   --plot-c2-heatmaps         Force regeneration of correlation heatmaps
   --contrast FLOAT           Override contrast parameter for simulated data
   --offset FLOAT             Override offset parameter for simulated data
   --phi-angles ANGLES        Override phi angles (comma-separated degrees)

**Logging Control:**

.. code-block:: bash

   --verbose, -v          Enable debug logging to console and file
   --quiet, -q            File logging only, no console output

**Examples:**

.. code-block:: bash

   # Basic analysis with classical methods
   homodyne

   # Comprehensive analysis with all methods
   homodyne --method all --verbose

   # Robust optimization for noisy data
   homodyne --method robust --static-isotropic

   # Data validation without fitting
   homodyne --plot-experimental-data --verbose

   # Custom configuration and output
   homodyne --config my_experiment.json --output-dir ./results

homodyne-config
~~~~~~~~~~~~~~~

Configuration file generator with intelligent defaults.

**Usage:**

.. code-block:: bash

   homodyne-config [OPTIONS]

**Options:**

.. code-block:: bash

   --mode MODE            Analysis mode:
                         • static_isotropic: 3-parameter isotropic
                         • static_anisotropic: 3-parameter anisotropic  
                         • laminar_flow: 7-parameter flow (default)
   --sample NAME          Sample identifier for metadata
   --author AUTHOR        Author name for documentation
   --experiment DESC      Experiment description
   --output FILE          Output configuration filename

**Examples:**

.. code-block:: bash

   # Generate laminar flow configuration
   homodyne-config --mode laminar_flow --sample protein_solution

   # Create isotropic mode config with metadata
   homodyne-config --mode static_isotropic --sample microgel \\
                   --author "Your Name" --experiment "Equilibrium dynamics"

Setup and Management Commands
-----------------------------

homodyne-post-install
~~~~~~~~~~~~~~~~~~~~~

Unified post-installation setup for shell completion and GPU acceleration.

**Usage:**

.. code-block:: bash

   homodyne-post-install [OPTIONS]

**Options:**

.. code-block:: bash

   --shell SHELL          Shell type (bash, zsh, fish)
   --gpu                  Enable GPU acceleration setup
   --advanced             Install advanced tools and optimization
   --interactive, -i      Interactive setup with guided choices
   --completion-only      Install only shell completion
   --gpu-only             Install only GPU acceleration
   --force                Force reinstallation of components

**Examples:**

.. code-block:: bash

   # Complete unified setup (recommended)
   homodyne-post-install --shell zsh --gpu --advanced

   # Interactive setup
   homodyne-post-install --interactive

   # Shell completion only
   homodyne-post-install --shell bash --completion-only

**Unified System Features:**

- **Cross-shell completion**: bash, zsh, fish, PowerShell
- **Smart GPU detection**: Automatic CUDA configuration
- **Advanced tools**: GPU optimization, system validation
- **Environment integration**: Virtual environment detection

homodyne-cleanup
~~~~~~~~~~~~~~~~

Environment cleanup utility for complete removal.

**Usage:**

.. code-block:: bash

   homodyne-cleanup [OPTIONS]

**Important:** Run this before ``pip uninstall homodyne-analysis`` to ensure complete cleanup of environment scripts.

Advanced Tools
--------------

homodyne-validate
~~~~~~~~~~~~~~~~~

Comprehensive system validation and health checking.

**Usage:**

.. code-block:: bash

   homodyne-validate [OPTIONS]

**Options:**

.. code-block:: bash

   --verbose, -v          Show detailed validation output
   --test TYPE            Run specific test category:
                         • environment: Platform and Python checks
                         • installation: Command availability
                         • completion: Shell completion testing  
                         • gpu: GPU setup verification
                         • integration: End-to-end testing
   --json                 Output results in JSON format
   --quick                Run only essential tests
   --fix                  Attempt to fix common issues automatically

**Test Categories:**

1. **Environment Detection**
   - Platform identification (Linux/macOS/Windows)
   - Python version compatibility (3.12+)
   - Virtual environment detection (conda, mamba, venv, virtualenv)
   - Shell type identification

2. **Installation Verification**
   - Command availability
   - Core module imports
   - Dependencies check
   - Help output validation

3. **Shell Completion**
   - Completion file presence
   - Activation script functionality
   - Alias availability
   - Cross-shell compatibility

4. **GPU Setup**
   - Hardware detection
   - JAX device availability
   - CUDA installation verification
   - Driver compatibility

5. **Integration Testing**
   - Component interaction validation
   - End-to-end workflow testing

**Examples:**

.. code-block:: bash

   # Full system validation
   homodyne-validate

   # Verbose diagnostic output
   homodyne-validate --verbose

   # Test specific components
   homodyne-validate --test gpu
   homodyne-validate --test completion

   # JSON output for CI/CD
   homodyne-validate --json > validation_report.json

   # Quick health check
   homodyne-validate --quick

homodyne-gpu-optimize
~~~~~~~~~~~~~~~~~~~~~

GPU optimization and performance benchmarking tool.

**Usage:**

.. code-block:: bash

   homodyne-gpu-optimize [OPTIONS]

**Options:**

.. code-block:: bash

   --benchmark            Run performance benchmarks
   --apply                Apply optimal settings automatically
   --interactive          Interactive optimization wizard
   --profile              Profile GPU memory usage
   --test                 Test GPU acceleration functionality

**Examples:**

.. code-block:: bash

   # Run benchmarks and apply optimal settings
   homodyne-gpu-optimize --benchmark --apply

   # Interactive optimization
   homodyne-gpu-optimize --interactive

   # Profile memory usage
   homodyne-gpu-optimize --profile

Unified System Aliases
-----------------------

After running ``homodyne-post-install --advanced``, these convenient aliases are available:

**Analysis Shortcuts:**

.. code-block:: bash

   hm --config config.json         # homodyne --method mcmc
   hc --config config.json         # homodyne --method classical  
   hr --config config.json         # homodyne --method robust
   ha --config config.json         # homodyne --method all
   hconfig                # homodyne-config

**System Tools:**

.. code-block:: bash

   gpu-status             # Check GPU activation status
   gpu-bench              # GPU benchmarking
   gpu-on                 # Manual GPU activation
   gpu-off                # Manual GPU deactivation

**Tab Completion:**

The unified system provides intelligent tab completion:

.. code-block:: bash

   homodyne --method <TAB>     # Shows: classical, robust, mcmc, all
   homodyne --config <TAB>     # Shows available config files
   hm <TAB>                    # Smart config file completion

Performance Optimization
-------------------------

**Threading Control:**

.. code-block:: bash

   # Set for reproducible performance
   export OMP_NUM_THREADS=4
   export OPENBLAS_NUM_THREADS=4
   export MKL_NUM_THREADS=4
   export NUMBA_DISABLE_INTEL_SVML=1

**GPU Acceleration (Linux + NVIDIA):**

.. code-block:: bash

   # Enable GPU acceleration
   homodyne-post-install --gpu
   
   # Check GPU status
   gpu-status
   
   # Optimize GPU settings
   homodyne-gpu-optimize --benchmark --apply

**Memory Optimization:**

.. code-block:: bash

   # For large datasets
   export NUMBA_CACHE_DIR=/tmp/numba_cache
   export HOMODYNE_PERFORMANCE_MODE=1

Common Usage Patterns
---------------------

**Quick Analysis Workflow:**

.. code-block:: bash

   # 1. Setup (one-time)
   pip install homodyne-analysis[all]
   homodyne-post-install --shell zsh --gpu --advanced
   
   # 2. Create config
   hconfig --mode laminar_flow --sample my_sample
   
   # 3. Run analysis
   ha --config config.json  # All methods with smart GPU/CPU selection

**Data Quality Check:**

.. code-block:: bash

   # Validate data before analysis
   homodyne --plot-experimental-data --config config.json --verbose

**Robust Analysis for Noisy Data:**

.. code-block:: bash

   # Use robust methods for noisy experimental data
   hr --config noisy_data.json --verbose

**Performance Testing:**

.. code-block:: bash

   # Benchmark and optimize
   homodyne-validate --test gpu
   homodyne-gpu-optimize --benchmark

Error Diagnosis
---------------

**Common Issues and Solutions:**

1. **Command not found:**
   
   .. code-block:: bash
   
      # Check installation
      homodyne-validate --test installation

2. **GPU not detected:**
   
   .. code-block:: bash
   
      # Validate GPU setup
      homodyne-validate --test gpu --verbose

3. **Shell completion not working:**
   
   .. code-block:: bash
   
      # Test completion system
      homodyne-validate --test completion

4. **Analysis failures:**
   
   .. code-block:: bash
   
      # Full diagnostic
      homodyne-validate --verbose

For comprehensive troubleshooting, see :doc:`../developer-guide/troubleshooting`.

Integration with Development Workflow
-------------------------------------

**CI/CD Integration:**

.. code-block:: bash

   # Automated validation in CI
   homodyne-validate --json --quick > validation.json

**Development Testing:**

.. code-block:: bash

   # Test installation after changes
   homodyne-validate --test installation --verbose

**Performance Monitoring:**

.. code-block:: bash

   # Regular performance checks
   homodyne-gpu-optimize --benchmark > performance_report.txt

This CLI interface provides comprehensive functionality for scientific analysis while maintaining ease of use through intelligent defaults and helpful validation tools.