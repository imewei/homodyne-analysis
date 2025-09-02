Post-Installation System
========================

The homodyne package includes a unified post-installation system that streamlines setup of shell completion, GPU acceleration, and advanced CLI tools.

.. automodule:: homodyne.post_install
   :members:
   :undoc-members:
   :show-inheritance:

Main Functions
--------------

install_shell_completion
~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: homodyne.post_install.install_shell_completion

Sets up shell completion for zsh, bash, and fish shells. Provides tab completion for commands, methods, and file paths.

**Example:**

.. code-block:: python

   from homodyne.post_install import install_shell_completion

   # Install completion for zsh
   success = install_shell_completion(shell_type="zsh", force=True)
   if success:
       print("Shell completion installed successfully")

install_gpu_acceleration
~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: homodyne.post_install.install_gpu_acceleration

Configures GPU acceleration on Linux systems with NVIDIA hardware. Sets up environment variables and activation scripts.

**Example:**

.. code-block:: python

   from homodyne.post_install import install_gpu_acceleration

   # Install GPU acceleration (Linux only)
   success = install_gpu_acceleration(force=True)
   if success:
       print("GPU acceleration configured")

install_advanced_features
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: homodyne.post_install.install_advanced_features

Installs advanced CLI tools including homodyne-gpu-optimize and homodyne-validate.

**Example:**

.. code-block:: python

   from homodyne.post_install import install_advanced_features

   # Install advanced CLI tools
   success = install_advanced_features()
   if success:
       print("Advanced features installed")

Utility Functions
-----------------

is_virtual_environment
~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: homodyne.post_install.is_virtual_environment

Detects if running in a virtual environment (conda, mamba, venv, virtualenv).

is_conda_environment
~~~~~~~~~~~~~~~~~~~~

.. autofunction:: homodyne.post_install.is_conda_environment

Checks if the environment is specifically a conda/mamba environment.

Command Line Interface
----------------------

The post-install system can be used from the command line:

.. code-block:: bash

   # Interactive setup
   homodyne-post-install

   # Non-interactive with all features
   homodyne-post-install --shell zsh --gpu --advanced

   # Shell completion only
   homodyne-post-install --shell bash

   # GPU and advanced features
   homodyne-post-install --gpu --advanced

Command Line Options
~~~~~~~~~~~~~~~~~~~~

- ``--shell SHELL`` - Install shell completion (zsh, bash, fish)
- ``--gpu`` - Install GPU acceleration (Linux only)
- ``--advanced`` - Install advanced CLI tools
- ``--force`` - Force installation even if not in virtual environment
- ``--non-interactive`` - Skip interactive prompts

Integration with Package Installation
-------------------------------------

The post-install system integrates seamlessly with pip installation:

.. code-block:: bash

   # Install package
   pip install homodyne-analysis[all]

   # Run unified setup
   homodyne-post-install --shell zsh --gpu --advanced

This replaces the previous separate installation steps and provides a consistent experience across all platforms and shells.

Cleanup and Uninstallation
---------------------------

The system works with the homodyne-cleanup tool for easy removal:

.. code-block:: bash

   # Interactive cleanup
   homodyne-cleanup

   # Remove all features
   homodyne-cleanup --all

   # Dry run to see what would be removed
   homodyne-cleanup --dry-run

See :doc:`../user-guide/installation` for complete installation instructions.
