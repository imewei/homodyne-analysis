Configuration
=============

Configuration files and parameter management for the homodyne package.

Configuration Format
--------------------

The homodyne package uses JSON-based configuration files with comprehensive validation. All configuration parameters are validated for physical constraints and data consistency.

Example Configuration
---------------------

.. code-block:: json

   {
     "data_file": "experimental_data.h5",
     "analysis_method": "classical",
     "optimizer": "nelder_mead",
     "static_mode": true,
     "static_submode": "isotropic",
     "angle_filtering": false,
     "num_threads": 4,
     "output_dir": "./homodyne_results"
   }

Parameter Validation
--------------------

- **Diffusion coefficient D(t)**: Automatically enforced to be positive (min 1e-10)
- **Shear rate γ̇(t)**: Automatically enforced to be positive (min 1e-10) 
- **Correlation functions**: c2 values expected in range [1.0, 2.0]
- **Scaling parameters**: contrast ∈ (0.05, 0.5], offset ∈ (0.05, 1.95]

Analysis Modes
--------------

1. **Static Isotropic** (3 parameters): ``static_mode: true, static_submode: "isotropic"``
2. **Static Anisotropic** (3 parameters): ``static_mode: true, static_submode: "anisotropic"``
3. **Laminar Flow** (7 parameters): ``static_mode: false``