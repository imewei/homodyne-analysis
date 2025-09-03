Frequently Asked Questions (FAQ)
==================================

This page addresses common questions about installing and using the Homodyne Analysis package.

Installation Questions
-----------------------

**Q: What Python version do I need?**

A: Python 3.12 or higher is required. The package will automatically check your version and provide helpful error messages if you're using an older version.

**Q: Can I use this package on Windows/macOS?**

A: Yes! The package is fully cross-platform. However, GPU acceleration is only available on Linux with NVIDIA GPUs.

**Q: How do I install with GPU support?**

A: For Linux with NVIDIA GPU:

.. code-block:: bash

   pip install homodyne-analysis[jax]
   homodyne-post-install --gpu

**Q: What if I get dependency conflicts during installation?**

A: Try installing in a clean environment:

.. code-block:: bash

   # Using conda/mamba
   conda create -n homodyne python=3.12
   conda activate homodyne
   pip install homodyne-analysis[all]

Usage Questions
---------------

**Q: Which analysis mode should I use?**

A: Choose based on your system:

* **Static Isotropic** (3 params): Equilibrium systems, fastest analysis
* **Static Anisotropic** (3 params): Equilibrium with angle dependence  
* **Laminar Flow** (7 params): Systems under shear, most comprehensive

**Q: Which optimization method is best?**

A: It depends on your data quality:

* **Clean data**: ``--method classical`` (fast)
* **Noisy data**: ``--method robust`` (noise-resistant)
* **Need uncertainties**: ``--method mcmc`` (comprehensive)
* **Complete analysis**: ``--method all``

**Q: How long does analysis take?**

A: Typical times:

* Classical methods: Minutes
* Robust methods: 10-30 minutes  
* MCMC: 1-4 hours (depending on parameters)

**Q: What data formats are supported?**

A: The package accepts:

* **HDF5 files** from XPCS experiments (requires ``xpcs-viewer``)
* **NPZ files** with correlation data structure ``(n_phi, n_t1, n_t2)``

Configuration Questions
-----------------------

**Q: How do I create a configuration file?**

A: Use the configuration generator:

.. code-block:: bash

   homodyne-config --mode laminar_flow --sample my_sample

**Q: Can I modify the generated configuration?**

A: Yes! The JSON configuration file is human-readable and can be edited. See :doc:`configuration` for details.

**Q: What if my data has different angle arrangements?**

A: Modify the ``phi_angles`` array in your configuration file to match your experimental setup.

Results Interpretation
----------------------

**Q: What do the parameter values mean physically?**

A: Key parameters:

* **D₀**: Diffusion coefficient amplitude [Ų/s]
* **α**: Time scaling exponent (≈ -1.5 for normal diffusion)
* **γ̇₀**: Shear rate amplitude [s⁻¹] 
* **β**: Shear scaling exponent
* **φ₀**: Angular offset [degrees]

**Q: How do I know if my fit is good?**

A: Check these quality indicators:

* **Chi-squared**: Lower is better
* **Reduced χ²**: Should be ≈ 1.0
* **R̂ values** (MCMC): Should be < 1.1
* **Visual inspection**: Check correlation heatmaps

**Q: What do the uncertainty values represent?**

A: From MCMC analysis, these are Bayesian posterior standard deviations representing parameter estimation uncertainties.

Performance Questions
---------------------

**Q: How can I speed up analysis?**

A: Several options:

.. code-block:: bash

   # Use classical methods only
   homodyne --method classical
   
   # Enable GPU (Linux + NVIDIA)
   homodyne-post-install --gpu
   
   # Optimize threading
   export OMP_NUM_THREADS=4

**Q: My MCMC is very slow. What can I do?**

A: Try:

* Reduce MCMC samples in configuration
* Enable GPU acceleration if available
* Use classical results as better starting points
* Consider simpler analysis mode (fewer parameters)

**Q: The analysis runs out of memory. Help!**

A: Options:

* Reduce data size by binning correlation functions
* Close other applications
* Use classical methods instead of MCMC
* Consider upgrading system memory

Troubleshooting
---------------

**Q: I get import errors when running analysis**

A: Check your installation:

.. code-block:: bash

   homodyne-validate --verbose

This will diagnose common issues.

**Q: The plots look strange/wrong**

A: Verify data quality first:

.. code-block:: bash

   homodyne --plot-experimental-data --config config.json

**Q: MCMC doesn't converge (R̂ > 1.1)**

A: Try:

* Increase number of MCMC samples
* Adjust initial parameter guesses
* Check if parameter bounds are reasonable
* Consider using classical results as starting point

**Q: I get "optimization failed" errors**

A: Common fixes:

* Check data quality and format
* Verify configuration parameters are reasonable  
* Try different optimization methods
* Use ``--verbose`` flag for detailed error messages

**Q: GPU acceleration isn't working**

A: Requirements for GPU acceleration:

* Linux operating system
* NVIDIA GPU with CUDA support
* CUDA 12.6+ and cuDNN 9.12+ installed
* Run ``gpu-status`` to check activation

Advanced Questions
------------------

**Q: Can I use my own data processing pipeline?**

A: Yes! As long as your data conforms to the expected NPZ structure ``(n_phi, n_t1, n_t2)``.

**Q: How do I cite this package?**

A: Please cite the original research paper:

.. code-block:: bibtex

   @article{he2024transport,
     title={Transport coefficient approach for characterizing nonequilibrium dynamics in soft matter},
     author={He, Hongrui and Liang, Hao and Chu, Miaoqi and Jiang, Zhang and de Pablo, Juan J and Tirrell, Matthew V and Narayanan, Suresh and Chen, Wei},
     journal={Proceedings of the National Academy of Sciences},
     volume={121},
     number={31},
     pages={e2401162121},
     year={2024},
     publisher={National Academy of Sciences},
     doi={10.1073/pnas.2401162121}
   }

**Q: Can I contribute to the project?**

A: Absolutely! See :doc:`../developer-guide/contributing` for guidelines on contributing code, documentation, or reporting issues.

**Q: Where can I get more help?**

A: Additional resources:

* **Documentation**: https://homodyne.readthedocs.io/
* **GitHub Issues**: https://github.com/imewei/homodyne/issues
* **Complete Tutorial**: :doc:`complete-workflow-tutorial`
* **Troubleshooting Guide**: :doc:`../developer-guide/troubleshooting`

Still Have Questions?
---------------------

If your question isn't answered here, please:

1. Check the complete documentation at https://homodyne.readthedocs.io/
2. Search existing issues at https://github.com/imewei/homodyne/issues
3. Create a new issue with details about your problem

We're happy to help improve both the software and documentation!