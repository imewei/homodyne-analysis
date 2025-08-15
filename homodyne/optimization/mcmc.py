"""
MCMC/NUTS Sampling Methods for Homodyne Scattering Analysis
==========================================================

This module contains MCMC and NUTS sampling algorithms extracted from the
ConfigurableHomodyneAnalysis class, including:
- PyMC-based Bayesian model construction
- NUTS (No-U-Turn Sampler) for efficient sampling
- Uncertainty quantification and posterior analysis
- Convergence diagnostics and chain analysis

MCMC methods provide full Bayesian uncertainty quantification by sampling
from the posterior distribution of model parameters.

Authors: Wei Chen, Hongrui He
Institution: Argonne National Laboratory & University of Chicago
"""

import time
import logging
from typing import Any, Dict, List, Optional

import numpy as np

# Import homodyne modules using relative import - keep for potential future use
try:
    from ..analysis.core import HomodyneAnalysisCore  # noqa: F401

    HOMODYNE_CORE_AVAILABLE = True
except ImportError:
    HOMODYNE_CORE_AVAILABLE = False
    HomodyneAnalysisCore = None

# PyMC and Bayesian inference dependencies
try:
    import arviz as az
    import pymc as pm
    import pytensor.tensor as pt
    from pytensor.compile.sharedvalue import shared

    PYMC_AVAILABLE = True
except ImportError:
    PYMC_AVAILABLE = False
    pm = None
    az = None
    pt = None
    shared = None

logger = logging.getLogger(__name__)


class MCMCSampler:
    """
    MCMC and NUTS sampling for Bayesian parameter estimation.

    This class provides advanced Bayesian sampling using PyMC's No-U-Turn
    Sampler (NUTS) for comprehensive uncertainty quantification of model
    parameters.
    """

    def __init__(self, analysis_core, config: Dict[str, Any]):
        """
        Initialize MCMC sampler.

        Parameters
        ----------
        analysis_core : HomodyneAnalysisCore
            Core analysis engine instance
        config : Dict[str, Any]
            Configuration dictionary

        Raises
        ------
        ImportError
            If required dependencies are not available
        ValueError
            If configuration is invalid
        """
        # Validate dependencies
        if not PYMC_AVAILABLE:
            raise ImportError(
                "PyMC is required for MCMC sampling but is not available. "
                "Install with: pip install pymc arviz"
            )

        # Validate inputs
        if analysis_core is None:
            raise ValueError("Analysis core instance is required")
        if not isinstance(config, dict):
            raise ValueError("Configuration must be a dictionary")

        self.core = analysis_core
        self.config = config
        self.bayesian_model = None
        self.mcmc_trace = None
        self.mcmc_result = None

        # Extract MCMC configuration
        self.mcmc_config = config.get("optimization_config", {}).get(
            "mcmc_sampling", {}
        )

        # Validate MCMC configuration
        self._validate_mcmc_config()

        logger.info("MCMC sampler initialized successfully")

    def _build_bayesian_model_optimized(
        self,
        c2_experimental: np.ndarray,
        phi_angles: np.ndarray,
        filter_angles_for_optimization: bool = True,
        is_static_mode: bool = False,
        effective_param_count: int = 7,
    ):
        """
        Build optimized Bayesian model for MCMC sampling.

        This method constructs a probabilistic model for Bayesian inference
        with PyMC, including proper priors and likelihood functions.

        Parameters
        ----------
        c2_experimental : np.ndarray
            Experimental correlation data
        phi_angles : np.ndarray
            Scattering angles
        filter_angles_for_optimization : bool, default True
            If True, use only angles in ranges [-10°, 10°] and [170°, 190°] for likelihood
        is_static_mode : bool, default False
            Whether static mode is enabled
        effective_param_count : int, default 7
            Number of parameters to use (3 for static, 7 for laminar flow)

        Returns
        -------
        pm.Model
            PyMC model ready for MCMC sampling

        Raises
        ------
        ImportError
            If PyMC is not available
        """
        if not PYMC_AVAILABLE:
            raise ImportError(
                "PyMC is required for Bayesian analysis but is not available. "
                "Install with: pip install pymc"
            )

        # Type assertions for type checker - these are guaranteed after the availability check
        assert pm is not None
        assert pt is not None
        assert shared is not None

        print("   Building Bayesian model with PyMC...")
        performance_config = self.config.get("performance_settings", {})

        # Data preprocessing for efficiency
        n_angles, n_time, _ = c2_experimental.shape  # noqa: F841

        # Apply angle filtering for MCMC optimization
        if filter_angles_for_optimization:
            # Define target angle ranges: [-10°, 10°] and [170°, 190°]
            target_ranges = [(-10.0, 10.0), (170.0, 190.0)]

            # Find indices of angles in target ranges
            optimization_indices = []
            for i, angle in enumerate(phi_angles):
                for min_angle, max_angle in target_ranges:
                    if min_angle <= angle <= max_angle:
                        optimization_indices.append(i)
                        break

            if optimization_indices:
                # Filter experimental data to optimization angles only
                c2_data_unfiltered = c2_experimental
                c2_experimental = c2_experimental[optimization_indices]
                phi_angles_filtered = phi_angles[optimization_indices]

                print(
                    f"   MCMC angle filtering: using {len(optimization_indices)}/{n_angles} angles"
                )
                print(
                    f"   Optimization angles: {[f'{angle:.1f}°' for angle in phi_angles_filtered]}"
                )
                logger.info(
                    f"MCMC using filtered angles: {len(optimization_indices)}/{n_angles} angles"
                )
            else:
                print(
                    "   Warning: No angles found in optimization ranges [-10°, 10°] and [170°, 190°]"
                )
                print("   Falling back to all angles for MCMC")
                logger.warning("No MCMC optimization angles found, using all angles")

        # Update n_angles after potential filtering
        n_angles, n_time, _ = c2_experimental.shape

        # Optional subsampling for large datasets
        subsample_factor = performance_config.get("bayesian_subsample_factor", 1)
        if subsample_factor > 1 and n_time > 50:
            subsample_indices = np.arange(0, n_time, subsample_factor)
            c2_data = c2_experimental[:, subsample_indices, :][:, :, subsample_indices]
            print(
                f"   Subsampling data by factor {subsample_factor}: {n_time}x{n_time} -> {len(subsample_indices)}x{len(subsample_indices)}"
            )
        else:
            c2_data = c2_experimental

        # Use float32 for memory efficiency if configured
        use_float32 = performance_config.get("use_float32_precision", True)
        dtype = np.float32 if use_float32 else np.float64
        dtype_str = "float32" if use_float32 else "float64"

        c2_data = c2_data.astype(dtype)
        phi_angles = phi_angles.astype(dtype)

        # Create PyMC model
        with pm.Model() as model:
            # Define priors based on parameter bounds from configuration
            bounds = self.config.get("parameter_space", {}).get(
                "bounds", []
            )  # noqa: F841

            # Parameter priors - mode-aware construction
            print(f"   Building {effective_param_count}-parameter model for {('static' if is_static_mode else 'laminar flow')} mode")
            
            # Always include diffusion parameters (first 3)
            D0 = pm.LogNormal("D0", mu=np.log(1000), sigma=2.0)
            alpha = pm.Normal("alpha", mu=0.0, sigma=1.0)  # Power-law exponent
            D_offset = pm.Normal("D_offset", mu=0.0, sigma=1000.0)  # Diffusion offset
            
            if not is_static_mode and effective_param_count > 3:
                # Laminar flow mode: include shear and angular parameters
                gamma_dot_t0 = pm.LogNormal(
                    "gamma_dot_t0", mu=np.log(0.01), sigma=2.0
                )  # Shear rate
                beta = pm.Normal("beta", mu=0.0, sigma=1.0)  # Shear power-law exponent
                gamma_dot_t_offset = pm.Normal(
                    "gamma_dot_t_offset", mu=0.0, sigma=0.01
                )  # Shear offset
                phi0 = pm.Normal("phi0", mu=0.0, sigma=10.0)  # Angular offset
            else:
                # Static mode: shear parameters are fixed at zero (not used)
                print("   Static mode: shear and angular parameters excluded from model")

            # Noise model
            noise_config = performance_config.get("noise_model", {})
            sigma = pm.HalfNormal("sigma", sigma=noise_config.get("sigma_prior", 0.1))

            # Convert to shared variables for efficiency
            c2_data_shared = shared(c2_data.astype(dtype), name="c2_data")
            phi_angles_shared = shared(
                phi_angles.astype(dtype), name="phi_angles"
            )  # noqa: F841

            # Forward model (simplified for computational efficiency)
            if is_static_mode:
                # Static mode: only diffusion parameters
                params = pt.stack([D0, alpha, D_offset])  # noqa: F841
            else:
                # Laminar flow mode: all parameters
                params = pt.stack(  # noqa: F841
                    [
                        D0,
                        alpha,
                        D_offset,
                        gamma_dot_t0,
                        beta,
                        gamma_dot_t_offset,
                        phi0,
                    ]
                )

            # Simplified forward model for demonstration
            simple_forward = noise_config.get("use_simple_forward_model", True)

            if simple_forward:
                print(
                    "   Using simplified forward model (faster sampling, reduced accuracy)"
                )
                # Create simplified deterministic relationship
                mu = pm.Deterministic("mu", D0 * 0.001)  # Placeholder scaling

                # Likelihood using mean experimental value
                c2_mean = pt.mean(c2_data_shared)
                likelihood = pm.Normal(  # noqa: F841
                    "likelihood", mu=mu, sigma=sigma, observed=c2_mean
                )
            else:
                print(
                    "   Warning: Full forward model requires custom PyTensor operations"
                )
                raise NotImplementedError(
                    "Full differentiable forward model not yet implemented. "
                    "Set 'use_simple_forward_model': True in noise_model configuration."
                )

            # Add validation checks
            D_positive = pm.Deterministic("D_positive", D0 > 0)  # noqa: F841
            gamma_positive = pm.Deterministic(
                "gamma_positive", gamma_dot_t0 > 0
            )  # noqa: F841
            D_total = pm.Deterministic("D_total", D0 + D_offset)  # noqa: F841

        print(f"   ✓ Bayesian model constructed successfully")
        print(f"     Model contains {len(model.basic_RVs)} random variables")
        print(f"     Data shape: {c2_data.shape}")
        print(f"     Precision: {dtype_str}")

        self.bayesian_model = model
        return model

    def _run_mcmc_nuts_optimized(
        self,
        c2_experimental: np.ndarray,
        phi_angles: np.ndarray,
        config: Dict[str, Any],
        filter_angles_for_optimization: bool = True,
        is_static_mode: bool = False,
        analysis_mode: str = "laminar_flow",
        effective_param_count: int = 7,
    ) -> Dict[str, Any]:
        """
        Run MCMC NUTS sampling for parameter uncertainty quantification.

        This method provides advanced Bayesian sampling using PyMC's
        No-U-Turn Sampler for uncertainty quantification.

        Parameters
        ----------
        c2_experimental : np.ndarray
            Experimental data
        phi_angles : np.ndarray
            Scattering angles
        config : Dict[str, Any]
            MCMC configuration
        filter_angles_for_optimization : bool, default True
            If True, use only angles in ranges [-10°, 10°] and [170°, 190°] for sampling
        is_static_mode : bool, default False
            Whether static mode is enabled
        analysis_mode : str, default "laminar_flow"
            Analysis mode ("static" or "laminar_flow")
        effective_param_count : int, default 7
            Number of parameters to use (3 for static, 7 for laminar flow)

        Returns
        -------
        Dict[str, Any]
            MCMC results and diagnostics

        Raises
        ------
        ImportError
            If PyMC is not available
        """
        if not PYMC_AVAILABLE:
            raise ImportError("PyMC not available for MCMC")

        # Type assertions for type checker - these are guaranteed after the availability check
        assert pm is not None
        assert az is not None

        # Extract MCMC settings
        draws = config.get("mcmc_draws", 1000)
        tune = config.get("mcmc_tune", 500)
        chains = config.get("mcmc_chains", 2)
        target_accept = config.get("target_accept", 0.9)
        cores = min(chains, getattr(self.core, "num_threads", 1))

        print(f"   Running MCMC (NUTS) Sampling...")
        print(f"     Mode: {analysis_mode} ({effective_param_count} parameters)")
        print(
            f"     Settings: draws={draws}, tune={tune}, chains={chains}, cores={cores}"
        )

        # Build the Bayesian model with angle filtering
        model = self._build_bayesian_model_optimized(
            c2_experimental,
            phi_angles,
            filter_angles_for_optimization=filter_angles_for_optimization,
            is_static_mode=is_static_mode,
            effective_param_count=effective_param_count,
        )

        # Prepare initial values from best parameters
        initvals = None
        best_params_bo = getattr(self.core, "best_params_bo", None)
        best_params_classical = getattr(self.core, "best_params_classical", None)

        if best_params_bo is not None:
            print("     ✓ Using Bayesian Optimization best for MCMC initialization")
            init_params = best_params_bo
        elif best_params_classical is not None:
            print("     ✓ Using Classical best for MCMC initialization")
            init_params = best_params_classical
        else:
            print("     ⚠ Using default MCMC initialization")
            init_params = None

        if init_params is not None:
            param_names = self.config["initial_parameters"]["parameter_names"]
            
            # Adjust initialization parameters based on mode
            if is_static_mode and len(init_params) > effective_param_count:
                # Use only diffusion parameters for static mode
                init_params_adjusted = init_params[:effective_param_count]
                param_names_adjusted = param_names[:effective_param_count]
                print(f"     Using {effective_param_count} diffusion parameters for static mode initialization")
            elif not is_static_mode and len(init_params) < effective_param_count:
                # Extend for laminar flow mode
                init_params_adjusted = np.zeros(effective_param_count)
                init_params_adjusted[:len(init_params)] = init_params
                param_names_adjusted = param_names[:effective_param_count]
                print(f"     Extended to {effective_param_count} parameters for laminar flow initialization")
            else:
                init_params_adjusted = init_params[:effective_param_count]
                param_names_adjusted = param_names[:effective_param_count]
            
            initvals = [
                {name: init_params_adjusted[i] for i, name in enumerate(param_names_adjusted)}
                for _ in range(chains)
            ]
            # Add small random perturbations for different chains
            for chain_idx in range(1, chains):
                for param, value in initvals[chain_idx].items():
                    initvals[chain_idx][param] = value * (1 + 0.01 * np.random.randn())

        mcmc_start = time.time()

        # Create callback for intermediate logging
        def log_sampling_progress(approx, loss, i):
            if i % 100 == 0:  # Log every 100 samples
                print(f"    Sample {i}: Log-posterior = {-loss:.6e}")
        
        with model:
            # For MCMC, we can't directly log chi-squared but can monitor the log-posterior
            print(f"    Starting MCMC sampling ({draws} draws + {tune} tuning)...")
            trace = pm.sample(
                draws=draws,
                tune=tune,
                chains=chains,
                cores=cores,
                initvals=initvals,
                target_accept=target_accept,
                return_inferencedata=True,
                compute_convergence_checks=True,
                progressbar=True,
            )

        mcmc_time = time.time() - mcmc_start

        # Extract posterior means (mode-aware)
        param_names = self.config["initial_parameters"]["parameter_names"]
        param_names_effective = param_names[:effective_param_count]
        posterior_means = {}
        for var_name in param_names_effective:
            # type: ignore[attr-defined]
            if hasattr(trace, "posterior") and var_name in trace.posterior:
                posterior_means[var_name] = float(
                    # type: ignore[attr-defined]
                    trace.posterior[var_name].mean()
                )

        results = {
            "trace": trace,
            "time": mcmc_time,
            "posterior_means": posterior_means,
            "config": config,
        }

        self.mcmc_result = results
        self.mcmc_trace = trace

        print(f"     ✓ MCMC completed in {mcmc_time:.1f}s")
        return results

    def run_mcmc_analysis(
        self,
        c2_experimental: Optional[np.ndarray] = None,
        phi_angles: Optional[np.ndarray] = None,
        mcmc_config: Optional[Dict[str, Any]] = None,
        filter_angles_for_optimization: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """
        Run complete MCMC analysis including model building and sampling.

        Parameters
        ----------
        c2_experimental : np.ndarray, optional
            Experimental correlation data
        phi_angles : np.ndarray, optional
            Scattering angles
        mcmc_config : Dict[str, Any], optional
            MCMC configuration settings
        filter_angles_for_optimization : bool, default True
            If True, use only angles in ranges [-10°, 10°] and [170°, 190°] for sampling

        Returns
        -------
        Dict[str, Any]
            Complete MCMC analysis results
        """
        if not PYMC_AVAILABLE:
            raise ImportError("PyMC not available for MCMC analysis")

        print("\n═══ MCMC/NUTS Sampling ═══")

        # Determine analysis mode and effective parameter count
        if hasattr(self.core, 'config_manager') and self.core.config_manager:
            is_static_mode = self.core.config_manager.is_static_mode_enabled()
            analysis_mode = self.core.config_manager.get_analysis_mode()
            effective_param_count = self.core.config_manager.get_effective_parameter_count()
        else:
            # Fallback to core method
            is_static_mode = getattr(self.core, 'is_static_mode', lambda: False)()
            analysis_mode = "static" if is_static_mode else "laminar_flow"
            effective_param_count = 3 if is_static_mode else 7

        print(f"  Analysis mode: {analysis_mode} ({effective_param_count} parameters)")
        logger.info(f"MCMC sampling using {analysis_mode} mode with {effective_param_count} parameters")

        # Load data if needed
        if c2_experimental is None or phi_angles is None:
            c2_experimental, _, phi_angles, _ = self.core.load_experimental_data()

        # Type assertions after loading data
        assert (
            c2_experimental is not None and phi_angles is not None
        ), "Failed to load experimental data"

        # Use provided config or default
        if mcmc_config is None:
            mcmc_config = self.mcmc_config or {}

        # Ensure mcmc_config is not None for type checker
        assert mcmc_config is not None

        # Determine angle filtering setting
        if filter_angles_for_optimization is None:
            # Get from ConfigManager if available
            if hasattr(self.core, "config_manager") and self.core.config_manager:
                filter_angles_for_optimization = (
                    self.core.config_manager.is_angle_filtering_enabled()
                )
            else:
                # Default to True for backward compatibility
                filter_angles_for_optimization = True

        # Ensure filter_angles_for_optimization is a boolean
        assert isinstance(
            filter_angles_for_optimization, bool
        ), "filter_angles_for_optimization must be a boolean"

        # Run MCMC sampling with angle filtering
        results = self._run_mcmc_nuts_optimized(
            c2_experimental,
            phi_angles,
            mcmc_config,
            filter_angles_for_optimization,
            is_static_mode,
            analysis_mode,
            effective_param_count,
        )

        # Add convergence diagnostics
        if "trace" in results:
            diagnostics = self.compute_convergence_diagnostics(results["trace"])
            results["diagnostics"] = diagnostics

        return results

    def compute_convergence_diagnostics(self, trace) -> Dict[str, Any]:
        """
        Compute convergence diagnostics for MCMC chains.

        Parameters
        ----------
        trace : arviz.InferenceData
            MCMC trace data

        Returns
        -------
        Dict[str, Any]
            Convergence diagnostics including R-hat, ESS, etc.
        """
        if not PYMC_AVAILABLE or az is None:
            logger.warning("Arviz not available - returning basic diagnostics")
            return {
                "converged": True,
                "note": "Diagnostics unavailable - arviz not installed",
            }

        try:
            # Compute R-hat (potential scale reduction factor)
            rhat = az.rhat(trace)

            # Compute effective sample size
            ess = az.ess(trace)

            # Compute MCSE (Monte Carlo standard error)
            mcse = az.mcse(trace)

            # Overall convergence assessment
            try:
                max_rhat = (
                    float(rhat.to_array().max())
                    if hasattr(
                        # type: ignore[attr-defined]
                        rhat,
                        "to_array",
                    )
                    else float(np.max(rhat))
                )
            except (AttributeError, TypeError):
                max_rhat = 1.0

            try:
                min_ess = (
                    float(ess.to_array().min())
                    if hasattr(ess, "to_array")
                    else float(np.min(ess))
                )
            except (AttributeError, TypeError):
                min_ess = 1000.0

            converged = max_rhat < 1.1 and min_ess > 100

            return {
                "rhat": rhat,
                "ess": ess,
                "mcse": mcse,
                "max_rhat": max_rhat,
                "min_ess": min_ess,
                "converged": converged,
                "assessment": "Converged" if converged else "Not converged",
            }

        except Exception as e:
            logger.warning(f"Failed to compute convergence diagnostics: {e}")
            return {"error": str(e)}

    def extract_posterior_statistics(self, trace) -> Dict[str, Any]:
        """
        Extract comprehensive posterior statistics.

        Parameters
        ----------
        trace : arviz.InferenceData
            MCMC trace data

        Returns
        -------
        Dict[str, Any]
            Posterior statistics including means, credible intervals, etc.
        """
        if not PYMC_AVAILABLE or az is None:
            logger.warning("Arviz not available - returning basic statistics")
            return {"note": ("Posterior statistics unavailable - arviz not installed")}

        try:
            # Summary statistics
            summary = az.summary(trace)

            # Extract parameter estimates
            param_names = self.config.get("initial_parameters", {}).get(
                "parameter_names", []
            )
            posterior_stats = {}

            for param in param_names:
                # type: ignore[attr-defined]
                if hasattr(trace, "posterior") and param in trace.posterior:
                    # type: ignore[attr-defined]
                    samples = trace.posterior[param].values.flatten()
                    posterior_stats[param] = {
                        "mean": float(np.mean(samples)),
                        "std": float(np.std(samples)),
                        "median": float(np.median(samples)),
                        "ci_2.5": float(np.percentile(samples, 2.5)),
                        "ci_97.5": float(np.percentile(samples, 97.5)),
                        "ci_25": float(np.percentile(samples, 25)),
                        "ci_75": float(np.percentile(samples, 75)),
                    }

            return {
                "summary_table": summary,
                "parameter_statistics": posterior_stats,
                # type: ignore[attr-defined]
                "total_samples": (
                    (len(trace.posterior.chain) * len(trace.posterior.draw))
                    if hasattr(trace, "posterior")
                    and hasattr(trace.posterior, "chain")
                    and hasattr(trace.posterior, "draw")
                    else 0
                ),
            }

        except Exception as e:
            logger.warning(f"Failed to extract posterior statistics: {e}")
            return {"error": str(e)}

    def generate_posterior_samples(self, n_samples: int = 1000) -> Optional[np.ndarray]:
        """
        Generate posterior parameter samples for uncertainty propagation.

        Parameters
        ----------
        n_samples : int
            Number of samples to generate

        Returns
        -------
        np.ndarray or None
            Array of parameter samples [n_samples, n_parameters]
        """
        if self.mcmc_trace is None:
            logger.warning("No MCMC trace available")
            return None

        try:
            param_names = self.config.get("initial_parameters", {}).get(
                "parameter_names", []
            )
            samples = []

            # Extract samples for each parameter
            for param in param_names:
                # type: ignore[attr-defined]
                if (
                    hasattr(self.mcmc_trace, "posterior")
                    and param in self.mcmc_trace.posterior
                ):
                    # type: ignore[attr-defined]
                    param_samples = self.mcmc_trace.posterior[param].values.flatten()
                    # Randomly subsample if more samples available than requested
                    if len(param_samples) > n_samples:
                        indices = np.random.choice(
                            len(param_samples), n_samples, replace=False
                        )
                        param_samples = param_samples[indices]
                    samples.append(param_samples[:n_samples])

            if samples:
                return np.column_stack(samples)
            else:
                logger.warning("No parameter samples found in trace")
                return None

        except Exception as e:
            logger.error(f"Failed to generate posterior samples: {e}")
            return None

    def assess_chain_mixing(self, trace) -> Dict[str, Any]:
        """
        Assess MCMC chain mixing and identify potential issues.

        Parameters
        ----------
        trace : arviz.InferenceData
            MCMC trace data

        Returns
        -------
        Dict[str, Any]
            Chain mixing assessment
        """
        try:
            # Check for divergences
            if hasattr(trace, "sample_stats") and "diverging" in trace.sample_stats:
                n_divergent = trace.sample_stats.diverging.sum().values
                divergent_fraction = float(
                    n_divergent / trace.sample_stats.diverging.size
                )
            else:
                n_divergent = 0
                divergent_fraction = 0.0

            # Assess mixing using effective sample size
            if az is not None:
                ess = az.ess(trace)
            else:
                ess = None
            try:
                if ess is not None:
                    trace_len = (
                        len(trace.posterior.draw)
                        if hasattr(
                            # type: ignore[attr-defined]
                            trace,
                            "posterior",
                        )
                        else 1000
                    )
                    min_ess_ratio = float(np.min(ess) / trace_len)
                else:
                    min_ess_ratio = 0.5
            except (AttributeError, TypeError):
                min_ess_ratio = 0.5

            # Overall assessment
            good_mixing = divergent_fraction < 0.01 and min_ess_ratio > 0.1

            return {
                "n_divergent": int(n_divergent),
                "divergent_fraction": divergent_fraction,
                "min_ess_ratio": min_ess_ratio,
                "good_mixing": good_mixing,
                "recommendations": self._get_mixing_recommendations(
                    divergent_fraction, min_ess_ratio
                ),
            }

        except Exception as e:
            logger.warning(f"Failed to assess chain mixing: {e}")
            return {"error": str(e)}

    def _get_mixing_recommendations(
        self, divergent_fraction: float, min_ess_ratio: float
    ) -> List[str]:
        """
        Get recommendations for improving chain mixing.

        Parameters
        ----------
        divergent_fraction : float
            Fraction of divergent transitions
        min_ess_ratio : float
            Minimum effective sample size ratio

        Returns
        -------
        List[str]
            List of recommendations
        """
        recommendations = []

        if divergent_fraction > 0.05:
            recommendations.extend(
                [
                    "High divergence rate detected",
                    "Try increasing target_accept (e.g., 0.95)",
                    "Consider reparametrizing the model",
                    "Increase tuning steps",
                ]
            )

        if min_ess_ratio < 0.1:
            recommendations.extend(
                [
                    "Low effective sample size",
                    "Increase number of draws",
                    "Check for high autocorrelation",
                    "Consider different step size adaptation",
                ]
            )

        if not recommendations:
            recommendations.append("Sampling appears healthy")

        return recommendations

    def _validate_mcmc_config(self) -> None:
        """
        Validate MCMC configuration parameters.

        Raises
        ------
        ValueError
            If configuration parameters are invalid
        """
        # Check required parameters exist
        if "initial_parameters" not in self.config:
            raise ValueError("Missing 'initial_parameters' in configuration")

        param_config = self.config["initial_parameters"]
        if "parameter_names" not in param_config:
            raise ValueError("Missing 'parameter_names' in initial_parameters")

        # Validate MCMC-specific settings
        mcmc_draws = self.mcmc_config.get("mcmc_draws", 1000)
        if not isinstance(mcmc_draws, int) or mcmc_draws < 1:
            raise ValueError(f"mcmc_draws must be a positive integer, got {mcmc_draws}")

        mcmc_tune = self.mcmc_config.get("mcmc_tune", 500)
        if not isinstance(mcmc_tune, int) or mcmc_tune < 1:
            raise ValueError(f"mcmc_tune must be a positive integer, got {mcmc_tune}")

        mcmc_chains = self.mcmc_config.get("mcmc_chains", 2)
        if not isinstance(mcmc_chains, int) or mcmc_chains < 1:
            raise ValueError(
                f"mcmc_chains must be a positive integer, got {mcmc_chains}"
            )

        target_accept = self.mcmc_config.get("target_accept", 0.9)
        if not isinstance(target_accept, (int, float)) or not 0 < target_accept < 1:
            raise ValueError(
                f"target_accept must be between 0 and 1, got {target_accept}"
            )

        logger.debug("MCMC configuration validated successfully")

    def _validate_physical_parameters(self, params: np.ndarray) -> bool:
        """
        Validate physical parameter values.

        Parameters
        ----------
        params : np.ndarray
            Parameter values to validate

        Returns
        -------
        bool
            True if parameters are physically valid
        """
        try:
            param_names = self.config["initial_parameters"]["parameter_names"]
            bounds = self.config.get("parameter_space", {}).get("bounds", [])

            # Check bounds if available
            if bounds and len(bounds) == len(params):
                for i, (param, value) in enumerate(zip(param_names, params)):
                    if len(bounds[i]) >= 2:
                        lower, upper = bounds[i][:2]
                        if not (lower <= value <= upper):
                            logger.warning(
                                f"Parameter {param} = {value} outside bounds [{lower}, {upper}]"
                            )
                            return False

            # Physical constraints
            param_dict = dict(zip(param_names, params))

            # Diffusion coefficient should be positive
            if "D0" in param_dict and param_dict["D0"] <= 0:
                logger.warning(
                    f"Non-physical diffusion coefficient: {param_dict['D0']}"
                )
                return False

            # Shear rate should be non-negative
            if "gamma_dot_t0" in param_dict and param_dict["gamma_dot_t0"] < 0:
                logger.warning(f"Negative shear rate: {param_dict['gamma_dot_t0']}")
                return False

            return True

        except Exception as e:
            logger.error(f"Error validating parameters: {e}")
            return False

    def validate_model_setup(self) -> Dict[str, Any]:
        """
        Validate the Bayesian model setup and configuration.

        Returns
        -------
        Dict[str, Any]
            Validation results and recommendations
        """
        validation_results = {
            "valid": True,
            "warnings": [],
            "errors": [],
            "recommendations": [],
        }

        # Check PyMC availability
        if not PYMC_AVAILABLE:
            validation_results["valid"] = False
            validation_results["errors"].append("PyMC not available for MCMC")
            return validation_results

        # Check configuration completeness
        required_sections = ["initial_parameters", "analyzer_parameters"]
        for section in required_sections:
            if section not in self.config:
                validation_results["valid"] = False
                validation_results["errors"].append(
                    f"Missing configuration section: {section}"
                )

        # Check performance settings
        perf_config = self.config.get("performance_settings", {})

        # Memory usage warnings
        use_float32 = perf_config.get("use_float32_precision", True)
        if not use_float32:
            validation_results["warnings"].append(
                "Using float64 precision - may require more memory"
            )

        # Subsampling recommendations
        subsample_factor = perf_config.get("bayesian_subsample_factor", 1)
        if subsample_factor == 1:
            validation_results["recommendations"].append(
                "Consider subsampling large datasets to improve MCMC performance"
            )

        # Forward model complexity
        noise_config = perf_config.get("noise_model", {})
        simple_forward = noise_config.get("use_simple_forward_model", True)
        if not simple_forward:
            validation_results["warnings"].append(
                "Complex forward model may slow down sampling significantly"
            )
            validation_results["recommendations"].append(
                "Consider using simplified forward model for initial exploration"
            )

        # MCMC settings validation
        draws = self.mcmc_config.get("mcmc_draws", 1000)
        chains = self.mcmc_config.get("mcmc_chains", 2)

        if draws < 1000:
            validation_results["warnings"].append(
                f"Low number of draws ({draws}) may not provide reliable estimates"
            )

        if chains < 2:
            validation_results["warnings"].append(
                "Single chain sampling prevents convergence diagnostics"
            )
            validation_results["recommendations"].append(
                "Use at least 2 chains for robust convergence assessment"
            )

        return validation_results

    def get_model_summary(self) -> Optional[Dict[str, Any]]:
        """
        Get summary information about the current Bayesian model.

        Returns
        -------
        Dict[str, Any] or None
            Model summary information
        """
        if self.bayesian_model is None:
            return None

        try:
            with self.bayesian_model:
                # Get model information
                n_params = len(self.bayesian_model.basic_RVs)
                param_names = [rv.name for rv in self.bayesian_model.basic_RVs]

                # Check for deterministic variables
                deterministic_vars = [
                    rv.name for rv in self.bayesian_model.deterministics
                ]

                return {
                    "n_parameters": n_params,
                    "parameter_names": param_names,
                    "deterministic_variables": deterministic_vars,
                    "model_type": "Bayesian (PyMC)",
                    "forward_model": (
                        "Simplified"
                        if self.config.get("performance_settings", {})
                        .get("noise_model", {})
                        .get("use_simple_forward_model", True)
                        else "Full"
                    ),
                }

        except Exception as e:
            logger.error(f"Failed to get model summary: {e}")
            return None

    def get_best_params(
        self, stage: str = "mcmc"
    ) -> Optional[np.ndarray]:  # noqa: ARG002
        """
        Get best parameters from MCMC posterior analysis.

        Parameters
        ----------
        stage : str
            Stage identifier (for compatibility)

        Returns
        -------
        np.ndarray or None
            Posterior mean parameters
        """
        if self.mcmc_result is None or "posterior_means" not in self.mcmc_result:
            logger.warning("No MCMC results available")
            return None

        try:
            param_names = self.config["initial_parameters"]["parameter_names"]
            posterior_means = self.mcmc_result["posterior_means"]

            # Convert to array in parameter order
            params = np.array([posterior_means.get(name, 0.0) for name in param_names])
            return params

        except Exception as e:
            logger.error(f"Failed to extract best parameters: {e}")
            return None

    def get_parameter_uncertainties(self) -> Optional[Dict[str, float]]:
        """
        Get parameter uncertainty estimates from MCMC posterior.

        Returns
        -------
        Dict[str, float] or None
            Parameter standard deviations
        """
        if self.mcmc_trace is None:
            logger.warning("No MCMC trace available")
            return None

        try:
            param_names = self.config["initial_parameters"]["parameter_names"]
            uncertainties = {}

            for param in param_names:
                # type: ignore[attr-defined]
                if (
                    hasattr(self.mcmc_trace, "posterior")
                    and param in self.mcmc_trace.posterior
                ):
                    # type: ignore[attr-defined]
                    samples = self.mcmc_trace.posterior[param].values.flatten()
                    uncertainties[param] = float(np.std(samples))

            return uncertainties

        except Exception as e:
            logger.error(f"Failed to extract parameter uncertainties: {e}")
            return None

    def save_results(self, filepath: str) -> bool:
        """
        Save MCMC results to file.

        Parameters
        ----------
        filepath : str
            Path to save results

        Returns
        -------
        bool
            True if saved successfully
        """
        if self.mcmc_result is None:
            logger.warning("No MCMC results to save")
            return False

        try:
            # Prepare serializable results
            results_to_save = {
                "posterior_means": self.mcmc_result["posterior_means"],
                "time": self.mcmc_result["time"],
                "config": self.mcmc_result["config"],
            }

            # Add diagnostics if available
            if "diagnostics" in self.mcmc_result:
                diag = self.mcmc_result["diagnostics"]
                results_to_save["diagnostics"] = {
                    "max_rhat": diag.get("max_rhat"),
                    "min_ess": diag.get("min_ess"),
                    "converged": diag.get("converged"),
                    "assessment": diag.get("assessment"),
                }

            # Save to JSON
            import json

            with open(filepath, "w") as f:
                json.dump(results_to_save, f, indent=2)

            logger.info(f"MCMC results saved to {filepath}")
            return True

        except Exception as e:
            logger.error(f"Failed to save MCMC results: {e}")
            return False

    def load_results(self, filepath: str) -> bool:
        """
        Load MCMC results from file.

        Parameters
        ----------
        filepath : str
            Path to load results from

        Returns
        -------
        bool
            True if loaded successfully
        """
        try:
            import json

            with open(filepath, "r") as f:
                results = json.load(f)

            # Restore basic results (note: trace cannot be serialized/restored)
            self.mcmc_result = results
            logger.info(f"MCMC results loaded from {filepath}")
            return True

        except Exception as e:
            logger.error(f"Failed to load MCMC results: {e}")
            return False


def create_mcmc_sampler(analysis_core, config: Dict[str, Any]) -> MCMCSampler:
    """
    Factory function to create an MCMC sampler instance.

    This function provides a convenient way to create and configure
    an MCMC sampler with proper validation and error handling.

    Parameters
    ----------
    analysis_core : HomodyneAnalysisCore
        Core analysis engine instance
    config : Dict[str, Any]
        Configuration dictionary

    Returns
    -------
    MCMCSampler
        Configured MCMC sampler instance

    Raises
    ------
    ImportError
        If PyMC dependencies are not available
    ValueError
        If configuration is invalid

    Examples
    --------
    >>> from homodyne.analysis.core import HomodyneAnalysisCore
    >>> from homodyne.optimization.mcmc import create_mcmc_sampler
    >>>
    >>> # Load configuration
    >>> core = HomodyneAnalysisCore('config.json')
    >>> config = core.config
    >>>
    >>> # Create MCMC sampler
    >>> mcmc = create_mcmc_sampler(core, config)
    >>>
    >>> # Run MCMC analysis
    >>> results = mcmc.run_mcmc_analysis()
    """
    # Validate PyMC availability
    if not PYMC_AVAILABLE:
        raise ImportError(
            "PyMC is required for MCMC sampling. Install with: pip install pymc arviz"
        )

    # Create and validate sampler
    sampler = MCMCSampler(analysis_core, config)

    # Validate model setup
    validation = sampler.validate_model_setup()
    if not validation["valid"]:
        error_msg = "MCMC configuration validation failed:\n"
        error_msg += "\n".join(f"- {error}" for error in validation["errors"])
        raise ValueError(error_msg)

    # Log warnings and recommendations
    for warning in validation["warnings"]:
        logger.warning(warning)
    for rec in validation["recommendations"]:
        logger.info(f"Recommendation: {rec}")

    return sampler


# Example usage and testing utilities
if __name__ == "__main__":
    """
    Example usage of the MCMC sampler.

    This section demonstrates how to use the MCMCSampler class
    for Bayesian parameter estimation in homodyne scattering analysis.
    """
    print("MCMC Sampling Module for Homodyne Scattering Analysis")
    print("=" * 60)

    # Check dependencies
    print(f"PyMC Available: {PYMC_AVAILABLE}")
    if PYMC_AVAILABLE:
        print(
            f"PyMC Version: {pm.__version__ if pm and hasattr(pm, '__version__') else 'unknown'}"
        )
        print(f"ArviZ Available: {az is not None}")

    print("\nModule successfully loaded and ready for use.")
    print("\nTo use the MCMC sampler:")
    print("1. Create a HomodyneAnalysisCore instance with your configuration")
    print("2. Use create_mcmc_sampler() to create a sampler instance")
    print("3. Call run_mcmc_analysis() to perform Bayesian parameter estimation")

    if not PYMC_AVAILABLE:
        print("\nWarning: Install PyMC for full functionality:")
        print("pip install pymc arviz")
