# MCMC Parameter Bounds Fix

## Issue Description

The MCMC sampler was failing during initialization with the error:
```
Initial evaluation of model at starting point failed!
```

This occurred when PyMC attempted to initialize MCMC sampling with the alpha parameter, which was set to its maximum boundary value (-1.5) with very narrow bounds [-1.6, -1.5].

## Root Cause Analysis

The problem arose from:

1. **Narrow parameter bounds**: Alpha had a width of only 0.1 units
2. **Boundary initialization**: Initial value was exactly at the upper bound (-1.5)
3. **PyMC jittering**: No room for the sampler to apply small perturbations during initialization
4. **Numerical instability**: This led to infinite/NaN values during model evaluation

## Solution Applied

### Configuration Changes

1. **Widened alpha bounds**: Changed from [-1.6, -1.5] to [-1.8, -1.2]
   - Increased width from 0.1 to 0.6 units (6x improvement)
   - Provides much more parameter space for exploration

2. **Centered initial value**: The initial value -1.5 is now perfectly centered
   - Distance from both boundaries: 0.3 units
   - Gives PyMC ample room for initialization jittering

3. **Fixed forward model**: Changed `use_simple_forward_model` from `true` to `false`
   - MCMC now ALWAYS uses scaling optimization (g₂ = offset + contrast × g₁)
   - Ensures consistent results with classical/Bayesian optimization methods
   - Eliminates "simplified model without scaling optimization" warnings

4. **Updated documentation**: Added explanatory notes in configuration

### File Changes

- `my_config.json`: Updated alpha parameter bounds and added documentation
- Added comprehensive regression test suite in `test_mcmc_parameter_bounds_regression.py`

### Validation

The fix was validated by:
1. Successfully running MCMC sampling without initialization errors
2. Verifying PyMC initialization logs show proper parameter jittering
3. Confirming sampling proceeds without infinite/NaN values
4. Creating comprehensive regression tests to prevent future issues

## Impact

- ✅ MCMC initialization now succeeds consistently
- ✅ Sampling starts properly with jitter+adapt_diag strategy  
- ✅ No more "Initial evaluation failed" errors
- ✅ Parameter exploration has adequate space for convergence
- ✅ MCMC uses full forward model with scaling optimization
- ✅ Consistent results across MCMC, classical, and Bayesian methods
- ✅ Regression tests ensure the fix remains stable

## Technical Details

### Parameter Space Comparison

**Before (Problematic)**:
- Bounds: [-1.6, -1.5], width = 0.1
- Initial: -1.5 (at max boundary)
- Jittering room: 0 above, 0.1 below

**After (Fixed)**:
- Bounds: [-1.8, -1.2], width = 0.6  
- Initial: -1.5 (perfectly centered)
- Jittering room: 0.3 above, 0.3 below

### PyMC Behavior

PyMC's NUTS sampler requires some parameter space around initial values for:
- Adaptive step size tuning
- Initial chain positioning
- Gradient-based exploration
- Numerical stability during warmup

The fix ensures these requirements are met for robust MCMC initialization.

### Forward Model Consistency

**Before (Inconsistent)**:
- MCMC used simplified forward model without scaling optimization
- Classical methods used full forward model with scaling (g₂ = offset + contrast × g₁)
- Results were not directly comparable between methods
- Warning messages about simplified model usage

**After (Consistent)**:
- All methods (MCMC, classical, Bayesian) use full forward model
- Scaling optimization accounts for instrumental response and background
- Results are directly comparable across optimization methods
- No more warnings about model inconsistencies

This ensures that MCMC provides results that can be meaningfully compared with classical optimization techniques.
