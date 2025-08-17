# Scaling Optimization in Homodyne XPCS Analysis

## Overview

**Scaling optimization is now always enabled** in all homodyne XPCS analysis modes. This document explains the mathematical foundation, physical significance, and implementation details.

## Mathematical Relationship

The scaling optimization determines the optimal relationship between experimental and theoretical correlation functions:

```
g₂ = offset + contrast × g₁
```

Where:
- **g₁**: Theoretical correlation function
- **g₂**: Experimental correlation function  
- **contrast**: Fitted scaling parameter (multiplicative factor)
- **offset**: Fitted baseline parameter (additive factor)

## Physical Significance

### Purpose
Scaling optimization accounts for systematic scaling factors that are unavoidably present in experimental data due to:

- **Instrumental response functions**: Detector and optical system responses
- **Background signals**: Electronic noise, scattered light, dark current
- **Detector gain variations**: Pixel-to-pixel sensitivity differences
- **Normalization differences**: Systematic differences in data normalization
- **Systematic measurement offsets**: Baseline shifts in correlation measurements

### Why Always Enabled
Without scaling optimization, direct comparison of raw theoretical values to experimental data would:
1. Ignore systematic scaling factors present in all real measurements
2. Produce meaningless chi-squared values
3. Lead to poor fitting quality assessment
4. Create inconsistencies between different optimization methods

## Mathematical Implementation

### Least Squares Solution
The optimal contrast and offset parameters are determined by solving:

```
A·x = b
```

Where:
- **A** = [theory, ones] (matrix with theory values and column of ones)
- **x** = [contrast, offset] (parameters to solve for)
- **b** = experimental (experimental data vector)

### Code Implementation
```python
A = np.vstack([theory, np.ones(len(theory))]).T
scaling, residuals, _, _ = np.linalg.lstsq(A, exp, rcond=None)
contrast, offset = scaling
fitted = theory * contrast + offset
```

## Chi-Squared Calculation

The chi-squared statistic is calculated using the fitted data:

```
χ² = Σ(experimental - fitted)²/σ²
```

Where:
- **fitted** = theory × contrast + offset (from optimal scaling)
- **σ²**: Measurement uncertainty variance

This formulation provides meaningful fitting quality assessment regardless of:
- Analysis mode (static vs. flow)
- Number of scattering angles
- Isotropic vs. anisotropic conditions

## Implementation Across Modules

### Core Analysis (`homodyne/analysis/core.py`)
- Lines 1159-1182: Main scaling optimization implementation
- Always performed for each scattering angle independently
- Integrated into chi-squared calculation pipeline

### Plotting (`homodyne/plotting.py`)
- Lines 191-204: Scaling for visualization consistency
- Ensures meaningful residual plots (exp - fitted)
- Maintains consistency with analysis methodology

### MCMC (`homodyne/optimization/mcmc.py`)
- Lines 296-302: Documentation of always-enabled scaling
- Ensures MCMC results comparable to classical optimization
- Fundamental to proper uncertainty quantification

## Configuration Changes

### Before (Removed)
```json
"chi_squared_calculation": {
    "scaling_optimization": true/false
}
```

### After (Always Enabled)
```json
"chi_squared_calculation": {
    "_scaling_optimization_note": "Scaling optimization is always enabled: g₂ = offset + contrast × g₁",
    "_scaling_details": {
        "_purpose": "Performs least squares fitting to determine contrast and offset parameters",
        "_accounts_for": ["instrumental response", "background signals", "detector variations", "normalization differences"],
        "_chi_squared_formula": "χ² = Σ(experimental - fitted)²/σ² where fitted = theory × contrast + offset",
        "_implementation": "Uses numpy.linalg.lstsq to solve A·x = b where A = [theory, ones], x = [contrast, offset]"
    }
}
```

## Benefits

### Scientific Accuracy
- Proper accounting for systematic experimental factors
- Meaningful chi-squared statistics for model validation
- Consistent results across different analysis approaches

### Methodological Consistency  
- Same scaling treatment in classical, MCMC, and Bayesian optimization
- Comparable results between different optimization methods
- Standardized approach across all analysis modes

### Practical Reliability
- Robust fitting quality assessment
- Reduced sensitivity to instrumental variations
- More reliable parameter uncertainty estimates

## Migration Notes

### For Users
- **No action required**: Scaling optimization now happens automatically
- **Configuration files**: Remove any `scaling_optimization` settings
- **Results**: Expect more consistent and physically meaningful fitting statistics

### For Developers
- **Code changes**: Remove conditional scaling logic 
- **Tests**: Update to expect scaling always enabled
- **Documentation**: Refer to this document for implementation details

## Technical Validation

All existing tests pass with scaling optimization always enabled:
- ✅ Isotropic mode integration tests
- ✅ MCMC scaling consistency tests  
- ✅ Plotting functionality tests
- ✅ Static mode analysis tests

## References

This implementation ensures the homodyne XPCS analysis provides physically meaningful results by properly accounting for the systematic scaling relationship between experimental measurements and theoretical predictions, as is standard practice in quantitative X-ray scattering analysis.