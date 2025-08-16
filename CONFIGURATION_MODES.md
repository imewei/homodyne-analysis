# Homodyne Analysis Configuration Modes

This document describes the three analysis modes available in the homodyne scattering analysis package and their corresponding configuration templates.

## Analysis Modes Overview

The homodyne analysis package supports three distinct analysis modes, each optimized for different experimental scenarios:

### 1. Static Isotropic Mode (`static_isotropic`)
**Use Case**: Static samples with isotropic scattering where results don't depend on scattering angle.

**Key Features**:
- No flow effects (γ̇ = 0, β = 0, γ̇_offset = 0, φ₀ = 0)
- Isotropic scattering - no angle selection needed
- `phi_angles_file` is NOT loaded - uses single dummy angle
- Angle filtering is automatically DISABLED
- Only 3 active parameters: `D0`, `alpha`, `D_offset`
- Faster optimization due to reduced parameter space

**Model Equation**:
```
g₁(t₁,t₂) = exp(-q² ∫ᵗ²ᵗ¹ D(t)dt) = g₁_diff(t₁,t₂)
g₂(t₁,t₂) = 1 + contrast × [g₁(t₁,t₂)]²
```
No angular dependence - results are identical for all scattering angles.

### 2. Static Anisotropic Mode (`static_anisotropic`)  
**Use Case**: Static samples where angle-dependent optimization efficiency is desired.

**Key Features**:
- No flow effects (γ̇ = 0, β = 0, γ̇_offset = 0, φ₀ = 0)
- Anisotropic scattering - uses angle selection
- `phi_angles_file` IS loaded for angle information  
- Angle filtering is ENABLED for optimization efficiency
- Only 3 active parameters: `D0`, `alpha`, `D_offset`
- Per-angle scaling optimization for better fit quality

**Model Equation**: Same as isotropic mode, but with angle filtering to focus optimization on specific angular ranges.

### 3. Laminar Flow Mode (`laminar_flow`)
**Use Case**: Samples under controlled flow or shear conditions with full physics model.

**Key Features**:
- All flow and diffusion effects included
- `phi_angles_file` IS REQUIRED for angle-dependent flow effects
- Angle filtering RECOMMENDED for computational efficiency
- All 7 parameters active: `D0`, `alpha`, `D_offset`, `gamma_dot_t0`, `beta`, `gamma_dot_t_offset`, `phi0`
- Complex parameter space with potential correlations
- MCMC analysis valuable for uncertainty quantification

**Model Equation**:
```
g₁(t₁,t₂) = g₁_diff(t₁,t₂) × g₁_shear(t₁,t₂)
where:
  g₁_diff = exp(-q² ∫ᵗ²ᵗ¹ D(t)dt)  [diffusion]
  g₁_shear = sinc²(Φ)  [flow-induced decorrelation]
  Φ = (qh/2π)cos(φ₀-φ) ∫|t₂-t₁| γ̇(t')dt'
```

## Configuration Templates

### Quick Start - Choose Your Template

1. **Static Isotropic Analysis**: Use `config_static_isotropic.json`
   ```bash
   cp homodyne/config_static_isotropic.json my_isotropic_config.json
   ```

2. **Static Anisotropic Analysis**: Use `config_static_anisotropic.json`
   ```bash
   cp homodyne/config_static_anisotropic.json my_anisotropic_config.json
   ```

3. **Laminar Flow Analysis**: Use `config_laminar_flow.json`
   ```bash
   cp homodyne/config_laminar_flow.json my_flow_config.json
   ```

### Configuration Settings

To specify the analysis mode in your configuration file:

```json
{
  "analysis_settings": {
    "static_mode": true/false,
    "static_submode": "isotropic" | "anisotropic" | null
  }
}
```

**Mode Selection Logic**:
- `static_mode: false` → **Laminar Flow Mode**
- `static_mode: true, static_submode: "isotropic"` → **Static Isotropic Mode**  
- `static_mode: true, static_submode: "anisotropic"` → **Static Anisotropic Mode**
- `static_mode: true, static_submode: null` → **Static Anisotropic Mode** (default)

### Key Configuration Differences

| Feature | Static Isotropic | Static Anisotropic | Laminar Flow |
|---------|-----------------|-------------------|--------------|
| **phi_angles_file loading** | ❌ Skipped | ✅ Loaded | ✅ Required |
| **Angle filtering** | ❌ Auto-disabled | ✅ Enabled | ✅ Recommended |
| **Active parameters** | 3 (D0, α, D_offset) | 3 (D0, α, D_offset) | 7 (all parameters) |
| **Scaling optimization** | ❌ Disabled | ✅ Enabled | ✅ Enabled |
| **MCMC draws (suggested)** | 8,000 | 8,000 | 10,000+ |
| **Optimization iterations** | 3,000 | 3,000 | 5,000 |

## Performance Considerations

### Static Isotropic Mode
- **Fastest**: Single angle, 3 parameters, no angle loading
- **Memory**: Minimal - single angle data
- **CPU**: Lowest computational cost
- **Recommended for**: Quick analysis, isotropic systems

### Static Anisotropic Mode  
- **Fast**: 3 parameters with angle filtering efficiency
- **Memory**: Standard - multi-angle data with filtering
- **CPU**: Moderate - angle filtering reduces data points
- **Recommended for**: Static analysis with optimization focus

### Laminar Flow Mode
- **Slowest**: 7 parameters, full angle coverage
- **Memory**: Highest - all angle data
- **CPU**: Highest - complex 7-parameter optimization
- **Recommended for**: Complete flow analysis, when computational resources allow

## Usage Examples

### Example 1: Static Isotropic Analysis
```python
from homodyne.analysis.core import HomodyneAnalysisCore

# Use isotropic configuration
core = HomodyneAnalysisCore("my_isotropic_config.json")

# Verify mode
print(f"Analysis mode: {core.config_manager.get_analysis_mode()}")
# Output: Analysis mode: static_isotropic

print(f"Active parameters: {core.get_effective_parameter_count()}")  
# Output: Active parameters: 3

print(f"Angle filtering: {core.config_manager.is_angle_filtering_enabled()}")
# Output: Angle filtering: False
```

### Example 2: Mode Comparison
```python
# Different modes for the same data
isotropic_core = HomodyneAnalysisCore("config_static_isotropic.json")
anisotropic_core = HomodyneAnalysisCore("config_static_anisotropic.json") 
flow_core = HomodyneAnalysisCore("config_laminar_flow.json")

# Compare parameters optimized
print(f"Isotropic: {isotropic_core.get_effective_parameter_count()} parameters")
print(f"Anisotropic: {anisotropic_core.get_effective_parameter_count()} parameters")
print(f"Flow: {flow_core.get_effective_parameter_count()} parameters")

# Output:
# Isotropic: 3 parameters
# Anisotropic: 3 parameters  
# Flow: 7 parameters
```

### Example 3: Configuration Override
```python
# Runtime mode switching
config_override = {
    "analysis_settings": {
        "static_mode": True,
        "static_submode": "isotropic"
    }
}

core = HomodyneAnalysisCore("config_template.json", config_override)
print(f"Mode: {core.config_manager.get_analysis_mode()}")
# Output: Mode: static_isotropic
```

## Migration Guide

### From Legacy Static Mode
If you have existing configurations with just `"static_mode": true`:

**Before** (legacy):
```json
{
  "analysis_settings": {
    "static_mode": true
  }
}
```

**After** (explicit):
```json
{
  "analysis_settings": {
    "static_mode": true,
    "static_submode": "anisotropic"  // or "isotropic"
  }
}
```

**Backward Compatibility**: Legacy configurations automatically default to `"anisotropic"` mode.

### Choosing the Right Mode

**Use Static Isotropic Mode when**:
- Sample is static (no applied flow or rotation)
- Scattering is isotropic (no angular anisotropy expected)
- You want fastest possible analysis
- Focus is only on diffusion dynamics D(t)

**Use Static Anisotropic Mode when**:
- Sample is static but you want angle filtering optimization
- You have limited computational resources but want some angle selection
- Backward compatibility with existing analysis workflows

**Use Laminar Flow Mode when**:
- Sample is under controlled shear or flow
- Both diffusion and flow dynamics are important  
- You have sufficient computational resources for 7-parameter fitting
- Angular dependence due to flow is measurable

## Troubleshooting

### Common Issues

**Issue**: "Angle filtering enabled but static_isotropic mode detected"
**Solution**: This is expected - angle filtering is automatically disabled in isotropic mode regardless of configuration.

**Issue**: "phi_angles_file not found" in static isotropic mode
**Solution**: This is expected - phi_angles_file is not loaded in isotropic mode. A dummy angle is used automatically.

**Issue**: Slow optimization in laminar flow mode
**Solution**: Enable angle filtering to reduce computational cost by 3-5x with minimal accuracy loss.

**Issue**: MCMC convergence problems with 7 parameters
**Solution**: 
- Increase tuning steps (`tune: 2000+`)
- Use better initial parameter estimates from classical optimization
- Increase target acceptance rate (`target_accept: 0.95`)

### Performance Optimization

**For Large Datasets**:
- Use isotropic mode when applicable (fastest)
- Enable angle filtering in anisotropic/flow modes
- Consider `data_type: "float32"` to reduce memory usage
- Reduce plotting DPI for faster visualization

**For Limited Resources**:
- Start with static isotropic mode to verify basic functionality
- Use anisotropic mode with angle filtering before full flow analysis
- Limit MCMC draws for initial testing

## Advanced Topics

### Custom Mode Detection
```python
from homodyne.core.config import ConfigManager

manager = ConfigManager("my_config.json")

# Detailed mode information
print(f"Static mode enabled: {manager.is_static_mode_enabled()}")
print(f"Static submode: {manager.get_static_submode()}")
print(f"Is isotropic: {manager.is_static_isotropic_enabled()}")
print(f"Is anisotropic: {manager.is_static_anisotropic_enabled()}")
print(f"Analysis mode: {manager.get_analysis_mode()}")
```

### Parameter Processing
```python
import numpy as np

# Example parameters [D0, α, D_offset, γ̇₀, β, γ̇_offset, φ₀]
params = np.array([1000.0, -0.1, 100.0, 0.01, -0.2, 0.001, 10.0])

# Static modes automatically zero flow parameters
effective_params = core.get_effective_parameters(params)
print(f"Effective parameters: {effective_params}")

# Static isotropic/anisotropic output: [1000.0, -0.1, 100.0, 0.0, 0.0, 0.0, 0.0]
# Laminar flow output: [1000.0, -0.1, 100.0, 0.01, -0.2, 0.001, 10.0]
```

### Testing Mode Functionality
```python
# Verify mode is working correctly
def test_analysis_mode(config_file):
    core = HomodyneAnalysisCore(config_file)
    mode = core.config_manager.get_analysis_mode()
    
    if mode == "static_isotropic":
        assert not core.config_manager.is_angle_filtering_enabled()
        assert core.get_effective_parameter_count() == 3
        print("✓ Static isotropic mode configured correctly")
        
    elif mode == "static_anisotropic":
        assert core.config_manager.is_angle_filtering_enabled()
        assert core.get_effective_parameter_count() == 3
        print("✓ Static anisotropic mode configured correctly")
        
    elif mode == "laminar_flow":
        assert core.get_effective_parameter_count() == 7
        print("✓ Laminar flow mode configured correctly")
        
    return mode

# Test your configuration
mode = test_analysis_mode("my_config.json")
print(f"Detected mode: {mode}")
```

---

For more detailed information about specific configuration parameters, see the individual template files:
- `config_static_isotropic.json` - Detailed isotropic mode settings
- `config_static_anisotropic.json` - Detailed anisotropic mode settings  
- `config_laminar_flow.json` - Detailed flow mode settings
- `config_template.json` - Comprehensive template with all options
