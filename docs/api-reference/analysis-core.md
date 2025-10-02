# `homodyne.analysis.core` - Core Analysis Engine

## Module Overview

The `homodyne.analysis.core` module provides the main analysis engine for homodyne
scattering in X-ray Photon Correlation Spectroscopy (XPCS). It implements the
theoretical framework from He et al. (2024) for analyzing nonequilibrium dynamics in
flowing soft matter systems.

**Module**: `homodyne.analysis.core` **Main Class**: `HomodyneAnalysisCore` **Authors**:
Wei Chen, Hongrui He (Argonne National Laboratory)

## Key Features

- JSON-based configuration management with comprehensive validation
- Three analysis modes: Static Isotropic, Static Anisotropic, Laminar Flow
- Experimental data loading with intelligent caching
- Correlation function calculation with Numba JIT optimization
- Parameter validation and bounds checking
- Memory-efficient matrix operations
- Automatic cache dimension validation and adjustment

## Class: `HomodyneAnalysisCore`

Main analysis engine for homodyne scattering analysis.

### Initialization

```python
from homodyne.analysis.core import HomodyneAnalysisCore

core = HomodyneAnalysisCore(config_file: str | Path)
```

**Parameters:**

- `config_file` (str | Path): Path to JSON configuration file

**Raises:**

- `FileNotFoundError`: Configuration file not found
- `json.JSONDecodeError`: Invalid JSON format
- `ValueError`: Invalid configuration parameters

**Example:**

```python
core = HomodyneAnalysisCore("config_laminar_flow.json")
print(f"Analysis mode: {core.static_mode}")
print(f"Time length: {core.time_length}")
print(f"Parameters: {core.n_params}")
```

### Attributes

#### Configuration Attributes

- **`config`** (dict): Complete configuration dictionary
- **`static_mode`** (bool): True for static analysis, False for laminar flow
- **`static_submode`** (str | None): "isotropic" or "anisotropic" for static mode
- **`n_params`** (int): Number of parameters (3 for static, 7 for laminar flow)

#### Experimental Parameters

- **`q_magnitude`** (float): Scattering wavevector magnitude [Å⁻¹]
- **`dt`** (float): Time step between frames [s/frame]
- **`stator_rotor_gap`** (float): Gap between stator and rotor [Å]
- **`start_frame`** (int): Starting frame number (1-based, inclusive)
- **`end_frame`** (int): Ending frame number (1-based, inclusive)
- **`time_length`** (int): Number of time points (calculated as
  `end_frame - start_frame + 1`)

#### Time Array

- **`time_array`** (np.ndarray): Time points array [seconds]
  - Shape: `(time_length,)`
  - Values: `[0, dt, 2*dt, ..., dt*(time_length-1)]`
  - Always starts at 0 for t1=t2=0 correlation

#### Parameter Bounds

- **`parameter_bounds`** (list\[tuple[float, float]\]): Parameter bounds for
  optimization
  - Format: `[(lower, upper), ...]` for each parameter
  - Length: `n_params` (3 for static, 7 for laminar flow)

### Methods

#### `load_experimental_data`

Load experimental correlation data from cached or raw files.

```python
c2_experimental = core.load_experimental_data(
    phi_angles: np.ndarray,
    n_angles: int
) -> np.ndarray
```

**Parameters:**

- `phi_angles` (np.ndarray): Scattering angles [degrees]
  - Shape: `(n_angles,)`
  - Values: φ ∈ \[0°, 360°)
- `n_angles` (int): Number of angles

**Returns:**

- `c2_experimental` (np.ndarray): Experimental correlation functions
  - Shape: `(n_angles, time_length, time_length)`
  - Values: c₂(φ, t₁, t₂) correlation matrices

**Raises:**

- `FileNotFoundError`: Cached data file not found
- `ValueError`: Data dimensions incompatible with configuration

**Features:**

- Automatic cache loading from NPZ files
- Cache dimension validation and auto-adjustment
- Support for both APS and APS-U data formats
- Intelligent fallback to raw data loading

**Example:**

```python
phi_angles = np.array([0, 36, 72, 108, 144])
c2_data = core.load_experimental_data(phi_angles, len(phi_angles))

print(f"Data shape: {c2_data.shape}")
# Output: Data shape: (5, 600, 600)

# Access correlation for first angle
c2_phi0 = c2_data[0]  # Shape: (600, 600)
```

#### `calculate_correlation_function`

Calculate theoretical correlation function for given parameters.

```python
c2_theoretical = core.calculate_correlation_function(
    params: np.ndarray,
    phi_angles: np.ndarray,
    time_array: np.ndarray | None = None
) -> np.ndarray
```

**Parameters:**

- `params` (np.ndarray): Model parameters
  - Static mode (3 params): `[D₀, α, D_offset]`
  - Laminar flow (7 params): `[D₀, α, D_offset, γ̇₀, β, γ̇_offset, φ₀]`
- `phi_angles` (np.ndarray): Scattering angles [degrees]
- `time_array` (np.ndarray | None): Time points [seconds] (default: use
  `self.time_array`)

**Returns:**

- `c2_theoretical` (np.ndarray): Theoretical correlation functions
  - Shape: `(n_angles, time_length, time_length)`
  - Values: Unscaled correlation c₂(φ, t₁, t₂)

**Physical Model:**

$$c_2(\\phi, t_1, t_2) = 1 + \\exp\\left[-q^2 \\int\_{t_1}^{t_2} D(t)dt\\right] \\times
\\text{sinc}^2\\left\[\\frac{1}{2\\pi} qh \\int\_{t_1}^{t_2}
\\dot{\\gamma}(t)\\cos(\\phi-\\phi_0)dt\\right\]$$

**Time-Dependent Coefficients:**

- $D(t) = D_0 t^\\alpha + D\_{\\text{offset}}$ (anomalous diffusion)
- $\\dot{\\gamma}(t) = \\dot{\\gamma}_0 t^\\beta + \\dot{\\gamma}_{\\text{offset}}$
  (time-dependent shear rate)

**Example:**

```python
# Static isotropic parameters
params = np.array([1e-12, 1.0, 0.0])  # [D₀, α, D_offset]
phi_angles = np.array([0, 45, 90])

c2_theory = core.calculate_correlation_function(params, phi_angles)

# Laminar flow parameters
params_flow = np.array([1e-12, 1.0, 0.0, 5.0, 0.5, 0.0, 45.0])
c2_flow = core.calculate_correlation_function(params_flow, phi_angles)
```

#### `calculate_chi_squared_optimized`

Calculate chi-squared goodness-of-fit metric with scaling optimization.

```python
chi_squared = core.calculate_chi_squared_optimized(
    params: np.ndarray,
    phi_angles: np.ndarray,
    c2_experimental: np.ndarray,
    return_details: bool = False
) -> float | dict
```

**Parameters:**

- `params` (np.ndarray): Model parameters
- `phi_angles` (np.ndarray): Scattering angles [degrees]
- `c2_experimental` (np.ndarray): Experimental correlation data
- `return_details` (bool): Return full optimization details (default: False)

**Returns:**

- `chi_squared` (float): Chi-squared value (if `return_details=False`)
- `details` (dict): Full results (if `return_details=True`)
  - `'chi_squared'`: Chi-squared value
  - `'contrast_params'`: Contrast parameters for each angle
  - `'offset_params'`: Offset parameters for each angle
  - `'c2_theoretical_scaled'`: Scaled theoretical correlations

**Chi-Squared Formula:**

$$\\chi^2 = \\sum\_{i,j,k} \\left\[\\frac{c\_{2,\\text{exp}}(i,j,k) - (A_i \\cdot
c\_{2,\\text{theo}}(i,j,k) + B_i)}{c\_{2,\\text{exp}}(i,j,k)}\\right\]^2$$

where $A_i$ and $B_i$ are contrast and offset scaling parameters for each angle.

**Performance:**

- Vectorized NumPy operations for all calculations
- 38% faster than previous implementation (1.33ms → 0.82ms)
- Cached configuration to avoid repeated dict lookups
- Memory-pooled result arrays

**Example:**

```python
params = np.array([1e-12, 1.0, 0.0])
chi2 = core.calculate_chi_squared_optimized(params, phi_angles, c2_data)
print(f"Chi-squared: {chi2:.6e}")

# Get full details
details = core.calculate_chi_squared_optimized(
    params, phi_angles, c2_data, return_details=True
)
print(f"Contrast params: {details['contrast_params']}")
print(f"Offset params: {details['offset_params']}")
```

#### `validate_parameters`

Validate parameters against physical bounds.

```python
is_valid = core.validate_parameters(params: np.ndarray) -> bool
```

**Parameters:**

- `params` (np.ndarray): Model parameters to validate

**Returns:**

- `is_valid` (bool): True if all parameters within bounds

**Validation Checks:**

1. Parameter count matches mode (3 for static, 7 for flow)
2. All parameters within configured bounds
3. No NaN or Inf values
4. Physical constraints (e.g., D₀ > 0)

**Example:**

```python
params = np.array([1e-12, 1.0, 0.0])
if core.validate_parameters(params):
    chi2 = core.calculate_chi_squared_optimized(params, phi_angles, c2_data)
else:
    print("Invalid parameters!")
```

## Frame Counting Convention

### Critical Implementation Detail

The module implements **1-based inclusive** frame counting in configurations:

```python
# In __init__ method
self.start_frame = config['data_settings']['start_frame']  # 1-based
self.end_frame = config['data_settings']['end_frame']      # 1-based

# Calculate time_length (INCLUSIVE)
self.time_length = self.end_frame - self.start_frame + 1  # Fixed in v0.6.5

# Create time array starting at 0
self.time_array = np.linspace(0, self.dt * (self.time_length - 1), self.time_length)
```

### Example Calculation

```python
# Configuration values
start_frame = 401
end_frame = 1000

# Time length calculation
time_length = 1000 - 401 + 1  # = 600 (NOT 599!)

# Time array
dt = 0.5
time_array = np.linspace(0, 0.5 * 599, 600)  # 600 points from 0 to 299.5 seconds
```

### Utility Functions

Use these utility functions for consistency:

```python
from homodyne.core.io_utils import calculate_time_length, config_frames_to_python_slice

# Calculate time_length
time_length = calculate_time_length(start_frame=401, end_frame=1000)
# Returns: 600

# Convert for array slicing
python_start, python_end = config_frames_to_python_slice(401, 1000)
# Returns: (400, 1000)
```

## Cache Dimension Validation

### Auto-Adjustment Feature

The analysis core automatically detects and adjusts for dimension mismatches:

```python
# In load_experimental_data method
if c2_experimental.shape[1] != self.time_length:
    logger.info(
        f"Auto-adjusting time_length from {self.time_length} "
        f"to {c2_experimental.shape[1]} to match cached data dimensions"
    )
    self.time_length = c2_experimental.shape[1]
    # Recreate time_array with correct dimensions
    self.time_array = np.linspace(0, self.dt * (self.time_length - 1), self.time_length)
```

**Use Cases:**

- Legacy cache files created with old frame counting formula
- Cache files from different configurations
- Partial data extraction for testing

**Recommendation:**

For production analysis, regenerate cache files to match configuration exactly.

## Analysis Modes

### Static Isotropic Mode

**Parameters**: 3 (D₀, α, D_offset) **Complexity**: O(N) **Use Case**: Brownian motion
only, isotropic systems

```python
config = {
    "analysis_settings": {
        "static_mode": true,
        "static_submode": "isotropic"
    }
}

core = HomodyneAnalysisCore("config_static_isotropic.json")
assert core.n_params == 3
assert core.static_mode == True
assert core.static_submode == "isotropic"
```

### Static Anisotropic Mode

**Parameters**: 3 (D₀, α, D_offset) **Complexity**: O(N log N) **Use Case**: Static
systems with angular dependence

```python
config = {
    "analysis_settings": {
        "static_mode": true,
        "static_submode": "anisotropic"
    }
}

core = HomodyneAnalysisCore("config_static_anisotropic.json")
assert core.n_params == 3
assert core.static_submode == "anisotropic"
```

### Laminar Flow Mode

**Parameters**: 7 (D₀, α, D_offset, γ̇₀, β, γ̇_offset, φ₀) **Complexity**: O(N²) **Use
Case**: Full nonequilibrium with flow and shear

```python
config = {
    "analysis_settings": {
        "static_mode": false,
        "static_submode": null
    }
}

core = HomodyneAnalysisCore("config_laminar_flow.json")
assert core.n_params == 7
assert core.static_mode == False
```

## Performance Optimizations

### Numba JIT Compilation

Core calculations use Numba for 3-5x speedup:

```python
@nb.jit(nopython=True)
def calculate_diffusion_integral_numba(time_array, D0, alpha, D_offset):
    """JIT-compiled diffusion integral calculation."""
    # Vectorized operations in compiled code
    pass
```

### Vectorized Operations

All array operations are vectorized:

```python
# Vectorized correlation calculation
time_diff = time_array[:, np.newaxis] - time_array[np.newaxis, :]
diffusion_term = np.exp(-q**2 * diffusion_integral)
```

### Memory Efficiency

- Pre-allocated result arrays
- In-place operations where possible
- Smart caching of intermediate results

**Performance Metrics** (v0.6.1+):

- Chi-squared calculation: 38% faster (1.33ms → 0.82ms)
- Memory overhead: Reduced through pooling
- JIT compatibility: Maintained while improving pure Python paths

## Error Handling

The module provides comprehensive error handling:

```python
from homodyne.analysis.core import HomodyneAnalysisCore

try:
    core = HomodyneAnalysisCore("config.json")
    c2_data = core.load_experimental_data(phi_angles, n_angles)
except FileNotFoundError as e:
    print(f"Configuration or data file not found: {e}")
except ValueError as e:
    print(f"Invalid configuration or data: {e}")
except json.JSONDecodeError as e:
    print(f"Invalid JSON in configuration: {e}")
```

## Complete Example

```python
import numpy as np
from homodyne.analysis.core import HomodyneAnalysisCore
from homodyne.optimization.classical import ClassicalOptimizer

# Initialize analysis core
config_file = "config_laminar_flow.json"
core = HomodyneAnalysisCore(config_file)

print(f"Analysis mode: {'Static' if core.static_mode else 'Laminar Flow'}")
print(f"Parameters: {core.n_params}")
print(f"Time length: {core.time_length}")
print(f"Time step: {core.dt} s/frame")

# Load experimental data
phi_angles = np.array([0, 36, 72, 108, 144])
c2_experimental = core.load_experimental_data(phi_angles, len(phi_angles))

print(f"Data shape: {c2_experimental.shape}")

# Calculate correlation for test parameters
if core.static_mode:
    test_params = np.array([1e-12, 1.0, 0.0])  # Static parameters
else:
    test_params = np.array([1e-12, 1.0, 0.0, 5.0, 0.5, 0.0, 45.0])  # Flow parameters

c2_theoretical = core.calculate_correlation_function(test_params, phi_angles)

# Calculate chi-squared
chi2_details = core.calculate_chi_squared_optimized(
    test_params, phi_angles, c2_experimental, return_details=True
)

print(f"Chi-squared: {chi2_details['chi_squared']:.6e}")
print(f"Contrast parameters: {chi2_details['contrast_params']}")

# Run optimization
optimizer = ClassicalOptimizer(core, core.config)
optimal_params, results = optimizer.run_classical_optimization_optimized(
    phi_angles=phi_angles,
    c2_experimental=c2_experimental
)

print(f"Optimal parameters: {optimal_params}")
print(f"Final chi-squared: {results['chi_squared']:.6e}")
```

## See Also

- **[core/io_utils.md](core/io_utils.md)**: Frame counting utility functions
- **[optimization/classical.md](optimization/classical.md)**: Classical optimization
- **[optimization/robust.md](optimization/robust.md)**: Robust optimization
- **[data/xpcs_loader.md](data/xpcs_loader.md)**: XPCS data loading

## References

1. He, H., et al. (2024). "Transport coefficient approach for characterizing
   nonequilibrium dynamics in soft matter." *PNAS*, 121(31), e2401162121.
   https://doi.org/10.1073/pnas.2401162121

______________________________________________________________________

**Module**: `homodyne.analysis.core` **Last Updated**: 2025-10-01 **Version**: 1.0.0
**Authors**: Wei Chen, Hongrui He (Argonne National Laboratory)
