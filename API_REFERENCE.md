# Homodyne API Reference

**Python 3.12+ Required** | **System CUDA JAX Integration Available** | **Code Quality: Black ✅
isort ✅ flake8 ~400** | **Trust Region Gurobi ✅**

## Core Modules

### `homodyne.analysis.core`

#### `HomodyneAnalysisCore`

Main analysis engine for XPCS data processing.

```python
class HomodyneAnalysisCore:
    def __init__(self, config: Dict[str, Any])

    # Properties
    phi_angles: np.ndarray              # Scattering angles
    c2_experimental: np.ndarray         # Experimental correlation data
    config: Dict[str, Any]              # Configuration settings

    # Methods
    def load_data(self, file_path: str) -> None
    def preprocess_data(self) -> None
    def validate_data(self) -> bool
    def get_experimental_parameters(self) -> Dict[str, Any]
```

**Example Usage:**

```python
from homodyne.analysis.core import HomodyneAnalysisCore

config = {"data": {"input_file": "data.h5"}}
core = HomodyneAnalysisCore(config)
core.load_data("experimental_data.h5")
core.preprocess_data()
```

______________________________________________________________________

### `homodyne.optimization.classical`

#### `ClassicalOptimizer`

Classical optimization methods (Nelder-Mead, Gurobi if available).

```python
class ClassicalOptimizer:
    def __init__(self, analysis_core: HomodyneAnalysisCore)

    # Methods
    def run_optimization(
        self,
        method: str = "nelder-mead",
        initial_params: Optional[np.ndarray] = None,
        bounds: Optional[List[Tuple[float, float]]] = None
    ) -> OptimizationResult

    def run_single_method(
        self,
        method: str,
        objective_func: Callable,
        initial_params: np.ndarray
    ) -> Tuple[bool, Union[scipy.optimize.OptimizeResult, Exception]]

    def create_objective_function(self) -> Callable
    def get_default_bounds(self) -> List[Tuple[float, float]]
```

**Example Usage:**

```python
from homodyne.optimization.classical import ClassicalOptimizer

optimizer = ClassicalOptimizer(analysis_core)
result = optimizer.run_optimization(method="nelder-mead")  # or "gurobi"

print(f"Optimal parameters: {result.x}")
print(f"Final chi-squared: {result.fun}")
print(f"Success: {result.success}")
```

**Note**: L-BFGS-B is no longer used. Available methods are "nelder-mead" and "gurobi"
(if licensed).

#### Gurobi Trust Region Implementation (Enhanced v0.6.5+)

The Gurobi optimization now uses an **iterative trust region SQP approach** for robust
convergence:

**Algorithm Overview:**

1. **Build quadratic approximation** around current point using finite differences
1. **Solve QP subproblem** with trust region constraints using native Gurobi QP solver
1. **Evaluate actual objective** at new candidate point
1. **Update trust region** based on actual vs. predicted improvement
1. **Iterate until convergence** or maximum iterations reached

**Key Features:**

- **Trust region management**: Radius adapts from 0.1 initial → 1e-8 to 1.0 range based
  on step quality
- **Parameter-scaled finite differences**: Epsilon scales with parameter magnitudes for
  numerical stability
- **Diagonal Hessian approximation**: More stable than full Hessian for chi-squared
  problems
- **Convergence criteria**: Gradient norm < tolerance, objective improvement \<
  tolerance, or trust region collapse
- **Parameter bounds**: Native Gurobi constraint support ensures physical parameter
  ranges
- **Progress logging**: Debug messages show iteration progress and χ² convergence

**Configuration Options:**

```python
method_options = {
    "Gurobi": {
        "max_iterations": 50,              # Outer trust region iterations
        "tolerance": 1e-6,                 # Convergence tolerance
        "trust_region_initial": 0.1,       # Initial trust region radius
        "trust_region_min": 1e-8,          # Minimum trust region radius  
        "trust_region_max": 1.0,           # Maximum trust region radius
        "output_flag": 0                   # Gurobi solver verbosity
    }
}
```

**Performance:** Expected convergence in 10-30 iterations for typical XPCS problems with
progressive χ² improvement.

______________________________________________________________________

### `homodyne.optimization.robust`

#### `RobustHomodyneOptimizer`

Robust optimization with uncertainty quantification.

```python
class RobustHomodyneOptimizer:
    def __init__(
        self,
        analysis_core: HomodyneAnalysisCore,
        config: Dict[str, Any]
    )

    # Methods
    def _solve_distributionally_robust(
        self,
        theta_init: np.ndarray,
        phi_angles: np.ndarray,
        c2_experimental: np.ndarray,
        uncertainty_radius: float = 0.05,
        solver: str = "clarabel"
    ) -> Tuple[Optional[np.ndarray], Dict[str, Any]]

    def _solve_scenario_based_robust(self, ...) -> Tuple[Optional[np.ndarray], Dict[str, Any]]
    def _solve_ellipsoidal_robust(self, ...) -> Tuple[Optional[np.ndarray], Dict[str, Any]]
```

**Available Methods:**

- `wasserstein`: Wasserstein Distributionally Robust Optimization
- `scenario`: Scenario-based robust optimization
- `ellipsoidal`: Ellipsoidal uncertainty sets

**Available Solvers:**

- `clarabel`: Default high-performance solver (CLARABEL)
- `scs`: Splitting Conic Solver (SCS)
- `cvxopt`: CVXOPT solver (fallback)

**Example Usage:**

```python
from homodyne.optimization.robust import RobustHomodyneOptimizer

# Initialize with analysis core and config
optimizer = RobustHomodyneOptimizer(analysis_core, config)

# Run distributionally robust optimization
optimal_params, info = optimizer._solve_distributionally_robust(
    theta_init=initial_params,
    phi_angles=phi_angles,
    c2_experimental=c2_experimental,
    uncertainty_radius=0.05,
    solver="clarabel"
)

print(f"Robust optimal parameters: {optimal_params}")
print(f"Final chi-squared: {info['final_chi_squared']}")
```

______________________________________________________________________

### `homodyne.optimization.mcmc`

#### `MCMCSampler`

Bayesian MCMC sampling using PyMC with NUTS.

```python
class MCMCSampler:
    def __init__(self, analysis_core: HomodyneAnalysisCore, config: Dict[str, Any])

    # Methods
    def run_mcmc_analysis(
        self,
        c2_experimental: Optional[np.ndarray] = None,
        phi_angles: Optional[np.ndarray] = None,
        mcmc_config: Optional[Dict[str, Any]] = None,
        filter_angles_for_optimization: Optional[bool] = None
    ) -> Dict[str, Any]

    def compute_convergence_diagnostics(self, trace) -> Dict[str, Any]
    def extract_posterior_statistics(self, trace) -> Dict[str, Any]
    def generate_posterior_samples(self, n_samples: int = 1000) -> Optional[np.ndarray]
```

**Return Structure:**

```python
{
    "trace": arviz.InferenceData,           # MCMC trace
    "posterior_means": Dict[str, float],    # Parameter means
    "chi_squared": float,                   # Model fit quality
    "diagnostics": Dict[str, Any],          # Convergence diagnostics
    "performance_metrics": Dict[str, Any]   # Timing information
}
```

**Example Usage:**

```python
from homodyne.optimization.mcmc import MCMCSampler

config = {
    "mcmc": {
        "chains": 4,
        "draws": 2000,
        "tune": 1000,
        "target_accept": 0.8
    }
}

sampler = MCMCSampler(analysis_core, config)
result = sampler.run_mcmc_analysis()

print(f"Posterior means: {result['posterior_means']}")
print(f"R-hat diagnostics: {result['diagnostics']['r_hat']}")
```

______________________________________________________________________

### `homodyne.core.config`

#### `ConfigManager`

Configuration management and validation.

```python
class ConfigManager:
    def __init__(self, config_path: Optional[str] = None)

    # Methods
    def load_config(self, config_path: str) -> Dict[str, Any]
    def validate_config(self, config: Dict[str, Any]) -> bool
    def get_default_config(self) -> Dict[str, Any]
    def merge_configs(self, *configs: Dict[str, Any]) -> Dict[str, Any]
    def save_config(self, config: Dict[str, Any], path: str) -> None

    # Analysis mode detection  
    def is_static_mode_enabled(self) -> bool
    def get_effective_parameter_count(self) -> int
    def get_analysis_settings(self) -> Dict[str, Any]  # Added in v0.6.5+
    def get_parameter_bounds(self) -> List[Tuple[float, float]]
```

**Example Usage:**

```python
from homodyne.core.config import ConfigManager

config_manager = ConfigManager("my_config.json")
config = config_manager.load_config()

if config_manager.validate_config(config):
    print("Configuration is valid")

is_static = config_manager.is_static_mode_enabled()
param_count = config_manager.get_effective_parameter_count()

# New in v0.6.5+: Enhanced analysis settings
analysis_settings = config_manager.get_analysis_settings()
print(f"Analysis settings: {analysis_settings}")
# Returns: {'static_mode': True, 'model_description': 'static_isotropic analysis with 3 parameters'}

# Get parameter bounds for optimization
bounds = config_manager.get_parameter_bounds()
print(f"Parameter bounds: {bounds}")
# Returns: [(1e-3, 1e6), (-2.0, 2.0), (-5000, 5000), ...]
```

______________________________________________________________________

### `homodyne.plotting`

#### Core Plotting Functions

```python
def plot_c2_heatmaps(
    c2_data: np.ndarray,
    phi_angles: np.ndarray,
    time_delays: Optional[np.ndarray] = None,
    output_dir: Optional[Path] = None,
    method_name: str = "analysis"
) -> bool

def plot_optimization_results(
    results: Dict[str, Any],
    output_dir: Path,
    config: Dict[str, Any]
) -> None

def plot_mcmc_diagnostics(
    trace,
    output_dir: Path,
    config: Dict[str, Any]
) -> None

def plot_robust_optimization_results(
    results: Dict[str, Any],
    output_dir: Path
) -> None
```

**Example Usage:**

```python
from homodyne.plotting import plot_c2_heatmaps, plot_optimization_results

# Generate correlation function heatmaps
success = plot_c2_heatmaps(
    c2_data=c2_experimental,
    phi_angles=phi_angles,
    output_dir=Path("results/plots"),
    method_name="classical"
)

# Plot optimization results
plot_optimization_results(
    results=optimization_result,
    output_dir=Path("results"),
    config=config
)
```

______________________________________________________________________

## Utility Functions

### `homodyne.utils.data`

```python
def load_experimental_data(file_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]
def validate_data_format(data: np.ndarray) -> bool
def preprocess_correlation_data(c2_data: np.ndarray) -> np.ndarray
def filter_angles_by_ranges(
    phi_angles: np.ndarray,
    ranges: List[Tuple[float, float]]
) -> np.ndarray
```

### `homodyne.core.kernels` (Performance Optimized v0.6.5+)

**JIT-Compiled Computational Kernels:**

```python
# High-performance correlation calculations with Numba JIT
def compute_g1_correlation_numba(diffusion_coeff, shear_rate, time_points, angles) -> np.ndarray
def create_time_integral_matrix_numba(time_dependent_array) -> np.ndarray  
def solve_least_squares_batch_numba(theory_batch, exp_batch) -> np.ndarray
def compute_chi_squared_batch_numba(theory_batch, exp_batch, contrast_batch, offset_batch) -> np.ndarray

# Fallback implementations (automatically used when Numba unavailable)
def _solve_least_squares_batch_fallback(theory_batch, exp_batch) -> np.ndarray
def _compute_chi_squared_batch_fallback(theory_batch, exp_batch, contrast_batch, offset_batch) -> np.ndarray
```

**Recent Improvements:**

- **Code cleanup**: Removed 308 lines of unused fallback implementations
- **Added missing functions**: `_solve_least_squares_batch_fallback` and
  `_compute_chi_squared_batch_fallback`
- **Performance optimization**: 3-5x speedup with JIT compilation
- **Numerical stability**: Enhanced finite difference calculations

### `homodyne.utils.performance`

```python
def benchmark_method(
    method_func: Callable,
    *args,
    **kwargs
) -> Dict[str, Any]

def monitor_memory_usage() -> Dict[str, float]
def optimize_numerical_environment() -> None
def profile_function(func: Callable) -> Callable
```

______________________________________________________________________

## Advanced Usage

### Custom Objective Functions

```python
from homodyne.optimization.classical import ClassicalOptimizer

class CustomOptimizer(ClassicalOptimizer):
    def create_objective_function(self) -> Callable:
        def custom_objective(params: np.ndarray) -> float:
            # Your custom objective implementation
            D0, alpha, D_offset = params[:3]

            # Calculate model predictions
            model_c2 = self.compute_model_correlation(params)

            # Custom chi-squared with regularization
            chi2 = np.sum((self.analysis_core.c2_experimental - model_c2)**2)
            regularization = 0.01 * np.sum(params**2)

            return chi2 + regularization

        return custom_objective
```

### Batch Processing

```python
from homodyne.analysis.core import HomodyneAnalysisCore
from homodyne.optimization.classical import ClassicalOptimizer
from pathlib import Path

def batch_analyze(data_files: List[str], config: Dict[str, Any]) -> List[Dict[str, Any]]:
    results = []

    for data_file in data_files:
        # Create analysis core for each dataset
        core = HomodyneAnalysisCore(config)
        core.load_data(data_file)

        # Run optimization
        optimizer = ClassicalOptimizer(core)
        result = optimizer.run_optimization()

        results.append({
            "file": data_file,
            "parameters": result.x,
            "chi_squared": result.fun,
            "success": result.success
        })

    return results
```

### Performance Monitoring

```python
from homodyne.core.config import performance_monitor

@performance_monitor
def analyze_with_monitoring(config_file: str):
    # Your analysis code here
    core = HomodyneAnalysisCore(config)
    optimizer = ClassicalOptimizer(core)
    return optimizer.run_optimization()

# Performance metrics automatically logged
result = analyze_with_monitoring("my_config.json")
```

______________________________________________________________________

## Error Handling

### Exception Types

```python
class HomodyneError(Exception):
    """Base exception for homodyne errors"""

class ConfigurationError(HomodyneError):
    """Configuration file errors"""

class DataFormatError(HomodyneError):
    """Data format or loading errors"""

class OptimizationError(HomodyneError):
    """Optimization convergence errors"""

class MCMCError(HomodyneError):
    """MCMC sampling errors"""
```

### Error Handling Example

```python
from homodyne.exceptions import ConfigurationError, OptimizationError

try:
    core = HomodyneAnalysisCore(config)
    optimizer = ClassicalOptimizer(core)
    result = optimizer.run_optimization()

except ConfigurationError as e:
    print(f"Configuration error: {e}")

except OptimizationError as e:
    print(f"Optimization failed: {e}")

except Exception as e:
    print(f"Unexpected error: {e}")
```

______________________________________________________________________

## CLI and Shell Integration

### Shell Tab Completion

The package provides comprehensive shell completion support with conda environment integration for enhanced CLI experience:

**Installation:**

```bash
# Install completion support
pip install homodyne-analysis[completion]

# Enable for your shell (one-time setup)
homodyne --install-completion bash    # For bash
homodyne --install-completion zsh     # For zsh  
homodyne --install-completion fish    # For fish
homodyne --install-completion powershell  # For PowerShell

# Conda environments: Completion is automatically integrated!
# The install command will show current status and provide guidance.

# To remove completion later
homodyne-cleanup                      # Conda environments (recommended)
homodyne --uninstall-completion bash  # Manual installations (bash)
homodyne --uninstall-completion zsh   # Manual installations (zsh)
```

**Features:**

- **Method completion**: `--method <TAB>` → classical, mcmc, robust, all
- **Config file completion**: `--config <TAB>` → available .json files
- **Directory completion**: `--output-dir <TAB>` → available directories
- **Context-aware**: Adapts based on current command context
- **Cross-platform**: Works on Linux, macOS, and Windows
- **Command shortcuts**: `hc`, `hm`, `hr`, `ha`, `hgm`, `hga`, `hc-iso`, `hc-aniso`, `hc-flow`
- **Conda integration**: Automatic setup in conda environments
- **Help system**: `homodyne_help`, `homodyne_gpu_status` functions

**Note**: Interactive CLI mode has been **removed** as of v0.6.5. Use shell completion
for enhanced CLI experience.

### Code Quality Standards (v0.6.5+)

The homodyne package maintains high code quality with comprehensive tooling:

**Formatting and Style:**

- ✅ **Black**: 100% compliant (88-character line length)
- ✅ **isort**: Import sorting and optimization
- ⚠️ **flake8**: ~400 remaining issues (mostly line length in data scripts)
- ⚠️ **mypy**: ~285 type annotation issues (missing library stubs)

**Recent Improvements:**

- **Code reduction**: Removed 308 lines of unused fallback implementations from
  kernels.py
- **Import optimization**: Cleaned up import patterns and resolved redefinition warnings
- **Critical fixes**: Fixed comparison operators (`== False` → `is False`) and missing
  function definitions
- **Enhanced algorithms**: Improved Gurobi optimization with trust region methods

______________________________________________________________________

## Backend Integration

### JAX Backend (System CUDA GPU Acceleration)

JAX integration provides system CUDA GPU acceleration and JIT compilation for MCMC sampling:

```python
# JAX backend is automatically detected and used when available
# Configuration in mcmc.py handles lazy importing:

from homodyne.optimization.mcmc import _lazy_import_jax
_lazy_import_jax()  # Automatically detects JAX availability

# MCMC sampling with JAX (when available)
sampler = MCMCSampler(analysis_core, config)
result = sampler.run_mcmc_analysis()  # Uses JAX if available, NumPy fallback

# Check if JAX is available
from homodyne.optimization.mcmc import JAX_AVAILABLE
print(f"JAX backend available: {JAX_AVAILABLE}")
```

**JAX Integration Features:**

- **Automatic detection**: JAX used when available, graceful NumPy fallback
- **System CUDA GPU acceleration**: Utilizes system CUDA 12.6+ and cuDNN 9.12+ when present
- **JIT compilation**: Additional performance boost beyond Numba
- **Lazy loading**: JAX imported only when needed

**Installation for System CUDA JAX:**

```bash
pip install jax jaxlib  # CPU version
# OR for system CUDA GPU support:
pip install jax[cuda12-local]  # System CUDA 12.6+ support
# Requires: system CUDA 12.6+ and cuDNN 9.12+ installed
```

### NumPy Backend (Default Fallback)

```python
# NumPy backend used when JAX unavailable
# No configuration needed - automatic fallback
config = {
    "mcmc": {
        "chains": 4,
        "draws": 2000
    }
}
```

______________________________________________________________________

## Testing and Validation

### Test Utilities

```python
from homodyne.tests.utils import (
    create_mock_analysis_core,
    generate_synthetic_data,
    validate_optimization_result
)

# Create mock data for testing
mock_core = create_mock_analysis_core(
    n_angles=10,
    n_times=50,
    true_parameters=[100.0, -0.5, 10.0]
)

# Generate synthetic data
c2_synthetic = generate_synthetic_data(
    phi_angles=np.linspace(-30, 30, 10),
    parameters=[100.0, -0.5, 10.0]
)

# Validate results
is_valid = validate_optimization_result(result, expected_params)
```

______________________________________________________________________

## Migration Guide

### From v1.x to v2.x

```python
# Old API (v1.x)
from homodyne import run_analysis
result = run_analysis(config_file="config.json")

# New API (v2.x)
from homodyne.analysis.core import HomodyneAnalysisCore
from homodyne.optimization.classical import ClassicalOptimizer

core = HomodyneAnalysisCore(config)
optimizer = ClassicalOptimizer(core)
result = optimizer.run_optimization()
```

### Configuration Changes

```python
# Old format
{
    "optimization_method": "nelder-mead",
    "mcmc_chains": 4
}

# New format
{
    "analysis": {
        "method": "classical"
    },
    "classical": {
        "method": "nelder-mead"
    },
    "mcmc": {
        "chains": 4
    }
}
```
