"""
Vulture whitelist for homodyne-analysis package.

This file contains intentionally unused code that should not be flagged as dead code:
- BLAS/LAPACK imports reserved for future performance optimizations
- API compatibility imports and exports
- Interface compliance parameters
- Future feature placeholders
"""

# ============================================================================
# BLAS/LAPACK Performance Imports (Reserved for Future Optimization)
# ============================================================================

# homodyne/core/blas_kernels.py and homodyne/statistics/chi_squared.py
dcopy = None
dger = None
dnrm2 = None
dsymm = None
dsymv = None
dgetrf = None
dgetrs = None
dpotri = None
dsygv = None

# homodyne/ui/visualization_optimizer.py
go = None
make_subplots = None

# homodyne/cli/core.py
get_available_backends = None
get_ml_backend_info = None

# homodyne/core/secure_io.py
_original_save_pickle = None

# ============================================================================
# Reserved Variables and Parameters
# ============================================================================

# Common patterns that are intentionally unused:
# - invalidate_on_change: caching parameter
# - owner: ownership tracking
# - custom_angles: angle customization placeholder
# - system_constraints: constraint system placeholder
# - parsed_args: argument parsing in adapters
# - _: intentional ignore of return values

invalidate_on_change = None
owner = None
custom_angles = None
system_constraints = None
parsed_args = None
_ = None

# homodyne/core/secure_io.py
_original_save_pickle = None

# ============================================================================
# Analysis and Optimization Parameters
# ============================================================================

# These are part of function signatures for API compatibility
# or reserved for future enhancements

kwargs = {}  # Generic kwargs for extensibility
args = {}  # Generic args for extensibility
initial_params = []
phi_angles = []
analyzer = None
param_hash = None
n_angles = 0
shell = None
backup = None
output_dir = None
