# Import Conventions and Standards

## Overview

This document defines the import conventions and best practices for the Homodyne
Analysis Package. Following these standards ensures code maintainability, IDE
compatibility, and optimal performance.

## Import Quality Metrics

**Current Status** (as of 2025-10-02):

- ✅ **Explicit Import Score**: 100.0% (no wildcard imports)
- ✅ **__all__ Coverage**: 86% (12/14 __init__.py files)
- ✅ **Import Sorting**: 100% (isort compliance)
- ⚠️ **Import Placement**: 89% (16 files with late imports for circular dependency
  prevention)

## Core Principles

### 1. Explicit Imports Only

**✅ DO**: Use explicit imports

```python
from homodyne.analysis.core import HomodyneAnalysisCore
from homodyne.optimization.classical import ClassicalOptimizer
import numpy as np
```

**❌ DON'T**: Use wildcard imports

```python
from homodyne.analysis import *  # NEVER DO THIS
from numpy import *               # NEVER DO THIS
```

**Rationale**:

- Explicit imports enable IDE autocomplete and navigation
- Prevents namespace pollution and name conflicts
- Makes dependencies obvious for maintainability
- Required for static analysis tools (mypy, pylint, etc.)

### 2. PEP 8 Import Organization

Organize imports in three groups, separated by blank lines:

```python
# Standard library imports
import json
import os
from pathlib import Path
from typing import Optional, Dict, Any

# Third-party imports
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Local/relative imports
from homodyne.core.config import ConfigManager
from homodyne.analysis.core import HomodyneAnalysisCore
from .kernels import static_isotropic_model
```

**Tool Support**: `isort` automatically handles this organization:

```bash
isort --profile=black --line-length=88 homodyne/
```

**Pre-commit Hook**: Already configured in `.pre-commit-config.yaml` (lines 104-111)

### 3. Lazy Imports for Performance

**Use Case**: Circular import prevention and startup performance optimization

**Pattern** (from `homodyne/__init__.py`):

```python
from typing import TYPE_CHECKING
import importlib

if TYPE_CHECKING:
    # Type hints only, not executed at runtime
    from homodyne.optimization.classical import ClassicalOptimizer

def get_optimizer():
    """Lazy loading to avoid circular imports."""
    optimizer_module = importlib.import_module('homodyne.optimization.classical')
    return optimizer_module.ClassicalOptimizer
```

**When to Use**:

- Preventing circular dependencies between modules
- Heavy modules that slow down package initialization
- Optional dependencies that might not be installed

**Files Currently Using Lazy Imports**: 16 files (intentional for performance)

### 4. Re-exports in __init__.py

**Pattern**: Explicit __all__ exports for public API

```python
# homodyne/statistics/__init__.py
from .chi_squared import (
    AdvancedChiSquaredAnalyzer,
    BLASChiSquaredKernels,
    batch_chi_squared_analysis,
)

__all__ = [
    "AdvancedChiSquaredAnalyzer",
    "BLASChiSquaredKernels",
    "batch_chi_squared_analysis",
]
```

**Benefits**:

- Defines public API contract explicitly
- Enables `from homodyne.statistics import AdvancedChiSquaredAnalyzer`
- Prevents accidental exposure of internal modules
- Supports IDE autocomplete at package level

**Current Coverage**: 12/14 __init__.py files have __all__ defined (86%)

**TODO**: Add __all__ to remaining 2 __init__.py files without re-exports

### 5. Import Aliases

**✅ DO**: Use conventional aliases for common packages

```python
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
```

**⚠️ AVOID**: Custom aliases that obscure intent

```python
import numpy as npy           # Unconventional
import scipy.optimize as opt  # Too generic
```

**Current Usage**: Only 6 aliases detected across 1840 imports (0.3%)

### 6. Relative vs Absolute Imports

**Relative Imports** (within package):

```python
# In homodyne/analysis/core.py
from ..optimization.classical import ClassicalOptimizer  # Relative
from .kernels import static_isotropic_model              # Relative
```

**Absolute Imports** (external or clarity):

```python
from homodyne.core.config import ConfigManager  # Absolute
import numpy as np                               # External
```

**Current Usage**:

- Relative imports: 121 (26.9%)
- Absolute imports: 447 (99.1% of external imports)

**Guidelines**:

- Use relative imports within the same package for maintainability
- Use absolute imports for external dependencies
- Prefer absolute imports when clarity is paramount

## Import Placement

### Standard Placement (Top of File)

**✅ DO**: Place imports at the top after module docstring

```python
"""Module docstring."""

import os
import sys

from homodyne.core.config import ConfigManager
```

### Conditional/Late Imports (16 files)

**When Acceptable**:

- Preventing circular dependencies
- Type checking only (`if TYPE_CHECKING:`)
- Optional heavy dependencies
- Function-scoped imports for lazy loading

**Example** (circular dependency prevention):

```python
def advanced_feature():
    """Heavy import only when feature is used."""
    from homodyne.advanced.module import AdvancedAnalyzer
    return AdvancedAnalyzer()
```

**Files Using Late Imports** (Intentional):

- `homodyne/ui/visualization_optimizer.py`
- `homodyne/ui/cli_enhancer.py`
- `homodyne/ui/interactive.py`
- ... (13 more files)

## Verification and Enforcement

### Pre-commit Hooks (Automatic)

Import quality is enforced via `.pre-commit-config.yaml`:

1. **isort** (lines 104-111): Automatic import sorting
2. **ruff** (lines 138-145): Detects unused imports (F401)
3. **flake8** (lines 148-159): Additional import linting

Run manually:

```bash
pre-commit run --all-files
```

### Manual Verification

Check import quality:

```bash
# Check for wildcard imports
grep -r "from .* import \*" homodyne/ --include="*.py"

# Check import organization
isort --check-only --diff homodyne/

# Check unused imports
ruff check homodyne/ --select F401
```

### IDE Configuration

**VSCode** (`settings.json`):

```json
{
  "python.linting.enabled": true,
  "python.linting.pylintEnabled": false,
  "python.linting.flake8Enabled": true,
  "python.formatting.provider": "black",
  "editor.formatOnSave": true,
  "[python]": {
    "editor.codeActionsOnSave": {
      "source.organizeImports": true
    }
  }
}
```

**PyCharm**: Enable "Optimize imports on the fly" in Settings → Editor → General → Auto
Import

## Common Patterns

### Configuration Loading

```python
import json
from pathlib import Path
from homodyne.core.config import ConfigManager

# Load configuration
with open("config.json") as f:
    config = json.load(f)
```

### Analysis Workflow

```python
from homodyne.analysis.core import HomodyneAnalysisCore
from homodyne.optimization.classical import ClassicalOptimizer

# Standard workflow
core = HomodyneAnalysisCore(config)
optimizer = ClassicalOptimizer(core, config)
params, result = optimizer.run_classical_optimization_optimized(
    phi_angles=phi_angles,
    c2_experimental=c2_data
)
```

### Optional Dependencies

```python
try:
    import numba
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

# Later in code
if HAS_NUMBA:
    @numba.jit(nopython=True)
    def fast_function():
        ...
```

## Migration Guide

### From Wildcard Imports

**Before**:

```python
from homodyne.analysis import *
```

**After**:

```python
from homodyne.analysis.core import (
    HomodyneAnalysisCore,
    calculate_chi_squared,
)
```

### From Unorganized Imports

**Before**:

```python
from homodyne.core import config
import os
from scipy import optimize
import sys
```

**After** (run `isort`):

```python
import os
import sys

from scipy import optimize

from homodyne.core import config
```

## Performance Considerations

### Import Overhead

- **Standard Import**: ~1ms per module
- **Lazy Import**: Deferred until first use
- **TYPE_CHECKING Import**: Zero runtime cost

### Optimization Strategies

1. **Lazy Loading** for heavy modules:

```python
def get_heavy_analyzer():
    from homodyne.heavy_module import HeavyAnalyzer
    return HeavyAnalyzer()
```

2. **Conditional Imports** for optional features:

```python
try:
    from homodyne.gpu import GPUAccelerator
    USE_GPU = True
except ImportError:
    USE_GPU = False
```

3. **TYPE_CHECKING** for type hints:

```python
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from homodyne.analysis.core import HomodyneAnalysisCore
```

## Troubleshooting

### Circular Import Errors

**Symptom**: `ImportError: cannot import name 'X' from partially initialized module`

**Solutions**:

1. Use lazy imports (function-scoped)
2. Move import to function that uses it
3. Restructure modules to break circular dependency

### Unused Import Warnings

**If Intentional** (e.g., re-exports in __init__.py):

```python
# Re-exported for convenience
from .submodule import PublicAPI  # noqa: F401
```

### Import Not Found

**Check**:

1. Module is in PYTHONPATH
2. Package is installed (`pip install -e .`)
3. No circular dependency issues
4. Correct spelling and case

## References

- **PEP 8**: https://peps.python.org/pep-0008/#imports
- **isort Documentation**: https://pycqa.github.io/isort/
- **Type Checking**: https://mypy.readthedocs.io/en/stable/

## Version History

- **2025-10-02**: Initial documentation based on comprehensive import analysis
  - Verified 100% explicit import compliance
  - Documented 16 intentional lazy import patterns
  - Established __all__ export standards

______________________________________________________________________

**Questions or Issues?** Open an issue at
https://github.com/imewei/homodyne-analysis/issues
