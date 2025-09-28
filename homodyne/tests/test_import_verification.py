"""
Comprehensive Import Verification Tests
======================================

Tests to ensure all imports in the homodyne package are properly used and optimized.
Detects unused imports, broken import chains, and validates import dependency structure.
"""

import ast
import importlib
import importlib.util
import os
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple
from unittest.mock import patch

import pytest

from homodyne.tests.conftest import PerformanceTimer


class ImportAnalyzer(ast.NodeVisitor):
    """AST visitor to analyze import statements and usage."""

    def __init__(self):
        self.imports = {}  # module_name -> {alias, line_number, used}
        self.from_imports = {}  # (module, name) -> {alias, line_number, used}
        self.names_used = set()  # All names referenced in the code

    def visit_Import(self, node):
        """Visit import statements."""
        for alias in node.names:
            name = alias.asname if alias.asname else alias.name
            self.imports[alias.name] = {
                'alias': name,
                'line_number': node.lineno,
                'used': False
            }

    def visit_ImportFrom(self, node):
        """Visit from ... import statements."""
        if node.module is None:
            return  # Skip relative imports without module

        for alias in node.names:
            name = alias.asname if alias.asname else alias.name
            self.from_imports[(node.module, alias.name)] = {
                'alias': name,
                'line_number': node.lineno,
                'used': False
            }

    def visit_Name(self, node):
        """Visit name references."""
        self.names_used.add(node.id)

    def visit_Attribute(self, node):
        """Visit attribute access."""
        if isinstance(node.value, ast.Name):
            self.names_used.add(node.value.id)
        self.generic_visit(node)


class ImportVerificationSuite:
    """Comprehensive import verification test suite."""

    def __init__(self, package_root: Path):
        self.package_root = package_root
        self.python_files = self._find_python_files()

    def _find_python_files(self) -> List[Path]:
        """Find all Python files in the package."""
        files = []
        for path in self.package_root.rglob("*.py"):
            # Skip test files and __pycache__ directories
            if not any(part.startswith("test_") or part == "__pycache__"
                      for part in path.parts):
                files.append(path)
        return files

    def analyze_file_imports(self, file_path: Path) -> Tuple[Dict, Dict, Set]:
        """Analyze imports and usage in a single file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except (UnicodeDecodeError, FileNotFoundError):
            return {}, {}, set()

        try:
            tree = ast.parse(content, filename=str(file_path))
        except SyntaxError:
            return {}, {}, set()

        analyzer = ImportAnalyzer()
        analyzer.visit(tree)

        # Mark imports as used if their names appear in the code
        for module_name, import_info in analyzer.imports.items():
            if import_info['alias'] in analyzer.names_used:
                import_info['used'] = True

        for (module, name), import_info in analyzer.from_imports.items():
            if import_info['alias'] in analyzer.names_used:
                import_info['used'] = True

        return analyzer.imports, analyzer.from_imports, analyzer.names_used

    def check_unused_imports(self) -> Dict[str, List[Dict]]:
        """Check for unused imports across all files."""
        unused_imports = {}

        for file_path in self.python_files:
            imports, from_imports, _ = self.analyze_file_imports(file_path)
            file_unused = []

            # Check regular imports
            for module_name, info in imports.items():
                if not info['used']:
                    file_unused.append({
                        'type': 'import',
                        'module': module_name,
                        'line': info['line_number'],
                        'alias': info['alias']
                    })

            # Check from imports
            for (module, name), info in from_imports.items():
                if not info['used']:
                    file_unused.append({
                        'type': 'from_import',
                        'module': module,
                        'name': name,
                        'line': info['line_number'],
                        'alias': info['alias']
                    })

            if file_unused:
                rel_path = file_path.relative_to(self.package_root)
                unused_imports[str(rel_path)] = file_unused

        return unused_imports

    def check_broken_imports(self) -> Dict[str, List[Dict]]:
        """Check for broken imports across all files."""
        broken_imports = {}

        for file_path in self.python_files:
            imports, from_imports, _ = self.analyze_file_imports(file_path)
            file_broken = []

            # Check if modules can be imported
            for module_name in imports.keys():
                try:
                    importlib.import_module(module_name)
                except ImportError as e:
                    file_broken.append({
                        'type': 'import',
                        'module': module_name,
                        'error': str(e)
                    })

            # Check from imports
            for module, name in from_imports.keys():
                try:
                    if module.startswith('.'):
                        # Relative import - try to resolve relative to package
                        package_name = self._get_package_name(file_path)
                        if package_name:
                            full_module = importlib.import_module(module, package_name)
                        else:
                            continue
                    else:
                        full_module = importlib.import_module(module)

                    # Check if the specific name exists
                    if not hasattr(full_module, name):
                        file_broken.append({
                            'type': 'from_import',
                            'module': module,
                            'name': name,
                            'error': f'Module {module} has no attribute {name}'
                        })

                except ImportError as e:
                    file_broken.append({
                        'type': 'from_import',
                        'module': module,
                        'name': name,
                        'error': str(e)
                    })

            if file_broken:
                rel_path = file_path.relative_to(self.package_root)
                broken_imports[str(rel_path)] = file_broken

        return broken_imports

    def _get_package_name(self, file_path: Path) -> str:
        """Get the package name for a file."""
        try:
            rel_path = file_path.relative_to(self.package_root)
            parts = rel_path.parts[:-1]  # Remove filename
            if parts and parts[0] == 'homodyne':
                return '.'.join(parts)
            return 'homodyne'
        except ValueError:
            return ''

    def analyze_import_dependencies(self) -> Dict[str, Set[str]]:
        """Analyze import dependency chains."""
        dependencies = {}

        for file_path in self.python_files:
            imports, from_imports, _ = self.analyze_file_imports(file_path)
            rel_path = str(file_path.relative_to(self.package_root))
            file_deps = set()

            # Add regular imports
            for module_name in imports.keys():
                if module_name.startswith('homodyne'):
                    file_deps.add(module_name)

            # Add from imports
            for module, _ in from_imports.keys():
                if module and (module.startswith('homodyne') or module.startswith('.')):
                    if module.startswith('.'):
                        # Convert relative to absolute
                        package_name = self._get_package_name(file_path)
                        if package_name:
                            try:
                                resolved = importlib.util.resolve_name(module, package_name)
                                file_deps.add(resolved)
                            except ImportError:
                                pass
                    else:
                        file_deps.add(module)

            dependencies[rel_path] = file_deps

        return dependencies


@pytest.fixture(scope="session")
def package_root():
    """Get the package root directory."""
    current_file = Path(__file__)
    # Go up from tests to homodyne package root
    return current_file.parent.parent


@pytest.fixture(scope="session")
def import_analyzer(package_root):
    """Create import analyzer for the package."""
    return ImportVerificationSuite(package_root)


class TestImportVerification:
    """Test suite for import verification."""

    @pytest.mark.integration
    def test_no_unused_imports(self, import_analyzer):
        """Test that there are no unused imports in the codebase."""
        unused_imports = import_analyzer.check_unused_imports()

        # Allow certain exceptions for commonly unused imports
        allowed_unused = {
            '__init__.py': {'TYPE_CHECKING'},  # Type checking imports
            'conftest.py': {'pytest'},  # Pytest fixtures
        }

        critical_unused = {}
        for file_path, unused_list in unused_imports.items():
            filename = os.path.basename(file_path)
            allowed = allowed_unused.get(filename, set())

            filtered_unused = []
            for unused in unused_list:
                import_name = unused.get('name', unused.get('module', ''))
                if import_name not in allowed:
                    filtered_unused.append(unused)

            if filtered_unused:
                critical_unused[file_path] = filtered_unused

        if critical_unused:
            error_msg = "Found unused imports:\n"
            for file_path, unused_list in critical_unused.items():
                error_msg += f"\n{file_path}:\n"
                for unused in unused_list:
                    if unused['type'] == 'import':
                        error_msg += f"  Line {unused['line']}: import {unused['module']}\n"
                    else:
                        error_msg += f"  Line {unused['line']}: from {unused['module']} import {unused['name']}\n"

            pytest.fail(error_msg)

    @pytest.mark.integration
    def test_no_broken_imports(self, import_analyzer):
        """Test that all imports can be resolved."""
        broken_imports = import_analyzer.check_broken_imports()

        if broken_imports:
            error_msg = "Found broken imports:\n"
            for file_path, broken_list in broken_imports.items():
                error_msg += f"\n{file_path}:\n"
                for broken in broken_list:
                    if broken['type'] == 'import':
                        error_msg += f"  import {broken['module']}: {broken['error']}\n"
                    else:
                        error_msg += f"  from {broken['module']} import {broken['name']}: {broken['error']}\n"

            pytest.fail(error_msg)

    @pytest.mark.integration
    def test_import_dependency_cycles(self, import_analyzer):
        """Test for circular import dependencies."""
        dependencies = import_analyzer.analyze_import_dependencies()

        # Simple cycle detection using DFS
        def has_cycle(node, graph, visited, rec_stack):
            visited.add(node)
            rec_stack.add(node)

            for neighbor in graph.get(node, set()):
                if neighbor not in visited:
                    if has_cycle(neighbor, graph, visited, rec_stack):
                        return True
                elif neighbor in rec_stack:
                    return True

            rec_stack.remove(node)
            return False

        visited = set()
        cycles_found = []

        for node in dependencies:
            if node not in visited:
                rec_stack = set()
                if has_cycle(node, dependencies, visited, rec_stack):
                    cycles_found.append(node)

        if cycles_found:
            error_msg = f"Found circular dependencies involving: {cycles_found}"
            pytest.fail(error_msg)

    @pytest.mark.integration
    def test_import_structure_consistency(self, import_analyzer):
        """Test that import structure follows package conventions."""
        issues = []

        for file_path in import_analyzer.python_files:
            imports, from_imports, _ = import_analyzer.analyze_file_imports(file_path)
            rel_path = file_path.relative_to(import_analyzer.package_root)

            # Check for imports that should be relative
            for module_name in imports.keys():
                if module_name.startswith('homodyne.'):
                    # Check if this could be a relative import
                    current_package = '.'.join(rel_path.parts[:-1])
                    if module_name.startswith(current_package + '.'):
                        issues.append(f"{rel_path}: Could use relative import for {module_name}")

            # Check for __init__.py files with missing __all__ when they have exports
            if file_path.name == '__init__.py' and (imports or from_imports):
                try:
                    with open(file_path, 'r') as f:
                        content = f.read()
                    if '__all__' not in content and 'from .' in content:
                        issues.append(f"{rel_path}: Missing __all__ definition with exports")
                except (IOError, UnicodeDecodeError):
                    pass

        # Report issues as warnings, not failures (these are style issues)
        if issues:
            import logging
            logger = logging.getLogger(__name__)
            for issue in issues:
                logger.warning(f"Import structure issue: {issue}")

    @pytest.mark.performance
    def test_critical_imports_are_fast(self):
        """Test that critical imports complete quickly."""
        critical_modules = [
            'homodyne',
            'homodyne.core.config',
            'homodyne.analysis.core',
            'homodyne.optimization.classical',
        ]

        max_import_time = 2.0  # seconds
        slow_imports = []

        for module_name in critical_modules:
            with PerformanceTimer(f"Import {module_name}") as timer:
                try:
                    # Clear module from cache to test cold import
                    if module_name in sys.modules:
                        del sys.modules[module_name]

                    importlib.import_module(module_name)

                except ImportError:
                    # Skip if module not available
                    continue

            if timer.elapsed_time and timer.elapsed_time > max_import_time:
                slow_imports.append((module_name, timer.elapsed_time))

        if slow_imports:
            error_msg = "Slow imports found:\n"
            for module, time_taken in slow_imports:
                error_msg += f"  {module}: {time_taken:.2f}s (max: {max_import_time}s)\n"
            pytest.fail(error_msg)

    @pytest.mark.performance
    def test_lazy_loading_effectiveness(self):
        """Test that lazy loading reduces initial import time."""
        import sys

        # Clear all homodyne modules
        to_remove = [name for name in sys.modules if name.startswith('homodyne')]
        for name in to_remove:
            del sys.modules[name]

        # Time basic import
        with PerformanceTimer("Basic import") as timer:
            import homodyne

        basic_import_time = timer.elapsed_time

        # Clear modules again
        to_remove = [name for name in sys.modules if name.startswith('homodyne')]
        for name in to_remove:
            del sys.modules[name]

        # Time import with immediate access to lazy-loaded items
        with PerformanceTimer("Import with lazy access") as timer:
            import homodyne
            # Access lazy-loaded items
            _ = homodyne.HomodyneAnalysisCore
            _ = homodyne.ClassicalOptimizer

        lazy_access_time = timer.elapsed_time

        # Basic import should be much faster than full access
        if basic_import_time is not None:
            assert basic_import_time < 1.0, f"Basic import too slow: {basic_import_time:.2f}s"

        if lazy_access_time is not None and basic_import_time is not None:
            speedup_ratio = lazy_access_time / basic_import_time
            assert speedup_ratio > 2.0, f"Lazy loading not effective: {speedup_ratio:.1f}x overhead"


class TestImportOptimization:
    """Test suite for import optimization strategies."""

    @pytest.mark.performance
    def test_numba_import_overhead(self):
        """Test that Numba imports don't slow down basic operations."""
        import sys

        # Clear numba-related modules
        to_remove = [name for name in sys.modules if 'numba' in name.lower()]
        for name in to_remove:
            del sys.modules[name]

        # Time import without numba
        with PerformanceTimer("Import without numba") as timer:
            with patch.dict('sys.modules', {'numba': None}):
                # Reload to ensure we're not using cached version
                importlib.reload(homodyne.core.kernels)

        non_numba_time = timer.elapsed_time

        # Clear modules
        to_remove = [name for name in sys.modules if name.startswith('homodyne')]
        for name in to_remove:
            del sys.modules[name]

        # Time import with numba (if available)
        with PerformanceTimer("Import with numba") as timer:
            try:
                importlib.reload(homodyne.core.kernels)
            except ImportError:
                pytest.skip("Numba not available")

        numba_time = timer.elapsed_time

        # Ensure numba doesn't add excessive overhead to basic imports
        if non_numba_time and numba_time:
            overhead_ratio = numba_time / non_numba_time
            assert overhead_ratio < 5.0, f"Excessive numba overhead: {overhead_ratio:.1f}x"

    @pytest.mark.integration
    def test_conditional_imports_work(self):
        """Test that conditional imports handle missing dependencies gracefully."""
        # Test modules that should handle missing dependencies
        conditional_modules = [
            'homodyne.optimization.robust',  # Should work without CVXPY
            'homodyne.visualization.plotting',  # Should work without matplotlib
            'homodyne.core.kernels',  # Should work without numba
        ]

        for module_name in conditional_modules:
            try:
                module = importlib.import_module(module_name)
                # Module should load without errors
                assert module is not None

                # Check if it has expected fallback behavior
                if hasattr(module, '__doc__'):
                    assert isinstance(module.__doc__, str)

            except ImportError as e:
                pytest.fail(f"Module {module_name} failed to import: {e}")

    @pytest.mark.performance
    def test_import_time_regression(self):
        """Test for import time regression compared to baseline."""
        # These are baseline expectations (in seconds)
        baselines = {
            'homodyne': 0.5,
            'homodyne.core': 0.3,
            'homodyne.analysis': 0.8,
            'homodyne.optimization': 1.0,
        }

        regressions = []

        for module_name, baseline_time in baselines.items():
            # Clear module cache
            to_remove = [name for name in sys.modules if name.startswith(module_name)]
            for name in to_remove:
                del sys.modules[name]

            with PerformanceTimer(f"Import {module_name}") as timer:
                try:
                    importlib.import_module(module_name)
                except ImportError:
                    continue  # Skip if module not available

            if timer.elapsed_time and timer.elapsed_time > baseline_time * 1.5:  # 50% tolerance
                regressions.append((module_name, timer.elapsed_time, baseline_time))

        if regressions:
            error_msg = "Import time regressions detected:\n"
            for module, actual, baseline in regressions:
                error_msg += f"  {module}: {actual:.2f}s (baseline: {baseline:.2f}s)\n"
            pytest.fail(error_msg)


# Additional utility functions for import analysis

def find_duplicate_imports(package_root: Path) -> Dict[str, List[str]]:
    """Find duplicate imports across the package."""
    analyzer = ImportVerificationSuite(package_root)
    all_imports = {}

    for file_path in analyzer.python_files:
        imports, from_imports, _ = analyzer.analyze_file_imports(file_path)
        rel_path = str(file_path.relative_to(package_root))

        for module_name in imports.keys():
            if module_name not in all_imports:
                all_imports[module_name] = []
            all_imports[module_name].append(rel_path)

    # Return only modules imported in multiple files
    return {module: files for module, files in all_imports.items() if len(files) > 1}


def suggest_import_optimizations(package_root: Path) -> List[str]:
    """Suggest import optimizations for the package."""
    analyzer = ImportVerificationSuite(package_root)
    suggestions = []

    # Find commonly imported modules
    duplicate_imports = find_duplicate_imports(package_root)

    for module, files in duplicate_imports.items():
        if len(files) > 3 and not module.startswith('homodyne'):
            suggestions.append(
                f"Consider centralizing '{module}' import in __init__.py "
                f"(used in {len(files)} files)"
            )

    return suggestions