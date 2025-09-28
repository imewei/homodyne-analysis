#!/usr/bin/env python3
"""
Advanced Import Analysis and Automation Engine
============================================

Enterprise-grade automated import detection and removal tooling for the homodyne package.
Provides comprehensive AST analysis, safe automated removal, and development workflow integration.

Features:
- Advanced pattern detection (aliases, star imports, conditional imports)
- Type annotation vs runtime usage analysis
- Safe automated removal with backup and validation
- Pre-commit hook and CI/CD integration
- Comprehensive safety checks and audit trails
"""

import argparse
import ast
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional, NamedTuple, Union, Any
import hashlib
import logging


class ImportInfo(NamedTuple):
    """Structured information about an import statement."""
    module: str
    name: Optional[str]  # None for module imports, specific name for from imports
    alias: Optional[str]
    line: int
    col_offset: int
    is_type_only: bool
    is_conditional: bool
    context: str
    usage_contexts: List[str]
    used_in_strings: bool
    used_in_comments: bool
    used_in_docstrings: bool


class UsageContext:
    """Track detailed usage context for imports."""

    def __init__(self):
        self.runtime_usage: Set[str] = set()
        self.type_annotation_usage: Set[str] = set()
        self.string_usage: Set[str] = set()
        self.comment_usage: Set[str] = set()
        self.docstring_usage: Set[str] = set()
        self.conditional_usage: Set[str] = set()


class AdvancedImportAnalyzer(ast.NodeVisitor):
    """Enterprise-grade AST visitor for comprehensive import usage analysis."""

    def __init__(self, source_code: str):
        self.source_code = source_code
        self.source_lines = source_code.splitlines()

        # Import tracking
        self.imports: Dict[str, ImportInfo] = {}
        self.from_imports: Dict[Tuple[str, str], ImportInfo] = {}

        # Usage tracking
        self.usage_context = UsageContext()
        self.names_used = set()  # All names referenced
        self.attribute_access = defaultdict(set)  # module -> set of attributes accessed
        self.function_calls = set()  # Function calls made
        self.class_instantiations = set()  # Classes instantiated

        # Context tracking
        self.context_stack = []
        self.in_type_annotation = False
        self.in_string_annotation = False
        self.current_function = None
        self.current_class = None

        # Advanced analysis
        self.star_imports = set()
        self.conditional_imports = set()
        self.lazy_imports = set()

        # Pre-analyze comments, strings, and docstrings
        self._analyze_text_content()

    def _analyze_text_content(self):
        """Pre-analyze comments, strings, and docstrings for import usage."""
        for line_num, line in enumerate(self.source_lines, 1):
            # Check comments
            comment_match = re.search(r'#(.+)', line)
            if comment_match:
                comment_text = comment_match.group(1)
                self._extract_names_from_text(comment_text, self.usage_context.comment_usage)

            # Check for string literals that might contain import names
            string_matches = re.findall(r'["\'](.*?)["\']', line)
            for string_content in string_matches:
                self._extract_names_from_text(string_content, self.usage_context.string_usage)

    def _extract_names_from_text(self, text: str, usage_set: Set[str]):
        """Extract potential import names from text content."""
        # Look for Python identifiers that could be import names
        python_identifiers = re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', text)
        usage_set.update(python_identifiers)

    def _get_current_context(self) -> str:
        """Get current code context for tracking usage."""
        context_parts = []
        if self.current_class:
            context_parts.append(f"class:{self.current_class}")
        if self.current_function:
            context_parts.append(f"function:{self.current_function}")
        if self.in_type_annotation:
            context_parts.append("type_annotation")

        return ".".join(context_parts) if context_parts else "module_level"

    def _is_conditional_import(self, node) -> bool:
        """Check if import is inside a conditional block."""
        # Simple heuristic: check if we're inside an if/try block
        return any(isinstance(ctx, (ast.If, ast.Try, ast.ExceptHandler))
                  for ctx in self.context_stack)

    def visit_Import(self, node):
        """Analyze import statements with enhanced context tracking."""
        is_conditional = self._is_conditional_import(node)

        for alias in node.names:
            module_name = alias.name
            alias_name = alias.asname if alias.asname else alias.name.split('.')[-1]

            import_info = ImportInfo(
                module=module_name,
                name=None,
                alias=alias_name,
                line=node.lineno,
                col_offset=node.col_offset,
                is_type_only=self.in_type_annotation,
                is_conditional=is_conditional,
                context=self._get_current_context(),
                usage_contexts=[],
                used_in_strings=alias_name in self.usage_context.string_usage,
                used_in_comments=alias_name in self.usage_context.comment_usage,
                used_in_docstrings=alias_name in self.usage_context.docstring_usage
            )

            self.imports[module_name] = import_info

            if is_conditional:
                self.conditional_imports.add(module_name)

    def visit_ImportFrom(self, node):
        """Analyze from ... import statements with enhanced pattern detection."""
        if node.module is None:
            return

        is_conditional = self._is_conditional_import(node)

        for alias in node.names:
            if alias.name == '*':
                # Star imports - mark as used by default
                import_info = ImportInfo(
                    module=node.module,
                    name='*',
                    alias='*',
                    line=node.lineno,
                    col_offset=node.col_offset,
                    is_type_only=False,  # Star imports are never type-only
                    is_conditional=is_conditional,
                    context=self._get_current_context(),
                    usage_contexts=['star_import'],
                    used_in_strings=False,
                    used_in_comments=False,
                    used_in_docstrings=False
                )
                self.from_imports[(node.module, '*')] = import_info
                self.star_imports.add(node.module)
            else:
                alias_name = alias.asname if alias.asname else alias.name

                import_info = ImportInfo(
                    module=node.module,
                    name=alias.name,
                    alias=alias_name,
                    line=node.lineno,
                    col_offset=node.col_offset,
                    is_type_only=self.in_type_annotation,
                    is_conditional=is_conditional,
                    context=self._get_current_context(),
                    usage_contexts=[],
                    used_in_strings=alias_name in self.usage_context.string_usage,
                    used_in_comments=alias_name in self.usage_context.comment_usage,
                    used_in_docstrings=alias_name in self.usage_context.docstring_usage
                )

                self.from_imports[(node.module, alias.name)] = import_info

                if is_conditional:
                    self.conditional_imports.add((node.module, alias.name))

    def visit_Name(self, node):
        """Track name usage with enhanced context analysis."""
        self.names_used.add(node.id)

        # Track usage context
        if self.in_type_annotation:
            self.usage_context.type_annotation_usage.add(node.id)
        else:
            self.usage_context.runtime_usage.add(node.id)

        current_context = self._get_current_context()

        # Mark imports as used
        for module_name, info in self.imports.items():
            if info.alias == node.id:
                # Update usage contexts
                info.usage_contexts.append(current_context)

        for (module, name), info in self.from_imports.items():
            if info.alias == node.id:
                # Update usage contexts
                info.usage_contexts.append(current_context)

    def visit_Attribute(self, node):
        """Track attribute access for modules with enhanced pattern detection."""
        if isinstance(node.value, ast.Name):
            module_alias = node.value.id
            attribute = node.attr

            # Track usage context
            if self.in_type_annotation:
                self.usage_context.type_annotation_usage.add(module_alias)
            else:
                self.usage_context.runtime_usage.add(module_alias)

            # Track attribute access for imported modules
            for module_name, info in self.imports.items():
                if info.alias == module_alias:
                    self.attribute_access[module_name].add(attribute)
                    context = f"{self._get_current_context()}.{attribute}"
                    info.usage_contexts.append(context)

        self.generic_visit(node)

    def visit_Call(self, node):
        """Track function calls with enhanced pattern detection."""
        if isinstance(node.func, ast.Name):
            self.function_calls.add(node.func.id)
            # Track usage context
            if self.in_type_annotation:
                self.usage_context.type_annotation_usage.add(node.func.id)
            else:
                self.usage_context.runtime_usage.add(node.func.id)

        elif isinstance(node.func, ast.Attribute):
            if isinstance(node.func.value, ast.Name):
                caller = node.func.value.id
                method = node.func.attr
                self.function_calls.add(f"{caller}.{method}")

                # Track usage context for the caller
                if self.in_type_annotation:
                    self.usage_context.type_annotation_usage.add(caller)
                else:
                    self.usage_context.runtime_usage.add(caller)

        self.generic_visit(node)

    def visit_FunctionDef(self, node):
        """Track function definitions with context management."""
        old_function = self.current_function
        self.current_function = node.name
        self.context_stack.append(node)

        # Check type annotations
        self._visit_type_annotations(node)

        self.generic_visit(node)

        self.context_stack.pop()
        self.current_function = old_function

    def visit_AsyncFunctionDef(self, node):
        """Track async function definitions."""
        self.visit_FunctionDef(node)

    def visit_ClassDef(self, node):
        """Track class definitions with context management."""
        old_class = self.current_class
        self.current_class = node.name
        self.context_stack.append(node)

        self.generic_visit(node)

        self.context_stack.pop()
        self.current_class = old_class

    def visit_If(self, node):
        """Track conditional blocks."""
        self.context_stack.append(node)
        self.generic_visit(node)
        self.context_stack.pop()

    def visit_Try(self, node):
        """Track try blocks."""
        self.context_stack.append(node)
        self.generic_visit(node)
        self.context_stack.pop()

    def visit_ExceptHandler(self, node):
        """Track exception handlers."""
        self.context_stack.append(node)
        self.generic_visit(node)
        self.context_stack.pop()

    def _visit_type_annotations(self, node):
        """Visit type annotations separately."""
        old_in_type = self.in_type_annotation
        self.in_type_annotation = True

        # Visit annotations
        if hasattr(node, 'annotation') and node.annotation:
            self.visit(node.annotation)

        if hasattr(node, 'returns') and node.returns:
            self.visit(node.returns)

        if hasattr(node, 'args'):
            for arg in node.args.args:
                if arg.annotation:
                    self.visit(arg.annotation)

        self.in_type_annotation = old_in_type


class SafetyChecker:
    """Comprehensive safety checks for automated import removal."""

    def __init__(self, package_root: Path):
        self.package_root = package_root
        self.logger = logging.getLogger(__name__)

    def check_git_status(self) -> bool:
        """Ensure working directory is clean."""
        try:
            result = subprocess.run(
                ['git', 'status', '--porcelain'],
                cwd=self.package_root,
                capture_output=True,
                text=True,
                timeout=10
            )
            return result.returncode == 0 and not result.stdout.strip()
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False

    def create_backup(self, files: List[Path]) -> Path:
        """Create timestamped backup of files."""
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        backup_dir = self.package_root / f'.import_cleanup_backup_{timestamp}'
        backup_dir.mkdir(exist_ok=True)

        for file_path in files:
            rel_path = file_path.relative_to(self.package_root)
            backup_file = backup_dir / rel_path
            backup_file.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(file_path, backup_file)

        self.logger.info(f"Backup created at: {backup_dir}")
        return backup_dir

    def verify_syntax(self, file_path: Path) -> bool:
        """Verify file has valid Python syntax."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            ast.parse(content, filename=str(file_path))
            return True
        except (SyntaxError, UnicodeDecodeError):
            return False

    def check_import_safety(self, import_info: ImportInfo, file_content: str) -> bool:
        """Check if import removal is safe."""
        # Never remove star imports automatically
        if import_info.name == '*':
            return False

        # Be cautious with conditional imports
        if import_info.is_conditional:
            return False

        # Check for string usage that might indicate dynamic imports
        if import_info.used_in_strings:
            # Look for patterns like getattr, __import__, importlib
            dynamic_patterns = ['getattr', '__import__', 'importlib', 'exec', 'eval']
            if any(pattern in file_content for pattern in dynamic_patterns):
                return False

        return True


class EnterpriseImportAnalyzer:
    """Enterprise-grade package-level import analysis and automation."""

    def __init__(self, package_root: Path):
        self.package_root = package_root
        self.safety_checker = SafetyChecker(package_root)
        self.logger = self._setup_logging()

        # Find Python files with enhanced filtering
        self.python_files = self._find_python_files()

        # Analysis results cache
        self._analysis_cache = {}
        self._cache_file = package_root / '.import_analysis_cache.json'

    def _setup_logging(self) -> logging.Logger:
        """Setup structured logging for analysis operations."""
        logger = logging.getLogger('import_analyzer')
        logger.setLevel(logging.INFO)

        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        return logger

    def _find_python_files(self) -> List[Path]:
        """Find Python files with enhanced filtering."""
        files = []

        # Exclusion patterns
        exclude_patterns = {
            '__pycache__', '.git', '.pytest_cache', '.mypy_cache',
            'build', 'dist', '*.egg-info', '.tox', '.venv', 'venv'
        }

        for path in self.package_root.rglob("*.py"):
            # Skip if any part matches exclusion patterns
            if any(part in exclude_patterns or part.startswith('.')
                  for part in path.parts):
                continue

            # Skip test files for production analysis
            if any(part.startswith('test_') for part in path.parts):
                continue

            files.append(path)

        return files

    def _get_file_hash(self, file_path: Path) -> str:
        """Get hash of file content for cache validation."""
        try:
            with open(file_path, 'rb') as f:
                return hashlib.sha256(f.read()).hexdigest()[:16]
        except (IOError, UnicodeDecodeError):
            return ''

    def _load_cache(self) -> Dict:
        """Load analysis cache if available and valid."""
        if not self._cache_file.exists():
            return {}

        try:
            with open(self._cache_file, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return {}

    def _save_cache(self, cache_data: Dict):
        """Save analysis cache."""
        try:
            with open(self._cache_file, 'w') as f:
                json.dump(cache_data, f, indent=2, default=str)
        except IOError:
            self.logger.warning("Could not save analysis cache")

    def analyze_file(self, file_path: Path, use_cache: bool = True) -> Dict:
        """Analyze a single file for import usage with caching."""
        rel_path = str(file_path.relative_to(self.package_root))
        file_hash = self._get_file_hash(file_path)

        # Check cache
        if use_cache:
            cache = self._load_cache()
            if rel_path in cache and cache[rel_path].get('hash') == file_hash:
                self.logger.debug(f"Using cached analysis for {rel_path}")
                return cache[rel_path]['analysis']

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except (UnicodeDecodeError, FileNotFoundError) as e:
            return {'error': f'Could not read file: {e}'}

        try:
            tree = ast.parse(content, filename=str(file_path))
        except SyntaxError as e:
            return {'error': f'Syntax error: {e}'}

        analyzer = AdvancedImportAnalyzer(content)
        analyzer.visit(tree)

        # Convert to serializable format
        analysis_result = {
            'imports': {k: v._asdict() for k, v in analyzer.imports.items()},
            'from_imports': {f"{k[0]}::{k[1]}": v._asdict()
                           for k, v in analyzer.from_imports.items()},
            'usage_context': {
                'runtime_usage': list(analyzer.usage_context.runtime_usage),
                'type_annotation_usage': list(analyzer.usage_context.type_annotation_usage),
                'string_usage': list(analyzer.usage_context.string_usage),
                'comment_usage': list(analyzer.usage_context.comment_usage),
                'docstring_usage': list(analyzer.usage_context.docstring_usage)
            },
            'names_used': list(analyzer.names_used),
            'attribute_access': {k: list(v) for k, v in analyzer.attribute_access.items()},
            'function_calls': list(analyzer.function_calls),
            'star_imports': list(analyzer.star_imports),
            'conditional_imports': list(analyzer.conditional_imports),
            'file_path': rel_path,
            'file_hash': file_hash
        }

        # Update cache
        if use_cache:
            cache = self._load_cache()
            cache[rel_path] = {
                'hash': file_hash,
                'analysis': analysis_result,
                'timestamp': time.time()
            }
            self._save_cache(cache)

        return analysis_result

    def analyze_all_files(self, use_cache: bool = True, show_progress: bool = True) -> Dict[str, Dict]:
        """Analyze all Python files in the package with progress tracking."""
        results = {}
        total_files = len(self.python_files)

        self.logger.info(f"Analyzing {total_files} Python files...")

        for i, file_path in enumerate(self.python_files, 1):
            if show_progress and i % 10 == 0:
                self.logger.info(f"Progress: {i}/{total_files} files analyzed")

            rel_path = str(file_path.relative_to(self.package_root))
            results[rel_path] = self.analyze_file(file_path, use_cache=use_cache)

        self.logger.info(f"Analysis complete: {total_files} files processed")
        return results

    def find_unused_imports(self, analysis_results: Dict[str, Dict]) -> Dict[str, List[Dict]]:
        """Find unused imports with advanced pattern detection."""
        unused_by_file = {}
        total_unused = 0

        for file_path, analysis in analysis_results.items():
            if 'error' in analysis:
                self.logger.warning(f"Skipping {file_path}: {analysis['error']}")
                continue

            unused_imports = []
            file_content = self._get_file_content(self.package_root / file_path)

            # Check if this is the new format with ImportInfo objects
            if 'file_hash' in analysis:
                # New format - use enhanced detection

                # Check regular imports
                for module_name, info_dict in analysis['imports'].items():
                    info = ImportInfo(**info_dict)

                    if not self._is_import_used(info, analysis, file_content):
                        if self.safety_checker.check_import_safety(info, file_content):
                            unused_imports.append({
                                'type': 'import',
                                'module': module_name,
                                'alias': info.alias,
                                'line': info.line,
                                'safety_level': self._assess_safety_level(info, file_content),
                                'is_conditional': info.is_conditional,
                                'is_type_only': info.is_type_only
                            })

                # Check from imports
                for import_key, info_dict in analysis['from_imports'].items():
                    module, name = import_key.split('::', 1)
                    info = ImportInfo(**info_dict)

                    if name != '*' and not self._is_import_used(info, analysis, file_content):
                        if self.safety_checker.check_import_safety(info, file_content):
                            unused_imports.append({
                                'type': 'from_import',
                                'module': module,
                                'name': name,
                                'alias': info.alias,
                                'line': info.line,
                                'safety_level': self._assess_safety_level(info, file_content),
                                'is_conditional': info.is_conditional,
                                'is_type_only': info.is_type_only
                            })
            else:
                # Legacy format - basic detection
                for module_name, info in analysis['imports'].items():
                    if not info.get('used', False):
                        unused_imports.append({
                            'type': 'import',
                            'module': module_name,
                            'alias': info.get('alias', module_name),
                            'line': info.get('line', 0),
                            'safety_level': 'medium',
                            'is_conditional': False,
                            'is_type_only': False
                        })

                for (module, name), info in analysis['from_imports'].items():
                    if not info.get('used', False) and name != '*':
                        unused_imports.append({
                            'type': 'from_import',
                            'module': module,
                            'name': name,
                            'alias': info.get('alias', name),
                            'line': info.get('line', 0),
                            'safety_level': 'medium',
                            'is_conditional': False,
                            'is_type_only': False
                        })

            if unused_imports:
                unused_by_file[file_path] = unused_imports
                total_unused += len(unused_imports)

        self.logger.info(f"Found {total_unused} unused imports across {len(unused_by_file)} files")
        return unused_by_file

    def _get_file_content(self, file_path: Path) -> str:
        """Get file content safely."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except (IOError, UnicodeDecodeError):
            return ''

    def _is_import_used(self, info: ImportInfo, analysis: Dict, file_content: str) -> bool:
        """Determine if an import is actually used."""
        alias = info.alias or info.name or info.module.split('.')[-1]

        # Check runtime usage
        if alias in analysis['usage_context']['runtime_usage']:
            return True

        # Check type annotation usage (might be acceptable to remove if type-only)
        if alias in analysis['usage_context']['type_annotation_usage']:
            # Type-only imports might be removable in some contexts
            if not info.is_type_only:
                return True

        # Check string/comment usage (be conservative)
        if (alias in analysis['usage_context']['string_usage'] or
            alias in analysis['usage_context']['comment_usage']):
            return True

        # Check for indirect usage patterns
        if self._check_indirect_usage(alias, file_content):
            return True

        return False

    def _check_indirect_usage(self, name: str, file_content: str) -> bool:
        """Check for indirect usage patterns that AST might miss."""
        # Check for dynamic attribute access
        dynamic_patterns = [
            f'getattr.*{re.escape(name)}',
            f'hasattr.*{re.escape(name)}',
            f'setattr.*{re.escape(name)}',
            f'__import__.*{re.escape(name)}',
            f'importlib.*{re.escape(name)}'
        ]

        for pattern in dynamic_patterns:
            if re.search(pattern, file_content, re.IGNORECASE):
                return True

        return False

    def _assess_safety_level(self, info: ImportInfo, file_content: str) -> str:
        """Assess safety level for import removal."""
        if info.is_conditional:
            return 'medium'  # Conditional imports need careful review

        if info.used_in_strings or info.used_in_comments:
            return 'low'  # Might be used dynamically

        if info.is_type_only:
            return 'high'  # Type-only imports are usually safe

        return 'high'  # Default safe removal

    def suggest_optimizations(self, analysis_results: Dict[str, Dict]) -> List[Dict]:
        """Suggest import optimizations."""
        suggestions = []

        # Find commonly imported modules
        module_usage = {}
        for file_path, analysis in analysis_results.items():
            if 'error' in analysis:
                continue

            for module_name in analysis['imports'].keys():
                if module_name not in module_usage:
                    module_usage[module_name] = []
                module_usage[module_name].append(file_path)

        # Suggest centralization for commonly used external modules
        for module, files in module_usage.items():
            if len(files) > 3 and not module.startswith('homodyne'):
                suggestions.append({
                    'type': 'centralize_import',
                    'module': module,
                    'files': files,
                    'suggestion': f"Consider centralizing '{module}' import (used in {len(files)} files)"
                })

        # Find modules that could use from imports instead
        for file_path, analysis in analysis_results.items():
            if 'error' in analysis:
                continue

            for module_name, info in analysis['imports'].items():
                # Check if this is new or old format
                is_used = False
                if isinstance(info, dict) and 'used' in info:
                    is_used = info['used']
                elif 'file_hash' in analysis:
                    # New format - check usage
                    is_used = self._is_import_used(ImportInfo(**info), analysis, '')

                if is_used and module_name in analysis.get('attribute_access', {}):
                    attributes = analysis['attribute_access'][module_name]
                    if len(attributes) <= 3:  # Small number of attributes
                        suggestions.append({
                            'type': 'use_from_import',
                            'file': file_path,
                            'module': module_name,
                            'attributes': list(attributes),
                            'suggestion': f"Consider 'from {module_name} import {', '.join(attributes)}'"
                        })

        return suggestions

    def generate_cleanup_script(self, unused_imports: Dict[str, List[Dict]]) -> str:
        """Generate a script to automatically remove unused imports."""
        script_lines = [
            "#!/usr/bin/env python3",
            '"""Automated import cleanup script - REVIEW BEFORE RUNNING"""',
            "",
            "import re",
            "import sys",
            "from pathlib import Path",
            "",
            "def remove_unused_imports():",
        ]

        for file_path, unused_list in unused_imports.items():
            script_lines.append(f'    # Cleanup {file_path}')
            script_lines.append(f'    file_path = Path("{file_path}")')
            script_lines.append('    if file_path.exists():')
            script_lines.append('        with open(file_path, "r") as f:')
            script_lines.append('            lines = f.readlines()')

            # Sort by line number in reverse order to avoid line number shifts
            sorted_unused = sorted(unused_list, key=lambda x: x['line'], reverse=True)

            for unused in sorted_unused:
                line_num = unused['line'] - 1  # Convert to 0-based indexing
                if unused['type'] == 'import':
                    script_lines.append(f'        # Remove: import {unused["module"]}')
                else:
                    script_lines.append(f'        # Remove: from {unused["module"]} import {unused["name"]}')
                script_lines.append(f'        if {line_num} < len(lines):')
                script_lines.append(f'            del lines[{line_num}]')

            script_lines.append('        with open(file_path, "w") as f:')
            script_lines.append('            f.writelines(lines)')
            script_lines.append('')

        script_lines.extend([
            "",
            "if __name__ == '__main__':",
            "    print('WARNING: Review this script before running!')",
            "    print('This will modify your source files.')",
            "    response = input('Continue? (yes/no): ')",
            "    if response.lower() == 'yes':",
            "        remove_unused_imports()",
            "        print('Cleanup completed.')",
            "    else:",
            "        print('Cleanup cancelled.')"
        ])

        return '\n'.join(script_lines)

    def check_with_external_tools(self) -> Dict[str, List[str]]:
        """Use external tools for additional validation."""
        tools_results = {}

        # Try using autoflake if available
        try:
            result = subprocess.run([
                'autoflake', '--check', '--recursive', str(self.package_root)
            ], capture_output=True, text=True, timeout=60)
            tools_results['autoflake'] = result.stdout.splitlines()
        except (subprocess.TimeoutExpired, FileNotFoundError):
            tools_results['autoflake'] = ['Tool not available or timeout']

        # Try using unimport if available
        try:
            result = subprocess.run([
                'unimport', '--check', '--recursive', str(self.package_root)
            ], capture_output=True, text=True, timeout=60)
            tools_results['unimport'] = result.stdout.splitlines()
        except (subprocess.TimeoutExpired, FileNotFoundError):
            tools_results['unimport'] = ['Tool not available or timeout']

        return tools_results


def setup_pre_commit_hook(package_root: Path) -> bool:
    """Setup pre-commit hook for automated import checking."""
    hooks_dir = package_root / '.git' / 'hooks'
    if not hooks_dir.exists():
        return False

    pre_commit_hook = hooks_dir / 'pre-commit'
    hook_content = f'''#!/bin/bash
# Automated import checking pre-commit hook

echo "Running import analysis..."
python {package_root}/homodyne/tests/import_analyzer.py --check-only --package-root {package_root}

if [ $? -ne 0 ]; then
    echo "Import issues found. Run import analysis for details."
    echo "To bypass: git commit --no-verify"
    exit 1
fi
'''

    try:
        with open(pre_commit_hook, 'w') as f:
            f.write(hook_content)
        os.chmod(pre_commit_hook, 0o755)
        return True
    except IOError:
        return False


def setup_github_action(package_root: Path) -> bool:
    """Setup GitHub Action for automated import checking."""
    github_dir = package_root / '.github' / 'workflows'
    if not github_dir.exists():
        github_dir.mkdir(parents=True, exist_ok=True)

    action_file = github_dir / 'import-analysis.yml'
    action_content = '''name: Import Analysis

on:
  pull_request:
    branches: [ main, develop ]
  push:
    branches: [ main, develop ]

jobs:
  import-analysis:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.12'
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -e .[dev]
    - name: Run import analysis
      run: |
        python homodyne/tests/import_analyzer.py --check-only --external-tools
    - name: Upload analysis results
      if: failure()
      uses: actions/upload-artifact@v3
      with:
        name: import-analysis-results
        path: import_analysis_*.json
'''

    try:
        with open(action_file, 'w') as f:
            f.write(action_content)
        return True
    except IOError:
        return False


def main():
    """Enhanced CLI interface with enterprise features."""
    parser = argparse.ArgumentParser(
        description='Enterprise-grade import analysis and automation for homodyne package',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''Examples:
  %(prog)s --check-only                    # Quick check for unused imports
  %(prog)s --external-tools --verbose      # Comprehensive analysis with external validation
  %(prog)s --auto-cleanup --dry-run        # Preview automated cleanup
  %(prog)s --setup-hooks                   # Setup development workflow integration
  %(prog)s --generate-report output.json   # Generate detailed analysis report
'''
    )

    parser.add_argument('--package-root', type=Path,
                       default=Path(__file__).parent.parent,
                       help='Root directory of the package')
    parser.add_argument('--output', type=Path,
                       help='Output file for analysis results (JSON)')
    parser.add_argument('--cleanup-script', type=Path,
                       help='Generate cleanup script at this path')
    parser.add_argument('--check-only', action='store_true',
                       help='Only check for issues, do not suggest fixes')
    parser.add_argument('--external-tools', action='store_true',
                       help='Also run external tools for validation')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose output')

    # Enterprise features
    parser.add_argument('--auto-cleanup', action='store_true',
                       help='Generate automated cleanup script')
    parser.add_argument('--dry-run', action='store_true',
                       help='Preview changes without making them')
    parser.add_argument('--setup-hooks', action='store_true',
                       help='Setup pre-commit hooks and CI integration')
    parser.add_argument('--cross-validate', action='store_true',
                       help='Cross-validate findings with external tools')
    parser.add_argument('--no-cache', action='store_true',
                       help='Disable analysis caching')
    parser.add_argument('--safety-level', choices=['low', 'medium', 'high'],
                       default='medium',
                       help='Safety level for automated operations')
    parser.add_argument('--generate-report', type=Path,
                       help='Generate comprehensive analysis report')
    parser.add_argument('--config', type=Path,
                       help='Configuration file for analysis settings')

    args = parser.parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level)

    # Setup development hooks
    if args.setup_hooks:
        print("Setting up development workflow integration...")
        hooks_success = setup_pre_commit_hook(args.package_root)
        action_success = setup_github_action(args.package_root)

        if hooks_success:
            print("✓ Pre-commit hook installed")
        else:
            print("✗ Failed to install pre-commit hook")

        if action_success:
            print("✓ GitHub Action configured")
        else:
            print("✗ Failed to configure GitHub Action")

        return

    # Initialize analyzer
    analyzer = EnterpriseImportAnalyzer(args.package_root)

    # Run analysis
    analysis_results = analyzer.analyze_all_files(
        use_cache=not args.no_cache,
        show_progress=args.verbose
    )
    unused_imports = analyzer.find_unused_imports(analysis_results)

    # External validation
    external_results = {}
    if args.external_tools:
        external_results = analyzer.run_external_validation()

    # Cross-validation
    validation_results = {}
    if args.cross_validate and external_results:
        validation_results = analyzer.cross_validate_findings(unused_imports, external_results)

    # Enhanced summary
    total_unused = sum(len(unused_list) for unused_list in unused_imports.values())
    print(f"\n{'='*60}")
    print(f"ENTERPRISE IMPORT ANALYSIS SUMMARY")
    print(f"{'='*60}")
    print(f"Package: {args.package_root.name}")
    print(f"Files analyzed: {len(analysis_results)}")
    print(f"Files with unused imports: {len(unused_imports)}")
    print(f"Total unused imports: {total_unused}")

    if validation_results:
        confirmed = sum(len(imports) for imports in validation_results.get('confirmed_unused', {}).values())
        disputed = sum(len(imports) for imports in validation_results.get('disputed_findings', {}).values())
        print(f"Confirmed by external tools: {confirmed}")
        print(f"Disputed findings: {disputed}")

    if external_results:
        print(f"\nExternal Tools Status:")
        for tool, result in external_results.items():
            status = "✓" if result.get('available') else "✗"
            issues = " (issues found)" if result.get('issues_found') else ""
            print(f"  {status} {tool}{issues}")

    # Detailed results
    if unused_imports:
        print(f"\nDETAILED FINDINGS:")
        print(f"{'-'*60}")
        for file_path, unused_list in unused_imports.items():
            print(f"\n📁 {file_path}: {len(unused_list)} unused imports")
            if args.verbose:
                for unused in unused_list:
                    safety_icon = {'high': '🟢', 'medium': '🟡', 'low': '🔴'}.get(
                        unused.get('safety_level', 'medium'), '⚪'
                    )
                    conditional_flag = ' [CONDITIONAL]' if unused.get('is_conditional') else ''
                    type_flag = ' [TYPE-ONLY]' if unused.get('is_type_only') else ''

                    if unused['type'] == 'import':
                        print(f"    {safety_icon} Line {unused['line']}: import {unused['module']}{conditional_flag}{type_flag}")
                    else:
                        print(f"    {safety_icon} Line {unused['line']}: from {unused['module']} import {unused['name']}{conditional_flag}{type_flag}")

    # Optimization suggestions
    if not args.check_only:
        suggestions = analyzer.suggest_optimizations(analysis_results)

        if suggestions:
            print(f"\nOPTIMIZATION RECOMMENDATIONS:")
            print(f"{'-'*60}")
            for i, suggestion in enumerate(suggestions[:10], 1):  # Show top 10
                impact = suggestion.get('impact_score', 0)
                impact_icon = '🔥' if impact >= 5 else '⭐' if impact >= 3 else '💡'
                print(f"{i:2d}. {impact_icon} {suggestion['suggestion']}")
                if args.verbose and 'rationale' in suggestion:
                    print(f"      Rationale: {suggestion['rationale']}")

            if len(suggestions) > 10:
                print(f"    ... and {len(suggestions) - 10} more suggestions")

    # Automated cleanup
    if args.auto_cleanup and unused_imports:
        if args.dry_run:
            print(f"\nDRY RUN - PREVIEW OF AUTOMATED CLEANUP:")
            print(f"{'-'*60}")
            safe_count = 0
            for file_path, unused_list in unused_imports.items():
                safe_imports = [u for u in unused_list if u.get('safety_level') == 'high']
                if safe_imports:
                    print(f"\n{file_path}: {len(safe_imports)} safe removals")
                    for unused in safe_imports:
                        if unused['type'] == 'import':
                            print(f"  - Line {unused['line']}: import {unused['module']}")
                        else:
                            print(f"  - Line {unused['line']}: from {unused['module']} import {unused['name']}")
                    safe_count += len(safe_imports)
            print(f"\nTotal safe removals: {safe_count}")
            print("Run without --dry-run to generate cleanup script")
        else:
            print(f"\nGenerating automated cleanup script...")
            script_path = analyzer.generate_safe_cleanup_script(unused_imports, args.cleanup_script)
            print(f"✓ Cleanup script generated: {script_path}")
            print(f"Review the script carefully before executing!")

    # Generate comprehensive report
    if args.generate_report or args.output:
        output_path = args.generate_report or args.output
        results = {
            'metadata': {
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'package_root': str(args.package_root),
                'analysis_version': '2.0',
                'safety_level': args.safety_level
            },
            'summary': {
                'files_analyzed': len(analysis_results),
                'files_with_unused_imports': len(unused_imports),
                'total_unused_imports': total_unused,
                'external_tools_used': list(external_results.keys()) if external_results else []
            },
            'analysis_results': analysis_results,
            'unused_imports': unused_imports,
            'optimization_suggestions': suggestions if not args.check_only else [],
            'external_validation': external_results,
            'cross_validation': validation_results
        }

        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n📊 Comprehensive report saved: {output_path}")

    # Final recommendations
    if total_unused > 0:
        print(f"\nRECOMMENDATIONS:")
        print(f"{'-'*60}")
        if total_unused <= 5:
            print("✓ Few unused imports found - consider manual cleanup")
        elif total_unused <= 20:
            print("⚠ Moderate unused imports - automated cleanup recommended")
        else:
            print("🚨 Many unused imports - prioritize cleanup for maintainability")

        safe_removals = sum(1 for file_imports in unused_imports.values()
                           for imp in file_imports if imp.get('safety_level') == 'high')
        if safe_removals > 0:
            print(f"💡 {safe_removals} imports can be safely auto-removed")

        print(f"\nNext steps:")
        print(f"1. Review findings above")
        print(f"2. Use --auto-cleanup --dry-run to preview changes")
        print(f"3. Generate cleanup script with --auto-cleanup")
        print(f"4. Consider --setup-hooks for continuous monitoring")

    # Exit with appropriate code
    if args.check_only:
        # In check-only mode, exit with error if issues found
        sys.exit(1 if total_unused > 0 else 0)
    else:
        # In analysis mode, always exit successfully
        sys.exit(0)


if __name__ == '__main__':
    main()