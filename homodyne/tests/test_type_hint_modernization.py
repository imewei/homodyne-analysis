"""
Type Hint Modernization System for Python 3.12+
===============================================

Comprehensive system for modernizing type hints to Python 3.12+ standards
with improved performance characteristics, better readability, and enhanced
static analysis support.

Authors: Wei Chen, Hongrui He
Institution: Argonne National Laboratory
"""

import ast
import logging
import re
from collections import defaultdict
from dataclasses import dataclass
from dataclasses import field
from datetime import datetime
from datetime import timezone
from pathlib import Path
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Set
from typing import Tuple
from typing import Union

import pytest

logger = logging.getLogger(__name__)


@dataclass
class TypeHintModernization:
    """Type hint modernization record."""
    file_path: str
    line_number: int
    old_hint: str
    new_hint: str
    modernization_type: str  # 'union', 'generic', 'optional', 'literal'
    performance_impact: str  # 'improved', 'neutral', 'minimal'


@dataclass
class TypeHintAnalysisResults:
    """Results of type hint analysis."""
    total_files: int
    files_with_hints: int
    total_type_hints: int
    modernizable_hints: int
    modernizations: List[TypeHintModernization] = field(default_factory=list)
    performance_improvements: int = 0
    coverage_percentage: float = 0.0


class Python312TypeHintModernizer:
    """Modernizes type hints to Python 3.12+ standards."""

    def __init__(self, package_root: Path):
        self.package_root = package_root
        self.modernization_patterns = self._get_modernization_patterns()

    def _get_modernization_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Get type hint modernization patterns."""
        return {
            # Union type modernization (PEP 604)
            "union_types": {
                "pattern": r"Union\[([^\]]+)\]",
                "replacement": lambda match: " | ".join(
                    part.strip() for part in match.group(1).split(",")
                ),
                "performance": "improved",
                "description": "Replace Union[X, Y] with X | Y"
            },

            # Optional type modernization
            "optional_types": {
                "pattern": r"Optional\[([^\]]+)\]",
                "replacement": lambda match: f"{match.group(1).strip()} | None",
                "performance": "improved",
                "description": "Replace Optional[X] with X | None"
            },

            # List/Dict/Set modernization (PEP 585)
            "generic_collections": {
                "patterns": {
                    r"List\[([^\]]+)\]": r"list[\1]",
                    r"Dict\[([^\]]+)\]": r"dict[\1]",
                    r"Set\[([^\]]+)\]": r"set[\1]",
                    r"Tuple\[([^\]]+)\]": r"tuple[\1]",
                    r"FrozenSet\[([^\]]+)\]": r"frozenset[\1]",
                    r"Deque\[([^\]]+)\]": r"deque[\1]",
                    r"DefaultDict\[([^\]]+)\]": r"defaultdict[\1]",
                    r"OrderedDict\[([^\]]+)\]": r"OrderedDict[\1]",
                    r"Counter\[([^\]]+)\]": r"Counter[\1]",
                    r"ChainMap\[([^\]]+)\]": r"ChainMap[\1]",
                },
                "performance": "improved",
                "description": "Use built-in generics instead of typing module"
            },

            # Type alias with TypeAlias annotation
            "type_aliases": {
                "pattern": r"^(\s*)([A-Z][A-Za-z0-9_]*)\s*=\s*([A-Za-z0-9_\[\],\s\|]+)$",
                "replacement": r"\1\2: TypeAlias = \3",
                "performance": "improved",
                "description": "Add TypeAlias annotation for better static analysis"
            },

            # Literal types for better type safety
            "literal_strings": {
                "pattern": r'str\s*=\s*["\']([a-zA-Z_][a-zA-Z0-9_]*)["\']',
                "replacement": r'Literal["\1"]',
                "performance": "improved",
                "description": "Use Literal types for string constants"
            },

            # Self type for method return types (PEP 673)
            "self_return_types": {
                "pattern": r"def\s+(\w+)\(self[^)]*\)\s*->\s*['\"]?([A-Z][A-Za-z0-9_]*)['\"]?:",
                "replacement": lambda match, class_name: f"def {match.group(1)}(self) -> Self:",
                "performance": "improved",
                "description": "Use Self type for method return types"
            },

            # Generic type parameter syntax (PEP 695)
            "generic_syntax": {
                "pattern": r"class\s+([A-Z][A-Za-z0-9_]*)\(Generic\[([T-Z][A-Za-z0-9_]*)\]\)",
                "replacement": r"class \1[\2]",
                "performance": "improved",
                "description": "Use new generic syntax"
            }
        }

    def analyze_file_type_hints(self, file_path: Path) -> List[TypeHintModernization]:
        """Analyze type hints in a single file and identify modernization opportunities."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            modernizations = []
            lines = content.split('\n')

            # Check each line for modernizable patterns
            for line_num, line in enumerate(lines, 1):
                modernizations.extend(
                    self._analyze_line_type_hints(line, str(file_path), line_num)
                )

            # Also do AST-based analysis for more complex cases
            try:
                tree = ast.parse(content)
                ast_modernizations = self._analyze_ast_type_hints(tree, str(file_path))
                modernizations.extend(ast_modernizations)
            except SyntaxError:
                logger.warning(f"Could not parse AST for {file_path}")

            return modernizations

        except Exception as e:
            logger.warning(f"Failed to analyze type hints in {file_path}: {e}")
            return []

    def _analyze_line_type_hints(self, line: str, file_path: str, line_num: int) -> List[TypeHintModernization]:
        """Analyze type hints in a single line."""
        modernizations = []

        # Union type modernization
        union_pattern = self.modernization_patterns["union_types"]["pattern"]
        for match in re.finditer(union_pattern, line):
            old_hint = match.group(0)
            new_hint = self.modernization_patterns["union_types"]["replacement"](match)

            modernizations.append(TypeHintModernization(
                file_path=file_path,
                line_number=line_num,
                old_hint=old_hint,
                new_hint=new_hint,
                modernization_type="union",
                performance_impact="improved"
            ))

        # Optional type modernization
        optional_pattern = self.modernization_patterns["optional_types"]["pattern"]
        for match in re.finditer(optional_pattern, line):
            old_hint = match.group(0)
            new_hint = self.modernization_patterns["optional_types"]["replacement"](match)

            modernizations.append(TypeHintModernization(
                file_path=file_path,
                line_number=line_num,
                old_hint=old_hint,
                new_hint=new_hint,
                modernization_type="optional",
                performance_impact="improved"
            ))

        # Generic collections modernization
        generic_patterns = self.modernization_patterns["generic_collections"]["patterns"]
        for old_pattern, new_pattern in generic_patterns.items():
            for match in re.finditer(old_pattern, line):
                old_hint = match.group(0)
                new_hint = re.sub(old_pattern, new_pattern, old_hint)

                modernizations.append(TypeHintModernization(
                    file_path=file_path,
                    line_number=line_num,
                    old_hint=old_hint,
                    new_hint=new_hint,
                    modernization_type="generic",
                    performance_impact="improved"
                ))

        return modernizations

    def _analyze_ast_type_hints(self, tree: ast.AST, file_path: str) -> List[TypeHintModernization]:
        """Analyze type hints using AST parsing."""
        modernizations = []

        class TypeHintVisitor(ast.NodeVisitor):
            def __init__(self):
                self.modernizations = []

            def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
                # Check return type annotation
                if node.returns:
                    self._check_annotation(node.returns, node.lineno)

                # Check parameter annotations
                for arg in node.args.args:
                    if arg.annotation:
                        self._check_annotation(arg.annotation, arg.lineno)

                self.generic_visit(node)

            def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
                """Visit annotated assignment."""
                if node.annotation:
                    self._check_annotation(node.annotation, node.lineno)
                self.generic_visit(node)

            def _check_annotation(self, annotation: ast.AST, line_num: int) -> None:
                """Check if annotation can be modernized."""
                if isinstance(annotation, ast.Subscript):
                    if isinstance(annotation.value, ast.Name):
                        name = annotation.value.id

                        # Check for Union types
                        if name == "Union":
                            old_hint = ast.unparse(annotation)
                            # Simplified modernization for AST
                            new_hint = old_hint.replace("Union[", "").replace("]", "").replace(", ", " | ")

                            self.modernizations.append(TypeHintModernization(
                                file_path=file_path,
                                line_number=line_num,
                                old_hint=old_hint,
                                new_hint=new_hint,
                                modernization_type="union",
                                performance_impact="improved"
                            ))

                        # Check for Optional types
                        elif name == "Optional":
                            old_hint = ast.unparse(annotation)
                            # Extract the inner type
                            if isinstance(annotation.slice, ast.Name):
                                inner_type = annotation.slice.id
                                new_hint = f"{inner_type} | None"
                            else:
                                inner_type = ast.unparse(annotation.slice)
                                new_hint = f"{inner_type} | None"

                            self.modernizations.append(TypeHintModernization(
                                file_path=file_path,
                                line_number=line_num,
                                old_hint=old_hint,
                                new_hint=new_hint,
                                modernization_type="optional",
                                performance_impact="improved"
                            ))

        visitor = TypeHintVisitor()
        visitor.visit(tree)
        modernizations.extend(visitor.modernizations)

        return modernizations

    def analyze_package_type_hints(self) -> TypeHintAnalysisResults:
        """Analyze type hints across the entire package."""
        logger.info("Analyzing type hints across package")

        total_files = 0
        files_with_hints = 0
        all_modernizations = []

        for py_file in self.package_root.rglob("*.py"):
            if self._should_skip_file(py_file):
                continue

            total_files += 1
            file_modernizations = self.analyze_file_type_hints(py_file)

            if file_modernizations:
                files_with_hints += 1
                all_modernizations.extend(file_modernizations)

        # Calculate performance improvements
        performance_improvements = len([
            m for m in all_modernizations
            if m.performance_impact == "improved"
        ])

        # Calculate coverage
        coverage_percentage = (files_with_hints / total_files * 100) if total_files > 0 else 0

        results = TypeHintAnalysisResults(
            total_files=total_files,
            files_with_hints=files_with_hints,
            total_type_hints=len(all_modernizations),
            modernizable_hints=len(all_modernizations),
            modernizations=all_modernizations,
            performance_improvements=performance_improvements,
            coverage_percentage=coverage_percentage
        )

        logger.info(f"Found {len(all_modernizations)} modernizable type hints")
        return results

    def apply_modernizations(self, modernizations: List[TypeHintModernization],
                           dry_run: bool = True) -> Dict[str, Any]:
        """Apply type hint modernizations to files."""
        logger.info(f"Applying {len(modernizations)} type hint modernizations (dry_run={dry_run})")

        # Group by file
        by_file = defaultdict(list)
        for mod in modernizations:
            by_file[mod.file_path].append(mod)

        results = {
            "files_modified": 0,
            "total_modernizations": len(modernizations),
            "successful_modernizations": 0,
            "failed_modernizations": 0,
            "errors": []
        }

        for file_path, file_modernizations in by_file.items():
            try:
                if self._apply_file_modernizations(file_path, file_modernizations, dry_run):
                    results["files_modified"] += 1
                    results["successful_modernizations"] += len(file_modernizations)
                else:
                    results["failed_modernizations"] += len(file_modernizations)
            except Exception as e:
                results["errors"].append(f"Failed to modernize {file_path}: {e}")
                results["failed_modernizations"] += len(file_modernizations)

        return results

    def _apply_file_modernizations(self, file_path: str, modernizations: List[TypeHintModernization],
                                  dry_run: bool) -> bool:
        """Apply modernizations to a single file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Sort modernizations by line number (descending) to avoid offset issues
            sorted_mods = sorted(modernizations, key=lambda m: m.line_number, reverse=True)

            lines = content.split('\n')
            modified = False

            for mod in sorted_mods:
                if mod.line_number <= len(lines):
                    line = lines[mod.line_number - 1]
                    if mod.old_hint in line:
                        new_line = line.replace(mod.old_hint, mod.new_hint)
                        lines[mod.line_number - 1] = new_line
                        modified = True
                        logger.debug(f"Modernized {mod.old_hint} -> {mod.new_hint} in {file_path}:{mod.line_number}")

            if modified and not dry_run:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write('\n'.join(lines))

            return modified

        except Exception as e:
            logger.error(f"Failed to apply modernizations to {file_path}: {e}")
            return False

    def _should_skip_file(self, file_path: Path) -> bool:
        """Check if file should be skipped."""
        skip_patterns = {
            "__pycache__", ".git", "build", "dist",
            ".pytest_cache", ".mypy_cache", "venv", ".venv"
        }
        return any(pattern in str(file_path) for pattern in skip_patterns)

    def generate_modernization_report(self, results: TypeHintAnalysisResults) -> Dict[str, Any]:
        """Generate comprehensive modernization report."""
        # Group modernizations by type
        by_type = defaultdict(int)
        by_performance = defaultdict(int)

        for mod in results.modernizations:
            by_type[mod.modernization_type] += 1
            by_performance[mod.performance_impact] += 1

        # Find most common modernizations
        file_counts = defaultdict(int)
        for mod in results.modernizations:
            file_counts[mod.file_path] += 1

        top_files = sorted(file_counts.items(), key=lambda x: x[1], reverse=True)[:10]

        report = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "summary": {
                "total_files_analyzed": results.total_files,
                "files_with_modernizable_hints": results.files_with_hints,
                "total_modernizable_hints": results.modernizable_hints,
                "performance_improvements": results.performance_improvements,
                "coverage_percentage": results.coverage_percentage
            },
            "modernization_types": dict(by_type),
            "performance_impact": dict(by_performance),
            "top_files_for_modernization": [
                {"file": file_path, "modernizations": count}
                for file_path, count in top_files
            ],
            "recommendations": self._generate_recommendations(results),
            "sample_modernizations": [
                {
                    "file": mod.file_path,
                    "line": mod.line_number,
                    "old": mod.old_hint,
                    "new": mod.new_hint,
                    "type": mod.modernization_type
                }
                for mod in results.modernizations[:10]
            ]
        }

        return report

    def _generate_recommendations(self, results: TypeHintAnalysisResults) -> List[str]:
        """Generate modernization recommendations."""
        recommendations = []

        if results.modernizable_hints > 50:
            recommendations.append("Consider running automated type hint modernization")

        if results.performance_improvements > 20:
            recommendations.append("High potential for performance improvements through modern type hints")

        # Check for specific patterns
        union_count = len([m for m in results.modernizations if m.modernization_type == "union"])
        if union_count > 10:
            recommendations.append(f"Replace {union_count} Union types with | syntax for better performance")

        optional_count = len([m for m in results.modernizations if m.modernization_type == "optional"])
        if optional_count > 10:
            recommendations.append(f"Replace {optional_count} Optional types with | None syntax")

        generic_count = len([m for m in results.modernizations if m.modernization_type == "generic"])
        if generic_count > 5:
            recommendations.append(f"Modernize {generic_count} generic collection types for better performance")

        if results.coverage_percentage < 50:
            recommendations.append("Consider adding type hints to improve static analysis")

        return recommendations

    def validate_modernized_syntax(self, file_path: Path) -> bool:
        """Validate that modernized syntax is correct."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Try to compile the code
            compile(content, str(file_path), 'exec')

            # Try to parse as AST
            ast.parse(content)

            return True
        except (SyntaxError, TypeError) as e:
            logger.error(f"Syntax validation failed for {file_path}: {e}")
            return False


class TestTypeHintModernization:
    """Test suite for type hint modernization system."""

    def test_union_type_modernization(self):
        """Test Union type modernization."""
        modernizer = Python312TypeHintModernizer(Path("homodyne"))

        # Test pattern matching
        line = "def func(x: Union[int, str]) -> Union[bool, None]:"
        modernizations = modernizer._analyze_line_type_hints(line, "test.py", 1)

        assert len(modernizations) == 2
        assert modernizations[0].modernization_type == "union"
        assert modernizations[0].new_hint == "int | str"
        assert modernizations[1].new_hint == "bool | None"

    def test_optional_type_modernization(self):
        """Test Optional type modernization."""
        modernizer = Python312TypeHintModernizer(Path("homodyne"))

        line = "def func(x: Optional[str]) -> Optional[int]:"
        modernizations = modernizer._analyze_line_type_hints(line, "test.py", 1)

        assert len(modernizations) == 2
        assert modernizations[0].modernization_type == "optional"
        assert modernizations[0].new_hint == "str | None"
        assert modernizations[1].new_hint == "int | None"

    def test_generic_collection_modernization(self):
        """Test generic collection modernization."""
        modernizer = Python312TypeHintModernizer(Path("homodyne"))

        line = "def func(x: List[int], y: Dict[str, Any]) -> Set[float]:"
        modernizations = modernizer._analyze_line_type_hints(line, "test.py", 1)

        assert len(modernizations) == 3
        assert any(m.new_hint == "list[int]" for m in modernizations)
        assert any(m.new_hint == "dict[str, Any]" for m in modernizations)
        assert any(m.new_hint == "set[float]" for m in modernizations)

    def test_package_analysis(self):
        """Test package-wide type hint analysis."""
        package_root = Path("homodyne")
        if not package_root.exists():
            pytest.skip("Package directory not found")

        modernizer = Python312TypeHintModernizer(package_root)
        results = modernizer.analyze_package_type_hints()

        assert isinstance(results, TypeHintAnalysisResults)
        assert results.total_files > 0
        # We expect some modernizable hints in a real package
        logger.info(f"Found {results.modernizable_hints} modernizable type hints")

    def test_modernization_report_generation(self):
        """Test modernization report generation."""
        package_root = Path("homodyne")
        if not package_root.exists():
            pytest.skip("Package directory not found")

        modernizer = Python312TypeHintModernizer(package_root)
        results = modernizer.analyze_package_type_hints()
        report = modernizer.generate_modernization_report(results)

        assert isinstance(report, dict)
        assert "summary" in report
        assert "modernization_types" in report
        assert "recommendations" in report

    def test_dry_run_modernization(self):
        """Test dry run modernization."""
        # Create a temporary file with old-style type hints
        import tempfile

        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write("""
from typing import Optional, Union, List

def test_func(x: Optional[str], y: Union[int, float]) -> List[str]:
    return []
""")
            temp_path = f.name

        try:
            modernizer = Python312TypeHintModernizer(Path("homodyne"))
            modernizations = modernizer.analyze_file_type_hints(Path(temp_path))

            # Should find modernizable hints
            assert len(modernizations) > 0

            # Test dry run
            results = modernizer.apply_modernizations(modernizations, dry_run=True)
            assert results["total_modernizations"] > 0

        finally:
            Path(temp_path).unlink()

    def test_syntax_validation(self):
        """Test syntax validation after modernization."""
        import tempfile

        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write("""
def test_func(x: str | None, y: int | float) -> list[str]:
    return []
""")
            temp_path = f.name

        try:
            modernizer = Python312TypeHintModernizer(Path("homodyne"))
            is_valid = modernizer.validate_modernized_syntax(Path(temp_path))
            assert is_valid

        finally:
            Path(temp_path).unlink()


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)

    package_root = Path("homodyne")
    if package_root.exists():
        modernizer = Python312TypeHintModernizer(package_root)

        # Analyze current type hints
        results = modernizer.analyze_package_type_hints()

        # Generate report
        report = modernizer.generate_modernization_report(results)

        # Save report
        import json
        with open("type_hint_modernization_report.json", 'w') as f:
            json.dump(report, f, indent=2)

        print(f"Type hint analysis complete:")
        print(f"  Files analyzed: {results.total_files}")
        print(f"  Modernizable hints: {results.modernizable_hints}")
        print(f"  Performance improvements: {results.performance_improvements}")
        print(f"  Coverage: {results.coverage_percentage:.1f}%")

        # Show sample modernizations
        if results.modernizations:
            print("\nSample modernizations:")
            for mod in results.modernizations[:5]:
                print(f"  {mod.old_hint} -> {mod.new_hint} ({mod.modernization_type})")
    else:
        print("Package directory 'homodyne' not found")
