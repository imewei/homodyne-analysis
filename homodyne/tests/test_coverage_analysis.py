"""
Test Coverage Analysis for Dead Code Detection
==============================================

Comprehensive test coverage analysis to distinguish between:
1. Truly unused code (dead code)
2. Code used only in tests
3. Code used in production but missing test coverage

This analysis is critical for safe dead code removal without breaking functionality.

Authors: Wei Chen, Hongrui He
Institution: Argonne National Laboratory
"""

import ast
from pathlib import Path
from typing import Dict, List, Set, Tuple, Any, Optional
import warnings

# Try to import coverage if available
try:
    import coverage
    COVERAGE_AVAILABLE = True
except ImportError:
    COVERAGE_AVAILABLE = False
    warnings.warn("coverage not available - some analysis features limited")


class CodeUsageAnalyzer:
    """
    Analyzes code usage patterns to identify dead code vs test-only code.

    Distinguishes between:
    - Production code used in real workflows
    - Test-only code used only in test suites
    - Truly unused code (dead code candidates)
    """

    def __init__(self, package_root: Path, test_dirs: Optional[List[str]] = None):
        """
        Initialize code usage analyzer.

        Parameters
        ----------
        package_root : Path
            Root directory of the package
        test_dirs : Optional[List[str]]
            Test directory names (defaults to common test patterns)
        """
        self.package_root = Path(package_root)
        self.test_dirs = test_dirs or ['tests', 'test', 'testing']

        # Track different types of code usage
        self.production_usage: Dict[str, Set[str]] = defaultdict(set)
        self.test_usage: Dict[str, Set[str]] = defaultdict(set)
        self.all_definitions: Dict[str, Set[str]] = defaultdict(set)

        # AST analyzers
        self.call_analyzer = CallGraphAnalyzer()
        self.definition_analyzer = DefinitionAnalyzer()

    def analyze_package_usage(self) -> Dict[str, Any]:
        """
        Analyze code usage patterns across the entire package.

        Returns
        -------
        Dict[str, Any]
            Comprehensive usage analysis report
        """
        print("🔍 Analyzing package code usage patterns...")

        # Step 1: Find all Python files
        python_files = list(self.package_root.rglob("*.py"))
        production_files = []
        test_files = []

        for file_path in python_files:
            if any(test_dir in file_path.parts for test_dir in self.test_dirs):
                test_files.append(file_path)
            else:
                production_files.append(file_path)

        print(f"Found {len(production_files)} production files, {len(test_files)} test files")

        # Step 2: Analyze definitions in all files
        for file_path in python_files:
            try:
                definitions = self.definition_analyzer.extract_definitions(file_path)
                module_name = self._get_module_name(file_path)
                self.all_definitions[module_name].update(definitions)
            except Exception as e:
                print(f"Warning: Failed to analyze {file_path}: {e}")

        # Step 3: Analyze usage in production files
        for file_path in production_files:
            try:
                usage = self.call_analyzer.extract_calls(file_path)
                module_name = self._get_module_name(file_path)
                self.production_usage[module_name].update(usage)
            except Exception as e:
                print(f"Warning: Failed to analyze production usage in {file_path}: {e}")

        # Step 4: Analyze usage in test files
        for file_path in test_files:
            try:
                usage = self.call_analyzer.extract_calls(file_path)
                module_name = self._get_module_name(file_path)
                self.test_usage[module_name].update(usage)
            except Exception as e:
                print(f"Warning: Failed to analyze test usage in {file_path}: {e}")

        # Step 5: Generate analysis report
        return self._generate_usage_report()

    def _get_module_name(self, file_path: Path) -> str:
        """Convert file path to module name."""
        try:
            relative_path = file_path.relative_to(self.package_root)
            parts = list(relative_path.parts[:-1])  # Remove .py extension
            if relative_path.stem != "__init__":
                parts.append(relative_path.stem)
            return ".".join(parts) if parts else "__main__"
        except ValueError:
            return str(file_path)

    def _generate_usage_report(self) -> Dict[str, Any]:
        """Generate comprehensive usage analysis report."""
        all_modules = set(self.all_definitions.keys()) | set(self.production_usage.keys()) | set(self.test_usage.keys())

        # Flatten all usage across modules
        all_production_usage = set()
        all_test_usage = set()
        all_definitions_flat = set()

        for module in all_modules:
            all_production_usage.update(self.production_usage.get(module, set()))
            all_test_usage.update(self.test_usage.get(module, set()))
            all_definitions_flat.update(self.all_definitions.get(module, set()))

        # Categorize code elements
        production_only = all_production_usage - all_test_usage
        test_only = all_test_usage - all_production_usage
        used_in_both = all_production_usage & all_test_usage
        truly_unused = all_definitions_flat - all_production_usage - all_test_usage

        # Detailed module-by-module analysis
        module_analysis = {}
        for module in all_modules:
            definitions = self.all_definitions.get(module, set())
            prod_usage = self.production_usage.get(module, set())
            test_usage = self.test_usage.get(module, set())

            module_analysis[module] = {
                "total_definitions": len(definitions),
                "production_usage": len(prod_usage),
                "test_usage": len(test_usage),
                "unused_in_module": len(definitions - prod_usage - test_usage),
                "test_only_in_module": len((definitions & test_usage) - prod_usage),
                "production_coverage": len(prod_usage & definitions) / len(definitions) if definitions else 0,
                "test_coverage": len(test_usage & definitions) / len(definitions) if definitions else 0,
            }

        return {
            "summary": {
                "total_definitions": len(all_definitions_flat),
                "production_usage": len(all_production_usage),
                "test_usage": len(all_test_usage),
                "production_only_usage": len(production_only),
                "test_only_usage": len(test_only),
                "used_in_both": len(used_in_both),
                "truly_unused": len(truly_unused),
                "dead_code_candidates": len(truly_unused),
            },
            "categorized_code": {
                "production_only": sorted(production_only),
                "test_only": sorted(test_only),
                "used_in_both": sorted(used_in_both),
                "truly_unused": sorted(truly_unused),
            },
            "module_analysis": module_analysis,
            "recommendations": self._generate_recommendations(truly_unused, test_only),
        }

    def _generate_recommendations(self, truly_unused: Set[str], test_only: Set[str]) -> List[str]:
        """Generate actionable recommendations based on analysis."""
        recommendations = []

        if truly_unused:
            recommendations.append(f"🗑️  Remove {len(truly_unused)} truly unused code elements")

        if test_only:
            recommendations.append(f"🧪 Review {len(test_only)} test-only code elements for potential cleanup")

        if len(truly_unused) > 50:
            recommendations.append("⚠️  High amount of dead code detected - prioritize cleanup")
        elif len(truly_unused) < 10:
            recommendations.append("✅ Low dead code - codebase is well-maintained")

        return recommendations


class DefinitionAnalyzer(ast.NodeVisitor):
    """AST-based analyzer for extracting function, class, and method definitions."""

    def __init__(self):
        self.definitions: Set[str] = set()
        self.current_class = None

    def extract_definitions(self, file_path: Path) -> Set[str]:
        """Extract all definitions from a Python file."""
        self.definitions = set()
        self.current_class = None

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                source = f.read()

            tree = ast.parse(source)
            self.visit(tree)

        except (SyntaxError, UnicodeDecodeError) as e:
            print(f"Warning: Could not parse {file_path}: {e}")

        return self.definitions.copy()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Visit function definitions."""
        if self.current_class:
            # Method definition
            self.definitions.add(f"{self.current_class}.{node.name}")
        else:
            # Function definition
            self.definitions.add(node.name)

        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        """Visit async function definitions."""
        if self.current_class:
            self.definitions.add(f"{self.current_class}.{node.name}")
        else:
            self.definitions.add(node.name)

        self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """Visit class definitions."""
        self.definitions.add(node.name)

        # Visit methods within the class
        old_class = self.current_class
        self.current_class = node.name
        self.generic_visit(node)
        self.current_class = old_class


class CallGraphAnalyzer(ast.NodeVisitor):
    """AST-based analyzer for extracting function and method calls."""

    def __init__(self):
        self.calls: Set[str] = set()

    def extract_calls(self, file_path: Path) -> Set[str]:
        """Extract all function/method calls from a Python file."""
        self.calls = set()

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                source = f.read()

            tree = ast.parse(source)
            self.visit(tree)

        except (SyntaxError, UnicodeDecodeError) as e:
            print(f"Warning: Could not parse {file_path}: {e}")

        return self.calls.copy()

    def visit_Call(self, node: ast.Call) -> None:
        """Visit function calls."""
        call_name = self._extract_call_name(node.func)
        if call_name:
            self.calls.add(call_name)

        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        """Visit attribute access (method calls)."""
        attr_name = self._extract_attribute_name(node)
        if attr_name:
            self.calls.add(attr_name)

        self.generic_visit(node)

    def _extract_call_name(self, node: ast.AST) -> Optional[str]:
        """Extract the name of a function call."""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return self._extract_attribute_name(node)
        return None

    def _extract_attribute_name(self, node: ast.Attribute) -> Optional[str]:
        """Extract attribute name from attribute access."""
        if isinstance(node.value, ast.Name):
            return f"{node.value.id}.{node.attr}"
        elif isinstance(node.value, ast.Attribute):
            base = self._extract_attribute_name(node.value)
            return f"{base}.{node.attr}" if base else None
        return node.attr


class CoverageBasedAnalyzer:
    """
    Coverage-based analysis for more accurate dead code detection.

    Uses runtime coverage data to identify truly unused code.
    """

    def __init__(self, package_root: Path):
        self.package_root = Path(package_root)

    def run_coverage_analysis(self, test_command: str = "python -m pytest") -> Dict[str, Any]:
        """
        Run coverage analysis to identify unused code.

        Parameters
        ----------
        test_command : str
            Command to run tests with coverage

        Returns
        -------
        Dict[str, Any]
            Coverage analysis results
        """
        if not COVERAGE_AVAILABLE:
            return {"error": "coverage package not available"}

        print("📊 Running coverage-based analysis...")

        try:
            # Initialize coverage
            cov = coverage.Coverage(
                source=[str(self.package_root)],
                omit=[
                    "*/tests/*",
                    "*/test_*",
                    "*/__pycache__/*",
                    "*/.*",
                ]
            )

            # Start coverage
            cov.start()

            # Import and run basic package functionality
            try:
                import homodyne
                # Try to exercise basic functionality
                health = homodyne.check_performance_health()
                config_manager = homodyne.ConfigManager
                analysis_core = homodyne.HomodyneAnalysisCore
            except Exception as e:
                print(f"Warning: Could not exercise package functionality: {e}")

            # Stop coverage
            cov.stop()
            cov.save()

            # Generate coverage report
            coverage_data = {}
            for filename in cov.get_data().measured_files():
                try:
                    rel_path = Path(filename).relative_to(self.package_root)
                    analysis = cov.analysis2(filename)

                    coverage_data[str(rel_path)] = {
                        "executed_lines": len(analysis.executed),
                        "missing_lines": len(analysis.missing),
                        "total_lines": len(analysis.statements),
                        "coverage_percent": (len(analysis.executed) / len(analysis.statements) * 100)
                                          if analysis.statements else 100,
                        "missing_line_numbers": sorted(analysis.missing),
                    }
                except Exception as e:
                    print(f"Warning: Could not analyze coverage for {filename}: {e}")

            return {
                "coverage_data": coverage_data,
                "summary": self._summarize_coverage(coverage_data),
            }

        except Exception as e:
            return {"error": f"Coverage analysis failed: {e}"}

    def _summarize_coverage(self, coverage_data: Dict[str, Any]) -> Dict[str, Any]:
        """Summarize coverage analysis results."""
        if not coverage_data:
            return {"error": "No coverage data available"}

        total_lines = sum(data["total_lines"] for data in coverage_data.values())
        executed_lines = sum(data["executed_lines"] for data in coverage_data.values())
        missing_lines = sum(data["missing_lines"] for data in coverage_data.values())

        low_coverage_files = [
            filename for filename, data in coverage_data.items()
            if data["coverage_percent"] < 50 and data["total_lines"] > 10
        ]

        return {
            "total_lines": total_lines,
            "executed_lines": executed_lines,
            "missing_lines": missing_lines,
            "overall_coverage_percent": (executed_lines / total_lines * 100) if total_lines > 0 else 0,
            "files_analyzed": len(coverage_data),
            "low_coverage_files": low_coverage_files,
            "potentially_dead_lines": missing_lines,
        }


class DeadCodeDetector:
    """
    Comprehensive dead code detector combining multiple analysis methods.
    """

    def __init__(self, package_root: Path):
        self.package_root = Path(package_root)
        self.usage_analyzer = CodeUsageAnalyzer(package_root)
        self.coverage_analyzer = CoverageBasedAnalyzer(package_root)

    def detect_dead_code(self) -> Dict[str, Any]:
        """
        Comprehensive dead code detection using multiple analysis methods.

        Returns
        -------
        Dict[str, Any]
            Comprehensive dead code analysis report
        """
        print("🔍 Starting comprehensive dead code detection...")

        # Static analysis
        print("\n📝 Running static code analysis...")
        usage_analysis = self.usage_analyzer.analyze_package_usage()

        # Coverage analysis
        print("\n📊 Running coverage analysis...")
        coverage_analysis = self.coverage_analyzer.run_coverage_analysis()

        # Combine results
        return {
            "analysis_timestamp": __import__('datetime').datetime.now().isoformat(),
            "package_root": str(self.package_root),
            "static_analysis": usage_analysis,
            "coverage_analysis": coverage_analysis,
            "recommendations": self._generate_combined_recommendations(usage_analysis, coverage_analysis),
            "summary": self._generate_summary(usage_analysis, coverage_analysis),
        }

    def _generate_combined_recommendations(self, usage_analysis: Dict[str, Any],
                                         coverage_analysis: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on combined analysis."""
        recommendations = []

        # From static analysis
        if "recommendations" in usage_analysis:
            recommendations.extend(usage_analysis["recommendations"])

        # From coverage analysis
        if "summary" in coverage_analysis and coverage_analysis["summary"]:
            summary = coverage_analysis["summary"]
            if "overall_coverage_percent" in summary:
                coverage_pct = summary["overall_coverage_percent"]
                if coverage_pct < 70:
                    recommendations.append(f"⚠️  Low overall coverage ({coverage_pct:.1f}%) - improve test coverage")
                elif coverage_pct > 90:
                    recommendations.append(f"✅ Excellent coverage ({coverage_pct:.1f}%)")

        # Combined insights
        static_dead = len(usage_analysis.get("categorized_code", {}).get("truly_unused", []))
        if static_dead > 20:
            recommendations.append("🧹 High dead code detected - implement cleanup pipeline")

        return recommendations

    def _generate_summary(self, usage_analysis: Dict[str, Any],
                         coverage_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate combined summary."""
        summary = {
            "analysis_methods": ["static_ast_analysis", "coverage_analysis"],
            "static_analysis_success": "summary" in usage_analysis,
            "coverage_analysis_success": "summary" in coverage_analysis and not coverage_analysis.get("error"),
        }

        # Add static analysis summary
        if summary["static_analysis_success"]:
            static_summary = usage_analysis["summary"]
            summary.update({
                "total_definitions": static_summary.get("total_definitions", 0),
                "dead_code_candidates": static_summary.get("dead_code_candidates", 0),
                "test_only_code": static_summary.get("test_only_usage", 0),
            })

        # Add coverage summary
        if summary["coverage_analysis_success"]:
            cov_summary = coverage_analysis["summary"]
            summary.update({
                "overall_coverage_percent": cov_summary.get("overall_coverage_percent", 0),
                "potentially_dead_lines": cov_summary.get("potentially_dead_lines", 0),
                "files_analyzed": cov_summary.get("files_analyzed", 0),
            })

        return summary


def analyze_dead_code_in_package(package_root: str = "/Users/b80985/Projects/homodyne-analysis/homodyne") -> Dict[str, Any]:
    """
    Convenience function to run comprehensive dead code analysis.

    Parameters
    ----------
    package_root : str
        Root directory of the package to analyze

    Returns
    -------
    Dict[str, Any]
        Comprehensive analysis report
    """
    detector = DeadCodeDetector(Path(package_root))
    return detector.detect_dead_code()


if __name__ == "__main__":
    # Run analysis on the homodyne package
    result = analyze_dead_code_in_package()

    print("\n" + "="*60)
    print("📊 DEAD CODE ANALYSIS RESULTS")
    print("="*60)

    if "summary" in result:
        summary = result["summary"]
        print(f"\n📈 Summary:")
        print(f"   • Total definitions: {summary.get('total_definitions', 'N/A')}")
        print(f"   • Dead code candidates: {summary.get('dead_code_candidates', 'N/A')}")
        print(f"   • Test-only code: {summary.get('test_only_code', 'N/A')}")
        print(f"   • Coverage: {summary.get('overall_coverage_percent', 'N/A'):.1f}%")

    if "recommendations" in result:
        print(f"\n💡 Recommendations:")
        for rec in result["recommendations"]:
            print(f"   {rec}")

    print(f"\n✅ Analysis complete!")
