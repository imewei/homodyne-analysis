"""
Code Quality Metrics Tracking and Baseline Establishment
=======================================================

Comprehensive code quality metrics system for tracking technical debt,
code complexity, maintainability, and establishing baselines for continuous
improvement of the homodyne analysis package.

Authors: Wei Chen, Hongrui He
Institution: Argonne National Laboratory
"""

import ast
import json
import logging
from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import field
from datetime import UTC
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest

logger = logging.getLogger(__name__)


@dataclass
class CodeQualityMetrics:
    """Comprehensive code quality metrics."""

    timestamp: str
    package_name: str

    # Basic metrics
    total_lines: int
    code_lines: int
    comment_lines: int
    blank_lines: int

    # Complexity metrics
    cyclomatic_complexity: float
    max_complexity: int
    avg_complexity: float
    high_complexity_functions: int

    # Dead code metrics
    unused_imports: int
    unused_functions: int
    unused_classes: int
    unused_variables: int

    # Maintainability metrics
    maintainability_index: float
    technical_debt_ratio: float
    code_duplication: float

    # Type hint coverage
    type_hint_coverage: float
    missing_type_hints: int

    # Documentation coverage
    docstring_coverage: float
    missing_docstrings: int

    # Test coverage
    test_coverage: float
    untested_functions: int

    # Security metrics
    security_issues: int
    high_severity_issues: int

    # Performance metrics
    performance_hotspots: int
    memory_inefficiencies: int


@dataclass
class CodeQualityBaseline:
    """Code quality baseline for tracking improvements."""

    name: str
    created_at: str
    package_version: str
    metrics: CodeQualityMetrics
    targets: dict[str, float] = field(default_factory=dict)
    thresholds: dict[str, float] = field(default_factory=dict)


class ComplexityAnalyzer:
    """Analyzes code complexity metrics."""

    def __init__(self, package_root: Path):
        self.package_root = package_root

    def analyze_complexity(self) -> dict[str, Any]:
        """Analyze cyclomatic complexity for all Python files."""
        complexity_results = {
            "total_functions": 0,
            "max_complexity": 0,
            "complexities": [],
            "high_complexity_functions": [],
            "file_complexities": {},
        }

        for py_file in self.package_root.rglob("*.py"):
            if self._should_skip_file(py_file):
                continue

            file_complexity = self._analyze_file_complexity(py_file)
            if file_complexity:
                complexity_results["file_complexities"][str(py_file)] = file_complexity

                for func_data in file_complexity["functions"]:
                    complexity = func_data["complexity"]
                    complexity_results["complexities"].append(complexity)
                    complexity_results["total_functions"] += 1

                    complexity_results["max_complexity"] = max(
                        complexity_results["max_complexity"], complexity
                    )

                    if complexity > 10:  # High complexity threshold
                        complexity_results["high_complexity_functions"].append(
                            {
                                "file": str(py_file),
                                "function": func_data["name"],
                                "complexity": complexity,
                                "line": func_data["line"],
                            }
                        )

        return complexity_results

    def _analyze_file_complexity(self, file_path: Path) -> dict[str, Any] | None:
        """Analyze complexity for a single file."""
        try:
            with open(file_path, encoding="utf-8") as f:
                content = f.read()

            tree = ast.parse(content)
            analyzer = ComplexityVisitor()
            analyzer.visit(tree)

            return {
                "total_complexity": sum(
                    func["complexity"] for func in analyzer.functions
                ),
                "function_count": len(analyzer.functions),
                "functions": analyzer.functions,
            }
        except Exception as e:
            logger.warning(f"Failed to analyze complexity for {file_path}: {e}")
            return None

    def _should_skip_file(self, file_path: Path) -> bool:
        """Check if file should be skipped."""
        skip_patterns = {
            "__pycache__",
            ".git",
            "build",
            "dist",
            ".pytest_cache",
            ".mypy_cache",
            "venv",
            ".venv",
        }
        return any(pattern in str(file_path) for pattern in skip_patterns)


class ComplexityVisitor(ast.NodeVisitor):
    """AST visitor to calculate cyclomatic complexity."""

    def __init__(self):
        self.functions = []
        self.current_complexity = 0
        self.current_function = None

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Visit function definition and calculate complexity."""
        old_complexity = self.current_complexity
        old_function = self.current_function

        self.current_complexity = 1  # Base complexity
        self.current_function = node.name

        self.generic_visit(node)

        self.functions.append(
            {
                "name": node.name,
                "line": node.lineno,
                "complexity": self.current_complexity,
            }
        )

        self.current_complexity = old_complexity
        self.current_function = old_function

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        """Visit async function definition."""
        self.visit_FunctionDef(node)

    def visit_If(self, node: ast.If) -> None:
        """Visit if statement."""
        self.current_complexity += 1
        self.generic_visit(node)

    def visit_While(self, node: ast.While) -> None:
        """Visit while loop."""
        self.current_complexity += 1
        self.generic_visit(node)

    def visit_For(self, node: ast.For) -> None:
        """Visit for loop."""
        self.current_complexity += 1
        self.generic_visit(node)

    def visit_AsyncFor(self, node: ast.AsyncFor) -> None:
        """Visit async for loop."""
        self.current_complexity += 1
        self.generic_visit(node)

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        """Visit except handler."""
        self.current_complexity += 1
        self.generic_visit(node)

    def visit_With(self, node: ast.With) -> None:
        """Visit with statement."""
        self.current_complexity += 1
        self.generic_visit(node)

    def visit_AsyncWith(self, node: ast.AsyncWith) -> None:
        """Visit async with statement."""
        self.current_complexity += 1
        self.generic_visit(node)


class LineCountAnalyzer:
    """Analyzes line counts and basic metrics."""

    def __init__(self, package_root: Path):
        self.package_root = package_root

    def analyze_lines(self) -> dict[str, int]:
        """Analyze line counts for the package."""
        totals = {
            "total_lines": 0,
            "code_lines": 0,
            "comment_lines": 0,
            "blank_lines": 0,
            "files_analyzed": 0,
        }

        for py_file in self.package_root.rglob("*.py"):
            if self._should_skip_file(py_file):
                continue

            file_stats = self._analyze_file_lines(py_file)
            if file_stats:
                for key in totals:
                    if key != "files_analyzed":
                        totals[key] += file_stats.get(key, 0)
                totals["files_analyzed"] += 1

        return totals

    def _analyze_file_lines(self, file_path: Path) -> dict[str, int] | None:
        """Analyze lines for a single file."""
        try:
            with open(file_path, encoding="utf-8") as f:
                lines = f.readlines()

            stats = {
                "total_lines": len(lines),
                "code_lines": 0,
                "comment_lines": 0,
                "blank_lines": 0,
            }

            for line in lines:
                stripped = line.strip()
                if not stripped:
                    stats["blank_lines"] += 1
                elif stripped.startswith("#"):
                    stats["comment_lines"] += 1
                elif '"""' in stripped or "'''" in stripped:
                    # Simple docstring detection - could be improved
                    stats["comment_lines"] += 1
                else:
                    stats["code_lines"] += 1

            return stats
        except Exception as e:
            logger.warning(f"Failed to analyze lines for {file_path}: {e}")
            return None

    def _should_skip_file(self, file_path: Path) -> bool:
        """Check if file should be skipped."""
        skip_patterns = {
            "__pycache__",
            ".git",
            "build",
            "dist",
            ".pytest_cache",
            ".mypy_cache",
            "venv",
            ".venv",
        }
        return any(pattern in str(file_path) for pattern in skip_patterns)


class TypeHintAnalyzer:
    """Analyzes type hint coverage."""

    def __init__(self, package_root: Path):
        self.package_root = package_root

    def analyze_type_hints(self) -> dict[str, Any]:
        """Analyze type hint coverage."""
        results = {
            "total_functions": 0,
            "functions_with_type_hints": 0,
            "functions_missing_type_hints": 0,
            "coverage_percentage": 0.0,
            "missing_hints": [],
        }

        for py_file in self.package_root.rglob("*.py"):
            if self._should_skip_file(py_file):
                continue

            file_results = self._analyze_file_type_hints(py_file)
            if file_results:
                results["total_functions"] += file_results["total_functions"]
                results["functions_with_type_hints"] += file_results[
                    "functions_with_type_hints"
                ]
                results["missing_hints"].extend(file_results["missing_hints"])

        results["functions_missing_type_hints"] = (
            results["total_functions"] - results["functions_with_type_hints"]
        )

        if results["total_functions"] > 0:
            results["coverage_percentage"] = (
                results["functions_with_type_hints"] / results["total_functions"] * 100
            )

        return results

    def _analyze_file_type_hints(self, file_path: Path) -> dict[str, Any] | None:
        """Analyze type hints for a single file."""
        try:
            with open(file_path, encoding="utf-8") as f:
                content = f.read()

            tree = ast.parse(content)
            visitor = TypeHintVisitor(str(file_path))
            visitor.visit(tree)

            return {
                "total_functions": visitor.total_functions,
                "functions_with_type_hints": visitor.functions_with_type_hints,
                "missing_hints": visitor.missing_hints,
            }
        except Exception as e:
            logger.warning(f"Failed to analyze type hints for {file_path}: {e}")
            return None

    def _should_skip_file(self, file_path: Path) -> bool:
        """Check if file should be skipped."""
        skip_patterns = {
            "__pycache__",
            ".git",
            "build",
            "dist",
            ".pytest_cache",
            ".mypy_cache",
            "venv",
            ".venv",
        }
        return any(pattern in str(file_path) for pattern in skip_patterns)


class TypeHintVisitor(ast.NodeVisitor):
    """AST visitor to analyze type hint coverage."""

    def __init__(self, file_path: str):
        self.file_path = file_path
        self.total_functions = 0
        self.functions_with_type_hints = 0
        self.missing_hints = []

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Visit function definition and check type hints."""
        self.total_functions += 1

        # Check if function has type hints
        has_type_hints = False

        # Check return type annotation
        if node.returns:
            has_type_hints = True

        # Check parameter type annotations
        for arg in node.args.args:
            if arg.annotation:
                has_type_hints = True
                break

        if has_type_hints:
            self.functions_with_type_hints += 1
        else:
            self.missing_hints.append(
                {"file": self.file_path, "function": node.name, "line": node.lineno}
            )

        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        """Visit async function definition."""
        self.visit_FunctionDef(node)


class CodeQualityTracker:
    """Main code quality tracking system."""

    def __init__(self, package_root: Path, config: dict[str, Any] | None = None):
        self.package_root = package_root
        self.config = config or self._get_default_config()
        self.complexity_analyzer = ComplexityAnalyzer(package_root)
        self.line_analyzer = LineCountAnalyzer(package_root)
        self.type_hint_analyzer = TypeHintAnalyzer(package_root)

    def _get_default_config(self) -> dict[str, Any]:
        """Get default configuration."""
        return {
            "complexity_threshold": 10,
            "maintainability_threshold": 70,
            "coverage_threshold": 80,
            "type_hint_threshold": 90,
            "docstring_threshold": 85,
        }

    def collect_metrics(self) -> CodeQualityMetrics:
        """Collect comprehensive code quality metrics."""
        logger.info("Collecting code quality metrics")

        # Basic line analysis
        line_stats = self.line_analyzer.analyze_lines()

        # Complexity analysis
        complexity_stats = self.complexity_analyzer.analyze_complexity()

        # Type hint analysis
        type_hint_stats = self.type_hint_analyzer.analyze_type_hints()

        # Calculate derived metrics
        avg_complexity = (
            sum(complexity_stats["complexities"])
            / len(complexity_stats["complexities"])
            if complexity_stats["complexities"]
            else 0
        )

        # Calculate maintainability index (simplified)
        maintainability_index = self._calculate_maintainability_index(
            line_stats, complexity_stats
        )

        # Get package version
        try:
            import importlib.metadata

            package_version = importlib.metadata.version("homodyne")
        except Exception:
            package_version = "unknown"

        metrics = CodeQualityMetrics(
            timestamp=datetime.now(UTC).isoformat(),
            package_name="homodyne",
            # Basic metrics
            total_lines=line_stats["total_lines"],
            code_lines=line_stats["code_lines"],
            comment_lines=line_stats["comment_lines"],
            blank_lines=line_stats["blank_lines"],
            # Complexity metrics
            cyclomatic_complexity=avg_complexity,
            max_complexity=complexity_stats["max_complexity"],
            avg_complexity=avg_complexity,
            high_complexity_functions=len(
                complexity_stats["high_complexity_functions"]
            ),
            # Dead code metrics (placeholder - would integrate with dead code detector)
            unused_imports=0,  # From previous analysis
            unused_functions=0,  # From previous analysis
            unused_classes=0,  # From previous analysis
            unused_variables=0,  # From previous analysis
            # Maintainability metrics
            maintainability_index=maintainability_index,
            technical_debt_ratio=self._calculate_technical_debt_ratio(complexity_stats),
            code_duplication=0.0,  # Would need duplication detector
            # Type hint coverage
            type_hint_coverage=type_hint_stats["coverage_percentage"],
            missing_type_hints=type_hint_stats["functions_missing_type_hints"],
            # Documentation coverage (placeholder)
            docstring_coverage=0.0,  # Would need docstring analyzer
            missing_docstrings=0,
            # Test coverage (placeholder)
            test_coverage=0.0,  # Would integrate with pytest-cov
            untested_functions=0,
            # Security metrics (placeholder)
            security_issues=0,  # Would integrate with bandit
            high_severity_issues=0,
            # Performance metrics (placeholder)
            performance_hotspots=0,  # Would integrate with profiler
            memory_inefficiencies=0,
        )

        logger.info(f"Collected metrics for {line_stats['files_analyzed']} files")
        return metrics

    def _calculate_maintainability_index(
        self, line_stats: dict[str, int], complexity_stats: dict[str, Any]
    ) -> float:
        """Calculate maintainability index."""
        # Simplified maintainability index calculation
        # Real implementation would use Halstead metrics

        if line_stats["code_lines"] == 0:
            return 100.0

        complexity_penalty = complexity_stats["max_complexity"] * 2
        comment_ratio = line_stats["comment_lines"] / line_stats["total_lines"] * 100

        # Base score minus complexity penalty plus comment bonus
        maintainability = max(0, 100 - complexity_penalty + comment_ratio)
        return min(100.0, maintainability)

    def _calculate_technical_debt_ratio(
        self, complexity_stats: dict[str, Any]
    ) -> float:
        """Calculate technical debt ratio."""
        total_functions = complexity_stats["total_functions"]
        if total_functions == 0:
            return 0.0

        high_complexity_count = len(complexity_stats["high_complexity_functions"])
        return (high_complexity_count / total_functions) * 100

    def establish_baseline(self, name: str) -> CodeQualityBaseline:
        """Establish a code quality baseline."""
        metrics = self.collect_metrics()

        # Define improvement targets
        targets = {
            "maintainability_index": min(95.0, metrics.maintainability_index + 10),
            "type_hint_coverage": min(100.0, metrics.type_hint_coverage + 15),
            "technical_debt_ratio": max(0.0, metrics.technical_debt_ratio - 5),
            "avg_complexity": max(1.0, metrics.avg_complexity - 1),
        }

        # Define quality thresholds
        thresholds = {
            "maintainability_index": self.config["maintainability_threshold"],
            "type_hint_coverage": self.config["type_hint_threshold"],
            "max_complexity": self.config["complexity_threshold"],
            "technical_debt_ratio": 20.0,  # Max 20% high complexity functions
        }

        baseline = CodeQualityBaseline(
            name=name,
            created_at=metrics.timestamp,
            package_version="1.0.0",  # Current version
            metrics=metrics,
            targets=targets,
            thresholds=thresholds,
        )

        # Save baseline
        baseline_path = Path(f"code_quality_baseline_{name}.json")
        with open(baseline_path, "w") as f:
            json.dump(asdict(baseline), f, indent=2)

        logger.info(f"Established baseline '{name}' - saved to {baseline_path}")
        return baseline

    def compare_to_baseline(self, baseline_name: str) -> dict[str, Any]:
        """Compare current metrics to baseline."""
        baseline_path = Path(f"code_quality_baseline_{baseline_name}.json")
        if not baseline_path.exists():
            raise FileNotFoundError(f"Baseline '{baseline_name}' not found")

        with open(baseline_path) as f:
            baseline_data = json.load(f)

        current_metrics = self.collect_metrics()
        baseline_metrics = CodeQualityMetrics(**baseline_data["metrics"])

        comparison = {
            "baseline_name": baseline_name,
            "baseline_date": baseline_data["created_at"],
            "current_date": current_metrics.timestamp,
            "improvements": {},
            "regressions": {},
            "targets_met": {},
            "thresholds_passed": {},
            "overall_score": 0.0,
        }

        # Compare key metrics
        key_metrics = [
            "maintainability_index",
            "type_hint_coverage",
            "technical_debt_ratio",
            "avg_complexity",
            "high_complexity_functions",
        ]

        total_score = 0
        for metric in key_metrics:
            current_value = getattr(current_metrics, metric)
            baseline_value = getattr(baseline_metrics, metric)

            if current_value > baseline_value:
                comparison["improvements"][metric] = {
                    "current": current_value,
                    "baseline": baseline_value,
                    "improvement": current_value - baseline_value,
                }
            elif current_value < baseline_value:
                comparison["regressions"][metric] = {
                    "current": current_value,
                    "baseline": baseline_value,
                    "regression": baseline_value - current_value,
                }

            # Check targets
            if metric in baseline_data["targets"]:
                target = baseline_data["targets"][metric]
                comparison["targets_met"][metric] = current_value >= target
                if comparison["targets_met"][metric]:
                    total_score += 1

            # Check thresholds
            if metric in baseline_data["thresholds"]:
                threshold = baseline_data["thresholds"][metric]
                comparison["thresholds_passed"][metric] = current_value >= threshold
                if comparison["thresholds_passed"][metric]:
                    total_score += 1

        comparison["overall_score"] = total_score / (len(key_metrics) * 2) * 100

        return comparison

    def generate_quality_report(
        self, baseline_name: str | None = None
    ) -> dict[str, Any]:
        """Generate comprehensive quality report."""
        current_metrics = self.collect_metrics()

        report = {
            "report_generated_at": datetime.now(UTC).isoformat(),
            "package_name": "homodyne",
            "current_metrics": asdict(current_metrics),
            "summary": {
                "total_lines": current_metrics.total_lines,
                "code_quality_score": current_metrics.maintainability_index,
                "complexity_score": 100 - current_metrics.technical_debt_ratio,
                "type_coverage_score": current_metrics.type_hint_coverage,
            },
        }

        if baseline_name:
            try:
                comparison = self.compare_to_baseline(baseline_name)
                report["baseline_comparison"] = comparison
            except FileNotFoundError:
                report["baseline_comparison"] = {
                    "error": f"Baseline '{baseline_name}' not found"
                }

        # Add recommendations
        recommendations = []

        if current_metrics.maintainability_index < 70:
            recommendations.append(
                "Improve code maintainability by reducing complexity"
            )

        if current_metrics.type_hint_coverage < 80:
            recommendations.append("Add type hints to improve code clarity")

        if current_metrics.technical_debt_ratio > 20:
            recommendations.append("Refactor high-complexity functions")

        if current_metrics.high_complexity_functions > 10:
            recommendations.append("Break down complex functions into smaller ones")

        report["recommendations"] = recommendations

        return report


class TestCodeQualityMetrics:
    """Test suite for code quality metrics system."""

    def test_complexity_analyzer(self):
        """Test complexity analysis."""
        package_root = Path("homodyne")
        if not package_root.exists():
            pytest.skip("Package directory not found")

        analyzer = ComplexityAnalyzer(package_root)
        results = analyzer.analyze_complexity()

        assert isinstance(results, dict)
        assert "total_functions" in results
        assert "max_complexity" in results
        assert "complexities" in results
        assert results["total_functions"] > 0

    def test_line_count_analyzer(self):
        """Test line count analysis."""
        package_root = Path("homodyne")
        if not package_root.exists():
            pytest.skip("Package directory not found")

        analyzer = LineCountAnalyzer(package_root)
        results = analyzer.analyze_lines()

        assert isinstance(results, dict)
        assert "total_lines" in results
        assert "code_lines" in results
        assert "comment_lines" in results
        assert results["total_lines"] > 0

    def test_type_hint_analyzer(self):
        """Test type hint analysis."""
        package_root = Path("homodyne")
        if not package_root.exists():
            pytest.skip("Package directory not found")

        analyzer = TypeHintAnalyzer(package_root)
        results = analyzer.analyze_type_hints()

        assert isinstance(results, dict)
        assert "total_functions" in results
        assert "coverage_percentage" in results
        assert results["total_functions"] > 0

    def test_quality_tracker(self):
        """Test quality tracking system."""
        package_root = Path("homodyne")
        if not package_root.exists():
            pytest.skip("Package directory not found")

        tracker = CodeQualityTracker(package_root)
        metrics = tracker.collect_metrics()

        assert isinstance(metrics, CodeQualityMetrics)
        assert metrics.total_lines > 0
        assert metrics.package_name == "homodyne"

    def test_baseline_establishment(self):
        """Test baseline establishment."""
        package_root = Path("homodyne")
        if not package_root.exists():
            pytest.skip("Package directory not found")

        tracker = CodeQualityTracker(package_root)
        baseline = tracker.establish_baseline("test_baseline")

        assert isinstance(baseline, CodeQualityBaseline)
        assert baseline.name == "test_baseline"
        assert baseline.metrics.total_lines > 0

        # Clean up
        baseline_path = Path("code_quality_baseline_test_baseline.json")
        if baseline_path.exists():
            baseline_path.unlink()

    def test_quality_report_generation(self):
        """Test quality report generation."""
        package_root = Path("homodyne")
        if not package_root.exists():
            pytest.skip("Package directory not found")

        tracker = CodeQualityTracker(package_root)
        report = tracker.generate_quality_report()

        assert isinstance(report, dict)
        assert "current_metrics" in report
        assert "summary" in report
        assert "recommendations" in report


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)

    package_root = Path("homodyne")
    if package_root.exists():
        tracker = CodeQualityTracker(package_root)

        # Establish baseline
        baseline = tracker.establish_baseline("phase2_dead_code_cleanup")
        print(f"Established baseline: {baseline.name}")

        # Generate report
        report = tracker.generate_quality_report("phase2_dead_code_cleanup")

        # Save report
        report_path = Path("code_quality_report.json")
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)

        print(f"Quality report saved to {report_path}")
        print(f"Quality score: {report['summary']['code_quality_score']:.1f}")
        print(f"Type coverage: {report['summary']['type_coverage_score']:.1f}%")
    else:
        print("Package directory 'homodyne' not found")
