"""
Advanced Complexity Analysis Tooling and Baseline Measurements
==============================================================

Comprehensive complexity analysis system for measuring, tracking, and
managing code complexity during refactoring activities. Provides detailed
metrics, baselines, and automated monitoring for complexity reduction.

Authors: Wei Chen, Hongrui He
Institution: Argonne National Laboratory
"""

import ast
import json
import logging
from dataclasses import asdict
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

import pytest

logger = logging.getLogger(__name__)


@dataclass
class ComplexityMetrics:
    """Detailed complexity metrics for a function."""
    name: str
    file_path: str
    line_number: int
    cyclomatic_complexity: int
    cognitive_complexity: int
    nesting_depth: int
    parameter_count: int
    line_count: int
    branch_count: int
    loop_count: int
    try_except_count: int
    return_count: int
    class_name: Optional[str] = None


@dataclass
class ComplexityBaseline:
    """Complexity baseline for tracking improvements."""
    name: str
    created_at: str
    package_version: str
    total_functions: int
    high_complexity_count: int
    max_complexity: int
    average_complexity: float
    complexity_distribution: Dict[str, int]
    function_metrics: List[ComplexityMetrics] = field(default_factory=list)


class AdvancedComplexityAnalyzer:
    """Advanced complexity analysis with multiple metrics."""

    def __init__(self, package_root: Path):
        self.package_root = package_root

    def analyze_function_complexity(self, node: ast.FunctionDef, file_path: str,
                                  class_name: Optional[str] = None) -> ComplexityMetrics:
        """Analyze complexity metrics for a single function."""
        visitor = ComplexityVisitor()
        visitor.visit(node)

        return ComplexityMetrics(
            name=node.name,
            file_path=file_path,
            line_number=node.lineno,
            cyclomatic_complexity=visitor.cyclomatic_complexity,
            cognitive_complexity=visitor.cognitive_complexity,
            nesting_depth=visitor.max_nesting_depth,
            parameter_count=len(node.args.args),
            line_count=node.end_lineno - node.lineno + 1 if node.end_lineno else 1,
            branch_count=visitor.branch_count,
            loop_count=visitor.loop_count,
            try_except_count=visitor.try_except_count,
            return_count=visitor.return_count,
            class_name=class_name
        )

    def analyze_package_complexity(self) -> Dict[str, Any]:
        """Analyze complexity across the entire package."""
        logger.info("Starting comprehensive complexity analysis")

        all_metrics = []
        file_count = 0

        for py_file in self.package_root.rglob("*.py"):
            if self._should_skip_file(py_file):
                continue

            file_metrics = self._analyze_file_complexity(py_file)
            all_metrics.extend(file_metrics)
            file_count += 1

        # Calculate aggregate statistics
        complexities = [m.cyclomatic_complexity for m in all_metrics]
        cognitive_complexities = [m.cognitive_complexity for m in all_metrics]

        results = {
            "total_functions": len(all_metrics),
            "files_analyzed": file_count,
            "max_cyclomatic_complexity": max(complexities) if complexities else 0,
            "max_cognitive_complexity": max(cognitive_complexities) if cognitive_complexities else 0,
            "average_cyclomatic_complexity": sum(complexities) / len(complexities) if complexities else 0,
            "average_cognitive_complexity": sum(cognitive_complexities) / len(cognitive_complexities) if cognitive_complexities else 0,
            "high_complexity_functions": [m for m in all_metrics if m.cyclomatic_complexity > 10],
            "very_high_complexity_functions": [m for m in all_metrics if m.cyclomatic_complexity > 20],
            "extremely_high_complexity_functions": [m for m in all_metrics if m.cyclomatic_complexity > 40],
            "all_metrics": all_metrics,
            "complexity_distribution": self._calculate_complexity_distribution(complexities),
            "cognitive_complexity_distribution": self._calculate_complexity_distribution(cognitive_complexities),
            "analysis_timestamp": datetime.now(timezone.utc).isoformat()
        }

        logger.info(f"Analyzed {len(all_metrics)} functions across {file_count} files")
        return results

    def _analyze_file_complexity(self, file_path: Path) -> List[ComplexityMetrics]:
        """Analyze complexity for all functions in a file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            tree = ast.parse(content)
            metrics = []

            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    # Check if function is inside a class
                    class_name = None
                    for parent in ast.walk(tree):
                        if isinstance(parent, ast.ClassDef):
                            for child in ast.walk(parent):
                                if child is node:
                                    class_name = parent.name
                                    break

                    metrics.append(self.analyze_function_complexity(node, str(file_path), class_name))

            return metrics

        except Exception as e:
            logger.warning(f"Failed to analyze {file_path}: {e}")
            return []

    def _should_skip_file(self, file_path: Path) -> bool:
        """Check if file should be skipped."""
        skip_patterns = {
            "__pycache__", ".git", "build", "dist",
            ".pytest_cache", ".mypy_cache", "venv", ".venv"
        }
        return any(pattern in str(file_path) for pattern in skip_patterns)

    def _calculate_complexity_distribution(self, complexities: List[int]) -> Dict[str, int]:
        """Calculate complexity distribution buckets."""
        distribution = {
            "low (1-5)": 0,
            "moderate (6-10)": 0,
            "high (11-20)": 0,
            "very_high (21-40)": 0,
            "extreme (>40)": 0
        }

        for complexity in complexities:
            if complexity <= 5:
                distribution["low (1-5)"] += 1
            elif complexity <= 10:
                distribution["moderate (6-10)"] += 1
            elif complexity <= 20:
                distribution["high (11-20)"] += 1
            elif complexity <= 40:
                distribution["very_high (21-40)"] += 1
            else:
                distribution["extreme (>40)"] += 1

        return distribution


class ComplexityVisitor(ast.NodeVisitor):
    """Enhanced AST visitor for detailed complexity analysis."""

    def __init__(self):
        self.cyclomatic_complexity = 1  # Base complexity
        self.cognitive_complexity = 0
        self.nesting_level = 0
        self.max_nesting_depth = 0
        self.branch_count = 0
        self.loop_count = 0
        self.try_except_count = 0
        self.return_count = 0

    def visit_If(self, node: ast.If) -> None:
        """Visit if statement."""
        self.cyclomatic_complexity += 1
        self.cognitive_complexity += (1 + self.nesting_level)
        self.branch_count += 1

        self.nesting_level += 1
        self.max_nesting_depth = max(self.max_nesting_depth, self.nesting_level)
        self.generic_visit(node)
        self.nesting_level -= 1

    def visit_While(self, node: ast.While) -> None:
        """Visit while loop."""
        self.cyclomatic_complexity += 1
        self.cognitive_complexity += (1 + self.nesting_level)
        self.loop_count += 1

        self.nesting_level += 1
        self.max_nesting_depth = max(self.max_nesting_depth, self.nesting_level)
        self.generic_visit(node)
        self.nesting_level -= 1

    def visit_For(self, node: ast.For) -> None:
        """Visit for loop."""
        self.cyclomatic_complexity += 1
        self.cognitive_complexity += (1 + self.nesting_level)
        self.loop_count += 1

        self.nesting_level += 1
        self.max_nesting_depth = max(self.max_nesting_depth, self.nesting_level)
        self.generic_visit(node)
        self.nesting_level -= 1

    def visit_AsyncFor(self, node: ast.AsyncFor) -> None:
        """Visit async for loop."""
        self.visit_For(node)

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        """Visit except handler."""
        self.cyclomatic_complexity += 1
        self.cognitive_complexity += (1 + self.nesting_level)
        self.try_except_count += 1

        self.nesting_level += 1
        self.max_nesting_depth = max(self.max_nesting_depth, self.nesting_level)
        self.generic_visit(node)
        self.nesting_level -= 1

    def visit_With(self, node: ast.With) -> None:
        """Visit with statement."""
        self.nesting_level += 1
        self.max_nesting_depth = max(self.max_nesting_depth, self.nesting_level)
        self.generic_visit(node)
        self.nesting_level -= 1

    def visit_AsyncWith(self, node: ast.AsyncWith) -> None:
        """Visit async with statement."""
        self.visit_With(node)

    def visit_Return(self, node: ast.Return) -> None:
        """Visit return statement."""
        self.return_count += 1
        if self.nesting_level > 0:
            self.cognitive_complexity += self.nesting_level
        self.generic_visit(node)

    def visit_BoolOp(self, node: ast.BoolOp) -> None:
        """Visit boolean operation (and/or)."""
        # Each additional condition adds to complexity
        self.cyclomatic_complexity += len(node.values) - 1
        self.cognitive_complexity += len(node.values) - 1
        self.generic_visit(node)

    def visit_Compare(self, node: ast.Compare) -> None:
        """Visit comparison operation."""
        # Multiple comparisons in one statement
        if len(node.comparators) > 1:
            self.cognitive_complexity += 1
        self.generic_visit(node)


class ComplexityBaselineManager:
    """Manages complexity baselines for tracking improvements."""

    def __init__(self, package_root: Path):
        self.package_root = package_root
        self.analyzer = AdvancedComplexityAnalyzer(package_root)

    def establish_baseline(self, name: str, description: str = "") -> ComplexityBaseline:
        """Establish a new complexity baseline."""
        logger.info(f"Establishing complexity baseline: {name}")

        analysis_results = self.analyzer.analyze_package_complexity()

        # Get package version
        try:
            import importlib.metadata
            package_version = importlib.metadata.version("homodyne")
        except Exception:
            package_version = "unknown"

        baseline = ComplexityBaseline(
            name=name,
            created_at=datetime.now(timezone.utc).isoformat(),
            package_version=package_version,
            total_functions=analysis_results["total_functions"],
            high_complexity_count=len(analysis_results["high_complexity_functions"]),
            max_complexity=analysis_results["max_cyclomatic_complexity"],
            average_complexity=analysis_results["average_cyclomatic_complexity"],
            complexity_distribution=analysis_results["complexity_distribution"],
            function_metrics=[asdict(m) for m in analysis_results["all_metrics"]]
        )

        # Save baseline
        baseline_path = self.package_root.parent / f"complexity_baseline_{name}.json"
        with open(baseline_path, 'w') as f:
            json.dump(asdict(baseline), f, indent=2)

        logger.info(f"Baseline saved to {baseline_path}")
        return baseline

    def compare_to_baseline(self, baseline_name: str) -> Dict[str, Any]:
        """Compare current complexity to a baseline."""
        baseline_path = self.package_root.parent / f"complexity_baseline_{baseline_name}.json"

        if not baseline_path.exists():
            raise FileNotFoundError(f"Baseline '{baseline_name}' not found at {baseline_path}")

        with open(baseline_path, 'r') as f:
            baseline_data = json.load(f)

        baseline = ComplexityBaseline(**{k: v for k, v in baseline_data.items() if k != 'function_metrics'})

        # Get current analysis
        current_analysis = self.analyzer.analyze_package_complexity()

        comparison = {
            "baseline_name": baseline_name,
            "baseline_date": baseline.created_at,
            "current_date": current_analysis["analysis_timestamp"],
            "improvements": {},
            "regressions": {},
            "summary": {}
        }

        # Compare key metrics
        metrics_comparison = {
            "total_functions": (baseline.total_functions, current_analysis["total_functions"]),
            "high_complexity_count": (baseline.high_complexity_count, len(current_analysis["high_complexity_functions"])),
            "max_complexity": (baseline.max_complexity, current_analysis["max_cyclomatic_complexity"]),
            "average_complexity": (baseline.average_complexity, current_analysis["average_cyclomatic_complexity"])
        }

        for metric, (baseline_value, current_value) in metrics_comparison.items():
            if current_value < baseline_value:
                comparison["improvements"][metric] = {
                    "baseline": baseline_value,
                    "current": current_value,
                    "improvement": baseline_value - current_value,
                    "percentage": ((baseline_value - current_value) / baseline_value * 100) if baseline_value > 0 else 0
                }
            elif current_value > baseline_value:
                comparison["regressions"][metric] = {
                    "baseline": baseline_value,
                    "current": current_value,
                    "regression": current_value - baseline_value,
                    "percentage": ((current_value - baseline_value) / baseline_value * 100) if baseline_value > 0 else 0
                }

        # Overall assessment
        improvement_count = len(comparison["improvements"])
        regression_count = len(comparison["regressions"])

        if improvement_count > regression_count:
            overall_status = "improved"
        elif regression_count > improvement_count:
            overall_status = "regressed"
        else:
            overall_status = "unchanged"

        comparison["summary"] = {
            "overall_status": overall_status,
            "improvements_count": improvement_count,
            "regressions_count": regression_count,
            "target_high_complexity_functions": 10,
            "current_high_complexity_functions": len(current_analysis["high_complexity_functions"]),
            "target_met": len(current_analysis["high_complexity_functions"]) <= 10
        }

        return comparison

    def generate_complexity_report(self, baseline_name: Optional[str] = None) -> Dict[str, Any]:
        """Generate comprehensive complexity report."""
        current_analysis = self.analyzer.analyze_package_complexity()

        report = {
            "report_timestamp": datetime.now(timezone.utc).isoformat(),
            "package_path": str(self.package_root),
            "current_analysis": current_analysis,
            "top_complex_functions": sorted(
                current_analysis["high_complexity_functions"],
                key=lambda x: x.cyclomatic_complexity,
                reverse=True
            )[:20],
            "recommendations": self._generate_recommendations(current_analysis)
        }

        if baseline_name:
            try:
                comparison = self.compare_to_baseline(baseline_name)
                report["baseline_comparison"] = comparison
            except FileNotFoundError:
                report["baseline_comparison"] = {"error": f"Baseline '{baseline_name}' not found"}

        return report

    def _generate_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on complexity analysis."""
        recommendations = []

        high_complexity_count = len(analysis["high_complexity_functions"])
        very_high_count = len(analysis["very_high_complexity_functions"])
        extreme_count = len(analysis["extremely_high_complexity_functions"])

        if extreme_count > 0:
            recommendations.append(f"URGENT: {extreme_count} functions have extreme complexity (>40). Immediate refactoring needed.")

        if very_high_count > 5:
            recommendations.append(f"HIGH PRIORITY: {very_high_count} functions have very high complexity (>20). Plan refactoring.")

        if high_complexity_count > 20:
            recommendations.append(f"MEDIUM PRIORITY: {high_complexity_count} functions have high complexity (>10). Gradual refactoring recommended.")

        if analysis["max_cyclomatic_complexity"] > 50:
            recommendations.append("Consider breaking down the most complex function using extract method pattern.")

        if analysis["average_cyclomatic_complexity"] > 8:
            recommendations.append("Average complexity is high. Focus on reducing overall complexity.")

        # Specific recommendations for top complex functions
        top_functions = sorted(analysis["all_metrics"], key=lambda x: x.cyclomatic_complexity, reverse=True)[:5]
        for func in top_functions:
            if func.cyclomatic_complexity > 30:
                recommendations.append(f"Refactor {func.name} (complexity: {func.cyclomatic_complexity}) in {func.file_path}")

        return recommendations


class ComplexityRefactoringPlanner:
    """Plans refactoring strategies based on complexity analysis."""

    def __init__(self, complexity_report: Dict[str, Any]):
        self.report = complexity_report

    def generate_refactoring_plan(self) -> Dict[str, Any]:
        """Generate a prioritized refactoring plan."""
        high_complexity_functions = self.report["current_analysis"]["high_complexity_functions"]

        # Group by priority
        critical_functions = [f for f in high_complexity_functions if f.cyclomatic_complexity > 40]
        high_priority_functions = [f for f in high_complexity_functions if 20 < f.cyclomatic_complexity <= 40]
        medium_priority_functions = [f for f in high_complexity_functions if 10 < f.cyclomatic_complexity <= 20]

        plan = {
            "plan_created": datetime.now(timezone.utc).isoformat(),
            "total_functions_to_refactor": len(high_complexity_functions),
            "critical_priority": {
                "count": len(critical_functions),
                "functions": [self._create_refactoring_task(f) for f in critical_functions],
                "estimated_effort": len(critical_functions) * 8,  # hours
                "recommended_approach": "Extract method and class patterns"
            },
            "high_priority": {
                "count": len(high_priority_functions),
                "functions": [self._create_refactoring_task(f) for f in high_priority_functions],
                "estimated_effort": len(high_priority_functions) * 4,  # hours
                "recommended_approach": "Extract method pattern"
            },
            "medium_priority": {
                "count": len(medium_priority_functions),
                "functions": [self._create_refactoring_task(f) for f in medium_priority_functions],
                "estimated_effort": len(medium_priority_functions) * 2,  # hours
                "recommended_approach": "Simplify conditionals and loops"
            },
            "total_estimated_effort": (len(critical_functions) * 8 +
                                     len(high_priority_functions) * 4 +
                                     len(medium_priority_functions) * 2)
        }

        return plan

    def _create_refactoring_task(self, func_metrics: ComplexityMetrics) -> Dict[str, Any]:
        """Create a refactoring task for a function."""
        return {
            "function_name": func_metrics.name,
            "file_path": func_metrics.file_path,
            "line_number": func_metrics.line_number,
            "current_complexity": func_metrics.cyclomatic_complexity,
            "target_complexity": min(10, func_metrics.cyclomatic_complexity // 2),
            "estimated_effort_hours": max(2, func_metrics.cyclomatic_complexity // 10),
            "suggested_techniques": self._suggest_refactoring_techniques(func_metrics)
        }

    def _suggest_refactoring_techniques(self, func_metrics: ComplexityMetrics) -> List[str]:
        """Suggest specific refactoring techniques."""
        techniques = []

        if func_metrics.cyclomatic_complexity > 40:
            techniques.append("Extract class - function is too large")
            techniques.append("Extract method - break into smaller functions")

        if func_metrics.parameter_count > 5:
            techniques.append("Parameter object - reduce parameter count")

        if func_metrics.nesting_depth > 4:
            techniques.append("Reduce nesting - use early returns")

        if func_metrics.branch_count > 10:
            techniques.append("Replace conditional with polymorphism")

        if func_metrics.loop_count > 3:
            techniques.append("Extract loop body into separate method")

        if func_metrics.return_count > 5:
            techniques.append("Consolidate return statements")

        return techniques


class TestAdvancedComplexityAnalysis:
    """Test suite for advanced complexity analysis tooling."""

    def test_complexity_analyzer_initialization(self):
        """Test complexity analyzer initialization."""
        package_root = Path("homodyne")
        analyzer = AdvancedComplexityAnalyzer(package_root)

        assert analyzer.package_root == package_root

    def test_complexity_visitor_metrics(self):
        """Test complexity visitor with known code patterns."""
        # Test simple function
        simple_code = """
def simple_function(x):
    return x + 1
"""
        tree = ast.parse(simple_code)
        func_node = tree.body[0]

        visitor = ComplexityVisitor()
        visitor.visit(func_node)

        assert visitor.cyclomatic_complexity == 1
        assert visitor.cognitive_complexity == 0

        # Test complex function
        complex_code = """
def complex_function(x, y):
    if x > 0:
        for i in range(x):
            if i % 2 == 0:
                try:
                    result = i / y
                except ZeroDivisionError:
                    return None
                if result > 10:
                    return result
            else:
                while y > 0:
                    y -= 1
    return 0
"""
        tree = ast.parse(complex_code)
        func_node = tree.body[0]

        visitor = ComplexityVisitor()
        visitor.visit(func_node)

        # Should have high complexity due to nested conditions and loops
        assert visitor.cyclomatic_complexity > 5
        assert visitor.cognitive_complexity > 5

    def test_package_complexity_analysis(self):
        """Test package-wide complexity analysis."""
        package_root = Path("homodyne")
        if not package_root.exists():
            pytest.skip("Package directory not found")

        analyzer = AdvancedComplexityAnalyzer(package_root)
        results = analyzer.analyze_package_complexity()

        assert isinstance(results, dict)
        assert "total_functions" in results
        assert "max_cyclomatic_complexity" in results
        assert "high_complexity_functions" in results
        assert results["total_functions"] > 0

        # Check that we found some high-complexity functions
        assert len(results["high_complexity_functions"]) > 0

    def test_baseline_establishment(self):
        """Test complexity baseline establishment."""
        package_root = Path("homodyne")
        if not package_root.exists():
            pytest.skip("Package directory not found")

        manager = ComplexityBaselineManager(package_root)
        baseline = manager.establish_baseline("test_baseline_task_3_2")

        assert isinstance(baseline, ComplexityBaseline)
        assert baseline.name == "test_baseline_task_3_2"
        assert baseline.total_functions > 0
        assert baseline.max_complexity > 0

        # Clean up
        baseline_path = package_root.parent / f"complexity_baseline_test_baseline_task_3_2.json"
        if baseline_path.exists():
            baseline_path.unlink()

    def test_complexity_report_generation(self):
        """Test complexity report generation."""
        package_root = Path("homodyne")
        if not package_root.exists():
            pytest.skip("Package directory not found")

        manager = ComplexityBaselineManager(package_root)
        report = manager.generate_complexity_report()

        assert isinstance(report, dict)
        assert "current_analysis" in report
        assert "top_complex_functions" in report
        assert "recommendations" in report

    def test_refactoring_plan_generation(self):
        """Test refactoring plan generation."""
        package_root = Path("homodyne")
        if not package_root.exists():
            pytest.skip("Package directory not found")

        manager = ComplexityBaselineManager(package_root)
        report = manager.generate_complexity_report()

        planner = ComplexityRefactoringPlanner(report)
        plan = planner.generate_refactoring_plan()

        assert isinstance(plan, dict)
        assert "critical_priority" in plan
        assert "high_priority" in plan
        assert "medium_priority" in plan
        assert "total_estimated_effort" in plan


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)

    package_root = Path("homodyne")
    if package_root.exists():
        # Establish baseline
        manager = ComplexityBaselineManager(package_root)
        baseline = manager.establish_baseline(
            "task_3_2_initial_baseline",
            "Initial complexity baseline before Task 3 refactoring"
        )

        print(f"Established baseline: {baseline.name}")
        print(f"Total functions: {baseline.total_functions}")
        print(f"High complexity functions: {baseline.high_complexity_count}")
        print(f"Maximum complexity: {baseline.max_complexity}")
        print(f"Average complexity: {baseline.average_complexity:.2f}")

        # Generate complexity report
        report = manager.generate_complexity_report("task_3_2_initial_baseline")

        # Save report
        report_path = Path("complexity_analysis_report.json")
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)

        print(f"\nComplexity report saved to {report_path}")

        # Generate refactoring plan
        planner = ComplexityRefactoringPlanner(report)
        plan = planner.generate_refactoring_plan()

        plan_path = Path("refactoring_plan.json")
        with open(plan_path, 'w') as f:
            json.dump(plan, f, indent=2)

        print(f"Refactoring plan saved to {plan_path}")
        print(f"Total estimated effort: {plan['total_estimated_effort']} hours")

    else:
        print("Package directory 'homodyne' not found")
