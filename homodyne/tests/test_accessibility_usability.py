"""
Accessibility and Usability Testing Framework
=============================================

Comprehensive accessibility and usability testing framework for Task 5.5.
Ensures software accessibility compliance and optimal user experience.

Authors: Wei Chen, Hongrui He
Institution: Argonne National Laboratory
"""

import json
import re
import time
import warnings
from dataclasses import asdict
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

warnings.filterwarnings("ignore")


class AccessibilityStandard(Enum):
    """Accessibility standards."""

    WCAG_2_1_A = "WCAG 2.1 Level A"
    WCAG_2_1_AA = "WCAG 2.1 Level AA"
    WCAG_2_1_AAA = "WCAG 2.1 Level AAA"
    SECTION_508 = "Section 508"
    ADA = "Americans with Disabilities Act"


class UsabilityMetric(Enum):
    """Usability metrics."""

    LEARNABILITY = "learnability"
    EFFICIENCY = "efficiency"
    MEMORABILITY = "memorability"
    ERROR_PREVENTION = "error_prevention"
    SATISFACTION = "satisfaction"


@dataclass
class AccessibilityIssue:
    """Accessibility issue data structure."""

    issue_id: str
    severity: str  # "critical", "major", "minor", "info"
    standard: AccessibilityStandard
    guideline: str
    description: str
    affected_component: str
    suggested_fix: str
    automated_fixable: bool = False


@dataclass
class UsabilityTest:
    """Usability test result."""

    test_name: str
    metric: UsabilityMetric
    score: float  # 0-100
    benchmark: float
    passed: bool
    recommendations: list[str]
    execution_time: float


class AccessibilityTester:
    """Advanced accessibility testing system."""

    def __init__(self):
        self.accessibility_issues = []
        self.standards_compliance = {}

    def test_cli_accessibility(self) -> list[AccessibilityIssue]:
        """Test CLI accessibility features."""
        issues = []

        # Test color usage in CLI output
        issues.extend(self._test_color_accessibility())

        # Test text readability
        issues.extend(self._test_text_readability())

        # Test keyboard navigation
        issues.extend(self._test_keyboard_accessibility())

        # Test screen reader compatibility
        issues.extend(self._test_screen_reader_compatibility())

        # Test alternative text and descriptions
        issues.extend(self._test_alternative_descriptions())

        self.accessibility_issues.extend(issues)
        return issues

    def _test_color_accessibility(self) -> list[AccessibilityIssue]:
        """Test color accessibility in CLI output."""
        issues = []

        # Check for ANSI color codes in CLI tools
        cli_files = list(Path().glob("**/cli/*.py"))

        for cli_file in cli_files[:5]:  # Limit for demo
            try:
                with open(cli_file, encoding="utf-8") as f:
                    content = f.read()

                # Look for color codes without alternative indicators
                color_patterns = [
                    r"\\033\[\d+m",  # ANSI escape codes
                    r"\\x1b\[\d+m",  # Hex escape codes
                    r"colorama\.",  # Colorama usage
                    r"termcolor\.",  # Termcolor usage
                ]

                has_color = any(
                    re.search(pattern, content) for pattern in color_patterns
                )
                has_alternative = any(
                    term in content.lower()
                    for term in ["alt", "describe", "label", "announce"]
                )

                if has_color and not has_alternative:
                    issues.append(
                        AccessibilityIssue(
                            issue_id=f"color_accessibility_{cli_file.name}",
                            severity="major",
                            standard=AccessibilityStandard.WCAG_2_1_AA,
                            guideline="1.4.1 Use of Color",
                            description="Color is used as the only visual means of conveying information",
                            affected_component=str(cli_file),
                            suggested_fix="Provide alternative text descriptions or symbols alongside color coding",
                            automated_fixable=False,
                        )
                    )

            except Exception:
                continue

        return issues

    def _test_text_readability(self) -> list[AccessibilityIssue]:
        """Test text readability and clarity."""
        issues = []

        # Check documentation and help text
        doc_files = list(Path().glob("**/*.md")) + list(Path().glob("**/*.rst"))

        for doc_file in doc_files[:10]:  # Limit for demo
            try:
                with open(doc_file, encoding="utf-8") as f:
                    content = f.read()

                # Check for overly complex sentences
                sentences = re.split(r"[.!?]+", content)
                complex_sentences = [s for s in sentences if len(s.split()) > 30]

                if (
                    len(complex_sentences) > len(sentences) * 0.3
                ):  # >30% complex sentences
                    issues.append(
                        AccessibilityIssue(
                            issue_id=f"readability_{doc_file.name}",
                            severity="minor",
                            standard=AccessibilityStandard.WCAG_2_1_AAA,
                            guideline="3.1.5 Reading Level",
                            description="Documentation contains many complex sentences that may be difficult to read",
                            affected_component=str(doc_file),
                            suggested_fix="Break down complex sentences into simpler, shorter sentences",
                            automated_fixable=False,
                        )
                    )

                # Check for missing headings structure
                if "# " in content and "## " not in content:
                    issues.append(
                        AccessibilityIssue(
                            issue_id=f"heading_structure_{doc_file.name}",
                            severity="major",
                            standard=AccessibilityStandard.WCAG_2_1_AA,
                            guideline="1.3.1 Info and Relationships",
                            description="Document lacks proper heading hierarchy",
                            affected_component=str(doc_file),
                            suggested_fix="Add subheadings to create proper document structure",
                            automated_fixable=True,
                        )
                    )

            except Exception:
                continue

        return issues

    def _test_keyboard_accessibility(self) -> list[AccessibilityIssue]:
        """Test keyboard navigation and accessibility."""
        issues = []

        # Check for keyboard shortcuts documentation
        cli_files = list(Path().glob("**/cli/*.py"))

        keyboard_features_found = False
        for cli_file in cli_files[:5]:
            try:
                with open(cli_file, encoding="utf-8") as f:
                    content = f.read()

                # Look for keyboard handling
                keyboard_indicators = [
                    "input(",
                    "raw_input(",
                    "click.option",
                    "argparse",
                    "keyboard",
                    "hotkey",
                    "shortcut",
                ]

                if any(indicator in content for indicator in keyboard_indicators):
                    keyboard_features_found = True
                    break

            except Exception:
                continue

        if not keyboard_features_found:
            issues.append(
                AccessibilityIssue(
                    issue_id="keyboard_navigation_missing",
                    severity="major",
                    standard=AccessibilityStandard.WCAG_2_1_AA,
                    guideline="2.1.1 Keyboard",
                    description="No clear keyboard navigation or input handling found",
                    affected_component="CLI interface",
                    suggested_fix="Implement comprehensive keyboard navigation and shortcuts",
                    automated_fixable=False,
                )
            )

        return issues

    def _test_screen_reader_compatibility(self) -> list[AccessibilityIssue]:
        """Test screen reader compatibility."""
        issues = []

        # Check for descriptive output and labels
        python_files = list(Path().glob("**/*.py"))[:20]  # Limit for demo

        for py_file in python_files:
            if "__pycache__" in str(py_file) or "test_" in str(py_file):
                continue

            try:
                with open(py_file, encoding="utf-8") as f:
                    content = f.read()

                # Look for print statements without descriptive context
                print_statements = re.findall(r"print\s*\([^)]+\)", content)

                unclear_prints = []
                for print_stmt in print_statements:
                    # Check if print statement has descriptive text
                    if not any(
                        word in print_stmt.lower()
                        for word in [
                            "error",
                            "warning",
                            "info",
                            "success",
                            "status",
                            "result",
                            "loading",
                            "processing",
                        ]
                    ):
                        unclear_prints.append(print_stmt)

                if len(unclear_prints) > 5:
                    issues.append(
                        AccessibilityIssue(
                            issue_id=f"screen_reader_compat_{py_file.name}",
                            severity="minor",
                            standard=AccessibilityStandard.WCAG_2_1_AA,
                            guideline="1.3.1 Info and Relationships",
                            description="Output lacks descriptive context for screen readers",
                            affected_component=str(py_file),
                            suggested_fix="Add descriptive labels and context to output messages",
                            automated_fixable=True,
                        )
                    )

            except Exception:
                continue

        return issues

    def _test_alternative_descriptions(self) -> list[AccessibilityIssue]:
        """Test alternative text and descriptions."""
        issues = []

        # Check for plots and visualizations without alt text
        viz_files = list(Path().glob("**/visualization/*.py"))

        for viz_file in viz_files:
            try:
                with open(viz_file, encoding="utf-8") as f:
                    content = f.read()

                # Look for plotting functions
                has_plots = any(
                    plot_func in content
                    for plot_func in [
                        "plt.",
                        "matplotlib",
                        "seaborn",
                        "plotly",
                        "bokeh",
                    ]
                )

                has_alt_text = any(
                    alt_term in content.lower()
                    for alt_term in [
                        "alt",
                        "alternative",
                        "describe",
                        "caption",
                        "title",
                        "label",
                    ]
                )

                if has_plots and not has_alt_text:
                    issues.append(
                        AccessibilityIssue(
                            issue_id=f"missing_alt_text_{viz_file.name}",
                            severity="major",
                            standard=AccessibilityStandard.WCAG_2_1_A,
                            guideline="1.1.1 Non-text Content",
                            description="Visualizations lack alternative text descriptions",
                            affected_component=str(viz_file),
                            suggested_fix="Add descriptive titles, labels, and alternative text for all visualizations",
                            automated_fixable=False,
                        )
                    )

            except Exception:
                continue

        return issues

    def generate_accessibility_report(self) -> dict[str, Any]:
        """Generate comprehensive accessibility report."""
        if not self.accessibility_issues:
            return {"message": "No accessibility issues found"}

        # Group issues by severity
        severity_groups = {}
        for issue in self.accessibility_issues:
            severity = issue.severity
            if severity not in severity_groups:
                severity_groups[severity] = []
            severity_groups[severity].append(issue)

        # Group issues by standard
        standard_groups = {}
        for issue in self.accessibility_issues:
            standard = issue.standard.value
            if standard not in standard_groups:
                standard_groups[standard] = []
            standard_groups[standard].append(issue)

        # Calculate compliance scores
        total_issues = len(self.accessibility_issues)
        critical_issues = len(severity_groups.get("critical", []))
        major_issues = len(severity_groups.get("major", []))

        # Compliance score (100 - weighted penalty)
        compliance_score = max(0, 100 - (critical_issues * 20 + major_issues * 10))

        # Convert issues to JSON-serializable format
        detailed_issues = []
        for issue in self.accessibility_issues:
            issue_dict = asdict(issue)
            issue_dict["standard"] = issue.standard.value  # Convert enum to string
            detailed_issues.append(issue_dict)

        return {
            "total_issues": total_issues,
            "severity_breakdown": {
                severity: len(issues) for severity, issues in severity_groups.items()
            },
            "standard_breakdown": {
                standard: len(issues) for standard, issues in standard_groups.items()
            },
            "compliance_score": compliance_score,
            "automated_fixable": len(
                [i for i in self.accessibility_issues if i.automated_fixable]
            ),
            "detailed_issues": detailed_issues,
        }


class UsabilityTester:
    """Advanced usability testing system."""

    def __init__(self):
        self.usability_tests = []
        self.benchmarks = {
            UsabilityMetric.LEARNABILITY: 80.0,
            UsabilityMetric.EFFICIENCY: 85.0,
            UsabilityMetric.MEMORABILITY: 75.0,
            UsabilityMetric.ERROR_PREVENTION: 90.0,
            UsabilityMetric.SATISFACTION: 80.0,
        }

    def test_cli_usability(self) -> list[UsabilityTest]:
        """Test CLI usability aspects."""
        tests = []

        # Test learnability
        tests.append(self._test_learnability())

        # Test efficiency
        tests.append(self._test_efficiency())

        # Test memorability
        tests.append(self._test_memorability())

        # Test error prevention
        tests.append(self._test_error_prevention())

        # Test user satisfaction
        tests.append(self._test_satisfaction())

        self.usability_tests.extend(tests)
        return tests

    def _test_learnability(self) -> UsabilityTest:
        """Test how easily new users can learn the interface."""
        start_time = time.perf_counter()

        score = 0
        recommendations = []

        # Check for help documentation
        help_indicators = 0

        # Check CLI files for help features
        cli_files = list(Path().glob("**/cli/*.py"))
        for cli_file in cli_files[:5]:
            try:
                with open(cli_file, encoding="utf-8") as f:
                    content = f.read()

                # Look for help features
                if any(
                    help_term in content
                    for help_term in ["--help", "-h", "help(", "usage", "argparse"]
                ):
                    help_indicators += 1

                # Look for examples
                if any(
                    example_term in content.lower()
                    for example_term in ["example", "demo", "tutorial"]
                ):
                    help_indicators += 1

            except Exception:
                continue

        # Check for README and documentation
        doc_files = list(Path().glob("README*")) + list(Path().glob("docs/**/*.md"))
        if doc_files:
            help_indicators += 2

        # Score based on help availability
        score = min(100, help_indicators * 20)

        if score < 60:
            recommendations.extend(
                [
                    "Add comprehensive help documentation",
                    "Include usage examples in CLI help",
                    "Create getting started guide",
                ]
            )

        execution_time = time.perf_counter() - start_time

        return UsabilityTest(
            test_name="CLI Learnability",
            metric=UsabilityMetric.LEARNABILITY,
            score=score,
            benchmark=self.benchmarks[UsabilityMetric.LEARNABILITY],
            passed=score >= self.benchmarks[UsabilityMetric.LEARNABILITY],
            recommendations=recommendations,
            execution_time=execution_time,
        )

    def _test_efficiency(self) -> UsabilityTest:
        """Test how efficiently users can perform tasks."""
        start_time = time.perf_counter()

        score = 0
        recommendations = []

        # Check for shortcuts and automation
        efficiency_indicators = 0

        cli_files = list(Path().glob("**/cli/*.py"))
        for cli_file in cli_files[:5]:
            try:
                with open(cli_file, encoding="utf-8") as f:
                    content = f.read()

                # Look for efficiency features
                if any(
                    shortcut in content for shortcut in ["--", "-", "alias", "shortcut"]
                ):
                    efficiency_indicators += 1

                # Look for batch processing
                if any(
                    batch_term in content
                    for batch_term in ["batch", "bulk", "multiple", "all"]
                ):
                    efficiency_indicators += 1

                # Look for configuration files
                if any(
                    config_term in content
                    for config_term in ["config", "settings", "profile"]
                ):
                    efficiency_indicators += 1

            except Exception:
                continue

        # Check for automation scripts
        script_files = list(Path().glob("scripts/*.py"))
        if script_files:
            efficiency_indicators += 1

        score = min(100, efficiency_indicators * 25)

        if score < 70:
            recommendations.extend(
                [
                    "Add command shortcuts and aliases",
                    "Implement batch processing capabilities",
                    "Provide configuration file support",
                ]
            )

        execution_time = time.perf_counter() - start_time

        return UsabilityTest(
            test_name="CLI Efficiency",
            metric=UsabilityMetric.EFFICIENCY,
            score=score,
            benchmark=self.benchmarks[UsabilityMetric.EFFICIENCY],
            passed=score >= self.benchmarks[UsabilityMetric.EFFICIENCY],
            recommendations=recommendations,
            execution_time=execution_time,
        )

    def _test_memorability(self) -> UsabilityTest:
        """Test how memorable the interface commands are."""
        start_time = time.perf_counter()

        score = 0
        recommendations = []

        # Check for intuitive command naming
        memorability_score = 0

        cli_files = list(Path().glob("**/cli/*.py"))
        for cli_file in cli_files[:5]:
            try:
                with open(cli_file, encoding="utf-8") as f:
                    content = f.read()

                # Look for intuitive function/command names
                function_matches = re.findall(r"def (\w+)", content)
                intuitive_names = 0

                for func_name in function_matches:
                    # Check if function name is descriptive
                    if any(
                        word in func_name.lower()
                        for word in [
                            "run",
                            "start",
                            "stop",
                            "create",
                            "delete",
                            "update",
                            "get",
                            "set",
                            "show",
                            "list",
                            "help",
                            "config",
                        ]
                    ):
                        intuitive_names += 1

                if function_matches:
                    memorability_score += (
                        intuitive_names / len(function_matches)
                    ) * 100

            except Exception:
                continue

        # Average memorability score
        score = memorability_score / max(len(cli_files), 1) if cli_files else 50

        if score < 60:
            recommendations.extend(
                [
                    "Use more descriptive and intuitive command names",
                    "Follow standard CLI naming conventions",
                    "Group related commands logically",
                ]
            )

        execution_time = time.perf_counter() - start_time

        return UsabilityTest(
            test_name="CLI Memorability",
            metric=UsabilityMetric.MEMORABILITY,
            score=score,
            benchmark=self.benchmarks[UsabilityMetric.MEMORABILITY],
            passed=score >= self.benchmarks[UsabilityMetric.MEMORABILITY],
            recommendations=recommendations,
            execution_time=execution_time,
        )

    def _test_error_prevention(self) -> UsabilityTest:
        """Test error prevention and validation."""
        start_time = time.perf_counter()

        score = 0
        recommendations = []

        # Check for input validation and error handling
        validation_indicators = 0

        python_files = list(Path().glob("**/*.py"))[:20]  # Limit for demo
        for py_file in python_files:
            if "__pycache__" in str(py_file) or "test_" in str(py_file):
                continue

            try:
                with open(py_file, encoding="utf-8") as f:
                    content = f.read()

                # Look for validation patterns
                if any(
                    validation_term in content
                    for validation_term in [
                        "validate",
                        "check",
                        "verify",
                        "assert",
                        "isinstance",
                        "hasattr",
                    ]
                ):
                    validation_indicators += 1

                # Look for error handling
                if any(
                    error_term in content
                    for error_term in [
                        "try:",
                        "except:",
                        "raise",
                        "ValueError",
                        "TypeError",
                    ]
                ):
                    validation_indicators += 1

            except Exception:
                continue

        # Score based on validation presence
        score = min(100, (validation_indicators / max(len(python_files), 1)) * 100)

        if score < 80:
            recommendations.extend(
                [
                    "Add comprehensive input validation",
                    "Implement better error messages",
                    "Provide input format examples",
                ]
            )

        execution_time = time.perf_counter() - start_time

        return UsabilityTest(
            test_name="Error Prevention",
            metric=UsabilityMetric.ERROR_PREVENTION,
            score=score,
            benchmark=self.benchmarks[UsabilityMetric.ERROR_PREVENTION],
            passed=score >= self.benchmarks[UsabilityMetric.ERROR_PREVENTION],
            recommendations=recommendations,
            execution_time=execution_time,
        )

    def _test_satisfaction(self) -> UsabilityTest:
        """Test overall user satisfaction indicators."""
        start_time = time.perf_counter()

        score = 0
        recommendations = []

        # Check for user-friendly features
        satisfaction_indicators = 0

        # Check for progress indicators
        python_files = list(Path().glob("**/*.py"))[:15]
        for py_file in python_files:
            if "__pycache__" in str(py_file):
                continue

            try:
                with open(py_file, encoding="utf-8") as f:
                    content = f.read()

                # Look for user-friendly features
                if any(
                    progress_term in content
                    for progress_term in [
                        "progress",
                        "tqdm",
                        "loading",
                        "status",
                        "percentage",
                    ]
                ):
                    satisfaction_indicators += 1

                # Look for informative output
                if any(
                    info_term in content
                    for info_term in ["print", "log", "info", "success", "complete"]
                ):
                    satisfaction_indicators += 1

            except Exception:
                continue

        # Check for user documentation
        if list(Path().glob("README*")):
            satisfaction_indicators += 2

        if list(Path().glob("examples/**/*")):
            satisfaction_indicators += 1

        score = min(100, satisfaction_indicators * 15)

        if score < 70:
            recommendations.extend(
                [
                    "Add progress indicators for long operations",
                    "Provide more informative success/completion messages",
                    "Include user examples and tutorials",
                ]
            )

        execution_time = time.perf_counter() - start_time

        return UsabilityTest(
            test_name="User Satisfaction",
            metric=UsabilityMetric.SATISFACTION,
            score=score,
            benchmark=self.benchmarks[UsabilityMetric.SATISFACTION],
            passed=score >= self.benchmarks[UsabilityMetric.SATISFACTION],
            recommendations=recommendations,
            execution_time=execution_time,
        )

    def generate_usability_report(self) -> dict[str, Any]:
        """Generate comprehensive usability report."""
        if not self.usability_tests:
            return {"message": "No usability tests completed"}

        # Calculate overall scores
        total_score = sum(test.score for test in self.usability_tests)
        average_score = total_score / len(self.usability_tests)

        passed_tests = sum(1 for test in self.usability_tests if test.passed)
        test_pass_rate = (passed_tests / len(self.usability_tests)) * 100

        # Group by metric
        metric_scores = {}
        for test in self.usability_tests:
            metric_scores[test.metric.value] = {
                "score": test.score,
                "benchmark": test.benchmark,
                "passed": test.passed,
                "recommendations": test.recommendations,
            }

        # Convert tests to JSON-serializable format
        detailed_tests = []
        for test in self.usability_tests:
            test_dict = asdict(test)
            test_dict["metric"] = test.metric.value  # Convert enum to string
            detailed_tests.append(test_dict)

        return {
            "overall_score": average_score,
            "test_pass_rate": test_pass_rate,
            "tests_passed": passed_tests,
            "total_tests": len(self.usability_tests),
            "metric_scores": metric_scores,
            "summary_recommendations": self._get_priority_recommendations(),
            "detailed_tests": detailed_tests,
        }

    def _get_priority_recommendations(self) -> list[str]:
        """Get prioritized recommendations."""
        all_recommendations = []
        for test in self.usability_tests:
            if not test.passed:
                all_recommendations.extend(test.recommendations)

        # Remove duplicates while preserving order
        seen = set()
        priority_recommendations = []
        for rec in all_recommendations:
            if rec not in seen:
                priority_recommendations.append(rec)
                seen.add(rec)

        return priority_recommendations[:10]  # Top 10 recommendations


def run_accessibility_usability_testing():
    """Main function to run accessibility and usability testing."""
    print("Accessibility and Usability Testing Framework - Task 5.5")
    print("=" * 65)

    # Create testing systems
    accessibility_tester = AccessibilityTester()
    usability_tester = UsabilityTester()

    print("Running accessibility testing...")
    accessibility_tester.test_cli_accessibility()
    accessibility_report = accessibility_tester.generate_accessibility_report()

    print("Running usability testing...")
    usability_tester.test_cli_usability()
    usability_report = usability_tester.generate_usability_report()

    # Compile comprehensive results
    comprehensive_results = {
        "accessibility_testing": accessibility_report,
        "usability_testing": usability_report,
        "combined_metrics": {
            "accessibility_score": accessibility_report.get("compliance_score", 0),
            "usability_score": usability_report.get("overall_score", 0),
            "total_issues": accessibility_report.get("total_issues", 0),
            "usability_tests_passed": usability_report.get("tests_passed", 0),
            "critical_accessibility_issues": accessibility_report.get(
                "severity_breakdown", {}
            ).get("critical", 0),
        },
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    # Display summary
    print("\nACCESSIBILITY AND USABILITY SUMMARY:")

    # Accessibility summary
    if accessibility_report.get("total_issues", 0) > 0:
        print(f"  Accessibility Issues Found: {accessibility_report['total_issues']}")
        print(
            f"  Compliance Score: {accessibility_report.get('compliance_score', 0):.1f}%"
        )
        severity_breakdown = accessibility_report.get("severity_breakdown", {})
        for severity, count in severity_breakdown.items():
            print(f"    {severity.title()}: {count}")
    else:
        print("  ✓ No accessibility issues found")

    # Usability summary
    if usability_report.get("total_tests", 0) > 0:
        print(f"  Usability Score: {usability_report.get('overall_score', 0):.1f}%")
        print(
            f"  Tests Passed: {usability_report.get('tests_passed', 0)}/{usability_report.get('total_tests', 0)}"
        )

        # Show metric scores
        metric_scores = usability_report.get("metric_scores", {})
        for metric, data in metric_scores.items():
            status = "✓ PASS" if data["passed"] else "✗ FAIL"
            print(
                f"    {status} {metric.title()}: {data['score']:.1f}% (benchmark: {data['benchmark']:.1f}%)"
            )

    # Combined assessment
    combined = comprehensive_results["combined_metrics"]
    overall_quality_score = (
        combined["accessibility_score"] + combined["usability_score"]
    ) / 2

    print("\nOVERALL QUALITY ASSESSMENT:")
    print(f"  Combined Quality Score: {overall_quality_score:.1f}%")

    if overall_quality_score >= 80:
        print("  🎯 Excellent accessibility and usability!")
    elif overall_quality_score >= 60:
        print("  ⚠️  Good accessibility and usability with room for improvement")
    else:
        print("  🔧 Significant improvements needed in accessibility and usability")

    # Show priority recommendations
    if usability_report.get("summary_recommendations"):
        print("\nPRIORITY RECOMMENDATIONS:")
        for i, rec in enumerate(usability_report["summary_recommendations"][:5], 1):
            print(f"  {i}. {rec}")

    # Save results
    results_dir = Path("accessibility_usability_results")
    results_dir.mkdir(exist_ok=True)

    results_file = results_dir / "task_5_5_accessibility_usability_report.json"
    with open(results_file, "w") as f:
        json.dump(comprehensive_results, f, indent=2)

    print(f"\n📄 Accessibility and usability report saved to: {results_file}")
    print("✅ Task 5.5 Accessibility and Usability Testing Complete!")
    print(f"🎯 Overall quality score: {overall_quality_score:.1f}%")
    print(f"♿ Accessibility compliance: {combined['accessibility_score']:.1f}%")
    print(f"👥 Usability score: {combined['usability_score']:.1f}%")

    return comprehensive_results


if __name__ == "__main__":
    run_accessibility_usability_testing()
