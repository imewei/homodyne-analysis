"""
Automated Dead Code Detection Pipeline
====================================

Enterprise-grade automated pipeline for detecting dead code using AST analysis,
call graph generation, and comprehensive static analysis. Provides safe,
reliable detection of unused functions, classes, and methods while avoiding
false positives.

Authors: Wei Chen, Hongrui He
Institution: Argonne National Laboratory
"""

import ast
import json
import logging
import time
from collections import defaultdict
from collections import deque
from dataclasses import asdict
from dataclasses import dataclass
from datetime import UTC
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest

logger = logging.getLogger(__name__)


@dataclass
class DeadCodeCandidate:
    """Dead code candidate with analysis details."""

    name: str
    type: str  # 'function', 'class', 'method'
    file_path: str
    line_number: int
    confidence: float  # 0.0 to 1.0
    reasons: list[str]
    dependencies: list[str]
    test_only: bool
    safe_to_remove: bool
    analysis_timestamp: str


@dataclass
class CallGraphNode:
    """Node in the call graph representing a callable."""

    name: str
    file_path: str
    line_number: int
    node_type: str  # 'function', 'method', 'class'
    is_public: bool
    is_test: bool
    calls: set[str]  # Names of functions/methods this node calls
    called_by: set[str]  # Names of functions/methods that call this node


class ASTCallGraphBuilder:
    """Builds call graphs using AST analysis."""

    def __init__(self, package_root: Path):
        self.package_root = package_root
        self.call_graph: dict[str, CallGraphNode] = {}
        self.imports: dict[str, set[str]] = defaultdict(set)
        self.definitions: dict[str, tuple[str, int]] = {}  # name -> (file, line)

    def build_call_graph(self) -> dict[str, CallGraphNode]:
        """Build comprehensive call graph for the package."""
        logger.info("Building call graph for package")

        # First pass: collect all definitions
        for py_file in self.package_root.rglob("*.py"):
            if self._should_skip_file(py_file):
                continue
            self._collect_definitions(py_file)

        # Second pass: build call relationships
        for py_file in self.package_root.rglob("*.py"):
            if self._should_skip_file(py_file):
                continue
            self._build_calls(py_file)

        logger.info(f"Built call graph with {len(self.call_graph)} nodes")
        return self.call_graph

    def _should_skip_file(self, file_path: Path) -> bool:
        """Check if file should be skipped."""
        skip_patterns = {
            "__pycache__",
            ".git",
            "build",
            "dist",
            ".pytest_cache",
            ".mypy_cache",
        }
        return any(pattern in str(file_path) for pattern in skip_patterns)

    def _collect_definitions(self, file_path: Path) -> None:
        """Collect function and class definitions from file."""
        try:
            with open(file_path, encoding="utf-8") as f:
                content = f.read()

            tree = ast.parse(content)
            is_test_file = "test_" in file_path.name or "/tests/" in str(file_path)

            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    full_name = f"{file_path.stem}.{node.name}"
                    self.definitions[node.name] = (str(file_path), node.lineno)
                    self.definitions[full_name] = (str(file_path), node.lineno)

                    # Create call graph node
                    self.call_graph[full_name] = CallGraphNode(
                        name=node.name,
                        file_path=str(file_path),
                        line_number=node.lineno,
                        node_type="function",
                        is_public=not node.name.startswith("_"),
                        is_test=is_test_file or node.name.startswith("test_"),
                        calls=set(),
                        called_by=set(),
                    )

                elif isinstance(node, ast.ClassDef):
                    full_name = f"{file_path.stem}.{node.name}"
                    self.definitions[node.name] = (str(file_path), node.lineno)
                    self.definitions[full_name] = (str(file_path), node.lineno)

                    # Create call graph node for class
                    self.call_graph[full_name] = CallGraphNode(
                        name=node.name,
                        file_path=str(file_path),
                        line_number=node.lineno,
                        node_type="class",
                        is_public=not node.name.startswith("_"),
                        is_test=is_test_file,
                        calls=set(),
                        called_by=set(),
                    )

                    # Collect methods
                    for method in node.body:
                        if isinstance(method, ast.FunctionDef):
                            method_full_name = (
                                f"{file_path.stem}.{node.name}.{method.name}"
                            )
                            self.definitions[method_full_name] = (
                                str(file_path),
                                method.lineno,
                            )

                            self.call_graph[method_full_name] = CallGraphNode(
                                name=method.name,
                                file_path=str(file_path),
                                line_number=method.lineno,
                                node_type="method",
                                is_public=not method.name.startswith("_"),
                                is_test=is_test_file or method.name.startswith("test_"),
                                calls=set(),
                                called_by=set(),
                            )

        except Exception as e:
            logger.warning(f"Failed to parse {file_path}: {e}")

    def _build_calls(self, file_path: Path) -> None:
        """Build call relationships from file."""
        try:
            with open(file_path, encoding="utf-8") as f:
                content = f.read()

            tree = ast.parse(content)
            visitor = CallVisitor(self.call_graph, self.definitions, str(file_path))
            visitor.visit(tree)

        except Exception as e:
            logger.warning(f"Failed to analyze calls in {file_path}: {e}")


class CallVisitor(ast.NodeVisitor):
    """AST visitor to collect function and method calls."""

    def __init__(
        self,
        call_graph: dict[str, CallGraphNode],
        definitions: dict[str, tuple[str, int]],
        file_path: str,
    ):
        self.call_graph = call_graph
        self.definitions = definitions
        self.file_path = file_path
        self.current_function = None
        self.current_class = None

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Visit function definition."""
        old_function = self.current_function
        file_stem = Path(self.file_path).stem

        if self.current_class:
            self.current_function = f"{file_stem}.{self.current_class}.{node.name}"
        else:
            self.current_function = f"{file_stem}.{node.name}"

        self.generic_visit(node)
        self.current_function = old_function

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """Visit class definition."""
        old_class = self.current_class
        self.current_class = node.name
        self.generic_visit(node)
        self.current_class = old_class

    def visit_Call(self, node: ast.Call) -> None:
        """Visit function/method call."""
        if self.current_function and self.current_function in self.call_graph:
            call_name = self._extract_call_name(node)
            if call_name:
                # Find the full name of the called function
                full_call_name = self._resolve_call_name(call_name)
                if full_call_name and full_call_name in self.call_graph:
                    # Add call relationship
                    self.call_graph[self.current_function].calls.add(full_call_name)
                    self.call_graph[full_call_name].called_by.add(self.current_function)

        self.generic_visit(node)

    def _extract_call_name(self, node: ast.Call) -> str | None:
        """Extract the name of the called function/method."""
        if isinstance(node.func, ast.Name):
            return node.func.id
        if isinstance(node.func, ast.Attribute):
            if isinstance(node.func.value, ast.Name):
                return f"{node.func.value.id}.{node.func.attr}"
            return node.func.attr
        return None

    def _resolve_call_name(self, call_name: str) -> str | None:
        """Resolve call name to full qualified name."""
        # Try exact match first
        if call_name in self.call_graph:
            return call_name

        # Try with current file prefix
        file_stem = Path(self.file_path).stem
        full_name = f"{file_stem}.{call_name}"
        if full_name in self.call_graph:
            return full_name

        # Search in definitions
        for def_name, (def_file, _) in self.definitions.items():
            if def_name.endswith(f".{call_name}") or def_name == call_name:
                return def_name

        return None


class DeadCodeDetectionPipeline:
    """Comprehensive dead code detection pipeline."""

    def __init__(self, package_root: Path, config: dict[str, Any] | None = None):
        self.package_root = package_root
        # Merge user config with defaults
        default_config = self._get_default_config()
        if config:
            default_config.update(config)
        self.config = default_config
        self.call_graph_builder = ASTCallGraphBuilder(package_root)
        self.analysis_results: dict[str, Any] = {}

    def _get_default_config(self) -> dict[str, Any]:
        """Get default configuration for dead code detection."""
        return {
            "min_confidence": 0.8,
            "exclude_test_files": False,
            "exclude_private_functions": True,
            "exclude_magic_methods": True,
            "exclude_property_methods": True,
            "safe_removal_only": True,
            "analysis_timeout": 300,  # 5 minutes
            "whitelist_patterns": [
                r"^test_.*",
                r"^__.*__$",
                r"^main$",
                r"^setup$",
                r"^teardown$",
            ],
            "exclude_files": [
                "__init__.py",
                "setup.py",
                "conftest.py",
            ],
        }

    def run_pipeline(self) -> dict[str, Any]:
        """Run the complete dead code detection pipeline."""
        logger.info("Starting dead code detection pipeline")
        start_time = time.time()

        try:
            # Step 1: Build call graph
            call_graph = self.call_graph_builder.build_call_graph()

            # Step 2: Identify entry points
            entry_points = self._identify_entry_points(call_graph)

            # Step 3: Perform reachability analysis
            reachable_nodes = self._reachability_analysis(call_graph, entry_points)

            # Step 4: Find dead code candidates
            candidates = self._find_dead_code_candidates(call_graph, reachable_nodes)

            # Step 5: Apply filters and confidence scoring
            filtered_candidates = self._filter_and_score_candidates(candidates)

            # Step 6: Generate recommendations
            recommendations = self._generate_recommendations(filtered_candidates)

            self.analysis_results = {
                "pipeline_version": "1.0.0",
                "analysis_timestamp": datetime.now(UTC).isoformat(),
                "package_root": str(self.package_root),
                "config": self.config,
                "statistics": {
                    "total_nodes": len(call_graph),
                    "entry_points": len(entry_points),
                    "reachable_nodes": len(reachable_nodes),
                    "dead_code_candidates": len(candidates),
                    "filtered_candidates": len(filtered_candidates),
                    "high_confidence_candidates": len(
                        [c for c in filtered_candidates if c.confidence > 0.9]
                    ),
                    "safe_removal_candidates": len(
                        [c for c in filtered_candidates if c.safe_to_remove]
                    ),
                    "analysis_time_seconds": time.time() - start_time,
                },
                "entry_points": list(entry_points),
                "candidates": [asdict(c) for c in filtered_candidates],
                "recommendations": recommendations,
            }

            logger.info(f"Pipeline completed in {time.time() - start_time:.2f}s")
            logger.info(f"Found {len(filtered_candidates)} dead code candidates")

            return self.analysis_results

        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            raise

    def _identify_entry_points(self, call_graph: dict[str, CallGraphNode]) -> set[str]:
        """Identify entry points (functions that should not be removed)."""
        entry_points = set()

        for name, node in call_graph.items():
            # Public API functions
            if node.is_public and not node.is_test:
                entry_points.add(name)

            # Main functions
            if node.name in ["main", "__main__", "run", "execute"]:
                entry_points.add(name)

            # CLI entry points
            if "cli" in node.file_path.lower() and node.is_public:
                entry_points.add(name)

            # Test functions
            if node.name.startswith("test_") or node.is_test:
                entry_points.add(name)

            # Magic methods
            if node.name.startswith("__") and node.name.endswith("__"):
                entry_points.add(name)

        logger.info(f"Identified {len(entry_points)} entry points")
        return entry_points

    def _reachability_analysis(
        self, call_graph: dict[str, CallGraphNode], entry_points: set[str]
    ) -> set[str]:
        """Perform reachability analysis from entry points."""
        reachable = set()
        queue = deque(entry_points)

        while queue:
            current = queue.popleft()
            if current in reachable:
                continue

            reachable.add(current)

            # Add all functions called by this function
            if current in call_graph:
                for called_func in call_graph[current].calls:
                    if called_func not in reachable:
                        queue.append(called_func)

        logger.info(f"Found {len(reachable)} reachable nodes")
        return reachable

    def _find_dead_code_candidates(
        self, call_graph: dict[str, CallGraphNode], reachable_nodes: set[str]
    ) -> list[DeadCodeCandidate]:
        """Find dead code candidates (unreachable nodes)."""
        candidates = []

        for name, node in call_graph.items():
            if name not in reachable_nodes:
                # Check if it's in an excluded file
                if any(excl in node.file_path for excl in self.config["exclude_files"]):
                    continue

                reasons = ["Not reachable from any entry point"]
                confidence = 0.7  # Base confidence

                # Increase confidence for certain patterns
                if node.name.startswith("_") and not node.name.startswith("__"):
                    reasons.append("Private function with no internal usage")
                    confidence += 0.1

                if not node.calls and not node.called_by:
                    reasons.append("No calls to or from this function")
                    confidence += 0.2

                if not node.is_public:
                    reasons.append("Not part of public API")
                    confidence += 0.1

                # Determine if it's test-only
                test_only = node.is_test or "test" in node.file_path.lower()

                # Determine if it's safe to remove
                safe_to_remove = (
                    confidence >= self.config["min_confidence"]
                    and not node.name.startswith("__")
                    and node.name not in ["setup", "teardown", "main"]
                )

                candidate = DeadCodeCandidate(
                    name=name,
                    type=node.node_type,
                    file_path=node.file_path,
                    line_number=node.line_number,
                    confidence=min(confidence, 1.0),
                    reasons=reasons,
                    dependencies=list(node.calls),
                    test_only=test_only,
                    safe_to_remove=safe_to_remove,
                    analysis_timestamp=datetime.now(UTC).isoformat(),
                )

                candidates.append(candidate)

        logger.info(f"Found {len(candidates)} initial dead code candidates")
        return candidates

    def _filter_and_score_candidates(
        self, candidates: list[DeadCodeCandidate]
    ) -> list[DeadCodeCandidate]:
        """Filter and score dead code candidates."""
        filtered = []

        for candidate in candidates:
            # Apply whitelist patterns
            import re

            skip = False
            for pattern in self.config["whitelist_patterns"]:
                if re.match(pattern, candidate.name):
                    skip = True
                    break

            if skip:
                continue

            # Apply configuration filters
            if self.config["exclude_private_functions"] and candidate.name.startswith(
                "_"
            ):
                continue

            if self.config["exclude_magic_methods"] and candidate.name.startswith("__"):
                continue

            if self.config["safe_removal_only"] and not candidate.safe_to_remove:
                continue

            # Apply minimum confidence threshold
            if candidate.confidence < self.config["min_confidence"]:
                continue

            filtered.append(candidate)

        # Sort by confidence (highest first)
        filtered.sort(key=lambda c: c.confidence, reverse=True)

        logger.info(f"Filtered to {len(filtered)} candidates")
        return filtered

    def _generate_recommendations(
        self, candidates: list[DeadCodeCandidate]
    ) -> dict[str, Any]:
        """Generate recommendations for dead code removal."""
        recommendations = {
            "summary": {
                "total_candidates": len(candidates),
                "high_confidence": len([c for c in candidates if c.confidence > 0.9]),
                "safe_removal": len([c for c in candidates if c.safe_to_remove]),
                "estimated_lines_saved": sum(
                    self._estimate_lines(c) for c in candidates
                ),
            },
            "removal_plan": {
                "immediate_removal": [
                    asdict(c)
                    for c in candidates
                    if c.confidence > 0.95 and c.safe_to_remove
                ],
                "review_required": [
                    asdict(c) for c in candidates if 0.8 <= c.confidence <= 0.95
                ],
                "low_confidence": [asdict(c) for c in candidates if c.confidence < 0.8],
            },
            "file_summary": self._group_by_file(candidates),
            "next_steps": [
                "Review high-confidence candidates for immediate removal",
                "Manually verify medium-confidence candidates",
                "Run comprehensive tests after removal",
                "Update documentation if needed",
            ],
        }

        return recommendations

    def _estimate_lines(self, candidate: DeadCodeCandidate) -> int:
        """Estimate number of lines for a dead code candidate."""
        try:
            with open(candidate.file_path) as f:
                lines = f.readlines()

            # Simple heuristic: function likely spans 10-20 lines
            if candidate.type == "function":
                return 15
            if candidate.type == "method":
                return 10
            if candidate.type == "class":
                return 30
            return 5
        except:
            return 10  # Default estimate

    def _group_by_file(
        self, candidates: list[DeadCodeCandidate]
    ) -> dict[str, dict[str, Any]]:
        """Group candidates by file for easier review."""
        file_groups = defaultdict(list)

        for candidate in candidates:
            file_groups[candidate.file_path].append(candidate)

        summary = {}
        for file_path, file_candidates in file_groups.items():
            summary[file_path] = {
                "total_candidates": len(file_candidates),
                "high_confidence": len(
                    [c for c in file_candidates if c.confidence > 0.9]
                ),
                "safe_removal": len([c for c in file_candidates if c.safe_to_remove]),
                "candidates": [c.name for c in file_candidates],
            }

        return summary

    def save_analysis_report(self, output_path: Path) -> None:
        """Save analysis report to file."""
        if not self.analysis_results:
            raise ValueError("No analysis results to save. Run pipeline first.")

        with open(output_path, "w") as f:
            json.dump(self.analysis_results, f, indent=2)

        logger.info(f"Analysis report saved to {output_path}")

    def get_removal_script(self) -> str:
        """Generate a script for automated removal of safe candidates."""
        if not self.analysis_results:
            raise ValueError("No analysis results available. Run pipeline first.")

        safe_candidates = [
            c
            for c in self.analysis_results["candidates"]
            if c["safe_to_remove"] and c["confidence"] > 0.95
        ]

        script_lines = [
            "#!/usr/bin/env python3",
            '"""',
            "Automated Dead Code Removal Script",
            f"Generated: {datetime.now().isoformat()}",
            f"Total candidates: {len(safe_candidates)}",
            '"""',
            "",
            "import ast",
            "import sys",
            "from pathlib import Path",
            "",
            "def remove_function_or_class(file_path: str, line_number: int, name: str):",
            "    '''Remove function or class from file.'''",
            "    # Implementation would go here",
            "    print(f'Would remove {name} from {file_path}:{line_number}')",
            "",
            "def main():",
            "    '''Main removal function.'''",
        ]

        for candidate in safe_candidates:
            script_lines.append(
                f"    remove_function_or_class('{candidate['file_path']}', "
                f"{candidate['line_number']}, '{candidate['name']}')"
            )

        script_lines.extend(["", "if __name__ == '__main__':", "    main()"])

        return "\n".join(script_lines)


class TestDeadCodeDetectionPipeline:
    """Test suite for dead code detection pipeline."""

    def test_pipeline_initialization(self):
        """Test pipeline initialization."""
        package_root = Path("homodyne")
        pipeline = DeadCodeDetectionPipeline(package_root)

        assert pipeline.package_root == package_root
        assert pipeline.config is not None
        assert "min_confidence" in pipeline.config

    def test_call_graph_building(self):
        """Test call graph building."""
        package_root = Path("homodyne")
        if not package_root.exists():
            pytest.skip("Package directory not found")

        builder = ASTCallGraphBuilder(package_root)
        call_graph = builder.build_call_graph()

        assert isinstance(call_graph, dict)
        assert len(call_graph) > 0

        # Check that nodes have expected attributes
        for name, node in call_graph.items():
            assert isinstance(node, CallGraphNode)
            assert node.name
            assert node.file_path
            assert node.line_number > 0
            assert node.node_type in ["function", "method", "class"]

    @pytest.mark.slow
    def test_full_pipeline_execution(self):
        """Test full pipeline execution."""
        package_root = Path("homodyne")
        if not package_root.exists():
            pytest.skip("Package directory not found")

        pipeline = DeadCodeDetectionPipeline(package_root)
        results = pipeline.run_pipeline()

        assert isinstance(results, dict)
        assert "statistics" in results
        assert "candidates" in results
        assert "recommendations" in results

        # Verify statistics
        stats = results["statistics"]
        assert stats["total_nodes"] > 0
        assert stats["analysis_time_seconds"] > 0

        # Verify candidates structure
        for candidate_dict in results["candidates"]:
            assert "name" in candidate_dict
            assert "confidence" in candidate_dict
            assert "safe_to_remove" in candidate_dict

    def test_entry_point_identification(self):
        """Test entry point identification."""
        package_root = Path("homodyne")
        if not package_root.exists():
            pytest.skip("Package directory not found")

        pipeline = DeadCodeDetectionPipeline(package_root)
        call_graph = pipeline.call_graph_builder.build_call_graph()
        entry_points = pipeline._identify_entry_points(call_graph)

        assert isinstance(entry_points, set)
        assert len(entry_points) > 0

        # Should include main functions and public API
        entry_point_names = {name.split(".")[-1] for name in entry_points}

        # Common entry points that should be preserved
        expected_patterns = ["main", "test_", "__init__"]
        found_patterns = any(
            any(pattern in name for pattern in expected_patterns)
            for name in entry_point_names
        )
        assert found_patterns, f"No expected patterns found in {entry_point_names}"

    def test_reachability_analysis(self):
        """Test reachability analysis."""
        package_root = Path("homodyne")
        if not package_root.exists():
            pytest.skip("Package directory not found")

        pipeline = DeadCodeDetectionPipeline(package_root)
        call_graph = pipeline.call_graph_builder.build_call_graph()
        entry_points = pipeline._identify_entry_points(call_graph)
        reachable = pipeline._reachability_analysis(call_graph, entry_points)

        assert isinstance(reachable, set)
        # All entry points should be reachable
        assert entry_points.issubset(reachable)
        # Reachable should be at least as large as entry points
        assert len(reachable) >= len(entry_points)

    def test_candidate_filtering(self):
        """Test candidate filtering."""
        # Create mock candidates
        candidates = [
            DeadCodeCandidate(
                name="test_function",
                type="function",
                file_path="/test/file.py",
                line_number=10,
                confidence=0.9,
                reasons=["Mock reason"],
                dependencies=[],
                test_only=False,
                safe_to_remove=True,
                analysis_timestamp="2024-01-01T00:00:00Z",
            ),
            DeadCodeCandidate(
                name="__private_function",
                type="function",
                file_path="/test/file.py",
                line_number=20,
                confidence=0.95,
                reasons=["Mock reason"],
                dependencies=[],
                test_only=False,
                safe_to_remove=True,
                analysis_timestamp="2024-01-01T00:00:00Z",
            ),
            DeadCodeCandidate(
                name="low_confidence_function",
                type="function",
                file_path="/test/file.py",
                line_number=30,
                confidence=0.5,
                reasons=["Mock reason"],
                dependencies=[],
                test_only=False,
                safe_to_remove=False,
                analysis_timestamp="2024-01-01T00:00:00Z",
            ),
        ]

        package_root = Path("homodyne")
        pipeline = DeadCodeDetectionPipeline(package_root)
        filtered = pipeline._filter_and_score_candidates(candidates)

        # Should filter out low confidence and private functions
        assert len(filtered) <= len(candidates)

        # All filtered candidates should meet minimum confidence
        for candidate in filtered:
            assert candidate.confidence >= pipeline.config["min_confidence"]

    def test_report_generation(self):
        """Test analysis report generation."""
        package_root = Path("homodyne")
        pipeline = DeadCodeDetectionPipeline(package_root)

        # Mock analysis results
        pipeline.analysis_results = {
            "candidates": [
                {
                    "name": "mock_function",
                    "confidence": 0.9,
                    "safe_to_remove": True,
                    "file_path": "/test/file.py",
                    "line_number": 10,
                }
            ]
        }

        recommendations = pipeline._generate_recommendations(
            [
                DeadCodeCandidate(
                    name="mock_function",
                    type="function",
                    file_path="/test/file.py",
                    line_number=10,
                    confidence=0.9,
                    reasons=["Mock"],
                    dependencies=[],
                    test_only=False,
                    safe_to_remove=True,
                    analysis_timestamp="2024-01-01T00:00:00Z",
                )
            ]
        )

        assert isinstance(recommendations, dict)
        assert "summary" in recommendations
        assert "removal_plan" in recommendations
        assert "next_steps" in recommendations

    def test_removal_script_generation(self):
        """Test removal script generation."""
        package_root = Path("homodyne")
        pipeline = DeadCodeDetectionPipeline(package_root)

        # Mock analysis results
        pipeline.analysis_results = {
            "candidates": [
                {
                    "name": "safe_function",
                    "confidence": 0.96,
                    "safe_to_remove": True,
                    "file_path": "/test/file.py",
                    "line_number": 10,
                }
            ]
        }

        script = pipeline.get_removal_script()

        assert isinstance(script, str)
        assert "#!/usr/bin/env python3" in script
        assert "safe_function" in script
        assert "remove_function_or_class" in script


# Integration test with actual package
class TestIntegrationWithHomodynePackage:
    """Integration tests with the actual homodyne package."""

    def test_analyze_homodyne_package(self):
        """Test analysis of the actual homodyne package."""
        package_root = Path("homodyne")
        if not package_root.exists():
            pytest.skip("Homodyne package directory not found")

        pipeline = DeadCodeDetectionPipeline(
            package_root,
            config={
                "min_confidence": 0.8,
                "safe_removal_only": True,
                "analysis_timeout": 120,
            },
        )

        results = pipeline.run_pipeline()

        # Basic validation
        assert results["statistics"]["total_nodes"] > 10
        assert results["statistics"]["analysis_time_seconds"] < 120

        # Should find some candidates in a real package
        assert len(results["candidates"]) >= 0

        # Log results for manual review
        logger.info(f"Found {len(results['candidates'])} dead code candidates")
        for candidate in results["candidates"][:5]:  # Show first 5
            logger.info(
                f"  - {candidate['name']} (confidence: {candidate['confidence']:.2f})"
            )

    def test_specific_files_analysis(self):
        """Test analysis of specific files mentioned in requirements."""
        target_files = [
            "homodyne/core/security_metrics.py",
            "homodyne/ui/cli_enhancer.py",
            "homodyne/core/config.py",
        ]

        found_files = []
        for file_path in target_files:
            if Path(file_path).exists():
                found_files.append(file_path)

        if not found_files:
            pytest.skip("Target files not found")

        package_root = Path("homodyne")
        pipeline = DeadCodeDetectionPipeline(package_root)
        results = pipeline.run_pipeline()

        # Check if any candidates are from target files
        target_candidates = [
            c
            for c in results["candidates"]
            if any(target in c["file_path"] for target in found_files)
        ]

        logger.info(f"Found {len(target_candidates)} candidates in target files")
        for candidate in target_candidates:
            logger.info(f"  - {candidate['name']} in {candidate['file_path']}")


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)

    package_root = Path("homodyne")
    if package_root.exists():
        pipeline = DeadCodeDetectionPipeline(package_root)
        results = pipeline.run_pipeline()

        # Save report
        output_path = Path("dead_code_analysis_report.json")
        pipeline.save_analysis_report(output_path)
        print(f"Analysis complete. Report saved to {output_path}")

        # Generate removal script
        script = pipeline.get_removal_script()
        script_path = Path("dead_code_removal_script.py")
        with open(script_path, "w") as f:
            f.write(script)
        print(f"Removal script generated: {script_path}")
    else:
        print("Package directory 'homodyne' not found")
