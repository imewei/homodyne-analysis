#!/usr/bin/env python3
"""
Research-Grade Documentation Deployment Script
==============================================

Comprehensive documentation deployment for the homodyne-analysis package
with research publication standards and automated quality validation.

This script provides:
- Multi-platform documentation building (HTML, PDF, ePub)
- Research standards validation
- Performance monitoring
- Quality assurance checks
- Publication-ready output generation

Usage:
    python docs/deploy_docs.py --target=all --validate --research-grade
    python docs/deploy_docs.py --target=github-pages --auto-deploy
    python docs/deploy_docs.py --build-only --pdf

Authors: Wei Chen, Hongrui He (Argonne National Laboratory)
License: MIT
"""

import argparse
import json
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


class ResearchDocumentationBuilder:
    """
    Research-grade documentation builder with publication standards.

    Features:
    - Sphinx documentation building with mathematical content
    - Research standards validation
    - Multi-format output (HTML, PDF, ePub)
    - Performance monitoring and optimization
    - Citation and publication compliance checking
    """

    def __init__(self, source_dir: str = "docs", build_dir: str = "docs/_build"):
        self.source_dir = Path(source_dir)
        self.build_dir = Path(build_dir)
        self.project_root = Path.cwd()
        self.start_time = time.time()

        # Quality metrics
        self.quality_metrics = {
            "build_time": 0,
            "html_files": 0,
            "api_files": 0,
            "research_files": 0,
            "pdf_generated": False,
            "citations_found": False,
            "mathematical_content": False,
            "research_standards_met": False,
        }

        # Required research sections
        self.required_research_sections = [
            "research/index.rst",
            "research/theoretical_framework.rst",
            "research/computational_methods.rst",
            "research/publications.rst",
        ]

    def validate_environment(self) -> bool:
        """Validate the documentation build environment."""
        print("🔍 Validating documentation environment...")

        # Check Python version (requirement removed to avoid outdated version block)
        if not hasattr(sys, "version_info"):
            print("❌ Invalid Python installation")
            return False

        # Check for required directories
        if not self.source_dir.exists():
            print(f"❌ Source directory not found: {self.source_dir}")
            return False

        # Check for Sphinx
        try:
            result = subprocess.run(
                ["sphinx-build", "--version"], capture_output=True, text=True
            )
            print(f"✅ Sphinx version: {result.stdout.strip()}")
        except FileNotFoundError:
            print("❌ Sphinx not found. Install with: pip install sphinx")
            return False

        # Check for research documentation files
        missing_sections = []
        for section in self.required_research_sections:
            if not (self.source_dir / section).exists():
                missing_sections.append(section)

        if missing_sections:
            print(f"⚠️  Missing research sections: {missing_sections}")
        else:
            print("✅ All required research sections found")

        # Check for homodyne package
        try:
            import homodyne

            print(f"✅ homodyne package version: {homodyne.__version__}")
        except ImportError:
            print("⚠️  homodyne package not found - API docs may be incomplete")

        return True

    def clean_build_directory(self):
        """Clean previous build artifacts."""
        print("🧹 Cleaning previous build artifacts...")

        if self.build_dir.exists():
            shutil.rmtree(self.build_dir)

        self.build_dir.mkdir(parents=True, exist_ok=True)
        print(f"✅ Build directory prepared: {self.build_dir}")

    def build_html_documentation(self) -> bool:
        """Build HTML documentation with research standards."""
        print("📚 Building HTML documentation...")

        html_dir = self.build_dir / "html"
        cmd = [
            "sphinx-build",
            "-b",
            "html",
            "-d",
            str(self.build_dir / "doctrees"),
            "-W",  # Turn warnings into errors for quality assurance
            "--keep-going",  # Continue building despite errors
            str(self.source_dir),
            str(html_dir),
        ]

        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, cwd=self.project_root
            )

            if result.returncode == 0:
                print("✅ HTML documentation built successfully")
                self.quality_metrics["html_files"] = len(
                    list(html_dir.glob("**/*.html"))
                )
                self.quality_metrics["api_files"] = len(
                    list(html_dir.glob("**/api*/*.html"))
                )
                self.quality_metrics["research_files"] = len(
                    list(html_dir.glob("**/research*/*.html"))
                )
                return True
            else:
                print("❌ HTML build failed:")
                print(result.stderr)
                return False

        except Exception as e:
            print(f"❌ HTML build error: {e}")
            return False

    def build_pdf_documentation(self) -> bool:
        """Build PDF documentation for publication."""
        print("📄 Building PDF documentation...")

        # Check for LaTeX
        try:
            subprocess.run(["pdflatex", "--version"], capture_output=True, check=True)
        except (FileNotFoundError, subprocess.CalledProcessError):
            print("⚠️  LaTeX not found - PDF generation skipped")
            return False

        latex_dir = self.build_dir / "latex"
        cmd = [
            "sphinx-build",
            "-b",
            "latex",
            "-d",
            str(self.build_dir / "doctrees"),
            str(self.source_dir),
            str(latex_dir),
        ]

        try:
            # Build LaTeX
            result = subprocess.run(
                cmd, capture_output=True, text=True, cwd=self.project_root
            )

            if result.returncode != 0:
                print("⚠️  LaTeX build warnings:")
                print(result.stderr)

            # Compile PDF
            pdf_cmd = ["make", "all-pdf"]
            pdf_result = subprocess.run(
                pdf_cmd, cwd=latex_dir, capture_output=True, text=True
            )

            if pdf_result.returncode == 0:
                pdf_files = list(latex_dir.glob("*.pdf"))
                if pdf_files:
                    print(f"✅ PDF documentation generated: {pdf_files[0].name}")
                    self.quality_metrics["pdf_generated"] = True
                    return True

            print("⚠️  PDF compilation completed with warnings")
            return False

        except Exception as e:
            print(f"❌ PDF build error: {e}")
            return False

    def build_epub_documentation(self) -> bool:
        """Build ePub documentation for mobile/tablet reading."""
        print("📱 Building ePub documentation...")

        epub_dir = self.build_dir / "epub"
        cmd = [
            "sphinx-build",
            "-b",
            "epub",
            "-d",
            str(self.build_dir / "doctrees"),
            str(self.source_dir),
            str(epub_dir),
        ]

        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, cwd=self.project_root
            )

            if result.returncode == 0:
                epub_files = list(epub_dir.glob("*.epub"))
                if epub_files:
                    print(f"✅ ePub documentation generated: {epub_files[0].name}")
                    return True

            print("⚠️  ePub build completed with warnings")
            return False

        except Exception as e:
            print(f"❌ ePub build error: {e}")
            return False

    def validate_research_standards(self) -> bool:
        """Validate documentation against research publication standards."""
        print("🔬 Validating research documentation standards...")

        html_dir = self.build_dir / "html"
        if not html_dir.exists():
            print("❌ HTML documentation not found for validation")
            return False

        validation_passed = True

        # Check for research sections
        for section in [
            "index.html",
            "theoretical_framework.html",
            "computational_methods.html",
            "publications.html",
        ]:
            research_file = html_dir / "research" / section
            if research_file.exists():
                print(f"✅ Research section found: {section}")
            else:
                print(f"❌ Missing research section: {section}")
                validation_passed = False

        # Check for citations
        publications_file = html_dir / "research" / "publications.html"
        if publications_file.exists():
            content = publications_file.read_text()
            if "10.1073/pnas.2401162121" in content:
                print("✅ Primary research citation found")
                self.quality_metrics["citations_found"] = True
            else:
                print("⚠️  Primary research citation may be missing")

        # Check for mathematical content
        framework_file = html_dir / "research" / "theoretical_framework.html"
        if framework_file.exists():
            content = framework_file.read_text()
            if "MathJax" in content or "math-container" in content:
                print("✅ Mathematical content properly formatted")
                self.quality_metrics["mathematical_content"] = True
            else:
                print("⚠️  Mathematical content formatting needs attention")

        # Check API documentation coverage
        api_files = len(list(html_dir.glob("**/api*/*.html")))
        if api_files >= 5:
            print(f"✅ Comprehensive API documentation ({api_files} files)")
        else:
            print(f"⚠️  Limited API documentation coverage ({api_files} files)")

        self.quality_metrics["research_standards_met"] = validation_passed
        return validation_passed

    def check_links(self) -> bool:
        """Check for broken links in documentation."""
        print("🔗 Checking documentation links...")

        linkcheck_dir = self.build_dir / "linkcheck"
        cmd = [
            "sphinx-build",
            "-b",
            "linkcheck",
            "-d",
            str(self.build_dir / "doctrees"),
            str(self.source_dir),
            str(linkcheck_dir),
        ]

        try:
            subprocess.run(
                cmd, capture_output=True, text=True, cwd=self.project_root, timeout=300
            )

            output_file = linkcheck_dir / "output.txt"
            if output_file.exists():
                content = output_file.read_text()
                if "broken" in content.lower():
                    print("⚠️  Broken links detected - check linkcheck report")
                    return False
                else:
                    print("✅ No broken links found")
                    return True

            print("✅ Link check completed")
            return True

        except subprocess.TimeoutExpired:
            print("⚠️  Link check timed out")
            return False
        except Exception as e:
            print(f"⚠️  Link check error: {e}")
            return False

    def generate_coverage_report(self):
        """Generate documentation coverage report."""
        print("📊 Generating documentation coverage report...")

        coverage_dir = self.build_dir / "coverage"
        cmd = [
            "sphinx-build",
            "-b",
            "coverage",
            "-d",
            str(self.build_dir / "doctrees"),
            str(self.source_dir),
            str(coverage_dir),
        ]

        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, cwd=self.project_root
            )

            if result.returncode == 0:
                print("✅ Coverage report generated")
            else:
                print("⚠️  Coverage report completed with warnings")

        except Exception as e:
            print(f"⚠️  Coverage report error: {e}")

    def deploy_to_github_pages(self) -> bool:
        """Deploy documentation to GitHub Pages."""
        print("🚀 Deploying to GitHub Pages...")

        try:
            # Check for ghp-import
            subprocess.run(["ghp-import", "--help"], capture_output=True, check=True)
        except (FileNotFoundError, subprocess.CalledProcessError):
            print("❌ ghp-import not found. Install with: pip install ghp-import")
            return False

        html_dir = self.build_dir / "html"
        if not html_dir.exists():
            print("❌ HTML documentation not found for deployment")
            return False

        cmd = [
            "ghp-import",
            "-n",  # Include .nojekyll file
            "-p",  # Push to remote
            "-f",  # Force push
            "-m",
            f"Update documentation - {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            str(html_dir),
        ]

        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, cwd=self.project_root
            )

            if result.returncode == 0:
                print("✅ Successfully deployed to GitHub Pages")
                return True
            else:
                print("❌ GitHub Pages deployment failed:")
                print(result.stderr)
                return False

        except Exception as e:
            print(f"❌ GitHub Pages deployment error: {e}")
            return False

    def generate_quality_report(self) -> dict:
        """Generate comprehensive quality metrics report."""
        self.quality_metrics["build_time"] = time.time() - self.start_time

        # Calculate quality score
        quality_score = 0
        max_score = 100

        if self.quality_metrics["html_files"] > 10:
            quality_score += 20
        if self.quality_metrics["api_files"] >= 5:
            quality_score += 15
        if self.quality_metrics["research_files"] >= 4:
            quality_score += 20
        if self.quality_metrics["pdf_generated"]:
            quality_score += 15
        if self.quality_metrics["citations_found"]:
            quality_score += 15
        if self.quality_metrics["mathematical_content"]:
            quality_score += 10
        if self.quality_metrics["research_standards_met"]:
            quality_score += 5

        self.quality_metrics["quality_score"] = quality_score
        self.quality_metrics["max_score"] = max_score

        return self.quality_metrics

    def save_quality_report(self, report: dict):
        """Save quality report to file."""
        report_file = self.build_dir / "quality_report.json"
        with open(report_file, "w") as f:
            json.dump(report, f, indent=2)

        # Also create markdown summary
        md_file = self.build_dir / "quality_summary.md"
        with open(md_file, "w") as f:
            f.write("# Documentation Quality Report\n\n")
            f.write(f"Generated: {datetime.now().isoformat()}\n\n")
            f.write(
                f"## Overall Quality Score: {report['quality_score']}/{report['max_score']}\n\n"
            )
            f.write("## Metrics\n\n")
            f.write(f"- Build time: {report['build_time']:.1f} seconds\n")
            f.write(f"- HTML files: {report['html_files']}\n")
            f.write(f"- API documentation files: {report['api_files']}\n")
            f.write(f"- Research documentation files: {report['research_files']}\n")
            f.write(f"- PDF generated: {'✅' if report['pdf_generated'] else '❌'}\n")
            f.write(
                f"- Citations found: {'✅' if report['citations_found'] else '❌'}\n"
            )
            f.write(
                f"- Mathematical content: {'✅' if report['mathematical_content'] else '❌'}\n"
            )
            f.write(
                f"- Research standards met: {'✅' if report['research_standards_met'] else '❌'}\n"
            )

        print(f"📊 Quality report saved: {report_file}")


def main():
    """Main documentation deployment function."""
    parser = argparse.ArgumentParser(
        description="Research-grade documentation builder for homodyne-analysis"
    )

    parser.add_argument(
        "--target",
        choices=["html", "pdf", "epub", "github-pages", "all"],
        default="html",
        help="Documentation build target",
    )

    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate research documentation standards",
    )

    parser.add_argument(
        "--research-grade",
        action="store_true",
        help="Enable research-grade quality checks",
    )

    parser.add_argument(
        "--auto-deploy",
        action="store_true",
        help="Automatically deploy to configured targets",
    )

    parser.add_argument(
        "--build-only",
        action="store_true",
        help="Build documentation without deployment",
    )

    parser.add_argument(
        "--clean", action="store_true", help="Clean build directory before building"
    )

    args = parser.parse_args()

    # Initialize builder
    builder = ResearchDocumentationBuilder()

    # Validate environment
    if not builder.validate_environment():
        sys.exit(1)

    # Clean if requested
    if args.clean:
        builder.clean_build_directory()

    print(f"\n🚀 Starting documentation build - Target: {args.target}")

    success = True

    # Build documentation
    if args.target in ["html", "all"]:
        if not builder.build_html_documentation():
            success = False

    if args.target in ["pdf", "all"]:
        if not builder.build_pdf_documentation():
            success = False

    if args.target in ["epub", "all"]:
        if not builder.build_epub_documentation():
            success = False

    # Validation
    if args.validate or args.research_grade:
        if not builder.validate_research_standards():
            success = False

        builder.check_links()
        builder.generate_coverage_report()

    # Deployment
    if not args.build_only and args.auto_deploy:
        if args.target in ["github-pages", "all"]:
            if not builder.deploy_to_github_pages():
                success = False

    # Generate quality report
    quality_report = builder.generate_quality_report()
    builder.save_quality_report(quality_report)

    # Final summary
    print(f"\n{'=' * 60}")
    print("📚 Documentation Build Summary")
    print(f"{'=' * 60}")
    print(f"✅ Build Status: {'SUCCESS' if success else 'FAILED'}")
    print(
        f"📊 Quality Score: {quality_report['quality_score']}/{quality_report['max_score']}"
    )
    print(f"⏱️  Build Time: {quality_report['build_time']:.1f} seconds")
    print(f"📁 HTML Files: {quality_report['html_files']}")
    print(
        f"🔬 Research Standards: {'MET' if quality_report['research_standards_met'] else 'NEEDS WORK'}"
    )

    if not success:
        print("\n❌ Some builds failed - check output above for details")
        sys.exit(1)
    else:
        print("\n🎉 All documentation builds completed successfully!")
        sys.exit(0)


if __name__ == "__main__":
    main()
