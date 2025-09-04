# Makefile for Sphinx documentation
# Enhanced documentation build system for homodyne-analysis
# Optimized for performance, reliability, and developer experience

# ======================================================================
# Configuration Variables
# ======================================================================

# Core build tools (can be overridden from command line or environment)
SPHINXOPTS    ?= -j auto -q --color
SPHINXBUILD   ?= sphinx-build
SPHINXAUTOBUILD ?= sphinx-autobuild
SOURCEDIR     = .
BUILDDIR      = _build

# Performance and quality options
STRICT_OPTS   = -W -E --keep-going
FAST_OPTS     = -j auto -q -E 
PARALLEL_JOBS ?= auto
VERBOSE_OPTS  = -v --color

# Build directories
HTMLDIR       = $(BUILDDIR)/html
DOCTREEDIR    = $(BUILDDIR)/doctrees
CACHEDIR      = $(BUILDDIR)/.docfx_cache

# Additional optimized options
ALLSPHINXOPTS = -d $(DOCTREEDIR) $(SPHINXOPTS)

# ======================================================================
# Help and Default Targets  
# ======================================================================

# Put it first so that "make" without argument is like "make help".
help:
	@echo "Homodyne Documentation Build System"
	@echo "====================================="
	@echo ""
	@echo "Quick Start Targets:"
	@echo "  html          - Build HTML documentation (optimized)"
	@echo "  fast          - Fast development build (minimal checks)"
	@echo "  live          - Live reload development server"
	@echo "  clean         - Clean all build artifacts"
	@echo ""
	@echo "Quality Assurance:"
	@echo "  strict        - Strict build with all warnings as errors"
	@echo "  check         - Run all quality checks (links, spelling, doctests)"
	@echo "  linkcheck     - Check external links"
	@echo "  spelling      - Check spelling (requires sphinxcontrib-spelling)"
	@echo ""
	@echo "Advanced Targets:"
	@echo "  all           - Build all formats (HTML, PDF, EPUB)"
	@echo "  pdf           - Build PDF via LaTeX"
	@echo "  epub          - Build EPUB format"
	@echo "  coverage      - Generate documentation coverage report"
	@echo ""
	@echo "Deployment:"
	@echo "  deploy        - Deploy to GitHub Pages"
	@echo "  preview       - Open built documentation in browser"
	@echo ""
	@echo "Environment Variables:"
	@echo "  SPHINXOPTS    - Additional Sphinx options (default: $(SPHINXOPTS))"
	@echo "  PARALLEL_JOBS - Parallel build jobs (default: $(PARALLEL_JOBS))"
	@echo ""

# Default target - build optimized HTML
default: html

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%:
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

# ======================================================================
# Core Build Targets
# ======================================================================

# Clean all build artifacts with progress indication
clean:
	@echo "🧹 Cleaning build directory..."
	@if [ -d "$(BUILDDIR)" ]; then \
		echo "  → Removing $(BUILDDIR)"; \
		rm -rf $(BUILDDIR); \
		echo "  ✅ Clean complete"; \
	else \
		echo "  ℹ️ Build directory already clean"; \
	fi

# Smart clean - preserves cached doctrees if source hasn't changed significantly
clean-cache:
	@echo "🗂️ Cleaning caches only..."
	@rm -rf $(CACHEDIR) $(BUILDDIR)/.buildinfo
	@echo "  ✅ Cache cleaned"

# Optimized HTML build with progress indication and error handling
html:
	@echo "🔨 Building HTML documentation..."
	@echo "  → Using $(PARALLEL_JOBS) parallel jobs"
	@mkdir -p $(HTMLDIR)
	@if $(SPHINXBUILD) -b html $(FAST_OPTS) "$(SOURCEDIR)" "$(HTMLDIR)" $(ALLSPHINXOPTS); then \
		echo "  ✅ HTML build complete → file://$(shell pwd)/$(HTMLDIR)/index.html"; \
	else \
		echo "  ❌ HTML build failed"; \
		exit 1; \
	fi

# Alias for html (backward compatibility)
docs: html

# Strict build with all warnings as errors - for CI/CD
strict:
	@echo "🔍 Building HTML documentation (strict mode)..."
	@mkdir -p $(HTMLDIR)
	@if $(SPHINXBUILD) -b html $(STRICT_OPTS) "$(SOURCEDIR)" "$(HTMLDIR)" $(ALLSPHINXOPTS); then \
		echo "  ✅ Strict build passed"; \
	else \
		echo "  ❌ Strict build failed - fix all warnings"; \
		exit 1; \
	fi

# Fast development build with minimal validation
fast:
	@echo "⚡ Fast development build..."
	@mkdir -p $(HTMLDIR) 
	@$(SPHINXBUILD) -b html -j $(PARALLEL_JOBS) -q -E "$(SOURCEDIR)" "$(HTMLDIR)" $(ALLSPHINXOPTS)
	@echo "  ✅ Fast build complete"

# Live reload development server with optimized settings  
live:
	@echo "🔄 Starting live reload server..."
	@echo "  → Server will be available at http://127.0.0.1:8000"
	@echo "  → Press Ctrl+C to stop"
	@$(SPHINXAUTOBUILD) --port 8000 --delay 1 \
		--ignore "*.tmp" --ignore "*.swp" --ignore "*~" \
		--watch "$(SOURCEDIR)" \
		"$(SOURCEDIR)" "$(HTMLDIR)" $(ALLSPHINXOPTS)

# Legacy alias for live reload
livehtml: live

# ======================================================================
# Quality Assurance Targets
# ======================================================================

# Comprehensive quality check suite
check: linkcheck spelling doctest
	@echo "✅ All quality checks completed"

# Fast link checking with progress indication
linkcheck:
	@echo "🔗 Checking external links..."
	@mkdir -p $(BUILDDIR)/linkcheck
	@if $(SPHINXBUILD) -b linkcheck -j $(PARALLEL_JOBS) "$(SOURCEDIR)" "$(BUILDDIR)/linkcheck" $(ALLSPHINXOPTS); then \
		echo "  ✅ Link check passed"; \
	else \
		echo "  ⚠️ Link check found issues - see $(BUILDDIR)/linkcheck/output.txt"; \
	fi

# Spelling check with enhanced feedback (requires sphinxcontrib-spelling)
spelling:
	@echo "📝 Checking spelling..."
	@mkdir -p $(BUILDDIR)/spelling
	@if command -v aspell >/dev/null 2>&1; then \
		if $(SPHINXBUILD) -b spelling "$(SOURCEDIR)" "$(BUILDDIR)/spelling" $(ALLSPHINXOPTS); then \
			echo "  ✅ Spelling check passed"; \
		else \
			echo "  ⚠️ Spelling issues found - see $(BUILDDIR)/spelling/output.txt"; \
		fi \
	else \
		echo "  ⚠️ aspell not found - install with: apt-get install aspell aspell-en"; \
	fi

# Test code examples in documentation with detailed output
doctest:
	@echo "🧪 Testing code examples..."
	@mkdir -p $(BUILDDIR)/doctest
	@if $(SPHINXBUILD) -b doctest "$(SOURCEDIR)" "$(BUILDDIR)/doctest" $(ALLSPHINXOPTS); then \
		echo "  ✅ All doctests passed"; \
	else \
		echo "  ❌ Some doctests failed - check $(BUILDDIR)/doctest/output.txt"; \
		exit 1; \
	fi

# Documentation coverage report with summary
coverage:
	@echo "📊 Generating documentation coverage report..."
	@mkdir -p $(BUILDDIR)/coverage
	@$(SPHINXBUILD) -b coverage "$(SOURCEDIR)" "$(BUILDDIR)/coverage" $(ALLSPHINXOPTS)
	@echo "  📈 Coverage report: $(BUILDDIR)/coverage/python.txt"
	@if [ -f "$(BUILDDIR)/coverage/python.txt" ]; then \
		echo "  📋 Summary:"; \
		tail -10 "$(BUILDDIR)/coverage/python.txt" | grep -E "(Undocumented|Total)"; \
	fi

# ======================================================================
# Advanced Format Targets
# ======================================================================

# Build all formats with progress tracking
all: clean html pdf epub
	@echo "🎯 All formats built successfully"

# Create PDF via LaTeX with error handling
pdf:
	@echo "📄 Building PDF documentation..."
	@mkdir -p $(BUILDDIR)/latex
	@echo "  → Building LaTeX source..."
	@if $(SPHINXBUILD) -b latex "$(SOURCEDIR)" "$(BUILDDIR)/latex" $(ALLSPHINXOPTS); then \
		echo "  → Compiling LaTeX to PDF..."; \
		if $(MAKE) -C $(BUILDDIR)/latex all-pdf LATEXMKOPTS="-silent"; then \
			echo "  ✅ PDF created: $(BUILDDIR)/latex/homodyne-analysis.pdf"; \
		else \
			echo "  ❌ PDF compilation failed"; \
			exit 1; \
		fi \
	else \
		echo "  ❌ LaTeX generation failed"; \
		exit 1; \
	fi

# Legacy alias for pdf
latexpdf: pdf

# Create EPUB format with validation
epub:
	@echo "📚 Building EPUB documentation..."
	@mkdir -p $(BUILDDIR)/epub
	@if $(SPHINXBUILD) -b epub "$(SOURCEDIR)" "$(BUILDDIR)/epub" $(ALLSPHINXOPTS); then \
		echo "  ✅ EPUB created: $(BUILDDIR)/epub/homodyne-analysis.epub"; \
	else \
		echo "  ❌ EPUB generation failed"; \
		exit 1; \
	fi

# JSON output for programmatic access
json:
	@echo "🔧 Building JSON documentation..."
	@mkdir -p $(BUILDDIR)/json
	@$(SPHINXBUILD) -b json -j $(PARALLEL_JOBS) "$(SOURCEDIR)" "$(BUILDDIR)/json" $(ALLSPHINXOPTS)
	@echo "  ✅ JSON build complete"

# ======================================================================
# Utility and Deployment Targets
# ======================================================================

# Open documentation in browser with cross-platform support
preview:
	@echo "🌐 Opening documentation in browser..."
	@if [ -f "$(HTMLDIR)/index.html" ]; then \
		if command -v xdg-open >/dev/null 2>&1; then \
			xdg-open "file://$(shell pwd)/$(HTMLDIR)/index.html"; \
		elif command -v open >/dev/null 2>&1; then \
			open "file://$(shell pwd)/$(HTMLDIR)/index.html"; \
		else \
			python3 -m webbrowser "file://$(shell pwd)/$(HTMLDIR)/index.html"; \
		fi; \
		echo "  ✅ Documentation opened"; \
	else \
		echo "  ❌ HTML documentation not found - run 'make html' first"; \
		exit 1; \
	fi

# Legacy alias for preview
open: preview

# Deploy to GitHub Pages with validation (requires ghp-import)
deploy:
	@echo "🚀 Deploying to GitHub Pages..."
	@if [ ! -d "$(HTMLDIR)" ]; then \
		echo "  ❌ HTML documentation not found - run 'make html' first"; \
		exit 1; \
	fi
	@if command -v ghp-import >/dev/null 2>&1; then \
		ghp-import -n -p -f $(HTMLDIR); \
		echo "  ✅ Deployed to GitHub Pages"; \
	else \
		echo "  ❌ ghp-import not found - install with: pip install ghp-import"; \
		exit 1; \
	fi

# Development server status and info  
status:
	@echo "📊 Documentation Build Status"
	@echo "=============================="
	@echo "Source directory: $(SOURCEDIR)"
	@echo "Build directory:  $(BUILDDIR)"
	@echo "HTML directory:   $(HTMLDIR)"
	@echo "Parallel jobs:    $(PARALLEL_JOBS)"
	@echo "Sphinx options:   $(SPHINXOPTS)"
	@echo ""
	@if [ -d "$(BUILDDIR)" ]; then \
		echo "📁 Build directory contents:"; \
		ls -la $(BUILDDIR) 2>/dev/null || true; \
		echo ""; \
		if [ -f "$(HTMLDIR)/index.html" ]; then \
			echo "✅ HTML documentation exists"; \
			echo "📊 HTML size: $$(du -sh $(HTMLDIR) 2>/dev/null | cut -f1)"; \
		else \
			echo "❌ HTML documentation not built"; \
		fi \
	else \
		echo "❌ No build directory found"; \
	fi

# Watch for changes and rebuild (alias for live)
watch: live

# ======================================================================
# PHONY Target Declaration
# ======================================================================

.PHONY: help default clean clean-cache html docs strict fast live livehtml \
        check linkcheck spelling doctest coverage all pdf latexpdf epub json \
        preview open deploy status watch
