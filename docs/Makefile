# Makefile for Sphinx documentation
# Enhanced documentation build system for homodyne-analysis

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?= -j auto
SPHINXBUILD   ?= sphinx-build
SPHINXAUTOBUILD ?= sphinx-autobuild
SOURCEDIR     = .
BUILDDIR      = _build

# Additional options
ALLSPHINXOPTS = -d $(BUILDDIR)/doctrees $(SPHINXOPTS)

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

# Custom targets
clean:
	@echo "Cleaning build directory..."
	rm -rf $(BUILDDIR)

html-strict:
	@echo "Building HTML documentation with strict warnings..."
	@$(SPHINXBUILD) -W -b html "$(SOURCEDIR)" "$(BUILDDIR)/html" $(SPHINXOPTS)

livehtml:
	@echo "Starting live reload server..."
	sphinx-autobuild "$(SOURCEDIR)" "$(BUILDDIR)/html" $(SPHINXOPTS)

linkcheck:
	@echo "Checking external links..."
	@$(SPHINXBUILD) -b linkcheck "$(SOURCEDIR)" "$(BUILDDIR)/linkcheck" $(SPHINXOPTS)

# Quick build without warnings for development
dev:
	@echo "Building HTML documentation (development mode)..."
	@$(SPHINXBUILD) -b html "$(SOURCEDIR)" "$(BUILDDIR)/html" $(ALLSPHINXOPTS)

# Build all formats
all: clean html latexpdf epub

# Check documentation quality
check: linkcheck doctest spelling

# Spelling check (requires sphinxcontrib-spelling)
spelling:
	@echo "Checking spelling..."
	@$(SPHINXBUILD) -b spelling "$(SOURCEDIR)" "$(BUILDDIR)/spelling" $(ALLSPHINXOPTS)

# Test code examples in documentation
doctest:
	@echo "Testing code examples..."
	@$(SPHINXBUILD) -b doctest "$(SOURCEDIR)" "$(BUILDDIR)/doctest" $(ALLSPHINXOPTS)

# Coverage report for documentation
coverage:
	@echo "Generating documentation coverage report..."
	@$(SPHINXBUILD) -b coverage "$(SOURCEDIR)" "$(BUILDDIR)/coverage" $(ALLSPHINXOPTS)
	@echo "Coverage report: $(BUILDDIR)/coverage/python.txt"

# Open documentation in browser
open:
	@echo "Opening documentation in browser..."
	@python -m webbrowser "file://$(shell pwd)/$(BUILDDIR)/html/index.html"

# Deploy to GitHub Pages (requires git)
deploy:
	@echo "Deploying to GitHub Pages..."
	@ghp-import -n -p -f $(BUILDDIR)/html

# Create PDF via LaTeX
latexpdf:
	@echo "Building LaTeX/PDF documentation..."
	@$(SPHINXBUILD) -b latex "$(SOURCEDIR)" "$(BUILDDIR)/latex" $(ALLSPHINXOPTS)
	@echo "Running LaTeX files through pdflatex..."
	@$(MAKE) -C $(BUILDDIR)/latex all-pdf

# Create EPUB format
epub:
	@echo "Building EPUB documentation..."
	@$(SPHINXBUILD) -b epub "$(SOURCEDIR)" "$(BUILDDIR)/epub" $(ALLSPHINXOPTS)

# JSON output for programmatic access
json:
	@echo "Building JSON documentation..."
	@$(SPHINXBUILD) -b json "$(SOURCEDIR)" "$(BUILDDIR)/json" $(ALLSPHINXOPTS)

# Watch for changes and rebuild
watch: livehtml

.PHONY: help clean html dev all check spelling doctest coverage open deploy latexpdf epub json watch livehtml linkcheck html-strict
