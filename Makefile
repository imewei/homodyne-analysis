# Homodyne Analysis Package - Development Makefile
# ================================================

.PHONY: help clean clean-all clean-build clean-pyc clean-test clean-venv clean-cache clean-cache-full install dev-install install-hooks install-build-deps test test-all test-fast test-discovery test-full-discovery test-discovery-report test-performance test-regression test-ci test-parallel test-cached lint format quality quality-fast ruff-fix pre-commit docs docs-serve docs-validate docs-check docs-stats docs-clean build upload check build-fast build-dev build-prod build-optimize watch benchmark cache-stats build-health profile-memory dev-optimize baseline-update baseline-reset baseline-report performance-analysis performance-analysis-quick

# Default target
help:
	@echo "Homodyne Analysis Package - Development Commands"
	@echo "=============================================="
	@echo
	@echo "Development:"
	@echo "  install      Install package in editable mode"
	@echo "  dev-install  Install package with all development dependencies"
	@echo
	@echo "Testing:"
	@echo "  test         Run tests with pytest"
	@echo "  test-all     Run tests with all optional dependencies"
	@echo "  test-discovery       Run comprehensive test discovery (no failure limits)"
	@echo "  test-full-discovery  Run full test suite discovery with detailed reporting"
	@echo "  test-discovery-report Generate comprehensive test discovery report"
	@echo "  test-performance     Run performance tests only"
	@echo "  test-regression      Run performance regression tests"
	@echo "  test-ci      Run CI-style performance tests"
	@echo "  lint         Run code linting (flake8, mypy)"
	@echo "  format       Format code with black"
	@echo
	@echo "Performance Baselines:"
	@echo "  baseline-update      Update performance baselines"
	@echo "  baseline-reset       Reset all performance baselines"
	@echo "  baseline-report      Generate performance report"
	@echo "  performance-analysis Run comprehensive performance analysis"
	@echo "  performance-analysis-quick  Run quick performance analysis"
	@echo
	@echo "Cleanup:"
	@echo "  clean        Clean all build artifacts and cache files (preserves virtual environment)"
	@echo "  clean-all    Clean everything including virtual environment"
	@echo "  clean-build  Remove build artifacts"
	@echo "  clean-pyc    Remove Python bytecode files"
	@echo "  clean-test   Remove test and coverage artifacts"
	@echo "  clean-venv   Remove virtual environment"
	@echo
	@echo "Documentation:"
	@echo "  docs           Build documentation"
	@echo "  docs-serve     Serve documentation locally"
	@echo "  docs-validate  Validate documentation completeness"
	@echo "  docs-check     Check documentation quality and links"
	@echo "  docs-stats     Show documentation statistics"
	@echo "  docs-clean     Clean documentation build artifacts"
	@echo
	@echo "Packaging:"
	@echo "  build        Build distribution packages"
	@echo "  upload       Upload to PyPI"
	@echo "  check        Check package metadata and distribution"

# Installation targets
install:
	pip install -e .

dev-install:
	pip install -e ".[all,dev,docs]"

# Testing targets
test:
	@echo "🧪 Running optimized test suite..."
	python -c "import sys; import os; os.environ['PYTHONWARNINGS'] = 'ignore'; sys.modules['numba'] = None; sys.modules['pymc'] = None; sys.modules['arviz'] = None; sys.modules['corner'] = None; import pytest; pytest.main(['-v', '--tb=short', '--continue-on-collection-errors'])"

test-all:
	@echo "🧪 Running comprehensive test suite with coverage..."
	pytest -v --cov=homodyne --cov-report=html --cov-report=term --continue-on-collection-errors

test-fast:
	@echo "⚡ Running fast test suite..."
	python -c "import sys; import os; os.environ['PYTHONWARNINGS'] = 'ignore'; sys.modules['numba'] = None; sys.modules['pymc'] = None; sys.modules['arviz'] = None; sys.modules['corner'] = None; import pytest; result = pytest.main(['-q', '--tb=no', '--continue-on-collection-errors', '--maxfail=3']); print(f'\nTest result code: {result}')"

# Comprehensive test discovery targets
test-discovery:
	@echo "🔍 Running comprehensive test discovery (no failure limits)..."
	python -c "import sys; import os; os.environ['PYTHONWARNINGS'] = 'ignore'; sys.modules['numba'] = None; sys.modules['pymc'] = None; sys.modules['arviz'] = None; sys.modules['corner'] = None; import pytest; pytest.main(['-v', '--tb=short', '--continue-on-collection-errors', '--maxfail=0'])"

test-full-discovery:
	@echo "🧪 Running full test suite discovery with detailed reporting..."
	pytest homodyne/tests/ -v --tb=short --continue-on-collection-errors --maxfail=0 --durations=0

test-discovery-report:
	@echo "📊 Running test discovery with comprehensive reporting..."
	pytest homodyne/tests/ -v --tb=short --continue-on-collection-errors --maxfail=0 --durations=0 --junit-xml=test-discovery-results.xml --html=test-discovery-report.html --self-contained-html

# Performance testing targets
test-performance:
	pytest homodyne/tests/ -v -m performance

test-regression:
	@echo "Running performance regression tests..."
	pytest homodyne/tests/test_performance.py -v -m regression

test-ci:
	@echo "Running CI-style performance tests..."
	pytest homodyne/tests/test_performance.py -v --tb=short

# Performance baseline management
baseline-update:
	@echo "Updating performance baselines..."
	pytest homodyne/tests/test_performance.py -v --update-baselines
	@echo "✓ Baselines updated successfully"

baseline-reset:
	@echo "Resetting performance baselines..."
	rm -f ci_performance_baselines.json
	rm -f homodyne_test_performance_baselines.json
	rm -f homodyne/tests/test_performance_baselines.json
	rm -f homodyne/tests/performance_baselines.json
	@echo "✓ Baselines reset"

baseline-report:
	@echo "Generating performance report..."
	pytest homodyne/tests/test_performance.py -v --tb=short --durations=0
	@echo "✓ Performance report completed"

# Performance analysis (integrated into test suite)
performance-analysis:
	@echo "🔍 Running performance analysis via test suite..."
	python -c "import pytest; pytest.main(['-v', '-m', 'benchmark', 'homodyne/tests/test_initialization_optimization.py::TestPerformanceBenchmarks::test_startup_time_benchmark'])"

performance-analysis-quick:
	@echo "⚡ Running quick performance check..."
	python -c "import time; start=time.time(); import homodyne; print(f'Import time: {time.time()-start:.4f}s')"

# Code quality targets
lint:
	@echo "🔍 Running linting checks..."
	ruff check homodyne/
	mypy homodyne/

format:
	@echo "✨ Formatting code..."
	ruff format homodyne/ tests/
	black homodyne/ tests/
	isort homodyne/ tests/

ruff-fix:
	@echo "🔧 Auto-fixing linting issues..."
	ruff check --fix homodyne/

quality: format lint
	@echo "✅ Code quality checks completed"

# Cleanup targets
clean: clean-build clean-pyc clean-test
	@echo "Cleaned all build artifacts and cache files"

clean-cache:
	find . -name '__pycache__' -type d -exec rm -rf {} +
	find . -name '*.pyc' -delete
	rm -rf .pytest_cache/ .mypy_cache/ .ruff_cache/
	@echo "Cleaned Python cache files"

clean-build:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .eggs/

clean-pyc:
	find . -name '*.pyc' -delete
	find . -name '*.pyo' -delete
	find . -name '*~' -delete
	find . -name '__pycache__' -type d -exec rm -rf {} +
	find . -name '*.orig' -delete
	find . -name '*.rej' -delete

clean-test:
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	rm -rf .benchmarks/
	rm -rf reports/
	rm -f bandit*.json
	rm -rf node_modules/
	rm -f test_data.json
	rm -f test_config.json
	rm -f test_array_data.npz
	rm -f test_report.txt
	rm -f cache_*.npz
	rm -rf homodyne_results/
	rm -rf ml_optimization_data/
	rm -rf performance_results/
	rm -f nonexistent_template
	rm -f static_isotropic
	rm -f test_*.npz
	rm -f test_*.json
	rm -f phi_angles_list.txt
	rm -rf homodyne_results/

clean-venv:
	rm -rf venv/
	rm -rf .venv/
	rm -rf env/
	rm -rf .env/

clean-all: clean-build clean-pyc clean-test clean-venv
	@echo "Cleaned all build artifacts, cache files, and virtual environment"

# Documentation targets
docs:
	$(MAKE) -C docs html

docs-serve:
	@echo "Serving documentation at http://localhost:8000"
	cd docs/_build/html && python -m http.server 8000

docs-validate:
	@echo "🔍 Validating documentation completeness..."
	@echo "Checking README.md..."
	@test -f README.md && echo "✓ README.md exists" || echo "✗ README.md missing"
	@echo "Checking API documentation..."
	@test -f docs/api/README.md && echo "✓ API index exists" || echo "✗ API index missing"
	@test -f docs/api/analysis_core.md && echo "✓ Core API docs exist" || echo "✗ Core API docs missing"
	@echo "Checking research documentation..."
	@test -f docs/research/methodology.md && echo "✓ Methodology docs exist" || echo "✗ Methodology docs missing"
	@echo "Checking documentation summary..."
	@test -f DOCUMENTATION_SUMMARY.md && echo "✓ Documentation summary exists" || echo "✗ Summary missing"
	@echo "✅ Documentation validation complete"

docs-check:
	@echo "🔍 Checking documentation quality..."
	@if command -v markdownlint >/dev/null 2>&1; then \
		markdownlint README.md docs/**/*.md; \
	else \
		echo "⚠️  markdownlint not installed (npm install -g markdownlint-cli)"; \
	fi
	@echo "Checking for broken links..."
	@grep -r "http" docs/*.md README.md | grep -v "^\s*#" || true
	@echo "✅ Documentation quality check complete"

docs-stats:
	@echo "📊 Documentation statistics:"
	@echo "README.md: $$(wc -l < README.md) lines"
	@echo "API docs: $$(find docs/api -name '*.md' -exec wc -l {} + | tail -1 | awk '{print $$1}') lines"
	@echo "Research docs: $$(find docs/research -name '*.md' -exec wc -l {} + | tail -1 | awk '{print $$1}') lines"
	@echo "Total documentation: $$(find . -name '*.md' -not -path './node_modules/*' -exec wc -l {} + | tail -1 | awk '{print $$1}') lines"
	@echo "✅ Documentation statistics complete"

docs-clean:
	@echo "🧹 Cleaning documentation build artifacts..."
	rm -rf docs/_build/
	@echo "✅ Documentation cleaned"

# Packaging targets
build: clean
	python -m build

upload: build
	python -m twine upload dist/*

check:
	python -m twine check dist/*
	python -m build --check

pre-commit:
	pre-commit run --all-files

install-hooks:
	pre-commit install

# Build Optimization Targets
# ==========================

# Fast development build with caching
build-fast:
	@echo "🚀 Fast development build with caching..."
	python build_optimizer.py --mode dev --cache

# Development build with hot reload
build-dev:
	@echo "🔥 Starting development server with hot reload..."
	python build_optimizer.py --mode dev --watch

# Production optimized build
build-prod:
	@echo "⚡ Production build with maximum optimization..."
	python build_optimizer.py --mode production --optimize

# Watch mode for continuous development
watch:
	@echo "👀 Starting file watcher for automatic rebuilds..."
	python build_optimizer.py --watch

# Build performance benchmarking
benchmark:
	@echo "📊 Running build performance benchmarks..."
	python build_optimizer.py --benchmark --compare-baseline

# Build optimization analysis
build-optimize:
	@echo "🔧 Analyzing and optimizing build performance..."
	python build_optimizer.py --mode production --optimize --benchmark

# Cache statistics and cleanup
cache-stats:
	@echo "📈 Build cache statistics:"
	@if [ -f .build_cache/cache_metrics.json ]; then \
		python -c "import json; data=json.load(open('.build_cache/cache_metrics.json')); print(f'Cache hit rate: {data.get(\"hit_rate\", 0):.1f}%'); print(f'Total entries: {data.get(\"total_entries\", 0)}'); print(f'Last updated: {data.get(\"last_updated\", \"Never\")}')"; \
	else \
		echo "No cache statistics available. Run 'make build-fast' first."; \
	fi

# Parallel testing with optimization
test-parallel:
	@echo "🧪 Running tests in parallel..."
	pytest -n auto --tb=short --continue-on-collection-errors

# Fast test execution with caching
test-cached:
	@echo "⚡ Running cached tests..."
	pytest --cache-clear -q --tb=no --continue-on-collection-errors

# Optimized code quality checks
quality-fast:
	@echo "✨ Running optimized code quality checks..."
	ruff check --fix homodyne/ & \
	ruff format homodyne/ & \
	mypy homodyne/ & \
	wait
	@echo "Code quality checks completed in parallel"

# Clean build cache
clean-cache-full:
	rm -rf .build_cache/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	find . -name '__pycache__' -type d -exec rm -rf {} +
	@echo "All caches cleaned"

# Development environment optimization
dev-optimize:
	@echo "🔧 Optimizing development environment..."
	pip install -e ".[performance,dev]" --upgrade
	pre-commit install
	python build_optimizer.py --clean --cache
	@echo "Development environment optimized"

# Memory profiling during builds
profile-memory:
	@echo "📊 Profiling memory usage during build..."
	python -m memory_profiler build_optimizer.py --mode dev

# Build system health check
build-health:
	@echo "🏥 Build system health check..."
	@python -c "import sys, psutil, multiprocessing; print(f'Python: {sys.version_info[:2]}'); print(f'CPU cores: {multiprocessing.cpu_count()}'); print(f'Memory: {psutil.virtual_memory().total // (1024**3)} GB'); print(f'Available: {psutil.virtual_memory().available // (1024**3)} GB')"
	@if command -v ccache >/dev/null 2>&1; then echo "ccache: available"; else echo "ccache: not installed (consider installing for faster C/C++ compilation)"; fi
	@if command -v ninja >/dev/null 2>&1; then echo "ninja: available"; else echo "ninja: not installed (consider installing for faster builds)"; fi

# Install build optimization dependencies
install-build-deps:
	@echo "📦 Installing build optimization dependencies..."
	pip install psutil toml watchdog cython joblib threadpoolctl
	if command -v brew >/dev/null 2>&1; then \
		brew install ccache ninja; \
	elif command -v apt-get >/dev/null 2>&1; then \
		sudo apt-get install ccache ninja-build; \
	fi
	@echo "Build optimization dependencies installed"
