# Configuration file for the Sphinx documentation builder.

import os
import sys

# Add the parent directory (containing homodyne package) to Python path
sys.path.insert(0, os.path.abspath(".."))

# Also add the homodyne package directory itself
homodyne_path = os.path.join(os.path.abspath(".."), "homodyne")
if os.path.exists(homodyne_path):
    sys.path.insert(0, homodyne_path)

# Set up import error handling for documentation builds
os.environ['SPHINX_BUILD'] = '1'

# -- Project information -----------------------------------------------------
project = "Homodyne Analysis"
copyright = "2024-2025, Wei Chen, Hongrui He (Argonne National Laboratory)"
author = "Wei Chen, Hongrui He"

# Get version dynamically
try:
    from importlib.metadata import version as get_version

    release = get_version("homodyne-analysis")
except Exception:
    release = "0.7.2"
version = ".".join(release.split(".")[:2])  # X.Y version

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.autosummary",
    "sphinx.ext.doctest",
    "sphinx.ext.todo",
    "sphinx.ext.coverage",
    "sphinx.ext.githubpages",
    "sphinx_autodoc_typehints",
    "sphinx_copybutton",
    "myst_parser",
    "numpydoc",
]

# Suppress specific warnings to reduce noise
suppress_warnings = [
    "misc.highlighting_failure",
    "autosummary",
    "autosummary.import_by_name",
    "autosummary.failed_import",
    "autosummary.failed_to_import",
    "autosummary.mock",
    "autodoc.import_object",
    "autodoc.mock",
    "toc.not_included",
    "ref.any",
    "ref.python",
    "toc.secnum",
    "image.not_readable",
    "download.not_readable",
]

# Performance optimizations - mock heavy dependencies
autodoc_mock_imports = [
    # Heavy scientific computing dependencies
    "numba",
    "pymc",
    "arviz",
    "pytensor",
    "jax",
    "jax.numpy",
    "numpyro",
    "numpyro.distributions",
    "numpyro.infer",
    "numpyro.diagnostics",
    "xpcs_viewer",
    "h5py",
    "matplotlib",
    "scipy",
    "numpy",
    # Mock problematic import dependencies that may not be available
    "gurobipy",
    "cvxpy",
    # Test modules that may cause import errors during doc build
    "pytest",
    "pytest_benchmark", 
    "pytest_mock",
]
autodoc_preserve_defaults = True

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# The default language to highlight source code in.
highlight_language = "python3"

# -- Options for extensions --------------------------------------------------

# autodoc configuration
autodoc_typehints = "description"
autodoc_member_order = "bysource"
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
    "special-members": "__init__",
    "exclude-members": "__weakref__",
}

# Optimize autodoc performance
autodoc_class_signature = "mixed"
autodoc_inherit_docstrings = True
autodoc_typehints_format = "short"

# napoleon configuration
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_type_aliases = None

# intersphinx configuration
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
}

# MyST parser configuration
myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "html_admonition",
    "html_image",
    "linkify",
    "replacements",
    "smartquotes",
    "substitution",
    "tasklist",
]

# Configure MyST parser for better performance
myst_heading_anchors = 2
myst_footnote_transition = True
myst_dmath_double_inline = True

# -- Extension configurations ------------------------------------------------

# AutoSummary configuration
autosummary_generate = True
autosummary_generate_overwrite = True
autosummary_imported_members = False  # Don't document imported members
autosummary_ignore_module_all = False  # Respect __all__ if defined
autosummary_mock_imports = True  # Allow mocking of imports in autosummary


# Configure autosummary to skip test modules that require pytest
def skip_test_modules(app, what, name, obj, skip, options):
    """Skip test modules and conftest files during documentation generation."""
    if skip:
        return skip

    # Skip modules that contain 'test' or 'conftest' in the name
    if "test" in name.lower() or "conftest" in name.lower():
        return True

    # Skip modules in tests directories
    if ".tests." in name:
        return True

    return skip


def setup(app):
    """Sphinx setup function."""
    app.connect("autodoc-skip-member", skip_test_modules)


# Copy button configuration
copybutton_prompt_text = r">>> |\.\.\. |\$ |In \[\d*\]: | {2,5}\.\.\.: | {5,8}: "
copybutton_prompt_is_regexp = True
copybutton_only_copy_prompt_lines = True
copybutton_remove_prompts = True

# NumPy docstring configuration
numpydoc_show_class_members = False
numpydoc_class_members_toctree = True
numpydoc_attributes_as_param_list = True
numpydoc_xref_param_type = True

# Todo configuration
todo_include_todos = False

# Type hints configuration
typehints_fully_qualified = False
always_document_param_types = True
typehints_document_rtype = True

# Add substitutions for common mathematical symbols
myst_substitutions = {
    "g1": r"$g_1$",
    "g2": r"$g_2$",
    "chi2": r"$\chi^2$",
    "alpha": r"$\alpha$",
    "beta": r"$\beta$",
    "gamma": r"$\gamma$",
}

# -- Options for HTML output -------------------------------------------------
html_theme = "furo"
html_title = f"{project} v{version}"
html_theme_options = {
    "sidebar_hide_name": False,
    "light_logo": "logo-light.png",
    "dark_logo": "logo-dark.png",
    "light_css_variables": {
        "color-brand-primary": "#2980b9",
        "color-brand-content": "#2c5282",
        "color-admonition-background": "transparent",
    },
    "dark_css_variables": {
        "color-brand-primary": "#4fc3f7",
        "color-brand-content": "#81d4fa",
        "color-admonition-background": "transparent",
    },
    "footer_icons": [
        {
            "name": "GitHub",
            "url": "https://github.com/imewei/homodyne",
            "html": """
                <svg stroke="currentColor" fill="currentColor" stroke-width="0" viewBox="0 0 16 16">
                    <path fill-rule="evenodd" d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0 0 16 8c0-4.42-3.58-8-8-8z"></path>
                </svg>
            """,
            "class": "",
        },
        {
            "name": "PyPI",
            "url": "https://pypi.org/project/homodyne-analysis/",
            "html": """
                <svg stroke="currentColor" fill="currentColor" stroke-width="0" viewBox="0 0 24 24">
                    <path d="M12.357 12.617l-1.037 5.732L8.7 10.617h-1.38l2.85 12.05h1.18l.9-5.01h.01l.9 5.01h1.18l2.85-12.05h-1.38l-2.62 7.732-.9-5.732h-1.01zm8.72-9.58V1.617c0-.55-.45-1-1-1H4.017c-.55 0-1 .45-1 1v14.52c0 .55.45 1 1 1h15.06c.55 0 1-.45 1-1V3.037zm-1 .42v13.1H5.017V2.037h14.06V3.037z"></path>
                </svg>
            """,
            "class": "",
        },
    ],
}

# Optimize HTML output
html_copy_source = False
html_show_sourcelink = False
html_compact_lists = True
html_secnumber_suffix = ". "

html_static_path = ["_static"]

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "sphinx"

# -- Options for LaTeX output ------------------------------------------------
latex_elements = {
    "papersize": "letterpaper",
    "pointsize": "10pt",
    "preamble": "",
    "fncychap": "\\usepackage[Bjornstrup]{fncychap}",
    "printindex": "\\footnotesize\\raggedright\\printindex",
}

latex_documents = [
    (
        "index",
        "homodyne-analysis.tex",
        "Homodyne Analysis Documentation",
        "Wei Chen, Hongrui He",
        "manual",
    ),
]

# -- Options for manual page output ------------------------------------------
man_pages = [
    ("index", "homodyne-analysis", "Homodyne Analysis Documentation", [author], 1)
]

# -- Options for Texinfo output ----------------------------------------------
texinfo_documents = [
    (
        "index",
        "homodyne-analysis",
        "Homodyne Analysis Documentation",
        author,
        "homodyne-analysis",
        "One line description of project.",
        "Miscellaneous",
    ),
]
