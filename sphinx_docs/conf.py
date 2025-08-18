# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
sys.path.insert(0, os.path.abspath('..'))

# -- Project information -----------------------------------------------------
project = 'Homodyne Analysis'
copyright = '2025, Wei Chen, Hongrui He'
author = 'Wei Chen, Hongrui He'
release = '6.0'
version = '6.0'

# -- General configuration ---------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
    'sphinx.ext.todo',
    'sphinx.ext.coverage',
    'sphinx.ext.autosummary',  # For better API documentation
    'myst_parser',
]

# Suppress specific warnings to reduce noise
suppress_warnings = [
    'misc.highlighting_failure',  # Suppress JSON highlighting warnings
    'autosummary',  # Suppress autosummary warnings
]

# Performance optimizations
autodoc_mock_imports = ['numba', 'pymc', 'arviz', 'pytensor']  # Mock heavy dependencies
autodoc_preserve_defaults = True  # Preserve default values in signatures

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
exclude_patterns = [
    '_build', 
    'Thumbs.db', 
    '.DS_Store',
    'GITHUB_ACTIONS_FIXES.md',
    '*.md',  # Exclude all markdown files except those explicitly included
]

# The default language to highlight source code in.
highlight_language = 'python3'

# -- Options for extensions --------------------------------------------------

# autodoc configuration
autodoc_typehints = 'description'
autodoc_member_order = 'bysource'
autodoc_default_options = {
    'members': True,
    'undoc-members': True,
    'show-inheritance': True,
    'special-members': '__init__',
    'exclude-members': '__weakref__',
}

# Optimize autodoc performance
autodoc_class_signature = 'mixed'  # Show both class and __init__ signatures
autodoc_inherit_docstrings = True  # Inherit docstrings from parent classes
autodoc_typehints_format = 'short'  # Use short form for type hints

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
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/', None),
    'matplotlib': ('https://matplotlib.org/stable/', None),
}

# todo extension
todo_include_todos = True

# MyST parser configuration
myst_enable_extensions = [
    'colon_fence',
    'deflist',
    'html_admonition',
    'html_image',
    'linkify',
    'replacements',
    'smartquotes',
    'substitution',
    'tasklist',
]

# Configure MyST parser for better performance
myst_heading_anchors = 2  # Generate anchors for h1 and h2 headings
myst_footnote_transition = True  # Add transition before footnotes
myst_dmath_double_inline = True  # Support $$ for inline math

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
html_theme = 'sphinx_rtd_theme'
html_theme_options = {
    'analytics_id': '',
    'logo_only': False,
    'prev_next_buttons_location': 'bottom',
    'style_external_links': False,
    'vcs_pageview_mode': '',
    'style_nav_header_background': '#2980b9',
    'collapse_navigation': True,
    'sticky_navigation': True,
    'navigation_depth': 3,  # Reduced depth for performance
    'includehidden': True,
    'titles_only': False
}

# Optimize HTML output
html_copy_source = False  # Don't copy source files to save space
html_show_sourcelink = False  # Don't show source links
html_compact_lists = True  # Use compact lists
html_secnumber_suffix = '. '  # Add period after section numbers

html_static_path = ['_static']
html_css_files = [
    'custom.css',
]

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'sphinx'

# Custom sidebar templates
html_sidebars = {
    '**': [
        'about.html',
        'navigation.html',
        'relations.html',
        'searchbox.html',
        'donate.html',
    ]
}

# -- Options for LaTeX output ------------------------------------------------
latex_elements = {
    'papersize': 'letterpaper',
    'pointsize': '10pt',
    'preamble': '',
    'fncychap': '\\usepackage[Bjornstrup]{fncychap}',
    'printindex': '\\footnotesize\\raggedright\\printindex',
}

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title,
#  author, documentclass [howto, manual, or own class]).
latex_documents = [
    ('index', 'homodyne-analysis.tex', 'Homodyne Analysis Documentation',
     'Wei Chen, Hongrui He', 'manual'),
]

# -- Options for manual page output ------------------------------------------
# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [
    ('index', 'homodyne-analysis', 'Homodyne Analysis Documentation',
     [author], 1)
]

# -- Options for Texinfo output ----------------------------------------------
# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
texinfo_documents = [
    ('index', 'homodyne-analysis', 'Homodyne Analysis Documentation',
     author, 'homodyne-analysis', 'One line description of project.',
     'Miscellaneous'),
]
