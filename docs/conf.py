# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
from unittest import mock
sys.path.insert(0, os.path.abspath('../'))

import asilib


# -- Project information -----------------------------------------------------

project = 'asilib'
copyright = '2024, Mykhaylo Shumko'
author = 'Mykhaylo Shumko'
version = str(asilib.__version__)


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.autosummary',
    'sphinx.ext.linkcode',
    'sphinx_copybutton',
    'nbsphinx',
    "sphinx_design"
]

copybutton_prompt_text = ">>> "
copybutton_copy_empty_lines = True

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', '../asilib/tests']


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "furo" # Tried these themes: sphinx_rtd_theme, pydata_sphinx_theme, furo

html_logo = './_static/asilib_logo.png'

html_theme_options = {
    # Toc options
    "sidebar_hide_name": True
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

napoleon_google_docstring = False
napoleon_use_ivar = True
napoleon_use_admonition_for_examples = True

autodoc_typehints = "none"

autodoc_default_options = {
    'member-order': 'bysource'
}

def linkcode_resolve(domain, info):
    if domain != 'py':
        return None
    if not info['module']:
        return None
    filename = info['module'].replace('.', '/')
    return "https://github.com/mshumko/asilib/tree/main/%s.py" % filename