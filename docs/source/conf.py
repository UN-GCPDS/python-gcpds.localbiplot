# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

sys.path.insert(0, os.path.abspath('../../gcpds'))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'localbiplot'
copyright = '2024, Jenniffer Carolina Triana Martinez'
author = 'Jenniffer Carolina Triana Martinez'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'nbsphinx',
    'dunderlab.docs',
]

templates_path = ['_templates']
exclude_patterns = ['_build', '**.ipynb_checkpoints']


html_logo = '_static/localbip_logo.png'
html_favicon = '_static/favicon.ico'


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'alabaster'
html_static_path = ['_static']
autodoc_mock_imports = ["matplotlib", 'numpy', 'pandas', 'seaborn', 'sklearn', 'scipy']
dunderlab_github_repository = "https://github.com/UN-GCPDS/python-gcpds.localbiplot"
