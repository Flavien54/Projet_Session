# Configuration file for the Sphinx documentation builder.
import os
import sys

sys.path.insert(0, os.path.abspath('../../'))
sys.path.insert(0, os.path.abspath('../tests'))
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'AeroPredict'
copyright = '2026, Melissa Flavien Vincent'
author = 'Melissa Flavien Vincent'
release = '2026'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',     
    'sphinx.ext.napoleon',   
    'sphinx.ext.viewcode'
]

templates_path = ['_templates']
exclude_patterns = []

language = 'fr'

autodoc_mock_imports = [
    'streamlit',
    'pandas',
    'numpy',
    'matplotlib',
    'seaborn',
    'plotly',
    'sklearn',
    'aerosandbox',
    'tensorflow',
    'keras',
    'scipy'
]
# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'alabaster'
html_static_path = ['_static']
