# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# Indiquer ou se trouve le code Python (racine du projet, deux niveaux
# au-dessus de docs/source/conf.py)
import os
import sys
sys.path.insert(0, os.path.abspath("../../"))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'AeroPredict'
copyright = '2026, Flavien Blanchard, Milissa Mechref, Vincent Condette'
author = 'Flavien Blanchard, Milissa Mechref, Vincent Condette'

version = '1.0'
release = '1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
]

templates_path = ['_templates']
exclude_patterns = []

language = 'fr'

# Dependances lourdes non installees dans l'environnement de build :
# autodoc simule leur presence pour pouvoir extraire les docstrings
# sans avoir besoin d'installer aerosandbox / tensorflow.
autodoc_mock_imports = ['aerosandbox', 'tensorflow']

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'alabaster'
html_static_path = ['_static']
