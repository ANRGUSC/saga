# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'SAGA'
copyright = '2024, Jared Coleman, Bhaskar Krishnamachari, Sanjana Adapala, Ravi Vivek Agrawal, Ebrahim Hirani, Saamarth Sethi, Deep Dodhiwala'
author = 'Jared Coleman, Bhaskar Krishnamachari, Sanjana Adapala, Ravi Vivek Agrawal, Ebrahim Hirani, Saamarth Sethi, Deep Dodhiwala'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',  # If you use NumPy or Google style docstrings
    'sphinx.ext.viewcode',  # To include source code links in your documentation
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
