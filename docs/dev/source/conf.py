project = 'FunSearch CVRP'
copyright = '2025, FunSearch CVRP Team'
author = 'FunSearch CVRP Team'
release = '0.1.0'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    'sphinx.ext.intersphinx',
]

templates_path = ['_templates']
exclude_patterns = []

html_theme = 'alabaster'
html_static_path = ['_static']

language = 'en'

locale_dirs = ['../locales/']
gettext_compact = False

intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
}
