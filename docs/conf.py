from pathlib import Path
import sys

# -- Path setup --------------------------------------------------------------

# Add project root to sys.path so Sphinx can import the package
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# -- Project information -----------------------------------------------------

project = "Cassini-UPyP"
author = "Aodren"
release = "0.1.0"

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "myst_parser"
]

autosummary_generate = True

napoleon_google_docstring = False
napoleon_numpy_docstring = True

templates_path = ["_templates"]
exclude_patterns = ["_build"]

# -- HTML output -------------------------------------------------------------

html_theme = "pydata_sphinx_theme"
html_title = "Cassini-UPyP documentation"

html_theme_options = {
    "navigation_depth": 4,
    "show_nav_level": 2,
    "show_prev_next": True,
}
autodoc_typehints = "signature"
autodoc_typehints_format = "short"
autodoc_preserve_defaults = False
napoleon_use_ivar = True
autoclass_content = "class"