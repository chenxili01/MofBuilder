"""Sphinx configuration for the MOFBuilder manual."""

from __future__ import annotations

import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.abspath("../../src"))

project = "MOFBuilder"
author = "Chenxi Li"
copyright = f"{datetime.now().year}, {author}"
release = "0.1.0"
version = release

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "generated/*"]

autosummary_generate = True
autosummary_generate_overwrite = True
autosummary_ignore_module_all = False
autosummary_imported_members = False

autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
    "member-order": "bysource",
    "imported-members": False,
    "exclude-members": "__dict__,__weakref__,__module__,__getattr__,__dir__",
}
autodoc_typehints = "description"

autodoc_mock_imports = [
    "veloxchem",
    "mpi4py",
    "openmm",
    "rdkit",
    "xtb",
    "py3Dmol",
]
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_param = True
napoleon_use_rtype = True

myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "fieldlist",
]
myst_heading_anchors = 3

try:
    import pydata_sphinx_theme  # noqa: F401
except ImportError:
    html_theme = "alabaster"
    html_theme_options = {}
else:
    html_theme = "pydata_sphinx_theme"
    html_theme_options = {
        "collapse_navigation": True,
        "show_nav_level": 2,
        "navigation_depth": 4,
        "show_prev_next": True,
        "navbar_persistent": ["search-button"],
        "navbar_end": ["theme-switcher", "navbar-icon-links"],
        "icon_links": [
            {
                "name": "GitHub",
                "url": "https://github.com/chenxili01/MofBuilder",
                "icon": "fa-brands fa-github",
                "type": "fontawesome",
            }
        ],
        "secondary_sidebar_items": ["page-toc"],
    }

html_context = {
    "github_user": "chenxili01",
    "github_repo": "MofBuilder",
    "github_version": "main",
    "doc_path": "docs/source",
}


html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_title = f"{project} {version} documentation"
html_last_updated_fmt = "%Y-%m-%d"

pygments_style = "sphinx"
pygments_dark_style = "monokai"
highlight_language = "python"
