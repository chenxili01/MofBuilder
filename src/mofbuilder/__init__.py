"""
MofBuilder: A Python library for building and analyzing Metal-Organic Frameworks (MOFs)

Copyright (C) 2024 MofBuilder Contributors

This library is free software; you can redistribute it and/or
modify it under the terms of the GNU Lesser General Public
License as published by the Free Software Foundation; either
version 3 of the License, or (at your option) any later version.

This library is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public
License along with this library; if not, write to the Free Software
Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
"""

from __future__ import annotations

import importlib
from typing import Any, Dict, Tuple

__version__ = "0.1.0"
__author__ = "MofBuilder Contributors"
__email__ = "chenxili@kth.se"
__license__ = "LGPL-3.0-or-later"

__all__ = [
    "MetalOrganicFrameworkBuilder", "core", "io", "utils", "analysis",
    "visualization", "md", "__version__"
]

_LAZY_MODULES = {"analysis", "core", "io", "utils", "visualization", "md"}
_LAZY_ATTRS: Dict[str, Tuple[str, str]] = {
    "MetalOrganicFrameworkBuilder":
    ("mofbuilder.core.builder", "MetalOrganicFrameworkBuilder"),
}


def __getattr__(name: str) -> Any:
    if name in _LAZY_MODULES:
        module = importlib.import_module(f".{name}", __name__)
        globals()[name] = module
        return module
    if name in _LAZY_ATTRS:
        module_name, attr_name = _LAZY_ATTRS[name]
        module = importlib.import_module(module_name)
        attr = getattr(module, attr_name)
        globals()[name] = attr
        return attr
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


def __dir__() -> list[str]:
    return sorted(set(globals()) | _LAZY_MODULES | set(_LAZY_ATTRS))
