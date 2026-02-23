"""
MofBuilder Utils Module for utility functions and helpers.

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

__all__ = [
    "fetch_pdbfile", "unit_cell_to_cartesian_matrix",
    "fractional_to_cartesian", "cartesian_to_fractional"
]

_LAZY_ATTRS: Dict[str, Tuple[str, str]] = {
    "fetch_pdbfile": ("mofbuilder.utils.fetch", "fetch_pdbfile"),
    "unit_cell_to_cartesian_matrix":
    ("mofbuilder.utils.geometry", "unit_cell_to_cartesian_matrix"),
    "fractional_to_cartesian":
    ("mofbuilder.utils.geometry", "fractional_to_cartesian"),
    "cartesian_to_fractional":
    ("mofbuilder.utils.geometry", "cartesian_to_fractional"),
}


def __getattr__(name: str) -> Any:
    if name in _LAZY_ATTRS:
        module_name, attr_name = _LAZY_ATTRS[name]
        module = importlib.import_module(module_name)
        attr = getattr(module, attr_name)
        globals()[name] = attr
        return attr
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY_ATTRS))
