"""
MofBuilder IO Module for reading and writing MOF structures.

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

from .basic import nl, nn

__all__ = [
    "CifReader", "CifWriter", "XyzReader", "XyzWriter", "PdbReader",
    "PdbWriter", "GroReader", "GroWriter", "nn", "nl"
]

_LAZY_ATTRS: Dict[str, Tuple[str, str]] = {
    "CifReader": ("mofbuilder.io.cif_reader", "CifReader"),
    "CifWriter": ("mofbuilder.io.cif_writer", "CifWriter"),
    "XyzReader": ("mofbuilder.io.xyz_reader", "XyzReader"),
    "XyzWriter": ("mofbuilder.io.xyz_writer", "XyzWriter"),
    "PdbReader": ("mofbuilder.io.pdb_reader", "PdbReader"),
    "PdbWriter": ("mofbuilder.io.pdb_writer", "PdbWriter"),
    "GroReader": ("mofbuilder.io.gro_reader", "GroReader"),
    "GroWriter": ("mofbuilder.io.gro_writer", "GroWriter"),
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
