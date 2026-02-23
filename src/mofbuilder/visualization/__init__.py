"""
MofBuilder Visualization Module for MOF structure visualization.

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
from typing import Any

__all__ = ["Viewer"]


def __getattr__(name: str) -> Any:
    if name == "Viewer":
        module = importlib.import_module("mofbuilder.visualization.viewer")
        viewer = getattr(module, "Viewer")
        globals()["Viewer"] = viewer
        return viewer
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


def __dir__() -> list[str]:
    return sorted(set(globals()) | {"Viewer"})
