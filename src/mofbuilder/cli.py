"""Command-line interface for MOFbuilder.

The CLI is intentionally dependency-light so users can query package metadata
and bundled database information without installing heavy optional stacks.
"""

from __future__ import annotations

import argparse
import importlib.metadata as importlib_metadata
from pathlib import Path
from typing import Dict, List, Optional


def _resolve_version() -> str:
    """Return package version, falling back to local source version."""
    try:
        return importlib_metadata.version("mofbuilder")
    except importlib_metadata.PackageNotFoundError:
        from . import __version__

        return __version__


def _get_data_path() -> Path:
    """Resolve bundled database path from local package layout."""
    return Path(__file__).resolve().parents[2] / "database"


def _read_topology_dict(topology_file: Path) -> Dict[str, List[str]]:
    """Parse MOF topology metadata into a family -> metals mapping."""
    if not topology_file.exists():
        raise FileNotFoundError(
            f"Could not find topology dictionary: {topology_file}")

    family_to_metals: Dict[str, List[str]] = {}
    with topology_file.open("r", encoding="utf-8") as handle:
        for raw_line in handle.readlines()[1:]:
            fields = raw_line.split()
            if len(fields) < 5:
                continue
            family, metal = fields[0].upper(), fields[2]
            family_to_metals.setdefault(family, [])
            if metal not in family_to_metals[family]:
                family_to_metals[family].append(metal)

    return family_to_metals


def _print_families() -> int:
    data_path = _get_data_path()
    topology = _read_topology_dict(data_path / "MOF_topology_dict")
    if not topology:
        print("No MOF families found.")
        return 1
    for family in sorted(topology):
        print(family)
    return 0


def _print_metals(mof_family: str) -> int:
    data_path = _get_data_path()
    topology = _read_topology_dict(data_path / "MOF_topology_dict")
    family = mof_family.upper()
    if family not in topology:
        print(f"Unknown MOF family: {mof_family}")
        return 2
    for metal in topology[family]:
        print(metal)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mofbuilder",
        description="MOFbuilder command-line interface",
    )
    parser.add_argument(
        "--version",
        action="store_true",
        help="Print MOFbuilder version and exit.",
    )

    subparsers = parser.add_subparsers(dest="command")
    subparsers.add_parser("data-path", help="Print bundled data directory path.")
    subparsers.add_parser("list-families",
                          help="List MOF families in MOF_topology_dict.")

    list_metals = subparsers.add_parser(
        "list-metals",
        help="List available metals for one MOF family.",
    )
    list_metals.add_argument(
        "--mof-family",
        required=True,
        help="MOF family name, e.g. UIO-66.",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.version:
        print(_resolve_version())
        return 0

    if args.command == "data-path":
        print(_get_data_path())
        return 0
    if args.command == "list-families":
        return _print_families()
    if args.command == "list-metals":
        return _print_metals(args.mof_family)

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
