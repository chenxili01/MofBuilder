# MOFBuilder

A Python toolkit for topology-driven construction and preparation of
metal-organic framework (MOF) structures.

[![Docs](https://img.shields.io/badge/docs-online-blue)](https://chenxili01.github.io/MOFBuilder/)
[![License: LGPL v3](https://img.shields.io/badge/License-LGPL%20v3-blue.svg)](https://www.gnu.org/licenses/lgpl-3.0)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyPI version](https://badge.fury.io/py/mofbuilder.svg)](https://badge.fury.io/py/mofbuilder)

## Overview

MOFBuilder follows a stable staged workflow:

1. select a MOF family and topology template
2. provide node and linker inputs
3. build the topology graph
4. optimize node placement and the cell
5. expand to a supercell and edge graph
6. materialize a `Framework` for export, editing, solvation, and MD setup

The public build path remains the single-role base case: one node fragment
selection and one linker fragment selection reused across the topology.

Internally, the code also carries deterministic topology role annotations and
role registries:

- topology role ids live on the graph as `node_role_id` and `edge_role_id`
- builder-owned registries live as `node_role_registry` and
  `edge_role_registry`
- families without role metadata normalize to `node:default` and
  `edge:default`

This internal model is used by the builder, optimizer, supercell, writer,
defect, and MD-preparation layers without changing the stable public workflow.

## Features

- topology-driven MOF construction from bundled template data
- export through the `Framework.write(...)` workflow
- defect editing through `Framework.remove(...)` and `Framework.replace(...)`
- solvation and MD preparation through `Framework.solvate(...)` and
  `Framework.md_prepare(...)`
- lightweight CLI commands for version and topology-library inspection

## Installation

MOFBuilder depends on several scientific packages distributed via `conda`,
including VeloxChem, RDKit, OpenMM, and `xtb-python`.

```bash
conda create -n mofbuilder python=3.10
conda activate mofbuilder
conda install -c veloxchem -c conda-forge \
  veloxchem ipykernel rdkit openmm xtb-python py3dmol
conda install -c conda-forge openmm-ml  # optional
git clone https://github.com/chenxili01/MofBuilder.git
cd MofBuilder
pip install -e .
```

## Quick Start

```python
from veloxchem.molecule import Molecule
from mofbuilder import MetalOrganicFrameworkBuilder

builder = MetalOrganicFrameworkBuilder(mof_family="HKUST-1")
builder.node_metal = "Cu"
builder.linker_molecule = Molecule.read_smiles(
    "O=C([O-])c1ccc(cc1)C(=O)[O-]"
)

framework = builder.build()
framework.write(format=["cif", "xyz", "gro"], filename="output/hkust1")
```

For the default single-role path above, MOFBuilder still normalizes the
internal runtime model to one-entry registries:

- `builder.node_role_registry["node:default"]`
- `builder.edge_role_registry["edge:default"]`

## CLI

The dependency-light CLI is useful for installation checks and topology
inspection:

```bash
mofbuilder --version
mofbuilder data-path
mofbuilder list-families
mofbuilder list-metals --mof-family UIO-66
```

## Documentation

Full documentation is available at
[chenxili01.github.io/MOFBuilder](https://chenxili01.github.io/MOFBuilder/).

## License

This project is licensed under the
`GNU Lesser General Public License v3.0 or later (LGPL-3.0-or-later)`.

## Links

- [Documentation](https://chenxili01.github.io/MOFBuilder/)
- [Source code](https://github.com/chenxili01/MOFBuilder)
- [Issues](https://github.com/chenxili01/MOFBuilder/issues)
- [Nodes/Net library for MOFBuilder](https://github.com/chenxili01/MOFBuilder_library)
