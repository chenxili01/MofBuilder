# MOFBuilder Documentation

MOFBuilder is a Python toolkit for constructing, exporting, and preparing
Metal-Organic Framework (MOF) structures for simulation workflows.

![MOFBuilder workflow](source/_static/images/mofbuilder_workflow.svg)

## What this documentation covers

- Topology-guided framework construction from linkers and node selections
- Practical export workflows (`cif`, `pdb`, `gro`, `xyz`)
- Solvation and MD setup pipeline
- Public API references focused on user-facing classes/functions

## Canonical documentation location

The canonical user manual lives under `docs/source/manual/`.
Root-level pages in `docs/` should route to those manual pages rather than
carry independent copies. `docs/quickstart.md` and `docs/examples.md` are now
include stubs that pull from the manual pages below to avoid future drift.

## Start Here

```{toctree}
:maxdepth: 2
:caption: Canonical Manual

source/manual/installation
source/manual/quickstart
source/manual/examples
api
```

## Core project links

- Repository: <https://github.com/chenxili01/MofBuilder>
- Issues: <https://github.com/chenxili01/MofBuilder/issues>
- Read the Docs: <https://mofbuilder.readthedocs.io>

```{note}
Update `docs/source/manual/*` when changing user-facing manual content. The
root `docs/` pages should remain routing or include stubs so there is only one
manual source of truth.
```
