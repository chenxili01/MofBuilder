# MofBuilder

[![License: LGPL v3](https://img.shields.io/badge/License-LGPL%20v3-blue.svg)](https://www.gnu.org/licenses/lgpl-3.0)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyPI version](https://badge.fury.io/py/mofbuilder.svg)](https://badge.fury.io/py/mofbuilder)

A Python library for **building and modelling Metal–Organic Framework (MOF) structures**.

---

## Overview

**MofBuilder** provides a modular and extensible framework for constructing and studying MOF systems, with integrated workflows for structure generation, analysis, and atomistic modelling.

The package is designed to interface with modern computational chemistry tools such as **VeloxChem**, **OpenMM**, **xTB**, and **RDKit**, enabling seamless workflows from **structure construction to molecular simulation**.

Core modules include:

- **Builder**  
  Construct MOF structures from nodes, linkers, and topologies.

- **Analysis**  
  Tools for structural analysis, geometric descriptors, and molecular properties.

- **Modelling**  
  Interfaces for molecular simulations and quantum chemistry calculations.

---

## Features

- Automated **MOF structure construction**
- Integration with **VeloxChem** for quantum chemistry workflows
- **OpenMM interface** for molecular dynamics
- Modular architecture designed for **high-throughput workflows**

---

## Installation

MofBuilder depends on several scientific packages distributed via **conda**, including VeloxChem, RDKit, OpenMM, and xtb-python.

### 1. Create environment

```bash
conda create -n mofbuilder python=3.10
conda activate mofbuilder
```

### 2. Install core scientific dependencies

```bash
conda install -c veloxchem -c conda-forge veloxchem ipykernel rdkit openmm xtb-python py3dmol
```

### 3. Optional: ML potentials for OpenMM

```bash
conda install -c conda-forge openmm-ml
```

### 4. Install MofBuilder from source

```bash
git clone https://github.com/chenxili01/MofBuilder.git
cd MofBuilder
pip install -e .
```

---

## Quick Start

Example workflow:

```python
from mofbuilder import MetalOrganicFrameworkBuilder as MofBuilder
# Build UiO-66 MOF
mof = MofBuilder(mof_family="UiO-66")
mof.linker_smiles = "O=C([O-])C(C=C1)=CC=C1C([O-])=O"
mof.node_metal = "Zr"
uio = mof.build()

#output
uio.write(format=["gro"], filename="uio66_original")

# Solvate and run MD
uio.solvate(padding_angstrom=20)
uio.md_prepare()
uio.md_driver.run_pipeline(steps=['em', 'nvt', 'npt'],
nvt_time=100, npt_time=100,
output_prefix="UiO66_MD")

```
---


## Documentation

Full documentation is available at:

https://mofbuilder.readthedocs.io

---

## License

This project is licensed under the **GNU Lesser General Public License v3.0 or later (LGPL-3.0-or-later)**.

See the [LICENSE](LICENSE) file for details.

---

## Citation

If you use **MofBuilder** in your research, please cite:

```bibtex
```

---

## Support

- **Issues:** https://github.com/chenxili01/MofBuilder/issues  
- **Discussions:** https://github.com/chenxili01/MofBuilder/discussions  
- **Documentation:** https://mofbuilder.readthedocs.io
