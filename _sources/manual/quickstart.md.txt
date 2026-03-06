# Quickstart

This quickstart demonstrates a minimal end-to-end build workflow:
select a MOF family, provide linker data, build the framework, and export files.

## 1. Import core classes

```python
from veloxchem.molecule import Molecule
from mofbuilder import MetalOrganicFrameworkBuilder
```

## 2. Inspect available topology options

```python
builder = MetalOrganicFrameworkBuilder(mof_family="HKUST-1")
builder.list_available_mof_families()
builder.list_available_metals("HKUST-1")
```

## 3. Configure and build a framework

```python
linker = Molecule.read_smiles("O=C([O-])c1ccc(cc1)C(=O)[O-]")

builder.node_metal = "Cu"
builder.linker_molecule = linker
builder.supercell = (1, 1, 1)

framework = builder.build()
```

At this point, `framework` is a `Framework` object containing graph/cell data
and writer/MD-preparation helpers.

## 4. Export structure files

```python
framework.write(
    format=["cif", "xyz", "gro"],
    filename="output/hkust1"
)
```

## 5. Optional next steps

- Add solvents with `framework.solvate(...)`
- Prepare MD input files with `framework.md_prepare()`
- Run dynamics with `framework.md_driver.run_pipeline(...)`

For complete examples, continue to {doc}`examples`.  
For full public API details, see {doc}`../api_reference`.
