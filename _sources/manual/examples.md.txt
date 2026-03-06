# Examples

This page collects practical examples that map directly to the current
`mofbuilder` API.

```{note}
Examples that use `veloxchem`, `openmm`, or `rdkit` require those optional
dependencies to be installed in your environment.
```

## Example 1: Build and export a MOF structure

This example builds an HKUST-1 framework from a linker molecule and writes
standard structure files.

```python
from veloxchem.molecule import Molecule
from mofbuilder import MetalOrganicFrameworkBuilder

# 1) Build a linker molecule (can also come from XYZ or SDF workflows)
linker = Molecule.read_smiles("O=C([O-])c1ccc(cc1)C(=O)[O-]")

# 2) Configure the builder
builder = MetalOrganicFrameworkBuilder(mof_family="HKUST-1")
builder.node_metal = "Cu"
builder.linker_molecule = linker
builder.supercell = (1, 1, 1)

# 3) Build and export
framework = builder.build()
framework.write(format=["cif", "xyz", "gro"], filename="output/hkust1")
```

Expected outcome:

- `output/hkust1.cif`
- `output/hkust1.xyz`
- `output/hkust1.gro`

## Example 2: Solvate and prepare an MD workflow

After building a framework, you can create a solvated system and stage OpenMM
inputs through MofBuilder's MD pipeline.

```python
from pathlib import Path

# Use a bundled solvent model (TIP3P)
tip3p_xyz = Path("database/solvents_database/TIP3P.xyz")

framework.target_directory = "output"
framework.solvate(
    solvents_files=[str(tip3p_xyz)],
    solvents_proportions=[1.0],
    solvents_quantities=[600],
    padding_angstrom=12,
)

framework.md_prepare()
framework.md_driver.run_pipeline(
    steps=["em", "nvt"],
    nvt_time=20,
    record_interval=1,
    output_prefix="output/hkust1_md",
    whole_traj=True,
)
```

What this does:

- Packs solvent molecules around the framework.
- Generates merged topology/force-field files for MD.
- Runs energy minimization (`em`) and NVT equilibration (`nvt`).

## Example 3: Defective MOF-5 workflow (from `examples/fig2_defective.ipynb`)

This workflow mirrors the notebook pattern used to prepare Figure 2-style
defect scenarios: build baseline structures, then apply targeted replacement
and removal operations.

### 3.1 Build uncapped and capped MOF-5 supercells

```python
from mofbuilder import MetalOrganicFrameworkBuilder as MofBuilder
import veloxchem as vlx

smi_str = {"bdc": "O=C([O-])C(C=C1)=CC=C1C([O-])=O"}
bdc = vlx.Molecule.read_smiles(smi_str["bdc"])

# Uncapped MOF-5 (2x2x2)
mof = MofBuilder(mof_family="MOF-5")
mof.ostream.mute()
mof.linker_molecule = bdc
mof.node_metal = "Zn"
mof.termination = False
mof.supercell = (2, 2, 2)
nocap = mof.build()

# Capped MOF-5 (2x2x2)
mof = MofBuilder(mof_family="MOF-5")
mof.ostream.mute()
mof.linker_molecule = bdc
mof.node_metal = "Zn"
mof.supercell = (2, 2, 2)
cap = mof.build()
```

### 3.2 Build a slab-like capped model

```python
mof = MofBuilder(mof_family="MOF-5")
mof.ostream.mute()
mof.linker_molecule = bdc
mof.node_metal = "Zn"
mof.supercell = (2, 2, 2)
mof.supercell_custom_fbox = ([0.2, 0.5], [0, 2], [0, 2])
slabcap = mof.build()
```

### 3.3 Replace selected linkers and remove selected components

```python
# Replace selected linker positions with a functionalized linker
new_linker = vlx.Molecule.read_smiles("O=C([O-])C(C=C1N)=C(N)C=C1C([O-])=O")
slabcap.ostream.mute()
rp_slab = slabcap.replace(
    replace_indices=[824, 816, 334, 804, 796, 344],
    new_linker_molecule=new_linker,
)

# Remove selected components to create defect variants
nocap.update_node_termination = True
nocap.ostream.mute()
rm_a = nocap.remove(
    remove_indices=[333, 223, 231, 117, 7, 79, 189, 1, 73, 183, 295, 111, 405, 217, 289, 327, 399]
)
rm_b = nocap.remove(remove_indices=[50, 370, 63, 368])
```

You can inspect any generated framework interactively with `show()`:

```python
rp_slab.show()
rm_a.show(residue_indices=False)
rm_b.show(residue_indices=False)
```

## Example 4: Quick CLI metadata queries

The CLI is intentionally lightweight for querying packaged topology metadata.

```bash
mofbuilder --version
mofbuilder data-path
mofbuilder list-families
mofbuilder list-metals --mof-family HKUST-1
```

This is useful for validating installation and discovering available MOF family
names before running a full build.
