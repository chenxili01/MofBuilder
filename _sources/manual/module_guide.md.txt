# Module Guide

This guide summarizes the main public modules in MofBuilder based on the
current codebase.

The analysis module is currently under development and is not yet part of the public interface.

## Builder (`mofbuilder.core`)

Primary entry points:

- `MetalOrganicFrameworkBuilder`
- `Framework`

Typical workflow:

1. Create `MetalOrganicFrameworkBuilder(mof_family=...)`
2. Set required inputs such as `node_metal` and linker source
3. Run `build()` to obtain a `Framework` object
4. Export files with `Framework.write(...)`

What it handles internally:

- Topology/template lookup (`MofTopLibrary`)
- Node/linker placement
- Rotation and cell optimization (`NetOptimizer`)
- Supercell and edge-graph generation (`SupercellBuilder`, `EdgeGraphBuilder`)
- Termination/defect handling (`TerminationDefectGenerator`)

## Modelling and Simulation (`mofbuilder.md`)

Primary entry points:

- `SolvationBuilder`
- `LinkerForceFieldGenerator`
- `GromacsForcefieldMerger`
- `OpenmmSetup`

Typical usage pattern:

1. Build framework with `MetalOrganicFrameworkBuilder`
2. Call `Framework.solvate(...)` to create a solvated system
3. Call `Framework.md_prepare()` to generate MD inputs
4. Run `Framework.md_driver.run_pipeline(...)` for EM/NVT/NPT stages

## I/O (`mofbuilder.io`)

Reader/writer classes provide explicit structure-file control:

- Readers: `CifReader`, `PdbReader`, `GroReader`, `XyzReader`
- Writers: `CifWriter`, `PdbWriter`, `GroWriter`, `XyzWriter`

These are useful when integrating MofBuilder into external workflows that
already provide structure files and only need conversion or export.

## Visualization (`mofbuilder.visualization`)

- `Viewer` is used by `Framework.show(...)` for quick inspection.

This is convenient for exploratory checks during build iterations before full
simulation setup.
