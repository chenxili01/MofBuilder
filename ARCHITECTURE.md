# MOFBuilder Architecture

## Project Overview

MOFBuilder is a data-driven Python package for constructing MOF structures from
topology templates plus node/linker fragments, then exporting, modifying,
solvating, and preparing them for simulation.

The repo is organized around a stable high-level workflow:

1. Select a MOF family and metal/node/linker inputs
2. Build topology and fragment representations
3. Optimize geometry and cell parameters
4. Expand to a supercell and derive the edge graph
5. Materialize a `Framework`
6. Export, edit defects, solvate, and prepare MD inputs

The package surface is intentionally lighter than the implementation:

- `mofbuilder` top-level imports are lazy
- `cli.py` is dependency-light
- `core` and `md` pull in heavy scientific dependencies once accessed

## Main Workflow / Mental Model

The central user object is `MetalOrganicFrameworkBuilder`. It is a stateful
orchestrator that gathers user inputs, pulls data from the bundled database,
constructs graph/fragment intermediates, and returns a built `Framework`.

`Framework` is the post-build container. It owns the assembled graph and merged
atom tables and provides the user-facing operations after construction:

- file export
- removal/replacement workflows
- visualization
- solvation
- linker force-field generation
- MD preparation

## Core Modules and Responsibilities

### Package Surface

- `src/mofbuilder/__init__.py`
  - lazy top-level imports
- `src/mofbuilder/core/__init__.py`
  - lazy exports for core classes
- `src/mofbuilder/cli.py`
  - dependency-light CLI for `--version`, `data-path`, `list-families`, and
    `list-metals`

### Core Construction

- `src/mofbuilder/core/builder.py`
  - `MetalOrganicFrameworkBuilder`
  - central orchestration layer
- `src/mofbuilder/core/framework.py`
  - `Framework`
  - built-object container plus post-build workflow API
- `src/mofbuilder/core/moftoplibrary.py`
  - `MofTopLibrary`
  - reads `database/MOF_topology_dict` and resolves template CIFs
- `src/mofbuilder/core/net.py`
  - `FrameNet`
  - parses topology CIFs into graph/unit-cell data
- `src/mofbuilder/core/node.py`
  - `FrameNode`
  - prepares node fragments
- `src/mofbuilder/core/linker.py`
  - `FrameLinker`
  - prepares linker fragments from XYZ, molecule objects, or SMILES
- `src/mofbuilder/core/termination.py`
  - `FrameTermination`
  - prepares termination fragments
- `src/mofbuilder/core/optimizer.py`
  - `NetOptimizer`
  - rotates/places fragments and optimizes the cell
- `src/mofbuilder/core/supercell.py`
  - `SupercellBuilder`, `EdgeGraphBuilder`
  - expands periodic structure and derives edge-graph representations
- `src/mofbuilder/core/defects.py`
  - `TerminationDefectGenerator`
  - removal/replacement/termination workflows
- `src/mofbuilder/core/write.py`
  - `MofWriter`
  - merged structure assembly and format export

### I/O and Helpers

- `src/mofbuilder/io/`
  - readers/writers for CIF, PDB, GRO, XYZ
- `src/mofbuilder/utils/geometry.py`
  - unit-cell and coordinate transforms plus geometric matching helpers
- `src/mofbuilder/utils/environment.py`
  - bundled data-path resolution
- `src/mofbuilder/utils/fetch.py`
  - file lookup support

### Post-Build Simulation Pipeline

- `src/mofbuilder/md/solvationbuilder.py`
  - `SolvationBuilder`
- `src/mofbuilder/md/linkerforcefield.py`
  - `LinkerForceFieldGenerator`, `ForceFieldMapper`
- `src/mofbuilder/md/gmxfilemerge.py`
  - `GromacsForcefieldMerger`
- `src/mofbuilder/md/setup.py`
  - `OpenmmSetup`

### Secondary / Less Mature Areas

- `src/mofbuilder/visualization/viewer.py`
  - `Viewer`
- `src/mofbuilder/analysis/`
  - `GraphAnalyzer`, `PorosityAnalyzer`
  - currently placeholder-level, not a mature subsystem

## Important Classes and Their Relationships

### `MetalOrganicFrameworkBuilder`

Owns:

- user inputs such as `mof_family`, `node_metal`, linker source, termination
  settings, supercell settings, defect options, and MD-prep options
- intermediate graph states
- helper instances such as `FrameNet`, `FrameNode`, `FrameLinker`,
  `FrameTermination`, `MofTopLibrary`, `NetOptimizer`, `MofWriter`,
  `TerminationDefectGenerator`

Depends on:

- bundled database assets
- `veloxchem`, `mpi4py`, `networkx`, `numpy`
- core helper classes and I/O readers

Relationship:

- populates and returns the `Framework` instance stored on `builder.framework`

### `Framework`

Owns:

- assembled graph and metadata copied from the builder
- merged Cartesian/fractional atom tables
- output settings and post-build workflow helpers

Depends on:

- `MofWriter`
- `TerminationDefectGenerator`
- `SolvationBuilder`
- linker force-field and OpenMM setup classes

Relationship:

- is the stable post-build interface for most user workflows
- `remove()` and `replace()` return new `Framework` instances
- `solvate()`, `generate_linker_forcefield()`, `md_prepare()`, and `show()`
  mutate the current instance

### `MofTopLibrary`

Owns:

- MOF-family metadata read from `MOF_topology_dict`
- template selection logic

### `FrameNet`

Owns:

- topology graph construction from template CIF
- cell information
- vertex/edge pairing and sorted topology metadata

### `FrameNode`, `FrameLinker`, `FrameTermination`

Own:

- fragment parsing and normalization for later placement

### `NetOptimizer`

Owns:

- rotation placement and optimized cell state

### `SupercellBuilder`, `EdgeGraphBuilder`

Own:

- supercell expansion
- edge-graph derivation
- matched node/edge bookkeeping used downstream

## Main Data Flow Through the Package

### 1. Database lookup

`MofTopLibrary` reads:

- `database/MOF_topology_dict`
- `database/template_database/`

It can also load optional additive family role metadata from:

- `database/MOF_topology_role_metadata.json`

This resolves node connectivity, linker topic, available metals, and the
template CIF file for the chosen MOF family.

`MOF_topology_dict` is treated as a tabular schema with columns for MOF family,
node connectivity, metal, linker topic, and topology stem.

### 2. Net parsing

`FrameNet` parses the topology CIF and derives:

- graph `G`
- unit-cell data
- sorted topology metadata
- linker connectivity information
- deterministic graph role annotations:
  - `G.nodes[n]["node_role_id"]`
  - `G.edges[e]["edge_role_id"]`

### 3. Fragment preparation

The builder prepares:

- node data via `FrameNode`
- linker data via `FrameLinker`
- optional termination data via `FrameTermination`
- builder-owned runtime role registries:
  - `node_role_registry`
  - `edge_role_registry`

For families without role metadata, the builder normalizes to the single-role
base case:

- `node:default`
- `edge:default`

### 4. Optimization

`NetOptimizer` aligns fragments onto the topology and optimizes:

- node rotations
- cell parameters

Role-aware fragment selection is driven by graph-stored role ids plus the
builder-owned registries; the single-role path remains the default fast path.

This yields the optimized graph `sG` and frame-unit-cell information.

### 5. Supercell and edge-graph generation

`SupercellBuilder` expands the system into `superG`.

`EdgeGraphBuilder` derives:

- `eG`
- `cleaved_eG`
- matching dictionaries and residue-related metadata used downstream

Role metadata is propagated through `superG`, `eG`, and `cleaved_eG`.

### 6. Framework materialization

`MetalOrganicFrameworkBuilder.build()` copies the resulting state into
`Framework`, applies termination logic, and materializes merged structure data
through `Framework.get_merged_data()`.

### 7. Post-build processing

`Framework` can then:

- write output files
- remove or replace substructures
- solvate the framework
- generate linker force-field data
- prepare MD inputs
- launch the OpenMM pipeline through `md_driver`

The built `Framework` retains the role-aware data needed by later post-build
steps, including the edge-role registry used by the current MD-preparation
path.

## Canonical Role Model

MOFBuilder now uses one internal role model across the pipeline:

- topology role ids are stored on graphs, not inferred from chemistry
- `FrameNet.G.nodes[n]["node_role_id"]` is the node-role source of truth
- `FrameNet.G.edges[e]["edge_role_id"]` is the edge-role source of truth
- `MetalOrganicFrameworkBuilder` owns `node_role_registry` and
  `edge_role_registry`
- families without role metadata normalize to `node:default` and
  `edge:default`

This role model is internal plumbing around the stable public workflow, not a
separate public orchestration path.

Specific behaviors that matter for modifications:

- `Framework.solvate()` defaults to TIP3P when no solvent list is provided
- `Framework.solvate()` writes `<mof_family>_in_solvent.gro`
- `Framework.md_prepare()` generates a topology with
  `GromacsForcefieldMerger`, writes a framework-only GRO if the system was not
  solvated, and then creates `OpenmmSetup`
- `Framework.generate_linker_forcefield()` can bypass generation when
  `provided_linker_itpfile` is set

## Important Design Constraints

- Public builder/framework APIs should remain stable
- Top-level imports and CLI behavior are intentionally dependency-light
- Graph state names already have established meaning:
  - `G`
  - `sG`
  - `superG`
  - `eG`
  - `cleaved_eG`
- `Framework.framework_data` and `Framework.framework_fcoords_data` are central
  downstream payloads; changes here affect writing, solvation, and MD prep
- Structural changes must keep merged data and residue metadata synchronized;
  current code does this by calling `get_merged_data()` after build/remove/
  replace flows
- Runtime behavior depends heavily on bundled data in `database/`
- Similar fixtures exist in `tests/database/`; data-format changes should keep
  them aligned

## Likely Extension Points

- new topology families or template assets via `MofTopLibrary`
- new fragment sources or preprocessing logic in `FrameNode`/`FrameLinker`
- geometry algorithms in `NetOptimizer` and `utils/geometry.py`
- new export behavior in `MofWriter` or `io/`
- expanded simulation-prep behavior in `md/`
- richer analysis implementations in `analysis/`
  - currently this area is mostly scaffolding

## Known Ambiguity / Areas Needing Caution

- `src/mofbuilder/analysis/` contains stub methods with `pass`
- Some documentation content is duplicated between `docs/` and
  `docs/source/manual/`
- `Framework.write()` references `remove_defects()`, but the visible public
  defect-removal method on `Framework` is `remove()`
  - inspect actual usage/tests before changing this area
- The repository root contains notebooks, outputs, and sample structure files;
  they appear exploratory or generated rather than architectural source files
- `optimizer.py`, `supercell.py`, `framework.py`, and `linkerforcefield.py`
  are the highest-risk modules for unintended regressions
- `tests/conftest.py` stubs heavy dependencies for testability
  - useful for interface coverage
  - not a substitute for validating real numerical/simulation behavior

## Refactoring Safely in This Repo

- Route each change to the narrowest owning module
- Prefer local edits over broad cleanup passes
- Preserve class names, public method names, and established attribute names
- Keep builder orchestration in `MetalOrganicFrameworkBuilder`
- Keep post-build behavior in `Framework`
- Preserve current mutation semantics
  - `build()` reuses `builder.framework`
  - `remove()` and `replace()` allocate new framework objects
- Preserve numerical behavior in geometry/optimization code unless explicitly
  improving it
- Re-check downstream consumers before changing graph attributes, coordinate
  conventions, or merged-data shape
- Avoid new dependencies unless there is a clear technical need
- Update tests and docs alongside behavior changes
- If scientific intent is unclear, inspect call sites, tests, comments, and
  docstrings before changing behavior

## Scope of Multi-Role Support

This feature introduces role-aware topology and fragment assignment.

It does not redesign:
- the builder workflow
- the graph-state architecture
- the optimization algorithms
- defect modeling

Changes should remain localized to role-awareness and fragment selection.
