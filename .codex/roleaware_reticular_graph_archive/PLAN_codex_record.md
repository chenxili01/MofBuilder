# PLANS.md

## Status

Frozen for implementation starting from Phase 1.

Do not revise this plan for style or completeness during implementation.
Only update it when a real schema/runtime/invariant conflict is discovered.

## Goal

Generalize MOFBuilder from templates that assume one reusable node role and one
reusable edge/linker role to templates that can carry multiple node roles and
multiple edge roles, while preserving the existing pipeline and keeping the
single-role case as the default/base case.

In this document, "single-role" means:

- one node fragment definition reused across all topology vertices
- one linker/edge fragment definition reused across all topology edges

It does **not** mean the topology has only one node or one edge instance.

## Public Workflow and Internal Integration Points That Must Stay Intact

### Stable public workflow

Keep this user-facing workflow and object model recognizable:

1. `MofTopLibrary.fetch(...)`
2. `FrameNet.create_net(...)`
3. `MetalOrganicFrameworkBuilder.load_framework()`
4. `MetalOrganicFrameworkBuilder.optimize_framework()`
5. `MetalOrganicFrameworkBuilder.make_supercell()`
6. `MetalOrganicFrameworkBuilder.build()`
7. Optional `Framework.write()`, `remove()`, `replace()`, `solvate()`,
   `generate_linker_forcefield()`, `md_prepare()`

### Stable internal integration points

Keep these internal handoff points intact even if role-aware data structures are
added behind them:

1. topology metadata loaded by `MofTopLibrary`
2. graph metadata attached by `FrameNet.create_net(...)`
3. normalized fragment registries assembled by
   `MetalOrganicFrameworkBuilder.load_framework()`
4. optimizer consumption of graph role ids plus normalized registries
5. graph-state progression through `G`, `sG`, `superG`, `eG`,
   `cleaved_eG`
6. `Framework.get_merged_data()` as the synchronization point after structural
   edits

## Hard Invariants

- Single-role templates remain the base case and should continue to use the
  current behavior by default.
- Do not break `MetalOrganicFrameworkBuilder` or `Framework` public method
  names.
- Keep `build()` returning `builder.framework`.
- Keep `Framework.remove()` and `Framework.replace()` returning new
  `Framework` objects.
- Keep top-level imports and `cli.py` dependency-light.
- Preserve existing graph-state names:
  - `G`
  - `sG`
  - `superG`
  - `eG`
  - `cleaved_eG`
- Preserve current numerical behavior unless a phase explicitly targets a
  numerical change.

## Canonical Internal Role Model

Use one canonical internal model throughout the effort. Do not let each phase
invent its own interpretation of "role".

## Role Source of Truth

Role identifiers live on the topology graph:

- FrameNet.G.nodes[n]["node_role_id"]
- FrameNet.G.edges[e]["edge_role_id"]

Fragment registries map role identifiers to fragment payloads:

- node_role_registry[role_id]
- edge_role_registry[role_id]

Downstream modules must read role identity from the graph and resolve fragment
payloads through the registries. They must not invent new role identifiers.

### Role semantics

- A **topology role label** is a stable topology-level identifier attached to a
  node or edge position in the template/graph.
- A **node role** means "these topology nodes are equivalent for fragment
  assignment purposes unless overridden later".
- An **edge role** means "these topology edges/edge-centers are equivalent for
  linker assignment purposes unless overridden later".
- A role is **not** automatically a different chemistry class.
  - two different topology role labels may still map to the same node fragment
    or the same linker chemistry
- a role is a topology classification key, not a chemical identity by itself
- chemical meaning is introduced only when a role id is resolved through the
  node/linker registries


### Normalized internal representation

Runtime role identity must be stored on the topology graph.

Core runtime structures:

- graph node attributes carrying `node_role_id`
- graph edge attributes carrying `edge_role_id`
- `node_role_registry`
  - per `node_role_id` -> normalized node fragment assignment/config
- `edge_role_registry`
  - per `edge_role_id` -> normalized linker fragment assignment/config

Optional helper views may exist, but they must be derived from graph-stored
role ids rather than maintained as competing sources of truth.

Minimum normalized fields expected in the registries:

- node role entry
  - `role_id`
  - `expected_connectivity`
  - fragment source/config for that role
- edge role entry
  - `role_id`
  - `linker_connectivity`
  - fragment source/config for that role

Every phase should be able to map:

- topology graph instance -> role id
- role id -> fragment/config payload


### Role source-of-truth and registry ownership rule

- Graph-stored role identifiers are the only source of truth for topology role
  identity at runtime.
- Builder registries are the only source of truth for resolving a role id to a
  fragment/config payload at runtime.

- `FrameNet` owns topology-instance -> role-id assignment on graph objects and
  related topology metadata.
- `MofTopLibrary` owns family/template metadata loading and normalization into
  passive role metadata structures.
- `MetalOrganicFrameworkBuilder` owns runtime role registries used for fragment
  selection and caching.
- Optimizer, supercell, writer, defects, and MD layers may consume these
  registries, but must not redefine role ids or create competing sources of
  truth for the same mapping.

### Single-role normalization

When no role metadata exists:

- every topology node normalizes to `node:default`
- every topology edge normalizes to `edge:default`

Legacy scalar builder inputs normalize into one-entry registries:

- `node_metal`, `dummy_atom_node`, and the selected node fragment become the
  payload for `node_role_registry["node:default"]`
- linker inputs such as `linker_xyzfile`, `linker_molecule`, or
  `linker_smiles` become the payload for `edge_role_registry["edge:default"]`

This default single-role normalization is the required backward-compatible base
case for all later phases.

## Runtime Data Flow

Topology parsing

CIF / template
      ↓
FrameNet.create_net()
      ↓
FrameNet.G (topology graph with role annotations)

Builder initialization

builder.load_framework()
      ↓
builder.node_role_registry
builder.edge_role_registry

Optimization

optimizer consumes
  FrameNet.G
  node_role_registry
  edge_role_registry

Result

Framework
  ├─ G
  ├─ sG
  ├─ superG
  ├─ eG
  └─ cleaved_eG

## Current Single-Role Bottlenecks in the Code

These are the main places where single-role assumptions are currently embedded:

- `src/mofbuilder/io/cif_reader.py`
  - stores a single `V_con` and a single `EC_con`
  - `get_type_atoms_fcoords_in_primitive_cell(target_type=...)` only selects by
    coarse atom type such as `"V"`, `"E"`, `"EC"`
  - `_extract_atoms_fcoords_from_lines()` calls `remove_tail_number(...)`,
    which collapses label detail that could otherwise distinguish roles
- `src/mofbuilder/core/moftoplibrary.py`
  - `MOF_topology_dict` resolves each family to one `node_connectivity`, one
    `linker_topic`, and one topology stem
- `src/mofbuilder/core/net.py`
  - exposes singular `max_degree`, `linker_connectivity`, `sorted_nodes`,
    `sorted_edges`, and `pair_vertex_edge`
  - distinguishes only ditopic vs multitopic linkers
- `src/mofbuilder/core/builder.py`
  - stores singular fields such as `node_metal`, `node_data`,
    `dummy_atom_node_dict`, `linker_*`, `termination_*`
  - `_read_node()` selects exactly one node fragment
  - `_read_linker()` prepares exactly one linker fragment family
- `src/mofbuilder/core/optimizer.py`
  - expects one `V_data` / `V_X_data`
  - expects one edge fragment family via `E_data`, `E_X_data`, and optionally
    `EC_data`, `EC_X_data`
- `src/mofbuilder/core/supercell.py`
  - branches on global `linker_connectivity == 2`
  - `_get_xoo_dict_of_node()` assumes node XOO ordering is globally reusable
- `src/mofbuilder/core/write.py`
  - uses one global `xoo_dict` and one global `dummy_atom_node_dict`
- `src/mofbuilder/core/defects.py`
  - uses global `node_connectivity`, `linker_connectivity`, `xoo_dict`, and
    `matched_vnode_xind`
- `src/mofbuilder/md/linkerforcefield.py` and
  `src/mofbuilder/md/gmxfilemerge.py`
  - assume one linker force-field family for the whole structure

## Strategy

Generalize bottom-up:

1. make topology parsing role-aware
2. make builder state role-aware
3. make optimizer consume role-aware fragments
4. propagate roles through supercell / edge-graph / writer / defects
5. only then generalize simulation-prep and force-field assembly

Do **not** start by changing `builder.py` or `framework.py` to accept many new
inputs before `net.py` and the graph metadata can actually represent roles.

## Controlled Generalization Rule

- At each phase, generalize exactly one layer to consume role-aware data and
  adapt adjacent layers only enough to preserve compatibility.
- Do not "finish the whole stack early" from inside a lower-layer phase.
- If a later-layer change becomes necessary to validate a phase, make the
  narrowest compatibility shim possible and stop there.
- Do not convert every scalar field, helper, cache, or convenience attribute
  into a multi-role structure unless that specific phase requires it.
- Prefer normalization at phase boundaries and keep local single-role fast paths
  where they remain correct.

## Test Rules Across All Phases

- Every phase that changes runtime behavior must include explicit regression
  coverage for the single-role path.
- The single-role path is not "implicitly covered" by the new role-aware path;
  add tests that prove it still normalizes to one default node role and one
  default edge role where relevant.
- Phase 3 must add direct normalization tests for legacy scalar builder inputs
  -> one-entry role registries.
- A heterogeneous multi-role test should be added only when the affected layer
  can actually consume it:
  - topology-only in Phase 1
  - metadata-only in Phase 2
  - runtime heterogeneous build behavior no earlier than Phase 4
- Do not add broad heterogeneous end-to-end tests before the underlying phase
  is ready.

## Phase Roadmap

### Phase 1: Role-Safe Topology Parsing

Objective:

- add internal role annotations at the topology/CIF layer without changing the
  current high-level pipeline

Files:

- `src/mofbuilder/io/cif_reader.py`
- `src/mofbuilder/core/net.py`
- `tests/test_io_reader.py`
- `tests/test_core_net.py`

Scope:

- preserve raw site-label information needed to distinguish node/edge roles
- avoid collapsing role-bearing labels too early in `CifReader`
- add explicit node-role / edge-role metadata onto `FrameNet.G`
- keep legacy outputs for single-role templates:
  - `linker_connectivity`
  - `max_degree`
  - `sorted_nodes`
  - `sorted_edges`
- ensure role labels are deterministic for the same input topology/template

Tests required:

- regression tests proving current single-role CIFs still produce the current
  scalar outputs
- one minimal topology-only heterogeneous-role test proving role labels survive
  parsing into graph metadata

Must **not** yet:

- change `MofTopLibrary`
- change builder inputs or builder runtime behavior
- change optimizer, supercell, writer, defects, or MD code
- introduce fragment-assignment logic based on the new role labels
- infer chemical fragment class directly from topology role labels
- redesign graph APIs beyond attaching stable role annotations

Exit criteria:

- single-role templates behave exactly as before
- `FrameNet` can emit stable per-node and per-edge role annotations
- higher layers may still ignore those roles for now
- role labels are stable and deterministic for repeated parsing of the same input

Recommended Codex thread boundary:

- production files: `cif_reader.py`, `net.py`
- test files: `test_io_reader.py`, `test_core_net.py`
- stop before touching `builder.py`

### Phase 2: Additive Family/Template Role Metadata

Objective:

- add an additive way to describe multi-role families without breaking the
  existing `MOF_topology_dict` contract

Files:

- `src/mofbuilder/core/moftoplibrary.py`
- `tests/test_core_moftoplibrary.py`
- optionally one new metadata fixture under `tests/database/`

Scope:

- keep `MOF_topology_dict` readable as-is for single-role families
- introduce optional role metadata as a sidecar mechanism instead of replacing
  the current table immediately
- expose role metadata in the canonical normalized form, but only as passive
  metadata accessors

Notes:

- choose exactly one additive metadata source for this phase and document it in
  code/tests; do not leave the schema open-ended across threads
- prefer an additive sidecar over breaking the current
  `"MOF node_connectivity metal linker_topic topology"` schema early
- this phase owns metadata schema and passive loading only
  - it does **not** own builder/runtime consumption
- choose exactly one normalized in-memory metadata shape for multi-role
  families; adapters may parse raw metadata sources, but downstream code must
  receive one stable normalized shape

Tests required:

- regression tests proving families with no role metadata still resolve exactly
  as before
- metadata tests proving a multi-role family can be loaded into the canonical
  normalized role model without invoking build/runtime code

Must **not** yet:

- refactor `builder.py`
- change fragment loading
- change optimizer inputs
- support multiple metadata formats in parallel "for flexibility"
- back-propagate runtime cache or fragment-loading concerns into the metadata
  schema

Exit criteria:

- `MofTopLibrary` can return legacy scalar metadata plus optional role metadata
- single-role families require no new metadata

Recommended Codex thread boundary:

- production file: `moftoplibrary.py`
- tests: `test_core_moftoplibrary.py`
- stop before modifying fragment loading

### Phase 3: Builder Input Normalization and Role Registries

Objective:

- make the builder internally role-aware while preserving current public scalar
  inputs for the single-role case

Files:

- `src/mofbuilder/core/builder.py`
- `tests/test_core_builder.py`

Scope:

- add normalized internal structures such as:
  - node-role specs
  - edge-role specs
  - role -> fragment cache/data maps
- keep existing fields such as `node_metal`, `linker_smiles`,
  `linker_xyzfile`, `linker_molecule`, `termination_name` as shorthand for the
  default single-role path
- preserve `load_framework() -> optimize_framework() -> make_supercell()`

Important constraint:

- do not generalize `FrameNode` and `FrameLinker` into multi-role managers if
  they can stay single-fragment processors instantiated per role
- do not redesign metadata schema here; consume the canonical metadata produced
  by Phase 2

Tests required:

- regression tests proving the current single-role builder inputs still behave
  the same
- explicit normalization tests showing scalar builder inputs become one-entry
  `node:default` / `edge:default` registries

Must **not** yet:

- change the Phase 2 metadata format
- modify optimizer placement logic
- modify supercell, writer, defects, or MD code
- generalize helper objects outside builder-owned normalization and registry
  setup

Exit criteria:

- builder can carry role-aware fragment specs internally
- single-role builds still use the current scalar attributes unchanged

Recommended Codex thread boundary:

- production file: `builder.py`
- tests: `test_core_builder.py`
- stop before modifying `optimizer.py`
- legacy scalar compatibility attributes may remain available, but should be
  derived from normalized role-aware state where possible

### Phase 4: Role-Aware Optimizer Inputs

Objective:

- let placement/optimization consume per-role node and edge fragments

Files:

- `src/mofbuilder/core/optimizer.py`
- `src/mofbuilder/utils/geometry.py`
- `tests/test_core_optimizer.py`

Scope:

- replace the assumption of one global node fragment and one global edge
  fragment with role-based lookups
- keep the current ditopic vs multitopic branch structure unless there is a
  clear need to refactor it later
- preserve the single-role path as a fast/simple path

Likely data changes:

- sorted nodes/edges remain, but role annotations drive which fragment data is
  selected for each node/edge
- legacy scalar fields should continue to work by normalizing into one-item
  registries

Tests required:

- regression tests proving current single-role optimizer behavior is unchanged
- one minimal heterogeneous multi-role runtime test at the optimizer boundary
  only; do not require full writer/MD success yet

Must **not** yet:

- refactor supercell/edge-graph logic
- generalize writer/defects/MD
- expand the public builder API surface
- redesign role ids or registry shapes introduced in Phases 1-3

Exit criteria:

- optimizer can place structures correctly when different node/edge roles use
  different fragment payloads
- single-role numerical behavior remains unchanged

Recommended Codex thread boundary:

- production files: `optimizer.py`, optionally `geometry.py`
- tests: `test_core_optimizer.py`
- stop before touching `supercell.py`

### Phase 5: Role Propagation Through Supercell and Edge Graph

Objective:

- keep node-role and edge-role identity alive after supercell expansion and eG
  generation

Files:

- `src/mofbuilder/core/supercell.py`
- `tests/test_core_supercell.py`

Scope:

- propagate role annotations into `superG`, `eG`, and `cleaved_eG`
- remove assumptions that one global `xoo_dict` shape/order is valid for every
  node role
- keep `matched_vnode_xind` semantics, but make them robust to role-specific
  node layouts

Tests required:

- regression tests proving single-role `superG` / `eG` behavior is unchanged
- one minimal heterogeneous-role test proving role labels survive into `eG` and
  `cleaved_eG`

Must **not** yet:

- redesign writer output formats
- redesign defect APIs
- generalize MD or force-field code
- repair earlier-phase normalization/schema issues inside this phase; stop and
  update the plan instead

Exit criteria:

- role metadata survives supercell and edge-graph construction
- single-role behavior is unchanged

Recommended Codex thread boundary:

- production file: `supercell.py`
- tests: `test_core_supercell.py`
- stop before touching writer/defects/framework

### Phase 6: Role-Aware Writer and Defect Metadata

Objective:

- generalize merged-data assembly and defect logic to use role-specific node
  metadata

Files:

- `src/mofbuilder/core/write.py`
- `src/mofbuilder/core/defects.py`
- `src/mofbuilder/core/framework.py`
- `tests/test_core_write.py`
- `tests/test_core_defects.py`
- `tests/test_core_framework.py`

Scope:

- replace the assumption of one global `dummy_atom_node_dict`
- replace the assumption of one global `xoo_dict`
- keep `Framework.get_merged_data()` as the synchronization point after
  structural edits
- preserve current mutation semantics:
  - `remove()` and `replace()` return new `Framework` objects

Tests required:

- regression tests proving current single-role merged data and defect behavior
  are unchanged
- one minimal heterogeneous-role test proving writer/defect code can consume
  role-specific node metadata after Phases 4-5 are complete

Must **not** yet:

- change force-field generation contracts
- generalize MD topology assembly
- refactor `Framework` public methods beyond role-aware internal plumbing
- invent writer-local or defect-local role identifiers

Exit criteria:

- merged output and defect-handling paths work with node-role-specific metadata
- single-role merged output remains stable

Recommended Codex thread boundary:

- production files: `write.py`, `defects.py`, `framework.py`
- tests: `test_core_write.py`, `test_core_defects.py`,
  `test_core_framework.py`
- stop before touching MD / force-field code

### Phase 7: Multi-Edge Force-Field and Simulation-Prep Support

Objective:

- let simulation prep handle more than one linker/edge role

Files:

- `src/mofbuilder/md/linkerforcefield.py`
- `src/mofbuilder/md/gmxfilemerge.py`
- `src/mofbuilder/core/framework.py`
- `tests/test_md_linkerforcefield.py`
- `tests/test_md_gmxfilemerge.py`
- `tests/test_core_framework.py`

Scope:

- support one force-field mapping/generation path per edge role
- merge multiple linker ITP outputs into the generated topology
- keep the current one-linker path unchanged

Important caution:

- this phase is not required to make multi-role structure construction work
- do not block phases 1-6 on MD generalization
- do not promise chemically complete heterogeneous force-field support before at
  least one minimal multi-edge path is proven end-to-end

Tests required:

- regression tests proving the current single-linker MD-prep path is unchanged
- one minimal heterogeneous multi-edge test proving topology generation and MD
  setup wiring can succeed at the current supported level

Must **not** yet:

- redesign topology parsing or metadata schema
- refactor earlier phases "for cleanup"
- broaden scope into general force-field research problems
- claim broad heterogeneous chemistry support from one minimal successful path

Exit criteria:

- at least one minimal heterogeneous multi-edge case can prepare simulation
  files at the currently supported level
- current single-linker MD prep remains unchanged

Recommended Codex thread boundary:

- production files: `linkerforcefield.py`, `gmxfilemerge.py`,
  possibly `framework.py`
- tests: `test_md_linkerforcefield.py`, `test_md_gmxfilemerge.py`,
  `test_core_framework.py`

### Phase 8: Documentation and Example Sync

Objective:

- document the new internal model after the code stabilizes

Files:

- `README.md`
- `docs/source/manual/*.md`
- `ARCHITECTURE.md`
- `CODEX_CONTEXT.md`
- `AGENTS.md`

Scope:

- explain the single-role base case and the new multi-role internal model
- document any additive family/template metadata introduced in phase 2
- add one multi-role example only after the code path is stable

Must **not** yet:

- mix docs work with algorithmic refactors
- document unsupported heterogeneous cases as complete

Exit criteria:

- docs match the implemented behavior
- no claims are made before tests exist

Recommended Codex thread boundary:

- docs only
- do not mix this with algorithmic changes

## Recommended Codex Thread Boundaries

Use one thread per phase. Do **not** combine these:

- Phase 1 with Phase 4
- Phase 3 with Phase 6
- Phase 5 with Phase 7

Recommended sequence:

1. Thread A: Phase 1 only
2. Thread B: Phase 2 only
3. Thread C: Phase 3 only
4. Thread D: Phase 4 only
5. Thread E: Phase 5 only
6. Thread F: Phase 6 only
7. Thread G: Phase 7 only
8. Thread H: Phase 8 only

Thread-size rule:

- no thread should modify more than 2-3 production modules unless the touched
  modules are already tightly coupled inside one phase
- each thread should include the nearest matching tests
- each thread should stop at a clean handoff boundary, even if the full feature
  is not yet complete
- if a thread discovers a schema/runtime boundary problem, stop and update the
  plan rather than fixing both sides in one pass

## Recommended Internal Modeling Rule

Use role registries internally and keep scalars as compatibility shorthands.

Recommended pattern:

- normalize the current scalar configuration into one-item role maps
- treat single-role templates as:
  - one node role
  - one edge role
- keep legacy scalar attributes readable/writable during the transition
- only remove duplicated internal code after the role-aware path is stable and
  the single-role base case is proven unchanged
- topology role labels and fragment assignment registries remain separate layers
- the normalized default ids are always `node:default` and `edge:default` when
  no role metadata is present

## Definition of Done for the Overall Effort

- single-role templates still pass their current tests unchanged
- legacy scalar inputs normalize into one-entry registries predictably
- `FrameNet` can emit stable role metadata
- builder/optimizer/supercell/writer/defects can consume multiple roles without
  forking the high-level pipeline
- MD prep either supports multiple edge roles or is explicitly documented as a
  later-phase limitation
- docs describe the new role-aware internals without breaking the existing user
  workflow

## Architecture Stop Rule

If a phase reveals a conflict between:

- the canonical role model
- the current pipeline
- or graph state invariants

Stop implementation and update this plan before continuing.

Do not redesign multiple layers in a single thread.
Do not resolve a schema/runtime mismatch by silently changing both sides in one
pass; stop at the boundary and make the conflict explicit.

## Execution Logging Rule

Each phase should produce a corresponding entry in `WORKLOG.md` with:

- intended scope
- files changed
- tests run
- result
- unresolved issues
- handoff notes

`PLANS.md` remains the controlling plan.
`WORKLOG.md` records execution history and checkpoint status.