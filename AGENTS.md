# AGENTS.md

## Repository Purpose

MOFBuilder is a Python toolkit for constructing and preparing metal-organic
framework (MOF) structures for simulation workflows. The codebase is
scientific/research software: correctness, API stability, and preservation of
geometry/graph conventions matter more than broad refactors.

## Main Technical Focus

- Topology-based MOF construction
- Node/linker/termination fragment assembly
- Geometry conversion and alignment
- Rotation and cell optimization
- Supercell and edge-graph generation
- Export, defect editing, solvation, and MD preparation

## Important Directories and Modules

- `src/mofbuilder/__init__.py`
  - Lazy top-level package surface
- `src/mofbuilder/cli.py`
  - Dependency-light CLI for version/data-path/topology queries
- `src/mofbuilder/core/`
  - Main construction pipeline
  - `builder.py`: `MetalOrganicFrameworkBuilder`
  - `framework.py`: built-object API
  - `moftoplibrary.py`: `MOF_topology_dict` lookup and template selection
  - `net.py`: topology graph extraction from template CIF
  - `node.py`, `linker.py`, `termination.py`: fragment preparation
  - `optimizer.py`: geometry and cell optimization
  - `supercell.py`: `SupercellBuilder`, `EdgeGraphBuilder`
  - `defects.py`: `TerminationDefectGenerator`
  - `write.py`: `MofWriter`
- `src/mofbuilder/io/`
  - CIF/PDB/GRO/XYZ readers and writers
- `src/mofbuilder/md/`
  - `SolvationBuilder`, `LinkerForceFieldGenerator`,
    `GromacsForcefieldMerger`, `OpenmmSetup`
- `src/mofbuilder/utils/`
  - Geometry helpers, data-path resolution, periodic table, file lookup
- `src/mofbuilder/visualization/`
  - `Viewer`
- `src/mofbuilder/analysis/`
  - Present but currently placeholder-level
- `database/`
  - Runtime assets
  - `MOF_topology_dict`, `template_database/`, `nodes_database/`,
    `terminations_database/`, `solvents_database/`, `nodes_itps/`,
    `terminations_itps/`, `mdps/`
- `tests/`
  - Unit/smoke tests and heavy-dependency stubs

## Public APIs to Preserve

Keep existing public APIs, function signatures, and attribute names unless
explicitly asked to change them.

Important public surfaces:

- `mofbuilder.MetalOrganicFrameworkBuilder`
- `mofbuilder.core.Framework`
- `mofbuilder.cli`
- `MetalOrganicFrameworkBuilder.list_available_mof_families`
- `MetalOrganicFrameworkBuilder.list_available_metals`
- `MetalOrganicFrameworkBuilder.list_available_terminations`
- `MetalOrganicFrameworkBuilder.list_available_solvents`
- `Framework.write`
- `Framework.remove`
- `Framework.replace`
- `Framework.solvate`
- `Framework.generate_linker_forcefield`
- `Framework.md_prepare`
- `Framework.show`

## Coding Rules

- Prefer minimal, localized edits
- Avoid unnecessary new dependencies
- Prefer NumPy vectorization where appropriate
- Follow existing data shapes and geometry conventions
- Preserve lazy-import behavior in package `__init__` files
- Keep orchestration in `core`, parsing in `io`, and simulation prep in `md`
- Reuse existing helpers before adding new ones
- Preserve current stdout/exit-code style for CLI and list/inspection helpers
  unless a task explicitly changes that interface

## Refactoring Rules

- Do not rename major classes or workflow methods unless explicitly requested
- Do not move heavy imports into `mofbuilder.__init__` or `cli.py`
- Do not assume the whole package is dependency-light
  - only the package surface and CLI are intentionally light
  - `core` and `md` import the heavy scientific stack directly
- Do not repurpose graph/state field names casually
  - `G`, `sG`, `superG`, `eG`, `cleaved_eG` already carry meaning
- Keep the builder/framework split intact
  - `MetalOrganicFrameworkBuilder` orchestrates construction
  - `Framework` owns post-build mutation/export/MD setup
- Preserve current object-lifetime semantics
  - `MetalOrganicFrameworkBuilder.build()` populates and returns
    `builder.framework`
  - `Framework.remove()` and `Framework.replace()` return new `Framework`
    instances
  - `Framework.solvate()`, `generate_linker_forcefield()`, `md_prepare()`, and
    `show()` mutate the current `Framework`
- When modifying graph-producing code, check downstream consumers before
  changing node/edge attributes or coordinate conventions
- When modifying framework mutation code, ensure merged data stays in sync
  through `get_merged_data()` or equivalent flow
- When editing `MOF_topology_dict` consumers, preserve the expected columns:
  MOF family, node connectivity, metal, linker topic, topology stem

## Phased Execution Rules

- Treat `PLANS.md` as the frozen execution roadmap for the multi-role effort.
  Do not edit it unless a real schema/runtime/invariant conflict is discovered.
- Work one phase at a time. Do not combine non-adjacent phases in one thread
  and do not pull later-phase cleanup into an earlier-phase change.
- Before starting a phase task, check `STATUS.md` for the active phase and
  update the matching checkpoint in `WORKLOG.md`.
- If implementation reveals a conflict with `PLANS.md`, the canonical role
  model, or graph/pipeline invariants, stop and record it in `WORKLOG.md` and
  `STATUS.md` before revising the plan.
- End each phase thread with a handoff log in `WORKLOG.md` and a short
  `STATUS.md` update covering files changed, verification performed, blockers,
  and the next checkpoint.

chmod +x scripts/run_tests.sh

## Test Execution Rule

All tests must be executed using the repository test runner:

scripts/run_tests.sh

Example:

scripts/run_tests.sh tests/test_io_reader.py
scripts/run_tests.sh tests/test_core_net.py

Rules:

- Do not run pytest directly.
- Do not attempt to reconstruct the environment manually.
- Always use `scripts/run_tests.sh` so the correct conda environment and
  PYTHONPATH are applied.
- Prefer running the narrowest relevant tests for the current phase.

Rules:

- Do not run pytest directly without the conda environment.
- Always set `PYTHONPATH=src`.
- Prefer running the narrowest relevant tests for the current phase.

## Testing and Verification Expectations

Tests exist and are organized by subsystem under `tests/`.

Useful targets:

- `tests/smoke/` for lazy imports and CLI behavior
- `tests/test_core_*.py` for builder/framework/topology logic
- `tests/test_io_*.py` for readers/writers
- `tests/test_md_*.py` for solvation/force-field/MD setup

Important facts:

- `tests/conftest.py` installs stubs for `veloxchem`, `mpi4py`, `openmm`,
  `scipy`, and other optional dependencies
- This allows much of the test suite to run without the full scientific stack
- These stubs are good for interface and control-flow coverage, not for proving
  real numerical or simulation correctness

Expected verification behavior:

- Run the narrowest relevant tests when the environment supports them
- If environment tooling is missing, do static verification and say clearly what
  was not run
- Do not claim runtime verification you did not perform
- If you edit package exports or `cli.py`, inspect
  `tests/smoke/test_smoke_imports.py` and `tests/smoke/test_smoke_cli.py`
- If you edit build or framework behavior, inspect
  `tests/test_core_builder.py` and `tests/test_core_framework.py`

## Documentation Rules

- Keep documentation synchronized with actual code
- Update `README.md` and `docs/source/manual/*.md` when public behavior changes
- Inspect both `docs/` and `docs/source/` before editing docs; similar material
  appears in both places
- Avoid editing `docs/source/api_generated/` manually unless the task is
  explicitly about generated API pages
- If behavior is uncertain, read code and tests before “improving” docs

## Scientific Software Caution Notes

- Respect existing scientific data structures and geometry conventions
- When editing optimization or geometry code, preserve numerical behavior unless
  explicitly improving it
- Be careful with unit-cell transforms, fractional/cartesian conversions, and
  linker fragment-length logic
- Preserve the current distinction between ditopic and multitopic linker
  handling; builder/optimizer code branches on linker connectivity
- Be conservative with tolerance changes, search ranges, or graph-matching
  logic
- When uncertain about scientific intent, inspect docstrings, inline comments,
  tests, and call sites before changing behavior

## Do Not

- Do not make flashy architectural rewrites
- Do not change public attribute names just to “clean up” style
- Do not add dependencies for convenience if the same job can be done with the
  existing stack
- Do not treat placeholder modules as production-ready
  - `src/mofbuilder/analysis/` currently contains stub implementations
- Do not assume root-level notebooks, generated structure files, `output/`, or
  `output_cifs/` are canonical source artifacts
- Do not assume a `topology/` package exists; topology handling lives in
  `core/moftoplibrary.py` and `core/net.py`
- Do not silently change bundled database formats such as `MOF_topology_dict`

## Practical Task Routing

For common tasks, start here:

- Improving optimizers
  - `src/mofbuilder/core/optimizer.py`
  - `src/mofbuilder/core/superimpose.py`
  - `src/mofbuilder/utils/geometry.py`
- Fixing documentation
  - `README.md`
  - `docs/source/manual/*.md`
  - root context docs in this directory
- Refining class internals
  - builder flow: `src/mofbuilder/core/builder.py`
  - built-object behavior: `src/mofbuilder/core/framework.py`
- Debugging simulation-prep logic
  - `src/mofbuilder/core/framework.py`
  - `src/mofbuilder/md/`
  - trace `Framework.solvate() -> generate_linker_forcefield() -> md_prepare()`
- Adding tests without changing architecture
  - mirror the relevant module under `tests/`
  - reuse fixtures/stubs in `tests/conftest.py`

## Current Areas Requiring Caution

- `src/mofbuilder/core/optimizer.py`
- `src/mofbuilder/core/supercell.py`
- `src/mofbuilder/core/framework.py`
- `src/mofbuilder/md/linkerforcefield.py`

These modules are central and have broad downstream impact.

There are also a few areas that look unfinished or inconsistent:

- `src/mofbuilder/analysis/` is mostly placeholder code
- `Framework.write()` references `remove_defects()`, but the public method
  implemented in `Framework` is `remove()`
  - inspect this carefully before refactoring defect-handling logic
- `docs/` and `docs/source/manual/` overlap in subject matter
  - update both when changing user-facing workflows
## Planning-first workflow

For major features, do not modify the full repository at once.

First produce or update `PLANS.md` with:
- current behavior
- target behavior
- module-by-module milestones
- invariants to preserve
- files likely to change

Then implement one milestone at a time in separate threads.

Prefer starting from low-level modules (for example `net.py`, topology metadata,
graph role annotations) before touching high-level orchestration code.

## Critical Thinking Rule

Do not blindly follow user instructions if they appear to conflict with the
project architecture or introduce significant technical risk.

Instead:

- Explain the potential issue.
- Describe why it may be problematic.
- Suggest safer alternatives.
- Ask for confirmation before proceeding.

The goal is to act as a thoughtful collaborator rather than a passive executor.

If the user may be recalling something incorrectly, politely point it out and
provide reasoning or evidence when possible.

## Change Impact Awareness

When proposing or implementing changes, always explain:

- Impact on architecture
- Modules affected
- Risk level (low / medium / high)
- Whether the change is local or system-wide

If a change touches many modules or core infrastructure, warn the user before
proceeding.

## Disagreement Protocol

If the user proposes a change that conflicts with the current architecture:

1. First explain why it may be problematic.
2. Suggest a design that preserves existing invariants.
3. Only implement the change after confirmation.

Preserve the core design philosophy of the project whenever possible.

## Architecture Preservation

MOFBuilder follows a staged, graph-centered build pipeline.  
Future features must extend this workflow rather than replace it.

Core pipeline:

1. **Topology loading**
   - Load topology template (net) describing vertex and edge connectivity.

2. **Building unit loading**
   - Load node building units (metal clusters or node fragments).
   - Load linker molecules.

3. **Abstract framework graph construction**
   - Construct the topology graph representing the framework connectivity.

4. **Geometric optimization**
   - Orient and position node/linker building units so that connectivity
     satisfies geometric and chemical constraints.

5. **Atomic structure realization**
   - Place atomic coordinates into the framework.
   - Store structural information as attributes in the graph.

6. **Derived graph generation**
   - Build edge-connected graphs used for further analysis.

7. **Supercell generation**
   - Expand the framework graph into periodic supercells.

8. **Framework modification**
   - Perform operations such as defect engineering or substitutions.


Rules:

- New features must integrate into one of these stages.
- The stage order must remain unchanged.
- The graph remains the central data structure.
- Single-node / single-linker templates remain the base case.
- Multi-node / multi-edge templates should generalize assignment logic
  without changing downstream stages.



  ### Invariants

The following concepts must remain stable:

- graph-centered architecture
- staged build pipeline
- backward compatibility for single-node templates
- topology templates drive connectivity
- atomic coordinates are derived from optimized graph state

## Performance Guardrail

Single-role builds must remain the fast path.

Multi-role support must not introduce significant overhead when only one role
exists.

Avoid unnecessary dictionary lookups or dynamic dispatch in tight loops when
the registry size is one.

## Work Log Rule

For each implementation phase, update `WORKLOG.md` at three points:

1. before coding: scope, invariants, out-of-scope modules
2. after coding: files changed, decisions made, tests run, result
3. at handoff: remaining issues, next phase readiness, plan conflicts

Do not rewrite old log entries except to add a clearly marked correction.

If a task reveals a conflict with `PLANS.md`, record the conflict in
`WORKLOG.md` first, then stop and update the plan explicitly.

## Architecture Lock

The following architecture is locked and must not be modified
unless explicitly instructed by PLANS.md.

Core pipeline:

1. MofTopLibrary.fetch(...)
2. FrameNet.create_net(...)
3. MetalOrganicFrameworkBuilder.load_framework()
4. MetalOrganicFrameworkBuilder.optimize_framework()
5. MetalOrganicFrameworkBuilder.make_supercell()
6. MetalOrganicFrameworkBuilder.build()

Graph states must remain:

G
sG
superG
eG
cleaved_eG

Rules:

- Do not rename these methods.
- Do not reorder pipeline steps.
- Do not merge pipeline stages.
- Do not introduce new top-level pipeline stages.
- Do not move responsibilities between modules.

If an implementation requires architecture changes,
stop and report the conflict instead of modifying it.

## Architecture Milestone Lock

The Phase 1-8 role-aware architecture milestone is now frozen as the approved
baseline for this repository.

Future debugging and feature work must extend the existing staged pipeline and
role-aware model rather than replacing them.

Do not casually mutate or redefine the completed milestone; start new work from
a new plan that preserves the locked pipeline, graph states, and graph-stored
role-id plus builder-registry architecture.

## Role Model Invariants

Role identifiers are the only topology classification mechanism.

node_role_id must live on:

FrameNet.G.nodes[n]["node_role_id"]

edge_role_id must live on:

FrameNet.G.edges[e]["edge_role_id"]

Role identifiers must never be:

- recomputed by downstream modules
- replaced by local role maps
- inferred from chemistry

Fragment registries must remain:

node_role_registry
edge_role_registry

Modules consuming roles must resolve fragment payloads
through these registries.

## Module Responsibility Lock

Each module owns specific responsibilities.

FrameNet
    topology graph construction
    topology role annotation

MofTopLibrary
    topology family metadata

MetalOrganicFrameworkBuilder
    fragment normalization
    runtime role registries

Optimizer
    node/linker placement

Supercell
    supercell expansion

Writer / Framework
    merged structure output

Defects
    defect operations

MD modules
    simulation preparation

Agents must not move responsibilities between modules.

## Phase Contract Rule

Before implementing a phase, generate a Phase Contract.

The contract defines:

goal
scope
allowed files
forbidden files
invariants
success criteria
stop rule

Implementation must remain inside the contract boundary.

## Phase Contract Flexibility Rule

Allowed-file lists define the primary execution boundary, but they do not need
to forbid every localized support seam by default.

Small, reviewable changes may be included in the active contract when all of
the following are true:

- they do not alter locked pipeline order, graph-state invariants, role-model
  invariants, or public runtime behavior
- they do not change `src/mofbuilder/core/builder.py` logic unless the active
  phase already authorizes `builder.py`
- they are limited to workflow/control plumbing, environment/configuration
  helpers, the repository test runner, or one narrow supporting regression test
- the reason for including them is recorded explicitly in `WORKLOG.md` and
  reflected in `STATUS.md`

Examples of allowed support seams when the criteria above are met:

- `workflow/run.py`
- `workflow/*.md`
- `scripts/run_tests.sh`
- narrow environment/configuration files
- one closely related test file that proves the localized change

Do not use this flexibility rule to broaden core scientific/runtime scope
silently. If the change would affect current MOFBuilder functions, builder
logic, phase-owned runtime behavior, or module responsibilities, stop and
record the conflict instead.


## Planner Scope Rule

The planner is planning-only.

The planner may:
- read all control documents
- generate a Phase Contract
- update `WORKLOG.md`
- update `STATUS.md`
- update workflow/prompt control markdown files when the user explicitly asks
  for rule refinement or when a review-blocking contract mismatch requires a
  targeted policy correction

The planner must not:
- modify source code
- modify tests
- modify `PLANS.md`, `ARCHITECTURE.md`, `CODEX_CONTEXT.md`, or `AGENTS.md`
  unless the user explicitly requests policy changes or a real control-doc
  conflict requires a narrowly scoped correction

Planner output should be limited to phase scope, invariants, allowed files,
forbidden files, required tests, success criteria, and stop rules.

Every phase implementation must pass a Reviewer check
before moving to the next phase.

Before coding, always summarize:

1. the goal
2. the scope
3. the invariants
4. the stop rule

Then confirm that the implementation will remain within the current phase
before producing code.
