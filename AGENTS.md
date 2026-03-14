# AGENTS.md

## Repository Purpose

MOFBuilder is scientific/research software for constructing and preparing
metal-organic frameworks (MOFs) for simulation workflows. Favor correctness,
API stability, and preservation of geometry/graph conventions over broad
refactors.

## Shared Control-Doc Authority

- `AGENTS.md` is the authority for repo-wide architecture, role-model, testing,
  and documentation rules.
- `PLANS.md` is the frozen multi-phase roadmap. Do not edit it unless a real
  schema, runtime, or invariant conflict is first recorded.
- `STATUS.md` is the live pointer to the active phase and checkpoint.
- `WORKLOG.md` is append-only execution history. Read `STATUS.md` first, then
  only the active checkpoint and Phase Contract unless older history is needed.
- `REVIEWER.md`, `PLANNER.md`, and `EXECUTOR.md` should reference section names
  in this file instead of copying unchanged invariant lists.

## Codebase Map

- `src/mofbuilder/__init__.py`: lazy top-level package surface
- `src/mofbuilder/cli.py`: dependency-light CLI
- `src/mofbuilder/core/`: build pipeline
  - `builder.py`: `MetalOrganicFrameworkBuilder`
  - `framework.py`: built-object API
  - `moftoplibrary.py`: topology/family lookup
  - `net.py`: topology graph extraction from template CIF
  - `optimizer.py`: geometry and cell optimization
  - `supercell.py`: `SupercellBuilder`, `EdgeGraphBuilder`
  - `defects.py`: `TerminationDefectGenerator`
  - `write.py`: `MofWriter`
- `src/mofbuilder/io/`: structure readers and writers
- `src/mofbuilder/md/`: solvation, force-field, and MD setup
- `src/mofbuilder/utils/`: geometry, element, and path helpers
- `src/mofbuilder/visualization/`: `Viewer`
- `src/mofbuilder/analysis/`: placeholder-level today
- `database/`: bundled runtime assets
- `tests/`: unit, smoke, and stubbed heavy-dependency coverage

## Public APIs to Preserve

Do not change public APIs, function signatures, or attribute names unless the
task explicitly requires it.

Important surfaces:

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

## Engineering Rules

- Prefer minimal, localized edits.
- Avoid unnecessary dependencies.
- Reuse existing helpers before adding new ones.
- Prefer NumPy vectorization where appropriate.
- Keep orchestration in `core`, parsing in `io`, and simulation prep in `md`.
- Preserve lazy-import behavior in `src/mofbuilder/__init__.py` and keep
  `cli.py` dependency-light.
- Do not move heavy imports into `mofbuilder.__init__` or `cli.py`.
- Preserve current stdout and exit-code behavior for the CLI unless explicitly
  changing that interface.
- Keep the builder/framework split intact:
  - `MetalOrganicFrameworkBuilder` orchestrates construction.
  - `Framework` owns post-build mutation, export, solvation, and MD setup.
- Preserve object-lifetime semantics:
  - `MetalOrganicFrameworkBuilder.build()` populates and returns
    `builder.framework`.
  - `Framework.remove()` and `Framework.replace()` return new `Framework`
    instances.
  - `Framework.solvate()`, `generate_linker_forcefield()`, `md_prepare()`, and
    `show()` mutate the current `Framework`.
- When modifying graph-producing code, check downstream consumers before
  changing node or edge attributes or coordinate conventions.
- When modifying framework mutation code, keep merged data synchronized through
  `get_merged_data()` or the existing equivalent flow.
- When editing `MOF_topology_dict` consumers, preserve the existing columns:
  MOF family, node connectivity, metal, linker topic, topology stem.

## Do Not

- Do not rename major classes, workflow methods, graph states, or public
  attributes without explicit approval.
- Do not make flashy architectural rewrites.
- Do not add convenience dependencies when the existing stack is sufficient.
- Do not treat placeholder modules as production-ready.
- Do not assume notebooks, generated structures, `output/`, or `output_cifs/`
  are canonical source artifacts.
- Do not assume a `topology/` package exists; topology lives in
  `core/moftoplibrary.py` and `core/net.py`.
- Do not silently change bundled database formats such as `MOF_topology_dict`.

## Test Execution Rule

Run all tests through the repository runner:

```bash
scripts/run_tests.sh <target>
```

Rules:

- Do not run `pytest` directly.
- Do not reconstruct the environment manually.
- Use the narrowest relevant test target for the current work.
- If tooling is missing, do static verification and say exactly what was not
  run.
- Never claim runtime verification you did not perform.

Useful targets:

- `tests/smoke/` for lazy imports and CLI behavior
- `tests/test_core_*.py` for builder, framework, and topology logic
- `tests/test_io_*.py` for readers and writers
- `tests/test_md_*.py` for solvation, force-field, and MD setup

Important facts:

- `tests/conftest.py` stubs many optional scientific dependencies.
- Those stubs are suitable for interface and control-flow coverage, not for
  proving numerical or simulation correctness.
- If you edit package exports or `cli.py`, inspect
  `tests/smoke/test_smoke_imports.py` and `tests/smoke/test_smoke_cli.py`.
- If you edit builder or framework behavior, inspect
  `tests/test_core_builder.py` and `tests/test_core_framework.py`.

## Documentation Rules

- Keep docs synchronized with actual code.
- Update `README.md` and `docs/source/manual/*.md` when public behavior
  changes.
- Inspect both `docs/` and `docs/source/` before editing; similar material
  exists in both places.
- Do not edit `docs/source/api_generated/` manually unless the task is
  explicitly about generated API pages.
- If behavior is uncertain, read code and tests before changing docs.

## Scientific Software Caution

- Respect existing scientific data structures and geometry conventions.
- Preserve numerical behavior unless explicitly improving it.
- Be careful with unit-cell transforms, fractional/cartesian conversions, and
  linker fragment-length logic.
- Preserve the ditopic versus multitopic linker distinction.
- Be conservative with tolerance, search-range, and graph-matching changes.
- When scientific intent is unclear, inspect docstrings, inline comments,
  tests, and call sites first.

## Common Task Routing

- Optimizers: `src/mofbuilder/core/optimizer.py`,
  `src/mofbuilder/core/superimpose.py`,
  `src/mofbuilder/utils/geometry.py`
- Builder internals: `src/mofbuilder/core/builder.py`
- Built-object behavior: `src/mofbuilder/core/framework.py`
- Simulation prep: `src/mofbuilder/core/framework.py`, `src/mofbuilder/md/`
- Docs: `README.md`, `docs/source/manual/*.md`, root control docs
- Tests: mirror the relevant module under `tests/` and reuse
  `tests/conftest.py`

High-risk areas:

- `src/mofbuilder/core/optimizer.py`
- `src/mofbuilder/core/supercell.py`
- `src/mofbuilder/core/framework.py`
- `src/mofbuilder/md/linkerforcefield.py`

Current caution points:

- `src/mofbuilder/analysis/` is mostly placeholder code.
- `Framework.write()` references `remove_defects()`, but the public method is
  `remove()`. Inspect carefully before refactoring defect handling.
- `docs/` and `docs/source/manual/` overlap; update both when user-facing
  workflows change.

## Planning And Risk Handling

- For major features, plan first in `PLANS.md`, then implement one milestone
  per thread.
- Prefer low-level changes before high-level orchestration changes.
- Before phase work, summarize:
  1. goal
  2. scope
  3. invariants
  4. stop rule
- If a request conflicts with the current architecture or carries material
  risk:
  1. explain the issue
  2. explain why it is risky
  3. suggest a safer design
  4. ask for confirmation before proceeding
- When proposing or implementing a change, state:
  - architectural impact
  - modules affected
  - risk level
  - whether the change is local or system-wide

## Architecture Preservation

MOFBuilder is a staged, graph-centered build pipeline. New features must extend
that workflow rather than replace it.

Conceptual stages:

1. topology loading
2. building unit loading
3. abstract framework graph construction
4. geometric optimization
5. atomic structure realization
6. derived graph generation
7. supercell generation
8. framework modification

Stable invariants:

- graph-centered architecture
- staged build pipeline
- topology templates drive connectivity
- atomic coordinates derive from optimized graph state
- single-node and single-linker templates remain the base case

## Performance Guardrail

Single-role builds must remain the fast path. Multi-role support must not add
significant overhead when only one role exists.

## Architecture Lock

The following runtime order is locked unless `PLANS.md` explicitly says
otherwise:

1. `MofTopLibrary.fetch(...)`
2. `FrameNet.create_net(...)`
3. `MetalOrganicFrameworkBuilder.load_framework()`
4. `MetalOrganicFrameworkBuilder.optimize_framework()`
5. `MetalOrganicFrameworkBuilder.make_supercell()`
6. `MetalOrganicFrameworkBuilder.build()`

Graph states that must remain:

- `G`
- `sG`
- `superG`
- `eG`
- `cleaved_eG`

Rules:

- Do not rename these methods or graph states.
- Do not reorder, merge, or add top-level pipeline stages.
- Do not move responsibilities between modules.
- If a task requires any of the above, stop and record the conflict instead of
  changing the architecture.

## Architecture Milestone Lock

The completed Phase 1-8 role-aware architecture is the approved baseline.
Future work must extend it rather than redefine it.

## Role Model Invariants

Role identifiers are the only topology classification mechanism.

- `node_role_id` lives on `FrameNet.G.nodes[n]["node_role_id"]`
- `edge_role_id` lives on `FrameNet.G.edges[e]["edge_role_id"]`
- role ids must not be inferred from chemistry
- role ids must not be recomputed downstream
- local role maps must not replace the canonical graph-stored ids
- fragment registries remain `node_role_registry` and `edge_role_registry`
- modules consuming roles must resolve payloads through those registries
- single-role templates must remain compatible with the canonical default-role
  base case

## Module Responsibility Lock

- `FrameNet`: topology graph construction and topology role annotation
- `MofTopLibrary`: topology family metadata
- `MetalOrganicFrameworkBuilder`: fragment normalization and runtime role
  registries
- `Optimizer`: node and linker placement
- `Supercell`: supercell expansion
- `Writer` / `Framework`: merged-structure output and framework-level behavior
- `Defects`: defect operations
- `md/`: simulation preparation

Do not move responsibilities between modules.

## Phase Execution Rules

- Treat `PLANS.md` as the frozen roadmap.
- Work one phase at a time; do not mix non-adjacent phases in one thread.
- Before phase work, check `STATUS.md`, then update the matching checkpoint in
  `WORKLOG.md`.
- If implementation reveals a conflict with `PLANS.md`, the role model, or any
  graph/pipeline invariant, record it in `WORKLOG.md` and `STATUS.md` before
  revising the plan.
- End each phase thread with a handoff log in `WORKLOG.md` and a concise
  `STATUS.md` update.

## Work Log Rule

For each implementation phase, update `WORKLOG.md` at:

1. before coding
2. after coding
3. handoff

Do not rewrite history except for clearly marked corrections.

## Phase Contract Rule

Every implementation phase must have a Phase Contract that defines:

- goal
- review context
- scope
- allowed files
- forbidden files
- shared invariants reference
- phase-specific constraints
- compatibility requirements
- required tests
- success criteria
- stop rule

Implementation must stay inside that contract.

## Phase Contract Flexibility Rule

Allowed-file lists may include small support seams only when all of the
following are true:

- locked pipeline, graph-state, role-model, and public-runtime behavior stay
  unchanged
- `src/mofbuilder/core/builder.py` logic is untouched unless it is already in
  scope
- the change is limited to workflow/control plumbing, environment/config
  helpers, the repository test runner, or one narrow supporting regression test
- the reason is recorded in `WORKLOG.md` and reflected in `STATUS.md`

Typical support seams:

- `workflow/run.py`
- `workflow/*.md`
- `scripts/run_tests.sh`
- narrow environment/configuration files
- one closely related regression test

Do not use this rule to smuggle in core scientific or runtime scope.

## Planner Scope Rule

The planner is planning-only.

The planner may:

- read all control documents
- generate a Phase Contract
- update `WORKLOG.md`
- update `STATUS.md`
- update workflow or prompt control markdown only when the user explicitly asks
  for rule refinement or a review-blocking contract mismatch requires a narrow
  policy correction

The planner must not modify source code, tests, `PLANS.md`, `ARCHITECTURE.md`,
`CODEX_CONTEXT.md`, or `AGENTS.md` unless the user explicitly requests a
policy-level change or a real control-doc conflict requires it.
