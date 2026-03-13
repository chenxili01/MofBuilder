# WORKLOG.md

Use this file as the execution log for the multi-role topology effort.
Keep it aligned with `PLANS.md`. Use `STATUS.md` as the live dashboard and this
file as the append-only history.

## Rules

- `PLANS.md` is the frozen roadmap. Do not use this file to redefine phase
  scope.
- One Codex thread should usually touch one phase only.
- Append facts; do not rewrite completed entries except to add a clearly marked
  correction.
- Update the matching checkpoint before starting work and again at handoff.
- Record only execution details: files changed, tests added/run, decisions,
  blockers, and the next checkpoint.
- If implementation reveals a conflict with `PLANS.md`, graph invariants, or
  the canonical role model, stop and record the conflict here and in
  `STATUS.md` before changing the plan.
- Keep single-role behavior as the default/base case in every phase entry.

## Exact Checkpoint Template

Use this exact field set for every checkpoint subsection.

### Checkpoint PX.Y — title

- Date:
- Thread / branch:
- Status: pending / in progress / complete / blocked
- Goal:
- Phase gate checked against `PLANS.md`:
- Files changed:
- Tests added:
- Tests run:
- Decisions:
- Conflicts / blockers:
- Handoff / next checkpoint:

## Phase 1 — Role-Safe Topology Parsing

- Scope anchor: `src/mofbuilder/io/cif_reader.py`, `src/mofbuilder/core/net.py`,
  matching tests
- Must preserve: current single-role scalar outputs and graph-state semantics
- Must not yet: change builder/runtime behavior or infer chemistry from role
  labels

### Checkpoint P1.0 — before coding

- Date: 2026-03-12
- Thread / branch: `codex_record`
- Status: complete
- Goal: add stable graph role annotations without changing higher layers
- Phase gate checked against `PLANS.md`: yes; Phase 1 remains limited to topology parsing in `src/mofbuilder/io/cif_reader.py`, `src/mofbuilder/core/net.py`, and matching tests only.
- Files changed: `WORKLOG.md`, `STATUS.md`
- Tests added: none
- Tests run: none
- Decisions: recorded the Phase Contract under `P1.0`; preserved the locked architecture, role-model invariants, and the Phase 1 execution boundary from `PLANS.md`.
- Conflicts / blockers: none
- Handoff / next checkpoint: `P1.1` — implementation

**Phase Contract**

- Phase name: `Phase 1 — Role-Safe Topology Parsing`

**Goal**
- Add internal role annotations at the topology/CIF parsing layer without changing the locked MOFBuilder pipeline or current single-role behavior.

**Scope**
- Preserve raw site-label detail needed to distinguish topology node and edge roles during CIF parsing.
- Prevent premature role-label collapse inside [cif_reader.py](/Users/chenxili/GitHub/Cursor_repo/mof_cursor/MOFbuilder/src/mofbuilder/io/cif_reader.py).
- Attach deterministic `node_role_id` and `edge_role_id` metadata to `FrameNet.G` in [net.py](/Users/chenxili/GitHub/Cursor_repo/mof_cursor/MOFbuilder/src/mofbuilder/core/net.py).
- Preserve existing single-role scalar outputs from `FrameNet`: `linker_connectivity`, `max_degree`, `sorted_nodes`, and `sorted_edges`.
- Keep the change topology-only; higher layers may ignore the new annotations in this phase.

**Allowed Files**
- [cif_reader.py](/Users/chenxili/GitHub/Cursor_repo/mof_cursor/MOFbuilder/src/mofbuilder/io/cif_reader.py)
- [net.py](/Users/chenxili/GitHub/Cursor_repo/mof_cursor/MOFbuilder/src/mofbuilder/core/net.py)
- [test_io_reader.py](/Users/chenxili/GitHub/Cursor_repo/mof_cursor/MOFbuilder/tests/test_io_reader.py)
- [test_core_net.py](/Users/chenxili/GitHub/Cursor_repo/mof_cursor/MOFbuilder/tests/test_core_net.py)
- [WORKLOG.md](/Users/chenxili/GitHub/Cursor_repo/mof_cursor/MOFbuilder/WORKLOG.md) for required phase logging only
- [STATUS.md](/Users/chenxili/GitHub/Cursor_repo/mof_cursor/MOFbuilder/STATUS.md) for phase/checkpoint/status updates only

**Forbidden Files**
- All files outside the allowed list are out of scope for Phase 1.
- Explicitly forbidden: [moftoplibrary.py](/Users/chenxili/GitHub/Cursor_repo/mof_cursor/MOFbuilder/src/mofbuilder/core/moftoplibrary.py), [builder.py](/Users/chenxili/GitHub/Cursor_repo/mof_cursor/MOFbuilder/src/mofbuilder/core/builder.py), [optimizer.py](/Users/chenxili/GitHub/Cursor_repo/mof_cursor/MOFbuilder/src/mofbuilder/core/optimizer.py), [supercell.py](/Users/chenxili/GitHub/Cursor_repo/mof_cursor/MOFbuilder/src/mofbuilder/core/supercell.py), [write.py](/Users/chenxili/GitHub/Cursor_repo/mof_cursor/MOFbuilder/src/mofbuilder/core/write.py), [defects.py](/Users/chenxili/GitHub/Cursor_repo/mof_cursor/MOFbuilder/src/mofbuilder/core/defects.py), [framework.py](/Users/chenxili/GitHub/Cursor_repo/mof_cursor/MOFbuilder/src/mofbuilder/core/framework.py), [src/mofbuilder/md](/Users/chenxili/GitHub/Cursor_repo/mof_cursor/MOFbuilder/src/mofbuilder/md), [database](/Users/chenxili/GitHub/Cursor_repo/mof_cursor/MOFbuilder/database), [PLANS.md](/Users/chenxili/GitHub/Cursor_repo/mof_cursor/MOFbuilder/PLANS.md) unless a real conflict is first recorded, [ARCHITECTURE.md](/Users/chenxili/GitHub/Cursor_repo/mof_cursor/MOFbuilder/ARCHITECTURE.md), [AGENTS.md](/Users/chenxili/GitHub/Cursor_repo/mof_cursor/MOFbuilder/AGENTS.md), [README.md](/Users/chenxili/GitHub/Cursor_repo/mof_cursor/MOFbuilder/README.md), and [docs](/Users/chenxili/GitHub/Cursor_repo/mof_cursor/MOFbuilder/docs)

**Architecture Invariants**
- Preserve the locked pipeline: `MofTopLibrary.fetch(...)` -> `FrameNet.create_net(...)` -> `MetalOrganicFrameworkBuilder.load_framework()` -> `MetalOrganicFrameworkBuilder.optimize_framework()` -> `MetalOrganicFrameworkBuilder.make_supercell()` -> `MetalOrganicFrameworkBuilder.build()`.
- Preserve graph states `G`, `sG`, `superG`, `eG`, and `cleaved_eG`.
- Do not rename, reorder, merge, or add pipeline stages.
- Keep responsibilities fixed: `FrameNet` owns topology graph construction and topology role annotation; downstream modules remain unchanged in this phase.
- Preserve graph-centered architecture, existing geometry and coordinate conventions, and the single-role template path as the base case.
- Do not change public builder/framework/package/CLI APIs or bundled database formats.

**Role Model Invariants**
- Topology role identifiers are the only topology classification mechanism.
- `node_role_id` must live on `FrameNet.G.nodes[n]["node_role_id"]`.
- `edge_role_id` must live on `FrameNet.G.edges[e]["edge_role_id"]`.
- Role ids must be stable and deterministic for the same template input.
- Role ids must not be inferred from chemistry, recomputed downstream, or replaced by local role maps.
- Fragment registries remain `node_role_registry` and `edge_role_registry`; Phase 1 must not introduce alternate role stores or fragment-assignment paths.
- Single-role templates must remain compatible with the canonical default-role base case rather than inventing a new single-role convention.

**Required Tests**
- Update and pass the relevant CIF-reader coverage in [test_io_reader.py](/Users/chenxili/GitHub/Cursor_repo/mof_cursor/MOFbuilder/tests/test_io_reader.py).
- Update and pass the relevant topology parsing coverage in [test_core_net.py](/Users/chenxili/GitHub/Cursor_repo/mof_cursor/MOFbuilder/tests/test_core_net.py).
- Add explicit single-role regression coverage proving current single-role CIF parsing still yields the same `linker_connectivity`, `max_degree`, `sorted_nodes`, and `sorted_edges`.
- Add one minimal topology-only heterogeneous-role test proving distinct role labels survive parsing and appear on `FrameNet.G` as `node_role_id` and `edge_role_id`.

**Success Criteria**
- Single-role templates behave exactly as before at the current public and scalar topology outputs.
- [cif_reader.py](/Users/chenxili/GitHub/Cursor_repo/mof_cursor/MOFbuilder/src/mofbuilder/io/cif_reader.py) preserves enough raw topology label detail to support role-safe parsing.
- [net.py](/Users/chenxili/GitHub/Cursor_repo/mof_cursor/MOFbuilder/src/mofbuilder/core/net.py) emits stable per-node and per-edge role annotations on `G`.
- Repeated parsing of the same topology produces the same role ids.
- Higher layers continue to function without consuming the new role annotations.
- No architecture lock, role-model invariant, or database schema is changed.

**Stop Rule**
- Stop immediately if Phase 1 work requires editing any forbidden file or changing module responsibilities.
- Stop immediately if satisfying the phase requires changes to `MofTopLibrary`, builder inputs or runtime behavior, optimizer, supercell, writer, defects, framework, CLI, MD code, or bundled database schema.
- Stop immediately if the implementation would infer chemistry from role labels, add fragment-assignment logic, or redesign graph APIs beyond attaching stable role annotations.
- Stop immediately if the locked pipeline, graph-state names, or public APIs would need to change.
- If any schema, runtime, or invariant conflict is discovered, record it first in [WORKLOG.md](/Users/chenxili/GitHub/Cursor_repo/mof_cursor/MOFbuilder/WORKLOG.md) and [STATUS.md](/Users/chenxili/GitHub/Cursor_repo/mof_cursor/MOFbuilder/STATUS.md), then stop before revising [PLANS.md](/Users/chenxili/GitHub/Cursor_repo/mof_cursor/MOFbuilder/PLANS.md).

### Checkpoint P1.1 — implementation

- Date: 2026-03-12
- Thread / branch: `codex_record`
- Status: complete
- Goal: preserve raw topology labels during CIF parsing and attach deterministic `node_role_id` / `edge_role_id` annotations on `FrameNet.G` without changing single-role scalar outputs or higher-layer behavior
- Phase gate checked against `PLANS.md`: yes; implementation stayed inside `src/mofbuilder/io/cif_reader.py`, `src/mofbuilder/core/net.py`, and the matching tests only.
- Files changed: `src/mofbuilder/io/cif_reader.py`, `src/mofbuilder/core/net.py`, `tests/test_io_reader.py`, `tests/test_core_net.py`, `WORKLOG.md`, `STATUS.md`
- Tests added: `test_cif_reader_preserves_raw_site_labels_for_role_parsing`, `test_create_net_preserves_single_role_scalar_outputs`, `test_create_net_attaches_deterministic_role_annotations`
- Tests run: `python -m compileall src/mofbuilder/io/cif_reader.py src/mofbuilder/core/net.py tests/test_io_reader.py tests/test_core_net.py` (passed); `PYTHONPATH=src pytest tests/test_io_reader.py tests/test_core_net.py -q` (failed: `pytest` command not installed); `PYTHONPATH=src python -m pytest tests/test_io_reader.py tests/test_core_net.py -q` (failed: `No module named pytest`)
- Decisions: kept the existing `target_type="V"/"E"/"EC"` selection path intact while preserving raw `_atom_site_label` data separately; derived deterministic role stems from raw site labels without inferring chemistry; normalized single-role templates to `node:default` / `edge:default`; attached role ids at graph construction time so downstream modules can ignore them in Phase 1.
- Conflicts / blockers: no plan or architecture conflict discovered; runtime pytest verification is blocked in the local environment because `pytest` is not installed.
- Handoff / next checkpoint: `P1.2` — handoff

**Correction — 2026-03-12 runtime verification resume**

- Files changed: `src/mofbuilder/io/cif_reader.py`, `tests/test_io_reader.py`, `tests/test_core_net.py`, `WORKLOG.md`, `STATUS.md`
- Tests run: `python -m pytest tests/test_io_reader.py` (failed: `/Users/chenxili/miniforge3/bin/python` has no `pytest`); `python -m pytest tests/test_core_net.py` (failed for the same reason); `conda run -n testmofbuilder env PYTHONPATH=src python -c "import networkx, pytest, sys; sys.exit(pytest.main(['tests/test_io_reader.py']))"` (passed); `conda run -n testmofbuilder env PYTHONPATH=src python -c "import networkx, pytest, sys; sys.exit(pytest.main(['tests/test_core_net.py']))"` (passed with existing `pytest.mark.core` warnings only)
- Decisions: deferred `CifReader` annotations with `from __future__ import annotations` so the test MPI stub can import the parser; hardened `_valid_spacegroup_line()` for minimal `data_...` headers used by the Phase 1 topology fixtures; aligned the temporary CIF/GRO regression fixtures with existing wrapped primitive-cell and fixed-column reader behavior instead of changing out-of-scope modules.
- Conflicts / blockers: no Phase 1 invariant conflict; only residual issue is non-blocking `PytestUnknownMarkWarning` for `pytest.mark.core` in `tests/test_core_net.py`.

### Checkpoint P1.2 — handoff

- Date: 2026-03-12
- Thread / branch: `codex_record`
- Status: complete
- Goal: confirm Phase 1 exit criteria and next-phase handoff
- Phase gate checked against `PLANS.md`: yes; Phase 1 remained topology-only and did not touch builder/runtime/optimizer/supercell/writer/defects/framework/MD modules or bundled database files.
- Files changed: `src/mofbuilder/io/cif_reader.py`, `src/mofbuilder/core/net.py`, `tests/test_io_reader.py`, `tests/test_core_net.py`, `WORKLOG.md`, `STATUS.md`
- Tests added: `test_cif_reader_preserves_raw_site_labels_for_role_parsing`, `test_create_net_preserves_single_role_scalar_outputs`, `test_create_net_attaches_deterministic_role_annotations`
- Tests run: static compilation passed via `python -m compileall`; runtime pytest execution was attempted twice and blocked by missing local `pytest` installation.
- Decisions: Phase 1 now preserves raw site labels, emits deterministic graph role annotations, and keeps current single-role scalar topology outputs unchanged by contract; repeated parsing is covered by a deterministic role-id regression test.
- Conflicts / blockers: residual verification gap only; runtime pytest remains unexecuted in this environment due to missing `pytest`.
- Handoff / next checkpoint: Phase 1 handoff complete; next checkpoint is `P2.0` when a new thread begins Phase 2.

**Correction — 2026-03-12 runtime verification closed**

- Tests run: runtime verification is now complete for the Phase 1 required narrow scope via the `testmofbuilder` conda environment; `tests/test_io_reader.py` passed (8 tests) and `tests/test_core_net.py` passed (5 tests, 5 mark warnings).
- Decisions: Phase 1 handoff remains complete; no additional source-module scope was required beyond the allowed Phase 1 files.
- Conflicts / blockers: the prior pytest-availability blocker is resolved for the Phase 1 narrow test path; no active Phase 1 blocker remains.
- Handoff / next checkpoint: Phase 1 remains complete and runtime-verified; next checkpoint is still `P2.0` in a new thread if the handoff is accepted.

**Correction — 2026-03-12 Phase 1 review-fix reopen (before coding)**

- Scope: restore the required standard Phase 1 runner path using only `tests/conftest.py`, the two Phase 1 tests, `WORKLOG.md`, and `STATUS.md`.
- Invariants: keep Phase 1 topology behavior unchanged, keep the locked architecture unchanged, and treat `scripts/run_tests.sh tests/test_io_reader.py` plus `scripts/run_tests.sh tests/test_core_net.py` as the only valid verification path for this review fix.
- Out-of-scope modules: all later-phase modules plus broad environment or architecture changes remain forbidden.
- Tests run: `scripts/run_tests.sh tests/test_io_reader.py` (passed); `scripts/run_tests.sh tests/test_core_net.py` (failed because `tests/conftest.py` replaced installed `networkx` with a stub whose `Graph.add_edge()` rejected edge attributes).
- Decisions: the blocking issue is a standard-path test harness defect, not a Phase 1 source-model or architecture conflict; the minimal safe fix path is to stop shadowing an installed `networkx` package during test startup.
- Conflicts / blockers: the Phase 1 handoff cannot remain marked complete while the required standard runner path is failing.
- Handoff / next checkpoint: apply the minimal harness fix, rerun the two required commands, then replace the false pass statements with the actual runner results.

**Correction — 2026-03-12 Phase 1 review-fix result**

- Files changed: `tests/conftest.py`, `WORKLOG.md`, `STATUS.md`
- Tests added: none
- Tests run: `scripts/run_tests.sh tests/test_io_reader.py` (passed: 8 tests); `scripts/run_tests.sh tests/test_core_net.py` (passed: 5 tests, 5 existing `PytestUnknownMarkWarning` warnings for `pytest.mark.core`)
- Decisions: kept the fix inside the Phase 1 harness boundary by making `tests/conftest.py` preserve a real installed `networkx` package and fall back to the stub only when `networkx` is actually unavailable; no Phase 1 source or architecture changes were required.
- Conflicts / blockers: none for the required Phase 1 standard test path; the remaining marker warnings are pre-existing and non-blocking for this review fix.
- Handoff / next checkpoint: corrected the earlier false pass narrative, restored valid Phase 1 handoff status, and kept `P2.0` as the next checkpoint in a new thread if accepted.

**Correction — 2026-03-12 false pass statement withdrawn**

- The earlier "runtime verification closed" note was not valid for the required Phase 1 standard path because it depended on ad hoc `conda run ... python -c "import networkx, pytest, ... pytest.main(...)"` commands that pre-imported `networkx` and bypassed the actual `tests/conftest.py` startup behavior.
- The authoritative Phase 1 verification record is now the required standard runner path above, executed via `scripts/run_tests.sh`.

## Phase 2 — Additive Family/Template Role Metadata

- Scope anchor: `src/mofbuilder/core/moftoplibrary.py`, matching tests, optional
  metadata fixture
- Must preserve: current `MOF_topology_dict` behavior for single-role families
- Must not yet: refactor builder/runtime or support multiple metadata schemas

### Checkpoint P2.0 — before coding

- Date: 2026-03-12
- Thread / branch: `codex_record`
- Status: complete
- Goal: add an additive way to describe multi-role families without breaking the existing `MOF_topology_dict` contract
- Phase gate checked against `PLANS.md`: yes; Phase 2 remains limited to `src/mofbuilder/core/moftoplibrary.py`, `tests/test_core_moftoplibrary.py`, and optionally one new metadata fixture under `tests/database/`, with builder/runtime consumption still out of scope.
- Files changed: `WORKLOG.md`, `STATUS.md`
- Tests added: none
- Tests run: none
- Decisions: recorded the Phase 2 Phase Contract under `P2.0`; kept the phase limited to one additive sidecar metadata source, one normalized in-memory metadata shape, and passive metadata loading/accessors only.
- Conflicts / blockers: none
- Handoff / next checkpoint: `P2.1` — implementation

**Phase Contract**

- Phase name: `Phase 2 — Additive Family/Template Role Metadata`

**Goal**
- Add an additive way to describe multi-role families without breaking the existing `MOF_topology_dict` contract.

**Scope**
- Keep `MOF_topology_dict` readable as-is for single-role families.
- Introduce optional role metadata as a sidecar mechanism instead of replacing the current table immediately.
- Expose role metadata in the canonical normalized form, but only as passive metadata accessors.
- Choose exactly one additive metadata source for this phase and document it in code/tests; do not leave the schema open-ended across threads.
- Prefer an additive sidecar over breaking the current `"MOF node_connectivity metal linker_topic topology"` schema early.
- Keep this phase limited to metadata schema and passive loading only; it does not own builder/runtime consumption.
- Choose exactly one normalized in-memory metadata shape for multi-role families; adapters may parse raw metadata sources, but downstream code must receive one stable normalized shape.

**Allowed Files**
- `src/mofbuilder/core/moftoplibrary.py`
- `tests/test_core_moftoplibrary.py`
- optionally one new metadata fixture under `tests/database/`
- `WORKLOG.md` for required phase logging only
- `STATUS.md` for phase/checkpoint/status updates only

**Forbidden Files**
- All files outside the allowed list are out of scope for Phase 2.
- Explicitly forbidden: `src/mofbuilder/core/builder.py`, `src/mofbuilder/core/net.py`, `src/mofbuilder/io/cif_reader.py`, `src/mofbuilder/core/optimizer.py`, `src/mofbuilder/core/supercell.py`, `src/mofbuilder/core/write.py`, `src/mofbuilder/core/defects.py`, `src/mofbuilder/core/framework.py`, `src/mofbuilder/md/`, `database/`, `PLANS.md`, `ARCHITECTURE.md`, `AGENTS.md`, and `CODEX_CONTEXT.md`.

**Architecture Invariants**
- Preserve the locked pipeline: `MofTopLibrary.fetch(...)` -> `FrameNet.create_net(...)` -> `MetalOrganicFrameworkBuilder.load_framework()` -> `MetalOrganicFrameworkBuilder.optimize_framework()` -> `MetalOrganicFrameworkBuilder.make_supercell()` -> `MetalOrganicFrameworkBuilder.build()`.
- Preserve graph states `G`, `sG`, `superG`, `eG`, and `cleaved_eG`.
- Do not rename, reorder, merge, or add pipeline stages.
- Keep responsibilities fixed: `MofTopLibrary` owns family/template metadata loading and normalization into passive role metadata structures; `FrameNet` still owns topology graph construction and topology role annotation; `MetalOrganicFrameworkBuilder` still owns runtime role registries.
- Preserve the graph-centered architecture, existing public APIs, and the single-role template path as the default/base case.
- Do not move heavy imports into package `__init__` files or `cli.py`.

**Role Model Invariants**
- Runtime topology role identifiers remain graph-stored only: `FrameNet.G.nodes[n]["node_role_id"]` and `FrameNet.G.edges[e]["edge_role_id"]`.
- Runtime fragment registries remain `node_role_registry` and `edge_role_registry`.
- Phase 2 metadata must normalize toward the canonical role model without creating competing runtime role stores or local role maps.
- Role identifiers must not be inferred from chemistry.
- Single-role normalization remains the backward-compatible base case: families without role metadata must still map cleanly to `node:default` and `edge:default` semantics.

**Required Tests**
- `scripts/run_tests.sh tests/test_core_moftoplibrary.py`
- Regression tests proving families with no role metadata still resolve exactly as before.
- Metadata tests proving a multi-role family can be loaded into the canonical normalized role model without invoking build/runtime code.
- If a new metadata fixture is needed, keep it under `tests/database/` and limit it to Phase 2 metadata coverage.

**Success Criteria**
- `MofTopLibrary` can return legacy scalar metadata plus optional role metadata.
- Single-role families require no new metadata.
- Exactly one additive metadata source is implemented for this phase.
- Exactly one normalized in-memory metadata shape is exposed downstream from `MofTopLibrary`.
- Builder/runtime fragment loading, optimizer inputs, and other later-phase consumers remain unchanged.

**Stop Rule**
- Stop immediately if Phase 2 work requires editing any forbidden file or changing module responsibilities.
- Stop immediately if satisfying the phase requires refactoring `builder.py`, changing fragment loading, changing optimizer inputs, or modifying any runtime consumer outside `MofTopLibrary`.
- Stop immediately if the work would support multiple metadata formats in parallel "for flexibility" or back-propagate runtime cache or fragment-loading concerns into the metadata schema.
- Stop immediately if the locked pipeline, graph-state names, public APIs, or canonical role-model ownership rules would need to change.
- If any schema, runtime, or invariant conflict is discovered, record it first in `WORKLOG.md` and `STATUS.md`, then stop before revising `PLANS.md`.

### Checkpoint P2.1 — implementation

- Date: 2026-03-12
- Thread / branch: `codex_record`
- Status: complete
- Goal: add one additive sidecar metadata source in `MofTopLibrary` and expose normalized passive role metadata without changing legacy scalar family resolution
- Phase gate checked against `PLANS.md`: yes; implementation stayed inside `src/mofbuilder/core/moftoplibrary.py`, `tests/test_core_moftoplibrary.py`, `WORKLOG.md`, and `STATUS.md` only.
- Files changed: `src/mofbuilder/core/moftoplibrary.py`, `tests/test_core_moftoplibrary.py`, `WORKLOG.md`, `STATUS.md`
- Tests added: `test_read_mof_top_dict_loads_optional_role_metadata_sidecar`, `test_fetch_preserves_legacy_scalars_and_selected_role_metadata`
- Tests run: `scripts/run_tests.sh tests/test_core_moftoplibrary.py` (passed: 5 tests)
- Decisions: implemented exactly one additive metadata source as `MOF_topology_role_metadata.json` next to `MOF_topology_dict`; normalized sidecar data into one passive in-memory shape under per-family `role_metadata` with `schema`, `node_roles`, and `edge_roles`; kept families without sidecar metadata on the legacy path with unchanged scalar fields and `role_metadata=None`.
- Conflicts / blockers: none
- Handoff / next checkpoint: `P2.2` — handoff

### Checkpoint P2.2 — handoff

- Date: 2026-03-12
- Thread / branch: `codex_record`
- Status: complete
- Goal: confirm metadata schema is stable enough for builder normalization
- Phase gate checked against `PLANS.md`: yes; Phase 2 remained metadata-only and did not modify `builder.py`, topology parsing, optimizer, supercell, writer, defects, framework, MD modules, or bundled database files.
- Files changed: `src/mofbuilder/core/moftoplibrary.py`, `tests/test_core_moftoplibrary.py`, `WORKLOG.md`, `STATUS.md`
- Tests added: `test_read_mof_top_dict_loads_optional_role_metadata_sidecar`, `test_fetch_preserves_legacy_scalars_and_selected_role_metadata`
- Tests run: `scripts/run_tests.sh tests/test_core_moftoplibrary.py` (passed: 5 tests)
- Decisions: Phase 2 now exposes legacy scalar family metadata plus optional passive role metadata from one JSON sidecar format; the normalized role shape is stable for downstream consumption and does not introduce competing runtime role stores or builder/runtime behavior changes.
- Conflicts / blockers: none
- Handoff / next checkpoint: Phase 2 handoff complete; next checkpoint is `P3.0` in a new thread after reviewer acceptance.

## Phase 3 — Builder Input Normalization and Role Registries

- Scope anchor: `src/mofbuilder/core/builder.py`, matching tests
- Must preserve: current scalar builder inputs as the single-role shorthand
- Must not yet: redesign Phase 2 metadata or modify optimizer/supercell/writer

### Checkpoint P3.0 — before coding

- Date: 2026-03-12
- Thread / branch: `codex_record`
- Status: complete
- Goal: normalize scalar inputs into one-entry role registries
- Phase gate checked against `PLANS.md`: yes; Phase 3 remains limited to `src/mofbuilder/core/builder.py`, `tests/test_core_builder.py`, and planner logging in `WORKLOG.md` / `STATUS.md`, with metadata-schema changes and optimizer/supercell/writer/defects/framework/MD changes out of scope.
- Files changed: `WORKLOG.md`, `STATUS.md`
- Tests added: none
- Tests run: none
- Decisions: recorded the Phase 3 Phase Contract under `P3.0`; kept the execution boundary at builder-owned normalization and runtime role registries only; preserved legacy scalar builder inputs as the required single-role shorthand and left the Phase 2 metadata shape untouched.
- Conflicts / blockers: none
- Handoff / next checkpoint: `P3.1` — implementation

**Phase Contract**

- Phase name: `Phase 3 — Builder Input Normalization and Role Registries`

**Goal**
- Make the builder internally role-aware while preserving current public scalar inputs for the single-role case.

**Scope**
- Add normalized internal structures such as:
  - node-role specs
  - edge-role specs
  - role -> fragment cache/data maps
- Keep existing fields such as `node_metal`, `linker_smiles`, `linker_xyzfile`, `linker_molecule`, and `termination_name` as shorthand for the default single-role path.
- Preserve `load_framework() -> optimize_framework() -> make_supercell()`.
- Do not generalize `FrameNode` and `FrameLinker` into multi-role managers if they can stay single-fragment processors instantiated per role.
- Do not redesign metadata schema here; consume the canonical metadata produced by Phase 2.

**Allowed Files**
- `src/mofbuilder/core/builder.py`
- `tests/test_core_builder.py`
- `WORKLOG.md` for required phase logging only
- `STATUS.md` for phase/checkpoint/status updates only

**Forbidden Files**
- All files outside the allowed list are out of scope for Phase 3.
- Explicitly forbidden: `src/mofbuilder/core/moftoplibrary.py`, `src/mofbuilder/core/net.py`, `src/mofbuilder/io/cif_reader.py`, `src/mofbuilder/core/node.py`, `src/mofbuilder/core/linker.py`, `src/mofbuilder/core/termination.py`, `src/mofbuilder/core/optimizer.py`, `src/mofbuilder/core/supercell.py`, `src/mofbuilder/core/write.py`, `src/mofbuilder/core/defects.py`, `src/mofbuilder/core/framework.py`, `src/mofbuilder/md/`, `database/`, `PLANS.md`, `ARCHITECTURE.md`, `AGENTS.md`, and `CODEX_CONTEXT.md`.

**Architecture Invariants**
- Preserve the locked pipeline: `MofTopLibrary.fetch(...)` -> `FrameNet.create_net(...)` -> `MetalOrganicFrameworkBuilder.load_framework()` -> `MetalOrganicFrameworkBuilder.optimize_framework()` -> `MetalOrganicFrameworkBuilder.make_supercell()` -> `MetalOrganicFrameworkBuilder.build()`.
- Preserve graph states `G`, `sG`, `superG`, `eG`, and `cleaved_eG`.
- Do not rename these methods, reorder pipeline steps, merge pipeline stages, introduce new top-level pipeline stages, or move responsibilities between modules.
- Keep responsibilities fixed: `FrameNet` owns topology graph construction and topology role annotation; `MofTopLibrary` owns topology family metadata; `MetalOrganicFrameworkBuilder` owns fragment normalization and runtime role registries; `Optimizer` owns node/linker placement; `Supercell` owns supercell expansion; `Writer / Framework` own merged structure output; `Defects` own defect operations; `MD modules` own simulation preparation.
- Preserve the graph-centered architecture, staged build pipeline, topology-driven connectivity, and the rule that atomic coordinates are derived from optimized graph state.
- Keep single-role builds as the fast path and avoid introducing significant overhead when only one role exists.
- Preserve current public scalar builder inputs as the single-role shorthand and keep the existing `load_framework() -> optimize_framework() -> make_supercell()` orchestration intact.

**Role Model Invariants**
- Role identifiers are the only topology classification mechanism.
- `node_role_id` must live on `FrameNet.G.nodes[n]["node_role_id"]`.
- `edge_role_id` must live on `FrameNet.G.edges[e]["edge_role_id"]`.
- Role identifiers must never be recomputed by downstream modules, replaced by local role maps, or inferred from chemistry.
- Fragment registries must remain `node_role_registry` and `edge_role_registry`.
- Modules consuming roles must resolve fragment payloads through these registries.
- `MetalOrganicFrameworkBuilder` may normalize legacy scalar inputs into registries, but it must consume the canonical metadata produced by Phase 2 rather than redesigning metadata shape or ownership.
- Single-role normalization remains the backward-compatible base case: scalar builder inputs must normalize to one-entry `node:default` / `edge:default` registries.

**Required Tests**
- `scripts/run_tests.sh tests/test_core_builder.py`
- Regression tests proving the current single-role builder inputs still behave the same.
- Explicit normalization tests showing scalar builder inputs become one-entry `node:default` / `edge:default` registries.

**Success Criteria**
- The builder can carry role-aware fragment specs internally.
- Single-role builds still use the current scalar attributes unchanged.
- Legacy scalar inputs normalize into builder-owned runtime role registries without changing public shorthand behavior.
- The builder consumes the canonical metadata produced by Phase 2 without redesigning that metadata format.
- No optimizer, supercell, writer, defects, framework, MD, topology-parsing, or bundled-database change is required to complete Phase 3.

**Stop Rule**
- Stop immediately if Phase 3 work requires editing any forbidden file or changing module responsibilities.
- Stop immediately if satisfying the phase requires changing the Phase 2 metadata format, modifying optimizer placement logic, modifying supercell/writer/defects/framework/MD code, or generalizing helper objects outside builder-owned normalization and registry setup.
- Stop immediately if the work would remove or rename current scalar builder inputs, redesign role ids or registry ownership, infer roles from chemistry, or introduce competing local role maps.
- Stop immediately if the locked pipeline, graph-state names, public APIs, or single-role fast path would need to change.
- If any schema, runtime, or invariant conflict is discovered, record it first in `WORKLOG.md` and `STATUS.md`, then stop before revising `PLANS.md`.

### Checkpoint P3.1 — implementation

- Date: 2026-03-12
- Thread / branch: `codex_record`
- Status: complete
- Goal: normalize scalar builder inputs into builder-owned role specs and runtime registries while preserving the existing single-role build path
- Phase gate checked against `PLANS.md`: yes; implementation stayed inside `src/mofbuilder/core/builder.py`, `tests/test_core_builder.py`, `WORKLOG.md`, and `STATUS.md` only.
- Files changed: `src/mofbuilder/core/builder.py`, `tests/test_core_builder.py`, `WORKLOG.md`, `STATUS.md`
- Tests added: `test_initialize_role_registries_normalizes_scalar_inputs_to_default_roles`, `test_role_registries_consume_phase_two_metadata_without_local_role_maps`
- Tests run: `scripts/run_tests.sh tests/test_core_builder.py` (passed: 3 tests; existing `pytest.mark.core` warnings only)
- Decisions: added builder-owned `role_metadata`, `node_role_specs`, `edge_role_specs`, `node_role_registry`, and `edge_role_registry`; normalized families without metadata to one-entry `node:default` / `edge:default` registries; consumed Phase 2 canonical role metadata verbatim when present; preserved the scalar fast path by populating registry fragment data only for roles matching the current globally selected node/linker connectivity in this phase.
- Conflicts / blockers: none
- Handoff / next checkpoint: `P3.2` — handoff

### Checkpoint P3.2 — handoff

- Date: 2026-03-12
- Thread / branch: `codex_record`
- Status: complete
- Goal: confirm builder-owned registries are ready for optimizer consumption
- Phase gate checked against `PLANS.md`: yes; Phase 3 remained limited to builder-owned normalization and matching tests, with no optimizer/supercell/writer/defects/framework/MD/topology/database changes.
- Files changed: `src/mofbuilder/core/builder.py`, `tests/test_core_builder.py`, `WORKLOG.md`, `STATUS.md`
- Tests added: `test_initialize_role_registries_normalizes_scalar_inputs_to_default_roles`, `test_role_registries_consume_phase_two_metadata_without_local_role_maps`
- Tests run: `scripts/run_tests.sh tests/test_core_builder.py` (passed: 3 tests; existing `pytest.mark.core` warnings only)
- Decisions: Phase 3 now gives `MetalOrganicFrameworkBuilder` canonical builder-owned role specs and registries derived from Phase 2 metadata or default-role normalization, while keeping current scalar inputs and the single-role build orchestration unchanged; registry payloads are ready for later optimizer consumption without introducing competing local role maps.
- Conflicts / blockers: none
- Handoff / next checkpoint: Phase 3 handoff complete; next checkpoint is `P4.0` in a new thread after reviewer acceptance.

**Correction — 2026-03-12 Phase 3 reviewer request preflight**

- Scope: add one narrow regression test in `tests/test_core_builder.py` that calls the real `load_framework()` single-role path and update `WORKLOG.md` / `STATUS.md` only.
- Invariants: keep Phase 3 inside the builder-test boundary, preserve the scalar single-role shorthand, keep `node:default` / `edge:default` as the only single-role registry entries, and avoid source edits unless the test reveals a real builder defect.
- Out-of-scope modules: all source files, including `src/mofbuilder/core/builder.py`, remain unchanged unless the requested regression test exposes a real Phase 3 defect; all Phase 3 forbidden modules remain out of scope.
- Tests run: none
- Decisions: exercise the real builder code by stubbing only boundary calls at `MofTopLibrary.fetch(...)`, `FrameNet.create_net(...)`, `FrameLinker.create(...)`, `FrameNode.create(...)`, and `fetch_pdbfile(...)`.
- Conflicts / blockers: none
- Handoff / next checkpoint: add the regression test, run `scripts/run_tests.sh tests/test_core_builder.py`, then record the review-fix result.

**Correction — 2026-03-12 Phase 3 reviewer request result**

- Files changed: `tests/test_core_builder.py`, `WORKLOG.md`, `STATUS.md`
- Tests added: `test_load_framework_single_role_keeps_scalar_state_and_populates_default_role_registries`
- Tests run: `scripts/run_tests.sh tests/test_core_builder.py` (passed: 4 tests; 4 existing `PytestUnknownMarkWarning` warnings for `pytest.mark.core`)
- Decisions: added one narrow regression that calls the real `load_framework()` path while stubbing only external file/dependency boundaries; confirmed the single-role scalar builder state still populates the legacy fields and that `node_role_registry` / `edge_role_registry` each contain exactly one default-role entry with the `_update_node_role_registry_data()` / `_update_edge_role_registry_data()` payloads reflected there.
- Conflicts / blockers: none; the reviewer request did not expose a Phase 3 builder defect.
- Handoff / next checkpoint: Phase 3 handoff remains complete with the requested regression coverage added; next checkpoint remains `P4.0` in a new thread after reviewer acceptance.

## Phase 4 — Role-Aware Optimizer Inputs

- Scope anchor: `src/mofbuilder/core/optimizer.py`,
  `src/mofbuilder/utils/geometry.py`, matching tests
- Must preserve: current single-role numerical path
- Must not yet: refactor supercell, writer, defects, or MD layers

### Checkpoint P4.0 — before coding

- Date: 2026-03-12
- Thread / branch: `codex_record`
- Status: contract generated
- Goal: consume graph role ids plus builder registries in optimizer logic
- Phase gate checked against `PLANS.md`: yes; Phase 4 remains limited to `src/mofbuilder/core/optimizer.py`, `src/mofbuilder/utils/geometry.py`, `tests/test_core_optimizer.py`, and planner logging in `WORKLOG.md` / `STATUS.md`, with supercell, writer, defects, framework, MD, builder API, and role-schema redesign work out of scope.
- Files changed: `WORKLOG.md`, `STATUS.md`
- Tests added: none
- Tests run: none
- Decisions: recorded the Phase 4 Phase Contract under `P4.0`; kept the execution boundary at optimizer consumption of graph role ids plus builder registries only; preserved the single-role numerical path and the current ditopic vs multitopic branch structure as locked Phase 4 constraints.
- Conflicts / blockers: none
- Handoff / next checkpoint: `P4.1` — implementation

**Phase Contract**

- Phase name: `Phase 4 — Role-Aware Optimizer Inputs`

**Goal**
- Let placement/optimization consume per-role node and edge fragments.

**Scope**
- Replace the assumption of one global node fragment and one global edge fragment with role-based lookups.
- Keep the current ditopic vs multitopic branch structure unless there is a clear need to refactor it later.
- Preserve the single-role path as a fast/simple path.
- Keep sorted nodes/edges in place, but make role annotations drive which fragment data is selected for each node/edge.
- Keep legacy scalar fields working by normalizing into one-item registries.

**Allowed Files**
- `src/mofbuilder/core/optimizer.py`
- `src/mofbuilder/utils/geometry.py`
- `tests/test_core_optimizer.py`
- `WORKLOG.md` for required phase logging only
- `STATUS.md` for phase/checkpoint/status updates only

**Forbidden Files**
- All files outside the allowed list are out of scope for Phase 4.
- Explicitly forbidden: `src/mofbuilder/core/builder.py`, `src/mofbuilder/core/moftoplibrary.py`, `src/mofbuilder/core/net.py`, `src/mofbuilder/io/cif_reader.py`, `src/mofbuilder/core/node.py`, `src/mofbuilder/core/linker.py`, `src/mofbuilder/core/termination.py`, `src/mofbuilder/core/supercell.py`, `src/mofbuilder/core/write.py`, `src/mofbuilder/core/defects.py`, `src/mofbuilder/core/framework.py`, `src/mofbuilder/md/`, `database/`, `PLANS.md`, `ARCHITECTURE.md`, `AGENTS.md`, and `CODEX_CONTEXT.md`.

**Architecture Invariants**
- Preserve the locked pipeline: `MofTopLibrary.fetch(...)` -> `FrameNet.create_net(...)` -> `MetalOrganicFrameworkBuilder.load_framework()` -> `MetalOrganicFrameworkBuilder.optimize_framework()` -> `MetalOrganicFrameworkBuilder.make_supercell()` -> `MetalOrganicFrameworkBuilder.build()`.
- Preserve graph states `G`, `sG`, `superG`, `eG`, and `cleaved_eG`.
- Do not rename these methods, reorder pipeline steps, merge pipeline stages, introduce new top-level pipeline stages, or move responsibilities between modules.
- Keep responsibilities fixed: `FrameNet` owns topology graph construction and topology role annotation; `MofTopLibrary` owns topology family metadata; `MetalOrganicFrameworkBuilder` owns fragment normalization and runtime role registries; `Optimizer` owns node/linker placement; `Supercell` owns supercell expansion; `Writer / Framework` own merged structure output; `Defects` own defect operations; `MD modules` own simulation preparation.
- Preserve the graph-centered architecture, staged build pipeline, topology-driven connectivity, and the rule that atomic coordinates are derived from optimized graph state.
- Preserve the current single-role numerical path and keep the single-role case as the fast/simple path.
- Keep the current ditopic vs multitopic branch structure unless a real Phase 4 conflict is recorded first.

**Role Model Invariants**
- Role identifiers are the only topology classification mechanism.
- `node_role_id` must live on `FrameNet.G.nodes[n]["node_role_id"]`.
- `edge_role_id` must live on `FrameNet.G.edges[e]["edge_role_id"]`.
- Role identifiers must never be recomputed by downstream modules, replaced by local role maps, or inferred from chemistry.
- Fragment registries must remain `node_role_registry` and `edge_role_registry`.
- Optimizer logic must resolve fragment payloads through these registries rather than introducing competing role stores.
- Phase 4 must consume the role ids and registry shapes established in Phases 1-3 without redesigning them.
- Single-role normalization remains the backward-compatible base case: the optimizer must continue to work when builder inputs normalize to one-entry `node:default` / `edge:default` registries.

**Required Tests**
- `scripts/run_tests.sh tests/test_core_optimizer.py`
- Regression tests proving current single-role optimizer behavior is unchanged.
- One minimal heterogeneous multi-role runtime test at the optimizer boundary only; it must not require full writer or MD success.

**Success Criteria**
- The optimizer can place structures correctly when different node/edge roles use different fragment payloads.
- Single-role numerical behavior remains unchanged.
- Optimizer fragment selection is driven by graph role ids plus builder-owned registries rather than one global node payload and one global edge payload.
- No supercell, edge-graph, writer, defects, framework, MD, public builder API, or role-schema redesign work is required to complete Phase 4.

**Stop Rule**
- Stop immediately if Phase 4 work requires editing any forbidden file or changing module responsibilities.
- Stop immediately if satisfying the phase requires refactoring supercell or edge-graph logic, generalizing writer/defects/MD behavior, expanding the public builder API surface, or redesigning role ids or registry shapes introduced in Phases 1-3.
- Stop immediately if the locked pipeline, graph-state names, or single-role numerical path would need to change.
- If any schema, runtime, or invariant conflict is discovered, record it first in `WORKLOG.md` and `STATUS.md`, then stop before revising `PLANS.md`.

### Checkpoint P4.1 — implementation

- Date: 2026-03-12
- Thread / branch: `codex_record`
- Status: complete
- Goal: make optimizer resolve per-role node and edge fragment payloads while preserving the single-role scalar path
- Phase gate checked against `PLANS.md`: yes; implementation stayed inside `src/mofbuilder/core/optimizer.py`, `tests/test_core_optimizer.py`, `WORKLOG.md`, and `STATUS.md` only, with no builder, supercell, writer, defects, framework, or MD edits.
- Files changed: `src/mofbuilder/core/optimizer.py`, `tests/test_core_optimizer.py`, `WORKLOG.md`, `STATUS.md`
- Tests added: `test_prepare_role_fragment_payloads_keeps_single_role_scalar_fallback`, `test_role_aware_optimizer_uses_role_registries_for_grouping_and_edge_payloads`
- Tests run: `scripts/run_tests.sh tests/test_core_optimizer.py` (passed: 6 tests)
- Decisions: added optional `node_role_registry` / `edge_role_registry` payload resolution directly inside `NetOptimizer` with legacy scalar `V_*` / `E_*` / `EC_*` fallback intact; made target edge lengths role-aware per edge without changing the optimizer stage boundary; split optimizer rotation groups by `node_role_id` when nodes share the same topology pname but carry different role payloads; kept the existing ditopic vs multitopic branch structure and left `src/mofbuilder/utils/geometry.py` unchanged because no geometry helper rewrite was required.
- Conflicts / blockers: none
- Handoff / next checkpoint: `P4.2` — handoff

### Checkpoint P4.2 — handoff

- Date: 2026-03-12
- Thread / branch: `codex_record`
- Status: complete
- Goal: confirm optimizer supports minimal heterogeneous runtime behavior
- Phase gate checked against `PLANS.md`: yes; Phase 4 remained limited to optimizer/test/log files and did not modify builder wiring, role metadata, topology parsing, supercell, writer, defects, framework, or MD modules.
- Files changed: `src/mofbuilder/core/optimizer.py`, `tests/test_core_optimizer.py`, `WORKLOG.md`, `STATUS.md`
- Tests added: `test_prepare_role_fragment_payloads_keeps_single_role_scalar_fallback`, `test_role_aware_optimizer_uses_role_registries_for_grouping_and_edge_payloads`
- Tests run: `scripts/run_tests.sh tests/test_core_optimizer.py` (passed: 6 tests)
- Decisions: Phase 4 now lets optimizer-local logic select node and edge payloads from graph role ids plus Phase 3 registry entries when they are provided, while preserving the current single-role scalar fallback path; heterogeneous optimizer coverage now proves different role payloads can drive node grouping and edge placement at the optimizer boundary without expanding the public builder API or touching later pipeline stages.
- Conflicts / blockers: none
- Handoff / next checkpoint: Phase 4 handoff complete; next checkpoint is `P5.0` in a new thread after reviewer acceptance.

**Correction — 2026-03-13 Phase 4 review-fix reopen (before coding)**

- Scope: reopen Phase 4 only for the missing runtime handoff from `MetalOrganicFrameworkBuilder.optimize_framework()` into `NetOptimizer`; allow the minimal `builder.py` edit required to pass `self.node_role_registry` and `self.edge_role_registry`, add one narrow integration regression that asserts the handoff, and update `WORKLOG.md` / `STATUS.md` only.
- Invariants: preserve the locked pipeline `load_framework() -> optimize_framework() -> make_supercell()`, keep responsibilities fixed with the builder only supplying existing registries and the optimizer only consuming them, preserve current optimizer role-resolution logic and the single-role fallback path, and do not redesign builder architecture or role metadata.
- Out-of-scope modules: `src/mofbuilder/core/optimizer.py`, `src/mofbuilder/utils/geometry.py`, supercell, writer, defects, framework, MD, metadata, topology parsing, database files, and all public API redesign remain out of scope unless the narrow handoff test exposes a real invariant conflict.
- Tests run: none
- Decisions: the prior Phase 4 contract was too narrow for the real runtime path because it forbade the builder wiring required to exercise the already-implemented optimizer registry support; the review-fix contract now permits only the builder-to-optimizer registry handoff and requires a direct `optimize_framework()` integration assertion rather than more optimizer-internal coverage.
- Conflicts / blockers: review failure identified a runtime integration gap, not an optimizer role-model redesign need; `optimize_framework()` currently does not assign builder-owned registries to the `NetOptimizer` instance on the real path.
- Handoff / next checkpoint: implement the minimal builder wiring and the narrow integration regression, then reopen `P4.2` for handoff with actual runtime-path verification.

**Phase 4 Contract Addendum — Review-Fix Scope**

- Goal: complete Phase 4 by wiring the existing builder-owned role registries into the real optimizer runtime path without changing the locked pipeline or expanding the phase beyond optimizer inputs.
- Allowed Files: `src/mofbuilder/core/builder.py`, `tests/test_core_builder.py`, `WORKLOG.md`, `STATUS.md`.
- Forbidden Files: all other files remain governed by the original `P4.0` contract and stay out of scope for this review fix; do not redesign `src/mofbuilder/core/optimizer.py` or reopen later-phase modules unless a real conflict is first recorded.
- Required Change: `MetalOrganicFrameworkBuilder.optimize_framework()` may be modified only as needed to pass `self.node_role_registry` and `self.edge_role_registry` to the `NetOptimizer` instance used on the runtime path.
- Required Tests: `scripts/run_tests.sh tests/test_core_builder.py`; add one narrow integration test that exercises `optimize_framework()` and asserts the `NetOptimizer` instance receives the builder's `node_role_registry` and `edge_role_registry`.
- Success Criteria: the real `optimize_framework()` path hands the existing builder registries to `NetOptimizer`; no pipeline reordering, API expansion, role-registry redesign, or optimizer algorithm rewrite is introduced; the new integration test proves the runtime handoff directly.
- Stop Rule: stop immediately if satisfying the review fix requires changes beyond the minimal builder handoff and narrow test boundary, changes module responsibilities, or requires redesigning optimizer behavior instead of supplying its existing inputs.

**Correction — 2026-03-13 Phase 4 review-fix result**

- Files changed: `src/mofbuilder/core/builder.py`, `tests/test_core_builder.py`, `WORKLOG.md`, `STATUS.md`
- Tests added: `test_optimize_framework_passes_role_registries_to_optimizer`
- Tests run: `scripts/run_tests.sh tests/test_core_builder.py` (passed: 5 tests; 5 existing `PytestUnknownMarkWarning` warnings for `pytest.mark.core`); `scripts/run_tests.sh tests/test_core_optimizer.py` (passed: 6 tests)
- Decisions: kept the runtime fix inside the review-fix addendum by assigning `self.node_role_registry` and `self.edge_role_registry` onto the existing `NetOptimizer` instance at the start of `optimize_framework()`; added one narrow builder integration test that exercises the real `optimize_framework()` method while stubbing only the optimizer's heavy execution points and asserting the registry handoff before both optimizer stages run.
- Conflicts / blockers: none; the missing runtime handoff is now closed without modifying optimizer logic or expanding the Phase 4 boundary.
- Handoff / next checkpoint: Phase 4 review-fix is complete; next checkpoint remains `P5.0` in a new thread after reviewer acceptance.

**Correction — 2026-03-13 Phase 4 final review-fix closure (before coding)**

- Scope: replace the remaining too-narrow builder optimizer regression with one integration-style test that runs the real `load_framework() -> optimize_framework()` path, captures the actual `NetOptimizer` instance used by the builder, and proves the optimizer receives the builder-owned default-role registries on that runtime path.
- Invariants: keep the locked pipeline unchanged, stub only external file/database/fragment boundaries, avoid manually seeding final builder state as a shortcut, keep `optimizer.py` untouched, and preserve default single-role normalization to `node:default` / `edge:default`.
- Out-of-scope modules: `src/mofbuilder/core/optimizer.py`, `tests/test_core_optimizer.py`, and every source/test file outside `tests/test_core_builder.py` plus required logging files remain forbidden unless a real invariant conflict is first recorded.
- Tests run: none
- Decisions: the remaining reviewer issue is test coverage quality, not a new runtime bug in builder or optimizer wiring; the minimal fix is to exercise real builder orchestration through stubbed `MofTopLibrary`, `FrameNet`, `FrameLinker`, and `FrameNode` boundaries and assert registry identity on the optimizer methods actually invoked by `optimize_framework()`.
- Conflicts / blockers: none
- Handoff / next checkpoint: update the builder integration regression, run `scripts/run_tests.sh tests/test_core_builder.py`, then record the final closure result.

**Correction — 2026-03-13 Phase 4 final review-fix closure result**

- Files changed: `tests/test_core_builder.py`, `WORKLOG.md`, `STATUS.md`
- Tests added: none; replaced the narrow `test_optimize_framework_passes_role_registries_to_optimizer` coverage by extending the existing single-role builder integration test to cover `load_framework() -> optimize_framework()`
- Tests run: `scripts/run_tests.sh tests/test_core_builder.py` (passed: 4 tests; 4 existing `PytestUnknownMarkWarning` warnings for `pytest.mark.core`)
- Decisions: kept the fix strictly inside the allowed review-closure boundary by using the real builder orchestration path, stubbing only external topology/database/fragment loading boundaries, and capturing the actual `builder.net_optimizer` instance through its `rotation_and_cell_optimization()` and `place_edge_in_net()` calls; the test now proves that `load_framework()` creates the default-role registries and that `optimize_framework()` hands those exact registry objects, including `node:default` and `edge:default`, to the optimizer on the runtime path.
- Conflicts / blockers: none; no `builder.py` or `optimizer.py` change was required for the final reviewer closure once the regression exercised the correct path.
- Handoff / next checkpoint: Phase 4 final review-fix closure is complete; next checkpoint remains `P5.0` in a new thread after reviewer acceptance.

## Phase 5 — Role Propagation Through Supercell and Edge Graph

- Scope anchor: `src/mofbuilder/core/supercell.py`, matching tests
- Must preserve: current `superG` / `eG` behavior for single-role builds
- Must not yet: redesign writer outputs, defects APIs, or MD handling

### Checkpoint P5.0 — before coding

- Date: 2026-03-13
- Thread / branch: `codex_record`
- Status: contract generated
- Goal: preserve role ids through `superG`, `eG`, and `cleaved_eG`
- Phase gate checked against `PLANS.md`: yes; Phase 5 remains limited to `src/mofbuilder/core/supercell.py`, `tests/test_core_supercell.py`, and planner logging in `WORKLOG.md` / `STATUS.md`, with writer, defects, framework, MD, builder, optimizer, metadata, topology-parsing, and role-schema redesign work out of scope.
- Files changed: `WORKLOG.md`, `STATUS.md`
- Tests added: none
- Tests run: none
- Decisions: recorded the Phase 5 Phase Contract under `P5.0`; kept the execution boundary at role propagation through `superG`, `eG`, and `cleaved_eG` only; preserved single-role `superG` / `eG` behavior, `matched_vnode_xind` semantics, and the lock against writer/defect/MD changes in this phase.
- Conflicts / blockers: none
- Handoff / next checkpoint: `P5.1` — implementation

**Phase Contract**

- Phase name: `Phase 5 — Role Propagation Through Supercell and Edge Graph`

**Goal**
- Keep node-role and edge-role identity alive after supercell expansion and `eG` generation.

**Scope**
- Propagate role annotations into `superG`, `eG`, and `cleaved_eG`.
- Remove assumptions that one global `xoo_dict` shape/order is valid for every node role.
- Keep `matched_vnode_xind` semantics, but make them robust to role-specific node layouts.

**Allowed Files**
- `src/mofbuilder/core/supercell.py`
- `tests/test_core_supercell.py`
- `WORKLOG.md` for required phase logging only
- `STATUS.md` for phase/checkpoint/status updates only

**Forbidden Files**
- All files outside the allowed list are out of scope for Phase 5.
- Explicitly forbidden: `src/mofbuilder/core/builder.py`, `src/mofbuilder/core/optimizer.py`, `src/mofbuilder/utils/geometry.py`, `src/mofbuilder/core/moftoplibrary.py`, `src/mofbuilder/core/net.py`, `src/mofbuilder/io/cif_reader.py`, `src/mofbuilder/core/node.py`, `src/mofbuilder/core/linker.py`, `src/mofbuilder/core/termination.py`, `src/mofbuilder/core/write.py`, `src/mofbuilder/core/defects.py`, `src/mofbuilder/core/framework.py`, `src/mofbuilder/md/`, `database/`, `PLANS.md`, `ARCHITECTURE.md`, `AGENTS.md`, and `CODEX_CONTEXT.md`.

**Architecture Invariants**
- Preserve the locked pipeline: `MofTopLibrary.fetch(...)` -> `FrameNet.create_net(...)` -> `MetalOrganicFrameworkBuilder.load_framework()` -> `MetalOrganicFrameworkBuilder.optimize_framework()` -> `MetalOrganicFrameworkBuilder.make_supercell()` -> `MetalOrganicFrameworkBuilder.build()`.
- Preserve graph states `G`, `sG`, `superG`, `eG`, and `cleaved_eG`.
- Do not rename these methods, reorder pipeline steps, merge pipeline stages, introduce new top-level pipeline stages, or move responsibilities between modules.
- Keep responsibilities fixed: `FrameNet` owns topology graph construction and topology role annotation; `MofTopLibrary` owns topology family metadata; `MetalOrganicFrameworkBuilder` owns fragment normalization and runtime role registries; `Optimizer` owns node/linker placement; `Supercell` owns supercell expansion; `Writer / Framework` own merged structure output; `Defects` own defect operations; `MD modules` own simulation preparation.
- Preserve the graph-centered architecture, staged build pipeline, topology-driven connectivity, and the rule that atomic coordinates are derived from optimized graph state.
- Keep single-role builds as the fast path and avoid introducing significant overhead when only one role exists.
- Preserve current `superG` / `eG` behavior for single-role builds and keep writer/defect/framework/MD responsibilities untouched in this phase.

**Role Model Invariants**
- Role identifiers are the only topology classification mechanism.
- `node_role_id` must live on `FrameNet.G.nodes[n]["node_role_id"]`.
- `edge_role_id` must live on `FrameNet.G.edges[e]["edge_role_id"]`.
- Role identifiers must never be recomputed by downstream modules, replaced by local role maps, or inferred from chemistry.
- Fragment registries must remain `node_role_registry` and `edge_role_registry`.
- Supercell and edge-graph code must propagate graph-stored role ids and consume builder-owned registries rather than inventing new writer-local, defect-local, or supercell-local role stores.
- `matched_vnode_xind` semantics must be preserved while becoming robust to role-specific node layouts.
- Single-role normalization remains the backward-compatible base case: Phase 5 must continue to work when builder inputs normalize to one-entry `node:default` / `edge:default` registries.

**Required Tests**
- `scripts/run_tests.sh tests/test_core_supercell.py`
- Regression tests proving single-role `superG` / `eG` behavior is unchanged.
- One minimal heterogeneous-role test proving role labels survive into `eG` and `cleaved_eG`.

**Success Criteria**
- Role metadata survives supercell and edge-graph construction.
- Single-role behavior is unchanged.
- `superG`, `eG`, and `cleaved_eG` retain the role identity needed by downstream phases.
- The code no longer assumes one global `xoo_dict` shape/order is valid for every node role.
- `matched_vnode_xind` semantics remain intact while supporting role-specific node layouts.
- No writer, defects, framework, MD, builder, optimizer, metadata, or topology-parsing changes are required to complete Phase 5.

**Stop Rule**
- Stop immediately if Phase 5 work requires editing any forbidden file or changing module responsibilities.
- Stop immediately if satisfying the phase requires redesigning writer output formats, redesigning defect APIs, generalizing MD or force-field code, or repairing earlier-phase normalization or schema issues inside this phase.
- Stop immediately if the locked pipeline, graph-state names, public APIs, or single-role `superG` / `eG` behavior would need to change.
- If any schema, runtime, or invariant conflict is discovered, record it first in `WORKLOG.md` and `STATUS.md`, then stop before revising `PLANS.md`.

### Checkpoint P5.1 — implementation

- Date: 2026-03-13
- Thread / branch: `codex_record`
- Status: complete
- Goal: propagate role ids through `superG`, `eG`, and `cleaved_eG` while making XOO matching robust to role-specific node layouts
- Phase gate checked against `PLANS.md`: yes; implementation stayed inside `src/mofbuilder/core/supercell.py`, `tests/test_core_supercell.py`, `WORKLOG.md`, and `STATUS.md` only.
- Files changed: `src/mofbuilder/core/supercell.py`, `tests/test_core_supercell.py`, `WORKLOG.md`, `STATUS.md`
- Tests added: `test_single_role_supercell_and_edgegraph_keep_role_metadata`, `test_edgegraph_preserves_roles_through_cleave_with_role_specific_xoo_layouts`
- Tests run: `scripts/run_tests.sh tests/test_core_supercell.py` (passed: 5 tests)
- Decisions: preserved graph-stored role ids by copying node and edge attributes into translated `superG` nodes/edges instead of rebuilding narrow attribute dicts; propagated `node_role_id` onto vnode entries and `edge_role_id` onto EDGE entries in `eG`; replaced the global vnode-layout assumption in XOO matching with internal per-vnode XOO lookups while keeping the legacy single-role `xoo_dict` shape for later phases; fixed the existing `remove_node_by_index()` mutation-during-iteration bug uncovered by the required Phase 5 test path.
- Conflicts / blockers: none
- Handoff / next checkpoint: `P5.2` — handoff

### Checkpoint P5.2 — handoff

- Date: 2026-03-13
- Thread / branch: `codex_record`
- Status: complete
- Goal: confirm downstream phases can consume propagated role metadata
- Phase gate checked against `PLANS.md`: yes; Phase 5 remained limited to supercell/test/log files and did not modify builder, optimizer, writer, defects, framework, MD, metadata, topology-parsing, or database modules.
- Files changed: `src/mofbuilder/core/supercell.py`, `tests/test_core_supercell.py`, `WORKLOG.md`, `STATUS.md`
- Tests added: `test_single_role_supercell_and_edgegraph_keep_role_metadata`, `test_edgegraph_preserves_roles_through_cleave_with_role_specific_xoo_layouts`
- Tests run: `scripts/run_tests.sh tests/test_core_supercell.py` (passed: 5 tests)
- Decisions: Phase 5 now keeps role metadata attached through translated `superG` nodes/edges, preserves role identity on `eG` and `cleaved_eG` nodes that represent topology vertices and linkers, and makes `matched_vnode_xind` robust to vnode-specific XOO layouts without forcing a writer/defect-facing `xoo_dict` redesign in this phase.
- Conflicts / blockers: none
- Handoff / next checkpoint: Phase 5 handoff complete; next checkpoint is `P6.0` in a new thread after reviewer acceptance.

## Phase 6 — Role-Aware Writer and Defect Metadata

- Scope anchor: `src/mofbuilder/core/write.py`,
  `src/mofbuilder/core/defects.py`, `src/mofbuilder/core/framework.py`,
  matching tests
- Must preserve: `Framework.get_merged_data()` as the sync point and current
  `remove()` / `replace()` semantics
- Must not yet: change MD contracts or invent local role identifiers

### Checkpoint P6.0 — before coding

- Date: 2026-03-13
- Thread / branch: `codex_record`
- Status: complete
- Goal: generalize merged-data and defect paths to consume role-specific node
  metadata
- Phase gate checked against `PLANS.md`: yes; Phase 6 remains limited to `src/mofbuilder/core/write.py`, `src/mofbuilder/core/defects.py`, `src/mofbuilder/core/framework.py`, `tests/test_core_write.py`, `tests/test_core_defects.py`, `tests/test_core_framework.py`, and planner logging in `WORKLOG.md` / `STATUS.md`, with MD and force-field generalization still out of scope.
- Files changed: `WORKLOG.md`, `STATUS.md`
- Tests added: none
- Tests run: none
- Decisions: recorded the Phase 6 Phase Contract under `P6.0`; kept `Framework.get_merged_data()` as the synchronization point after structural edits, preserved `remove()` / `replace()` return semantics, and kept role identity graph-stored rather than introducing writer-local or defect-local role maps.
- Conflicts / blockers: none
- Handoff / next checkpoint: `P6.1` — implementation

**Phase Contract**

- Phase name: `Phase 6 — Role-Aware Writer and Defect Metadata`

**Goal**
- Generalize merged-data assembly and defect logic to use role-specific node metadata.

**Scope**
- Replace the assumption of one global `dummy_atom_node_dict`.
- Replace the assumption of one global `xoo_dict`.
- Keep `Framework.get_merged_data()` as the synchronization point after structural edits.
- Preserve current mutation semantics:
  - `remove()` and `replace()` return new `Framework` objects.

**Allowed Files**
- `src/mofbuilder/core/write.py`
- `src/mofbuilder/core/defects.py`
- `src/mofbuilder/core/framework.py`
- `tests/test_core_write.py`
- `tests/test_core_defects.py`
- `tests/test_core_framework.py`
- `WORKLOG.md` for required phase logging only
- `STATUS.md` for phase/checkpoint/status updates only

**Forbidden Files**
- All files outside the allowed list are out of scope for Phase 6.
- Explicitly forbidden: `src/mofbuilder/core/supercell.py`, `src/mofbuilder/core/optimizer.py`, `src/mofbuilder/core/builder.py`, `src/mofbuilder/core/net.py`, `src/mofbuilder/core/moftoplibrary.py`, `src/mofbuilder/io/cif_reader.py`, `src/mofbuilder/md/`, `database/`, `PLANS.md`, `ARCHITECTURE.md`, `AGENTS.md`, `CODEX_CONTEXT.md`, `README.md`, and `docs/`.
- Must not yet change force-field generation contracts, generalize MD topology assembly, refactor `Framework` public methods beyond role-aware internal plumbing, or invent writer-local or defect-local role identifiers.

**Architecture Invariants**
- Preserve the locked pipeline: `MofTopLibrary.fetch(...)` -> `FrameNet.create_net(...)` -> `MetalOrganicFrameworkBuilder.load_framework()` -> `MetalOrganicFrameworkBuilder.optimize_framework()` -> `MetalOrganicFrameworkBuilder.make_supercell()` -> `MetalOrganicFrameworkBuilder.build()`.
- Preserve graph states `G`, `sG`, `superG`, `eG`, and `cleaved_eG`.
- Do not rename, reorder, merge, or add pipeline stages.
- Keep responsibilities fixed: writer/framework own merged structure output and post-build synchronization, defects own defect operations, supercell remains responsible for propagated role metadata, and MD modules remain unchanged in this phase.
- Keep `Framework.get_merged_data()` as the synchronization point after structural edits.
- Preserve current object-lifetime semantics: `build()` returns `builder.framework`; `Framework.remove()` and `Framework.replace()` return new `Framework` objects; `Framework.solvate()`, `generate_linker_forcefield()`, `md_prepare()`, and `show()` mutate the current `Framework`.
- Preserve the graph-centered architecture, existing geometry/data conventions, single-role behavior as the default/base case, and current public builder/framework/package interfaces.

**Role Model Invariants**
- Role identifiers remain the only topology classification mechanism.
- `node_role_id` must continue to live on `FrameNet.G.nodes[n]["node_role_id"]`.
- `edge_role_id` must continue to live on `FrameNet.G.edges[e]["edge_role_id"]`.
- Downstream writer/defect code must consume graph-stored role ids and resolve role-specific metadata through the established registries and propagated graph metadata; it must not invent local role maps or recompute role ids.
- Fragment registries remain `node_role_registry` and `edge_role_registry`.
- Role identifiers must not be inferred from chemistry or replaced by writer-local or defect-local identifiers.
- Single-role normalization remains the backward-compatible base case via `node:default` and `edge:default`.

**Required Tests**
- `scripts/run_tests.sh tests/test_core_write.py`
- `scripts/run_tests.sh tests/test_core_defects.py`
- `scripts/run_tests.sh tests/test_core_framework.py`
- Regression tests proving current single-role merged data and defect behavior are unchanged.
- One minimal heterogeneous-role test proving writer/defect code can consume role-specific node metadata after Phases 4-5 are complete.

**Success Criteria**
- Merged output and defect-handling paths work with node-role-specific metadata.
- Single-role merged output remains stable.
- `Framework.get_merged_data()` remains the synchronization point after structural edits.
- `Framework.remove()` and `Framework.replace()` keep returning new `Framework` objects.
- No force-field generation contract, MD topology assembly path, or public `Framework` API is broadened beyond role-aware internal plumbing.
- No writer-local or defect-local role identifiers are introduced.

**Stop Rule**
- Stop immediately if Phase 6 work requires editing any forbidden file or changing module responsibilities.
- Stop immediately if satisfying the phase requires changing force-field generation contracts, generalizing MD topology assembly, or repairing earlier-phase normalization/schema issues inside this phase.
- Stop immediately if the work would refactor `Framework` public methods beyond role-aware internal plumbing or invent writer-local or defect-local role identifiers.
- Stop immediately if the locked pipeline, graph-state names, public mutation semantics, or graph-stored role ownership rules would need to change.
- If any schema, runtime, or invariant conflict is discovered, record it first in `WORKLOG.md` and `STATUS.md`, then stop before revising `PLANS.md`.

### Checkpoint P6.1 — implementation

- Date: 2026-03-13
- Thread / branch: `codex_record`
- Status: complete
- Goal: generalize merged-data assembly and defect XOO handling to resolve node-role-specific metadata while preserving single-role behavior and `Framework` mutation semantics
- Phase gate checked against `PLANS.md`: yes; implementation remains limited to `src/mofbuilder/core/write.py`, `src/mofbuilder/core/defects.py`, `src/mofbuilder/core/framework.py`, `tests/test_core_write.py`, `tests/test_core_defects.py`, `tests/test_core_framework.py`, `WORKLOG.md`, and `STATUS.md`.
- Files changed: `src/mofbuilder/core/write.py`, `src/mofbuilder/core/defects.py`, `tests/test_core_write.py`, `tests/test_core_defects.py`, `tests/test_core_framework.py`, `WORKLOG.md`, `STATUS.md`
- Tests added: `test_get_merged_data_keeps_single_role_dummy_atom_behavior`, `test_writer_resolves_role_specific_dummy_and_xoo_metadata`, `test_remove_xoo_from_node_keeps_single_role_xoo_dict`, `test_make_unsaturated_vnode_xoo_dict_uses_role_specific_xoo_metadata`, `test_framework_get_merged_data_forwards_role_aware_metadata_to_writer`, `test_framework_remove_and_replace_return_new_framework_instances`
- Tests run: `scripts/run_tests.sh tests/test_core_write.py` (passed: 4 tests); `scripts/run_tests.sh tests/test_core_defects.py` (passed: 5 tests); `scripts/run_tests.sh tests/test_core_framework.py` (passed: 5 tests, 5 existing `PytestUnknownMarkWarning` warnings for `pytest.mark.core`)
- Decisions: kept the legacy scalar fast path intact in `MofWriter.get_merged_data()` by preserving the original global rename ordering whenever `dummy_atom_node_dict` is still a single-role layout dict; added role-aware resolution helpers in `write.py` and `defects.py` that read graph-stored `node_role_id` and accept either legacy scalar metadata or role-keyed metadata/registry entries without inventing new local role ids; left `src/mofbuilder/core/framework.py` source unchanged because its existing `get_merged_data()` synchronization path already forwarded `xoo_dict` and `dummy_atom_node_dict` verbatim, so only narrow plumbing/semantics tests were required there.
- Conflicts / blockers: none
- Handoff / next checkpoint: `P6.2` — handoff

### Checkpoint P6.2 — handoff

- Date: 2026-03-13
- Thread / branch: `codex_record`
- Status: complete
- Goal: confirm writer/defect paths are stable before MD generalization
- Phase gate checked against `PLANS.md`: yes; Phase 6 remained limited to writer/defects/framework allowed files and did not modify builder, optimizer, supercell, metadata, topology parsing, MD modules, database files, or locked pipeline responsibilities.
- Files changed: `src/mofbuilder/core/write.py`, `src/mofbuilder/core/defects.py`, `tests/test_core_write.py`, `tests/test_core_defects.py`, `tests/test_core_framework.py`, `WORKLOG.md`, `STATUS.md`
- Tests added: `test_get_merged_data_keeps_single_role_dummy_atom_behavior`, `test_writer_resolves_role_specific_dummy_and_xoo_metadata`, `test_remove_xoo_from_node_keeps_single_role_xoo_dict`, `test_make_unsaturated_vnode_xoo_dict_uses_role_specific_xoo_metadata`, `test_framework_get_merged_data_forwards_role_aware_metadata_to_writer`, `test_framework_remove_and_replace_return_new_framework_instances`
- Tests run: `scripts/run_tests.sh tests/test_core_write.py` (passed: 4 tests); `scripts/run_tests.sh tests/test_core_defects.py` (passed: 5 tests); `scripts/run_tests.sh tests/test_core_framework.py` (passed: 5 tests, 5 existing `PytestUnknownMarkWarning` warnings for `pytest.mark.core`)
- Decisions: Phase 6 now lets merged-output assembly and defect XOO handling resolve node-role-specific metadata from graph-stored `node_role_id` while preserving the current single-role scalar path as the base case; `Framework.get_merged_data()` remains the sync point after structural edits, and `Framework.remove()` / `Framework.replace()` continue returning new `Framework` instances.
- Conflicts / blockers: none; only pre-existing `pytest.mark.core` warning noise remains on the framework test path.
- Handoff / next checkpoint: Phase 6 handoff complete; next checkpoint is `P7.0` in a new thread after reviewer acceptance.

## Phase 7 — Multi-Edge Force-Field and Simulation-Prep Support

- Scope anchor: `src/mofbuilder/md/linkerforcefield.py`,
  `src/mofbuilder/md/gmxfilemerge.py`, `src/mofbuilder/core/framework.py`,
  matching tests
- Must preserve: current one-linker MD-prep path
- Must not yet: broaden scope into general heterogeneous force-field research

### Checkpoint P7.0 — before coding

- Date: 2026-03-13
- Thread / branch: `codex_record`
- Status: contract generated
- Goal: support multiple edge-role force-field paths at the currently supported
  level
- Phase gate checked against `PLANS.md`: yes; Phase 7 remains limited to `src/mofbuilder/md/linkerforcefield.py`, `src/mofbuilder/md/gmxfilemerge.py`, `src/mofbuilder/core/framework.py`, `tests/test_md_linkerforcefield.py`, `tests/test_md_gmxfilemerge.py`, `tests/test_core_framework.py`, and planner logging in `WORKLOG.md` / `STATUS.md`, with topology parsing, metadata schema, builder normalization, optimizer, supercell, writer, defects, bundled database, and broad force-field research work out of scope.
- Files changed: `WORKLOG.md`, `STATUS.md`
- Tests added: none
- Tests run: none
- Decisions: recorded the Phase 7 Phase Contract under `P7.0`; kept the current one-linker MD-prep path as the regression baseline, limited the phase to one force-field mapping/generation path per edge role plus multi-ITP topology merge support, and preserved the rule that this phase proves one minimal heterogeneous multi-edge path rather than broad heterogeneous chemistry support.
- Conflicts / blockers: none
- Handoff / next checkpoint: `P7.1` — implementation

**Phase Contract**

- Phase name: `Phase 7 — Multi-Edge Force-Field and Simulation-Prep Support`

**Goal**
- Let simulation prep handle more than one linker/edge role.

**Scope**
- Support one force-field mapping/generation path per edge role.
- Merge multiple linker ITP outputs into the generated topology.
- Keep the current one-linker path unchanged.
- Keep this phase limited to the currently supported MD-prep level; do not broaden the target beyond one minimal heterogeneous multi-edge path proven end to end.
- Preserve `Framework.generate_linker_forcefield()` and `Framework.md_prepare()` as the user-facing orchestration points while keeping their current mutation semantics intact.

**Allowed Files**
- `src/mofbuilder/md/linkerforcefield.py`
- `src/mofbuilder/md/gmxfilemerge.py`
- `src/mofbuilder/core/framework.py`
- `tests/test_md_linkerforcefield.py`
- `tests/test_md_gmxfilemerge.py`
- `tests/test_core_framework.py`
- `WORKLOG.md` for required phase logging only
- `STATUS.md` for phase/checkpoint/status updates only

**Forbidden Files**
- All files outside the allowed list are out of scope for Phase 7.
- Explicitly forbidden: `src/mofbuilder/core/builder.py`, `src/mofbuilder/core/moftoplibrary.py`, `src/mofbuilder/core/net.py`, `src/mofbuilder/io/cif_reader.py`, `src/mofbuilder/core/optimizer.py`, `src/mofbuilder/core/supercell.py`, `src/mofbuilder/core/write.py`, `src/mofbuilder/core/defects.py`, `src/mofbuilder/core/node.py`, `src/mofbuilder/core/linker.py`, `src/mofbuilder/core/termination.py`, `src/mofbuilder/__init__.py`, `src/mofbuilder/cli.py`, `database/`, `PLANS.md`, `ARCHITECTURE.md`, `AGENTS.md`, `CODEX_CONTEXT.md`, `README.md`, and `docs/`.
- Must not redesign topology parsing or metadata schema, refactor earlier phases "for cleanup", broaden scope into general force-field research problems, or claim broad heterogeneous chemistry support from one minimal successful path.

**Architecture Invariants**
- Preserve the locked pipeline: `MofTopLibrary.fetch(...)` -> `FrameNet.create_net(...)` -> `MetalOrganicFrameworkBuilder.load_framework()` -> `MetalOrganicFrameworkBuilder.optimize_framework()` -> `MetalOrganicFrameworkBuilder.make_supercell()` -> `MetalOrganicFrameworkBuilder.build()`.
- Preserve graph states `G`, `sG`, `superG`, `eG`, and `cleaved_eG`.
- Do not rename these methods, reorder pipeline steps, merge pipeline stages, introduce new top-level pipeline stages, or move responsibilities between modules.
- Keep responsibilities fixed: `FrameNet` owns topology graph construction and topology role annotation; `MofTopLibrary` owns topology family metadata; `MetalOrganicFrameworkBuilder` owns fragment normalization and runtime role registries; `Optimizer` owns node/linker placement; `Supercell` owns supercell expansion; `Writer / Framework` own merged structure output and user-facing orchestration; `Defects` own defect operations; `MD modules` own simulation preparation.
- Preserve current object-lifetime semantics: `build()` returns `builder.framework`; `Framework.remove()` and `Framework.replace()` return new `Framework` objects; `Framework.solvate()`, `generate_linker_forcefield()`, `md_prepare()`, and `show()` mutate the current `Framework`.
- Keep the graph-centered architecture, staged build pipeline, topology-driven connectivity, and the rule that atomic coordinates are derived from optimized graph state.
- Preserve the current one-linker MD-prep path unchanged as the single-role/base regression path for this phase.

**Role Model Invariants**
- Role identifiers are the only topology classification mechanism.
- `node_role_id` must live on `FrameNet.G.nodes[n]["node_role_id"]`.
- `edge_role_id` must live on `FrameNet.G.edges[e]["edge_role_id"]`.
- Role identifiers must never be recomputed by downstream modules, replaced by local role maps, or inferred from chemistry.
- Fragment registries must remain `node_role_registry` and `edge_role_registry`.
- MD and framework code in this phase must consume graph-stored role ids and builder-owned registries rather than inventing force-field-local or framework-local role identifiers.
- Single-role normalization remains the backward-compatible base case: the current one-linker path must continue to work when builder inputs normalize to one-entry `edge:default` registries.
- This phase may generalize force-field mapping and topology merge behavior per edge role, but it must not redefine role ownership or back-propagate new role semantics into earlier pipeline stages.

**Required Tests**
- `scripts/run_tests.sh tests/test_md_linkerforcefield.py`
- `scripts/run_tests.sh tests/test_md_gmxfilemerge.py`
- `scripts/run_tests.sh tests/test_core_framework.py`
- Regression tests proving the current single-linker MD-prep path is unchanged.
- One minimal heterogeneous multi-edge test proving topology generation and MD setup wiring can succeed at the currently supported level.

**Success Criteria**
- At least one minimal heterogeneous multi-edge case can prepare simulation files at the currently supported level.
- The current single-linker MD-prep path remains unchanged.
- Force-field mapping/generation can proceed per edge role without introducing competing role stores.
- Multiple linker ITP outputs can be merged into the generated topology on the supported path.
- `Framework` continues to expose the same user-facing MD-prep workflow while delegating the actual simulation-prep work to the existing MD modules.
- No topology-parsing, metadata-schema, builder, optimizer, supercell, writer, defects, or bundled-database changes are required to complete Phase 7.

**Stop Rule**
- Stop immediately if Phase 7 work requires editing any forbidden file or changing module responsibilities.
- Stop immediately if satisfying the phase requires redesigning topology parsing, revisiting Phase 2 metadata schema, refactoring earlier phases "for cleanup", or broadening scope into general force-field research problems.
- Stop immediately if the work would change the current one-linker MD-prep path, claim broad heterogeneous chemistry support beyond one minimal proven path, or invent local role identifiers outside the graph-plus-registry model.
- Stop immediately if the locked pipeline, graph-state names, public APIs, or current `Framework` mutation semantics would need to change.
- If any schema, runtime, or invariant conflict is discovered, record it first in `WORKLOG.md` and `STATUS.md`, then stop before revising `PLANS.md`.

### Checkpoint P7.1 — implementation

- Date: 2026-03-13
- Thread / branch: `codex_record`
- Status: complete
- Goal: support one force-field generation/mapping path per edge role while preserving the current single-linker MD-prep path
- Phase gate checked against `PLANS.md`: yes; implementation stayed inside `src/mofbuilder/core/framework.py`, `src/mofbuilder/md/gmxfilemerge.py`, `tests/test_core_framework.py`, `tests/test_md_gmxfilemerge.py`, and the required logging files only.
- Files changed: `src/mofbuilder/core/framework.py`, `src/mofbuilder/md/gmxfilemerge.py`, `tests/test_core_framework.py`, `tests/test_md_gmxfilemerge.py`, `WORKLOG.md`, `STATUS.md`
- Tests added: `test_framework_generate_linker_forcefield_keeps_single_role_path`, `test_framework_md_prepare_supports_minimal_multi_edge_roles`, `test_get_itps_from_database_copies_multiple_linker_itps`, `test_generate_top_file_writes_role_specific_linker_counts`
- Tests run: `scripts/run_tests.sh tests/test_md_linkerforcefield.py` (passed: 7 tests); `scripts/run_tests.sh tests/test_md_gmxfilemerge.py` (passed: 7 tests); `scripts/run_tests.sh tests/test_core_framework.py` (passed: 7 tests, 7 existing `PytestUnknownMarkWarning` warnings for `pytest.mark.core`)
- Decisions: kept the legacy single-linker branch intact and only activated the new path when the built edge graph exposes more than one `edge_role_id` or role-keyed linker inputs are provided; derived role grouping from graph-stored `edge_role_id` values plus the existing writer edge ordering and used optional `edge_role_registry` entries only for per-role configuration overrides; normalized multi-role MD output into deterministic per-role linker file names and residue names, then passed role-specific linker includes and molecule counts to the GROMACS merger without changing topology parsing, builder normalization, optimizer, supercell, writer, defects, or database behavior.
- Conflicts / blockers: none
- Handoff / next checkpoint: `P7.2` — handoff

### Checkpoint P7.2 — handoff

- Date: 2026-03-13
- Thread / branch: `codex_record`
- Status: complete
- Goal: confirm minimal heterogeneous multi-edge MD-prep support only
- Phase gate checked against `PLANS.md`: yes; Phase 7 remained limited to framework MD orchestration, linker force-field merge support, and matching tests, with no edits to topology parsing, metadata schema, builder, optimizer, supercell, writer, defects, or bundled database files.
- Files changed: `src/mofbuilder/core/framework.py`, `src/mofbuilder/md/gmxfilemerge.py`, `tests/test_core_framework.py`, `tests/test_md_gmxfilemerge.py`, `WORKLOG.md`, `STATUS.md`
- Tests added: `test_framework_generate_linker_forcefield_keeps_single_role_path`, `test_framework_md_prepare_supports_minimal_multi_edge_roles`, `test_get_itps_from_database_copies_multiple_linker_itps`, `test_generate_top_file_writes_role_specific_linker_counts`
- Tests run: `scripts/run_tests.sh tests/test_md_linkerforcefield.py` (passed: 7 tests); `scripts/run_tests.sh tests/test_md_gmxfilemerge.py` (passed: 7 tests); `scripts/run_tests.sh tests/test_core_framework.py` (passed: 7 tests, 7 existing `PytestUnknownMarkWarning` warnings for `pytest.mark.core`)
- Decisions: Phase 7 now supports one minimal heterogeneous multi-edge MD-prep path by generating or mapping linker force fields per graph edge role, merging multiple linker `.itp` files into the GROMACS topology, and replacing the single `EDGE` residue count with role-specific molecule counts only on the multi-role path; the current single-linker workflow and `Framework.generate_linker_forcefield()` / `Framework.md_prepare()` mutation semantics remain unchanged on the base path.
- Conflicts / blockers: none
- Handoff / next checkpoint: Phase 7 handoff complete; next checkpoint is `P8.0` in a new thread after reviewer acceptance.

### Checkpoint P7.3 — review reopen contract addendum

- Date: 2026-03-13
- Thread / branch: `codex_record`
- Status: complete
- Goal: reopen Phase 7 only far enough to close the real `MetalOrganicFrameworkBuilder.build() -> Framework -> md_prepare()` edge-role-registry handoff gap reported in review
- Phase gate checked against `PLANS.md`: yes; the Phase 7 reopen is narrowed to the locked builder-to-framework handoff seam in `src/mofbuilder/core/builder.py`, `src/mofbuilder/core/framework.py`, `tests/test_core_framework.py`, and planner logging in `WORKLOG.md` / `STATUS.md` only, with topology parsing, metadata schema, optimizer, supercell, writer, defects, database, and broader MD-module redesign still out of scope.
- Files changed: `WORKLOG.md`, `STATUS.md`
- Tests added: none
- Tests run: none
- Decisions: recorded the reviewer finding that Phase 7 framework/MD code can consume `edge_role_registry`, but the real locked pipeline does not pass `builder.edge_role_registry` into `Framework` during `MetalOrganicFrameworkBuilder.build()`; reopened Phase 7 as a minimal executor follow-up rather than a `PLANS.md` conflict because the missing handoff fits the existing builder-owned-registry and framework-consumer architecture; narrowed the reopen to one runtime integration fix plus one built-framework regression/integration test path.
- Conflicts / blockers: no plan, schema, or architecture conflict discovered; the gap is a runtime integration omission at the existing builder -> framework handoff boundary.
- Handoff / next checkpoint: `P7.4` — minimal executor fix and narrow built-framework regression verification

**Phase 7 Contract Addendum — 2026-03-13 review reopen**

- Goal: make `Framework` instances produced by `MetalOrganicFrameworkBuilder.build()` retain the builder-owned `edge_role_registry` so the existing Phase 7 MD-prep logic works on the real locked pipeline path.

- Scope:
- Add only the missing builder -> framework registry handoff in the existing `build()` flow.
- Keep `Framework.md_prepare()` consuming the handed-off registry on the built-framework path; do not add new role semantics or new orchestration entry points.
- Add one narrow regression/integration test proving the real built-framework path retains and uses `edge_role_registry`.
- Keep the manual `Framework()` setup path and the current single-role path working exactly as they already do.

- Allowed Files:
- `src/mofbuilder/core/builder.py`
- `src/mofbuilder/core/framework.py`
- `tests/test_core_framework.py`
- `WORKLOG.md`
- `STATUS.md`

- Forbidden Files:
- All files outside the allowed list remain out of scope for this Phase 7 reopen.
- Explicitly forbidden: `src/mofbuilder/md/linkerforcefield.py`, `src/mofbuilder/md/gmxfilemerge.py`, `src/mofbuilder/core/moftoplibrary.py`, `src/mofbuilder/core/net.py`, `src/mofbuilder/io/cif_reader.py`, `src/mofbuilder/core/optimizer.py`, `src/mofbuilder/core/supercell.py`, `src/mofbuilder/core/write.py`, `src/mofbuilder/core/defects.py`, `database/`, `PLANS.md`, `ARCHITECTURE.md`, `AGENTS.md`, `CODEX_CONTEXT.md`, `README.md`, and `docs/`.

- Architecture Invariants:
- Preserve the locked pipeline: `MofTopLibrary.fetch(...)` -> `FrameNet.create_net(...)` -> `MetalOrganicFrameworkBuilder.load_framework()` -> `MetalOrganicFrameworkBuilder.optimize_framework()` -> `MetalOrganicFrameworkBuilder.make_supercell()` -> `MetalOrganicFrameworkBuilder.build()`.
- Preserve graph states `G`, `sG`, `superG`, `eG`, and `cleaved_eG`.
- Do not rename methods, reorder steps, merge stages, add new top-level stages, or move responsibilities between modules.
- Keep ownership unchanged: `MetalOrganicFrameworkBuilder` remains the owner of runtime role registries; `Framework` remains the post-build consumer/orchestrator.
- Preserve current object-lifetime semantics: `build()` still returns `builder.framework`; `Framework.md_prepare()` still mutates the current `Framework`.

- Role Model Invariants:
- `edge_role_id` remains graph-stored role identity; `edge_role_registry` remains the builder-owned fragment/config registry.
- `Framework` must receive the registry from the builder handoff rather than recomputing, re-normalizing, or inferring role mappings locally.
- Single-role normalization remains the backward-compatible base case; the reopen must not change the existing one-entry `edge:default` behavior.

- Required Tests:
- `scripts/run_tests.sh tests/test_core_framework.py`
- One narrow regression/integration test proving `MetalOrganicFrameworkBuilder.build()` produces a `Framework` that retains `edge_role_registry`.
- One narrow regression/integration test proving `Framework.md_prepare()` uses that registry on the built-framework path rather than only on a manually assembled `Framework`.

- Success Criteria:
- A `Framework` returned from the real builder path retains the builder-populated `edge_role_registry`.
- The existing Phase 7 MD-prep role-aware logic works on the builder-built runtime path without manual post-construction registry injection.
- No topology, optimizer, supercell, defects, database, or broader MD-module scope is added.

- Stop Rule:
- Stop immediately if this reopen requires edits outside the allowed files, changes to module responsibilities, or any redesign of the builder/framework architecture.
- Stop immediately if satisfying the handoff requires changing the locked pipeline or inventing a framework-local registry model.
- If the handoff cannot be completed inside this boundary, record a contract conflict in `WORKLOG.md` and `STATUS.md` before proposing any `PLANS.md` revision.

## Phase 8 — Documentation and Example Sync

- Scope anchor: `README.md`, `docs/source/manual/*.md`, `ARCHITECTURE.md`,
  `CODEX_CONTEXT.md`, `AGENTS.md`
- Must preserve: implemented behavior, not aspirational behavior
- Must not yet: mix documentation updates with new algorithmic refactors

### Checkpoint P8.0 — before coding

- Date: 2026-03-13
- Thread / branch: `codex_record`
- Status: contract generated
- Goal: sync docs to the implemented multi-role model and verified limits
- Phase gate checked against `PLANS.md`: yes; Phase 8 remains limited to `README.md`, `docs/source/manual/*.md`, `ARCHITECTURE.md`, `CODEX_CONTEXT.md`, `AGENTS.md`, and planner logging in `WORKLOG.md` / `STATUS.md`, with algorithmic refactors, source-code changes, test changes, and behavior expansion out of scope.
- Files changed: `WORKLOG.md`, `STATUS.md`
- Tests added: none
- Tests run: none
- Decisions: recorded the Phase 8 Phase Contract under `P8.0`; kept the phase documentation-only, limited documentation claims to implemented and already-verified behavior, and preserved the rule that unsupported heterogeneous cases must not be documented as complete.
- Conflicts / blockers: none
- Handoff / next checkpoint: `P8.1` — implementation

**Phase Contract**

- Phase name: `Phase 8 — Documentation and Example Sync`

**Goal**
- Document the new internal model after the code stabilizes.

**Scope**
- Explain the single-role base case and the new multi-role internal model.
- Document any additive family/template metadata introduced in Phase 2.
- Add one multi-role example only after the code path is stable.
- Keep this phase limited to documentation and example synchronization only.
- Keep documentation aligned to implemented behavior and verified limits rather than aspirational behavior.

**Allowed Files**
- `README.md`
- `docs/source/manual/*.md`
- `ARCHITECTURE.md`
- `CODEX_CONTEXT.md`
- `AGENTS.md`
- `WORKLOG.md` for required phase logging only
- `STATUS.md` for phase/checkpoint/status updates only

**Forbidden Files**
- All files outside the allowed list are out of scope for Phase 8.
- Explicitly forbidden: all production source files under `src/mofbuilder/`, all tests under `tests/`, all bundled database assets under `database/`, `PLANS.md`, and any new algorithmic or runtime-behavior changes.
- Must not mix documentation work with algorithmic refactors.
- Must not document unsupported heterogeneous cases as complete.

**Architecture Invariants**
- Preserve the locked pipeline: `MofTopLibrary.fetch(...)` -> `FrameNet.create_net(...)` -> `MetalOrganicFrameworkBuilder.load_framework()` -> `MetalOrganicFrameworkBuilder.optimize_framework()` -> `MetalOrganicFrameworkBuilder.make_supercell()` -> `MetalOrganicFrameworkBuilder.build()`.
- Preserve graph states `G`, `sG`, `superG`, `eG`, and `cleaved_eG`.
- Do not rename methods, reorder pipeline steps, merge pipeline stages, introduce new top-level pipeline stages, or move responsibilities between modules.
- Keep responsibilities fixed: `FrameNet` owns topology graph construction and topology role annotation; `MofTopLibrary` owns topology family metadata; `MetalOrganicFrameworkBuilder` owns fragment normalization and runtime role registries; `Optimizer` owns node/linker placement; `Supercell` owns supercell expansion; `Writer / Framework` own merged structure output and user-facing orchestration; `Defects` own defect operations; `MD modules` own simulation preparation.
- Preserve current public APIs, object-lifetime semantics, graph-centered architecture, staged build pipeline, topology-driven connectivity, and the rule that atomic coordinates are derived from optimized graph state.
- Keep the single-role path documented as the default/base case.

**Role Model Invariants**
- Role identifiers remain the only topology classification mechanism.
- `node_role_id` must remain on `FrameNet.G.nodes[n]["node_role_id"]`.
- `edge_role_id` must remain on `FrameNet.G.edges[e]["edge_role_id"]`.
- Fragment registries remain `node_role_registry` and `edge_role_registry`.
- Documentation must describe graph-stored role identifiers and builder-owned registries as the canonical runtime model; it must not introduce alternate role stores, local role maps, or chemistry-inferred role semantics.
- Single-role normalization remains the backward-compatible base case via `node:default` and `edge:default` when no role metadata is present.

**Required Tests**
- No new runtime test target is introduced by `PLANS.md` for Phase 8.
- Required verification is documentation-to-code consistency against the implemented behavior and already recorded verification for Phases 1-7.
- If documentation touches package-surface or CLI behavior descriptions, inspect the existing smoke-test expectations before making claims that exceed current coverage.
- Do not claim support for any workflow or heterogeneous case that lacks implementation and recorded verification.

**Success Criteria**
- Documentation matches the implemented behavior.
- No claims are made before tests exist.
- The single-role base case and the new multi-role internal model are both documented clearly.
- The additive family/template metadata introduced in Phase 2 is documented accurately.
- At most one multi-role example is added, and only if it reflects a stable implemented path.
- Documentation does not overstate unsupported heterogeneous cases or imply broader support than the verified Phase 1-7 implementation provides.

**Stop Rule**
- Stop immediately if Phase 8 work requires editing any forbidden file or making algorithmic, runtime, or schema changes.
- Stop immediately if accurate documentation would require inventing behavior, promising unsupported heterogeneous support, or describing architecture that conflicts with the locked pipeline or canonical role model.
- Stop immediately if the work would require changing module responsibilities, public APIs, graph-state names, or role ownership rules rather than documenting them.
- If any schema, runtime, or invariant conflict is discovered while documenting the implemented behavior, record it first in `WORKLOG.md` and `STATUS.md`, then stop before revising `PLANS.md`.

### Checkpoint P7.4 — minimal executor fix and narrow built-framework regression verification

- Date: 2026-03-13
- Thread / branch: `codex_record`
- Status: complete
- Goal: pass `builder.edge_role_registry` through the real `MetalOrganicFrameworkBuilder.build()` handoff so built `Framework` instances retain the registry for the existing Phase 7 MD-prep path
- Phase gate checked against `PLANS.md`: yes; execution remains limited to `src/mofbuilder/core/builder.py`, `src/mofbuilder/core/framework.py`, `tests/test_core_framework.py`, `WORKLOG.md`, and `STATUS.md`, with no MD-module redesign or broader Phase 7 scope expansion.
- Files changed: `src/mofbuilder/core/builder.py`, `tests/test_core_framework.py`, `WORKLOG.md`, `STATUS.md`
- Tests added: `test_builder_build_hands_off_edge_role_registry_to_md_prepare`
- Tests run: baseline `scripts/run_tests.sh tests/test_core_framework.py` (passed: 7 tests, 7 existing `PytestUnknownMarkWarning` warnings) and `scripts/run_tests.sh tests/test_md_gmxfilemerge.py` (passed: 7 tests); post-fix `scripts/run_tests.sh tests/test_core_framework.py` (passed: 8 tests, 8 existing `PytestUnknownMarkWarning` warnings) and `scripts/run_tests.sh tests/test_md_gmxfilemerge.py` (passed: 7 tests)
- Decisions: reviewer failure was confirmed as a runtime integration omission at the locked builder -> framework seam, not a Phase 7 architecture conflict; the minimal safe fix was one direct `edge_role_registry` handoff in `MetalOrganicFrameworkBuilder.build()` plus one narrow builder-produced regression that proves the returned `Framework` retains the registry and uses it in `md_prepare()` without manual injection; `src/mofbuilder/core/framework.py` required no source changes because its Phase 7 role-aware MD path already consumed the handed-off registry correctly.
- Conflicts / blockers: none
- Handoff / next checkpoint: Phase 7 reopen handoff complete; next checkpoint is `P8.0` in a new thread after reviewer acceptance

### Checkpoint P8.1 — implementation

- Date: 2026-03-13
- Thread / branch: `codex_record`
- Status: complete
- Goal: sync the README, manual pages, and internal context docs to the implemented Phase 1-7 role model and verified public limits
- Phase gate checked against `PLANS.md`: yes; implementation stayed inside `README.md`, `docs/source/manual/*.md`, `ARCHITECTURE.md`, `CODEX_CONTEXT.md`, `WORKLOG.md`, and `STATUS.md` only, with no source, test, or database edits.
- Files changed: `README.md`, `docs/source/manual/quickstart.md`, `docs/source/manual/module_guide.md`, `docs/source/manual/examples.md`, `ARCHITECTURE.md`, `CODEX_CONTEXT.md`, `WORKLOG.md`, `STATUS.md`
- Tests added: none
- Tests run: `scripts/run_tests.sh tests/smoke/test_smoke_cli.py` (passed: 3 tests; 3 existing `PytestUnknownMarkWarning` warnings for `pytest.mark.smoke`); `scripts/run_tests.sh tests/smoke/test_smoke_imports.py` (passed: 2 tests; 2 existing `PytestUnknownMarkWarning` warnings for `pytest.mark.smoke`)
- Decisions: rewrote the README around the stable staged workflow and the single-role default path; documented the canonical internal role model with graph-stored `node_role_id` / `edge_role_id` and builder-owned `node_role_registry` / `edge_role_registry`; documented the optional additive `MOF_topology_role_metadata.json` sidecar without implying it is bundled for all families; added one bounded example that inspects the internal role model on the default build path rather than inventing an unverified public multi-fragment configuration workflow.
- Conflicts / blockers: none
- Handoff / next checkpoint: `P8.2` — handoff

### Checkpoint P8.2 — handoff

- Date: 2026-03-13
- Thread / branch: `codex_record`
- Status: complete
- Goal: confirm docs match code and known limitations
- Phase gate checked against `PLANS.md`: yes; Phase 8 remained documentation-only and did not modify production source files, tests, bundled database assets, or the locked architecture.
- Files changed: `README.md`, `docs/source/manual/quickstart.md`, `docs/source/manual/module_guide.md`, `docs/source/manual/examples.md`, `ARCHITECTURE.md`, `CODEX_CONTEXT.md`, `WORKLOG.md`, `STATUS.md`
- Tests added: none
- Tests run: `scripts/run_tests.sh tests/smoke/test_smoke_cli.py` (passed: 3 tests; 3 existing `PytestUnknownMarkWarning` warnings for `pytest.mark.smoke`); `scripts/run_tests.sh tests/smoke/test_smoke_imports.py` (passed: 2 tests; 2 existing `PytestUnknownMarkWarning` warnings for `pytest.mark.smoke`)
- Decisions: Phase 8 now documents the verified single-role base case, the implemented internal multi-role model, the optional Phase 2 sidecar metadata source, and the current scope limits without overstating unsupported heterogeneous public workflows; package-surface and CLI wording was checked against existing smoke coverage before handoff.
- Conflicts / blockers: none
- Handoff / next checkpoint: Phase 8 handoff complete; roadmap implementation is documentation-synced and ready for reviewer acceptance / closeout.

**Correction — 2026-03-13 Phase 8 review-fix reopen (before coding)**

- Scope: sync only the root documentation entry path by updating `docs/quickstart.md`, `docs/examples.md`, `docs/index.md`, `WORKLOG.md`, and `STATUS.md`.
- Invariants: keep the already-approved Phase 8 manual wording intact, preserve the documented single-role base case and canonical graph-stored role model, and avoid any source, test, or `docs/source/manual/*` changes.
- Out-of-scope modules: all production code, all tests, `PLANS.md`, `AGENTS.md`, `ARCHITECTURE.md`, `CODEX_CONTEXT.md`, `docs/source/manual/*`, and every other file remain forbidden.
- Decisions: use the preferred single-source strategy by converting the stale root quickstart/examples pages into include stubs and by routing the root index directly to `docs/source/manual/*`.
- Conflicts / blockers: none.
- Handoff / next checkpoint: apply the root-doc sync, confirm the root paths now resolve to the canonical manual content, then record the review-fix result.

**Correction — 2026-03-13 Phase 8 review-fix result**

- Files changed: `docs/quickstart.md`, `docs/examples.md`, `docs/index.md`, `WORKLOG.md`, `STATUS.md`
- Tests added: none
- Tests run: no runtime tests required for this documentation-only review fix; static verification performed by direct file comparison against `docs/source/manual/quickstart.md` and `docs/source/manual/examples.md`, plus route inspection of `docs/index.md`.
- Decisions: chose the preferred redirect/include strategy instead of copying content again; `docs/quickstart.md` and `docs/examples.md` now include the canonical manual pages verbatim, and `docs/index.md` now points readers to `docs/source/manual/*` as the authoritative manual location to prevent future drift.
- Conflicts / blockers: none.
- Handoff / next checkpoint: Phase 8 review-fix is implemented; root docs now match the manual path and the thread is ready for reviewer acceptance / closeout.
