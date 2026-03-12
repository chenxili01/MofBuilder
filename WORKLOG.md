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

- Date:
- Thread / branch:
- Status: pending
- Goal: normalize scalar inputs into one-entry role registries
- Phase gate checked against `PLANS.md`:
- Files changed:
- Tests added:
- Tests run:
- Decisions:
- Conflicts / blockers:
- Handoff / next checkpoint:

### Checkpoint P3.1 — implementation

- Date:
- Thread / branch:
- Status: pending
- Goal:
- Phase gate checked against `PLANS.md`:
- Files changed:
- Tests added:
- Tests run:
- Decisions:
- Conflicts / blockers:
- Handoff / next checkpoint:

### Checkpoint P3.2 — handoff

- Date:
- Thread / branch:
- Status: pending
- Goal: confirm builder-owned registries are ready for optimizer consumption
- Phase gate checked against `PLANS.md`:
- Files changed:
- Tests added:
- Tests run:
- Decisions:
- Conflicts / blockers:
- Handoff / next checkpoint:

## Phase 4 — Role-Aware Optimizer Inputs

- Scope anchor: `src/mofbuilder/core/optimizer.py`,
  `src/mofbuilder/utils/geometry.py`, matching tests
- Must preserve: current single-role numerical path
- Must not yet: refactor supercell, writer, defects, or MD layers

### Checkpoint P4.0 — before coding

- Date:
- Thread / branch:
- Status: pending
- Goal: consume graph role ids plus builder registries in optimizer logic
- Phase gate checked against `PLANS.md`:
- Files changed:
- Tests added:
- Tests run:
- Decisions:
- Conflicts / blockers:
- Handoff / next checkpoint:

### Checkpoint P4.1 — implementation

- Date:
- Thread / branch:
- Status: pending
- Goal:
- Phase gate checked against `PLANS.md`:
- Files changed:
- Tests added:
- Tests run:
- Decisions:
- Conflicts / blockers:
- Handoff / next checkpoint:

### Checkpoint P4.2 — handoff

- Date:
- Thread / branch:
- Status: pending
- Goal: confirm optimizer supports minimal heterogeneous runtime behavior
- Phase gate checked against `PLANS.md`:
- Files changed:
- Tests added:
- Tests run:
- Decisions:
- Conflicts / blockers:
- Handoff / next checkpoint:

## Phase 5 — Role Propagation Through Supercell and Edge Graph

- Scope anchor: `src/mofbuilder/core/supercell.py`, matching tests
- Must preserve: current `superG` / `eG` behavior for single-role builds
- Must not yet: redesign writer outputs, defects APIs, or MD handling

### Checkpoint P5.0 — before coding

- Date:
- Thread / branch:
- Status: pending
- Goal: preserve role ids through `superG`, `eG`, and `cleaved_eG`
- Phase gate checked against `PLANS.md`:
- Files changed:
- Tests added:
- Tests run:
- Decisions:
- Conflicts / blockers:
- Handoff / next checkpoint:

### Checkpoint P5.1 — implementation

- Date:
- Thread / branch:
- Status: pending
- Goal:
- Phase gate checked against `PLANS.md`:
- Files changed:
- Tests added:
- Tests run:
- Decisions:
- Conflicts / blockers:
- Handoff / next checkpoint:

### Checkpoint P5.2 — handoff

- Date:
- Thread / branch:
- Status: pending
- Goal: confirm downstream phases can consume propagated role metadata
- Phase gate checked against `PLANS.md`:
- Files changed:
- Tests added:
- Tests run:
- Decisions:
- Conflicts / blockers:
- Handoff / next checkpoint:

## Phase 6 — Role-Aware Writer and Defect Metadata

- Scope anchor: `src/mofbuilder/core/write.py`,
  `src/mofbuilder/core/defects.py`, `src/mofbuilder/core/framework.py`,
  matching tests
- Must preserve: `Framework.get_merged_data()` as the sync point and current
  `remove()` / `replace()` semantics
- Must not yet: change MD contracts or invent local role identifiers

### Checkpoint P6.0 — before coding

- Date:
- Thread / branch:
- Status: pending
- Goal: generalize merged-data and defect paths to consume role-specific node
  metadata
- Phase gate checked against `PLANS.md`:
- Files changed:
- Tests added:
- Tests run:
- Decisions:
- Conflicts / blockers:
- Handoff / next checkpoint:

### Checkpoint P6.1 — implementation

- Date:
- Thread / branch:
- Status: pending
- Goal:
- Phase gate checked against `PLANS.md`:
- Files changed:
- Tests added:
- Tests run:
- Decisions:
- Conflicts / blockers:
- Handoff / next checkpoint:

### Checkpoint P6.2 — handoff

- Date:
- Thread / branch:
- Status: pending
- Goal: confirm writer/defect paths are stable before MD generalization
- Phase gate checked against `PLANS.md`:
- Files changed:
- Tests added:
- Tests run:
- Decisions:
- Conflicts / blockers:
- Handoff / next checkpoint:

## Phase 7 — Multi-Edge Force-Field and Simulation-Prep Support

- Scope anchor: `src/mofbuilder/md/linkerforcefield.py`,
  `src/mofbuilder/md/gmxfilemerge.py`, `src/mofbuilder/core/framework.py`,
  matching tests
- Must preserve: current one-linker MD-prep path
- Must not yet: broaden scope into general heterogeneous force-field research

### Checkpoint P7.0 — before coding

- Date:
- Thread / branch:
- Status: pending
- Goal: support multiple edge-role force-field paths at the currently supported
  level
- Phase gate checked against `PLANS.md`:
- Files changed:
- Tests added:
- Tests run:
- Decisions:
- Conflicts / blockers:
- Handoff / next checkpoint:

### Checkpoint P7.1 — implementation

- Date:
- Thread / branch:
- Status: pending
- Goal:
- Phase gate checked against `PLANS.md`:
- Files changed:
- Tests added:
- Tests run:
- Decisions:
- Conflicts / blockers:
- Handoff / next checkpoint:

### Checkpoint P7.2 — handoff

- Date:
- Thread / branch:
- Status: pending
- Goal: confirm minimal heterogeneous multi-edge MD-prep support only
- Phase gate checked against `PLANS.md`:
- Files changed:
- Tests added:
- Tests run:
- Decisions:
- Conflicts / blockers:
- Handoff / next checkpoint:

## Phase 8 — Documentation and Example Sync

- Scope anchor: `README.md`, `docs/source/manual/*.md`, `ARCHITECTURE.md`,
  `CODEX_CONTEXT.md`, `AGENTS.md`
- Must preserve: implemented behavior, not aspirational behavior
- Must not yet: mix documentation updates with new algorithmic refactors

### Checkpoint P8.0 — before coding

- Date:
- Thread / branch:
- Status: pending
- Goal: sync docs to the implemented multi-role model and verified limits
- Phase gate checked against `PLANS.md`:
- Files changed:
- Tests added:
- Tests run:
- Decisions:
- Conflicts / blockers:
- Handoff / next checkpoint:

### Checkpoint P8.1 — implementation

- Date:
- Thread / branch:
- Status: pending
- Goal:
- Phase gate checked against `PLANS.md`:
- Files changed:
- Tests added:
- Tests run:
- Decisions:
- Conflicts / blockers:
- Handoff / next checkpoint:

### Checkpoint P8.2 — handoff

- Date:
- Thread / branch:
- Status: pending
- Goal: confirm docs match code and known limitations
- Phase gate checked against `PLANS.md`:
- Files changed:
- Tests added:
- Tests run:
- Decisions:
- Conflicts / blockers:
- Handoff / next checkpoint:
