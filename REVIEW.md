**1. Phase and checkpoint being reviewed**

Current status points to `Phase 2 — Additive Family/Template Role Metadata`, checkpoint `P2.2` handoff in [STATUS.md:8](/Users/chenxili/GitHub/Cursor_repo/mof_cursor/MOFbuilder/STATUS.md), [STATUS.md:9](/Users/chenxili/GitHub/Cursor_repo/mof_cursor/MOFbuilder/STATUS.md), but the fresh controlled executor run recorded in [WORKLOG.md:213](/Users/chenxili/GitHub/Cursor_repo/mof_cursor/MOFbuilder/WORKLOG.md) through [WORKLOG.md:322](/Users/chenxili/GitHub/Cursor_repo/mof_cursor/MOFbuilder/WORKLOG.md) only reaches the implementation checkpoint and hands off to `P2.2`; it does not contain a matching new `P2.2` handoff entry for this 2026-03-13 run.

**2. Files modified**

Files modified in the working tree:
- [src/mofbuilder/core/moftoplibrary.py](/Users/chenxili/GitHub/Cursor_repo/mof_cursor/MOFbuilder/src/mofbuilder/core/moftoplibrary.py)
- [tests/test_core_moftoplibrary.py](/Users/chenxili/GitHub/Cursor_repo/mof_cursor/MOFbuilder/tests/test_core_moftoplibrary.py)
- [WORKLOG.md](/Users/chenxili/GitHub/Cursor_repo/mof_cursor/MOFbuilder/WORKLOG.md)
- [STATUS.md](/Users/chenxili/GitHub/Cursor_repo/mof_cursor/MOFbuilder/STATUS.md)

Tests added, per [WORKLOG.md:318](/Users/chenxili/GitHub/Cursor_repo/mof_cursor/MOFbuilder/WORKLOG.md):
- `test_read_mof_top_dict_without_sidecar_keeps_role_metadata_none`
- `test_read_mof_top_dict_loads_canonical_role_metadata_sidecar`
- `test_read_mof_top_dict_rejects_invalid_schema_at_library_boundary`

Decisions recorded, per [WORKLOG.md:320](/Users/chenxili/GitHub/Cursor_repo/mof_cursor/MOFbuilder/WORKLOG.md):
- canonical passive schema fields added at the `MofTopLibrary` boundary
- lightweight validation added for aliases, path grammar, bundle ownership, null-edge declarations, and unresolved-edge null fallback
- execution stayed in `fixture-only` mode

Scope boundary from the Phase Contract, per [WORKLOG.md:242](/Users/chenxili/GitHub/Cursor_repo/mof_cursor/MOFbuilder/WORKLOG.md) through [WORKLOG.md:308](/Users/chenxili/GitHub/Cursor_repo/mof_cursor/MOFbuilder/WORKLOG.md):
- allowed files were limited to `moftoplibrary.py`, `test_core_moftoplibrary.py`, metadata fixtures under `tests/database/`, `WORKLOG.md`, and `STATUS.md`
- builder/runtime consumption was explicitly out of scope

**3. Architecture compliance check**

- File-level scope compliance: passed. The actual modified files stayed within the Phase 2 allowed set.
- Locked pipeline order: unchanged. No edits touched `MofTopLibrary.fetch(...) -> FrameNet.create_net(...) -> MetalOrganicFrameworkBuilder.load_framework() -> optimize_framework() -> make_supercell() -> build`.
- Graph-state names: unchanged. No edits touched `G`, `sG`, `superG`, `eG`, or `cleaved_eG`.
- Responsibility placement: mostly correct at the file boundary. The implementation stays in `MofTopLibrary`, but it changes the shape of `role_metadata` returned by that boundary in a way that conflicts with the current downstream builder consumer.

Primary finding:
- [P1] The Phase 2 change replaces the exposed `role_metadata` structure with the new canonical schema in [src/mofbuilder/core/moftoplibrary.py:748](/Users/chenxili/GitHub/Cursor_repo/mof_cursor/MOFbuilder/src/mofbuilder/core/moftoplibrary.py), [src/mofbuilder/core/moftoplibrary.py:840](/Users/chenxili/GitHub/Cursor_repo/mof_cursor/MOFbuilder/src/mofbuilder/core/moftoplibrary.py), but the existing builder still consumes `metadata.get("node_roles")` and `metadata.get("edge_roles")` in [src/mofbuilder/core/builder.py:341](/Users/chenxili/GitHub/Cursor_repo/mof_cursor/MOFbuilder/src/mofbuilder/core/builder.py). The current builder test fixture in [tests/test_core_builder.py:183](/Users/chenxili/GitHub/Cursor_repo/mof_cursor/MOFbuilder/tests/test_core_builder.py) confirms that the repo baseline still expects the old Phase 2 shape. That means this Phase 2 thread has changed an existing runtime seam without updating the forbidden downstream consumer, which violates the contract’s stop rule in [WORKLOG.md:304](/Users/chenxili/GitHub/Cursor_repo/mof_cursor/MOFbuilder/WORKLOG.md).

**4. Role-model compliance check**

- `FrameNet.G.nodes[n]["node_role_id"]`: untouched in this phase.
- `FrameNet.G.edges[e]["edge_role_id"]`: untouched in this phase.
- Deterministic role ids: the new metadata normalization derives canonical ids deterministically from aliases in [src/mofbuilder/core/moftoplibrary.py:89](/Users/chenxili/GitHub/Cursor_repo/mof_cursor/MOFbuilder/src/mofbuilder/core/moftoplibrary.py) through [src/mofbuilder/core/moftoplibrary.py:170](/Users/chenxili/GitHub/Cursor_repo/mof_cursor/MOFbuilder/src/mofbuilder/core/moftoplibrary.py).
- No chemistry inference from topology roles: passed within `MofTopLibrary`; the validation is alias/schema based, not chemistry based.

Caveat:
- Although graph-stored role-id invariants are untouched, the current runtime handoff to builder-owned registries is now inconsistent with the repository baseline. The builder will ignore the new schema and fall back to default single-role registry construction because `node_roles` / `edge_roles` are no longer present.

**5. Test coverage review**

Required phase test:
- `scripts/run_tests.sh tests/test_core_moftoplibrary.py` passed locally: 7 tests.

Additional reviewer verification:
- `scripts/run_tests.sh tests/test_core_builder.py` passed locally: 4 tests, 4 existing `pytest.mark.core` warnings.

Coverage assessment:
- The required Phase 2 tests exist and pass.
- Legacy-family regression exists.
- No-sidecar regression exists.
- Invalid-schema regression exists.
- The gap is integration coverage at the existing runtime seam: nothing verifies that `MofTopLibrary.role_metadata` remains consumable by the already-present builder logic.

**6. Scope violations (if any)**

- No forbidden files were modified.
- There is a contract/process violation: `STATUS.md` claims the active checkpoint is `P2.2` handoff, but the fresh controlled 2026-03-13 Phase 2 entry in `WORKLOG.md` stops at implementation and only says the next checkpoint is `P2.2`. The required handoff log for this thread is missing.
- There is a semantic scope violation: the phase was supposed to remain passive and avoid runtime impact, but changing the exposed `role_metadata` structure breaks the already-existing downstream builder contract without touching the forbidden consumer.

**7. Recommended fixes (if needed)**

- Restore compatibility at the `MofTopLibrary` boundary before approval. The safe Phase 2 fix is to preserve the currently consumed `node_roles` / `edge_roles` shape for `self.role_metadata`, or expose the new canonical schema through a separate additive field/accessor that Phase 3 can adopt. Do not change `builder.py` in this phase.
- Add a regression that exercises the current runtime seam: load canonical sidecar metadata through `MofTopLibrary`, then prove the builder does not silently collapse back to `node:default` / `edge:default`. If that cannot be done within Phase 2 scope, record the conflict and stop the phase instead of changing the exposed schema.
- Synchronize execution logs: either add the missing fresh `P2.2` handoff entry to [WORKLOG.md](/Users/chenxili/GitHub/Cursor_repo/mof_cursor/MOFbuilder/WORKLOG.md) or move [STATUS.md](/Users/chenxili/GitHub/Cursor_repo/mof_cursor/MOFbuilder/STATUS.md) back to the implementation checkpoint.

**8. Final decision**

FAILED