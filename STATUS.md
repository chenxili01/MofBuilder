# STATUS.md

## Workflow Status

- Phase: Phase 3
- Checkpoint: phase-3-builder-registry-complete
- Status: COMPLETED
- Next step: planner
- Last update: 2026-03-15

## Branch

typed-attachment-hardening

## Objective

Systematically remove hard-coded universal attachment-atom assumptions so fragment loading,
builder/runtime compilation, and optimizer-local placement use typed attachment sources and
resolved anchors rather than a universal literal `X` bucket.

## Current Focus

Phase 3 implementation is complete. Builder-owned surfaces now retain typed
attachment tables and coordinate registries keyed by preserved
`source_atom_type`, while legacy literal-`X` payloads remain available as
compatibility views for existing callers.

## Executor Handoff

1. Phase 3 completed without widening into resolved-anchor compilation or
   optimizer migration.
2. `MetalOrganicFrameworkBuilder` now keeps builder-owned typed attachment
   tables and coordinate registries for node, linker-center, and linker-outer
   fragment surfaces.
3. Builder role registries now store typed attachment payloads alongside the
   existing legacy `node_X_data`, `linker_center_X_data`, and
   `linker_outer_X_data` compatibility fields.
4. Builder loading paths now preserve Phase 2 typed fragment inputs into
   builder-owned registries, including recentered outer-linker attachment
   coordinates.
5. Phase 3 tests were updated to cover a builder-surface typed attachment
   registry case and a legacy literal-`X` compatibility case.
6. Validation was limited in this shell:
   `python -m compileall` passed for the changed files, but
   `python -m pytest` could not run because `pytest` is not installed in the
   active interpreter.
7. Next step is planner handoff for Phase 4 only.

## Invariants

1. Topology graph remains the source of truth.
2. Builder owns role interpretation and runtime/snapshot compilation.
3. Optimizer consumes compiled attachment semantics rather than inferring role meaning from raw fragment atoms.
4. Framework remains role-agnostic.
5. Backward compatibility must be preserved.
6. Primitive-first optimization must be preserved.
7. Null-edge semantics must remain explicit.
8. Snapshots remain derived API views, not new sources of truth.
9. Slot/path semantics determine legality before geometry.
10. Attachment-source typing must not be collapsed to a universal literal `X` assumption.
