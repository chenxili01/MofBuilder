# STATUS.md

## Workflow Status

- Phase: Phase 4
- Checkpoint: phase-4-resolved-anchor-compilation-complete
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

Phase 4 implementation is complete. Builder-owned runtime and optimization
snapshot compilation now resolve `source_atom_type` from slot/path semantics,
export explicit resolved-anchor metadata, and preserve legacy literal-`X`
families as compatibility views rather than the canonical semantic model.

## Planner Handoff

1. Phase 4 completed without widening into optimizer-consumption migration,
   framework changes, graph grammar changes, or constructor-signature changes.
2. `MetalOrganicFrameworkBuilder` now compiles resolved slot anchors into
   runtime and optimization snapshot slot rules, including
   `source_atom_type`, `anchor_source_type`, `anchor_source_ordinal`, and
   `anchor_vector`.
3. Optimization graph node and edge records now carry explicit per-edge target
   anchor metadata derived from semantic graph geometry when node `ccoords`
   are available, so downstream code can consume compiled anchors instead of
   reconstructing them from raw attachment buckets.
4. Legacy literal-`X` fallback remains valid: slot/path resolution prefers a
   typed source match first and falls back to literal `X` only as
   compatibility behavior.
5. Builder tests now cover one typed resolved-anchor compilation case and one
   legacy literal-`X` resolved-anchor compatibility case.
6. Validation in this shell remained limited:
   `python -m compileall src/mofbuilder/core/builder.py tests/test_core_builder.py`
   passed, but `pytest` is unavailable and the active interpreter also lacks
   runtime test dependencies such as `networkx`.
7. Next step is planner handoff for Phase 5 only.

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
