# STATUS.md

## Workflow Status

- Phase: Phase 6
- Checkpoint: phase-6-executor-complete
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

Phase 6 is complete: compatibility layer and guarded rollout are now bounded
explicitly at the optimizer local-placement seam.
Legacy literal-`X` behavior remains the guard-off path, while guard-enabled
placement requires builder-owned resolved-anchor semantics and stays limited to
covered optimizer-local placement flows.

## Executor Handoff

1. Planner should start from the completed Phase 6 seam:
   guard-off optimizer placement remains legacy literal-`X`,
   guard-enabled placement consumes builder-compiled resolved anchors only, and
   explicit `anchor_source_type == "X"` compatibility records remain supported.
2. Supported versus unsupported rollout scope is now documented in
   `ARCHITECTURE.md`; do not assume typed-family coverage beyond the covered
   optimizer-local placement path.
3. Phase 7 may expand regression coverage and debug surfaces, but it must not
   blur builder ownership, remove compatibility paths prematurely, or widen the
   rollout beyond documented support without replanning.

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
