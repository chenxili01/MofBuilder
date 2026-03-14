# STATUS.md

## Workflow Status

- Phase: Phase 1
- Checkpoint: phase-1-contract-complete
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

Phase 1 documentation is complete. The typed-attachment failure class, required
terminology, and builder/optimizer ownership seam are now pinned in the control docs
without modifying production code, tests, or runtime schemas.

## Executor Handoff

1. Phase 1 completed as documentation-only work in workflow/control markdown files.
2. The failure class is explicit:
   valid typed attachment atoms must not be collapsed or dropped into a universal literal
   `X` assumption before builder-owned resolution.
3. The ownership seam is explicit:
   raw fragment atom typing is upstream input, builder resolves `source_atom_type` from
   slot/path semantics and compiles resolved anchors, optimizer consumes only those
   compiled anchors.
4. The migration constraint is explicit:
   legacy literal-`X` families remain valid, but they are a compatibility path rather
   than the universal attachment model.
5. Executor stopped at the phase boundary without widening into parser, builder,
   optimizer, test, or runtime-schema implementation.

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
