# STATUS.md

## Workflow Status

- Phase: Phase 6
- Checkpoint: workflow-complete
- Status: COMPLETED
- Next step: done
- Last update: 2026-03-14

## Branch

role-runtime-contract

## Objective

Establish a clean, builder-owned snapshot API that future optimizer/rotation reconstruction can consume safely, without exposing arbitrary mutable builder internals.

## Current Focus

Prepare the documentation and branch handoff for the later optimizer/rotation reconstruction branch by documenting snapshot fields, the expected node-local contract, the SVD/Kabsch initializer plus constrained-refinement handoff, and unresolved follow-up decisions without changing production behavior.

## Invariants

1. Topology graph remains the source of truth.
2. Builder owns role interpretation.
3. Framework remains role-agnostic in this branch.
4. Backward compatibility must be preserved.
5. Primitive-first optimization must be preserved.
6. Null-edge semantics must remain consistent with the checkpoints.
7. Bundle ownership and resolver boundaries must remain consistent with the checkpoints.
8. Snapshots are derived API views, not new sources of truth.
