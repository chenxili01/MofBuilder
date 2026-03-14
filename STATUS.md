# STATUS.md

## Workflow Status

- Phase: Phase 7
- Checkpoint: phase-7-executor-implemented
- Status: COMPLETED_PENDING_PLANNER
- Next step: Planner reviews completion and decides whether to advance
- Last update: 2026-03-14

## Branch

optimizer-reconstruction

## Objective

Reconstruct the optimizer / rotation logic so local placement is driven by the completed snapshot contract from the `role-runtime-contract` branch, using semantics-first legality, node-local contract compilation, SVD/Kabsch initialization, and constrained refinement while preserving the legacy optimizer path initially.

## Current Focus

Phase 7 executor work is complete and pending planner review. The builder and optimizer now expose an explicit `use_role_aware_local_placement` guard, and the guarded path seeds the existing optimizer from the optimizer-owned local refinement helper for one representative `V`-class case while preserving the legacy path as the default when the guard is off or no snapshot is provided. Scope remained limited to optimizer/builder guarded wiring, tests, and workflow status updates; framework behavior, FrameNet, snapshot ownership/schema, graph grammar, primitive-first ordering, and unguarded legacy-path replacement were not modified.

## Invariants

1. Topology graph remains the source of truth.
2. Builder owns role interpretation and snapshot compilation.
3. Optimizer consumes the narrowed snapshot seam rather than arbitrary builder internals.
4. Framework remains role-agnostic in this branch.
5. Backward compatibility must be preserved.
6. Primitive-first optimization must be preserved.
7. Null-edge semantics must remain consistent with the checkpoints.
8. Bundle ownership and resolver boundaries must remain consistent with the checkpoints.
9. Snapshots remain derived API views, not new sources of truth.
10. Semantics must determine legality before geometry.
