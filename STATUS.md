# STATUS.md

## Workflow Status

- Phase: Phase 5
- Checkpoint: phase-5-executor-implemented
- Status: COMPLETED_PENDING_PLANNER
- Next step: Planner reviews completion and decides whether to advance
- Last update: 2026-03-14

## Branch

optimizer-reconstruction

## Objective

Reconstruct the optimizer / rotation logic so local placement is driven by the completed snapshot contract from the `role-runtime-contract` branch, using semantics-first legality, node-local contract compilation, SVD/Kabsch initialization, and constrained refinement while preserving the legacy optimizer path initially.

## Current Focus

Phase 5 executor work is complete and ready for planner review. The optimizer now has a passive local constrained refinement helper that runs only after a known legal correspondence and rigid initialization, keeps the same legal mapping fixed, and uses a documented narrow objective of anchor mismatch plus angle alignment when chemistry-direction target vectors are present. Scope remained limited to optimizer-owned modules and tests; no builder changes, framework changes, supercell changes, guarded integration wiring, null-edge-specific behavior changes, or legacy-path replacement were introduced.

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
