# STATUS.md

## Workflow Status

- Phase: Phase 8
- Checkpoint: workflow-complete
- Status: COMPLETED
- Next step: done
- Last update: 2026-03-14

## Branch

optimizer-reconstruction

## Objective

Reconstruct the optimizer / rotation logic so local placement is driven by the completed snapshot contract from the `role-runtime-contract` branch, using semantics-first legality, node-local contract compilation, SVD/Kabsch initialization, and constrained refinement while preserving the legacy optimizer path initially.

## Current Focus

Phase 8 executor work is complete and pending planner review. The guarded role-aware path now covers the prior representative `V`-class case plus a small additional already-supported `C`-class case, with optimizer-owned debug records for selected and fallback guarded outcomes. Validation is currently limited by the local environment: compile and diff checks passed, but full guarded-path tests still require an interpreter with the runtime dependencies installed together.

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
