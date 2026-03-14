# STATUS.md

## Workflow Status

- Phase: Phase 3
- Checkpoint: phase-3-semantics-implemented
- Status: COMPLETED_PENDING_PLANNER
- Next step: Planner reviews completion and decides whether to advance
- Last update: 2026-03-14

## Branch

role-runtime-contract

## Objective

Establish a clean, builder-owned snapshot API that future optimizer/rotation reconstruction can consume safely, without exposing arbitrary mutable builder internals.

## Current Focus

Populate `OptimizationSemanticSnapshot` with the minimum builder-owned semantic contract needed for future node placement logic, including graph role ids, slot rules, incident edge constraints, bundle/order hints, null-edge rules, and resolve modes, without changing optimizer behavior, framework behavior, FrameNet graph stamping, or the build pipeline.

## Invariants

1. Topology graph remains the source of truth.
2. Builder owns role interpretation.
3. Framework remains role-agnostic in this branch.
4. Backward compatibility must be preserved.
5. Primitive-first optimization must be preserved.
6. Null-edge semantics must remain consistent with the checkpoints.
7. Bundle ownership and resolver boundaries must remain consistent with the checkpoints.
8. Snapshots are derived API views, not new sources of truth.
