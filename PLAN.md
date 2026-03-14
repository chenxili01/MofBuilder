# PLAN.md

## Workflow

Agents:

```
Planner
Executor
```

No reviewer role.

Executor must self-check with `CHECKLIST.md`.

Executor must update:

```
STATUS.md
WORKLOG.md
```

after every completed phase.

Executor implements **one phase only**.

---

# Branch Objective

Branch:

```
role-runtime-contract
```

Goal:

Establish a **clean snapshot API** that lets the builder expose a stable, read-only, role-aware optimization contract without forcing the optimizer to reach into builder internals.

This branch is **not** the full optimizer rewrite branch.

This branch prepares the seam that the later optimizer/rotation reconstruction will consume.

---

# Architectural Invariants

Must not change:

```
Builder → Framework separation
Primitive optimization before supercell
Graph states: G → sG → superG → eG → cleaved_eG
Role ids stored on graph
Single-role families remain valid
Graph grammar limited to V-E-V and V-E-C
Builder owns role interpretation
Framework remains role-agnostic
```

Pipeline remains:

```
FrameNet
fragment preparation
NetOptimizer
supercell expansion
Framework assembly
```

The snapshot API is an **extension seam**, not a pipeline redesign.

---

# Snapshot API Model

The branch introduces a stable, builder-owned, read-only runtime contract for downstream consumption.

Working names in this plan:

```
RoleRuntimeSnapshot
OptimizationSemanticSnapshot
FrameworkInputSnapshot
```

The exact final class names may vary slightly, but the API must preserve the same ownership and meaning.

---

# Ownership Rules

## Builder

Owns:

```
role metadata ingestion
graph role normalization
role registries
bundle compilation
resolve scaffolding
snapshot compilation
```

Builder exports snapshots.

Builder does **not** hand optimizer its full mutable internal state.

## Optimizer

Consumes a narrowed snapshot.

Optimizer does **not** reinterpret builder internals and does **not** become the owner of role metadata meaning.

## Framework

Consumes resolved/materialization inputs only.

Framework remains role-agnostic unless a later explicit phase expands its input surface.

---

# Snapshot Design Rules

Snapshots must be:

```
read-only by convention
typed or structurally explicit
builder-owned
minimal
stable across phases
debuggable
```

Snapshots must not:

```
duplicate competing sources of truth
replace graph-stored role ids
require optimizer to read arbitrary builder fields
force public API breakage
```

---

# Phase Roadmap

Executor implements **phases sequentially**.

---

# Phase 1 — Snapshot Architecture and Record Types

Add typed record definitions and architecture-safe container objects for the new branch.

Primary goal:

```
replace ad hoc dict surfaces with explicit runtime records
```

Scope:

```
builder-owned snapshot/record definitions only
no optimizer behavior change
```

Outputs should include record types for:

```
node role records
edge role records
bundle records
resolve instruction records
null-edge policy records
provenance records
resolved-state records
```

and top-level snapshot containers for:

```
RoleRuntimeSnapshot
OptimizationSemanticSnapshot
FrameworkInputSnapshot
```

---

# Phase 2 — Builder Runtime Snapshot Export

Add builder-side snapshot compilation/export methods.

Primary goal:

```
builder can export a stable runtime snapshot without changing existing build behavior
```

Builder should expose narrow getters such as:

```
get_role_runtime_snapshot()
get_optimization_semantic_snapshot()
get_framework_input_snapshot()
```

These methods must compile from existing builder-owned state rather than introduce new ownership.

No optimizer or framework behavior changes yet.

---

# Phase 3 — Optimization Snapshot Semantics

Populate the optimization snapshot with the minimum semantic contract needed for future node placement logic.

This phase should compile, at minimum:

```
graph role ids
slot rules
incident edge constraints
bundle/order hints
null-edge rules
resolve modes
```

This is still a builder/export phase.

No optimizer rewrite yet.

---

# Phase 4 — Snapshot Validation and Compatibility Tests

Harden the snapshot seam with focused validation and compatibility tests.

Goals:

```
single-role families still work
role-aware families export stable snapshots
snapshot contents stay phase-bounded
no builder/framework/optimizer ownership drift
```

No new runtime behavior is introduced in this phase.

---

# Phase 5 — Optional Optimizer Snapshot Ingestion Hook

Add only the smallest optional hook that allows the optimizer to accept the new snapshot object without changing default behavior.

Examples:

```
rotation_and_cell_optimization(..., semantic_snapshot=None)
NetOptimizer(..., semantic_snapshot=None)
```

This phase must preserve the old optimizer path completely.

No role-aware placement algorithm rewrite yet.

---

# Phase 6 — Documentation and Handoff for Rotation Rewrite

Prepare the branch handoff for the later optimizer/rotation reconstruction branch.

Goals:

```
document snapshot fields
document expected node-local contract
document SVD + constrained refinement plan
record unresolved decisions cleanly
```

No additional production behavior changes.

---

# Executor Rules

Executor must:

```
read PLAN.md
detect current phase from STATUS.md
implement only that phase
self-check with CHECKLIST.md
update STATUS.md
update WORKLOG.md
```

Executor must not modify architecture outside the active phase.

---

# Stop Rule

Stop immediately if a task requires:

```
optimizer algorithm rewrite
framework materialization redesign
supercell semantic redesign
new graph grammar beyond V-E-V / V-E-C
moving role interpretation out of builder
```

Those belong to later branches/phases.

---

# End of Plan

This branch is successful when the repository has a stable, builder-owned snapshot API that future optimizer work can consume safely.
