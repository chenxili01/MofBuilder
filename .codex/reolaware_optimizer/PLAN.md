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
optimizer-reconstruction
```

Goal:

Reconstruct the optimizer / rotation logic so local placement is driven by the completed
snapshot contract from the `role-runtime-contract` branch.

This branch must implement the agreed future flow:

```text
semantic legality
→ node-local contract compilation
→ legal correspondence compilation
→ SVD/Kabsch rigid initialization
→ constrained local chemistry-aware refinement
→ optional broader/global residual handling
```

This branch must **not** redesign builder ownership, graph semantics, or framework ownership.

---

# Upstream Contract Baseline

The optimizer reconstruction branch must treat the snapshot seam as already implemented upstream.

Authoritative references:

- `SNAPSHOT_API_HANDOFF.md`
- `OPTIMIZER_DISCUSSION_MEMORY.md`
- `OPTIMIZER_TODO_ROADMAP.md`
- `ROUND1_CHECKPOINT.md`
- `ROUND2_CHECKPOINT.md`

The implemented upstream snapshot contract includes:

- `RoleRuntimeSnapshot`
- `OptimizationSemanticSnapshot`
- `FrameworkInputSnapshot`
- `GraphNodeSemanticRecord`
- `GraphEdgeSemanticRecord`

The optimizer branch must consume that contract rather than re-invent it.

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
Snapshots are derived API views, not new sources of truth
Null edge remains distinct from zero-length real edge
```

Pipeline remains:

```
FrameNet
fragment preparation
NetOptimizer
supercell expansion
Framework assembly
```

The optimizer rewrite is a **consumer-side reconstruction**, not a pipeline redesign.

---

# Core Design Rules

## 1. Builder owns meaning

The optimizer must consume compiled semantics from `OptimizationSemanticSnapshot`.

The optimizer must not:
- reinterpret family metadata ad hoc
- inspect arbitrary builder internals
- redefine canonical order
- infer slot legality from geometry first

## 2. Semantic legality is hard

The optimizer may refine geometry, but it must not violate:
- legal path class
- legal slot correspondence
- null-edge semantics
- bundle ownership assumptions

## 3. SVD is the initializer, not the whole optimizer

When correspondence is known, use SVD/Kabsch for local rigid placement.

Then optionally refine with a local chemistry-aware objective.

## 4. Old path must remain available initially

The existing optimizer behavior must remain available as a fallback while the new path is introduced behind explicit guards.

---

# Phase Roadmap

Executor implements **phases sequentially**.

---

# Phase 1 — Node-Local Placement Contract

Build a node-local semantic contract helper derived from `OptimizationSemanticSnapshot`.

Primary goal:

```
compile the minimum per-node placement input from the snapshot seam
```

This phase should create a helper structure such as:

```
NodePlacementContract
```

It must not yet rewrite the full optimizer algorithm.

---

# Phase 2 — Legal Correspondence Compilation

Implement legality-first slot/edge correspondence compilation.

Primary goal:

```
determine which incident edges may legally map to which local slots
```

This phase must use semantics first, not geometry first.

---

# Phase 3 — SVD / Kabsch Local Rigid Initialization

Implement deterministic local rigid placement when a legal correspondence is known.

Primary goal:

```
compute node-local rigid pose directly from legal correspondences
```

This phase introduces SVD/Kabsch-based initialization only.

---

# Phase 4 — Discrete Ambiguity Handling

Handle small legal candidate sets caused by:
- repeated slot types
- local symmetry
- mirrored or near-degenerate assignments

Primary goal:

```
enumerate legal discrete candidates, solve SVD for each, and score them
```

No broad continuous search should replace the legal candidate model.

---

# Phase 5 — Local Constrained Refinement

Add a small local chemistry-aware refinement stage after SVD.

Primary goal:

```
recover chemically realistic local pose without breaking semantic legality
```

This phase should start with a minimal local objective only.

---

# Phase 6 — Null-Edge-Specific Behavior

Add explicit null-edge and alignment-only handling inside the new local placement path.

Primary goal:

```
let null edges influence orientation differently from normal linker-length chemistry
```

Null-edge semantics must remain explicit.

---

# Phase 7 — Optional Integrated Optimizer Path

Wire the new node-local path into the optimizer behind an optional guarded path.

Examples:

```python
semantic_snapshot=None
use_role_aware_local_placement=False
```

Primary goal:

```
integrate the new path without breaking the old path
```

Start with one representative family/case only.

---

# Phase 8 — Expanded Coverage, Debug Surfaces, and Handoff

Broaden coverage cautiously and harden observability.

Goals:

- expand from one representative node-local case
- add debug records for local assignments and scores
- add compatibility coverage
- document unresolved follow-ups if broader integration is still pending

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
builder semantic ownership redesign
framework semantic redesign
new graph grammar beyond V-E-V / V-E-C
supercell semantic redesign
deleting the old optimizer path before new path is proven
changing snapshot schema ownership ad hoc
```

Those require explicit replanning.

---

# End of Plan

This branch is successful when the optimizer can consume the completed snapshot seam and place at least one representative role-aware case through legality-first correspondence, SVD initialization, and constrained refinement while preserving the legacy path.
