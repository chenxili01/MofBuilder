# OPTIMIZER_TODO_ROADMAP.md

## Purpose

This document converts the optimizer discussion memory into a **step-by-step implementation roadmap** for future optimizer work.

It is meant to be used **after** the `role-runtime-contract` branch establishes the clean snapshot API.

This document is intentionally practical:
- what to implement
- in what order
- what to avoid
- what counts as success

It should be read together with:

- `OPTIMIZER_DISCUSSION_MEMORY.md`
- `SNAPSHOT_API_HANDOFF.md`
- `ROUND1_CHECKPOINT.md`
- `ROUND2_CHECKPOINT.md`
- the snapshot API documents created for the `role-runtime-contract` branch

---

# 1. Branch Goal

Future branch goal:

> Reconstruct the optimizer / rotation logic so node placement is driven by compiled semantic constraints first, then refined by chemistry-aware optimization.

This means moving away from:
- blind nearest-distance pairing
- optimizer-owned semantic interpretation
- generic continuous rotation search as the primary mechanism

and toward:

```text
semantic legality
→ deterministic local correspondence
→ SVD/Kabsch initialization
→ constrained chemistry-aware refinement
→ optional global/cell residual optimization
```

---

# 2. Preconditions Before Starting

Do **not** start the optimizer rewrite until all of the following exist:

## 2.1 Clean snapshot seam exists
The builder must already export a stable optimizer-facing snapshot.

Expected concept:

```python
OptimizationSemanticSnapshot
```

Use `SNAPSHOT_API_HANDOFF.md` as the concrete contract reference for the implemented Phase 5 snapshot surface.

## 2.2 Snapshot contents are sufficient
At minimum the snapshot must expose:

- graph phase / graph identity
- node ids
- node role ids
- edge ids
- edge role ids
- slot rules / slot typing
- incident edge constraints
- bundle/order hints where relevant
- null-edge rules
- resolve mode hints affecting geometry

The currently implemented fields and record names should be taken from `SNAPSHOT_API_HANDOFF.md`, not redefined ad hoc in the optimizer branch.

## 2.3 Snapshot compatibility is tested
Legacy/default-role families must still work.

Role-aware families must export deterministic snapshot content.

If these are not true yet, stop and finish the snapshot work first.

---

# 3. Non-Negotiable Design Rules

These rules must guide the implementation.

## 3.1 Builder owns meaning
The optimizer must consume compiled semantics.

It must not:
- reinterpret family metadata ad hoc
- inspect random builder internals directly
- redefine canonical order

## 3.2 Graph remains the topology source of truth
Graph role ids stay on:

```python
G.nodes[n]["node_role_id"]
G.edges[e]["edge_role_id"]
```

## 3.3 Slot labels are not role ids
Never compare:
- edge role names
- node role names
- slot labels

as if they were the same namespace.

Only compare:
- required slot type
- actual slot type

## 3.4 Null edge remains explicit
Null edge:
- is still an `E*` role
- is not a zero-length real edge
- may constrain orientation differently from normal chemistry

## 3.5 Semantic legality is a hard constraint
The optimizer may refine geometry, but it must not break:
- legal path class
- legal slot correspondence
- null-edge semantics
- bundle ownership assumptions

---

# 4. Recommended Implementation Order

This is the most important section.

---

## Step 1 — Build a node-local semantic contract view

### Goal
Create a helper that compiles the minimum local placement data for one node from the snapshot.

This helper should be derived from the documented `OptimizationSemanticSnapshot` contract in `SNAPSHOT_API_HANDOFF.md`.

### Suggested output
For one node, produce a structure like:

```python
NodePlacementContract(
    node_id=...,
    node_role_id=...,
    local_slots=[...],
    incident_edges=[...],
    target_directions=[...],
    null_edge_flags=[...],
    resolve_modes=[...],
)
```

### Must include
- node id
- node role id
- node role class
- local slot indices
- local slot types
- local anchor coordinates or local anchor vectors
- incident edge ids
- incident edge roles
- required slot type at this endpoint
- target graph direction for each incident edge
- null-edge / alignment-only flags
- null payload model where present
- optional local order hints if needed
- resolve-mode hints that affect geometry interpretation

### Why first
This is the smallest correct unit of future placement logic.

Do not touch the whole optimizer before this view exists.

---

## Step 2 — Implement legal correspondence compilation

### Goal
For one node, determine which incident edges can legally map to which local slots.

### Rules
A correspondence is legal only if:
- endpoint slot requirement matches local slot type
- path/endpoint semantics are satisfied
- any family constraints from snapshot are respected

### Output
A legal assignment space, such as:
- one unique mapping
- or a small discrete set of legal mappings

### Important
Do not use geometry to decide legality.
Use semantics first.

### Success condition
Given a fully coordinated node, the legal assignment set is small and explicit.

---

## Step 3 — Implement deterministic local rigid placement with SVD

### Goal
Given one legal correspondence set, compute the local rigid transform directly.

### Method
Use:
- SVD / Kabsch
- local anchor vectors or anchor points
- graph-derived target directions / targets

The handoff document intentionally leaves the exact target representation open. Pick one explicit representation in the later branch plan before implementation and document why.

### Intended behavior
For a fully coordinated node:
- if correspondence is unique, the first local pose should be computed directly
- no blind continuous search should be required for that node

### Success condition
The node can be placed deterministically from one legal correspondence set.

---

## Step 4 — Add discrete ambiguity handling

### Goal
Handle nodes where multiple legal correspondences remain because of:
- repeated slot types
- local symmetry
- mirrored or near-degenerate arrangements

### Method
For each legal discrete candidate:
1. solve local rigid pose with SVD
2. compute a semantic/geometry score
3. choose the best candidate

### Important
This is a **small discrete search**, not a generic blind optimizer.

### Success condition
The solver stays inside the semantically legal candidate space.

---

## Step 5 — Add local constrained refinement

### Goal
After SVD initialization, improve the pose with chemistry-aware refinement.

### Why needed
SVD is a rigid best-fit initializer, but it does not encode:
- angle preferences
- bond distortion penalties
- steric clashes
- null-edge weighting differences

### Suggested minimal energy terms
Start with:
- anchor mismatch penalty
- bond-distance penalty
- angle penalty
- clash penalty
- null-edge alignment penalty

These terms are still handoff-stage candidates. Treat them as next-branch design work, not already-settled runtime behavior.

### Constraint rule
Illegal correspondence is still forbidden.
The refinement step must not leave the legal semantic neighborhood.

### Success condition
The node pose becomes chemically more realistic without breaking semantic legality.

---

## Step 6 — Add null-edge-specific behavior

### Goal
Handle null/alignment-only edges explicitly.

### Needed because
Null edge:
- is explicit
- may affect orientation strongly
- should not necessarily behave like normal linker-length chemistry

### Likely behavior
Null/alignment-only edges may:
- contribute to orientation scoring
- contribute less or differently to distance terms
- contribute no normal linker-length penalty

### Success condition
Null-edge influence is explicit and distinct from real edges.

---

## Step 7 — Integrate as an optional optimizer path

### Goal
Wire the new logic into the optimizer without breaking the old path.

### Recommended rule
Use a guarded path such as:

```python
semantic_snapshot=None
```

If no snapshot:
- old behavior remains

If snapshot present:
- new node-local contract path becomes available

### Important
Do not remove the old optimizer path initially.

### Success condition
Backwards compatibility remains intact.

---

## Step 8 — Expand from one representative case to broader coverage

### Goal
After the prototype works, broaden coverage gradually.

### Recommended order
1. one fully coordinated `V-E-C` case
2. one mixed-role `V-E-V` / `V-E-C` case
3. one null-edge case
4. one symmetry/ambiguity case
5. only then broader family coverage

### Important
Do not start by rewriting all families at once.

---

# 5. Suggested Milestone Plan

## Milestone A — Prototype contract and one-node placement
Deliver:
- `NodePlacementContract`
- legal correspondence compiler
- SVD placement for one representative node

## Milestone B — Ambiguity handling
Deliver:
- discrete candidate enumeration
- candidate scoring
- one symmetry case

## Milestone C — Chemistry-aware local refinement
Deliver:
- small local objective
- refinement after SVD
- null-edge-aware weighting

## Milestone D — Optional optimizer integration
Deliver:
- snapshot ingestion path
- feature-guarded use in optimizer
- backwards compatibility

## Milestone E — Expanded test matrix
Deliver:
- multiple family cases
- null-edge case
- legacy fallback case
- debug traces

---

# 6. What To Prototype First

Start with exactly one representative node-local case.

Best candidate discussed:

- a fully coordinated `VA`
- mixed incident edge roles
- enough metadata to define legal slot correspondence clearly

Why:
- this is the case that exposed the current weakness
- it tests slot legality + target directions + local rigid solve
- it avoids prematurely rewriting global optimizer behavior

---

# 7. Debug Outputs To Add Early

Add explicit debug surfaces from the start.

Recommended per-node debug record:

```python
{
    "node_id": ...,
    "node_role_id": ...,
    "incident_edges": [...],
    "required_slot_types": [...],
    "legal_assignments": [...],
    "selected_assignment": ...,
    "svd_score": ...,
    "refined_score": ...,
    "null_edge_flags": [...],
}
```

This is important because the biggest failure mode is “geometrically okay but semantically wrong.”

---

# 8. Common Failure Modes To Avoid

## 8.1 Reintroducing nearest-distance legality
Do not let geometry decide which slot is legal.

## 8.2 Mixing role ids and slot labels again
Do not use naming similarity as matching logic.

## 8.3 Letting optimizer consume builder internals directly
Always go through the snapshot.

## 8.4 Starting with global optimization
Do not begin with cell-level or full-graph rewrite.
Start node-local.

## 8.5 Removing the old path too early
Keep backwards compatibility until the new path is proven.

---

# 9. Success Criteria For The Future Rewrite

The optimizer rewrite is successful when:

1. Fully coordinated nodes can be placed from compiled semantic constraints rather than blind search.
2. Local rigid initialization uses SVD/Kabsch when correspondence is known.
3. A constrained refinement step restores chemical realism.
4. Null-edge semantics remain explicit and distinct.
5. Legacy/default-role workflows still function.
6. The optimizer consumes a clean snapshot instead of arbitrary builder internals.
7. Debug outputs make semantic misassignment visible.

---

# 10. Final Guidance

The practical principle is:

> **Do the smallest semantically correct thing first.**

That means:

1. one node-local contract
2. one legal correspondence solver
3. one SVD initializer
4. one constrained refinement step
5. only then wider optimizer integration

This roadmap should keep the future optimizer branch focused and prevent it from rewriting too many things at once.

For the exact implemented snapshot layers, record types, and ownership guardrails, defer to `SNAPSHOT_API_HANDOFF.md`.
