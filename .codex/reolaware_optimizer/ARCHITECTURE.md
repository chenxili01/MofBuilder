# ARCHITECTURE.md

# MOFBuilder Optimizer Reconstruction Architecture

## Branch Context

This branch starts **after** the `role-runtime-contract` branch established the snapshot seam.

The branch does **not** redesign:

- graph role identity
- builder semantic ownership
- framework ownership
- primitive-first pipeline order

Instead, it reconstructs the optimizer so local placement is driven by the completed snapshot contract.

---

# Upstream Contract Baseline

The optimizer branch must treat the following as authoritative upstream inputs:

## Graph truth

```python
G.nodes[n]["node_role_id"]
G.edges[e]["edge_role_id"]
```

## Snapshot seam

Builder exports:

- `RoleRuntimeSnapshot`
- `OptimizationSemanticSnapshot`
- `FrameworkInputSnapshot`

Optimizer reconstruction uses:

- `OptimizationSemanticSnapshot`

The exact implemented fields are defined in `SNAPSHOT_API_HANDOFF.md`.

---

# Core Principle

The new optimizer behavior follows this order:

```text
semantic legality
→ node-local contract compilation
→ legal correspondence compilation
→ SVD/Kabsch rigid initialization
→ constrained local chemistry-aware refinement
→ optional broader/global residual handling
```

This means:

- semantics determine what is allowed
- local rigid pose is computed directly when correspondence is known
- refinement restores chemistry
- broad/global optimization handles only what remains coupled

---

# Why This Branch Exists

The earlier optimizer behavior was too distance-first.

Failure pattern:

- geometrically acceptable rotation
- semantically wrong local assignment

Representative issue:

- mixed incident edge-role families around a single node
- shortest-distance pairing can choose the wrong local correspondence

Therefore, the optimizer must stop acting as the primary semantic interpreter.

---

# Ownership Boundaries

## Graph

The graph remains the topology source of truth.

## Builder

Builder remains the owner of:
- role meaning
- slot semantics compilation
- bundle compilation
- resolve scaffolding
- snapshot compilation

## Optimizer

Optimizer becomes:
- a consumer of the snapshot seam
- a node-local contract compiler
- a legality-first correspondence engine
- an SVD/Kabsch initializer
- a constrained local refinement layer
- an optional broader/global residual optimizer

Optimizer does **not** become the owner of role meaning.

## Framework

Framework remains role-agnostic in this branch.

---

# Optimizer Reconstruction Layers

## 1. Node-local contract layer

Purpose:

```
compile the minimum local placement inputs for one node
```

Expected contents:
- node id
- node role id
- node role class
- local slot rules / slot types
- incident edge ids / role ids
- endpoint-aware slot requirements
- target graph directions or equivalent target references
- bundle/order hints where relevant
- null-edge flags
- resolve-mode hints that affect geometry interpretation

This layer is derived from `OptimizationSemanticSnapshot`.

---

## 2. Legal correspondence layer

Purpose:

```
determine which incident edges may legally map to which local slots
```

Semantics-first rules:
- required slot type must match actual local slot type
- path/endpoint semantics must be respected
- null-edge and alignment-only constraints must remain explicit

Geometry must not decide legality.

---

## 3. Local rigid initialization layer

Purpose:

```
compute a deterministic local rigid pose from a legal correspondence
```

Method:
- SVD / Kabsch
- explicit source anchors/vectors
- explicit target anchors/vectors/directions

For fully coordinated nodes, this should often determine the first local pose directly.

---

## 4. Discrete ambiguity layer

Purpose:

```
handle symmetry or repeated-slot ambiguity within the legal candidate space
```

Method:
- enumerate legal candidates
- solve SVD for each
- score candidates
- keep best legal candidate

This remains a **small discrete search**, not a blind optimizer.

---

## 5. Local constrained refinement layer

Purpose:

```
recover chemistry-aware realism after rigid initialization
```

Possible terms:
- anchor mismatch penalty
- bond-distance penalty
- angle penalty
- clash penalty
- null-edge alignment penalty

Refinement must stay inside the semantically legal neighborhood.

---

## 6. Optional integrated path layer

Purpose:

```
allow the new local placement path to coexist with the old path during migration
```

Examples:
- `semantic_snapshot=None`
- `use_role_aware_local_placement=False`

The default legacy path must remain available initially.

---

# Null-Edge Behavior

Null edge remains:

- an explicit `E*` role
- not a zero-length real edge
- possibly orientation-relevant
- not automatically equivalent to normal linker-length chemistry

Therefore the new optimizer must allow null edges to influence local placement differently from real edges.

---

# Compatibility Requirements

The branch must preserve:

- existing default-role fallback
- old optimizer path during migration
- primitive-first optimization
- graph grammar limited to `V-E-V` and `V-E-C`

Single-role / legacy families must continue to work.

---

# Target Outcome

This branch is successful when the optimizer can place at least one representative role-aware case from the snapshot seam using:

- legality-first local correspondence
- SVD/Kabsch initialization
- constrained local refinement

while preserving the old path and all ownership boundaries.
