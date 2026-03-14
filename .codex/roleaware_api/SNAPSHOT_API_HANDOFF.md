# SNAPSHOT_API_HANDOFF.md

## Purpose

This document is the Phase 6 handoff for the `role-runtime-contract` branch.

It records the finalized snapshot contract that now exists in the branch and the downstream expectations for the later optimizer / rotation reconstruction branch.

This document is descriptive. It does not introduce new runtime behavior.

---

# 1. Branch Status at Handoff

The branch now provides:

- explicit snapshot record types
- builder-owned snapshot compilation
- a narrowed optimizer-facing semantic snapshot
- an optional optimizer ingestion hook

The branch does not provide:

- a role-aware optimizer rewrite
- node-local legal correspondence compilation
- SVD/Kabsch placement logic
- constrained chemistry-aware refinement
- framework semantic expansion

This branch remains a snapshot seam branch.

---

# 2. Ownership Summary

## Graph

The topology graph remains the source of truth for graph role identity:

```python
G.nodes[n]["node_role_id"]
G.edges[e]["edge_role_id"]
```

## Builder

Builder remains the owner of:

- role interpretation
- role registry normalization
- bundle compilation
- resolve scaffolding
- null-edge policy compilation
- snapshot compilation

## Optimizer

Optimizer is only a consumer of a narrowed semantic view.

In this branch, the optimizer can accept:

```python
semantic_snapshot=None
```

but it does not yet use that snapshot to change placement behavior.

## Framework

Framework remains role-agnostic in this branch.

`FrameworkInputSnapshot` exists as a stable downstream handoff concept only.

---

# 3. Implemented Snapshot Layers

The branch currently exposes three builder-owned snapshot layers.

## 3.1 `RoleRuntimeSnapshot`

Purpose:

```text
broad builder-owned runtime semantic view
```

Current fields:

- `family_name`
- `graph_phase`
- `node_role_records`
- `edge_role_records`
- `bundle_records`
- `resolve_instruction_records`
- `null_edge_policy_records`
- `provenance_records`
- `resolved_state_records`
- `metadata`

## 3.2 `OptimizationSemanticSnapshot`

Purpose:

```text
narrow optimizer-facing semantic contract
```

Current fields:

- `family_name`
- `graph_phase`
- `graph_node_records`
- `graph_edge_records`
- `node_role_records`
- `edge_role_records`
- `bundle_records`
- `resolve_instruction_records`
- `null_edge_policy_records`
- `metadata`

This is the authoritative Phase 5 optimizer-facing handoff surface for the later rewrite branch.

## 3.3 `FrameworkInputSnapshot`

Purpose:

```text
narrow resolved/materialization handoff
```

Current fields:

- `family_name`
- `graph_phase`
- `bundle_records`
- `provenance_records`
- `resolved_state_records`
- `metadata`

---

# 4. Implemented Record Surface

The snapshot seam currently includes these explicit record types:

- `NodeRoleRecord`
- `EdgeRoleRecord`
- `BundleRecord`
- `ResolveInstructionRecord`
- `NullEdgePolicyRecord`
- `ProvenanceRecord`
- `ResolvedStateRecord`
- `GraphNodeSemanticRecord`
- `GraphEdgeSemanticRecord`

The graph-level semantic records are the main optimizer-relevant records for the next branch.

---

# 5. Concrete Optimizer-Facing Fields Available Now

This section records the concrete Phase 5-ready semantic fields already present in `OptimizationSemanticSnapshot`.

## 5.1 `GraphNodeSemanticRecord`

Current fields:

- `node_id`
- `role_id`
- `role_class`
- `slot_rules`
- `incident_edge_ids`
- `incident_edge_role_ids`
- `incident_edge_constraints`
- `bundle_id`
- `bundle_order_hint`
- `metadata`

## 5.2 `GraphEdgeSemanticRecord`

Current fields:

- `edge_id`
- `graph_edge`
- `edge_role_id`
- `path_type`
- `endpoint_node_ids`
- `endpoint_role_ids`
- `endpoint_pattern`
- `slot_index`
- `slot_rules`
- `bundle_id`
- `bundle_order_index`
- `resolve_mode`
- `is_null_edge`
- `null_payload_model`
- `allows_null_fallback`
- `metadata`

## 5.3 Supporting optimizer-facing records

The optimizer-facing snapshot also includes:

- `node_role_records`
- `edge_role_records`
- `bundle_records`
- `resolve_instruction_records`
- `null_edge_policy_records`

These support later node-local contract compilation but do not transfer role interpretation ownership out of the builder.

---

# 6. Expected Node-Local Contract for the Later Rewrite

The next optimizer / rotation branch should compile a downstream node-local contract view from `OptimizationSemanticSnapshot`.

That node-local view is not a new source of truth. It is a per-node derived helper for local placement logic.

## 6.1 Minimum expected fields

For one node, the downstream contract should compile at least:

- node id
- node role id
- node role class
- local slot rules and slot types
- incident edge ids
- incident edge role ids
- endpoint-aware slot requirements for each incident edge
- target graph directions or equivalent target vectors
- bundle id and local order hints where relevant
- null-edge flags
- null payload model where present
- resolve-mode hints that affect geometry interpretation

## 6.2 Source mapping

That future node-local contract should compile from:

- `graph_node_records`
- `graph_edge_records`
- `node_role_records`
- `edge_role_records`
- `bundle_records`
- `resolve_instruction_records`
- `null_edge_policy_records`

It should not require the optimizer to inspect random builder fields.

## 6.3 Guardrails

The next branch should preserve:

- graph role ids remain on graph elements
- builder remains owner of role meaning
- optimizer consumes compiled semantics only
- framework remains role-agnostic unless a later plan changes that explicitly

---

# 7. Intended Future Placement Flow

The agreed downstream flow remains:

```text
semantic legality
-> node-local contract compilation
-> legal correspondence compilation
-> SVD/Kabsch rigid initialization
-> constrained local refinement
-> optional broader optimizer/global residual handling
```

This is a later-branch execution plan, not current-branch runtime behavior.

---

# 8. Explicit Non-Goals of This Branch

This branch does not:

- implement node-local placement
- implement legal correspondence enumeration
- implement SVD/Kabsch placement in optimizer
- implement constrained refinement
- redefine bundle ownership
- move semantic ownership into optimizer
- widen framework inputs beyond the documented handoff snapshot

---

# 9. Open Decisions for the Next Branch

These remain unresolved on purpose and should be planned explicitly in the later optimizer branch.

## 9.1 Exact node-local target representation

Still open:

- anchor vectors
- anchor-point clouds
- local frames
- hybrid representations

## 9.2 Candidate enumeration for symmetric cases

Still open:

- how to enumerate legal discrete assignments
- how to rank symmetry-equivalent candidates
- how to handle handedness and mirrored poses

## 9.3 Minimal constrained refinement objective

Still open:

- anchor mismatch penalty
- bond-distance penalty
- angle penalty
- clash penalty
- null-edge alignment weighting

## 9.4 Behavioral semantic class fields

Still open:

- whether future optimizer behavior needs extra compiled behavioral classes beyond the current snapshot records
- whether any such field would drive real behavior rather than duplicate edge-role naming

No new such field should be invented without a later planning step.

---

# 10. Readiness Statement

At Phase 6 handoff, the branch is ready for later optimizer rewrite planning because:

- the snapshot seam exists
- ownership boundaries are documented
- optimizer-facing fields are explicit
- the node-local downstream contract is described
- the SVD/Kabsch plus constrained-refinement handoff is documented
- unresolved next-branch decisions are recorded cleanly

The next step belongs to the planner for the later optimizer / rotation reconstruction work.
