# CODEX_CONTEXT.md

# Repository Quick Map

```
MOFBuilder
│
├─ builder.py
│   MetalOrganicFrameworkBuilder
│   → orchestrates the entire construction pipeline
│
├─ framework.py
│   Framework
│   → container for the built MOF structure and post-build workflows
│
├─ moftoplibrary.py
│   MofTopLibrary
│   → loads topology metadata and template CIFs
│
├─ net.py
│   FrameNet
│   → parses topology CIF → graph G
│
├─ node.py / linker.py / termination.py
│   → fragment preparation
│
├─ optimizer.py
│   NetOptimizer
│   → fragment placement and cell optimization
│
└─ supercell.py
    → supercell expansion and edge graph generation
```

---

# Branch Quick Context

Branch:

```
role-runtime-contract
```

Immediate goal:

```
create a clean snapshot API before rebuilding optimizer/rotation logic
```

This is a **snapshot-first** branch, not the full role-aware optimizer branch.

---

# Core Pipeline

```
Topology (FrameNet)
      ↓
Fragment preparation
      ↓
Optimization (NetOptimizer)
      ↓
Supercell expansion
      ↓
Framework materialization
      ↓
Post-build workflows
```

This branch must **not** redesign that pipeline.

---

# Current Problem Context

The system already has builder-owned role-aware state from prior work, such as:

```
node_role_registry
edge_role_registry
bundle_registry
resolve_instructions
fragment_lookup_map
null_edge_rules
provenance_map
```

The next optimizer work needs semantic inputs, but should not consume arbitrary builder internals directly.

This branch introduces a stable seam via snapshots.

---

# Snapshot Goal

The branch should converge on three conceptual snapshot layers:

## RoleRuntimeSnapshot
Broad builder-owned semantic runtime view.

## OptimizationSemanticSnapshot
Narrow optimizer-facing semantic contract.

## FrameworkInputSnapshot
Narrow downstream handoff concept.

---

# Architectural Invariants

Do **not** change without explicit planning approval:

```
Builder → Framework separation
Graph state names (G, sG, superG, eG, cleaved_eG)
Primitive optimization before supercell expansion
Role IDs stored on graph elements
Single-role families must continue working
Graph grammar limited to V-E-V and V-E-C
```

---

# Ownership Summary

## Graph
Stores role ids and topology-derived metadata.

## Builder
Owns role interpretation, registries, bundle maps, resolve scaffolding, and snapshot compilation.

## Optimizer
May later consume `OptimizationSemanticSnapshot`.
Does not own role meaning.

## Framework
Remains role-agnostic in this branch.

---

# Operating Rule

If a task appears to require:
- optimizer algorithm redesign
- framework semantic redesign
- new graph grammar
- moving semantics out of builder

stop and hand back to planner.
