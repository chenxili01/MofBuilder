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
optimizer-reconstruction
```

Immediate goal:

```
rebuild local optimizer / rotation logic on top of the completed snapshot seam
```

This branch comes **after** the completed `role-runtime-contract` work.

---

# Upstream Snapshot Baseline

The upstream branch already provides:

- explicit snapshot record types
- builder-owned snapshot compilation
- `OptimizationSemanticSnapshot`
- optional optimizer snapshot ingestion hook

Use `SNAPSHOT_API_HANDOFF.md` as the contract reference.

Do not redesign that seam casually.

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

This branch reconstructs optimizer behavior only.

It must **not** redesign the pipeline.

---

# Current Problem Context

The previous optimizer behavior was too distance-first.

Failure pattern:
- geometrically acceptable
- semantically wrong local assignment

The new branch must move to:

```text
semantics first
→ legal correspondence
→ SVD/Kabsch initialization
→ local constrained refinement
```

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
Consumes `OptimizationSemanticSnapshot` and reconstructs local placement behavior.

Does not own role meaning.

## Framework
Remains role-agnostic in this branch.

---

# Operating Rule

If a task appears to require:
- builder semantic ownership redesign
- framework semantic redesign
- new graph grammar
- supercell semantic redesign
- deleting the old optimizer path before the new path is proven

stop and hand back to planner.
