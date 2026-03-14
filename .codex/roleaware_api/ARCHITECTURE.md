# ARCHITECTURE.md

# MOFBuilder Snapshot-First Architecture for `role-runtime-contract`

## Branch Context

This branch does **not** redesign the builder pipeline.

Instead, it formalizes a **snapshot-first seam** between:

```
Builder
→ Optimizer
→ Framework-facing materialization inputs
```

The goal is to prevent later optimizer work from reading arbitrary mutable builder internals.

---

# High-Level Workflow

The stable high-level workflow remains:

1. Topology parsing
2. Fragment preparation
3. Geometry optimization
4. Supercell generation
5. Framework materialization
6. Post-build workflows

The current branch only formalizes **builder-owned runtime views** that can be exported safely.

---

# Core Principle

The system already has meaningful builder-owned role state:

```
node_role_registry
edge_role_registry
bundle_registry
resolve_instructions
fragment_lookup_map
null_edge_rules
provenance_map
resolved-state maps
```

Those are valuable implementation artifacts, but they are not yet a clean API surface.

This branch groups and narrows that state into explicit snapshot contracts.

---

# Snapshot Layers

## 1. RoleRuntimeSnapshot

The broad builder-owned semantic runtime view.

Purpose:

```
stable internal API surface for builder-owned role semantics
```

Expected contents may include:

- family name
- role metadata reference
- graph snapshot reference or graph-phase metadata
- node role records
- edge role records
- bundle records
- resolve instruction records
- null-edge policy records
- provenance records
- resolved-state records

This snapshot is **builder-owned** and reflects semantic runtime state without exposing arbitrary internal maps ad hoc.

---

## 2. OptimizationSemanticSnapshot

A narrowed semantic view intended for optimizer consumption.

Purpose:

```
give optimizer only the data it needs
```

Expected contents may include:

- graph phase / graph identity information
- per-node role ids
- per-edge role ids
- slot rules / slot typing
- incident edge constraints
- target-order or bundle hints where relevant
- null-edge rules
- resolve mode hints that affect geometry interpretation

This snapshot must not let optimizer become a second role-interpretation engine.

---

## 3. FrameworkInputSnapshot

A narrowed resolved/materialization input view.

Purpose:

```
prepare a stable downstream handoff for framework materialization without leaking builder internals
```

This branch does not require framework behavior changes, but it defines the clean boundary that later branches may consume.

---

# Ownership Boundaries

## MofTopLibrary

Owns passive metadata only.

## FrameNet

Owns graph construction and topology-derived annotations.

Examples:

```
node_role_id
edge_role_id
slot_index
cyclic_edge_order
```

## Builder

Owns:
- metadata ingestion
- role normalization
- registries
- bundle compilation
- resolve scaffolding
- snapshot compilation

## Optimizer

Consumes the optimization snapshot only.

Does not own role interpretation.

## Framework

Consumes resolved/materialization inputs only.

Remains role-agnostic in this branch.

---

# Source of Truth Rules

## Graph

Graph role ids remain stored on graph objects:

```
G.nodes[n]["node_role_id"]
G.edges[e]["edge_role_id"]
```

The graph remains the topology source of truth.

## Builder

Builder remains the source of truth for runtime interpretation and compilation.

## Snapshot

Snapshots are **API views**, not new sources of truth.

They compile from builder-owned state and graph-owned identity.

---

# Snapshot Design Requirements

Snapshots must be:

- stable
- explicit
- typed or structurally clear
- inspectable
- phase-bounded
- backward compatible for legacy/default-role families

Snapshots must not:

- duplicate conflicting role semantics
- require optimizer to read builder internals directly
- replace graph role ids
- force immediate public API breakage

---

# Compatibility Requirements

Families without role metadata must continue to work through the existing fallback:

```
node:default
edge:default
```

The snapshot API must support that path without requiring all families to adopt new role-aware metadata.

---

# Future Handoff

The next branch or later phases may use `OptimizationSemanticSnapshot` to drive:

- legal slot-edge correspondence
- deterministic node-local placement
- SVD/Kabsch initialization
- constrained local refinement
- null-edge-aware placement rules

That future work depends on the clean boundary created here.

This branch itself does not implement that rotation logic.
