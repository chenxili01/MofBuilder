# ARCHITECTURE_DECISIONS.md

## Purpose

This document records **architectural decisions and their rationale** for MOFBuilder.

It serves several purposes:

* explain **why the system is designed the way it is**
* prevent accidental architectural drift
* guide future contributors
* help Codex agents understand design intent

Each decision entry includes:

* **Context**
* **Decision**
* **Consequences**

---

# ADR-001: Stable Builder–Framework Separation

## Context

MOFBuilder must support a full workflow:

* topology interpretation
* fragment preparation
* geometry optimization
* supercell expansion
* structure export
* defect manipulation
* MD preparation

These operations occur at **different conceptual stages**.

## Decision

Separate responsibilities between:

```
MetalOrganicFrameworkBuilder
Framework
```

Builder:

* orchestrates the **construction process**

Framework:

* represents the **completed MOF structure**
* provides **post-build operations**

## Consequences

Advantages:

* clear lifecycle boundary
* stable user-facing object
* easier extension of post-build workflows

Tradeoffs:

* requires copying state from builder to framework
* two objects must remain consistent

---

# ADR-002: Graph-Centric Internal Representation

## Context

MOF topologies are naturally expressed as **graphs**.

Nodes represent:

* metal clusters
* linker centers

Edges represent:

* linker connectors
* topology constraints

## Decision

Represent MOF topology internally as **NetworkX graphs**.

Primary graph states:

```
G         primitive topology graph
sG        optimized primitive graph
superG    expanded supercell graph
eG        edge graph
cleaved_eG processed edge graph
```

Graph attributes store semantic metadata.

## Consequences

Advantages:

* flexible topology representation
* easier neighbor queries
* natural fit for reticular chemistry

Tradeoffs:

* graph metadata must remain consistent across pipeline stages
* graph copying between states requires careful handling

---

# ADR-003: Role IDs Stored on Graph Elements

## Context

Traditional MOF builders infer fragment placement from **chemistry heuristics**.

However, topology-driven construction requires **explicit semantic roles**.

Examples:

* metal cluster centers
* linker centers
* connector edges

## Decision

Store role identifiers directly on graph elements.

```
G.nodes[n]["node_role_id"]
G.edges[e]["edge_role_id"]
```

These become the **source of truth** for role semantics.

Fragment assignment is resolved through builder-managed registries.

## Consequences

Advantages:

* deterministic fragment assignment
* explicit topology semantics
* easier multi-role family support

Tradeoffs:

* requires metadata normalization
* role registry must remain synchronized with graph state

---

# ADR-004: Role Registry Managed by Builder

## Context

Graph elements identify roles, but they do not directly contain fragment payloads.

Fragment payloads must be resolved dynamically.

## Decision

The builder maintains runtime registries:

```
node_role_registry
edge_role_registry
```

Registries map role IDs to:

* fragment payloads
* slot metadata
* connector information

## Consequences

Advantages:

* clean separation of topology and chemistry
* easier fragment substitution
* supports data-driven MOF families

Tradeoffs:

* builder becomes central orchestration layer
* registry state must propagate through pipeline

---

# ADR-005: Prefix-Based Role Semantics

## Context

MOF families require multiple role types.

Examples:

```
VA VB VC
EA EB
CA
```

However, roles must remain **family-specific**.

## Decision

Use prefix-based role semantics.

```
V*  node center
C*  linker center
E*  connector edge
```

Suffix letters represent **family-local identifiers**.

```
VA != globally special node
VA = one node role within a family
```

## Consequences

Advantages:

* flexible role system
* no global role namespace conflicts
* supports arbitrary topology families

Tradeoffs:

* role semantics must be interpreted using metadata

---

# ADR-006: Restricted Graph Grammar

## Context

Reticular frameworks connect fragments through **two fundamental patterns**.

```
node — linker — node
node — connector — linker center
```

Allowing arbitrary patterns would dramatically increase complexity.

## Decision

Limit the topology grammar to:

```
V-E-V
V-E-C
```

Where:

```
V node center
C linker center
E connector edge
```

## Consequences

Advantages:

* simpler optimizer logic
* deterministic linker reconstruction
* easier validation

Tradeoffs:

* more exotic topology patterns are not supported yet

---

# ADR-007: Linker Bundle Ownership

## Context

Multitopic linkers consist of:

* central scaffold
* connector arms

Topology must reconstruct these fragments consistently.

## Decision

Define **linker center roles (`C*`) as bundle owners**.

Bundle contents:

```
C center
+ incident E connectors
```

## Consequences

Advantages:

* deterministic linker reconstruction
* simpler fragment assembly logic

Tradeoffs:

* node roles cannot own linker fragments

---

# ADR-008: Canonical Ordering Computed at Topology Stage

## Context

Linker arms must be assigned consistently.

Without deterministic ordering, fragment placement becomes unstable.

## Decision

Compute canonical ordering during **FrameNet topology parsing**.

Ordering metadata is attached to:

```
C nodes
incident E edges
```

## Consequences

Advantages:

* deterministic fragment placement
* consistent linker reconstruction

Tradeoffs:

* topology parser must compute ordering metadata

---

# ADR-009: Null Edge Representation

## Context

Some MOF topologies contain **connections without explicit linker atoms**.

Examples:

* rod-like SBUs
* shared metal clusters

## Decision

Represent these using **null edges**.

Canonical representation:

```
two overlapping anchor points
```

Null edges remain explicit edge roles.
Null edges represent topology connectivity without linker atoms.

Important distinction:

```
null edge != zero-length chemical edge
```

## Consequences

Advantages:

* supports rod-like structures
* preserves topology connectivity

Tradeoffs:

* additional metadata needed to distinguish null edges

---

# ADR-010: Primitive Optimization Before Supercell Expansion

## Context

Optimizing large supercells is computationally expensive.

## Decision

Perform optimization only on the **primitive topology graph**.

```
optimize primitive cell
→ expand to supercell
```

Supercell generation uses **translation-based replication**.

## Consequences

Advantages:

* faster optimization
* simpler geometry operations

Tradeoffs:

* supercell must correctly propagate role metadata

---

# ADR-011: Provenance Tracking for Structural Resolve

## Context

Some construction steps may transfer atoms between fragments.

Examples:

* coordination group borrowing
* termination placement

## Decision

Track **provenance metadata** for fragment ownership changes.

Provenance survives until framework materialization.

## Consequences

Advantages:

* enables unsaturated site detection
* enables termination workflows
* easier debugging

Tradeoffs:

* additional metadata management

---

# ADR-012: Builder Pipeline Stability

## Context

Many modules depend on the builder workflow.

Changing pipeline structure risks breaking downstream systems.

## Decision

Keep the builder pipeline stable.

Core stages remain:

```
topology
→ fragment preparation
→ optimization
→ supercell expansion
→ framework materialization
```

## Consequences

Advantages:

* stable user workflows
* easier maintenance

Tradeoffs:

* architectural evolution must remain incremental

---

# ADR-013: Lazy Import Strategy

## Context

Scientific libraries introduce heavy dependencies.

Users often only need a subset of functionality.

## Decision

Use lazy imports for:

```
mofbuilder
mofbuilder.core
```

CLI remains dependency-light.

## Consequences

Advantages:

* faster startup
* lower installation friction

Tradeoffs:

* delayed import errors possible

---

# ADR-014: Compatibility with Single-Role Families

## Context

Many MOF families do not require multi-role semantics.

## Decision

Normalize families without role metadata to:

```
node:default
edge:default
```

This preserves legacy workflows.

## Consequences

Advantages:

* backward compatibility
* simpler default behavior

Tradeoffs:

* dual-path logic in builder

---

# ADR-015: Role-Aware Architecture as Internal Feature

## Context

Role-aware topology significantly expands internal flexibility.

However, exposing it directly in the public API would increase complexity.

## Decision

Treat role-aware behavior as **internal plumbing**.

Public builder workflow remains unchanged.

## Consequences

Advantages:

* stable public API
* easier adoption

Tradeoffs:

* role-aware features require internal understanding

---

# Updating This Document

When adding a major architectural change:

1. Create a new ADR entry
2. Explain context
3. Describe decision
4. Describe consequences

Do not remove previous decisions.

Architecture evolves through **additive decisions**, not rewriting history.


