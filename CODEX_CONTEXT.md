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
│   FrameNode / FrameLinker / FrameTermination
│   → fragment preparation
│
├─ optimizer.py
│   NetOptimizer
│   → fragment placement and cell optimization
│
├─ supercell.py
│   SupercellBuilder / EdgeGraphBuilder
│   → supercell expansion and edge graph generation
│
├─ defects.py
│   TerminationDefectGenerator
│   → remove / replace / termination workflows
│
├─ write.py
│   MofWriter
│   → merged structure assembly and export
│
└─ md/
    → solvation, force-field generation, MD setup
```

---

# Core Pipeline

```
Topology (FrameNet)
      ↓
Fragment preparation (FrameNode / FrameLinker)
      ↓
Optimization (NetOptimizer)
      ↓
Supercell expansion (SupercellBuilder)
      ↓
Framework materialization (Framework)
      ↓
Post-build workflows (defects / MD / export)
```

---

# Role-Aware Extension (Current Development)

New internal capability being added:

```
Topology roles
    V = node center
    C = linker center
    E = connector edge
```

Graph grammar currently limited to:

```
V-E-V
V-E-C
```

Role identifiers are stored on graph elements:

```
G.nodes[n]["node_role_id"]
G.edges[e]["edge_role_id"]
```

Builder resolves payload fragments via:

```
node_role_registry
edge_role_registry
```

---

# Architectural Invariants

Do **not change** without explicit planning phase:

```
Builder → Framework separation
Graph state names (G, sG, superG, eG, cleaved_eG)
Primitive optimization before supercell expansion
Role IDs stored on graph elements
Single-role families must continue working
```

---

# Development Workflow

This repository uses a **Planner → Executor Codex workflow**.

```
Planner → defines phases in PLAN.md
Executor → implements exactly one phase
```

Executor must perform **self-review using CHECKLIST.md** before completion.

---





## Purpose

This document provides **high-level architectural context for Codex agents** working in the MOFBuilder repository.

Codex should read this file **before modifying code**.

The goals are:

* prevent architectural drift
* clarify module responsibilities
* summarize core data models
* define the current development scope

This file is **not a full specification**.
For detailed architecture see:

```
ARCHITECTURE.md
ARCHITECTURE_DECISIONS.md
PLAN.md
```

---

# Codex Operating Rules

Codex agents working in this repository must follow these rules.

Violating these rules is considered an architectural error.

---

## 1. Do Not Redesign the Architecture

MOFBuilder already has a stable architecture.

Do **not** redesign:

* the builder pipeline
* the graph data model
* the builder/framework separation

If a task appears to require architectural redesign, **stop and escalate**.

---

## 2. Always Check the Active Phase

All development is **plan-driven**.

Before modifying code:

1. read `PLAN.md`
2. identify the current phase
3. confirm allowed files/modules

Do not expand scope beyond the phase.

---

## 3. Do Not Break the Builder → Framework Model

Builder constructs structures.

Framework represents built structures.

Never merge these responsibilities.

---

## 4. Preserve Graph State Names

These graph states have fixed meanings:

```
G
sG
superG
eG
cleaved_eG
```

Do not rename or repurpose them.

---

## 5. Role IDs Live on the Graph

Topology roles must remain stored on graph elements.

```
G.nodes[n]["node_role_id"]
G.edges[e]["edge_role_id"]
```

Do not move role semantics to chemistry inference.

---

## 6. Respect Graph Grammar Limits

The current system only supports:

```
V-E-V
V-E-C
```

Where:

```
V = node center
C = linker center
E = connector edge
```

Do not introduce new grammar forms without planning approval.

---

## 7. Preserve Single-Role Compatibility

Families without role metadata must continue to work.

They normalize to:

```
node:default
edge:default
```

Do not break this behavior.

---

## 8. Primitive Optimization Comes First

Optimization occurs on the **primitive topology graph**.

Supercell generation happens afterwards.

Do not reverse this order.

---

## 9. Avoid Broad Refactors

Prefer **small, localized edits**.

Do not clean up unrelated code while implementing a phase.

---

## 10. If the Architecture Is Unclear, Stop

Never guess architectural intent.

Stop and report if:

* module responsibilities conflict
* plan instructions are unclear
* invariants cannot be preserved

---

# How Codex Should Work

For every task:

1. Read
   `CODEX_CONTEXT.md`
   `ARCHITECTURE.md`
   `PLAN.md`

2. Identify the current phase.

3. Follow the checklist in:

```
CHECKLIST.md
```

4. Implement only the allowed scope.

5. Perform executor self-review before finishing.

---

# Key Principle

MOFBuilder architecture is defined by three principles:

```
graph-driven topology
fragment-based construction
builder-framework separation
```

All modifications must respect these principles.



# Repository Purpose

MOFBuilder is a **Python package for constructing Metal–Organic Framework (MOF) structures** from topology templates and fragment libraries.

The package supports:

* topology-driven structure construction
* fragment placement
* geometry optimization
* supercell expansion
* defect manipulation
* structure export
* molecular simulation preparation

The repository combines:

```
reticular chemistry
graph algorithms
geometric optimization
molecular simulation preparation
```

---

# Core Mental Model

The system has two primary user-facing objects:

```
MetalOrganicFrameworkBuilder
Framework
```

### Builder

The builder orchestrates **construction of the MOF**.

Responsibilities include:

* reading topology templates
* loading fragment libraries
* building topology graphs
* placing fragments
* optimizing geometry
* generating supercells
* constructing the final framework

### Framework

The framework represents **the completed MOF structure**.

Responsibilities include:

* structure export
* defect modification
* solvation
* force-field preparation
* MD setup
* visualization

Builder creates frameworks.

Framework objects represent **post-build structures**.

---

# Core Pipeline

The builder workflow follows a stable pipeline.

```
Topology parsing
→ Fragment preparation
→ Geometry optimization
→ Supercell generation
→ Framework materialization
→ Post-build workflows
```

The architecture intentionally keeps this pipeline stable.

---

# Internal Graph Model

MOFBuilder represents structures using **graph objects**.

Primary graph states:

```
G          primitive topology graph
sG         optimized primitive graph
superG     supercell graph
eG         edge graph
cleaved_eG processed edge graph
```

Graph objects are stored using **NetworkX**.

Graph attributes store structural metadata.

---

# Role Model

MOFBuilder uses **topology roles** to assign fragments.

Roles are stored on graph elements.

```
G.nodes[n]["node_role_id"]
G.edges[e]["edge_role_id"]
```

Builder resolves fragments using registries.

```
node_role_registry
edge_role_registry
```

---

# Role Prefix Semantics

Role identifiers use prefix-based meaning.

```
V* node center
C* linker center
E* connector edge
```

Examples:

```
VA
VB
CA
EA
EB
```

Important rule:

Suffix letters are **family-local identifiers**.

Example:

```
VA ≠ global metal node type
VA = node role within a specific topology family
```

---

# Graph Grammar Constraint

The current architecture supports only two topology patterns.

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

Other patterns are currently unsupported.

Do **not introduce additional grammar forms** unless explicitly planned.

---

# Linker Bundle Model

Multitopic linkers are reconstructed as bundles.

Bundle structure:

```
C center
+ incident E connectors
```

`C` roles act as **bundle owners**.

`V` roles represent inorganic centers.

---

# Null Edge Model


Some topologies contain connections without explicit linker atoms.

These are represented using **null edges**.

Null edges are stored as:

```
two overlapping anchor points
```

Important distinction:

```
null edge ≠ zero-length real edge
```

Null edges preserve topology connectivity.

---

# Module Responsibilities

## Builder

Central orchestration layer.

File:

```
src/mofbuilder/core/builder.py
```

Responsibilities:

* manage user inputs
* load topology metadata
* coordinate construction pipeline
* maintain role registries
* produce `Framework`

---

## Framework

Represents built structure.

File:

```
src/mofbuilder/core/framework.py
```

Responsibilities:

* merged atom tables
* export operations
* defect workflows
* simulation preparation

---

## MofTopLibrary

File:

```
src/mofbuilder/core/moftoplibrary.py
```

Responsibilities:

* read topology metadata
* resolve topology templates
* load family metadata

---

## FrameNet

File:

```
src/mofbuilder/core/net.py
```

Responsibilities:

* parse topology CIF files
* construct graph `G`
* assign topology roles

---

## FrameNode / FrameLinker

Files:

```
node.py
linker.py
```

Responsibilities:

* fragment preparation
* fragment normalization

---

## NetOptimizer

File:

```
optimizer.py
```

Responsibilities:

* fragment alignment
* geometry optimization

---

## SupercellBuilder

File:

```
supercell.py
```

Responsibilities:

* supercell expansion
* edge graph generation

---

## Defects / Termination

Files:

```
defects.py
termination.py
```

Responsibilities:

* removal workflows
* replacement workflows
* termination placement

---

## Writer

File:

```
write.py
```

Responsibilities:

* structure export
* merged structure assembly

---

# Key Architectural Invariants

The following must remain stable unless explicitly changed in a planning phase.

### Builder–Framework separation

Builder constructs structures.

Framework represents built structures.

---

### Stable graph state names

```
G
sG
superG
eG
cleaved_eG
```

Do not rename these states.

---

### Role metadata stored on graph

Roles must remain stored on graph attributes.

---

### Single-role compatibility

Families without role metadata must continue working.

They normalize to:

```
node:default
edge:default
```

---

### Primitive optimization

Optimization occurs on the primitive graph.

Supercells are generated afterwards.

---

# High-Risk Modules

Changes to these modules require extra caution.

```
optimizer.py
supercell.py
framework.py
linkerforcefield.py
```

They interact with multiple subsystems.

---

# Current Development Focus

The active development branch introduces **role-aware topology support**.

Goals include:

* deterministic fragment assignment
* support for multi-role topologies
* explicit connector semantics
* better topology metadata

This work **extends the architecture** but does not redesign the core pipeline.

---

# Out-of-Scope Changes

Codex agents should not attempt to:

* redesign the builder pipeline
* replace the graph data model
* change public APIs without planning approval
* introduce new heavy dependencies
* redesign optimization algorithms

---

# How Codex Should Approach Changes

Before implementing code:

1. Read:

```
PLAN.md
ARCHITECTURE.md
ARCHITECTURE_DECISIONS.md
AGENTS.md
```

2. Identify the active phase in `PLAN.md`.

3. Modify only the modules allowed by that phase.

4. Preserve all architectural invariants.

---

# If Architectural Ambiguity Appears

Codex must stop and report when:

* scope becomes unclear
* invariants may be violated
* topology semantics become ambiguous
* module responsibilities conflict

Do not guess architectural intent.

---

# Summary

MOFBuilder architecture is defined by three core principles:

```
graph-driven topology
fragment-based construction
builder–framework workflow separation
```

All new features must integrate within these principles.

