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

# Architectural Invariants

Must not change:

```
Builder → Framework separation
Primitive optimization before supercell
Graph states: G → sG → superG → eG → cleaved_eG
Role ids stored on graph
Single-role families remain valid
```

Pipeline remains:

```
FrameNet
fragment preparation
NetOptimizer
supercell expansion
Framework assembly
```

Role system is an **extension**, not redesign.

---

# Role Model

Role identifiers:

```
V*  node center
C*  linker center
E*  connector edge
```

Examples:

```
VA
VB
CA
EA
EB
```

Runtime ids:

```
node:VA
node:CA
edge:EA
```

Suffix letters are **family-local**.

---

# Graph Grammar

Allowed paths:

```
V-E-V
V-E-C
```

Meaning:

```
V = node center
C = linker center
E = connector
```

No other patterns allowed.

---

# Null Edge

Null edge is still an **E role**.

Metadata:

```
edge_kind: null
```

Representation:

```
two overlapping anchor points
```

Null edges are explicit graph objects. 
Null edges represent topology connectivity without linker atoms.
They are implemented as two overlapping anchor points.
---

# Module Responsibilities

## MofTopLibrary

Owns topology metadata:

```
role declarations
connectivity
path rules
edge-kind metadata
family policies
fragment lookup hints
```

Metadata example:

```
VA(EA,EA,EB,EB)
VA-EA-CA
VA-EB-VA
EB: null
```

---

## FrameNet

`create_net()` builds the topology graph.

Responsibilities:

```
construct primitive graph
attach node_role_id
attach edge_role_id
attach slot metadata
compute cyclic order around C nodes
store ordering metadata
```

FrameNet does **not resolve chemistry**.

---

## Builder

`MetalOrganicFrameworkBuilder` orchestrates compilation.

Responsibilities:

```
load topology metadata
normalize role ids
build role registries
call FrameNet validation
compile bundle maps
prepare resolve instructions
```

Builder owns:

```
node_role_registry
edge_role_registry
bundle registry
resolve scaffolding
```

---

# Resolve Timing

Two-stage model.

Stage 1 (Builder):

```
prepare resolve instructions
compile bundle maps
prepare provenance scaffolding
```

Stage 2 (post-optimization):

```
resolve fragments
commit bundle ownership
merge fragments
```

Resolution happens **before Framework assembly**.

---

# Development Phases

Executor implements **phases sequentially**.

---

# Phase 1 — Topology Metadata Loader

Add to `MofTopLibrary`:

```
role metadata loader
JSON-readable metadata format
expose roles
expose connectivity
expose path rules
expose edge-kind metadata
```

Constraints:

```
no builder logic change
```

---

# Phase 2 — Role Graph in FrameNet

Modify `FrameNet.create_net()`.

Add:

```
node_role_id on nodes
edge_role_id on edges
slot metadata
cyclic order computation for C nodes
ordering metadata on graph
```

Constraints:

```
no optimizer change
no chemistry resolution
```

---

# Phase 3 — FrameNet Validation

Add:

```
FrameNet.validate_roles()
```

Validation checks:

```
legal role prefixes
valid grammar (V-E-V, V-E-C)
connectivity consistency
slot metadata presence
ordering metadata sanity
null-edge metadata consistency
```

Builder must call validation before build.

---

# Phase 4 — Builder Role Registries

Add builder structures:

```
node_role_registry
edge_role_registry
```

Responsibilities:

```
normalize role ids
map roles to payload definitions
prepare bundle scaffolding
compile ordering data
```

Constraints:

```
optimizer unchanged
```

---

# Phase 5 — Bundle Compilation

Builder compiles linker bundles.

Use ordering metadata from FrameNet.

Bundle definition:

```
C center + incident E edges
```

Builder produces:

```
bundle registry
```

---

# Phase 6 — Resolve Preparation

Builder prepares:

```
resolve instructions
fragment lookup hints
null-edge rules
provenance scaffolding
```

No execution yet.

---

# Phase 7 — Post-Optimization Resolve

After `NetOptimizer`.

Builder performs:

```
resolve node fragments
resolve linker bundles
resolve edge fragments
commit ownership
merge fragments
```

Output becomes **Framework assembly input**.

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

# End of Plan

Scope limited to grammar:

```
V-E-V
V-E-C
```

Future graph types are **out of scope**.

