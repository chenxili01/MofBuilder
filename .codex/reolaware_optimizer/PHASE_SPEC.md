# PHASE_SPEC.md

## Purpose

Defines **implementation boundaries** for each optimizer reconstruction phase described in `PLAN.md`.

Executor must follow this file to avoid:

* architecture drift
* builder/optimizer ownership blur
* accidental framework coupling
* replacing the old optimizer path too early
* widening the graph grammar

Each phase specifies:

```
Allowed modules
Required work
Forbidden changes
Completion criteria
```

Executor must **stop immediately** if the phase requirements are satisfied.

---

# Global Rules

Executor must follow these rules for **all phases**.

### Default allowed production modules

```
mofbuilder/core/optimizer.py
mofbuilder/core/ (new optimizer helper module allowed)
tests/
workflow markdown files
```

### Conditionally allowed modules

`mofbuilder/core/builder.py` may only be touched in phases that explicitly allow wiring changes.

### Do not modify unless the phase explicitly allows it

```
Framework behavior
FrameNet graph stamping
MofTopLibrary metadata ownership
supercell expansion behavior
fragment library parsing behavior
MD modules
writer/export behavior
```

### Do not change

```
existing builder constructor signature
graph state names
primitive-first optimization order
snapshot ownership boundaries
```

### Role IDs must remain stored

```
on graph nodes
on graph edges
```

Never move graph role identity out of the graph.

---

# Phase 1 — Node-Local Placement Contract

## Allowed Modules

```
mofbuilder/core/optimizer.py
mofbuilder/core/ (new optimizer helper module allowed)
tests/
```

## Required Work

Add the node-local semantic contract layer derived from `OptimizationSemanticSnapshot`.

Must support a helper structure such as:

```
NodePlacementContract
```

It must compile, at minimum:

- node id
- node role id
- node role class
- local slot rules / slot types
- incident edge ids
- incident edge role ids
- endpoint-aware slot requirements
- target direction placeholders or equivalent target references
- bundle/order hints where relevant
- null-edge flags
- resolve-mode hints that affect geometry interpretation

## Forbidden Changes

Do not modify:

```
builder snapshot schema
framework
FrameNet
global optimizer objective
```

## Completion Criteria

```
node-local contract helper exists
it compiles from the snapshot only
tests cover default-role and role-aware contract construction
no placement behavior change yet
```

---

# Phase 2 — Legal Correspondence Compilation

## Allowed Modules

```
mofbuilder/core/optimizer.py
mofbuilder/core/ (new optimizer helper module allowed)
tests/
```

## Required Work

Implement legality-first correspondence compilation.

Rules must be based on:
- required endpoint slot type
- local slot type
- endpoint/path semantics
- snapshot-provided constraints

Output should be:
- one legal mapping
- or a small discrete set of legal mappings

## Forbidden Changes

Do not modify:

```
builder
framework
global continuous optimizer behavior
```

## Completion Criteria

```
legal correspondence compiler exists
geometry is not used to determine legality
tests cover legal vs illegal mappings
```

---

# Phase 3 — SVD / Kabsch Local Rigid Initialization

## Allowed Modules

```
mofbuilder/core/optimizer.py
mofbuilder/core/ (new optimizer helper module allowed)
tests/
```

## Required Work

Implement deterministic local rigid initialization when a legal correspondence is known.

Method should use:
- SVD / Kabsch
- explicit local anchors or vectors
- explicit target directions / targets

The exact target representation must be documented in code/tests for this phase.

## Forbidden Changes

Do not modify:

```
builder snapshot schema
framework
broad global optimizer loop
```

## Completion Criteria

```
local SVD initializer exists
it operates only on legal correspondences
tests cover one representative fully coordinated case
```

---

# Phase 4 — Discrete Ambiguity Handling

## Allowed Modules

```
mofbuilder/core/optimizer.py
mofbuilder/core/ (new optimizer helper module allowed)
tests/
```

## Required Work

Handle small legal candidate sets caused by:
- repeated slot types
- symmetry
- mirrored / near-degenerate arrangements

For each legal candidate:
1. solve SVD
2. score candidate
3. choose best candidate

## Forbidden Changes

Do not modify:

```
builder
framework
global optimizer pipeline order
```

## Completion Criteria

```
discrete candidate handling exists
it stays inside the legal semantic candidate space
tests cover at least one ambiguity case
```

---

# Phase 5 — Local Constrained Refinement

## Allowed Modules

```
mofbuilder/core/optimizer.py
mofbuilder/core/ (new optimizer helper module allowed)
tests/
```

## Required Work

Add a small local chemistry-aware refinement stage after SVD.

Suggested minimal terms:
- anchor mismatch penalty
- bond-distance penalty
- angle penalty
- clash penalty
- null-edge alignment penalty

The final term set may be smaller, but it must be documented.

Refinement must not break semantic legality.

## Forbidden Changes

Do not modify:

```
builder
framework
supercell
broad global force-field redesign
```

## Completion Criteria

```
local refinement exists
it runs after SVD
it does not escape the legal correspondence neighborhood
tests cover at least one refined local case
```

---

# Phase 6 — Null-Edge-Specific Behavior

## Allowed Modules

```
mofbuilder/core/optimizer.py
mofbuilder/core/ (new optimizer helper module allowed)
tests/
```

## Required Work

Add explicit local placement behavior for:
- `is_null_edge`
- alignment-only semantics
- null payload model differences

Null edges may affect orientation differently from real edges.

## Forbidden Changes

Do not modify:

```
builder ownership
framework
graph grammar
```

## Completion Criteria

```
null-edge behavior is explicit in the new local path
tests cover at least one null-edge-aware case
null edge remains distinct from zero-length real edge
```

---

# Phase 7 — Optional Integrated Optimizer Path

## Allowed Modules

```
mofbuilder/core/optimizer.py
mofbuilder/core/builder.py
tests/
```

## Required Work

Integrate the new node-local path behind an explicit optional guard.

Examples:

```
semantic_snapshot=None
use_role_aware_local_placement=False
```

Default legacy behavior must remain intact.

## Forbidden Changes

Do not modify:

```
framework
FrameNet
snapshot ownership schema
```

## Completion Criteria

```
new path can be enabled explicitly
old path remains available
tests cover no-snapshot and snapshot-enabled cases
```

---

# Phase 8 — Expanded Coverage, Debug Surfaces, and Handoff

## Allowed Modules

```
mofbuilder/core/optimizer.py
mofbuilder/core/builder.py (only if minimal debug/wiring changes are required)
tests/
workflow markdown files
```

## Required Work

Broaden cautiously:
- add debug records for selected assignments and scores
- extend representative coverage
- document remaining gaps and handoff state

## Forbidden Changes

Do not modify:

```
framework
graph grammar
broad snapshot schema ownership
```

## Completion Criteria

```
debug surfaces exist
coverage is expanded beyond the initial prototype
handoff docs are updated honestly
```

---

# Executor Termination Rule

Executor must stop when:

```
phase completion criteria satisfied
tests/checks completed or blockers documented
STATUS.md updated for planner handoff
WORKLOG.md updated concretely
```
