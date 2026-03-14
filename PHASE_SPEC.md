# PHASE_SPEC.md

## Purpose

Defines **implementation boundaries** for each typed-attachment-hardening phase described in `PLAN.md`.

Executor must follow this file to avoid:

* architecture drift
* builder/optimizer ownership blur
* accidental framework coupling
* replacing compatibility surfaces too early
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
mofbuilder/core/basic.py
mofbuilder/core/pdb_reader.py
mofbuilder/core/node.py
mofbuilder/core/linker.py
mofbuilder/core/builder.py
mofbuilder/core/optimizer.py
mofbuilder/core/ (new helper module allowed)
tests/
workflow markdown files
```

### Do not modify unless the phase explicitly allows it

```
Framework behavior
FrameNet graph stamping
MofTopLibrary metadata ownership
supercell expansion behavior
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

# Phase 1 — Attachment Semantics Audit and Contract

## Allowed Modules

```
workflow markdown files
```

## Required Work

- Define the typed attachment terminology and branch objective.
- Document where hard-coded universal `X` assumptions are forbidden.
- Record the builder-owned seam between raw fragment atoms and optimizer-consumable resolved anchors.

## Forbidden Changes

Do not modify production code.

## Completion Criteria

```
control docs are initialized
ownership boundaries are explicit
hard-coded X failure class is named directly
```

---

# Phase 2 — Reader / Parser Typed Attachment Preservation

## Allowed Modules

```
mofbuilder/core/basic.py
mofbuilder/core/pdb_reader.py
mofbuilder/core/node.py
mofbuilder/core/linker.py
tests/
workflow markdown files
```

## Required Work

- Preserve typed attachment candidates during parsing and fragment loading.
- Stop filtering attachment payloads to literal `X` only.
- Add tests covering at least one typed attachment case and one legacy literal `X` case.

## Forbidden Changes

Do not modify:

```
builder runtime schema
framework
broad optimizer behavior beyond defensive failure handling
```

## Completion Criteria

```
reader preserves typed attachment classes
legacy literal X still works
typed atoms are no longer dropped at the reader boundary
```

---

# Phase 3 — Builder Typed Attachment Registry

## Allowed Modules

```
mofbuilder/core/builder.py
mofbuilder/core/node.py
mofbuilder/core/linker.py
mofbuilder/core/fetch.py
mofbuilder/core/ (new helper module allowed)
tests/
workflow markdown files
```

## Required Work

- Introduce builder-owned typed attachment payloads such as `attachment_coords_by_type`.
- Preserve compatibility for any legacy literal-`X` helper surfaces still needed.
- Keep fragment-local typed attachment coordinates accessible for later resolution.

## Forbidden Changes

Do not modify:

```
framework
FrameNet
broad optimizer logic
```

## Completion Criteria

```
typed attachment tables exist in builder-owned surfaces
compatibility behavior for literal X remains documented and preserved
```

---

# Phase 4 — Resolved Anchor Compilation

## Allowed Modules

```
mofbuilder/core/builder.py
mofbuilder/core/optimizer.py
mofbuilder/core/ (new helper module allowed)
tests/
workflow markdown files
```

## Required Work

- Resolve slot rules to fragment-local source atom types.
- Compile resolved anchor metadata into runtime/snapshot records.
- Preserve semantics-first legality and clear ownership boundaries.

## Forbidden Changes

Do not modify:

```
framework
FrameNet
graph grammar
```

## Completion Criteria

```
resolved anchor records are compiled for downstream consumers
builder remains the owner of source-type interpretation
```

---

# Phase 5 — Optimizer Consumption Migration

## Allowed Modules

```
mofbuilder/core/optimizer.py
mofbuilder/core/builder.py
mofbuilder/core/ (new helper module allowed)
tests/
workflow markdown files
```

## Required Work

- Move local placement helpers to consume resolved anchor records rather than universal `X` buckets.
- Add explicit semantic errors for missing resolved anchors.
- Preserve legality-before-geometry and compatibility boundaries.

## Forbidden Changes

Do not modify:

```
framework
graph grammar
broad pipeline order
```

## Completion Criteria

```
optimizer local helpers no longer require node_X_data as the universal anchor source
missing-anchor failures are semantic and explainable
```

---

# Phase 6 — Compatibility Layer and Guarded Rollout

## Allowed Modules

```
mofbuilder/core/optimizer.py
mofbuilder/core/builder.py
tests/
workflow markdown files
```

## Required Work

- Keep legacy literal-`X` paths working.
- Add guarded rollout for typed-attachment paths where required.
- Document supported versus unsupported families honestly.

## Forbidden Changes

Do not modify:

```
framework
snapshot ownership schema
legacy path removal
```

## Completion Criteria

```
legacy literal X families remain available
new typed-attachment path is bounded and explicit
fallback behavior is documented and test-covered
```

---

# Phase 7 — Regression Coverage, Debug Surfaces, and Handoff

## Allowed Modules

```
mofbuilder/core/optimizer.py
mofbuilder/core/builder.py
tests/
workflow markdown files
```

## Required Work

- Add regression cases for typed-attachment families and mixed-source nodes.
- Add explicit debug/failure surfaces where helpful.
- Document unresolved risks and remaining unsupported patterns.
- Finalize handoff notes.

## Forbidden Changes

Do not modify:

```
framework
graph grammar
broad snapshot ownership changes
```

## Completion Criteria

```
regression coverage exists for typed and legacy attachment families
remaining gaps are documented honestly
handoff state is clear
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
