# PHASE_SPEC.md

## Purpose

Defines **implementation boundaries** for each development phase described in `PLAN.md`.

Executor must follow this file to avoid:

* architecture drift
* scope expansion
* module corruption

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

### Do not modify

```
NetOptimizer
Framework
MD modules
supercell expansion
fragment placement
geometry utilities
```

### Allowed modules

```
mofbuilder/core/mof_top_library.py
mofbuilder/core/frame_net.py
mofbuilder/core/builder.py
```

Additional helper modules may be added **only inside `core/`**.

### Do not change

```
existing public APIs
existing builder constructor signature
existing optimizer pipeline
```

### Role IDs must be stored

```
on graph nodes
on graph edges
```

Never store roles outside the graph.

---

# Phase 1 — Topology Metadata Loader

## Allowed Modules

```
mofbuilder/core/mof_top_library.py
```

## Required Work

Add role metadata support.

Must support metadata fields:

```
node_roles
edge_roles
connectivity
path_rules
edge_kind
```

Loader must accept **JSON-compatible dictionary format**.

Expose metadata via:

```
get_role_metadata()
get_node_roles()
get_edge_roles()
```

### Example structure

```
{
  "node_roles": ["VA","VB","CA"],
  "edge_roles": ["EA","EB"],
  "connectivity": {
     "VA":4,
     "CA":2
  },
  "edge_kind": {
     "EB":"null"
  }
}
```

## Forbidden Changes

Do not modify:

```
FrameNet
Builder
Optimizer
```

## Completion Criteria

```
metadata loader exists
metadata accessible via MofTopLibrary
unit test loads metadata successfully
```

---

# Phase 2 — Role Graph in FrameNet

## Allowed Modules

```
mofbuilder/core/frame_net.py
```

## Required Work

Modify `FrameNet.create_net()`.

Attach role identifiers.

Node attribute:

```
node_role_id
```

Edge attribute:

```
edge_role_id
```

Add slot metadata:

```
slot_index
```

Compute cyclic order for linker centers.

Store:

```
cyclic_edge_order
```

Only compute ordering for:

```
C* nodes
```

## Forbidden Changes

Do not modify:

```
Builder
Optimizer
fragment resolution
```

## Completion Criteria

Graph contains:

```
node_role_id
edge_role_id
slot_index
cyclic_edge_order
```

---

# Phase 3 — FrameNet Validation

## Allowed Modules

```
mofbuilder/core/frame_net.py
```

## Required Work

Add validation function:

```
FrameNet.validate_roles()
```

Validation checks:

```
legal role prefixes
valid grammar (V-E-V, V-E-C)
connectivity consistency
slot metadata present
cyclic order valid
null-edge metadata valid
```

Return format:

```
ValidationResult
{
  ok: bool
  errors: list
}
```

Builder must call validation before optimization.

## Forbidden Changes

Do not modify:

```
Builder internals
Optimizer
```

## Completion Criteria

```
validate_roles() implemented
validation errors descriptive
builder calls validation
```

---

# Phase 4 — Builder Role Registries

## Allowed Modules

```
mofbuilder/core/builder.py
```

## Required Work

Add builder registries:

```
node_role_registry
edge_role_registry
```

Registry content:

```
role_id
connectivity
metadata reference
```

Builder must normalize role identifiers:

```
VA -> node:VA
EA -> edge:EA
```

Store normalized roles in graph.

## Forbidden Changes

Do not modify:

```
FrameNet internals
Optimizer
fragment placement
```

## Completion Criteria

```
builder builds registries
role ids normalized
graph role ids consistent
```

---

# Phase 5 — Bundle Compilation

## Allowed Modules

```
mofbuilder/core/builder.py
```

## Required Work

Compile linker bundles.

Bundle definition:

```
C node + incident E edges
```

Builder must:

```
read cyclic_edge_order
group edges into bundle
assign bundle_id
store bundle metadata
```

Bundle metadata:

```
bundle_id
center_node
edge_list
ordering
```

## Forbidden Changes

Do not modify:

```
fragment resolution
optimizer
framework assembly
```

## Completion Criteria

Builder produces:

```
bundle_registry
```

---

# Phase 6 — Resolve Preparation

## Allowed Modules

```
mofbuilder/core/builder.py
```

## Required Work

Builder prepares resolve scaffolding.

Produce structures:

```
resolve_instructions
fragment_lookup_map
null_edge_rules
provenance_map
```

Do **not execute resolution** yet.

## Forbidden Changes

Do not modify:

```
fragment merging
geometry placement
Framework assembly
```

## Completion Criteria

Resolve data structures exist.

No fragments modified.

---

# Phase 7 — Post-Optimization Resolve

## Allowed Modules

```
mofbuilder/core/builder.py
```

## Required Work

After `NetOptimizer`.

Perform resolution:

```
resolve node fragments
resolve linker bundles
resolve edge fragments
commit bundle ownership
merge fragments
```

Resolution order:

```
node → linker bundle → edge
```

Output becomes input to:

```
Framework assembly
```

## Forbidden Changes

Do not modify:

```
optimizer
geometry algorithms
supercell expansion
```

## Completion Criteria

```
fragment ownership resolved
bundles merged correctly
framework assembly receives resolved fragments
```

---

# Executor Termination Rule

Executor must stop when:

```
phase completion criteria satisfied
tests pass
CHECKLIST.md satisfied
```

Then update:

```
STATUS.md
WORKLOG.md
```


