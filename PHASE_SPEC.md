# PHASE_SPEC.md

## Purpose

Defines **implementation boundaries** for each development phase described in `PLAN.md`.

Executor must follow this file to avoid:

* architecture drift
* optimizer-scope leakage
* framework/API breakage
* snapshot contract instability

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

### Do not modify unless the phase explicitly allows it

```
NetOptimizer behavior
Framework behavior
MD modules
supercell expansion logic
fragment placement behavior
geometry utilities
```

### Default allowed modules

```
mofbuilder/core/builder.py
mofbuilder/core/
tests/
workflow markdown files
```

Additional helper modules may be added **only inside `core/`** if they are snapshot/record focused.

### Do not change

```
existing public APIs
existing builder constructor signature
primitive-first optimization order
```

### Role IDs must remain stored

```
on graph nodes
on graph edges
```

Never move graph role identity out of the graph.

---

# Phase 1 — Snapshot Architecture and Record Types

## Allowed Modules

```
mofbuilder/core/builder.py
mofbuilder/core/ (new helper module allowed)
tests/
```

## Required Work

Add explicit record types and snapshot containers.

Must support records for:

```
NodeRoleRecord
EdgeRoleRecord
BundleRecord
ResolveInstructionRecord
NullEdgePolicyRecord
ProvenanceRecord
ResolvedStateRecord
```

Must support top-level snapshot containers for:

```
RoleRuntimeSnapshot
OptimizationSemanticSnapshot
FrameworkInputSnapshot
```

This phase may define:
- dataclasses
- typed mappings
- conversion helpers

## Forbidden Changes

Do not modify:

```
NetOptimizer behavior
Framework
FrameNet behavior
build pipeline order
```

## Completion Criteria

```
record types exist
snapshot containers exist
tests cover construction/basic fields
no behavior change to optimizer/framework
```

---

# Phase 2 — Builder Runtime Snapshot Export

## Allowed Modules

```
mofbuilder/core/builder.py
mofbuilder/core/ (snapshot helpers)
tests/
```

## Required Work

Add builder-owned snapshot export methods.

Builder should expose narrow getters such as:

```
get_role_runtime_snapshot()
get_optimization_semantic_snapshot()
get_framework_input_snapshot()
```

These methods must compile from existing builder-owned state.

Legacy/default-role families must continue to work.

## Forbidden Changes

Do not modify:

```
optimizer logic
framework logic
FrameNet graph stamping
supercell behavior
```

## Completion Criteria

```
builder exports snapshots
snapshots are compiled from builder state
legacy families still work
tests cover export behavior
```

---

# Phase 3 — Optimization Snapshot Semantics

## Allowed Modules

```
mofbuilder/core/builder.py
mofbuilder/core/ (snapshot helpers)
tests/
```

## Required Work

Populate `OptimizationSemanticSnapshot` with the minimum role-aware contract needed for future placement logic.

Must include, at minimum:

```
node ids / node role ids
edge ids / edge role ids
slot rules
incident edge constraints
bundle/order hints
null-edge rules
resolve modes
```

This phase may add compilation helpers to derive node-local constraint views.

## Forbidden Changes

Do not modify:

```
optimizer algorithm
framework materialization
fragment placement behavior
```

## Completion Criteria

```
optimization snapshot contains the required semantic fields
tests cover role-aware and default-role cases
builder remains the owner of interpretation
```

---

# Phase 4 — Snapshot Validation and Compatibility Tests

## Allowed Modules

```
mofbuilder/core/builder.py
mofbuilder/core/ (snapshot helpers)
tests/
```

## Required Work

Add validation/consistency checks around snapshot compilation.

Checks should cover:

```
missing role registry data
graph/snapshot consistency
bundle ordering consistency
null-edge rule consistency
legacy/default fallback stability
```

Add tests proving compatibility for:
- legacy/default families
- role-aware families
- empty/partial optional data where allowed

## Forbidden Changes

Do not modify:

```
optimizer behavior
framework behavior
public constructor signatures
```

## Completion Criteria

```
snapshot validation exists
tests prove compatibility
no production behavior change outside snapshot export
```

---

# Phase 5 — Optional Optimizer Snapshot Ingestion Hook

## Allowed Modules

```
mofbuilder/core/builder.py
mofbuilder/core/optimizer.py
tests/
```

## Required Work

Add only the smallest optional hook that allows the optimizer to accept a semantic snapshot.

Examples:

```
semantic_snapshot=None
```

The old optimizer path must remain fully intact.

No role-aware placement algorithm rewrite yet.

## Forbidden Changes

Do not modify:

```
optimizer scoring/objective logic beyond wiring the hook
framework
supercell
```

## Completion Criteria

```
optimizer can accept snapshot optionally
default path unchanged
tests cover no-snapshot and snapshot-present construction paths
```

---

# Phase 6 — Documentation and Handoff for Rotation Rewrite

## Allowed Modules

```
workflow markdown files
tests/ (doc-adjacent only if needed)
```

## Required Work

Document:
- snapshot fields
- node-local contract expectations
- SVD initializer + constrained refinement handoff
- unresolved decisions for the next branch

## Forbidden Changes

Do not modify:

```
production source behavior
```

## Completion Criteria

```
handoff docs complete
status/worklog updated
branch ready for later optimizer rewrite
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
