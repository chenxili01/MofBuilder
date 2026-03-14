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

# Branch Objective

Branch:

```
typed-attachment-hardening
```

Goal:

Systematically remove hard-coded universal attachment-atom assumptions so fragment loading,
builder/runtime compilation, and optimizer-local placement all operate on typed attachment
sources derived from slot/path semantics rather than a universal literal `X` bucket.

This branch must implement the agreed future flow:

```text
typed attachment preservation
→ builder-owned source-type resolution
→ resolved anchor compilation
→ optimizer consumption of resolved anchors
→ guarded compatibility for legacy literal-X families
```

This branch must **not** redesign builder ownership, graph semantics, or framework ownership.

---

# Upstream Branch Baseline

This branch reuses the same development routine that worked in the completed
`optimizer-reconstruction` branch:

- phase-bounded planner/executor workflow
- explicit phase roadmap in `PLAN.md`
- hard module boundaries in `PHASE_SPEC.md`
- append-only `WORKLOG.md`
- explicit `STATUS.md` handoff after planner and executor steps
- conservative architectural invariants and stop rules

The new branch changes the problem target, not the workflow discipline.

---

# Architectural Invariants

Must not change:

```
Builder → Framework separation
Primitive optimization before supercell
Graph states: G → sG → superG → eG → cleaved_eG
Role ids stored on graph
Single-role families remain valid
Graph grammar limited to V-E-V and V-E-C
Builder owns role interpretation
Framework remains role-agnostic
Snapshots are derived API views, not new sources of truth
Null edge remains distinct from zero-length real edge
```

Pipeline remains:

```
FrameNet
fragment preparation
builder/runtime compilation
NetOptimizer
supercell expansion
Framework assembly
```

This branch is a **semantic seam hardening**, not a pipeline redesign.

---

# Core Design Rules

## 1. Builder owns attachment meaning

The optimizer must consume compiled attachment semantics from builder-owned runtime/snapshot
records.

The optimizer must not:
- reinterpret fragment atom labels ad hoc
- inspect arbitrary builder internals
- invent slot legality from geometry first
- assume all attachable atoms are literal `X`

## 2. Slot semantics are hard

Attachment legality comes from slot/path semantics first.

The code must distinguish:
- `slot_type` as semantic meaning
- `source_atom_type` as fragment-local attachment lookup class

## 3. Typed sources must be preserved

Fragment readers/parsers must preserve typed attachment candidates such as:
- `X`
- `XA`
- `XB`
- `Al`

rather than filtering everything into or through one literal class.

## 4. Builder compiles resolved anchors

Builder-owned runtime/snapshot outputs should carry resolved anchor metadata so downstream
code consumes explicit anchor records rather than reconstructing them from raw atom buckets.

## 5. Legacy paths remain available initially

Existing literal-`X` families must continue to work during migration.

Old payloads such as `node_X_data` may remain temporarily for compatibility, but must no
longer be the only attachment representation.

---

# Phase Roadmap

Executor implements **phases sequentially**.

---

# Phase 1 — Attachment Semantics Audit and Contract

Define the typed-attachment branch contract and pin the exact ownership boundaries.

Primary goal:

```
name the failure class explicitly and freeze the builder/optimizer seam
```

This phase is documentation-only. It establishes terminology such as:

```
slot_type
source_atom_type
resolved anchor
legacy literal-X compatibility
```

It must not modify production code.

Execution plan for this phase only:

1. Update the control docs so the branch objective, typed-attachment failure
   class, and builder/optimizer ownership seam are stated directly and
   consistently.
2. Record the required Phase 1 terminology in the docs:
   `slot_type`, `source_atom_type`, `resolved anchor`, and
   `legacy literal-X compatibility`.
3. Document the forbidden assumption set for later phases:
   fragment readers, builder compilation, and optimizer helpers must not treat
   literal `X` as the only attachable atom class.
4. Freeze the downstream ownership rule:
   builder-owned runtime/snapshot records are the only valid source for
   optimizer-consumable resolved-anchor semantics.
5. Keep the executor scope documentation-only:
   allowed edits are workflow/control markdown files; production code, tests,
   and runtime schemas remain out of scope in Phase 1.

Executor handoff constraints:

- Allowed files: workflow markdown files only.
- Required outcome: the docs must name the failure mode as typed attachments
  being collapsed or dropped into a universal literal-`X` assumption before
  builder-owned resolution.
- Required seam statement: raw fragment atom typing is upstream input, builder
  resolves source types and compiles anchors, optimizer consumes compiled
  anchors only.
- Required compatibility statement: legacy literal-`X` families remain valid
  during migration, but they are no longer the semantic model for all
  attachments.
- Stop rule: stop immediately if the work would require editing production
  modules, tests, runtime payloads, or widening Phase 1 into parser, builder,
  or optimizer implementation.

---

# Phase 2 — Reader / Parser Typed Attachment Preservation

Preserve typed attachment candidates during low-level fragment parsing and loading.

Primary goal:

```
stop losing valid typed attachment atoms at the reader boundary
```

Representative targets include:
- atom-name normalization behavior
- PDB note / anchor filtering logic
- node/linker fragment attachment extraction

This phase should eliminate the specific “typed atom dropped before runtime compilation”
failure mode, but it must not yet redesign builder runtime payloads broadly.

Execution plan for this phase only:

1. Audit the reader/parser path in the Phase 2 allowed modules to find every place
   typed attachment atoms can be normalized, filtered, or collapsed before fragment
   loading completes.
2. Update parsing and fragment-loading logic so attachment candidates preserve their
   fragment-local `source_atom_type` instead of being reduced to literal `X` only.
3. Keep legacy literal-`X` behavior working as a compatibility path while making sure
   typed candidates such as `XA`, `XB`, or other valid attachment labels survive the
   reader boundary unchanged enough for later builder-owned resolution.
4. Add bounded regression tests that prove:
   one typed attachment case is preserved through the reader/loading boundary, and
   one legacy literal-`X` case still behaves as before.
5. Limit Phase 2 scope to reader/parser preservation only:
   do not introduce builder-owned runtime schema changes, resolved-anchor compilation,
   or broad optimizer migration in this phase.

Executor handoff constraints:

- Allowed files: `mofbuilder/core/basic.py`, `mofbuilder/core/pdb_reader.py`,
  `mofbuilder/core/node.py`, `mofbuilder/core/linker.py`, `tests/`, and workflow
  markdown files only.
- Required outcome: valid typed attachment atoms are no longer dropped, filtered,
  or collapsed to literal `X` during parsing or fragment loading.
- Required compatibility statement: legacy literal-`X` attachment families must still
  load successfully after the preservation changes.
- Required test coverage: at least one typed attachment preservation case and at least
  one legacy literal-`X` compatibility case.
- Stop rule: stop immediately if the work would require changing builder runtime
  schemas, resolved-anchor payloads, framework behavior, or broad optimizer logic.

---

# Phase 3 — Builder Typed Attachment Registry

Add builder-owned typed attachment registries for fragment-local attachment coordinates.

Primary goal:

```
store typed attachment coordinates without collapsing them into one universal bucket
```

Representative output shape:

```python
attachment_coords_by_type = {
    "X": [...],
    "XA": [...],
    "Al": [...],
}
```

This phase should remain builder-owned and compatibility-aware.

Execution plan for this phase only:

1. Audit the Phase 3 allowed modules to find the builder-owned surfaces that still
   collapse fragment-local attachment coordinates into legacy literal-`X` buckets
   or expose only `*_X_data` compatibility payloads.
2. Introduce builder-owned typed attachment registries such as
   `attachment_coords_by_type` in the node/linker-to-builder path so fragment-local
   attachment coordinates remain grouped by preserved `source_atom_type`.
3. Keep existing literal-`X` compatibility helpers available for current callers,
   but ensure they are derived compatibility views rather than the only builder-owned
   attachment representation.
4. Add bounded regression tests that prove:
   typed attachment coordinates survive into builder-owned surfaces without being
   collapsed to one universal bucket, and legacy literal-`X` compatibility behavior
   still works for existing families.
5. Limit Phase 3 scope to the typed registry layer only:
   do not compile resolved anchors yet, do not migrate optimizer placement logic,
   and do not change framework or FrameNet ownership boundaries.

Executor handoff constraints:

- Allowed files: `mofbuilder/core/builder.py`, `mofbuilder/core/node.py`,
  `mofbuilder/core/linker.py`, `mofbuilder/core/fetch.py`,
  `mofbuilder/core/` new helper modules if needed, `tests/`, and workflow markdown
  files only.
- Required outcome: builder-owned surfaces expose typed attachment coordinate tables
  keyed by preserved `source_atom_type` instead of relying on a universal literal-`X`
  bucket as the only representation.
- Required compatibility statement: legacy literal-`X` helper surfaces may remain,
  but they must be preserved as guarded compatibility outputs rather than the
  semantic source of truth for all attachment types.
- Required test coverage: at least one builder-surface typed attachment registry case
  and at least one legacy literal-`X` compatibility case.
- Stop rule: stop immediately if the work would require resolved-anchor compilation,
  optimizer-local placement migration, framework changes, FrameNet changes, or
  broader pipeline redesign.

---

# Phase 4 — Resolved Anchor Compilation

Compile slot-rule-resolved anchors from typed fragment attachments into runtime/snapshot records.

Primary goal:

```
export explicit resolved anchors so downstream code no longer has to infer them
```

Representative outputs may include:
- `anchor_vector`
- `anchor_source_type`
- `anchor_source_ordinal`

This phase must preserve semantics-first legality and builder ownership.

Execution plan for this phase only:

1. Audit the builder runtime/snapshot compilation path in the Phase 4 allowed
   modules and identify every downstream-facing surface that still requires
   attachment meaning to be reconstructed from raw typed attachment tables or
   legacy literal-`X` compatibility buckets.
2. Introduce a builder-owned resolved-anchor compilation step that uses
   slot/path semantics to choose the fragment-local `source_atom_type` for each
   attachable role before any optimizer-local consumption occurs.
3. Compile explicit resolved-anchor records into the builder-owned runtime
   and/or snapshot surfaces using narrow metadata that downstream consumers can
   read directly, including the chosen source type and enough anchor identity
   to avoid re-derivation from raw atom buckets.
4. Preserve legacy literal-`X` compatibility payloads as compatibility views
   only, but make sure the newly compiled resolved anchors become the canonical
   builder-owned semantic output for later optimizer migration.
5. Add bounded tests that prove:
   one typed attachment case compiles a resolved anchor from preserved typed
   builder data, and one legacy literal-`X` case still produces a valid
   resolved anchor without changing framework behavior or broad optimizer flow.
6. Keep Phase 4 bounded to builder-owned resolution and anchor compilation
   only:
   do not migrate optimizer local placement to depend on the new records yet,
   and do not widen the work into framework, graph grammar, or rollout-control
   changes.

Executor handoff constraints:

- Allowed files: `mofbuilder/core/builder.py`, `mofbuilder/core/optimizer.py`,
  `mofbuilder/core/` new helper modules if needed, `tests/`, and workflow
  markdown files only.
- Required outcome: builder-owned runtime/snapshot surfaces compile explicit
  resolved-anchor records from slot/path semantics and preserved typed
  attachment inputs so downstream code no longer needs raw-bucket inference.
- Required seam statement: builder remains the only owner of
  `source_atom_type` resolution and resolved-anchor compilation; optimizer must
  not reinterpret fragment-local atom labels or re-resolve slot legality in
  this phase.
- Required compatibility statement: legacy literal-`X` families must still
  compile valid resolved anchors, but `node_X_data`-style payloads remain
  compatibility surfaces rather than the canonical semantic model.
- Required test coverage: at least one typed resolved-anchor compilation case
  and at least one legacy literal-`X` resolved-anchor compatibility case.
- Stop rule: stop immediately if the work would require optimizer-consumption
  migration, framework changes, graph grammar changes, constructor-signature
  changes, or compatibility-rollout controls beyond compiling the new anchor
  records.

---

# Phase 5 — Optimizer Consumption Migration

Move optimizer local placement helpers to consume resolved anchors instead of universal `X`
buckets.

Primary goal:

```
make optimizer local placement depend on compiled anchor semantics, not raw X-only payloads
```

This phase should cover:
- local rigid initialization inputs
- narrow helper surfaces that still assume universal `X`
- explicit semantic guardrails for missing anchors

It must not broaden into framework or global optimizer redesign.

---

# Phase 6 — Compatibility Layer and Guarded Rollout

Preserve old behavior while allowing typed-attachment cases to run through the new seam.

Primary goal:

```
keep legacy literal-X behavior stable while enabling typed attachment paths carefully
```

Examples:

```python
use_typed_attachment_runtime=False
semantic_snapshot=None
```

The exact guard surface may vary, but rollout must remain narrow and explicit.

---

# Phase 7 — Regression Coverage, Debug Surfaces, and Handoff

Broaden coverage cautiously and make failures inspectable.

Goals:

- add regression coverage for literal `X`, typed `XA`, and mixed-source cases such as `XA` + `Al`
- add explicit semantic failure messages for missing/unresolved anchors
- document remaining unsupported families honestly
- finalize handoff notes

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

# Stop Rule

Stop immediately if a task requires:

```
builder semantic ownership redesign
framework semantic redesign
new graph grammar beyond V-E-V / V-E-C
supercell semantic redesign
deleting compatibility surfaces before typed paths are proven
changing snapshot ownership ad hoc
```

Those require explicit replanning.

---

# End of Plan

This branch is successful when typed attachment atoms are preserved, builder compiles
resolved anchors from typed sources, optimizer consumes those compiled anchors for covered
paths, and legacy literal-`X` behavior remains available during migration.
