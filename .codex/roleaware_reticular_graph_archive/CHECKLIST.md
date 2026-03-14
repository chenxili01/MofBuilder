# CHECKLIST.md

## Purpose

This checklist is the **mandatory pre-flight and post-change safety list** for Codex work in the MOFBuilder repository.

It exists to reduce the risk of:

* architectural drift
* hidden scope expansion
* silent API breakage
* graph-state inconsistency
* role-metadata regressions
* broken downstream workflows

This checklist must be used together with:

* `AGENTS.md`
* `PLAN.md`
* `ARCHITECTURE.md`
* `ARCHITECTURE_DECISIONS.md`
* `CODEX_CONTEXT.md`

---

# 1. Before Doing Any Work

Before planning or editing anything, confirm all of the following.

## 1.1 Read required repo documents

Confirm you have read:

* `AGENTS.md`
* `PLAN.md`
* `ARCHITECTURE.md`
* `ARCHITECTURE_DECISIONS.md`
* `CODEX_CONTEXT.md`
* `STATUS.md`
* `WORKLOG.md`

If the active branch is a planning branch, also read:

* `PLAN_codex_record.md`

Do not proceed until these are read.

---

## 1.2 Identify the active role

Confirm which role you are acting as:

* Planner
* Executor

If this is unclear, stop.

---

## 1.3 Identify the active phase

Confirm:

* the exact current phase in `PLAN.md`
* the objective of the phase
* allowed files/modules
* forbidden files/modules
* invariants for the phase
* exit criteria for the phase

If any of these are unclear, stop.

---

## 1.4 Confirm the task type

Confirm whether the current task is one of:

* planning only
* implementation
* documentation only
* test-only
* validation/refactor within an existing phase

If this is unclear, stop.

---

# 2. Architecture Safety Check

Before making changes, confirm the planned work does **not** violate the core architecture.

## 2.1 Builder–Framework separation

Confirm you are not collapsing or blurring the distinction between:

* `MetalOrganicFrameworkBuilder`
* `Framework`

Builder constructs.
Framework represents the built result.

---

## 2.2 Stable graph states

Confirm you are preserving these graph-state names and meanings:

* `G`
* `sG`
* `superG`
* `eG`
* `cleaved_eG`

Do not rename or repurpose them unless the phase explicitly allows it.

---

## 2.3 Graph role source of truth

Confirm topology role ids remain stored on graph elements:

* `G.nodes[n]["node_role_id"]`
* `G.edges[e]["edge_role_id"]`

Do not replace this with chemistry inference.

---

## 2.4 Registry ownership

Confirm builder-owned runtime registries remain the payload/config source of truth:

* `node_role_registry`
* `edge_role_registry`

Do not introduce competing parallel registries without explicit planning approval.

---

## 2.5 Grammar restriction

For the current role-aware cycle, confirm you are not broadening graph grammar beyond:

* `V-E-V`
* `V-E-C`

Do not silently introduce additional graph forms.

---

## 2.6 Bundle ownership

Confirm `C*` roles remain linker bundle owners.

Do not move bundle ownership to `V*` roles unless a future explicit plan allows it.

---

## 2.7 Null-edge distinction

Confirm you are preserving the distinction between:

* null edge
* zero-length real edge

Do not collapse them into the same concept.

---

## 2.8 Primitive-first optimization

Confirm optimization remains primitive-cell-first unless the active phase explicitly changes that.

Supercell generation should remain downstream of primitive optimization in the current architecture.

---

# 3. Compatibility Check

Before changing code, confirm compatibility expectations.

## 3.1 Public API stability

Confirm the current phase does not unintentionally break public APIs.

If a public API change is needed, the phase must explicitly allow it.

---

## 3.2 Single-role fast path

Confirm single-role workflows remain supported.

Families without role metadata must continue to normalize to:

* `node:default`
* `edge:default`

---

## 3.3 Legacy family behavior

Confirm current topology families without the new metadata still work.

Do not require all families to adopt the new schema immediately.

---

## 3.4 Legacy linker support

Confirm current `linker.py` split-linker logic is preserved unless the phase explicitly says otherwise.

Generalize around it; do not hard-delete it casually.

---

## 3.5 Dependency discipline

Confirm you are not adding new dependencies unless the current phase explicitly allows it.

Preferred current stack remains limited to existing project dependencies.

---

# 4. Scope Check

Before editing files, confirm scope discipline.

## 4.1 Allowed files only

List the files you expect to touch.

Confirm they are allowed in the current phase.

---

## 4.2 Forbidden files untouched

List the files you explicitly must not touch in this phase.

If a needed change appears to require one of these files, stop and escalate.

---

## 4.3 No hidden refactor

Confirm you are not doing broad cleanup, renaming, or “while I’m here” edits outside phase scope.

Keep edits narrow.

---

## 4.4 No speculative redesign

Confirm you are implementing only settled decisions from `PLAN.md`.

Do not invent new architecture during execution.

---

# 5. Validation Planning Check

Before editing, identify what validation will prove the phase is correct.

## 5.1 Phase-specific validation

Write down:

* what will be validated
* how it will be validated
* which tests or checks should pass

If you cannot say this clearly, stop.

---

## 5.2 Failure conditions

Identify what would count as a failed phase result, such as:

* broken single-role workflows
* wrong role normalization
* silent null-edge fallback masking missing chemistry
* lost provenance
* broken graph-state propagation

---

## 5.3 Stop conditions

Identify the point where you must stop and escalate, for example:

* role semantics become ambiguous
* module boundaries conflict
* phase requires forbidden files
* invariant cannot be preserved cleanly

---

# 6. Implementation Checklist

Use this while editing.

## 6.1 Preserve ownership boundaries

Check that logic is going into the correct owning module.

Examples:

* passive metadata in `moftoplibrary.py`
* topology parsing / graph stamping in `net.py`
* high-level compilation in `builder.py`
* optimization logic in `optimizer.py`
* final materialization in `framework.py`

---

## 6.2 Preserve existing names when possible

Avoid unnecessary changes to:

* class names
* public method names
* established attribute names
* graph-state names

---

## 6.3 Keep metadata explicit

Prefer explicit metadata and validation over hidden heuristics.

Especially for:

* role classes
* path grammar
* slot typing
* null-edge behavior
* resolve mode
* provenance

---

## 6.4 Avoid silent fallback unless planned

If a fallback behavior exists, ensure it is:

* explicit
* validated
* policy-controlled

Silent fallback is dangerous unless the plan clearly allows it.

---

## 6.5 Do not duplicate semantics

Avoid encoding the same semantic truth in multiple competing places.

Examples of source-of-truth rules:

* graph carries role ids
* builder owns runtime registries
* topology metadata defines role/path grammar
* final framework owns merged structure data

---

## 6.6 Preserve debugability

When introducing new metadata or behavior, keep it inspectable.

This can include:

* graph attributes
* bundle ids
* provenance tags
* explicit validation messages

---

# 7. Role-Aware Specific Checklist

Use this for the role-aware reticular graph cycle.

## 7.1 Prefix semantics

Confirm only the prefix has global meaning:

* `V` = node center
* `C` = linker center
* `E` = connector edge

Suffix letters are family-local only.

---

## 7.2 Alias and canonical ids

Confirm both identity levels are handled consistently where needed:

* alias form, e.g. `VA`, `EA`
* canonical form, e.g. `node:VA`, `edge:EA`

---

## 7.3 Ordered path semantics

Confirm path semantics remain ordered where the plan requires it.

Do not assume `V-E-C` and `C-E-V` are interchangeable.

---

## 7.4 Slot typing

Confirm typed slots are validated before geometry matching.

Examples:

* `XA -> XA`
* `XB -> XB`

---

## 7.5 Ordering persistence

If canonical ordering is computed upstream, confirm later phases consume it rather than recomputing it inconsistently.

---

## 7.6 Null-edge payload model

If handling null edges, confirm the runtime representation matches the plan.

Current canonical null payload model:

* two overlapping anchor points

---

## 7.7 Resolve/provenance

If the current phase touches resolve behavior, confirm:

* resolve mode is explicit
* ownership changes are tracked
* provenance survives until final merge

---

# 8. Testing Checklist

After changes, confirm test and validation coverage.

## 8.1 Minimum expectations

Confirm the relevant tests/checks for the phase have been run or documented.

Examples may include:

* topology grammar validation
* role metadata normalization
* slot typing validation
* null-edge fallback policy
* canonical order persistence
* primitive optimization behavior
* supercell semantic propagation
* provenance persistence
* unsaturated-site / termination logic
* debug export behavior

---

## 8.2 No regression of critical workflows

Confirm no obvious regression in:

* existing builder workflow
* framework export behavior
* default family handling
* primitive-to-supercell flow

---

## 8.3 Heavy dependency caution

If tests use stubs/mocks for heavy scientific dependencies, remember:

* interface validation is not full scientific validation
* do not overclaim confidence from stubbed tests alone

---

# 9. Executor Self-Review Checklist

Because this repo uses Planner → Executor without a separate reviewer, executor must self-review before finishing.

Confirm all of the following:

## 9.1 Scope respected

* I modified only allowed files
* I did not expand phase scope silently

## 9.2 Invariants preserved

* builder/framework separation preserved
* graph states preserved
* role source-of-truth preserved
* compatibility preserved
* grammar not broadened unintentionally

## 9.3 Validation completed

* planned checks/tests were performed
* results match phase expectations

## 9.4 Risks documented

* any unresolved risks are documented
* any deferred follow-up is explicit
* no hidden known issues are left unmentioned

---

# 10. Completion Report Template

When finishing a task, summarize:

* active phase:
* files changed:
* files intentionally not changed:
* invariants checked:
* validations/tests run:
* known risks:
* deferred items:

Use this even for small changes.

---

# 11. Immediate Stop Rules

Stop immediately if any of the following occurs:

* you cannot identify the active phase
* the change requires forbidden files
* the architecture docs and plan conflict
* role semantics become ambiguous
* a public API break is needed but not planned
* graph grammar would need to be broadened
* source-of-truth ownership becomes unclear
* you are tempted to “clean up” unrelated code

When in doubt, stop and escalate rather than guessing.

---

# 12. Short Practical Rule

If a change feels larger than the current phase, it probably is.

Do the narrowest correct thing.

