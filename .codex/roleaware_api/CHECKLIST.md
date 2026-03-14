# CHECKLIST.md

## Purpose

This checklist is the mandatory pre-flight and post-change safety list for Codex work in the `role-runtime-contract` branch.

Use this together with:

* `AGENTS.md`
* `PLAN.md`
* `PHASE_SPEC.md`
* `ARCHITECTURE.md`
* `ARCHITECTURE_DECISIONS.md`
* `CODEX_CONTEXT.md`
* `STATUS.md`
* `WORKLOG.md`

---

# 1. Before Doing Any Work

## 1.1 Read required repo documents

Confirm you have read:

* `AGENTS.md`
* `PLAN.md`
* `PHASE_SPEC.md`
* `ARCHITECTURE.md`
* `ARCHITECTURE_DECISIONS.md`
* `CODEX_CONTEXT.md`
* `STATUS.md`
* `WORKLOG.md`
* `ROUND1_CHECKPOINT.md`
* `ROUND2_CHECKPOINT.md`

Do not proceed until these are read.

## 1.2 Identify the active role

Confirm which role you are acting as:
* Planner
* Executor

If unclear, stop.

## 1.3 Identify the active phase

Confirm:
* exact current phase
* objective
* allowed modules
* forbidden modules
* invariants
* exit criteria

If unclear, stop.

---

# 2. Architecture Safety Check

## 2.1 Builder–Framework separation

Confirm you are not collapsing the distinction between:
* `MetalOrganicFrameworkBuilder`
* `Framework`

## 2.2 Stable graph states

Preserve:
* `G`
* `sG`
* `superG`
* `eG`
* `cleaved_eG`

## 2.3 Graph role source of truth

Confirm topology role ids remain on graph elements:
* `G.nodes[n]["node_role_id"]`
* `G.edges[e]["edge_role_id"]`

## 2.4 Registry ownership

Confirm builder-owned runtime registries remain the payload/config source of truth:
* `node_role_registry`
* `edge_role_registry`

## 2.5 Grammar restriction

Do not broaden graph grammar beyond:
* `V-E-V`
* `V-E-C`

## 2.6 Bundle ownership

Confirm `C*` roles remain bundle owners.

## 2.7 Null-edge distinction

Preserve:
* null edge
* zero-length real edge

Do not collapse them.

## 2.8 Snapshot rule

Confirm snapshots are:
* derived API views
* not competing sources of truth
* builder-owned exports

---

# 3. Compatibility Check

## 3.1 Public API stability

Do not unintentionally break public APIs.

## 3.2 Single-role fast path

Families without role metadata must still normalize to:
* `node:default`
* `edge:default`

## 3.3 Legacy family behavior

Do not require all families to adopt new snapshot-specific metadata.

---

# 4. Scope Check

## 4.1 Allowed files only

List files you expect to touch.
Confirm they are allowed in the phase.

## 4.2 Forbidden files untouched

List the files you must not touch in the phase.

## 4.3 No hidden refactor

Do not do broad cleanup or unrelated renaming.

## 4.4 No speculative redesign

Implement only settled decisions from the plan/checkpoints.

---

# 5. Snapshot-Specific Checklist

## 5.1 Runtime snapshot boundary

Does the new or modified snapshot compile from existing builder-owned state rather than inventing a new ownership model?

## 5.2 Narrow optimizer view

If touching optimizer-facing snapshot code, does it stay narrower than full builder internals?

## 5.3 Graph consistency

Can the snapshot contents be traced back to graph role ids and builder registries clearly?

## 5.4 Legacy fallback

Does the snapshot still work for default-role families?

## 5.5 Debuggability

Are new snapshot fields inspectable and explicit?

---

# 6. Validation Planning Check

Before editing, identify:
* what will be validated
* how it will be validated
* what would count as failure
* when to stop and escalate

---

# 7. Executor Self-Review

Before finishing, explicitly confirm:

* phase scope was respected
* ownership boundaries preserved
* no future-phase leakage
* null-edge semantics remain correct
* builder/framework boundary remains correct
* changed files match phase allowance
* validations were actually performed
* `WORKLOG.md` updated
* `STATUS.md` updated
