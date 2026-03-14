# CHECKLIST.md

## Purpose

This checklist is the mandatory pre-flight and post-change safety list for Codex work in the optimizer reconstruction branch.

Use this together with:

* `AGENTS.md`
* `PLAN.md`
* `PHASE_SPEC.md`
* `ARCHITECTURE.md`
* `ARCHITECTURE_DECISIONS.md`
* `CODEX_CONTEXT.md`
* `STATUS.md`
* `WORKLOG.md`
* `SNAPSHOT_API_HANDOFF.md`
* `OPTIMIZER_DISCUSSION_MEMORY.md`
* `OPTIMIZER_TODO_ROADMAP.md`

---

# 1. Before Doing Any Work

## 1.1 Read required documents

Confirm you have read:

* `AGENTS.md`
* `PLAN.md`
* `PHASE_SPEC.md`
* `ARCHITECTURE.md`
* `ARCHITECTURE_DECISIONS.md`
* `CODEX_CONTEXT.md`
* `STATUS.md`
* `WORKLOG.md`
* `SNAPSHOT_API_HANDOFF.md`
* `OPTIMIZER_DISCUSSION_MEMORY.md`
* `OPTIMIZER_TODO_ROADMAP.md`
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

## 2.4 Snapshot seam rule

Confirm the optimizer is consuming the snapshot seam rather than arbitrary builder internals.

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

## 2.8 Semantics before geometry

Confirm legality is determined from semantics first.

Geometry may rank legal candidates, but must not decide legality.

---

# 3. Compatibility Check

## 3.1 Public API stability

Do not unintentionally break public APIs.

## 3.2 Single-role fast path

Families without role metadata must still normalize to:
* `node:default`
* `edge:default`

## 3.3 Legacy optimizer path

Do not remove or silently change the old optimizer path before the new path is proven.

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

Implement only settled decisions from the handoff docs and the active phase plan.

---

# 5. Optimizer-Specific Checklist

## 5.1 Node-local contract

If the phase touches local placement, does it compile a clear node-local contract from the snapshot?

## 5.2 Legal correspondence

If the phase touches matching, is legality determined from slot/path semantics rather than geometry?

## 5.3 SVD layer

If the phase touches rigid placement, is SVD/Kabsch used only after legality is known?

## 5.4 Local refinement

If the phase touches refinement, does it remain inside the legal correspondence neighborhood?

## 5.5 Null-edge behavior

If the phase touches null edges, does it preserve explicit null-edge semantics?

## 5.6 Old-path safety

If the phase touches integration, is the old path still available?

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
