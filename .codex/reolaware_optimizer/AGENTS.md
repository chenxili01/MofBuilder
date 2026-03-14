# AGENTS.md

## Purpose

This repository branch uses a **two-agent Codex workflow** to develop the optimizer reconstruction work in controlled phases.

The agents are:

1. **Planner**
2. **Executor**

The workflow ensures:

* architectural stability
* narrow phase-based development
* minimal regression risk
* clean builder/optimizer ownership boundaries
* legality-first optimizer reconstruction

---

# Repository Governance Model

Development is **plan-driven**.

The authoritative documents are:

| File                        | Purpose                                |
| --------------------------- | -------------------------------------- |
| `PLAN.md`                   | Active development roadmap             |
| `PHASE_SPEC.md`             | Phase boundaries / allowed modules     |
| `ARCHITECTURE.md`           | Branch architecture description        |
| `ARCHITECTURE_DECISIONS.md` | Architectural decisions and rationale  |
| `CHECKLIST.md`              | Development safety checklist           |
| `CODEX_CONTEXT.md`          | Context summary for Codex              |
| `STATUS.md`                 | Current repository state               |
| `WORKLOG.md`                | Chronological development log          |
| `SNAPSHOT_API_HANDOFF.md`   | Upstream snapshot seam contract        |
| `OPTIMIZER_DISCUSSION_MEMORY.md` | Reasoning memory for optimizer branch |
| `OPTIMIZER_TODO_ROADMAP.md` | Practical future optimizer roadmap     |
| `ROUND1_CHECKPOINT.md`      | Semantic contract reference            |
| `ROUND2_CHECKPOINT.md`      | Ownership / phase boundary reference   |

Agents **must read these files before acting**.

---

# Agent Roles

## 1. Planner

Planner agents **design development phases**.

Planner responsibilities:

* interpret `PLAN.md`
* map the active phase from `STATUS.md`
* define exact phase-bounded tasks
* define validation rules
* define stop rules
* keep the branch aligned with the snapshot handoff contract

Planner agents **do not implement code**.

Planner outputs include:

* updates to `STATUS.md`
* phase-bounded implementation instructions
* architecture clarifications
* roadmap adjustments within the active branch goal

Planner agents must ensure:

* consistency with `SNAPSHOT_API_HANDOFF.md`
* consistency with `OPTIMIZER_DISCUSSION_MEMORY.md`
* consistency with `OPTIMIZER_TODO_ROADMAP.md`
* consistency with `ROUND1_CHECKPOINT.md`
* consistency with `ROUND2_CHECKPOINT.md`
* conservative scope control

---

## 2. Executor

Executor agents **implement one phase at a time**.

Executor responsibilities:

* read the current phase in `STATUS.md`
* implement only the allowed scope
* respect module ownership boundaries
* preserve invariants
* maintain compatibility with existing APIs
* update `WORKLOG.md`
* update `STATUS.md`

Executors must:

* avoid touching modules not listed in the phase scope
* keep code changes minimal and isolated
* follow validation rules defined in the phase
* stop if scope becomes unclear

---

# Executor Self-Review Requirement

Because this workflow does not use a reviewer agent, **Executor must perform a self-review step before completion.**

Executor must verify:

1. All phase invariants remain valid.
2. No forbidden modules were modified.
3. Existing APIs still function unless the phase explicitly allows changes.
4. No architectural invariants were violated.
5. Tests or validation checks for the phase pass, or blockers are documented honestly.

The executor must summarize:

* files modified
* invariants verified
* unresolved concerns

before concluding the task.

---

# Architectural Invariants

These invariants must not be violated unless a future planning phase explicitly allows it.

## Graph Grammar

Current development cycle supports only:

```
V-E-V
V-E-C
```

## Role Identity Model

Roles have two identities:

Family alias:
```
VA
VB
CA
EA
EB
```

Canonical runtime identity:
```
node:VA
node:CA
edge:EA
edge:EB
```

Only the **prefix** has universal meaning.

Suffix letters are family-local identifiers.

## Graph Role Source of Truth

Role ids remain on graph elements:

```
G.nodes[n]["node_role_id"]
G.edges[e]["edge_role_id"]
```

## Builder Ownership

Builder remains the owner of:
- role interpretation
- role registries
- bundle compilation
- resolve scaffolding
- snapshot compilation

## Optimizer Role

Optimizer consumes a narrowed semantic snapshot.

Optimizer may reconstruct local placement behavior, but it does not become the owner of role meaning.

## Framework Boundary

Framework remains role-agnostic in this branch.

## Primitive-First Optimization

Optimization remains primitive-cell-first.

## Null-Edge Distinction

Preserve:

```
null edge != zero-length real edge
```

---

# Execution Safety Rules

Agents must follow these rules:

1. Never bypass `PLAN.md`.
2. Never silently expand scope.
3. Never modify architecture without planner approval.
4. Never break APIs unless the active phase explicitly allows it.
5. Never move role interpretation out of builder.
6. Never let optimizer inspect arbitrary builder internals directly.
7. Never remove the old optimizer path before the new path is proven.
8. Never use geometry to decide legality before semantics.
