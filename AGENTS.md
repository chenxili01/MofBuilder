# AGENTS.md
## Purpose

This repository uses a **two-agent Codex workflow** to develop MOFBuilder in controlled phases.

The agents are:

1. **Planner**
2. **Executor**

The workflow ensures:

* architectural stability
* controlled phase-based development
* minimal regression risk
* clear design authority

---

# Repository Governance Model

Development is **plan-driven**.

The authoritative documents are:

| File                   | Purpose                           |
| ---------------------- | --------------------------------- |
| `PLAN.md`              | Active development roadmap        |
| `PLAN_codex_record.md` | Frozen historical planning record |
| `ARCHITECTURE.md`      | System architecture description   |
| `CHECKLIST.md`         | Development safety checklist      |
| `CODEX_CONTEXT.md`     | Context summary for Codex         |
| `STATUS.md`            | Current repository state          |
| `WORKLOG.md`           | Chronological development log     |

Agents **must read these files before acting**.

---

# Agent Roles

## 1. Planner

Planner agents **design development phases**.

Planner responsibilities:

* interpret `PLAN.md`
* design phase structure
* define invariants
* define scope boundaries
* define validation rules
* define stop rules

Planner agents **do not implement code**.

Planner outputs include:

* updates to `PLAN.md`
* planning checkpoints
* architecture clarifications
* roadmap adjustments

Planner agents must ensure:

* compatibility with `PLAN_codex_record.md`
* architectural consistency
* phase isolation
* conservative scope control

---

## 2. Executor

Executor agents **implement one phase at a time**.

Executor responsibilities:

* read the current phase in `PLAN.md`
* implement only the allowed scope
* respect module ownership boundaries
* preserve invariants
* maintain compatibility with existing APIs

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
5. Tests or validation checks for the phase pass.

The executor must summarize:

* files modified
* invariants verified
* unresolved concerns

before concluding the task.

---

# Development Phases

All development must follow phases defined in `PLAN.md`.

Each phase must define:

* objective
* allowed modules
* forbidden modules
* invariants
* validation goals
* exit criteria
* rollback risks

Executor **must not move to the next phase automatically.**

Each phase requires a new planning step.

---

# Architectural Invariants

These invariants must not be violated unless a future planning phase explicitly allows it.

---

## Graph Grammar

Current development cycle supports only:

```
V-E-V
V-E-C
```

Where:

```
V = node center
C = linker center
E = connector edge
```

Other graph patterns are not allowed in this cycle.

---

## Role Identity Model

Roles have two identities.

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
node:VB
node:CA
edge:EA
edge:EB
```

Only the **prefix** (`V`, `C`, `E`) has universal meaning.

Suffix letters are **family-local identifiers**.

---

## Bundle Ownership

`C*` roles own linker bundles.

A multitopic linker is reconstructed from:

```
C center
+ incident E connectors
+ optional coordination resolve
```

`V*` nodes do not own bundles.

---

## Slot Matching

Attachment slots are typed.

Examples:

```
XA
XB
```

Matching must preserve slot types.

Slot systems must support:

* stable attachment indices
* slot types
* optional cyclic ordering

---

## Null Edge Semantics

Null edges are explicit `E*` roles with metadata.

Canonical representation:

```
two overlapping anchor points
```

Important rule:

```
null edge != zero-length real edge
```

---

## Resolve Behavior

Resolve behavior is controlled by family metadata.

Possible modes:

```
alignment_only
ownership_transfer
```

Resolve occurs **after optimization and before final merge**.

---

## Provenance

All ownership changes must record provenance metadata.

Provenance must survive until final framework materialization.

This enables:

* unsaturated site detection
* termination placement
* defect modeling
* debugging

---

# Module Ownership

Modules have defined responsibilities.

---

## MofTopLibrary

Owns passive family metadata:

* role declarations
* connectivity metadata
* path rules
* null-edge policies
* fragment lookup hints

---

## FrameNet

Owns topology construction:

* build net graph
* stamp role ids
* compute cyclic ordering around `C`
* attach ordering metadata

Also owns **initial topology validation**.

---

## Builder

Acts as the **compilation manager**.

Responsibilities:

* ingest family metadata
* normalize role registries
* resolve payloads
* apply null fallback policies
* validate slot/path compatibility
* prepare resolve scaffolding

---

## Optimizer

Performs geometry optimization.

Must understand:

```
V-E-C
V-E-V
slot matching
null-edge constraints
```

---

## Linker

Contains helper logic for linker splitting.

Legacy behavior must remain supported.

---

## Supercell

Handles replication.

Replication must preserve:

* role ids
* bundle ids
* provenance
* resolve metadata

Replication occurs **after primitive-cell optimization**.

---

## Framework

Performs final structure materialization.

Responsibilities:

* merge atoms
* apply resolve ownership
* produce final structure
* mark unsaturated sites
* define termination anchors

---

## Defects / Termination

Consume provenance output to determine:

* unsaturated sites
* termination placement

---

## Write

Handles output.

Must support:

* standard structure export
* optional debug export

Debug export may include:

```
role ids
bundle ids
provenance
unsaturated markers
```

---

# Compatibility Requirements

Early phases must preserve:

* existing public APIs
* single-role workflows
* current topology families without new metadata

Role-aware features must remain **optional paths** initially.

---

# Validation Policy

FrameNet must provide a validation function checking:

* role prefix legality
* graph grammar validity
* connectivity consistency
* slot metadata
* null-edge declarations

Builder must call this validation before compilation.

---

# Execution Safety Rules

Agents must follow these rules:

1. Never bypass `PLAN.md`.
2. Never silently expand scope.
3. Never modify architecture without planner approval.
4. Never break APIs unless the plan explicitly allows it.
5. Stop and report if architectural contradictions appear.

---

# Codex Workflow

Development flow:

```
Planner → Executor
```

Planner defines phase scope.
Executor implements that phase and performs a self-review.

---

# Stop Rule

Agents must stop if:

* architecture becomes unclear
* invariants would be violated
* scope is exceeded
* plan instructions conflict with repository state
