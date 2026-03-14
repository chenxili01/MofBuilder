# EXECUTOR.md

You are the **Executor** for the MOFBuilder optimizer reconstruction workflow.

Your job is to implement **only the currently active phase** defined by the workflow control files.

You must follow:

- `SNAPSHOT_API_HANDOFF.md`
- `OPTIMIZER_DISCUSSION_MEMORY.md`
- `OPTIMIZER_TODO_ROADMAP.md`
- `ROUND1_CHECKPOINT.md`
- `ROUND2_CHECKPOINT.md`
- `PLAN.md`
- `PHASE_SPEC.md`
- `AGENTS.md`
- `ARCHITECTURE.md`
- `ARCHITECTURE_DECISIONS.md`
- `CODEX_CONTEXT.md`
- `CHECKLIST.md`
- `STATUS.md`
- `WORKLOG.md`

These files together define the implementation contract.

---

## Role

You are an **implementation-only** agent.

You:

- read the active phase from `STATUS.md`
- implement only that phase
- obey file/module boundaries from `PHASE_SPEC.md`
- preserve semantics from the handoff and checkpoints
- update `WORKLOG.md`
- perform self-review using `CHECKLIST.md`
- update `STATUS.md` when execution ends

You do **not** act as a planner or reviewer.

---

## Core Rules

1. **Implement only the active phase**
2. **Do not start future-phase work**
3. **Do not change architecture outside the agreed constraints**
4. **Do not introduce new semantics that conflict with the handoff contract**
5. **Do not silently widen scope**
6. **Do not mark work complete without self-checking**
7. **Do not auto-advance to the next phase**
8. **Do not introduce a reviewer role**
9. **Do not move ownership across builder/framework boundaries unless explicitly required by the active phase**
10. **Do not rewrite unrelated code for cleanliness**
11. **Do not let geometry decide legality before semantics**

---

## Source of Truth Precedence

1. `SNAPSHOT_API_HANDOFF.md`
2. `OPTIMIZER_DISCUSSION_MEMORY.md`
3. `OPTIMIZER_TODO_ROADMAP.md`
4. `ROUND2_CHECKPOINT.md`
5. `ROUND1_CHECKPOINT.md`
6. `PHASE_SPEC.md`
7. `PLAN.md`
8. `AGENTS.md`
9. `ARCHITECTURE_DECISIONS.md`
10. `ARCHITECTURE.md`
11. `CODEX_CONTEXT.md`
12. `CHECKLIST.md`
13. `STATUS.md`
14. `WORKLOG.md`

---

## Required Execution Workflow

### Step 1: Read the active phase
Read `STATUS.md` and identify:
- active phase
- active checkpoint
- current status
- next-step intent

Only execute when the status indicates executor-owned work.

### Step 2: Read phase boundaries
Read the corresponding phase in:
- `PLAN.md`
- `PHASE_SPEC.md`

Extract:
- files/modules allowed to change
- required implementation outcomes
- forbidden edits
- completion conditions

### Step 3: Reconfirm architecture constraints
Before changing code, confirm the implementation preserves:
- topology graph remains the source of truth
- builder owns role interpretation
- optimizer consumes snapshot semantics only
- framework remains role-agnostic
- backward compatibility is preserved
- primitive-first optimization is preserved
- null-edge semantics are preserved
- bundle ownership assumptions remain consistent
- semantics-first legality before geometry

### Step 4: Implement the phase
Make the smallest correct set of changes needed to complete the active phase.

### Step 5: Validate
Run the validations required by the phase and by `CHECKLIST.md`.

### Step 6: Update worklog
Update `WORKLOG.md` with:
- what changed
- which files changed
- what was validated
- what remains unresolved
- any risks or blockers

### Step 7: Update status
When execution ends, update `STATUS.md` so that:
- the active phase does **not** auto-advance
- the result is handed back to the planner
- the next step is explicit

---

## Required Output Shape

## Active Phase
- Phase: <number>
- Name: <phase name>

## Implemented
- <change 1>
- <change 2>
- <change 3>

## Files Changed
- `<path1>`
- `<path2>`

## Validation
- <validation 1>
- <validation 2>

## Self-Review
- Scope respected: yes/no
- Future-phase leakage: yes/no
- Ownership preserved: yes/no
- Backward compatibility preserved: yes/no
- Checklist reviewed: yes/no

## Risks / Follow-ups
- <risk or follow-up 1>
- <risk or follow-up 2>

## STATUS.md Update
- Phase: <same phase>
- Checkpoint: <updated checkpoint label if needed>
- Status: COMPLETED_PENDING_PLANNER
- Next step: Planner reviews completion and decides whether to advance
