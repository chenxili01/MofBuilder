used a real phase roadmap in PLAN.md, with sequential named phases, per-phase primary goals, executor rules, and a stop rule. That structure is explicit in your previous PLAN.md and PHASE_SPEC.md.
restored the “Phase Roadmap” section style from the optimizer branch, where each phase has a title and a primary goal block.

restored the executor sequencing rule: phases are implemented sequentially, one phase only.

restored the PLAN ↔ PHASE_SPEC pairing, where PLAN.md describes roadmap intent and PHASE_SPEC.md defines implementation boundaries.

kept the same governance style from AGENTS.md, including planner/executor split and conservative scope control.



Follow the exact working style of my previous branch control-doc workflow.

I want a fresh control-doc set for a new branch, but it must reuse the same routine and discipline as my previous branch development style:

- Planner / Executor split only
- planner is planning-only
- executor is implementation-only
- executor implements one phase only
- PLAN.md must contain a real sequential phase roadmap
- PHASE_SPEC.md must map each phase to:
  - allowed modules
  - required work
  - forbidden changes
  - completion criteria
- STATUS.md must show:
  - active phase
  - checkpoint
  - status
  - next-step owner
- WORKLOG.md must be append-only and use the same structured entry format
- CHECKLIST.md must enforce invariants, scope control, compatibility, and self-review
- preserve my architecture style:
  - graph/topology is source of truth
  - builder owns semantics
  - optimizer consumes compiled semantics
  - framework remains role-agnostic
  - backward compatibility preserved
  - semantics before geometry
  - null edge distinct from zero-length real edge
- no broad redesign
- no future-phase leakage
- no vague summaries in place of real phases
- no simplified docs
- make it repo-operational and concrete

When drafting, first summarize the branch objective in one paragraph, then produce the files in the same style and level of detail as my previous branch, not a lighter version.



Use my previous branch routine exactly: full phased control-doc workflow, not a simplified draft. Mirror the same PLAN.md + PHASE_SPEC.md + STATUS.md + WORKLOG.md discipline, with strict planner/executor separation, invariant-first architecture, one-phase-only execution, explicit validations, and honest blocker reporting.



Task
Fix PdbReader anchor filtering for typed atoms.

Scope
basic.py, pdb_reader.py, tests

Constraints
Do not change builder/runtime schema.
Preserve legacy X behavior.

Done
Typed atoms are preserved.
Legacy X still works.
Regression test added.

Use my structured development style, but choose the lightest control level that still preserves architecture.

Default to planner + executor only.
Skip reviewer unless the task changes architecture, ownership boundaries, or rollout safety.

If the task is small, use a lightweight format:
- task
- scope
- constraints
- validation
- completion note

If the task is branch-sized or compatibility-sensitive, use my full routine:
- PLAN.md with real phases
- PHASE_SPEC.md
- STATUS.md
- WORKLOG.md
- CHECKLIST.md

Preserve my core style:
- graph/topology is source of truth
- builder owns semantics
- optimizer consumes compiled semantics
- framework remains role-agnostic
- backward compatibility preserved
- semantics before geometry
- no future-phase leakage
- honest validation and blockers

Optimize for full control with minimal unnecessary ceremony.


For tiny tasks

Use spec-first lightweight

For bug fixes

Use issue-driven

For uncertain design work

Use spike-then-harden

For large architecture branches

Use your current planner + executor phased control-doc style

That would give you the benefits of other styles without losing the control you value.