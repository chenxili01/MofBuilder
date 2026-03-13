You are the executor for MOFBuilder development.Prefer minimal code changes required to satisfy the Phase Contract.
Run tests using scripts/run_tests.sh according to the Test Execution Rule in AGENTS.md.


Read these files first:
- PLANS.md
- AGENTS.md
- ARCHITECTURE.md
- CODEX_CONTEXT.md
- STATUS.md
- WORKLOG.md

Follow the current Phase Contract recorded in WORKLOG.md for the phase and checkpoint
identified by STATUS.md.

Executor scope rule:

You may modify ONLY:
- the source files explicitly allowed by the current Phase Contract
- the test files explicitly allowed by the current Phase Contract
- WORKLOG.md for implementation and handoff updates only
- STATUS.md for checkpoint/status updates only

You must NOT modify:
- PLANS.md
- AGENTS.md
- ARCHITECTURE.md
- CODEX_CONTEXT.md
- any source or test file outside the current Phase Contract
- any other files

------------------------------------------------
Task
------------------------------------------------

Execute the current phase according to the Phase Contract.

Steps:

1. Determine the current phase and checkpoint from STATUS.md.

2. In WORKLOG.md:
   - locate the matching phase section automatically
   - locate the active checkpoint automatically
   - read the Phase Contract recorded for that checkpoint
   - use that contract as the execution boundary

3. Before coding, provide a short execution preflight containing:
   - current phase
   - current checkpoint
   - goal
   - files you are allowed to modify
   - key invariants you must preserve
   - confirmation that you will stay within scope

4. Implement only what is required by the current Phase Contract.

5. Add or update only the tests required by the current Phase Contract.

6. After implementation:
   - update the implementation checkpoint in WORKLOG.md
   - record files changed, tests added/run, key decisions, and any blockers
   - update STATUS.md to the next appropriate checkpoint/state

------------------------------------------------
Hard rules
------------------------------------------------

- Follow AGENTS.md architecture locks.
- Preserve the single-role path as the default/base case.
- Do not infer chemistry from topology role labels unless the contract explicitly allows it.
- Do not expand scope into later phases.
- Do not silently refactor forbidden modules.
- If implementation reveals a conflict with:
  - PLANS.md
  - the Phase Contract
  - architecture locks
  - graph-state invariants
  - role-model invariants
  then STOP, record the conflict in WORKLOG.md and STATUS.md, and do not continue coding past that boundary.

- Never delete existing WORKLOG history.
- Only update the active implementation/handoff checkpoint and STATUS.md.

------------------------------------------------
Output
------------------------------------------------

Return:

1. Execution preflight
2. Implementation approach
3. Code changes
4. Tests added or updated
5. WORKLOG.md updates
6. STATUS.md updates
7. Any conflicts or reasons for stopping
8. 
Phase-1 reminder:
- preserve current single-role scalar outputs
- attach deterministic node_role_id / edge_role_id annotations to FrameNet.G
- do not modify builder/runtime/optimizer/supercell/writer/defects/MD code
- do not redesign graph APIs beyond attaching stable role annotations

Follow AGENTS.md architecture locks.
