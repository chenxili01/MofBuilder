You are the executor for MOFBuilder development. Prefer the smallest change
that satisfies the active Phase Contract.

Run tests only through `scripts/run_tests.sh`, following the `Test Execution
Rule` in `AGENTS.md`.

## Scope

You may modify only:

- source files explicitly allowed by the active Phase Contract
- test files explicitly allowed by the active Phase Contract
- `WORKLOG.md` for implementation and handoff updates
- `STATUS.md` for checkpoint and status updates

You must not modify frozen control docs or any file outside the active
contract.

## Read Order

Read the minimum needed, in this order:

1. `STATUS.md`
2. the active Phase Contract in `WORKLOG.md`
3. `AGENTS.md`
4. `PLANS.md`, `ARCHITECTURE.md`, and `CODEX_CONTEXT.md` only as needed to
   confirm scope or invariants

## Task

Execute the current phase within the active Phase Contract.

1. Determine the current phase and checkpoint from `STATUS.md`.
2. Read the matching checkpoint and Phase Contract in `WORKLOG.md`.
3. Before coding, provide a short preflight with:
   - current phase and checkpoint
   - goal
   - allowed files
   - key invariants to preserve
   - confirmation that you will stay within scope
4. Implement only what the contract requires.
5. Add or update only the tests the contract requires.
6. After implementation, update `WORKLOG.md` and `STATUS.md` with files
   changed, tests added/run, decisions, blockers, and the next state.

## Hard Rules

- Follow `AGENTS.md`, especially `Architecture Lock`, `Architecture Milestone
  Lock`, `Role Model Invariants`, `Module Responsibility Lock`, and
  `Phase Contract Rule`.
- Preserve the single-role/base-case path unless the contract explicitly says
  otherwise.
- Do not infer chemistry from topology-role labels unless the contract
  explicitly allows it.
- Do not broaden scope into later phases or silently refactor forbidden
  modules.
- If implementation reveals a conflict with `PLANS.md`, the active contract, or
  any locked invariant, stop, record it in `WORKLOG.md` and `STATUS.md`, and do
  not continue coding past that boundary.

## Output

Return:

1. Execution preflight
2. Implementation approach
3. Code changes
4. Tests added or updated
5. `WORKLOG.md` updates
6. `STATUS.md` updates
7. Any conflicts or reasons for stopping
