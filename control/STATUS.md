Phase: P1.0 – Planning / remediation
State: BLOCKED

Summary:
Active remediation contract authorizes workflow/WORKLOG.md and workflow/STATUS.md,
but the authoritative control documents are the root files WORKLOG.md and STATUS.md.

Current issue:
The nested workflow copies do not exist, and the executor sandbox is restricted to
/workflow, so it cannot update the real root control docs.

Decision:
Execution stopped correctly under contract rules rather than reinterpreting scope.

Required next action:
Correct the remediation contract to authorize the root control docs:
- WORKLOG.md
- STATUS.md

Notes:
- No runtime, implementation, or test changes were performed in this remediation pass.
- This is a contract/path mismatch, not a code-path failure.
