# CHECKLIST.md

Confirm:
- Phase scope stayed documentation-only.
- Typed attachment terminology is recorded: `slot_type`, `source_atom_type`,
  `resolved anchor`, and `legacy literal-X compatibility`.
- Control docs state the failure class directly:
  valid typed attachment atoms must not be collapsed or dropped into a universal
  literal-`X` assumption before builder-owned resolution.
- Control docs state the ownership seam directly:
  raw fragment atom typing is upstream input, builder resolves `source_atom_type`
  from slot/path semantics and compiles resolved anchors, optimizer consumes only
  those compiled anchors.
- Control docs state the forbidden assumption directly:
  fragment readers, builder compilation, and optimizer helpers must not treat
  literal `X` as the only attachable atom class.
- Control docs preserve migration compatibility:
  legacy literal-`X` families remain valid, but they are not the universal
  attachment model.
