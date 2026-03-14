# Typed Attachment Hardening Architecture

Existing code assumes a universal attachment atom class such as `X`.
This branch replaces that assumption with typed attachment sources while preserving
legacy literal-`X` compatibility during migration.

## Phase 1 Contract

The Phase 1 failure class is explicit:
valid typed attachment atoms must not be collapsed, filtered, or dropped into a
universal literal-`X` assumption before builder-owned resolution.

The required terminology for this branch is:

- `slot_type`: semantic attachment meaning derived from slot/path rules
- `source_atom_type`: fragment-local attachment lookup class preserved from raw fragment atoms
- `resolved anchor`: builder-compiled attachment record produced after source-type resolution
- `legacy literal-X compatibility`: temporary support for existing literal-`X` families without
  treating them as the universal attachment model

## Ownership Seam

Raw fragment atom typing is upstream input only.
Builder-owned runtime and snapshot compilation must:

- preserve typed attachment inputs long enough to resolve `source_atom_type`
- resolve attachment legality from slot/path semantics first
- compile optimizer-consumable `resolved anchor` records

The optimizer must consume only builder-compiled resolved-anchor semantics.
It must not reconstruct attachment meaning from raw fragment atom buckets or assume
all attachable atoms are literal `X`.

## Phase 1 Scope

Phase 1 is documentation-only.
Production code, tests, runtime payloads, and runtime schemas remain unchanged in this phase.
