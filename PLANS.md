# PLAN.md

## 1. Title

Role-Aware Reticular Graph Planning Cycle for `role-aware-reticular-graph`

## 2. Status / intent

This is the active planning document for the `role-aware-reticular-graph`
branch.

`PLAN_codex_record.md` remains the frozen historical implementation record for
the completed Phase 1-8 role-aware baseline and must not be mutated during this
cycle except for context reading.

This document is planning-only. It defines the next architecture-preserving
planning cycle and does not authorize runtime changes outside a later phase
contract.

## 3. Executive summary

The repository already has a frozen role-aware baseline built around
graph-stored `node_role_id` / `edge_role_id`, builder-owned registries, the
locked staged pipeline, and current single-role compatibility behavior. This
planning cycle does not replace that baseline. It sharpens it into a stricter
role-aware reticular graph model with an explicit G2 graph grammar and clearer
module ownership for topology metadata, validation, bundle compilation,
optimization constraints, resolve timing, provenance, unsaturated-site
generation, and debug export.

The goal is to make later executor threads work from a concrete spec instead of
inventing semantics module-by-module. The plan therefore fixes the graph
grammar, naming model, ownership boundaries, phase contracts, validation goals,
stop rules, and failure-containment rules up front while preserving the current
public workflow and single-role fast path.

## 4. Scope

This planning cycle is limited to medium-broad G2 scope.

Supported topology path grammar in this cycle:

- `V-E-V`
- `V-E-C`

Within that grammar, this cycle covers:

- node-center roles (`V*`)
- linker-center roles (`C*`)
- connector-edge roles (`E*`)
- family-local role aliases plus canonical runtime ids
- bundle ownership on `C*`
- typed attachment slots
- stable attachment indices
- optional cyclic order, mandatory when a family reconstructs multitopic
  linkers
- ordered endpoint/path semantics
- explicit null-edge semantics
- family-controlled unresolved-edge fallback
- family-controlled resolve behavior
- provenance requirements from build preparation through final merge

This cycle does not broaden the grammar beyond the legal G2 paths above.

## 5. Non-goals

Early phases explicitly do not attempt:

- arbitrary reticular graph grammars beyond `V-E-V` and `V-E-C`
- direct support for `V-V`, `C-C`, `E-E`, or longer unrestricted path classes
- force-field redesign
- MD workflow redesign
- new external dependencies beyond the current stack
- aggressive cross-module refactors
- public API renaming or signature changes
- reordering or replacing the locked pipeline
- silent unification of null edges and zero-length real edges

## 6. Architecture invariants

The following invariants are locked for this cycle unless a later explicit plan
revision says otherwise:

- The locked pipeline remains intact:
  `MofTopLibrary.fetch(...) -> FrameNet.create_net(...) ->
  MetalOrganicFrameworkBuilder.load_framework() ->
  MetalOrganicFrameworkBuilder.optimize_framework() ->
  MetalOrganicFrameworkBuilder.make_supercell() ->
  MetalOrganicFrameworkBuilder.build()`.
- Graph-state names remain intact:
  `G`, `sG`, `superG`, `eG`, `cleaved_eG`.
- Graph role ids remain the topology source of truth.
- Role identifiers are the only topology classification mechanism at runtime.
- Runtime registries remain the payload/config resolution source of truth.
- `FrameNet.G.nodes[n]["node_role_id"]` and
  `FrameNet.G.edges[e]["edge_role_id"]` remain canonical runtime storage
  locations for role identity.
- Downstream modules must not recompute role ids from chemistry or invent local
  competing role maps.
- Existing public APIs remain working in early phases, including
  `MetalOrganicFrameworkBuilder`, `Framework`, and current builder/framework
  method names.
- Single-role workflows remain the default fast path.
- Later implementations must keep the single-role path cheap when registry size
  is one and avoid unnecessary per-instance dynamic overhead in tight loops.
- Families without new reticular metadata must continue to work.
- Existing additive role-aware behavior already in the codebase must be treated
  as baseline compatibility, not discarded.
- Old families without new metadata must continue to normalize into existing
  single-role-compatible behavior.
- Current linker splitting logic in `linker.py` remains a supported helper/base
  path rather than being hard-deleted.
- Primitive-cell optimization followed by translation-based supercell
  replication remains the intended performance path.
- Unresolved new metadata must fail clearly, not silently, unless the family
  policy explicitly allows null fallback for the relevant edge role.
- No new dependency may be introduced for schema validation or graph handling
  in early phases.

## 7. Semantic model summary

This cycle adopts the settled Round 1 semantic contract:

- Only the prefixes `V`, `C`, and `E` have universal meaning.
- `V*` roles are node-center anchors.
- `C*` roles are linker-center anchors and bundle owners.
- `E*` roles are connector-edge anchors.
- The only legal ordered path types are `V-E-V` and `V-E-C`.
- Graph roles are conceptual topology anchors, not inherently fully
  materialized chemistry.
- Runtime payload kinds remain separate from graph-role meaning and may be
  atomic, X-point, orientation, or null payloads.
- `C*` always owns the linker bundle.
- Matching must preserve slot type before geometry matching.
- Ordered endpoint semantics matter; `V-E-C` is not interchangeable with
  `C-E-V`.
- A null edge is still an `E*` role, explicitly represented as a runtime edge
  payload with duplicated zero-length anchors.
- A null edge and a zero-length real edge are distinct metadata concepts.
- Unresolved-edge fallback is controlled by family policy, not by silent global
  default.
- Resolve behavior is family-controlled and must support at least
  `alignment-only` and `ownership-transfer`.
- Borrowed coordination-group use during resolve is permitted where required by
  family chemistry and must preserve provenance.
- Provenance must survive through final merged structure materialization.
- Rod-like or chain-like overlap behavior must be represented through explicit
  edge semantics rather than implicit overlap guessing.

## 8. Role naming model

Prefix semantics:

- `V*` means node-center role class
- `C*` means linker-center role class
- `E*` means edge role class

Suffix semantics:

- The suffix is a family-local alias only.
- `VA`, `VB`, `VC`, `CA`, `CB`, `EA`, and `EB` are meaningful only inside the
  family that declares them.
- `VA` is not globally special across families.

Identity model:

- Each role has a family alias, such as `VA`, `CA`, `EA`, `EB`.
- Each role also has a canonical runtime id used in graph/runtime plumbing.
- Canonical id mapping in this cycle is:
  - `V*` aliases map to `node:<alias>`
  - `C*` aliases map to `node:<alias>`
  - `E*` aliases map to `edge:<alias>`

Examples:

- `VA -> node:VA`
- `CA -> node:CA`
- `EA -> edge:EA`
- `EB -> edge:EB`

The alias remains family-facing metadata. The canonical id remains the runtime
key stored on graphs and used to index builder-owned registries.

## 9. Metadata grammar draft

The following field names are the provisional Phase 1 schema names for this
planning cycle. They are proposed now so that later Phase 2 work can implement
one concrete JSON-readable schema instead of improvising structure later.

Planned schema concepts:

- `schema_name`
- `schema_version`
- `family_name`
- `roles`
- `connectivity_rules`
- `path_rules`
- `bundle_rules`
- `slot_rules`
- `cyclic_order_rules`
- `edge_kind_rules`
- `resolve_rules`
- `unresolved_edge_policy`
- `fragment_lookup_hints`

Planned field meanings:

- `roles`
  - passive declarations of family aliases and canonical runtime ids
- `roles.<alias>.role_class`
  - one of `V`, `C`, `E`
- `roles.<alias>.canonical_role_id`
  - canonical runtime id, such as `node:VA` or `edge:EA`
- `connectivity_rules.<alias>.incident_edge_aliases`
  - ordered or multiplicity-bearing declaration of expected incident edge
    aliases
- `path_rules[]`
  - ordered endpoint rule for each edge alias; this is the canonical
    declaration of allowed ordered path shape
- `bundle_rules.<alias>.bundle_owner`
  - identifies the owner class for a linker bundle; in this cycle only `C`
    aliases may declare linker-bundle ownership
- `slot_rules.<alias>[]`
  - attachment-slot metadata with `attachment_index`, `slot_type`, and
    optional endpoint-side annotations
- `cyclic_order_rules.<alias>.ordered_attachment_indices`
  - canonical local order over attachment indices where required
- `edge_kind_rules.<edge_alias>.edge_kind`
  - `real` or `null`
- `edge_kind_rules.<edge_alias>.null_payload_model`
  - explicit declaration for null edges; planned canonical value is
    `duplicated_zero_length_anchors`
- `resolve_rules.<edge_alias>.resolve_mode`
  - at minimum `alignment_only` or `ownership_transfer`
- `unresolved_edge_policy.default_action`
  - phase-level family default for unresolved edge roles, expected to be either
    `error` or `allow_null_fallback`
- `unresolved_edge_policy.allowed_null_fallback_edge_aliases`
  - optional narrowed allow-list for edge aliases that may default to null
- `fragment_lookup_hints.<alias>`
  - passive lookup hints only; not runtime fragment payloads

Minimal canonical example snippet for a target family:

```json
{
  "schema_name": "mof_reticular_role_metadata",
  "schema_version": 1,
  "family_name": "MIL53_LIKE_ROD",
  "roles": {
    "VA": {
      "role_class": "V",
      "canonical_role_id": "node:VA"
    },
    "CA": {
      "role_class": "C",
      "canonical_role_id": "node:CA"
    },
    "EA": {
      "role_class": "E",
      "canonical_role_id": "edge:EA"
    },
    "EB": {
      "role_class": "E",
      "canonical_role_id": "edge:EB"
    }
  },
  "connectivity_rules": {
    "VA": {
      "incident_edge_aliases": ["EA", "EA", "EB", "EB"]
    },
    "CA": {
      "incident_edge_aliases": ["EA", "EA"]
    }
  },
  "path_rules": [
    {
      "edge_alias": "EA",
      "endpoint_pattern": ["VA", "EA", "CA"]
    },
    {
      "edge_alias": "EB",
      "endpoint_pattern": ["VA", "EB", "VA"]
    }
  ],
  "bundle_rules": {
    "CA": {
      "bundle_owner": "linker",
      "attachment_edge_aliases": ["EA", "EA"]
    }
  },
  "slot_rules": {
    "VA": [
      {"attachment_index": 0, "slot_type": "XA"},
      {"attachment_index": 1, "slot_type": "XA"},
      {"attachment_index": 2, "slot_type": "XB"},
      {"attachment_index": 3, "slot_type": "XB"}
    ],
    "CA": [
      {"attachment_index": 0, "slot_type": "XA"},
      {"attachment_index": 1, "slot_type": "XA"}
    ],
    "EA": [
      {"attachment_index": 0, "slot_type": "XA", "endpoint_side": "V"},
      {"attachment_index": 1, "slot_type": "XA", "endpoint_side": "C"}
    ],
    "EB": [
      {"attachment_index": 0, "slot_type": "XB", "endpoint_side": "V"},
      {"attachment_index": 1, "slot_type": "XB", "endpoint_side": "V"}
    ]
  },
  "cyclic_order_rules": {
    "CA": {
      "ordered_attachment_indices": [0, 1],
      "order_kind": "clockwise_local_topology"
    }
  },
  "edge_kind_rules": {
    "EA": {
      "edge_kind": "real"
    },
    "EB": {
      "edge_kind": "null",
      "null_payload_model": "duplicated_zero_length_anchors"
    }
  },
  "resolve_rules": {
    "EA": {
      "resolve_mode": "ownership_transfer"
    },
    "EB": {
      "resolve_mode": "alignment_only"
    }
  },
  "unresolved_edge_policy": {
    "default_action": "error",
    "allowed_null_fallback_edge_aliases": ["EB"]
  },
  "fragment_lookup_hints": {
    "VA": {
      "library": "nodes_database",
      "keywords": ["2c", "rod", "Al"]
    },
    "CA": {
      "library": "linker_input",
      "fragment_kind": "center"
    },
    "EA": {
      "library": "linker_input",
      "fragment_kind": "connector"
    },
    "EB": {
      "library": "family_metadata",
      "fragment_kind": "null_edge"
    }
  }
}
```

Schema guardrails:

- This schema is passive family metadata only.
- It is not a replacement for graph annotations or builder registries.
- It must remain JSON-readable with a lightweight Python validator.
- It must remain additive with respect to current families that only rely on
  `MOF_topology_dict`.

## 10. Canonical example family

Motivating exemplar: a MIL-53-like rod family with a multitopic-linker bundle.

Planned semantic reading of the exemplar:

- `VA`
  - a rod or chain repeat anchor on the inorganic node side
- `CA`
  - the linker-center bundle owner for the multitopic organic linker
- `EA`
  - a real connector edge joining the rod-side anchor to the linker-center
    bundle
- `EB`
  - a `V-E-V` null edge used to represent rod/chain continuity or overlap-like
    interpretation explicitly rather than by geometric guessing

What the example is meant to prove:

- the same family can legitimately use both `V-E-C` and `V-E-V`
- the linker bundle is owned by `CA`, not by `VA`
- slot typing and local order are family-level declarations, not optimizer
  inventions
- null edge behavior can coexist with real connector behavior in one family
- borrowed coordination-group semantics can be family-controlled without making
  graph roles chemically overloaded

What the example does not pre-commit:

- exact final JSON filename
- exact fragment-library filenames
- exact coordinate-transfer algorithm during resolve
- exact provenance serialization layout in final writer output

## 11. Module ownership matrix

| Module / surface | Owns in this cycle | Must not own in this cycle |
| --- | --- | --- |
| `MofTopLibrary` | Passive family metadata, role declarations, connectivity declarations, ordered path rules, edge-kind/null metadata, unresolved-edge family policy, lookup hints | Graph construction, runtime fragment resolution, geometry resolve, provenance commit |
| `FrameNet.create_net()` | Topology graph construction, graph role stamping, topology-derived slot/path metadata, local cyclic order around `C` nodes, storage of topology hints on graph objects | Chemistry resolve, null fallback policy, payload lookup, final provenance semantics |
| New `FrameNet` validation function | Validation of role prefixes/classes, legal G2 path grammar, connectivity consistency, slot/path metadata presence, ordering sanity, null-edge declaration consistency, structured status/errors/hints | Runtime payload resolution, chemistry inference, optimizer-stage resolve |
| `builder.py` | Family metadata ingestion, normalization, registry compilation, null fallback policy, call order for `FrameNet` validation, bundle-id compilation, resolve/provenance scaffolding, cross-module coordination | Topology parsing, topology role invention, writer-side final merge, defect inference from scratch |
| `linker.py` | Linker splitting and fragment-preparation helper behavior, reused as the base processor underneath role-aware handling | Family policy, graph grammar enforcement, bundle ownership definition |
| `optimizer.py` | Role-aware consumption of graph/registry inputs, slot-typed anchor matching, `V-E-C` vs `V-E-V` constraint handling, null-edge zero-length constraints | Canonical order invention, family metadata ownership, final ownership transfer/provenance commit |
| `supercell.py` | Semantic propagation through replication, preserving role ids, bundle membership, provenance scaffolding, unresolved/resolve-pending metadata through supercell expansion | Redefining topology semantics, chemistry resolve policy, writer-specific export logic |
| `framework.py` | Post-build materialization, resolve commit before final merge, provenance-preserving merged structure, unsaturated markers, termination-anchor data handoff | Topology parsing, family metadata validation, upstream role-id invention |
| `defects.py` | Consumption of explicit resolve/provenance output, unsaturated-site handling after materialization, defect mutation on already-resolved framework structures | Reconstructing provenance by guesswork, family policy definition, writer formatting |
| `termination.py` | Termination fragment preparation and explicit termination-anchor consumption | Topology semantics, unsaturated-site discovery from scratch |
| `write.py` | Normal output plus debug/checkpoint export of role ids, bundle ids, provenance, unsaturated markers | Topology validation, registry normalization, resolve policy |

Ownership lock for this cycle:

- `FrameNet` remains topology-owner.
- `builder.py` remains compilation manager and policy coordinator.
- `optimizer.py` remains placement/constraint consumer.
- `supercell.py` remains semantic propagation through replication.
- `framework.py` remains final materialization and merge owner.
- `defects.py`, `termination.py`, and `write.py` remain downstream consumers of
  explicit resolved state.

## 12. Legacy compatibility contract

The following compatibility rules are mandatory in early phases:

- Old public APIs remain working.
- Single-role workflows remain the default fast path.
- Current topology families without new metadata still work.
- The current additive metadata path remains compatible until deliberately
  superseded by a later implemented schema migration.
- Old linker splitting logic remains supported as a helper/base.
- Role-aware reticular behavior remains optional until later phases complete.
- Legacy scalar builder inputs must continue to normalize into a valid default
  runtime path.
- Metadata failures must be explicit unless family policy permits null fallback
  for the relevant unresolved edge role.
- Primitive-cell optimization followed by translation-based supercell
  replication remains the preferred performance path.

## 13. Decision log

- Decision: initial grammar is limited to `V-E-V` and `V-E-C`.
  Rationale: this is the smallest graph grammar that captures the target
  chemistry and bundle semantics without reopening the entire pipeline around
  unrestricted path classes.

- Decision: `C*` owns the linker bundle.
  Rationale: multitopic linker reconstruction, slot order, and resolve behavior
  become coherent only when a single topology anchor class owns the bundle.

- Decision: canonical cyclic order is computed once upstream from topology.
  Rationale: order must be deterministic and family-defined before geometry
  placement; allowing downstream modules to redefine it would create silent
  drift.

- Decision: null edges and zero-length real edges remain distinct.
  Rationale: the first is a virtual semantic placeholder; the second is real
  chemistry with short or collapsed length. They must not share fallback or
  validation behavior.

- Decision: unresolved-edge fallback is family-policy-controlled.
  Rationale: a universal null fallback would silently mask missing chemistry and
  corrupt validation in families that require explicit connector resolution.

- Decision: `builder.py` remains the high-level compiler/manager.
  Rationale: this preserves the existing builder/framework split and prevents
  schema, registry, bundle, and policy logic from being duplicated in lower
  layers.

- Decision: optimization remains primitive-cell-first.
  Rationale: this matches the locked pipeline and keeps supercell generation a
  translation-based semantic propagation step rather than a new optimization
  problem.

- Decision: ownership-sensitive resolve is prepared before optimization but
  committed only before final merge.
  Rationale: this keeps topology metadata stable early while postponing final
  chemistry ownership to the point where geometry and provenance are both
  available.

## 14. Roadmap at a glance

| Phase | Name | Main output |
| --- | --- | --- |
| 1 | Planning/spec | Active `PLAN.md` with phase contracts and schema draft |
| 2 | Passive metadata extension | Concrete JSON-readable family metadata contract in `MofTopLibrary` |
| 3 | Builder normalization and validation | Compiled runtime role/bundle scaffolding and explicit validation entrypoint |
| 4 | Optimizer role-aware consumption | Placement constraints aware of path class, slot typing, and null-edge rules |
| 5 | Post-opt semantic propagation and final resolve/merge | Supercell semantic preservation plus framework-side resolve/provenance commit |
| 6 | Defects and termination integration | Explicit unsaturated-site and termination-anchor consumption |
| 7 | Writer and debug export | Normal plus debug/checkpoint outputs carrying semantic metadata |
| 8 | Docs/examples/tests hardening | Documentation, examples, and regression hardening aligned with implemented behavior |

## 15. Phase roadmap

### Phase 1 - Planning/spec

Objective:

- Produce this active plan and freeze the semantic, ownership, and stop-rule
  contracts for later execution threads.

Preconditions:

- `PLAN_codex_record.md` is frozen.
- Round 1 and Round 2 checkpoints are accepted as settled inputs.

In-scope modules/files:

- `PLAN.md`
- planning/control docs for read-only synthesis

Explicitly out of scope:

- all runtime modules
- all test modules
- frozen historical planning records

Invariants to preserve:

- no code implementation
- no source or public API changes
- no phase broadening beyond planning/spec

Validation goals:

- the plan must encode the settled semantic contract
- the plan must encode the settled module ownership contract
- the plan must respect repository architecture locks and compatibility rules

Tests to add/run later:

- none in this planning phase

Exit criteria:

- `PLAN.md` exists as the active planning document
- the document contains explicit phase contracts, invariants, and stop rules
- the document includes at least one minimal metadata example snippet

Rollback risks / failure risks:

- under-specification that forces later executors to invent architecture
- accidental contradiction of the frozen baseline

Planner/executor/reviewer notes:

- planner owns this phase
- executor threads must not reinterpret Phase 1 as permission to implement
- reviewer should only check completeness and internal consistency

Stop and escalate if:

- a planning requirement would force runtime edits
- the frozen historical record would need mutation
- any settled Round 1 or Round 2 decision would need reopening

### Phase 2 - `MofTopLibrary` passive metadata extension

Objective:

- Phase 2 is limited to passive parsing and schema-checking of family metadata
  at the `MofTopLibrary` boundary only.
- Expose the reticular role grammar as one JSON-readable passive schema with
  lightweight validation only; runtime compilation and runtime policy work are
  deferred to Phase 3.

Preconditions:

- Phase 1 plan accepted
- exact schema names and semantics taken from this plan unless a real conflict
  is documented first

In-scope modules/files:

- `src/mofbuilder/core/moftoplibrary.py`
- `tests/test_core_moftoplibrary.py`
- metadata fixture(s) under `tests/database/`
- bundled database edits are forbidden by default in Phase 2

Explicitly out of scope:

- `src/mofbuilder/core/net.py`
- `src/mofbuilder/core/builder.py`
- `src/mofbuilder/core/optimizer.py`
- `src/mofbuilder/core/supercell.py`
- `src/mofbuilder/core/framework.py`
- `src/mofbuilder/core/defects.py`
- `src/mofbuilder/core/termination.py`
- `src/mofbuilder/core/write.py`
- `src/mofbuilder/core/linker.py`
- MD modules
- builder-facing compilation helpers
- registry construction
- bundle-id derivation
- unresolved-edge-policy normalization
- structured runtime validation APIs
- any bundled-database edit that is not explicitly authorized by a plan
  revision that names the exact allowed families

Invariants to preserve:

- `MOF_topology_dict` remains readable for existing families
- metadata remains passive and additive
- no graph or runtime behavior changes
- no heavy schema dependency introduced
- old families without new metadata still resolve exactly as before
- `MofTopLibrary.fetch()` key names remain unchanged for existing bundled
  families
- `MofTopLibrary.fetch()` return types remain unchanged for existing bundled
  families
- `MofTopLibrary.fetch()` family-selection behavior remains unchanged
- legacy-family `fetch()` results remain unchanged except for an optional
  additive `role_metadata` field

Validation goals:

- metadata roles must declare legal prefixes/classes
- canonical ids must be derivable from aliases
- path rules must be limited to `V-E-V` and `V-E-C`
- bundle ownership may only be declared on `C*`
- null-edge declarations must be explicit
- unresolved-edge fallback policy must be explicit where used
- validation stays library-boundary only and must not introduce structured
  runtime validation APIs in Phase 2

Tests to add/run later:

- run `scripts/run_tests.sh tests/test_core_moftoplibrary.py`
- at least one legacy-family regression
- at least one no-role-metadata regression
- at least one invalid-schema negative test
- only if bundled database mode is explicitly approved by a plan revision that
  names exact families, at least one regression proving existing bundled
  families are unchanged except for an additive `role_metadata` field

Exit criteria:

- `MofTopLibrary` can load one normalized passive reticular metadata shape
- legacy scalar metadata remains intact
- invalid metadata fails at the library boundary, not downstream

Rollback risks / failure risks:

- schema drift between metadata file and runtime consumers
- overloading the metadata layer with runtime payload ownership

Planner/executor/reviewer notes:

- executor must keep this phase metadata-only
- Phase 2 default execution mode is `fixture-only`
- every executor thread must begin by reopening the relevant current-cycle
  control documents before coding; at minimum this means `PLAN.md` plus any
  applicable frozen control docs named in Section 20
- the first Phase 2 executor thread must reopen `STATUS.md` and `WORKLOG.md`
  and explicitly confirm in its pre-coding summary that it is operating in the
  new planning cycle before touching `src/mofbuilder/core/moftoplibrary.py`,
  `tests/test_core_moftoplibrary.py`, or any metadata fixture/database file
- every Phase 2 pre-coding summary must declare whether the thread is
  `fixture-only` or `fixture + database metadata`
- every Phase 2 pre-coding summary must explicitly declare whether
  `database/MOF_topology_role_metadata.json` will be edited
- a pre-coding summary cannot self-authorize bundled-database edits
- editing `database/MOF_topology_role_metadata.json` is forbidden unless a plan
  revision explicitly authorizes that file and names the exact allowed
  families
- builder-facing compilation helpers, registry construction, bundle-id
  derivation, unresolved-edge-policy normalization, and structured runtime
  validation APIs begin in Phase 3, not Phase 2
- reviewer must reject any builder/runtime consumption introduced here
- reviewer approval requires both:
  - the runtime-seam checklist passing for the submitted Phase 2 scope
  - direct verification with
    `scripts/run_tests.sh tests/test_core_moftoplibrary.py`

Stop and escalate if:

- the phase requires runtime fragment resolution
- the phase requires graph mutation outside passive metadata loading
- the phase requires multiple concurrent schema variants without a canonical
  normalized representation
- the work would alter `MofTopLibrary.fetch()` results for an existing bundled
  family, even if no file outside `src/mofbuilder/core/moftoplibrary.py` is
  edited

### Phase 3 - Builder normalization, validation, bundle maps, and null policy

Objective:

- Make `builder.py` the explicit compilation manager for reticular metadata,
  validation, canonical bundle structures, and null-fallback policy without
  changing public APIs.

Preconditions:

- Phase 2 passive metadata contract exists
- planned `FrameNet` validation API is specified in the phase contract before
  code starts
- Phase 3 is the first phase allowed to add builder-facing compilation helpers,
  registry construction, bundle-id derivation, unresolved-edge-policy
  normalization, and structured runtime validation APIs

In-scope modules/files:

- `src/mofbuilder/core/builder.py`
- `src/mofbuilder/core/net.py` only for the new validation function and the
  minimum graph-hint storage needed by builder consumption
- `tests/test_core_builder.py`
- `tests/test_core_net.py` for validation-boundary coverage only

Explicitly out of scope:

- `src/mofbuilder/core/optimizer.py`
- `src/mofbuilder/core/supercell.py`
- `src/mofbuilder/core/framework.py`
- `src/mofbuilder/core/defects.py`
- `src/mofbuilder/core/termination.py`
- `src/mofbuilder/core/write.py`
- `src/mofbuilder/core/linker.py` except read-only dependency usage
- MD modules

Invariants to preserve:

- builder remains the orchestration manager
- graph role ids remain on `FrameNet.G`
- registries remain builder-owned
- public scalar builder inputs remain accepted
- single-role fast path remains cheap
- old families without new metadata remain supported

Validation goals:

- builder calls `FrameNet` validation before optimization
- validation returns structured status/errors/hints
- builder compiles canonical bundle ids from topology order hints
- builder normalizes unresolved-edge policy explicitly by family
- builder prepares, but does not yet commit, resolve/provenance scaffolding

Tests to add/run later:

- `scripts/run_tests.sh tests/test_core_builder.py`
- `scripts/run_tests.sh tests/test_core_net.py`
- normalization tests for legacy scalar inputs
- validation tests for illegal prefixes, illegal endpoint patterns, missing slot
  metadata, missing order metadata where required, and invalid null-edge policy
- bundle-map tests proving canonical bundle ids are deterministic

Exit criteria:

- builder can ingest passive reticular metadata and compile runtime-friendly
  structures
- `FrameNet` validation exists and fails early on invalid topology/metadata
  combinations
- null fallback is explicit and family-driven

Rollback risks / failure risks:

- builder-local shadow role maps
- policy leakage into optimizer or writer
- breaking single-role builder behavior while generalizing internals

Planner/executor/reviewer notes:

- supercell, writer, and defects remain out of scope even if downstream gaps
  become visible
- reviewer should demand explicit structured validation output, not ad hoc
  warnings only

Stop and escalate if:

- the phase requires placement logic changes in optimizer
- builder would need to redefine topology order instead of compiling upstream
  hints
- family policy cannot be represented without reopening the Phase 2 schema

### Phase 4 - Optimizer role-aware consumption

Objective:

- Make placement and primitive-cell optimization consume reticular path class,
  slot typing, and null-edge constraints without redefining topology ownership.

Preconditions:

- builder can hand off validated role-aware graph and registry state
- canonical order and bundle ids already exist upstream

In-scope modules/files:

- `src/mofbuilder/core/optimizer.py`
- `src/mofbuilder/core/superimpose.py` only if narrowly required by matching
  logic
- `src/mofbuilder/core/other.py` only if narrowly required by typed pairing or
  ordering helpers
- `tests/test_core_optimizer.py`

Explicitly out of scope:

- `src/mofbuilder/core/framework.py`
- `src/mofbuilder/core/supercell.py`
- `src/mofbuilder/core/defects.py`
- `src/mofbuilder/core/termination.py`
- `src/mofbuilder/core/write.py`
- metadata schema changes
- public builder API changes

Invariants to preserve:

- optimizer consumes graph/registry inputs, it does not redefine them
- primitive-cell optimization remains the core numerical path
- canonical cyclic order is not recomputed arbitrarily
- single-role numerical behavior remains unchanged

Validation goals:

- optimizer distinguishes `V-E-C` from `V-E-V`
- typed slot matching is enforced before geometric pairing
- null-edge zero-length constraints are handled explicitly
- role-aware grouping does not break the default single-role fast path

Tests to add/run later:

- `scripts/run_tests.sh tests/test_core_optimizer.py`
- single-role regression tests for current placement behavior
- targeted role-aware tests for slot-typed matching
- tests proving null-edge constraints do not masquerade as zero-length real-edge
  behavior
- tests proving canonical order is consumed, not invented, by optimizer

Exit criteria:

- optimizer accepts validated reticular role-aware input without redefining the
  topology model
- primitive-cell optimization still completes on single-role paths
- at least one narrow heterogeneous G2 case is covered at the optimizer
  boundary

Rollback risks / failure risks:

- canonical order drift
- silent fallback from typed slots to pure geometry matching
- numerical regressions in single-role paths

Planner/executor/reviewer notes:

- executor must not roll resolve/provenance commit into this phase
- reviewer should reject any redefinition of bundle ownership or order

Stop and escalate if:

- canonical order is missing upstream and would need optimizer invention
- null-edge handling cannot be represented without reopening the builder/schema
  contract
- the phase requires supercell or framework changes beyond narrow interface
  shims

### Phase 5 - Post-optimization semantic propagation and final resolve/merge

Objective:

- Preserve reticular semantics through supercell replication and commit
  ownership-sensitive resolve/provenance before final merged structure
  materialization.

Preconditions:

- Phase 4 provides primitive-cell optimized role-aware geometry
- builder resolve scaffolding exists but is not yet committed

In-scope modules/files:

- `src/mofbuilder/core/supercell.py`
- `src/mofbuilder/core/framework.py`
- `src/mofbuilder/core/builder.py` only for narrow handoff wiring required to
  pass resolve/provenance scaffolding forward
- `tests/test_core_supercell.py`
- `tests/test_core_framework.py`

Explicitly out of scope:

- `src/mofbuilder/core/defects.py`
- `src/mofbuilder/core/termination.py`
- `src/mofbuilder/core/write.py`
- `src/mofbuilder/core/moftoplibrary.py`
- `src/mofbuilder/core/linker.py` except existing helper consumption
- MD modules

Invariants to preserve:

- optimize primitive cell first, then replicate by translation
- supercell preserves semantics, not only coordinates
- resolve is committed before final merge, not during topology parsing
- graph-state names and phase order remain locked

Validation goals:

- replicated nodes/edges retain role ids
- bundle membership survives replication
- provenance and pending-resolve metadata survive until commit
- framework materialization commits ownership transfer where required
- unsaturated-site and termination-anchor precursor data are emitted explicitly

Tests to add/run later:

- `scripts/run_tests.sh tests/test_core_supercell.py`
- `scripts/run_tests.sh tests/test_core_framework.py`
- semantic propagation tests for role ids, bundle ids, and provenance through
  `superG`, `eG`, and `cleaved_eG`
- resolve/provenance tests proving ownership transfer occurs before merged atom
  assembly
- unsaturated marker generation tests at the framework boundary

Exit criteria:

- supercell propagation preserves the semantic state required by later modules
- framework materialization has explicit resolved/provenance-rich state before
  final merge
- no downstream module needs to rediscover ownership or provenance by guesswork

Rollback risks / failure risks:

- provenance loss before final merge
- semantic data dropped during supercell replication
- accidental change to translation-based fast path performance

Planner/executor/reviewer notes:

- this phase intentionally includes `supercell.py` even though the agreed phase
  label centers on framework/build resolve, because semantic propagation must be
  correct before framework merge can be trusted
- reviewer should insist on explicit pre-merge resolve evidence

Stop and escalate if:

- resolve commit would require reordering the locked pipeline
- supercell semantic preservation requires a new top-level graph state
- framework merge cannot preserve provenance without reopening earlier schema
  contracts

### Phase 6 - Defects and termination integration

Objective:

- Make defect editing and termination placement consume explicit resolve,
  provenance, unsaturated-site, and termination-anchor outputs from Phase 5.

Preconditions:

- Phase 5 emits explicit resolved ownership and unsaturated/termination-anchor
  data

In-scope modules/files:

- `src/mofbuilder/core/defects.py`
- `src/mofbuilder/core/termination.py`
- `src/mofbuilder/core/framework.py` only for narrow handoff plumbing to the
  defect/termination layer
- `tests/test_core_defects.py`
- `tests/test_core_termination.py`
- `tests/test_core_framework.py` if a framework-facing regression is required

Explicitly out of scope:

- `src/mofbuilder/core/moftoplibrary.py`
- `src/mofbuilder/core/net.py`
- `src/mofbuilder/core/optimizer.py`
- `src/mofbuilder/core/supercell.py`
- `src/mofbuilder/core/write.py`
- MD modules

Invariants to preserve:

- defects consume explicit outputs; they do not rediscover topology meaning
- termination remains a fragment-preparation/placement concern, not a topology
  policy owner
- current mutation semantics remain intact

Validation goals:

- leftover coordination groups can define unsaturated sites explicitly
- termination placement consumes explicit termination anchors when available
- removal/replacement flows preserve resolved/provenance metadata on resulting
  framework objects

Tests to add/run later:

- `scripts/run_tests.sh tests/test_core_defects.py`
- `scripts/run_tests.sh tests/test_core_termination.py`
- `scripts/run_tests.sh tests/test_core_framework.py`
- unsaturated-site tests for partial resolve leftovers
- termination-anchor tests proving no rediscovery by guesswork
- regression tests for `Framework.remove()` and `Framework.replace()`

Exit criteria:

- defect and termination code operate on explicit semantic outputs from Phase 5
- unsaturated-site behavior is consistent with resolve/provenance rules
- mutation APIs remain stable

Rollback risks / failure risks:

- defects reconstructing role/path meaning locally
- inconsistent unsaturated markers after removal/replacement

Planner/executor/reviewer notes:

- reviewer should reject any topology-schema work in this phase
- any missing precursor data from Phase 5 is a stop-and-escalate condition, not
  a reason to rebuild semantics here

Stop and escalate if:

- defects or termination need to infer provenance that Phase 5 did not provide
- the phase requires writer/debug export changes beyond narrow test harness
  needs

### Phase 7 - Writer and debug export

Objective:

- Support both normal output and explicit debug/checkpoint export carrying the
  reticular semantic metadata required for diagnosis and audit.

Preconditions:

- Phase 5 and Phase 6 provide stable resolved/provenance-rich graph/materialized
  state

In-scope modules/files:

- `src/mofbuilder/core/write.py`
- `src/mofbuilder/core/framework.py` only for narrow writer handoff plumbing
- `tests/test_core_write.py`
- `tests/test_core_framework.py` if writer-facing framework regressions require
  it

Explicitly out of scope:

- metadata schema changes
- optimizer or supercell changes
- defect policy changes
- MD modules

Invariants to preserve:

- normal output remains available and compatible
- debug export is additive, not a forced replacement path
- writer remains a consumer of explicit semantic metadata, not a topology owner

Validation goals:

- normal export keeps current output expectations
- debug/checkpoint export can carry role ids, bundle ids, provenance, and
  unsaturated markers
- writer does not silently discard semantic metadata required for audit

Tests to add/run later:

- `scripts/run_tests.sh tests/test_core_write.py`
- `scripts/run_tests.sh tests/test_core_framework.py`
- regression tests for current merged-data behavior
- debug-export tests for role ids, bundle ids, provenance, and unsaturated
  markers

Exit criteria:

- writer supports both normal and debug/checkpoint export modes
- exported debug metadata is sufficient for later diagnosis of resolve and
  bundle behavior

Rollback risks / failure risks:

- silent metadata loss in export paths
- accidental breakage of current normal writer behavior

Planner/executor/reviewer notes:

- reviewer should keep this phase additive and reject output-format churn that
  is not required by the plan

Stop and escalate if:

- writer would need to invent semantic metadata that upstream phases did not
  preserve
- normal output compatibility would be broken to make debug export work

### Phase 8 - Docs, examples, and test hardening

Objective:

- Synchronize docs, examples, and regression coverage with the implemented
  reticular role model after runtime behavior stabilizes.

Preconditions:

- Phases 2-7 complete and reviewed

In-scope modules/files:

- `README.md`
- `docs/`
- `docs/source/manual/`
- `tests/test_core_moftoplibrary.py`
- `tests/test_core_builder.py`
- `tests/test_core_net.py`
- `tests/test_core_optimizer.py`
- `tests/test_core_supercell.py`
- `tests/test_core_framework.py`
- `tests/test_core_defects.py`
- `tests/test_core_termination.py`
- `tests/test_core_write.py`
- other narrow test files required to close verified gaps

Explicitly out of scope:

- new production behavior unrelated to documentation/test hardening
- broad algorithmic refactors
- new external dependencies

Invariants to preserve:

- docs describe implemented behavior only
- tests use `scripts/run_tests.sh`
- single-role compatibility remains explicitly covered

Validation goals:

- docs explain G2 scope and non-goals clearly
- examples do not overclaim unsupported grammars or chemistry
- regression coverage proves no single-role breakage

Tests to add/run later:

- the narrowest relevant phase-targeted commands via `scripts/run_tests.sh`
- smoke tests if package surface or CLI language changes
- targeted end-to-end tests only after the underlying layers are already proven

Exit criteria:

- docs, examples, and tests match the implemented reticular behavior
- no unsupported scope is documented as complete
- single-role and legacy-family compatibility are explicitly regression-covered

Rollback risks / failure risks:

- documentation overstating support
- tests implicitly assuming broad arbitrary-graph support

Planner/executor/reviewer notes:

- reviewer should reject documentation that reopens settled semantics or
  broadens scope beyond G2

Stop and escalate if:

- documentation uncovers a real architecture conflict with implemented runtime
  behavior
- test hardening would require reopening a prior phase contract

## 16. Phase-by-phase allowed files

This section is a compact execution map. Later phase contracts may narrow these
lists further, but they must not silently broaden them.

| Phase | Allowed files | Files that should not be touched in that phase |
| --- | --- | --- |
| 1 | `PLAN.md` | All runtime modules, tests, frozen records |
| 2 | `src/mofbuilder/core/moftoplibrary.py`, `tests/test_core_moftoplibrary.py`, metadata fixtures under `tests/database/`; `database/MOF_topology_role_metadata.json` only if a plan revision explicitly authorizes that file and names exact families | `builder.py`, `net.py`, `optimizer.py`, `supercell.py`, `framework.py`, `defects.py`, `termination.py`, `write.py`, MD modules |
| 3 | `src/mofbuilder/core/builder.py`, `src/mofbuilder/core/net.py` for validation only, `tests/test_core_builder.py`, `tests/test_core_net.py` | `optimizer.py`, `supercell.py`, `framework.py`, `defects.py`, `termination.py`, `write.py`, MD modules |
| 4 | `src/mofbuilder/core/optimizer.py`, narrow helper files if strictly required, `tests/test_core_optimizer.py` | `supercell.py`, `framework.py`, `defects.py`, `termination.py`, `write.py`, metadata files |
| 5 | `src/mofbuilder/core/supercell.py`, `src/mofbuilder/core/framework.py`, narrow builder handoff plumbing, `tests/test_core_supercell.py`, `tests/test_core_framework.py` | `defects.py`, `termination.py`, `write.py`, metadata files, MD modules |
| 6 | `src/mofbuilder/core/defects.py`, `src/mofbuilder/core/termination.py`, narrow framework plumbing, matching tests | `moftoplibrary.py`, `net.py`, `optimizer.py`, `supercell.py`, `write.py`, MD modules |
| 7 | `src/mofbuilder/core/write.py`, narrow `framework.py` plumbing, matching tests | metadata files, optimizer/supercell logic, defects policy, MD modules |
| 8 | docs and test files only | unrelated production modules except for narrowly justified review fixes tied to documentation/test correctness |

## 17. Test strategy

All later runtime verification must use the repository test runner:

- `scripts/run_tests.sh`

Do not run `pytest` directly in executor threads.

Test categories required across the cycle:

- topology grammar validation tests
  - illegal prefixes, illegal path classes, invalid ordered endpoint rules
- metadata normalization tests
  - passive schema loading, alias-to-canonical-id mapping, family-policy
    loading, lookup hints
- slot typing tests
  - slot-type-preserving matching and explicit failure on mismatched slot types
- ordered path / cyclic order persistence tests
  - topology-derived order survives from metadata through graph storage and
    downstream use
- null-edge fallback tests
  - explicit null edges versus zero-length real edges
  - family-policy-controlled unresolved fallback
- primitive-cell optimizer role-aware tests
  - `V-E-C` versus `V-E-V`
  - zero-length null constraints
  - single-role numerical regression
- supercell semantic propagation tests
  - role ids, bundle ids, resolve-pending state, provenance survival
- resolve / provenance tests
  - pre-merge ownership transfer, provenance retention, leftover coordination
    groups
- unsaturated-site / termination tests
  - explicit unsaturated markers and termination-anchor consumption
- writer debug-export tests
  - role ids, bundle ids, provenance, unsaturated markers

Suggested narrow verification targets by phase:

- Phase 2:
  `scripts/run_tests.sh tests/test_core_moftoplibrary.py`
  plus required Phase 2 regression evidence:
  - one legacy-family regression
  - one no-role-metadata regression
  - one invalid-schema negative test
  - only if bundled database mode is explicitly approved by plan revision, one
    regression proving existing bundled families are unchanged except for an
    additive `role_metadata` field
- Phase 3:
  `scripts/run_tests.sh tests/test_core_builder.py`
  `scripts/run_tests.sh tests/test_core_net.py`
- Phase 4:
  `scripts/run_tests.sh tests/test_core_optimizer.py`
- Phase 5:
  `scripts/run_tests.sh tests/test_core_supercell.py`
  `scripts/run_tests.sh tests/test_core_framework.py`
- Phase 6:
  `scripts/run_tests.sh tests/test_core_defects.py`
  `scripts/run_tests.sh tests/test_core_termination.py`
- Phase 7:
  `scripts/run_tests.sh tests/test_core_write.py`
  `scripts/run_tests.sh tests/test_core_framework.py`
- Phase 8:
  narrow reruns of prior phase tests plus smoke tests if documentation or
  package-surface wording requires them

## 18. Risk register

- Silent null-edge masking of missing chemistry.
  Impact: invalid families could appear to build while silently defaulting to a
  null edge.
  Containment: family-policy-controlled fallback plus explicit validation
  failures.

- Loss of canonical order.
  Impact: multitopic linker reconstruction becomes non-deterministic.
  Containment: compute order once upstream and store it explicitly.

- Breaking single-role workflows.
  Impact: regressions in the current public path.
  Containment: explicit single-role regression tests in every runtime phase.

- Provenance loss before final merge.
  Impact: ownership transfer, unsaturated-site marking, and termination logic
  become untrustworthy.
  Containment: resolve scaffolding prepared upstream and committed before final
  merge.

- Role/path ambiguity between modules.
  Impact: topology meaning drifts as it crosses module boundaries.
  Containment: fixed ownership matrix and structured validation.

- Accidental public API breakage.
  Impact: user workflows and smoke tests regress.
  Containment: preserve APIs in early phases and run narrow framework/builder
  regressions.

- Performance regression in the supercell path.
  Impact: current translation-based fast path degrades.
  Containment: keep primitive-cell-first optimization and translation-based
  replication.

- Schema/runtime drift.
  Impact: metadata passes library validation but cannot be consumed by builder.
  Containment: builder phase stops and escalates instead of silently adapting
  incompatible schema.

## 19. Deferred decisions / open questions

These are intentionally bounded implementation questions. They do not reopen
the settled semantics above.

- Exact serialized shape of the structured `FrameNet` validation result.
  Constraint: must include status, structured errors, and hints/messages.

- Exact on-graph attribute names for topology-derived cyclic order and bundle
  hints.
  Constraint: they must be stored on graph objects and treated as topology
  hints, not chemistry results.

- Exact debug-export surface in `write.py`.
  Constraint: normal output remains supported and debug output must include role
  ids, bundle ids, provenance, and unsaturated markers.

## 20. Execution rules for later Codex roles

- Planner threads define phase scope, invariants, allowed files, forbidden
  files, success criteria, and stop rules.
- Executor threads implement only the current approved phase.
- Reviewer threads verify invariants, tests, stop rules, and failure
  containment.
- No phase may silently broaden scope.
- If an executor discovers a schema/runtime/invariant conflict, the executor
  must stop, log the conflict, and request a plan revision instead of repairing
  both sides in one pass.
- During this planning cycle, the following are read-only control docs for
  executor and reviewer threads unless a real conflict is found, logged, and
  escalated before any edit:
  `AGENTS.md`, `ARCHITECTURE.md`, `ARCHITECTURE_DECISIONS.md`, `README.md`,
  and `CODEX_CONTEXT.md`.
- `STATUS.md` and `WORKLOG.md` are read-only for semantic/scope edits, but
  append-only checkpoint/status logging required by `AGENTS.md` is permitted.
- Every executor thread must begin by reopening the relevant control docs for
  the current cycle before coding. The minimum required set is `PLAN.md` plus
  any phase-specific control docs or frozen records needed to verify scope,
  invariants, and stop rules.
- Section C execution-control patch for Phase 2 is active in this planning
  cycle. Equivalent mandatory wording is:
  - the first Phase 2 executor thread must reopen `STATUS.md` and `WORKLOG.md`
    and explicitly confirm it is working in the new cycle before touching
    `src/mofbuilder/core/moftoplibrary.py`, `tests/test_core_moftoplibrary.py`,
    `tests/database/`, or `database/MOF_topology_role_metadata.json`
  - Phase 2 default mode is `fixture-only`
  - every Phase 2 pre-coding summary must state whether the thread is
    `fixture-only` or `fixture + database metadata`
  - every Phase 2 pre-coding summary must explicitly state whether
    `database/MOF_topology_role_metadata.json` will be edited
  - a pre-coding summary cannot self-authorize bundled-database edits
  - editing `database/MOF_topology_role_metadata.json` is forbidden unless a
    plan revision explicitly authorizes that file and names exact allowed
    families
  - reviewer approval for Phase 2 requires the runtime-seam checklist plus
    direct verification with
    `scripts/run_tests.sh tests/test_core_moftoplibrary.py`
    and the following minimum evidence:
    - one legacy-family regression
    - one no-role-metadata regression
    - one invalid-schema negative test
    - only if bundled database mode is explicitly approved by a plan revision,
      one regression proving existing bundled families are unchanged except for
      an additive `role_metadata` field
- Runtime-seam checklist for reviewer approval:
  - the change remains passive metadata loading/normalization only
  - no builder-facing compilation helpers, registry construction, bundle-id
    derivation, unresolved-edge-policy normalization, or structured runtime
    validation APIs are introduced in Phase 2
  - `MOF_topology_dict` legacy behavior remains intact for existing families
  - `MofTopLibrary.fetch()` key names, return types, and family-selection
    behavior remain unchanged
  - legacy-family `fetch()` results remain unchanged except for an optional
    additive `role_metadata` field
  - no builder, optimizer, framework, supercell, writer, defects, termination,
    or MD runtime consumption is introduced
  - any fixture or database metadata added is validated at the
    `MofTopLibrary` boundary and not by downstream repair logic
  - the submitted thread stayed within its declared Phase 2 mode:
    `fixture-only` or `fixture + database metadata`
  - if bundled database mode was not explicitly approved by plan revision,
    `database/MOF_topology_role_metadata.json` is unchanged
  - stop immediately if the work would alter `fetch()` results for any existing
    bundled family, even when only
    `src/mofbuilder/core/moftoplibrary.py` is edited

## 21. Stop rule for this planning run

This run only creates or updates `PLAN.md`.

It does not:

- implement code
- refactor modules
- modify tests
- change public APIs
- mutate `PLAN_codex_record.md`

The planning run ends after:

- drafting `PLAN.md`
- self-reviewing it once against the Round 1 checkpoint, Round 2 checkpoint,
  and repository control docs
- applying at most one revision if needed
