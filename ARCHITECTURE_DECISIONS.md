# Architecture Decisions

Concise ADR-style notes for the frozen Phase 1-8 role-aware architecture
baseline.

## ADR-001: Role IDs Live on the Topology Graph

- Status: accepted
- Decision: topology role identity is stored on graph instances as
  `FrameNet.G.nodes[n]["node_role_id"]` and
  `FrameNet.G.edges[e]["edge_role_id"]`.
- Rationale: runtime consumers read role identity from one deterministic source
  of truth attached to the graph, which keeps later pipeline stages aligned.

## ADR-002: Registries Resolve Roles, Graph Labels Do Not Carry Fragment Payloads

- Status: accepted
- Decision: fragment/config resolution stays in
  `node_role_registry` and `edge_role_registry`, owned by
  `MetalOrganicFrameworkBuilder`.
- Rationale: topology labels remain passive classification keys while builder
  registries own runtime fragment assignment and normalization.

## ADR-003: Single-Role Remains the Base Case

- Status: accepted
- Decision: the default path remains the existing single-role workflow, with
  normalization to `node:default` and `edge:default` when no role metadata is
  present.
- Rationale: backward compatibility and performance stay anchored to the
  established public workflow.

## ADR-004: The Staged Pipeline Remains Locked

- Status: accepted
- Decision: the canonical pipeline remains
  `MofTopLibrary.fetch(...) -> FrameNet.create_net(...) -> MetalOrganicFrameworkBuilder.load_framework() -> MetalOrganicFrameworkBuilder.optimize_framework() -> MetalOrganicFrameworkBuilder.make_supercell() -> MetalOrganicFrameworkBuilder.build()`.
- Rationale: the role-aware architecture was implemented as an extension of the
  existing graph-centered workflow, not as a replacement for it.
