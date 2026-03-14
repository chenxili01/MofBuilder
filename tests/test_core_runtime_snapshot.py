from dataclasses import FrozenInstanceError

import pytest

from mofbuilder.core.runtime_snapshot import (
    BundleRecord,
    EdgeRoleRecord,
    FrameworkInputSnapshot,
    GraphEdgeSemanticRecord,
    GraphNodeSemanticRecord,
    NodeRoleRecord,
    NullEdgePolicyRecord,
    OptimizationSemanticSnapshot,
    ProvenanceRecord,
    ResolveInstructionRecord,
    ResolvedStateRecord,
    RoleRuntimeSnapshot,
)


@pytest.mark.core
def test_snapshot_records_support_default_role_construction():
    node_record = NodeRoleRecord(
        role_id="node:default",
        family_alias="default",
        role_class="V",
        expected_connectivity=6,
        metadata_reference={"source": "legacy_default"},
    )
    edge_record = EdgeRoleRecord(
        role_id="edge:default",
        family_alias="default",
        role_class="E",
        linker_connectivity=2,
        edge_kind="real",
        metadata_reference={"source": "legacy_default"},
    )
    resolved_state = ResolvedStateRecord(
        state_id="node:default:fragment",
        role_id="node:default",
        state_kind="node_fragment",
        is_resolved=False,
        metadata={"reason": "not_compiled_in_phase_1"},
    )

    runtime_snapshot = RoleRuntimeSnapshot(
        family_name="MOF-TEST",
        graph_phase="G",
        node_role_records={"node:default": node_record},
        edge_role_records={"edge:default": edge_record},
        resolved_state_records={resolved_state.state_id: resolved_state},
        metadata={"source": "phase_1_record_test"},
    )

    assert runtime_snapshot.family_name == "MOF-TEST"
    assert runtime_snapshot.graph_phase == "G"
    assert runtime_snapshot.node_role_records["node:default"].expected_connectivity == 6
    assert runtime_snapshot.edge_role_records["edge:default"].linker_connectivity == 2
    assert runtime_snapshot.resolved_state_records[resolved_state.state_id].is_resolved is False
    assert runtime_snapshot.metadata["source"] == "phase_1_record_test"

    with pytest.raises(TypeError):
        runtime_snapshot.node_role_records["node:extra"] = node_record
    with pytest.raises(FrozenInstanceError):
        runtime_snapshot.family_name = "OTHER"


@pytest.mark.core
def test_snapshot_records_capture_role_aware_bundle_resolve_and_null_edge_fields():
    null_policy = NullEdgePolicyRecord(
        edge_role_id="edge:EB",
        edge_kind="null",
        is_null_edge=True,
        null_payload_model="duplicated_zero_length_anchors",
        unresolved_action="error",
        allows_null_fallback=True,
        metadata={"semantic_distinction": "null_edge_not_zero_length_real_edge"},
    )
    node_record = NodeRoleRecord(
        role_id="node:CA",
        family_alias="CA",
        role_class="C",
        expected_connectivity=2,
        topology_labels=("C0",),
        incident_edge_aliases=("EA", "EA"),
        slot_rules=(
            {"attachment_index": 0, "slot_type": "XA"},
            {"attachment_index": 1, "slot_type": "XA"},
        ),
        metadata_reference={"source": "canonical_role_metadata"},
    )
    edge_record = EdgeRoleRecord(
        role_id="edge:EA",
        family_alias="EA",
        role_class="E",
        linker_connectivity=4,
        topology_labels=("E0",),
        endpoint_pattern=("VA", "EA", "CA"),
        slot_rules=(
            {"attachment_index": 0, "slot_type": "XA", "endpoint_side": "V"},
            {"attachment_index": 1, "slot_type": "XA", "endpoint_side": "C"},
        ),
        edge_kind="real",
        resolve_mode="ownership_transfer",
        metadata_reference={"source": "canonical_role_metadata"},
    )
    bundle_record = BundleRecord(
        bundle_id="bundle:CA:0",
        owner_role_id="node:CA",
        attachment_edge_role_ids=("edge:EA", "edge:EA"),
        ordered_attachment_indices=(0, 1),
        order_kind="clockwise_local_topology",
        metadata={"bundle_owner": "linker"},
    )
    resolve_instruction = ResolveInstructionRecord(
        instruction_id="resolve:EA:0",
        edge_role_id="edge:EA",
        resolve_mode="ownership_transfer",
        endpoint_pattern=("VA", "EA", "CA"),
        bundle_id="bundle:CA:0",
        source_role_id="node:VA",
        target_role_id="node:CA",
        metadata={"phase_boundary": "builder_owned"},
    )
    provenance_record = ProvenanceRecord(
        record_id="prov:edge:EA",
        role_id="edge:EA",
        source_kind="graph_edge",
        source_ref="('V0','C0')",
        metadata={"edge_role_id": "edge:EA"},
    )
    resolved_state = ResolvedStateRecord(
        state_id="resolved:bundle:CA:0",
        role_id="node:CA",
        state_kind="bundle_fragment",
        is_resolved=True,
        payload_ref="fragment:linker_center",
        fragment_key="center",
        metadata={"stage": "resolved_view"},
    )

    runtime_snapshot = RoleRuntimeSnapshot(
        family_name="TEST-MULTI",
        graph_phase="sG",
        node_role_records={"node:CA": node_record},
        edge_role_records={"edge:EA": edge_record},
        bundle_records={bundle_record.bundle_id: bundle_record},
        resolve_instruction_records=(resolve_instruction,),
        null_edge_policy_records={null_policy.edge_role_id: null_policy},
        provenance_records={provenance_record.record_id: provenance_record},
        resolved_state_records={resolved_state.state_id: resolved_state},
        metadata={"role_ids_live_on_graph": True},
    )
    optimization_snapshot = OptimizationSemanticSnapshot(
        family_name="TEST-MULTI",
        graph_phase="sG",
        graph_node_records={
            "C0": GraphNodeSemanticRecord(
                node_id="C0",
                role_id="node:CA",
                role_class="C",
                slot_rules=node_record.slot_rules,
                incident_edge_ids=("V0|C0",),
                incident_edge_role_ids=("edge:EA",),
                incident_edge_constraints=(
                    {
                        "edge_id": "V0|C0",
                        "edge_role_id": "edge:EA",
                        "slot_index": 0,
                    },
                ),
                bundle_id="bundle:CA:0",
                bundle_order_hint={"ordered_attachment_indices": (0, 1)},
            )
        },
        graph_edge_records={
            "V0|C0": GraphEdgeSemanticRecord(
                edge_id="V0|C0",
                graph_edge=("V0", "C0"),
                edge_role_id="edge:EA",
                path_type="V-E-C",
                endpoint_node_ids=("V0", "C0"),
                endpoint_role_ids=("node:VA", "node:CA"),
                endpoint_pattern=edge_record.endpoint_pattern,
                slot_index={"V0": 0, "C0": 0},
                slot_rules=edge_record.slot_rules,
                bundle_id="bundle:CA:0",
                bundle_order_index=0,
                resolve_mode="ownership_transfer",
            )
        },
        node_role_records=runtime_snapshot.node_role_records,
        edge_role_records=runtime_snapshot.edge_role_records,
        bundle_records=runtime_snapshot.bundle_records,
        resolve_instruction_records=runtime_snapshot.resolve_instruction_records,
        null_edge_policy_records=runtime_snapshot.null_edge_policy_records,
        metadata={"consumer": "future_optimizer"},
    )
    framework_snapshot = FrameworkInputSnapshot(
        family_name="TEST-MULTI",
        graph_phase="cleaved_eG",
        bundle_records=runtime_snapshot.bundle_records,
        provenance_records=runtime_snapshot.provenance_records,
        resolved_state_records=runtime_snapshot.resolved_state_records,
        metadata={"consumer": "future_framework_handoff"},
    )

    assert runtime_snapshot.bundle_records["bundle:CA:0"].owner_role_id == "node:CA"
    assert runtime_snapshot.resolve_instruction_records[0].endpoint_pattern == ("VA", "EA", "CA")
    assert runtime_snapshot.null_edge_policy_records["edge:EB"].is_null_edge is True
    assert (
        runtime_snapshot.null_edge_policy_records["edge:EB"].metadata["semantic_distinction"]
        == "null_edge_not_zero_length_real_edge"
    )
    assert optimization_snapshot.metadata["consumer"] == "future_optimizer"
    assert optimization_snapshot.graph_node_records["C0"].bundle_id == "bundle:CA:0"
    assert optimization_snapshot.graph_edge_records["V0|C0"].path_type == "V-E-C"
    assert framework_snapshot.provenance_records["prov:edge:EA"].source_kind == "graph_edge"
    assert framework_snapshot.resolved_state_records["resolved:bundle:CA:0"].payload_ref == (
        "fragment:linker_center"
    )

    with pytest.raises(TypeError):
        runtime_snapshot.resolve_instruction_records[0].metadata["extra"] = "forbidden"
