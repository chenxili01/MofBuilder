import numpy as np
import networkx as nx
import pytest

from mofbuilder.core import optimizer as opt
from mofbuilder.core.optimizer_contract import (
    NodeLocalConstrainedRefinement,
    NodeDiscreteAmbiguityResolution,
    LegalNodeCorrespondence,
    NodeLocalRigidInitialization,
    NodePlacementContract,
    compile_local_constrained_refinement,
    compile_discrete_ambiguity_resolution,
    compile_local_rigid_initialization,
    compile_legal_node_correspondences,
    compile_node_placement_contract,
)
from mofbuilder.core.runtime_snapshot import (
    BundleRecord,
    EdgeRoleRecord,
    GraphEdgeSemanticRecord,
    GraphNodeSemanticRecord,
    NodeRoleRecord,
    NullEdgePolicyRecord,
    OptimizationSemanticSnapshot,
)


def _fragment_table(rows):
    return np.array(rows, dtype=object)


def _fragment_row(atom_name, atom_type, coords):
    return [atom_name, atom_type, 0, 0, 0, *coords]


def test_recenter_and_norm_vectors_returns_unit_rows():
    vectors = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

    normed, center = opt.recenter_and_norm_vectors(vectors)

    assert center.shape == (3,)
    assert np.allclose(np.linalg.norm(normed, axis=1), 1.0)


def test_get_connected_nodes_vectors_extracts_neighbor_coords():
    g = nx.Graph()
    g.add_node("A", ccoords=np.array([0.0, 0.0, 0.0]))
    g.add_node("B", ccoords=np.array([1.0, 0.0, 0.0]))
    g.add_node("C", ccoords=np.array([0.0, 1.0, 0.0]))
    g.add_edge("A", "B")
    g.add_edge("A", "C")

    vecs, center = opt.get_connected_nodes_vectors("A", g)

    assert len(vecs) == 2
    assert np.allclose(center, [0.0, 0.0, 0.0])


def test_expand_set_rots_maps_one_rotation_per_group():
    pname_set = {
        "V": {"ind_ofsortednodes": [0, 2]},
        "W": {"ind_ofsortednodes": [1]},
    }
    rot_v = np.eye(3)
    rot_w = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    set_rots = np.array([rot_v, rot_w]).reshape(-1)

    out = opt.expand_set_rots(pname_set, set_rots, ["a", "b", "c"])

    assert out.shape == (3, 3, 3)
    assert np.allclose(out[0], rot_v)
    assert np.allclose(out[1], rot_w)
    assert np.allclose(out[2], rot_v)


def test_optimizer_optional_semantic_snapshot_hook_defaults_and_accepts_snapshot():
    optimizer = opt.NetOptimizer()

    assert optimizer.semantic_snapshot is None

    semantic_snapshot = OptimizationSemanticSnapshot(
        family_name="TEST-FAMILY",
        graph_phase="G",
        metadata={"phase_bounded": "phase_5_hook"},
    )
    optimizer_with_snapshot = opt.NetOptimizer(semantic_snapshot=semantic_snapshot)

    assert optimizer_with_snapshot.semantic_snapshot is semantic_snapshot


def test_rotation_and_cell_optimization_stores_optional_semantic_snapshot(
    monkeypatch,
):
    optimizer = opt.NetOptimizer()
    optimizer.constant_length = 1.54
    optimizer.G = nx.Graph()
    semantic_snapshot = OptimizationSemanticSnapshot(
        family_name="TEST-FAMILY",
        graph_phase="G",
    )

    def stop_after_hook(_graph):
        raise RuntimeError("stop-after-hook")

    monkeypatch.setattr(optimizer, "_prepare_role_fragment_payloads", stop_after_hook)

    with pytest.raises(RuntimeError, match="stop-after-hook"):
        optimizer.rotation_and_cell_optimization(
            semantic_snapshot=semantic_snapshot,
            use_role_aware_local_placement=True,
        )

    assert optimizer.semantic_snapshot is semantic_snapshot
    assert optimizer.use_role_aware_local_placement is True


def test_compile_role_aware_initial_rotations_requires_explicit_opt_in(monkeypatch):
    semantic_snapshot = OptimizationSemanticSnapshot(
        family_name="ROLE-AWARE",
        graph_phase="sG",
        graph_node_records={
            "V0": GraphNodeSemanticRecord(
                node_id="V0",
                role_id="node:VA",
                role_class="V",
            ),
        },
    )
    optimizer = opt.NetOptimizer(semantic_snapshot=semantic_snapshot)
    optimizer.sorted_nodes = ["V0"]

    calls = []

    def fake_refinement(node_id, semantic_snapshot=None, **_kwargs):
        calls.append((node_id, semantic_snapshot))
        return NodeLocalConstrainedRefinement(
            node_id=node_id,
            node_role_id="node:VA",
            correspondence=LegalNodeCorrespondence(
                node_id=node_id,
                node_role_id="node:VA",
            ),
            rigid_initialization=NodeLocalRigidInitialization(
                node_id=node_id,
                node_role_id="node:VA",
                correspondence=LegalNodeCorrespondence(
                    node_id=node_id,
                    node_role_id="node:VA",
                ),
                anchor_pairs=(),
                rotation_matrix=((1.0, 0.0, 0.0),
                                 (0.0, 1.0, 0.0),
                                 (0.0, 0.0, 1.0)),
                translation_vector=(0.0, 0.0, 0.0),
                rmsd=0.0,
                source_anchor_representation="anchor_vector",
                target_anchor_representation="target_point",
            ),
            rotation_matrix=((1.0, 0.0, 0.0),
                             (0.0, 1.0, 0.0),
                             (0.0, 0.0, 1.0)),
            translation_vector=(0.0, 0.0, 0.0),
        )

    monkeypatch.setattr(
        optimizer,
        "compile_local_constrained_refinement",
        fake_refinement,
    )

    disabled = optimizer._compile_role_aware_initial_rotations(
        {"group:V": {"ind_ofsortednodes": [0]}},
        semantic_snapshot=semantic_snapshot,
    )

    assert disabled == {}
    assert calls == []

    optimizer.use_role_aware_local_placement = True
    enabled = optimizer._compile_role_aware_initial_rotations(
        {"group:V": {"ind_ofsortednodes": [0]}},
        semantic_snapshot=semantic_snapshot,
    )

    assert calls == [("V0", semantic_snapshot)]
    assert np.allclose(enabled["group:V"], np.eye(3))
    assert "V0" in optimizer.role_aware_local_placement_records


def test_compile_role_aware_initial_rotations_supports_v_and_c_guarded_cases(
    monkeypatch,
):
    semantic_snapshot = OptimizationSemanticSnapshot(
        family_name="ROLE-AWARE",
        graph_phase="sG",
        graph_node_records={
            "V0": GraphNodeSemanticRecord(
                node_id="V0",
                role_id="node:VA",
                role_class="V",
            ),
            "C0": GraphNodeSemanticRecord(
                node_id="C0",
                role_id="node:CA",
                role_class="C",
            ),
        },
    )
    optimizer = opt.NetOptimizer(semantic_snapshot=semantic_snapshot)
    optimizer.sorted_nodes = ["V0", "C0"]
    optimizer.use_role_aware_local_placement = True

    def fake_contract(node_id, semantic_snapshot=None):
        return NodePlacementContract(
            node_id=node_id,
            node_role_id=semantic_snapshot.graph_node_records[node_id].role_id,
            node_role_class=semantic_snapshot.graph_node_records[node_id].role_class,
            local_slot_types=("XA", "XB"),
            incident_edge_ids=(f"{node_id}|E0", f"{node_id}|E1"),
            resolve_mode_hints=("ownership_transfer",),
            null_edge_flags={f"{node_id}|E0": False, f"{node_id}|E1": False},
        )

    def fake_correspondences(_node_id, **_kwargs):
        return (
            LegalNodeCorrespondence(
                node_id="selected",
                node_role_id="role",
                edge_to_slot_index={"edge:0": 0, "edge:1": 1},
            ),
        )

    def fake_rigid(node_id, **_kwargs):
        return NodeLocalRigidInitialization(
            node_id=node_id,
            node_role_id=semantic_snapshot.graph_node_records[node_id].role_id,
            correspondence=LegalNodeCorrespondence(
                node_id=node_id,
                node_role_id=semantic_snapshot.graph_node_records[node_id].role_id,
                edge_to_slot_index={"edge:0": 0, "edge:1": 1},
            ),
            anchor_pairs=(),
            rotation_matrix=((1.0, 0.0, 0.0),
                             (0.0, 1.0, 0.0),
                             (0.0, 0.0, 1.0)),
            translation_vector=(0.0, 0.0, 0.0),
            rmsd=0.0,
            source_anchor_representation="anchor_vector",
            target_anchor_representation="target_point",
            metadata={"orientation_only_pair_count": 0},
        )

    def fake_refinement(node_id, semantic_snapshot=None, rigid_initialization=None, **_kwargs):
        return NodeLocalConstrainedRefinement(
            node_id=node_id,
            node_role_id=semantic_snapshot.graph_node_records[node_id].role_id,
            correspondence=rigid_initialization.correspondence,
            rigid_initialization=rigid_initialization,
            rotation_matrix=((1.0, 0.0, 0.0),
                             (0.0, 1.0, 0.0),
                             (0.0, 0.0, 1.0)),
            translation_vector=(0.0, 0.0, 0.0),
            objective_value=0.0,
            initial_objective_value=0.0,
        )

    monkeypatch.setattr(optimizer, "compile_node_placement_contract", fake_contract)
    monkeypatch.setattr(
        optimizer,
        "compile_legal_node_correspondences",
        fake_correspondences,
    )
    monkeypatch.setattr(optimizer, "compile_local_rigid_initialization", fake_rigid)
    monkeypatch.setattr(
        optimizer,
        "compile_local_constrained_refinement",
        fake_refinement,
    )

    rotations = optimizer._compile_role_aware_initial_rotations(
        {
            "group:V": {"ind_ofsortednodes": [0]},
            "group:C": {"ind_ofsortednodes": [1]},
        },
        semantic_snapshot=semantic_snapshot,
    )

    assert set(rotations) == {"group:V", "group:C"}
    assert np.allclose(rotations["group:V"], np.eye(3))
    assert np.allclose(rotations["group:C"], np.eye(3))
    assert set(optimizer.role_aware_local_placement_records) == {"V0", "C0"}
    assert optimizer.role_aware_local_placement_debug_records["V0"]["status"] == "selected"
    assert optimizer.role_aware_local_placement_debug_records["C0"]["status"] == "selected"


def test_compile_role_aware_initial_rotations_records_guard_disabled_and_missing_snapshot():
    optimizer = opt.NetOptimizer()
    optimizer.sorted_nodes = ["V0", "C0"]

    disabled = optimizer._compile_role_aware_initial_rotations(
        {
            "group:V": {"ind_ofsortednodes": [0]},
            "group:C": {"ind_ofsortednodes": [1]},
        },
        semantic_snapshot=OptimizationSemanticSnapshot(
            family_name="ROLE-AWARE",
            graph_phase="sG",
        ),
    )

    assert disabled == {}
    assert optimizer.role_aware_local_placement_debug_records["V0"]["fallback_reason"] == "guard_disabled"
    assert optimizer.role_aware_local_placement_debug_records["C0"]["fallback_reason"] == "guard_disabled"

    optimizer.use_role_aware_local_placement = True
    missing_snapshot = optimizer._compile_role_aware_initial_rotations(
        {
            "group:V": {"ind_ofsortednodes": [0]},
            "group:C": {"ind_ofsortednodes": [1]},
        },
        semantic_snapshot=None,
    )

    assert missing_snapshot == {}
    assert optimizer.role_aware_local_placement_debug_records["V0"]["fallback_reason"] == "missing_semantic_snapshot"
    assert optimizer.role_aware_local_placement_debug_records["C0"]["fallback_reason"] == "missing_semantic_snapshot"


def test_compile_role_aware_initial_rotations_records_selected_and_fallback_debug_details():
    rotation_expected = np.array(
        [
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    translation_expected = np.array([5.0, -2.0, 0.5])
    source_anchor_real = (2.0, 0.0, 0.0)
    source_anchor_real_2 = (0.0, 0.0, 2.0)
    source_direction_null = (0.0, 1.0, 0.0)
    target_anchor_real = tuple(
        np.dot(np.asarray(source_anchor_real), rotation_expected) + translation_expected
    )
    target_anchor_real_2 = tuple(
        np.dot(np.asarray(source_anchor_real_2), rotation_expected) + translation_expected
    )
    target_direction_null = tuple(
        np.dot(np.asarray(source_direction_null), rotation_expected)
    )

    semantic_snapshot = OptimizationSemanticSnapshot(
        family_name="ROLE-AWARE",
        graph_phase="sG",
        graph_node_records={
            "V0": GraphNodeSemanticRecord(
                node_id="V0",
                role_id="node:VA",
                role_class="V",
                slot_rules=(
                    {
                        "attachment_index": 0,
                        "slot_type": "XA",
                        "anchor_vector": source_anchor_real,
                        "chemistry_direction": source_anchor_real,
                    },
                    {
                        "attachment_index": 1,
                        "slot_type": "XB",
                        "anchor_vector": source_direction_null,
                        "chemistry_direction": source_direction_null,
                    },
                    {
                        "attachment_index": 2,
                        "slot_type": "XC",
                        "anchor_vector": source_anchor_real_2,
                        "chemistry_direction": source_anchor_real_2,
                    },
                ),
                incident_edge_ids=("V0|V1", "V0|V2", "V0|V3"),
                incident_edge_role_ids=("edge:EA", "edge:EB", "edge:EC"),
                incident_edge_constraints=(
                    {"edge_id": "V0|V1", "edge_role_id": "edge:EA", "slot_index": 0},
                    {"edge_id": "V0|V2", "edge_role_id": "edge:EB", "slot_index": 1},
                    {"edge_id": "V0|V3", "edge_role_id": "edge:EC", "slot_index": 2},
                ),
            ),
            "Q0": GraphNodeSemanticRecord(
                node_id="Q0",
                role_id="node:QA",
                role_class="Q",
            ),
        },
        graph_edge_records={
            "V0|V1": GraphEdgeSemanticRecord(
                edge_id="V0|V1",
                graph_edge=("V0", "V1"),
                edge_role_id="edge:EA",
                path_type="V-E-V",
                endpoint_node_ids=("V0", "V1"),
                endpoint_role_ids=("node:VA", "node:VB"),
                endpoint_pattern=("VA", "EA", "VB"),
                slot_index={"V0": 0, "V1": 0},
                slot_rules=(
                    {"attachment_index": 0, "slot_type": "XA", "endpoint_side": "V"},
                ),
                metadata={"target_point_by_node": {"V0": target_anchor_real}},
            ),
            "V0|V2": GraphEdgeSemanticRecord(
                edge_id="V0|V2",
                graph_edge=("V0", "V2"),
                edge_role_id="edge:EB",
                path_type="V-E-V",
                endpoint_node_ids=("V0", "V2"),
                endpoint_role_ids=("node:VA", "node:VC"),
                endpoint_pattern=("VA", "EB", "VC"),
                slot_index={"V0": 1, "V2": 0},
                slot_rules=(
                    {"attachment_index": 1, "slot_type": "XB", "endpoint_side": "V"},
                ),
                resolve_mode="alignment_only",
                is_null_edge=True,
                null_payload_model="duplicated_zero_length_anchors",
                metadata={"target_vector_by_node": {"V0": target_direction_null}},
            ),
            "V0|V3": GraphEdgeSemanticRecord(
                edge_id="V0|V3",
                graph_edge=("V0", "V3"),
                edge_role_id="edge:EC",
                path_type="V-E-V",
                endpoint_node_ids=("V0", "V3"),
                endpoint_role_ids=("node:VA", "node:VD"),
                endpoint_pattern=("VA", "EC", "VD"),
                slot_index={"V0": 2, "V3": 0},
                slot_rules=(
                    {"attachment_index": 2, "slot_type": "XC", "endpoint_side": "V"},
                ),
                metadata={"target_point_by_node": {"V0": target_anchor_real_2}},
            ),
        },
        node_role_records={
            "node:VA": NodeRoleRecord(
                role_id="node:VA",
                family_alias="VA",
                role_class="V",
                slot_rules=(
                    {
                        "attachment_index": 0,
                        "slot_type": "XA",
                        "anchor_vector": source_anchor_real,
                        "chemistry_direction": source_anchor_real,
                    },
                    {
                        "attachment_index": 1,
                        "slot_type": "XB",
                        "anchor_vector": source_direction_null,
                        "chemistry_direction": source_direction_null,
                    },
                    {
                        "attachment_index": 2,
                        "slot_type": "XC",
                        "anchor_vector": source_anchor_real_2,
                        "chemistry_direction": source_anchor_real_2,
                    },
                ),
            ),
        },
        edge_role_records={
            "edge:EA": EdgeRoleRecord(
                role_id="edge:EA",
                family_alias="EA",
                role_class="E",
                endpoint_pattern=("VA", "EA", "VB"),
                slot_rules=(
                    {"attachment_index": 0, "slot_type": "XA", "endpoint_side": "V"},
                ),
            ),
            "edge:EB": EdgeRoleRecord(
                role_id="edge:EB",
                family_alias="EB",
                role_class="E",
                endpoint_pattern=("VA", "EB", "VC"),
                slot_rules=(
                    {"attachment_index": 1, "slot_type": "XB", "endpoint_side": "V"},
                ),
                resolve_mode="alignment_only",
                null_edge_policy=NullEdgePolicyRecord(
                    edge_role_id="edge:EB",
                    edge_kind="null",
                    is_null_edge=True,
                    null_payload_model="duplicated_zero_length_anchors",
                ),
            ),
            "edge:EC": EdgeRoleRecord(
                role_id="edge:EC",
                family_alias="EC",
                role_class="E",
                endpoint_pattern=("VA", "EC", "VD"),
                slot_rules=(
                    {"attachment_index": 2, "slot_type": "XC", "endpoint_side": "V"},
                ),
            ),
        },
        null_edge_policy_records={
            "edge:EB": NullEdgePolicyRecord(
                edge_role_id="edge:EB",
                edge_kind="null",
                is_null_edge=True,
                null_payload_model="duplicated_zero_length_anchors",
            ),
        },
    )

    optimizer = opt.NetOptimizer(semantic_snapshot=semantic_snapshot)
    optimizer.sorted_nodes = ["V0", "Q0"]
    optimizer.use_role_aware_local_placement = True

    rotations = optimizer._compile_role_aware_initial_rotations(
        {
            "group:V": {"ind_ofsortednodes": [0]},
            "group:Q": {"ind_ofsortednodes": [1]},
        },
        semantic_snapshot=semantic_snapshot,
    )

    assert set(rotations) == {"group:V"}
    assert np.allclose(rotations["group:V"], rotation_expected)

    selected_debug = optimizer.role_aware_local_placement_debug_records["V0"]
    assert selected_debug["status"] == "selected"
    assert selected_debug["candidate_count"] == 1
    assert selected_debug["selected_assignment"] == {"V0|V1": 0, "V0|V2": 1, "V0|V3": 2}
    assert selected_debug["null_edge_count"] == 1
    assert selected_debug["alignment_only_count"] == 1
    assert selected_debug["resolve_mode_hints"] == ("alignment_only",)
    assert selected_debug["orientation_only_pair_count"] == 2
    assert selected_debug["used_ambiguity_resolution"] is False

    fallback_debug = optimizer.role_aware_local_placement_debug_records["Q0"]
    assert fallback_debug["status"] == "fallback"
    assert fallback_debug["fallback_reason"] == "unsupported_role_class"
    assert fallback_debug["candidate_count"] == 0


def test_compile_node_placement_contract_supports_default_role_snapshot():
    semantic_snapshot = OptimizationSemanticSnapshot(
        family_name="DEFAULT-FAMILY",
        graph_phase="G",
        graph_node_records={
            "V0": GraphNodeSemanticRecord(
                node_id="V0",
                role_id="node:default",
                role_class="V",
                slot_rules=(
                    {"attachment_index": 0, "slot_type": "XA"},
                    {"attachment_index": 1, "slot_type": "XB"},
                ),
                incident_edge_ids=("V0|V1", "V0|V2"),
                incident_edge_role_ids=("edge:default", "edge:default"),
                incident_edge_constraints=(
                    {"edge_id": "V0|V1", "edge_role_id": "edge:default", "slot_index": 0},
                    {"edge_id": "V0|V2", "edge_role_id": "edge:default", "slot_index": 1},
                ),
            ),
        },
        graph_edge_records={
            "V0|V1": GraphEdgeSemanticRecord(
                edge_id="V0|V1",
                graph_edge=("V0", "V1"),
                edge_role_id="edge:default",
                path_type="V-E-V",
                endpoint_node_ids=("V0", "V1"),
                endpoint_role_ids=("node:default", "node:default"),
                endpoint_pattern=("VA", "EA", "VA"),
                slot_index={"V0": 0, "V1": 0},
                slot_rules=(
                    {"attachment_index": 0, "slot_type": "XA", "endpoint_side": "V"},
                ),
            ),
            "V0|V2": GraphEdgeSemanticRecord(
                edge_id="V0|V2",
                graph_edge=("V0", "V2"),
                edge_role_id="edge:default",
                path_type="V-E-V",
                endpoint_node_ids=("V0", "V2"),
                endpoint_role_ids=("node:default", "node:default"),
                endpoint_pattern=("VA", "EA", "VA"),
                slot_index={"V0": 1, "V2": 0},
                slot_rules=(
                    {"attachment_index": 1, "slot_type": "XB", "endpoint_side": "V"},
                ),
            ),
        },
        node_role_records={
            "node:default": NodeRoleRecord(
                role_id="node:default",
                family_alias="default",
                role_class="V",
                slot_rules=(
                    {"attachment_index": 0, "slot_type": "XA"},
                    {"attachment_index": 1, "slot_type": "XB"},
                ),
            ),
        },
        edge_role_records={
            "edge:default": EdgeRoleRecord(
                role_id="edge:default",
                family_alias="default",
                role_class="E",
                endpoint_pattern=("VA", "EA", "VA"),
                slot_rules=(
                    {"attachment_index": 0, "slot_type": "XA", "endpoint_side": "V"},
                    {"attachment_index": 1, "slot_type": "XB", "endpoint_side": "V"},
                ),
            ),
        },
    )

    contract = compile_node_placement_contract(semantic_snapshot, "V0")

    assert isinstance(contract, NodePlacementContract)
    assert contract.node_role_id == "node:default"
    assert contract.local_slot_types == ("XA", "XB")
    assert contract.incident_edge_ids == ("V0|V1", "V0|V2")
    assert contract.incident_requirements[0].required_slot_type == "XA"
    assert contract.incident_requirements[1].required_slot_type == "XB"
    assert contract.incident_requirements[0].target_direction.remote_node_id == "V1"
    assert contract.null_edge_flags == {"V0|V1": False, "V0|V2": False}


def test_compile_node_placement_contract_preserves_role_aware_bundle_and_null_edge_hints():
    semantic_snapshot = OptimizationSemanticSnapshot(
        family_name="ROLE-AWARE",
        graph_phase="sG",
        graph_node_records={
            "C0": GraphNodeSemanticRecord(
                node_id="C0",
                role_id="node:CA",
                role_class="C",
                slot_rules=(
                    {"attachment_index": 0, "slot_type": "XA"},
                    {"attachment_index": 1, "slot_type": "XB"},
                ),
                incident_edge_ids=("V0|C0", "V1|C0"),
                incident_edge_role_ids=("edge:EA", "edge:EB"),
                incident_edge_constraints=(
                    {"edge_id": "V0|C0", "edge_role_id": "edge:EA", "slot_index": 0},
                    {"edge_id": "V1|C0", "edge_role_id": "edge:EB", "slot_index": 1},
                ),
                bundle_id="bundle:CA:0",
                bundle_order_hint={"ordered_attachment_indices": (0, 1)},
            ),
        },
        graph_edge_records={
            "V0|C0": GraphEdgeSemanticRecord(
                edge_id="V0|C0",
                graph_edge=("V0", "C0"),
                edge_role_id="edge:EA",
                path_type="V-E-C",
                endpoint_node_ids=("V0", "C0"),
                endpoint_role_ids=("node:VA", "node:CA"),
                endpoint_pattern=("VA", "EA", "CA"),
                slot_index={"V0": 0, "C0": 0},
                slot_rules=(
                    {"attachment_index": 0, "slot_type": "XA", "endpoint_side": "C"},
                ),
                bundle_id="bundle:CA:0",
                bundle_order_index=0,
                resolve_mode="ownership_transfer",
            ),
            "V1|C0": GraphEdgeSemanticRecord(
                edge_id="V1|C0",
                graph_edge=("V1", "C0"),
                edge_role_id="edge:EB",
                path_type="V-E-C",
                endpoint_node_ids=("V1", "C0"),
                endpoint_role_ids=("node:VB", "node:CA"),
                endpoint_pattern=("VB", "EB", "CA"),
                slot_index={"V1": 0, "C0": 1},
                slot_rules=(
                    {"attachment_index": 1, "slot_type": "XB", "endpoint_side": "C"},
                ),
                bundle_id="bundle:CA:0",
                bundle_order_index=1,
                resolve_mode="alignment_only",
                is_null_edge=True,
                null_payload_model="duplicated_zero_length_anchors",
                allows_null_fallback=True,
            ),
        },
        node_role_records={
            "node:CA": NodeRoleRecord(
                role_id="node:CA",
                family_alias="CA",
                role_class="C",
                slot_rules=(
                    {"attachment_index": 0, "slot_type": "XA"},
                    {"attachment_index": 1, "slot_type": "XB"},
                ),
            ),
        },
        edge_role_records={
            "edge:EA": EdgeRoleRecord(
                role_id="edge:EA",
                family_alias="EA",
                role_class="E",
                endpoint_pattern=("VA", "EA", "CA"),
                slot_rules=(
                    {"attachment_index": 0, "slot_type": "XA", "endpoint_side": "C"},
                ),
                resolve_mode="ownership_transfer",
            ),
            "edge:EB": EdgeRoleRecord(
                role_id="edge:EB",
                family_alias="EB",
                role_class="E",
                endpoint_pattern=("VB", "EB", "CA"),
                slot_rules=(
                    {"attachment_index": 1, "slot_type": "XB", "endpoint_side": "C"},
                ),
                resolve_mode="alignment_only",
            ),
        },
        bundle_records={
            "bundle:CA:0": BundleRecord(
                bundle_id="bundle:CA:0",
                owner_role_id="node:CA",
                attachment_edge_role_ids=("edge:EA", "edge:EB"),
                ordered_attachment_indices=(0, 1),
                order_kind="clockwise_local_topology",
            ),
        },
        null_edge_policy_records={
            "edge:EB": NullEdgePolicyRecord(
                edge_role_id="edge:EB",
                edge_kind="null",
                is_null_edge=True,
                null_payload_model="duplicated_zero_length_anchors",
                allows_null_fallback=True,
            ),
        },
    )

    optimizer = opt.NetOptimizer(semantic_snapshot=semantic_snapshot)
    contract = optimizer.compile_node_placement_contract("C0")

    assert contract.node_role_id == "node:CA"
    assert contract.bundle_id == "bundle:CA:0"
    assert contract.bundle_ordered_attachment_indices == (0, 1)
    assert contract.bundle_order_kind == "clockwise_local_topology"
    assert contract.resolve_mode_hints == ("ownership_transfer", "alignment_only")
    assert contract.null_edge_flags == {"V0|C0": False, "V1|C0": True}
    assert contract.incident_requirements[1].required_slot_type == "XB"
    assert contract.incident_requirements[1].is_null_edge is True
    assert contract.incident_requirements[1].target_direction.remote_role_id == "node:VB"


def test_compile_node_placement_contract_requires_snapshot():
    optimizer = opt.NetOptimizer()

    with pytest.raises(ValueError, match="OptimizationSemanticSnapshot is required"):
        optimizer.compile_node_placement_contract("V0")


def test_compile_legal_node_correspondences_returns_single_semantic_mapping():
    semantic_snapshot = OptimizationSemanticSnapshot(
        family_name="DEFAULT-FAMILY",
        graph_phase="G",
        graph_node_records={
            "V0": GraphNodeSemanticRecord(
                node_id="V0",
                role_id="node:default",
                role_class="V",
                slot_rules=(
                    {"attachment_index": 0, "slot_type": "XA"},
                    {"attachment_index": 1, "slot_type": "XB"},
                ),
                incident_edge_ids=("V0|V1", "V0|V2"),
                incident_edge_role_ids=("edge:default", "edge:default"),
                incident_edge_constraints=(
                    {"edge_id": "V0|V1", "edge_role_id": "edge:default"},
                    {"edge_id": "V0|V2", "edge_role_id": "edge:default"},
                ),
            ),
        },
        graph_edge_records={
            "V0|V1": GraphEdgeSemanticRecord(
                edge_id="V0|V1",
                graph_edge=("V0", "V1"),
                edge_role_id="edge:default",
                path_type="V-E-V",
                endpoint_node_ids=("V0", "V1"),
                endpoint_role_ids=("node:default", "node:default"),
                endpoint_pattern=("default", "default", "default"),
                slot_rules=(
                    {"attachment_index": 0, "slot_type": "XA", "endpoint_side": "V"},
                ),
            ),
            "V0|V2": GraphEdgeSemanticRecord(
                edge_id="V0|V2",
                graph_edge=("V0", "V2"),
                edge_role_id="edge:default",
                path_type="V-E-V",
                endpoint_node_ids=("V0", "V2"),
                endpoint_role_ids=("node:default", "node:default"),
                endpoint_pattern=("default", "default", "default"),
                slot_rules=(
                    {"attachment_index": 1, "slot_type": "XB", "endpoint_side": "V"},
                ),
            ),
        },
        node_role_records={
            "node:default": NodeRoleRecord(
                role_id="node:default",
                family_alias="default",
                role_class="V",
                slot_rules=(
                    {"attachment_index": 0, "slot_type": "XA"},
                    {"attachment_index": 1, "slot_type": "XB"},
                ),
            ),
        },
        edge_role_records={
            "edge:default": EdgeRoleRecord(
                role_id="edge:default",
                family_alias="default",
                role_class="E",
                endpoint_pattern=("default", "default", "default"),
                slot_rules=(
                    {"attachment_index": 0, "slot_type": "XA", "endpoint_side": "V"},
                    {"attachment_index": 1, "slot_type": "XB", "endpoint_side": "V"},
                ),
            ),
        },
    )

    correspondences = compile_legal_node_correspondences(semantic_snapshot, "V0")

    assert len(correspondences) == 1
    assert isinstance(correspondences[0], LegalNodeCorrespondence)
    assert correspondences[0].edge_to_slot_index == {"V0|V1": 0, "V0|V2": 1}
    assert tuple(
        assignment.slot_type for assignment in correspondences[0].assignments
    ) == ("XA", "XB")


def test_compile_legal_node_correspondences_rejects_illegal_slot_mapping():
    semantic_snapshot = OptimizationSemanticSnapshot(
        family_name="DEFAULT-FAMILY",
        graph_phase="G",
        graph_node_records={
            "V0": GraphNodeSemanticRecord(
                node_id="V0",
                role_id="node:default",
                role_class="V",
                slot_rules=(
                    {"attachment_index": 0, "slot_type": "XA"},
                ),
                incident_edge_ids=("V0|V1",),
                incident_edge_role_ids=("edge:default",),
                incident_edge_constraints=(
                    {"edge_id": "V0|V1", "edge_role_id": "edge:default"},
                ),
            ),
        },
        graph_edge_records={
            "V0|V1": GraphEdgeSemanticRecord(
                edge_id="V0|V1",
                graph_edge=("V0", "V1"),
                edge_role_id="edge:default",
                path_type="V-E-V",
                endpoint_node_ids=("V0", "V1"),
                endpoint_role_ids=("node:default", "node:default"),
                endpoint_pattern=("default", "default", "default"),
                slot_rules=(
                    {"attachment_index": 0, "slot_type": "XB", "endpoint_side": "V"},
                ),
            ),
        },
        node_role_records={
            "node:default": NodeRoleRecord(
                role_id="node:default",
                family_alias="default",
                role_class="V",
                slot_rules=(
                    {"attachment_index": 0, "slot_type": "XA"},
                ),
            ),
        },
        edge_role_records={
            "edge:default": EdgeRoleRecord(
                role_id="edge:default",
                family_alias="default",
                role_class="E",
                endpoint_pattern=("default", "default", "default"),
                slot_rules=(
                    {"attachment_index": 0, "slot_type": "XB", "endpoint_side": "V"},
                ),
            ),
        },
    )

    correspondences = compile_legal_node_correspondences(semantic_snapshot, "V0")

    assert correspondences == ()


def test_compile_legal_node_correspondences_enumerates_small_legal_candidate_set():
    semantic_snapshot = OptimizationSemanticSnapshot(
        family_name="ROLE-AWARE",
        graph_phase="sG",
        graph_node_records={
            "V0": GraphNodeSemanticRecord(
                node_id="V0",
                role_id="node:VA",
                role_class="V",
                slot_rules=(
                    {"attachment_index": 0, "slot_type": "XA"},
                    {"attachment_index": 1, "slot_type": "XA"},
                ),
                incident_edge_ids=("V0|C0", "V0|C1"),
                incident_edge_role_ids=("edge:EA", "edge:EA"),
                incident_edge_constraints=(
                    {"edge_id": "V0|C0", "edge_role_id": "edge:EA"},
                    {"edge_id": "V0|C1", "edge_role_id": "edge:EA"},
                ),
            ),
        },
        graph_edge_records={
            "V0|C0": GraphEdgeSemanticRecord(
                edge_id="V0|C0",
                graph_edge=("V0", "C0"),
                edge_role_id="edge:EA",
                path_type="V-E-C",
                endpoint_node_ids=("V0", "C0"),
                endpoint_role_ids=("node:VA", "node:CA"),
                endpoint_pattern=("VA", "EA", "CA"),
                slot_rules=(
                    {"attachment_index": 0, "slot_type": "XA", "endpoint_side": "V"},
                ),
            ),
            "V0|C1": GraphEdgeSemanticRecord(
                edge_id="V0|C1",
                graph_edge=("V0", "C1"),
                edge_role_id="edge:EA",
                path_type="V-E-C",
                endpoint_node_ids=("V0", "C1"),
                endpoint_role_ids=("node:VA", "node:CA"),
                endpoint_pattern=("VA", "EA", "CA"),
                slot_rules=(
                    {"attachment_index": 0, "slot_type": "XA", "endpoint_side": "V"},
                ),
            ),
        },
        node_role_records={
            "node:VA": NodeRoleRecord(
                role_id="node:VA",
                family_alias="VA",
                role_class="V",
                slot_rules=(
                    {"attachment_index": 0, "slot_type": "XA"},
                    {"attachment_index": 1, "slot_type": "XA"},
                ),
            ),
        },
        edge_role_records={
            "edge:EA": EdgeRoleRecord(
                role_id="edge:EA",
                family_alias="EA",
                role_class="E",
                endpoint_pattern=("VA", "EA", "CA"),
                slot_rules=(
                    {"attachment_index": 0, "slot_type": "XA", "endpoint_side": "V"},
                ),
            ),
        },
    )

    optimizer = opt.NetOptimizer(semantic_snapshot=semantic_snapshot)
    correspondences = optimizer.compile_legal_node_correspondences("V0")

    assert len(correspondences) == 2
    assert {tuple(candidate.edge_to_slot_index.items()) for candidate in correspondences} == {
        (("V0|C0", 0), ("V0|C1", 1)),
        (("V0|C0", 1), ("V0|C1", 0)),
    }


def test_compile_legal_node_correspondences_requires_snapshot():
    optimizer = opt.NetOptimizer()

    with pytest.raises(ValueError, match="OptimizationSemanticSnapshot is required"):
        optimizer.compile_legal_node_correspondences("V0")


def test_compile_local_rigid_initialization_returns_deterministic_svd_pose():
    rotation_expected = np.array(
        [
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    translation_expected = np.array([2.0, -1.0, 0.5])
    source_anchors = {
        0: (1.0, 0.0, 0.0),
        1: (0.0, 1.0, 0.0),
        2: (0.0, 0.0, 1.0),
    }
    target_points = {
        slot_index: tuple(np.dot(np.asarray(anchor), rotation_expected) + translation_expected)
        for slot_index, anchor in source_anchors.items()
    }

    semantic_snapshot = OptimizationSemanticSnapshot(
        family_name="ROLE-AWARE",
        graph_phase="sG",
        graph_node_records={
            "V0": GraphNodeSemanticRecord(
                node_id="V0",
                role_id="node:VA",
                role_class="V",
                slot_rules=(
                    {"attachment_index": 0, "slot_type": "XA", "anchor_vector": source_anchors[0]},
                    {"attachment_index": 1, "slot_type": "XB", "anchor_vector": source_anchors[1]},
                    {"attachment_index": 2, "slot_type": "XC", "anchor_vector": source_anchors[2]},
                ),
                incident_edge_ids=("V0|V1", "V0|V2", "V0|V3"),
                incident_edge_role_ids=("edge:EA", "edge:EB", "edge:EC"),
                incident_edge_constraints=(
                    {"edge_id": "V0|V1", "edge_role_id": "edge:EA", "slot_index": 0},
                    {"edge_id": "V0|V2", "edge_role_id": "edge:EB", "slot_index": 1},
                    {"edge_id": "V0|V3", "edge_role_id": "edge:EC", "slot_index": 2},
                ),
            ),
        },
        graph_edge_records={
            "V0|V1": GraphEdgeSemanticRecord(
                edge_id="V0|V1",
                graph_edge=("V0", "V1"),
                edge_role_id="edge:EA",
                path_type="V-E-V",
                endpoint_node_ids=("V0", "V1"),
                endpoint_role_ids=("node:VA", "node:VB"),
                endpoint_pattern=("VA", "EA", "VB"),
                slot_index={"V0": 0, "V1": 0},
                slot_rules=(
                    {"attachment_index": 0, "slot_type": "XA", "endpoint_side": "V"},
                ),
                metadata={"target_point_by_node": {"V0": target_points[0]}},
            ),
            "V0|V2": GraphEdgeSemanticRecord(
                edge_id="V0|V2",
                graph_edge=("V0", "V2"),
                edge_role_id="edge:EB",
                path_type="V-E-V",
                endpoint_node_ids=("V0", "V2"),
                endpoint_role_ids=("node:VA", "node:VC"),
                endpoint_pattern=("VA", "EB", "VC"),
                slot_index={"V0": 1, "V2": 0},
                slot_rules=(
                    {"attachment_index": 1, "slot_type": "XB", "endpoint_side": "V"},
                ),
                metadata={"target_point_by_node": {"V0": target_points[1]}},
            ),
            "V0|V3": GraphEdgeSemanticRecord(
                edge_id="V0|V3",
                graph_edge=("V0", "V3"),
                edge_role_id="edge:EC",
                path_type="V-E-V",
                endpoint_node_ids=("V0", "V3"),
                endpoint_role_ids=("node:VA", "node:VD"),
                endpoint_pattern=("VA", "EC", "VD"),
                slot_index={"V0": 2, "V3": 0},
                slot_rules=(
                    {"attachment_index": 2, "slot_type": "XC", "endpoint_side": "V"},
                ),
                metadata={"target_point_by_node": {"V0": target_points[2]}},
            ),
        },
        node_role_records={
            "node:VA": NodeRoleRecord(
                role_id="node:VA",
                family_alias="VA",
                role_class="V",
                slot_rules=(
                    {"attachment_index": 0, "slot_type": "XA", "anchor_vector": source_anchors[0]},
                    {"attachment_index": 1, "slot_type": "XB", "anchor_vector": source_anchors[1]},
                    {"attachment_index": 2, "slot_type": "XC", "anchor_vector": source_anchors[2]},
                ),
            ),
        },
        edge_role_records={
            "edge:EA": EdgeRoleRecord(
                role_id="edge:EA",
                family_alias="EA",
                role_class="E",
                endpoint_pattern=("VA", "EA", "VB"),
                slot_rules=(
                    {"attachment_index": 0, "slot_type": "XA", "endpoint_side": "V"},
                ),
            ),
            "edge:EB": EdgeRoleRecord(
                role_id="edge:EB",
                family_alias="EB",
                role_class="E",
                endpoint_pattern=("VA", "EB", "VC"),
                slot_rules=(
                    {"attachment_index": 1, "slot_type": "XB", "endpoint_side": "V"},
                ),
            ),
            "edge:EC": EdgeRoleRecord(
                role_id="edge:EC",
                family_alias="EC",
                role_class="E",
                endpoint_pattern=("VA", "EC", "VD"),
                slot_rules=(
                    {"attachment_index": 2, "slot_type": "XC", "endpoint_side": "V"},
                ),
            ),
        },
    )

    rigid_init = compile_local_rigid_initialization(semantic_snapshot, "V0")

    assert isinstance(rigid_init, NodeLocalRigidInitialization)
    assert rigid_init.metadata["anchor_count"] == 3
    assert "anchor_vector" in rigid_init.source_anchor_representation
    assert "target_direction metadata" in rigid_init.target_anchor_representation
    assert np.allclose(np.asarray(rigid_init.rotation_matrix), rotation_expected)
    assert np.allclose(np.asarray(rigid_init.translation_vector), translation_expected)
    assert rigid_init.rmsd == pytest.approx(0.0)


def test_compile_local_rigid_initialization_requires_single_known_correspondence():
    semantic_snapshot = OptimizationSemanticSnapshot(
        family_name="ROLE-AWARE",
        graph_phase="sG",
        graph_node_records={
            "V0": GraphNodeSemanticRecord(
                node_id="V0",
                role_id="node:VA",
                role_class="V",
                slot_rules=(
                    {"attachment_index": 0, "slot_type": "XA", "anchor_vector": (1.0, 0.0, 0.0)},
                    {"attachment_index": 1, "slot_type": "XA", "anchor_vector": (0.0, 1.0, 0.0)},
                ),
                incident_edge_ids=("V0|C0", "V0|C1"),
                incident_edge_role_ids=("edge:EA", "edge:EA"),
                incident_edge_constraints=(
                    {"edge_id": "V0|C0", "edge_role_id": "edge:EA"},
                    {"edge_id": "V0|C1", "edge_role_id": "edge:EA"},
                ),
            ),
        },
        graph_edge_records={
            "V0|C0": GraphEdgeSemanticRecord(
                edge_id="V0|C0",
                graph_edge=("V0", "C0"),
                edge_role_id="edge:EA",
                path_type="V-E-C",
                endpoint_node_ids=("V0", "C0"),
                endpoint_role_ids=("node:VA", "node:CA"),
                endpoint_pattern=("VA", "EA", "CA"),
                slot_rules=(
                    {"attachment_index": 0, "slot_type": "XA", "endpoint_side": "V"},
                ),
                metadata={"target_point_by_node": {"V0": (0.0, 0.0, 0.0)}},
            ),
            "V0|C1": GraphEdgeSemanticRecord(
                edge_id="V0|C1",
                graph_edge=("V0", "C1"),
                edge_role_id="edge:EA",
                path_type="V-E-C",
                endpoint_node_ids=("V0", "C1"),
                endpoint_role_ids=("node:VA", "node:CA"),
                endpoint_pattern=("VA", "EA", "CA"),
                slot_rules=(
                    {"attachment_index": 0, "slot_type": "XA", "endpoint_side": "V"},
                ),
                metadata={"target_point_by_node": {"V0": (1.0, 1.0, 0.0)}},
            ),
        },
        node_role_records={
            "node:VA": NodeRoleRecord(
                role_id="node:VA",
                family_alias="VA",
                role_class="V",
                slot_rules=(
                    {"attachment_index": 0, "slot_type": "XA", "anchor_vector": (1.0, 0.0, 0.0)},
                    {"attachment_index": 1, "slot_type": "XA", "anchor_vector": (0.0, 1.0, 0.0)},
                ),
            ),
        },
        edge_role_records={
            "edge:EA": EdgeRoleRecord(
                role_id="edge:EA",
                family_alias="EA",
                role_class="E",
                endpoint_pattern=("VA", "EA", "CA"),
                slot_rules=(
                    {"attachment_index": 0, "slot_type": "XA", "endpoint_side": "V"},
                ),
            ),
        },
    )

    with pytest.raises(ValueError, match="single legal correspondence is required"):
        compile_local_rigid_initialization(semantic_snapshot, "V0")


def test_compile_local_rigid_initialization_treats_null_alignment_edges_as_orientation_only():
    rotation_expected = np.array(
        [
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    translation_expected = np.array([5.0, -2.0, 0.5])
    source_anchor_real = (2.0, 0.0, 0.0)
    source_anchor_real_2 = (0.0, 0.0, 2.0)
    source_direction_null = (0.0, 1.0, 0.0)
    target_anchor_real = tuple(
        np.dot(np.asarray(source_anchor_real), rotation_expected) + translation_expected
    )
    target_anchor_real_2 = tuple(
        np.dot(np.asarray(source_anchor_real_2), rotation_expected) + translation_expected
    )
    target_direction_null = tuple(
        np.dot(np.asarray(source_direction_null), rotation_expected)
    )

    semantic_snapshot = OptimizationSemanticSnapshot(
        family_name="ROLE-AWARE",
        graph_phase="sG",
        graph_node_records={
            "V0": GraphNodeSemanticRecord(
                node_id="V0",
                role_id="node:VA",
                role_class="V",
                slot_rules=(
                    {
                        "attachment_index": 0,
                        "slot_type": "XA",
                        "anchor_vector": source_anchor_real,
                        "chemistry_direction": source_anchor_real,
                    },
                    {
                        "attachment_index": 1,
                        "slot_type": "XB",
                        "anchor_vector": source_direction_null,
                        "chemistry_direction": source_direction_null,
                    },
                    {
                        "attachment_index": 2,
                        "slot_type": "XC",
                        "anchor_vector": source_anchor_real_2,
                        "chemistry_direction": source_anchor_real_2,
                    },
                ),
                incident_edge_ids=("V0|V1", "V0|V2", "V0|V3"),
                incident_edge_role_ids=("edge:EA", "edge:EB", "edge:EC"),
                incident_edge_constraints=(
                    {"edge_id": "V0|V1", "edge_role_id": "edge:EA", "slot_index": 0},
                    {"edge_id": "V0|V2", "edge_role_id": "edge:EB", "slot_index": 1},
                    {"edge_id": "V0|V3", "edge_role_id": "edge:EC", "slot_index": 2},
                ),
            ),
        },
        graph_edge_records={
            "V0|V1": GraphEdgeSemanticRecord(
                edge_id="V0|V1",
                graph_edge=("V0", "V1"),
                edge_role_id="edge:EA",
                path_type="V-E-V",
                endpoint_node_ids=("V0", "V1"),
                endpoint_role_ids=("node:VA", "node:VB"),
                endpoint_pattern=("VA", "EA", "VB"),
                slot_index={"V0": 0, "V1": 0},
                slot_rules=(
                    {"attachment_index": 0, "slot_type": "XA", "endpoint_side": "V"},
                ),
                metadata={"target_point_by_node": {"V0": target_anchor_real}},
            ),
            "V0|V2": GraphEdgeSemanticRecord(
                edge_id="V0|V2",
                graph_edge=("V0", "V2"),
                edge_role_id="edge:EB",
                path_type="V-E-V",
                endpoint_node_ids=("V0", "V2"),
                endpoint_role_ids=("node:VA", "node:VC"),
                endpoint_pattern=("VA", "EB", "VC"),
                slot_index={"V0": 1, "V2": 0},
                slot_rules=(
                    {"attachment_index": 1, "slot_type": "XB", "endpoint_side": "V"},
                ),
                resolve_mode="alignment_only",
                is_null_edge=True,
                null_payload_model="duplicated_zero_length_anchors",
                metadata={"target_vector_by_node": {"V0": target_direction_null}},
            ),
            "V0|V3": GraphEdgeSemanticRecord(
                edge_id="V0|V3",
                graph_edge=("V0", "V3"),
                edge_role_id="edge:EC",
                path_type="V-E-V",
                endpoint_node_ids=("V0", "V3"),
                endpoint_role_ids=("node:VA", "node:VD"),
                endpoint_pattern=("VA", "EC", "VD"),
                slot_index={"V0": 2, "V3": 0},
                slot_rules=(
                    {"attachment_index": 2, "slot_type": "XC", "endpoint_side": "V"},
                ),
                metadata={"target_point_by_node": {"V0": target_anchor_real_2}},
            ),
        },
        node_role_records={
            "node:VA": NodeRoleRecord(
                role_id="node:VA",
                family_alias="VA",
                role_class="V",
                slot_rules=(
                    {
                        "attachment_index": 0,
                        "slot_type": "XA",
                        "anchor_vector": source_anchor_real,
                        "chemistry_direction": source_anchor_real,
                    },
                    {
                        "attachment_index": 1,
                        "slot_type": "XB",
                        "anchor_vector": source_direction_null,
                        "chemistry_direction": source_direction_null,
                    },
                    {
                        "attachment_index": 2,
                        "slot_type": "XC",
                        "anchor_vector": source_anchor_real_2,
                        "chemistry_direction": source_anchor_real_2,
                    },
                ),
            ),
        },
        edge_role_records={
            "edge:EA": EdgeRoleRecord(
                role_id="edge:EA",
                family_alias="EA",
                role_class="E",
                endpoint_pattern=("VA", "EA", "VB"),
                slot_rules=(
                    {"attachment_index": 0, "slot_type": "XA", "endpoint_side": "V"},
                ),
            ),
            "edge:EB": EdgeRoleRecord(
                role_id="edge:EB",
                family_alias="EB",
                role_class="E",
                endpoint_pattern=("VA", "EB", "VC"),
                slot_rules=(
                    {"attachment_index": 1, "slot_type": "XB", "endpoint_side": "V"},
                ),
                resolve_mode="alignment_only",
                null_edge_policy=NullEdgePolicyRecord(
                    edge_role_id="edge:EB",
                    edge_kind="null",
                    is_null_edge=True,
                    null_payload_model="duplicated_zero_length_anchors",
                ),
            ),
            "edge:EC": EdgeRoleRecord(
                role_id="edge:EC",
                family_alias="EC",
                role_class="E",
                endpoint_pattern=("VA", "EC", "VD"),
                slot_rules=(
                    {"attachment_index": 2, "slot_type": "XC", "endpoint_side": "V"},
                ),
            ),
        },
        null_edge_policy_records={
            "edge:EB": NullEdgePolicyRecord(
                edge_role_id="edge:EB",
                edge_kind="null",
                is_null_edge=True,
                null_payload_model="duplicated_zero_length_anchors",
            ),
        },
    )

    rigid_init = compile_local_rigid_initialization(semantic_snapshot, "V0")

    assert np.allclose(np.asarray(rigid_init.rotation_matrix), rotation_expected)
    assert np.allclose(np.asarray(rigid_init.translation_vector), translation_expected)
    assert rigid_init.metadata["real_anchor_pair_count"] == 2
    assert rigid_init.metadata["orientation_only_pair_count"] == 2
    assert rigid_init.metadata["translation_mode"] == "real_anchor_centroid"
    assert sum(
        pair.metadata["pair_kind"] == "orientation_only" for pair in rigid_init.anchor_pairs
    ) == 2


def test_compile_discrete_ambiguity_resolution_scores_legal_candidates_and_selects_best():
    source_anchors = {
        0: (1.0, 0.0, 0.0),
        1: (0.0, 1.0, 0.0),
    }

    semantic_snapshot = OptimizationSemanticSnapshot(
        family_name="ROLE-AWARE",
        graph_phase="sG",
        graph_node_records={
            "V0": GraphNodeSemanticRecord(
                node_id="V0",
                role_id="node:VA",
                role_class="V",
                slot_rules=(
                    {"attachment_index": 0, "slot_type": "XA", "anchor_vector": source_anchors[0]},
                    {"attachment_index": 1, "slot_type": "XA", "anchor_vector": source_anchors[1]},
                ),
                incident_edge_ids=("V0|C0", "V0|C1"),
                incident_edge_role_ids=("edge:EA", "edge:EA"),
                incident_edge_constraints=(
                    {"edge_id": "V0|C0", "edge_role_id": "edge:EA"},
                    {"edge_id": "V0|C1", "edge_role_id": "edge:EA"},
                ),
            ),
        },
        graph_edge_records={
            "V0|C0": GraphEdgeSemanticRecord(
                edge_id="V0|C0",
                graph_edge=("V0", "C0"),
                edge_role_id="edge:EA",
                path_type="V-E-C",
                endpoint_node_ids=("V0", "C0"),
                endpoint_role_ids=("node:VA", "node:CA"),
                endpoint_pattern=("VA", "EA", "CA"),
                slot_rules=(
                    {"attachment_index": 0, "slot_type": "XA", "endpoint_side": "V"},
                ),
                metadata={"target_point_by_node": {"V0": source_anchors[0]}},
            ),
            "V0|C1": GraphEdgeSemanticRecord(
                edge_id="V0|C1",
                graph_edge=("V0", "C1"),
                edge_role_id="edge:EA",
                path_type="V-E-C",
                endpoint_node_ids=("V0", "C1"),
                endpoint_role_ids=("node:VA", "node:CA"),
                endpoint_pattern=("VA", "EA", "CA"),
                slot_rules=(
                    {"attachment_index": 0, "slot_type": "XA", "endpoint_side": "V"},
                ),
                metadata={"target_point_by_node": {"V0": source_anchors[1]}},
            ),
        },
        node_role_records={
            "node:VA": NodeRoleRecord(
                role_id="node:VA",
                family_alias="VA",
                role_class="V",
                slot_rules=(
                    {"attachment_index": 0, "slot_type": "XA", "anchor_vector": source_anchors[0]},
                    {"attachment_index": 1, "slot_type": "XA", "anchor_vector": source_anchors[1]},
                ),
            ),
        },
        edge_role_records={
            "edge:EA": EdgeRoleRecord(
                role_id="edge:EA",
                family_alias="EA",
                role_class="E",
                endpoint_pattern=("VA", "EA", "CA"),
                slot_rules=(
                    {"attachment_index": 0, "slot_type": "XA", "endpoint_side": "V"},
                ),
            ),
        },
    )

    optimizer = opt.NetOptimizer(semantic_snapshot=semantic_snapshot)
    resolution = optimizer.compile_discrete_ambiguity_resolution("V0")

    assert isinstance(resolution, NodeDiscreteAmbiguityResolution)
    assert resolution.metadata["candidate_count"] == 2
    assert len(resolution.candidates) == 2
    assert resolution.selected_candidate.score == pytest.approx(0.0)
    assert resolution.selected_candidate.tie_break_signature == (0, 1)
    assert resolution.selected_correspondence.edge_to_slot_index == {"V0|C0": 0, "V0|C1": 1}
    assert resolution.selected_initialization.rmsd == pytest.approx(0.0)
    assert sorted(candidate.score for candidate in resolution.candidates)[1] > 0.0


def test_compile_discrete_ambiguity_resolution_keeps_unique_case_trivial():
    semantic_snapshot = OptimizationSemanticSnapshot(
        family_name="ROLE-AWARE",
        graph_phase="sG",
        graph_node_records={
            "V0": GraphNodeSemanticRecord(
                node_id="V0",
                role_id="node:VA",
                role_class="V",
                slot_rules=(
                    {"attachment_index": 0, "slot_type": "XA", "anchor_vector": (1.0, 0.0, 0.0)},
                    {"attachment_index": 1, "slot_type": "XB", "anchor_vector": (0.0, 1.0, 0.0)},
                ),
                incident_edge_ids=("V0|V1", "V0|V2"),
                incident_edge_role_ids=("edge:EA", "edge:EB"),
                incident_edge_constraints=(
                    {"edge_id": "V0|V1", "edge_role_id": "edge:EA", "slot_index": 0},
                    {"edge_id": "V0|V2", "edge_role_id": "edge:EB", "slot_index": 1},
                ),
            ),
        },
        graph_edge_records={
            "V0|V1": GraphEdgeSemanticRecord(
                edge_id="V0|V1",
                graph_edge=("V0", "V1"),
                edge_role_id="edge:EA",
                path_type="V-E-V",
                endpoint_node_ids=("V0", "V1"),
                endpoint_role_ids=("node:VA", "node:VB"),
                endpoint_pattern=("VA", "EA", "VB"),
                slot_index={"V0": 0, "V1": 0},
                slot_rules=(
                    {"attachment_index": 0, "slot_type": "XA", "endpoint_side": "V"},
                ),
                metadata={"target_point_by_node": {"V0": (1.0, 0.0, 0.0)}},
            ),
            "V0|V2": GraphEdgeSemanticRecord(
                edge_id="V0|V2",
                graph_edge=("V0", "V2"),
                edge_role_id="edge:EB",
                path_type="V-E-V",
                endpoint_node_ids=("V0", "V2"),
                endpoint_role_ids=("node:VA", "node:VC"),
                endpoint_pattern=("VA", "EB", "VC"),
                slot_index={"V0": 1, "V2": 0},
                slot_rules=(
                    {"attachment_index": 1, "slot_type": "XB", "endpoint_side": "V"},
                ),
                metadata={"target_point_by_node": {"V0": (0.0, 1.0, 0.0)}},
            ),
        },
        node_role_records={
            "node:VA": NodeRoleRecord(
                role_id="node:VA",
                family_alias="VA",
                role_class="V",
                slot_rules=(
                    {"attachment_index": 0, "slot_type": "XA", "anchor_vector": (1.0, 0.0, 0.0)},
                    {"attachment_index": 1, "slot_type": "XB", "anchor_vector": (0.0, 1.0, 0.0)},
                ),
            ),
        },
        edge_role_records={
            "edge:EA": EdgeRoleRecord(
                role_id="edge:EA",
                family_alias="EA",
                role_class="E",
                endpoint_pattern=("VA", "EA", "VB"),
                slot_rules=(
                    {"attachment_index": 0, "slot_type": "XA", "endpoint_side": "V"},
                ),
            ),
            "edge:EB": EdgeRoleRecord(
                role_id="edge:EB",
                family_alias="EB",
                role_class="E",
                endpoint_pattern=("VA", "EB", "VC"),
                slot_rules=(
                    {"attachment_index": 1, "slot_type": "XB", "endpoint_side": "V"},
                ),
            ),
        },
    )

    resolution = compile_discrete_ambiguity_resolution(semantic_snapshot, "V0")

    assert len(resolution.candidates) == 1
    assert resolution.selected_candidate_index == 0
    assert resolution.selected_candidate.score == pytest.approx(0.0)
    assert resolution.selected_correspondence.edge_to_slot_index == {"V0|V1": 0, "V0|V2": 1}


def test_get_rot_trans_matrix_uses_superimpose_result(monkeypatch):
    g = nx.Graph()
    g.add_node("N", ccoords=np.array([0.0, 0.0, 0.0]))
    g.add_node("B", ccoords=np.array([1.0, 0.0, 0.0]))
    g.add_node("C", ccoords=np.array([0.0, 1.0, 0.0]))
    g.add_edge("N", "B")
    g.add_edge("N", "C")

    xdict = {0: np.array([["X", 1.0, 0.0, 0.0], ["X", 0.0, 1.0, 0.0]], dtype=object)}

    rot_expected = np.eye(3)
    tran_expected = np.array([0.1, 0.2, 0.3])

    monkeypatch.setattr(
        opt,
        "superimpose_rotation_only",
        lambda a, b: (0.0, rot_expected, tran_expected),
    )

    rot, tran = opt.get_rot_trans_matrix("N", g, ["N"], xdict)

    assert np.allclose(rot, rot_expected)
    assert np.allclose(tran, tran_expected)


def test_prepare_role_fragment_payloads_keeps_single_role_scalar_fallback():
    optimizer = opt.NetOptimizer()
    optimizer.constant_length = 1.54
    optimizer.linker_frag_length = 3.0
    optimizer.fake_edge = False
    optimizer.sorted_nodes = ["V0_[0 0 0]", "V1_[0 0 0]"]
    optimizer.sorted_edges = [("V0_[0 0 0]", "V1_[0 0 0]")]
    optimizer.V_data = _fragment_table([
        _fragment_row("N", "N", [0.0, 0.0, 0.0]),
        _fragment_row("C", "C", [0.5, 0.0, 0.0]),
    ])
    optimizer.V_X_data = _fragment_table([
        _fragment_row("X", "X", [2.0, 0.0, 0.0]),
    ])
    optimizer.E_data = _fragment_table([
        _fragment_row("L", "L", [-1.0, 0.0, 0.0]),
        _fragment_row("L", "L", [1.0, 0.0, 0.0]),
    ])
    optimizer.E_X_data = _fragment_table([
        _fragment_row("X", "X", [-1.0, 0.0, 0.0]),
        _fragment_row("X", "X", [1.0, 0.0, 0.0]),
    ])

    g = nx.Graph()
    g.add_node("V0_[0 0 0]",
               ccoords=np.array([0.0, 0.0, 0.0]),
               node_role_id="node:default")
    g.add_node("V1_[0 0 0]",
               ccoords=np.array([4.0, 0.0, 0.0]),
               node_role_id="node:default")
    g.add_edge("V0_[0 0 0]", "V1_[0 0 0]", edge_role_id="edge:default")

    optimizer._prepare_role_fragment_payloads(g)
    target_edge_lengths = optimizer._get_target_edge_lengths()

    expected = 3.0 + 2 * 1.54 + 2.0 + 2.0
    assert np.isclose(
        target_edge_lengths[("V0_[0 0 0]", "V1_[0 0 0]")], expected)
    assert np.allclose(
        optimizer.node_fragment_payloads["V0_[0 0 0]"]["coords"],
        optimizer.V_data[:, 5:8].astype(float),
    )
    assert np.allclose(
        optimizer.edge_fragment_payloads[("V0_[0 0 0]",
                                          "V1_[0 0 0]")]["coords"],
        optimizer.E_data[:, 5:8].astype(float),
    )


def test_role_aware_optimizer_uses_role_registries_for_grouping_and_edge_payloads(
    monkeypatch,
):
    optimizer = opt.NetOptimizer()
    optimizer.constant_length = 1.54
    optimizer.linker_frag_length = 1.0
    optimizer.fake_edge = False
    optimizer.sorted_nodes = ["V0_[0 0 0]", "V0_[1 0 0]"]
    optimizer.sorted_edges = [("V0_[0 0 0]", "V0_[1 0 0]")]
    optimizer.V_data = _fragment_table([
        _fragment_row("G", "G", [0.0, 0.0, 0.0]),
    ])
    optimizer.V_X_data = _fragment_table([
        _fragment_row("X", "X", [9.0, 0.0, 0.0]),
    ])
    optimizer.E_data = _fragment_table([
        _fragment_row("GLOBAL", "GLOBAL", [-1.0, 0.0, 0.0]),
        _fragment_row("GLOBAL", "GLOBAL", [1.0, 0.0, 0.0]),
    ])
    optimizer.E_X_data = _fragment_table([
        _fragment_row("X", "X", [-1.0, 0.0, 0.0]),
        _fragment_row("X", "X", [1.0, 0.0, 0.0]),
    ])
    optimizer.node_role_registry = {
        "node:alpha": {
            "role_id": "node:alpha",
            "node_data": _fragment_table([
                _fragment_row("A", "A", [0.0, 0.0, 0.0]),
            ]),
            "node_X_data": _fragment_table([
                _fragment_row("X", "X", [1.0, 0.0, 0.0]),
            ]),
        },
        "node:beta": {
            "role_id": "node:beta",
            "node_data": _fragment_table([
                _fragment_row("B", "B", [0.0, 1.0, 0.0]),
            ]),
            "node_X_data": _fragment_table([
                _fragment_row("X", "X", [2.0, 0.0, 0.0]),
            ]),
        },
    }
    optimizer.edge_role_registry = {
        "edge:long": {
            "role_id": "edge:long",
            "linker_connectivity": 2,
            "linker_center_data": _fragment_table([
                _fragment_row("ROLE", "ROLE", [-2.0, 0.0, 0.0]),
                _fragment_row("ROLE", "ROLE", [2.0, 0.0, 0.0]),
            ]),
            "linker_center_X_data": _fragment_table([
                _fragment_row("X", "X", [-2.0, 0.0, 0.0]),
                _fragment_row("X", "X", [2.0, 0.0, 0.0]),
            ]),
            "linker_frag_length": 4.0,
            "linker_fake_edge": False,
        },
    }

    g = nx.Graph()
    g.add_node("V0_[0 0 0]",
               ccoords=np.array([0.0, 0.0, 0.0]),
               node_role_id="node:alpha")
    g.add_node("V0_[1 0 0]",
               ccoords=np.array([4.0, 0.0, 0.0]),
               node_role_id="node:beta")
    g.add_edge("V0_[0 0 0]", "V0_[1 0 0]", edge_role_id="edge:long")

    optimizer._prepare_role_fragment_payloads(g)
    node_pos_dict, node_x_pos_dict = optimizer._generate_pos_dict(g)
    monkeypatch.setattr(
        opt,
        "get_rot_trans_matrix",
        lambda node, graph, sorted_nodes, xdict: (np.eye(3), np.zeros(3)),
    )
    pname_set, pname_set_dict = optimizer._generate_pname_set(
        g, optimizer.sorted_nodes, node_x_pos_dict)

    assert len(pname_set) == 2
    assert np.allclose(optimizer.node_fragment_payloads["V0_[0 0 0]"]["coords"],
                       [[0.0, 0.0, 0.0]])
    assert np.allclose(optimizer.node_fragment_payloads["V0_[1 0 0]"]["coords"],
                       [[0.0, 1.0, 0.0]])
    assert np.allclose(
        optimizer.edge_fragment_payloads[("V0_[0 0 0]",
                                          "V0_[1 0 0]")]["coords"],
        [[-2.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
    )

    optimizer.pname_set_dict = pname_set_dict
    optimizer.sG = g.copy()
    optimizer.sc_unit_cell_inv = np.eye(3)
    optimizer.sc_rot_node_X_pos = {
        0: np.array([[0, 1.0, 0.0, 0.0]], dtype=object),
        1: np.array([[0, 3.0, 0.0, 0.0]], dtype=object),
    }
    optimizer.sc_rot_node_pos = node_pos_dict
    optimizer.optimized_pair = {("V0_[0 0 0]", "V0_[1 0 0]"): (0, 0)}

    monkeypatch.setattr(opt, "superimpose_rotation_only",
                        lambda a, b: (0.0, np.eye(3), np.zeros(3)))

    placed = optimizer.place_edge_in_net()

    assert placed.edges[("V0_[0 0 0]", "V0_[1 0 0]")]["c_points"][0, 0] == "ROLE"
    assert placed.nodes["V0_[0 0 0]"]["c_points"][0, 0] == "A"
    assert placed.nodes["V0_[1 0 0]"]["c_points"][0, 0] == "B"


def test_place_edge_in_net_uses_typed_resolved_anchors_for_role_aware_placement(
    monkeypatch,
):
    optimizer = opt.NetOptimizer()
    optimizer.constant_length = 0.0
    optimizer.use_role_aware_local_placement = True
    optimizer.sorted_nodes = ["V0", "V1"]
    optimizer.optimized_pair = {("V0", "V1"): (0, 0)}
    optimizer.sc_rot_node_X_pos = {
        0: np.array([[0, 9.0, 0.0, 0.0]], dtype=object),
        1: np.array([[0, 13.0, 0.0, 0.0]], dtype=object),
    }
    optimizer.sc_rot_node_attachment_lookup = {
        "V0": {("XA", 0): np.array([1.0, 0.0, 0.0])},
        "V1": {("XA", 0): np.array([3.0, 0.0, 0.0])},
    }
    optimizer.sc_rot_node_pos = {
        0: np.array([[0, 0.0, 0.0, 0.0]], dtype=object),
        1: np.array([[0, 4.0, 0.0, 0.0]], dtype=object),
    }
    optimizer.sc_unit_cell_inv = np.eye(3)
    optimizer.nodes_atom = {
        "V0": _fragment_table([["A", "A"]]),
        "V1": _fragment_table([["B", "B"]]),
    }
    optimizer.edge_fragment_payloads = {
        ("V0", "V1"): {
            "atom": _fragment_table([["ROLE", "ROLE"], ["ROLE", "ROLE"]]),
            "coords": np.array([[-2.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
            "x_coords": np.array([[-10.0, 0.0, 0.0], [10.0, 0.0, 0.0]]),
            "attachment_coords_by_type": {
                "XA": np.array([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
            },
            "linker_frag_length": 2.0,
            "fake_edge": False,
        },
        ("V1", "V0"): {
            "atom": _fragment_table([["ROLE", "ROLE"], ["ROLE", "ROLE"]]),
            "coords": np.array([[-2.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
            "x_coords": np.array([[-10.0, 0.0, 0.0], [10.0, 0.0, 0.0]]),
            "attachment_coords_by_type": {
                "XA": np.array([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
            },
            "linker_frag_length": 2.0,
            "fake_edge": False,
        },
    }
    optimizer.sG = nx.Graph()
    optimizer.sG.add_node("V0", ccoords=np.array([0.0, 0.0, 0.0]))
    optimizer.sG.add_node("V1", ccoords=np.array([4.0, 0.0, 0.0]))
    optimizer.sG.add_edge("V0", "V1", edge_role_id="edge:EA")
    optimizer.semantic_snapshot = OptimizationSemanticSnapshot(
        family_name="ROLE-AWARE",
        graph_phase="sG",
        graph_node_records={
            "V0": GraphNodeSemanticRecord(
                node_id="V0",
                role_id="node:VA",
                role_class="V",
                slot_rules=(
                    {
                        "attachment_index": 0,
                        "slot_type": "XA",
                        "anchor_source_type": "XA",
                        "anchor_source_ordinal": 0,
                    },
                ),
            ),
            "V1": GraphNodeSemanticRecord(
                node_id="V1",
                role_id="node:VB",
                role_class="V",
                slot_rules=(
                    {
                        "attachment_index": 1,
                        "slot_type": "XA",
                        "anchor_source_type": "XA",
                        "anchor_source_ordinal": 0,
                    },
                ),
            ),
        },
        graph_edge_records={
            "V0|V1": GraphEdgeSemanticRecord(
                edge_id="V0|V1",
                graph_edge=("V0", "V1"),
                edge_role_id="edge:EA",
                slot_index={"V0": 0, "V1": 1},
                slot_rules=(
                    {
                        "attachment_index": 0,
                        "slot_type": "XA",
                        "endpoint_side": "V",
                        "anchor_source_type": "XA",
                        "anchor_source_ordinal": 0,
                    },
                    {
                        "attachment_index": 1,
                        "slot_type": "XA",
                        "endpoint_side": "V",
                        "anchor_source_type": "XA",
                        "anchor_source_ordinal": 1,
                    },
                ),
            ),
        },
    )

    monkeypatch.setattr(
        opt,
        "superimpose_rotation_only",
        lambda a, b: (0.0, np.eye(3), np.zeros(3)),
    )

    placed = optimizer.place_edge_in_net()

    assert np.allclose(placed.edges[("V0", "V1")]["coords"], [2.0, 0.0, 0.0])


def test_place_edge_in_net_preserves_legacy_x_behavior_when_role_aware_guard_is_disabled(
    monkeypatch,
):
    optimizer = opt.NetOptimizer()
    optimizer.constant_length = 0.0
    optimizer.use_role_aware_local_placement = False
    optimizer.sorted_nodes = ["V0", "V1"]
    optimizer.optimized_pair = {("V0", "V1"): (0, 0)}
    optimizer.sc_rot_node_X_pos = {
        0: np.array([[0, 1.0, 0.0, 0.0]], dtype=object),
        1: np.array([[0, 3.0, 0.0, 0.0]], dtype=object),
    }
    optimizer.sc_rot_node_attachment_lookup = {
        "V0": {("XA", 0): np.array([9.0, 0.0, 0.0])},
        "V1": {("XA", 0): np.array([13.0, 0.0, 0.0])},
    }
    optimizer.sc_rot_node_pos = {
        0: np.array([[0, 0.0, 0.0, 0.0]], dtype=object),
        1: np.array([[0, 4.0, 0.0, 0.0]], dtype=object),
    }
    optimizer.sc_unit_cell_inv = np.eye(3)
    optimizer.nodes_atom = {
        "V0": _fragment_table([["A", "A"]]),
        "V1": _fragment_table([["B", "B"]]),
    }
    optimizer.edge_fragment_payloads = {
        ("V0", "V1"): {
            "atom": _fragment_table([["ROLE", "ROLE"], ["ROLE", "ROLE"]]),
            "coords": np.array([[-2.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
            "x_coords": np.array([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
            "attachment_coords_by_type": {
                "XA": np.array([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
            },
            "linker_frag_length": 2.0,
            "fake_edge": False,
        },
        ("V1", "V0"): {
            "atom": _fragment_table([["ROLE", "ROLE"], ["ROLE", "ROLE"]]),
            "coords": np.array([[-2.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
            "x_coords": np.array([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
            "attachment_coords_by_type": {
                "XA": np.array([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
            },
            "linker_frag_length": 2.0,
            "fake_edge": False,
        },
    }
    optimizer.sG = nx.Graph()
    optimizer.sG.add_node("V0", ccoords=np.array([0.0, 0.0, 0.0]))
    optimizer.sG.add_node("V1", ccoords=np.array([4.0, 0.0, 0.0]))
    optimizer.sG.add_edge("V0", "V1", edge_role_id="edge:EA")
    optimizer.semantic_snapshot = OptimizationSemanticSnapshot(
        family_name="ROLE-AWARE",
        graph_phase="sG",
        graph_node_records={
            "V0": GraphNodeSemanticRecord(
                node_id="V0",
                role_id="node:VA",
                role_class="V",
                slot_rules=(
                    {
                        "attachment_index": 0,
                        "slot_type": "XA",
                        "anchor_source_type": "XA",
                        "anchor_source_ordinal": 0,
                    },
                ),
            ),
            "V1": GraphNodeSemanticRecord(
                node_id="V1",
                role_id="node:VB",
                role_class="V",
                slot_rules=(
                    {
                        "attachment_index": 1,
                        "slot_type": "XA",
                        "anchor_source_type": "XA",
                        "anchor_source_ordinal": 0,
                    },
                ),
            ),
        },
        graph_edge_records={
            "V0|V1": GraphEdgeSemanticRecord(
                edge_id="V0|V1",
                graph_edge=("V0", "V1"),
                edge_role_id="edge:EA",
                slot_index={"V0": 0, "V1": 1},
                slot_rules=(
                    {
                        "attachment_index": 0,
                        "slot_type": "XA",
                        "endpoint_side": "V",
                        "anchor_source_type": "XA",
                        "anchor_source_ordinal": 0,
                    },
                    {
                        "attachment_index": 1,
                        "slot_type": "XA",
                        "endpoint_side": "V",
                        "anchor_source_type": "XA",
                        "anchor_source_ordinal": 1,
                    },
                ),
            ),
        },
    )

    monkeypatch.setattr(
        opt,
        "superimpose_rotation_only",
        lambda a, b: (0.0, np.eye(3), np.zeros(3)),
    )

    placed = optimizer.place_edge_in_net()

    assert np.allclose(placed.edges[("V0", "V1")]["coords"], [2.0, 0.0, 0.0])


def test_place_edge_in_net_preserves_legacy_literal_x_compatibility_through_resolved_anchors(
    monkeypatch,
):
    optimizer = opt.NetOptimizer()
    optimizer.constant_length = 0.0
    optimizer.use_role_aware_local_placement = True
    optimizer.sorted_nodes = ["V0", "V1"]
    optimizer.optimized_pair = {("V0", "V1"): (0, 0)}
    optimizer.sc_rot_node_X_pos = {
        0: np.array([[0, 1.0, 0.0, 0.0]], dtype=object),
        1: np.array([[0, 3.0, 0.0, 0.0]], dtype=object),
    }
    optimizer.sc_rot_node_attachment_lookup = {}
    optimizer.sc_rot_node_pos = {
        0: np.array([[0, 0.0, 0.0, 0.0]], dtype=object),
        1: np.array([[0, 4.0, 0.0, 0.0]], dtype=object),
    }
    optimizer.sc_unit_cell_inv = np.eye(3)
    optimizer.nodes_atom = {
        "V0": _fragment_table([["A", "A"]]),
        "V1": _fragment_table([["B", "B"]]),
    }
    optimizer.edge_fragment_payloads = {
        ("V0", "V1"): {
            "atom": _fragment_table([["ROLE", "ROLE"], ["ROLE", "ROLE"]]),
            "coords": np.array([[-2.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
            "x_coords": np.array([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
            "attachment_coords_by_type": {},
            "linker_frag_length": 2.0,
            "fake_edge": False,
        },
        ("V1", "V0"): {
            "atom": _fragment_table([["ROLE", "ROLE"], ["ROLE", "ROLE"]]),
            "coords": np.array([[-2.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
            "x_coords": np.array([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
            "attachment_coords_by_type": {},
            "linker_frag_length": 2.0,
            "fake_edge": False,
        },
    }
    optimizer.sG = nx.Graph()
    optimizer.sG.add_node("V0", ccoords=np.array([0.0, 0.0, 0.0]))
    optimizer.sG.add_node("V1", ccoords=np.array([4.0, 0.0, 0.0]))
    optimizer.sG.add_edge("V0", "V1", edge_role_id="edge:EA")
    optimizer.semantic_snapshot = OptimizationSemanticSnapshot(
        family_name="ROLE-AWARE",
        graph_phase="sG",
        graph_node_records={
            "V0": GraphNodeSemanticRecord(
                node_id="V0",
                role_id="node:VA",
                role_class="V",
                slot_rules=(
                    {
                        "attachment_index": 0,
                        "slot_type": "XA",
                        "anchor_source_type": "X",
                        "anchor_source_ordinal": 0,
                    },
                ),
            ),
            "V1": GraphNodeSemanticRecord(
                node_id="V1",
                role_id="node:VB",
                role_class="V",
                slot_rules=(
                    {
                        "attachment_index": 1,
                        "slot_type": "XA",
                        "anchor_source_type": "X",
                        "anchor_source_ordinal": 0,
                    },
                ),
            ),
        },
        graph_edge_records={
            "V0|V1": GraphEdgeSemanticRecord(
                edge_id="V0|V1",
                graph_edge=("V0", "V1"),
                edge_role_id="edge:EA",
                slot_index={"V0": 0, "V1": 1},
                slot_rules=(
                    {
                        "attachment_index": 0,
                        "slot_type": "XA",
                        "endpoint_side": "V",
                        "anchor_source_type": "X",
                        "anchor_source_ordinal": 0,
                    },
                    {
                        "attachment_index": 1,
                        "slot_type": "XA",
                        "endpoint_side": "V",
                        "anchor_source_type": "X",
                        "anchor_source_ordinal": 1,
                    },
                ),
            ),
        },
    )

    monkeypatch.setattr(
        opt,
        "superimpose_rotation_only",
        lambda a, b: (0.0, np.eye(3), np.zeros(3)),
    )

    placed = optimizer.place_edge_in_net()

    assert np.allclose(placed.edges[("V0", "V1")]["coords"], [2.0, 0.0, 0.0])


def test_place_edge_in_net_raises_semantic_error_for_missing_resolved_anchor_inputs():
    optimizer = opt.NetOptimizer()
    optimizer.constant_length = 0.0
    optimizer.use_role_aware_local_placement = True
    optimizer.sorted_nodes = ["V0", "V1"]
    optimizer.optimized_pair = {("V0", "V1"): (0, 0)}
    optimizer.sc_rot_node_X_pos = {
        0: np.array([[0, 1.0, 0.0, 0.0]], dtype=object),
        1: np.array([[0, 3.0, 0.0, 0.0]], dtype=object),
    }
    optimizer.sc_rot_node_attachment_lookup = {"V0": {}, "V1": {}}
    optimizer.sc_rot_node_pos = {
        0: np.array([[0, 0.0, 0.0, 0.0]], dtype=object),
        1: np.array([[0, 4.0, 0.0, 0.0]], dtype=object),
    }
    optimizer.sc_unit_cell_inv = np.eye(3)
    optimizer.nodes_atom = {
        "V0": _fragment_table([["A", "A"]]),
        "V1": _fragment_table([["B", "B"]]),
    }
    optimizer.edge_fragment_payloads = {
        ("V0", "V1"): {
            "atom": _fragment_table([["ROLE", "ROLE"], ["ROLE", "ROLE"]]),
            "coords": np.array([[-2.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
            "x_coords": np.array([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
            "attachment_coords_by_type": {},
            "linker_frag_length": 2.0,
            "fake_edge": False,
        },
        ("V1", "V0"): {
            "atom": _fragment_table([["ROLE", "ROLE"], ["ROLE", "ROLE"]]),
            "coords": np.array([[-2.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
            "x_coords": np.array([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
            "attachment_coords_by_type": {},
            "linker_frag_length": 2.0,
            "fake_edge": False,
        },
    }
    optimizer.sG = nx.Graph()
    optimizer.sG.add_node("V0", ccoords=np.array([0.0, 0.0, 0.0]))
    optimizer.sG.add_node("V1", ccoords=np.array([4.0, 0.0, 0.0]))
    optimizer.sG.add_edge("V0", "V1", edge_role_id="edge:EA")
    optimizer.semantic_snapshot = OptimizationSemanticSnapshot(
        family_name="ROLE-AWARE",
        graph_phase="sG",
        graph_node_records={
            "V0": GraphNodeSemanticRecord(
                node_id="V0",
                role_id="node:VA",
                role_class="V",
                slot_rules=(
                    {
                        "attachment_index": 0,
                        "slot_type": "XA",
                        "anchor_source_type": "XA",
                        "anchor_source_ordinal": 0,
                    },
                ),
            ),
            "V1": GraphNodeSemanticRecord(
                node_id="V1",
                role_id="node:VB",
                role_class="V",
                slot_rules=(
                    {
                        "attachment_index": 1,
                        "slot_type": "XA",
                        "anchor_source_type": "XA",
                        "anchor_source_ordinal": 0,
                    },
                ),
            ),
        },
        graph_edge_records={
            "V0|V1": GraphEdgeSemanticRecord(
                edge_id="V0|V1",
                graph_edge=("V0", "V1"),
                edge_role_id="edge:EA",
                slot_index={"V0": 0, "V1": 1},
                slot_rules=(
                    {
                        "attachment_index": 0,
                        "slot_type": "XA",
                        "endpoint_side": "V",
                        "anchor_source_type": "XA",
                        "anchor_source_ordinal": 0,
                    },
                    {
                        "attachment_index": 1,
                        "slot_type": "XA",
                        "endpoint_side": "V",
                        "anchor_source_type": "XA",
                        "anchor_source_ordinal": 1,
                    },
                ),
            ),
        },
    )

    with pytest.raises(ValueError, match="Missing builder-compiled resolved anchor"):
        optimizer.place_edge_in_net()


def test_compile_local_constrained_refinement_improves_combined_local_objective_without_remapping():
    theta = np.deg2rad(35.0)
    ideal_rotation = np.array(
        [
            [np.cos(theta), -np.sin(theta), 0.0],
            [np.sin(theta), np.cos(theta), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    translation = np.array([1.5, -0.5, 0.25])
    source_anchors = {
        0: (1.0, 0.0, 0.0),
        1: (0.0, 1.0, 0.0),
        2: (0.0, 0.0, 1.0),
    }
    target_vectors = {
        slot_index: tuple(np.dot(np.asarray(anchor), ideal_rotation))
        for slot_index, anchor in source_anchors.items()
    }
    target_points = {
        slot_index: tuple(np.dot(np.asarray(anchor), ideal_rotation) + translation)
        for slot_index, anchor in source_anchors.items()
    }
    noisy_target_points = {
        0: target_points[0],
        1: tuple(np.asarray(target_points[1]) + np.array([0.2, -0.1, 0.0])),
        2: target_points[2],
    }

    semantic_snapshot = OptimizationSemanticSnapshot(
        family_name="ROLE-AWARE",
        graph_phase="sG",
        graph_node_records={
            "V0": GraphNodeSemanticRecord(
                node_id="V0",
                role_id="node:VA",
                role_class="V",
                slot_rules=(
                    {"attachment_index": 0, "slot_type": "XA", "anchor_vector": source_anchors[0], "chemistry_direction": source_anchors[0]},
                    {"attachment_index": 1, "slot_type": "XB", "anchor_vector": source_anchors[1], "chemistry_direction": source_anchors[1]},
                    {"attachment_index": 2, "slot_type": "XC", "anchor_vector": source_anchors[2], "chemistry_direction": source_anchors[2]},
                ),
                incident_edge_ids=("V0|V1", "V0|V2", "V0|V3"),
                incident_edge_role_ids=("edge:EA", "edge:EB", "edge:EC"),
                incident_edge_constraints=(
                    {"edge_id": "V0|V1", "edge_role_id": "edge:EA", "slot_index": 0},
                    {"edge_id": "V0|V2", "edge_role_id": "edge:EB", "slot_index": 1},
                    {"edge_id": "V0|V3", "edge_role_id": "edge:EC", "slot_index": 2},
                ),
            ),
        },
        graph_edge_records={
            "V0|V1": GraphEdgeSemanticRecord(
                edge_id="V0|V1",
                graph_edge=("V0", "V1"),
                edge_role_id="edge:EA",
                path_type="V-E-V",
                endpoint_node_ids=("V0", "V1"),
                endpoint_role_ids=("node:VA", "node:VB"),
                endpoint_pattern=("VA", "EA", "VB"),
                slot_index={"V0": 0, "V1": 0},
                slot_rules=(
                    {"attachment_index": 0, "slot_type": "XA", "endpoint_side": "V"},
                ),
                metadata={
                    "target_point_by_node": {"V0": noisy_target_points[0]},
                    "target_vector_by_node": {"V0": target_vectors[0]},
                },
            ),
            "V0|V2": GraphEdgeSemanticRecord(
                edge_id="V0|V2",
                graph_edge=("V0", "V2"),
                edge_role_id="edge:EB",
                path_type="V-E-V",
                endpoint_node_ids=("V0", "V2"),
                endpoint_role_ids=("node:VA", "node:VC"),
                endpoint_pattern=("VA", "EB", "VC"),
                slot_index={"V0": 1, "V2": 0},
                slot_rules=(
                    {"attachment_index": 1, "slot_type": "XB", "endpoint_side": "V"},
                ),
                metadata={
                    "target_point_by_node": {"V0": noisy_target_points[1]},
                    "target_vector_by_node": {"V0": target_vectors[1]},
                },
            ),
            "V0|V3": GraphEdgeSemanticRecord(
                edge_id="V0|V3",
                graph_edge=("V0", "V3"),
                edge_role_id="edge:EC",
                path_type="V-E-V",
                endpoint_node_ids=("V0", "V3"),
                endpoint_role_ids=("node:VA", "node:VD"),
                endpoint_pattern=("VA", "EC", "VD"),
                slot_index={"V0": 2, "V3": 0},
                slot_rules=(
                    {"attachment_index": 2, "slot_type": "XC", "endpoint_side": "V"},
                ),
                metadata={
                    "target_point_by_node": {"V0": noisy_target_points[2]},
                    "target_vector_by_node": {"V0": target_vectors[2]},
                },
            ),
        },
        node_role_records={
            "node:VA": NodeRoleRecord(
                role_id="node:VA",
                family_alias="VA",
                role_class="V",
                slot_rules=(
                    {"attachment_index": 0, "slot_type": "XA", "anchor_vector": source_anchors[0], "chemistry_direction": source_anchors[0]},
                    {"attachment_index": 1, "slot_type": "XB", "anchor_vector": source_anchors[1], "chemistry_direction": source_anchors[1]},
                    {"attachment_index": 2, "slot_type": "XC", "anchor_vector": source_anchors[2], "chemistry_direction": source_anchors[2]},
                ),
            ),
        },
        edge_role_records={
            "edge:EA": EdgeRoleRecord(
                role_id="edge:EA",
                family_alias="EA",
                role_class="E",
                endpoint_pattern=("VA", "EA", "VB"),
                slot_rules=(
                    {"attachment_index": 0, "slot_type": "XA", "endpoint_side": "V"},
                ),
            ),
            "edge:EB": EdgeRoleRecord(
                role_id="edge:EB",
                family_alias="EB",
                role_class="E",
                endpoint_pattern=("VA", "EB", "VC"),
                slot_rules=(
                    {"attachment_index": 1, "slot_type": "XB", "endpoint_side": "V"},
                ),
            ),
            "edge:EC": EdgeRoleRecord(
                role_id="edge:EC",
                family_alias="EC",
                role_class="E",
                endpoint_pattern=("VA", "EC", "VD"),
                slot_rules=(
                    {"attachment_index": 2, "slot_type": "XC", "endpoint_side": "V"},
                ),
            ),
        },
    )

    optimizer = opt.NetOptimizer(semantic_snapshot=semantic_snapshot)
    rigid_initialization = optimizer.compile_local_rigid_initialization("V0")
    refinement = optimizer.compile_local_constrained_refinement(
        "V0",
        rigid_initialization=rigid_initialization,
        objective_weights={"anchor_mismatch": 1.0, "angle_alignment": 0.75},
    )

    assert isinstance(refinement, NodeLocalConstrainedRefinement)
    assert refinement.correspondence.edge_to_slot_index == {"V0|V1": 0, "V0|V2": 1, "V0|V3": 2}
    assert refinement.correspondence.edge_to_slot_index == rigid_initialization.correspondence.edge_to_slot_index
    assert refinement.objective_value < refinement.initial_objective_value
    assert refinement.metadata["direction_pair_count"] == 3
    assert refinement.metadata["search_strategy"] == "deterministic coordinate descent around passive SVD pose"
    assert "angle alignment penalty" in refinement.metadata["objective_terms"][1]
    assert not np.allclose(
        np.asarray(refinement.rotation_matrix),
        np.asarray(rigid_initialization.rotation_matrix),
    )


def test_compile_local_constrained_refinement_tracks_null_edge_alignment_terms_explicitly():
    rotation_expected = np.array(
        [
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    translation_expected = np.array([5.0, -2.0, 0.5])
    source_anchor_real = (2.0, 0.0, 0.0)
    source_anchor_real_2 = (0.0, 0.0, 2.0)
    source_direction_null = (0.0, 1.0, 0.0)
    target_anchor_real = tuple(
        np.dot(np.asarray(source_anchor_real), rotation_expected) + translation_expected
    )
    target_anchor_real_2 = tuple(
        np.dot(np.asarray(source_anchor_real_2), rotation_expected) + translation_expected
    )
    target_direction_null = tuple(
        np.dot(np.asarray(source_direction_null), rotation_expected)
    )

    semantic_snapshot = OptimizationSemanticSnapshot(
        family_name="ROLE-AWARE",
        graph_phase="sG",
        graph_node_records={
            "V0": GraphNodeSemanticRecord(
                node_id="V0",
                role_id="node:VA",
                role_class="V",
                slot_rules=(
                    {
                        "attachment_index": 0,
                        "slot_type": "XA",
                        "anchor_vector": source_anchor_real,
                        "chemistry_direction": source_anchor_real,
                    },
                    {
                        "attachment_index": 1,
                        "slot_type": "XB",
                        "anchor_vector": source_direction_null,
                        "chemistry_direction": source_direction_null,
                    },
                    {
                        "attachment_index": 2,
                        "slot_type": "XC",
                        "anchor_vector": source_anchor_real_2,
                        "chemistry_direction": source_anchor_real_2,
                    },
                ),
                incident_edge_ids=("V0|V1", "V0|V2", "V0|V3"),
                incident_edge_role_ids=("edge:EA", "edge:EB", "edge:EC"),
                incident_edge_constraints=(
                    {"edge_id": "V0|V1", "edge_role_id": "edge:EA", "slot_index": 0},
                    {"edge_id": "V0|V2", "edge_role_id": "edge:EB", "slot_index": 1},
                    {"edge_id": "V0|V3", "edge_role_id": "edge:EC", "slot_index": 2},
                ),
            ),
        },
        graph_edge_records={
            "V0|V1": GraphEdgeSemanticRecord(
                edge_id="V0|V1",
                graph_edge=("V0", "V1"),
                edge_role_id="edge:EA",
                path_type="V-E-V",
                endpoint_node_ids=("V0", "V1"),
                endpoint_role_ids=("node:VA", "node:VB"),
                endpoint_pattern=("VA", "EA", "VB"),
                slot_index={"V0": 0, "V1": 0},
                slot_rules=(
                    {"attachment_index": 0, "slot_type": "XA", "endpoint_side": "V"},
                ),
                metadata={"target_point_by_node": {"V0": target_anchor_real}},
            ),
            "V0|V2": GraphEdgeSemanticRecord(
                edge_id="V0|V2",
                graph_edge=("V0", "V2"),
                edge_role_id="edge:EB",
                path_type="V-E-V",
                endpoint_node_ids=("V0", "V2"),
                endpoint_role_ids=("node:VA", "node:VC"),
                endpoint_pattern=("VA", "EB", "VC"),
                slot_index={"V0": 1, "V2": 0},
                slot_rules=(
                    {"attachment_index": 1, "slot_type": "XB", "endpoint_side": "V"},
                ),
                resolve_mode="alignment_only",
                is_null_edge=True,
                null_payload_model="duplicated_zero_length_anchors",
                metadata={"target_vector_by_node": {"V0": target_direction_null}},
            ),
            "V0|V3": GraphEdgeSemanticRecord(
                edge_id="V0|V3",
                graph_edge=("V0", "V3"),
                edge_role_id="edge:EC",
                path_type="V-E-V",
                endpoint_node_ids=("V0", "V3"),
                endpoint_role_ids=("node:VA", "node:VD"),
                endpoint_pattern=("VA", "EC", "VD"),
                slot_index={"V0": 2, "V3": 0},
                slot_rules=(
                    {"attachment_index": 2, "slot_type": "XC", "endpoint_side": "V"},
                ),
                metadata={"target_point_by_node": {"V0": target_anchor_real_2}},
            ),
        },
        node_role_records={
            "node:VA": NodeRoleRecord(
                role_id="node:VA",
                family_alias="VA",
                role_class="V",
                slot_rules=(
                    {
                        "attachment_index": 0,
                        "slot_type": "XA",
                        "anchor_vector": source_anchor_real,
                        "chemistry_direction": source_anchor_real,
                    },
                    {
                        "attachment_index": 1,
                        "slot_type": "XB",
                        "anchor_vector": source_direction_null,
                        "chemistry_direction": source_direction_null,
                    },
                    {
                        "attachment_index": 2,
                        "slot_type": "XC",
                        "anchor_vector": source_anchor_real_2,
                        "chemistry_direction": source_anchor_real_2,
                    },
                ),
            ),
        },
        edge_role_records={
            "edge:EA": EdgeRoleRecord(
                role_id="edge:EA",
                family_alias="EA",
                role_class="E",
                endpoint_pattern=("VA", "EA", "VB"),
                slot_rules=(
                    {"attachment_index": 0, "slot_type": "XA", "endpoint_side": "V"},
                ),
            ),
            "edge:EB": EdgeRoleRecord(
                role_id="edge:EB",
                family_alias="EB",
                role_class="E",
                endpoint_pattern=("VA", "EB", "VC"),
                slot_rules=(
                    {"attachment_index": 1, "slot_type": "XB", "endpoint_side": "V"},
                ),
                resolve_mode="alignment_only",
            ),
            "edge:EC": EdgeRoleRecord(
                role_id="edge:EC",
                family_alias="EC",
                role_class="E",
                endpoint_pattern=("VA", "EC", "VD"),
                slot_rules=(
                    {"attachment_index": 2, "slot_type": "XC", "endpoint_side": "V"},
                ),
            ),
        },
        null_edge_policy_records={
            "edge:EB": NullEdgePolicyRecord(
                edge_role_id="edge:EB",
                edge_kind="null",
                is_null_edge=True,
                null_payload_model="duplicated_zero_length_anchors",
            ),
        },
    )

    refinement = compile_local_constrained_refinement(semantic_snapshot, "V0")

    assert refinement.metadata["null_edge_direction_pair_count"] == 1
    assert refinement.metadata["weights"]["null_edge_alignment"] == pytest.approx(0.5)
    assert refinement.objective_breakdown["null_edge_alignment_penalty"] == pytest.approx(0.0)
    assert "null-edge alignment penalty" in refinement.metadata["objective_terms"][2]
