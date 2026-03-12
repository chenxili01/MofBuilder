import networkx as nx
import numpy as np

from mofbuilder.core.supercell import EdgeGraphBuilder, SupercellBuilder, remove_node_by_index


def _make_vnode_f_points(xcoord, x_index_offset=0):
    rows = []
    if x_index_offset:
        rows.append(["C0", "C", xcoord - 0.05, 0.0, 0.0])
    rows.extend([
        [f"X{x_index_offset + 1}", "X", xcoord, 0.0, 0.0],
        [f"O{x_index_offset + 1}", "O", xcoord, 0.02, 0.0],
        [f"O{x_index_offset + 2}", "O", xcoord, -0.02, 0.0],
    ])
    return np.array(rows, dtype=object)


def _make_edge_f_points(*xcoords):
    return np.array([[f"X{i + 1}", "X", xcoord, 0.0, 0.0]
                     for i, xcoord in enumerate(xcoords)],
                    dtype=object)


def _build_edgegraph(superg):
    builder = EdgeGraphBuilder()
    builder.superG = superg
    builder.linker_connectivity = 2
    builder.sc_unit_cell = np.eye(3)
    builder.supercell = [1, 1, 1]
    builder.node_connectivity = 1
    builder.custom_fbox = None
    builder.build_edgeG_from_superG()
    return builder


def test_is_ditopic_linker_flag():
    builder = SupercellBuilder()
    builder.linker_connectivity = 2
    assert builder._is_ditopic_linker() is True
    builder.linker_connectivity = 4
    assert builder._is_ditopic_linker() is False


def test_remove_node_by_index_removes_target_nodes_and_edges():
    g = nx.Graph()
    g.add_node("V_0", index=1)
    g.add_node("EDGE_0", index=-2)
    g.add_node("V_1", index=3)

    out = remove_node_by_index(g, remove_node_list=[1], remove_edge_list=[2])

    assert "V_0" not in out.nodes
    assert "EDGE_0" not in out.nodes
    assert "V_1" in out.nodes


def test_edgegraph_update_matched_nodes_xind_filters_invalid_entries():
    b = EdgeGraphBuilder()
    matched = [("V_0", 1, "EDGE_0"), ("V_1", 2, "EDGE_1"), ("V_2", 3, "EDGE_2")]

    out = b._update_matched_nodes_xind(["V_0"], ["EDGE_2"], matched)

    assert out == [("V_1", 2, "EDGE_1")]


def test_single_role_supercell_and_edgegraph_keep_role_metadata():
    sg = nx.Graph()
    sg.add_node("V0_[0. 0. 0.]",
                f_points=_make_vnode_f_points(0.25),
                fcoords=np.array([0.25, 0.0, 0.0]),
                type="V",
                note="V",
                node_role_id="node:default")
    sg.add_node("V1_[0. 0. 0.]",
                f_points=_make_vnode_f_points(0.75),
                fcoords=np.array([0.75, 0.0, 0.0]),
                type="V",
                note="V",
                node_role_id="node:default")
    sg.add_edge("V0_[0. 0. 0.]",
                "V1_[0. 0. 0.]",
                f_points=_make_edge_f_points(0.26, 0.74),
                fcoords=np.array([[0.25, 0.0, 0.0], [0.75, 0.0, 0.0]]),
                type="E",
                edge_role_id="edge:default")

    supercell_builder = SupercellBuilder()
    supercell_builder.sG = sg
    supercell_builder.linker_connectivity = 2
    supercell_builder.supercell = [1, 1, 1]
    supercell_builder.cell_info = [1.0, 1.0, 1.0, 90.0, 90.0, 90.0]

    superg = supercell_builder.build_supercellGraph()

    assert "V0_[0. 0. 0.]" in superg.nodes
    assert "V1_[0. 0. 0.]" in superg.nodes
    assert superg.has_edge("V0_[0. 0. 0.]", "V1_[0. 0. 0.]")
    assert {data["node_role_id"] for _, data in superg.nodes(data=True)} == {
        "node:default"
    }
    assert {data["edge_role_id"] for _, _, data in superg.edges(data=True)} == {
        "edge:default"
    }

    direct_superg = nx.Graph()
    direct_superg.add_node("V0_[0. 0. 0.]",
                           f_points=_make_vnode_f_points(0.25),
                           fcoords=np.array([0.25, 0.0, 0.0]),
                           type="V",
                           note="V",
                           node_role_id="node:default")
    direct_superg.add_node("V1_[0. 0. 0.]",
                           f_points=_make_vnode_f_points(0.75),
                           fcoords=np.array([0.75, 0.0, 0.0]),
                           type="V",
                           note="V",
                           node_role_id="node:default")
    direct_superg.add_edge("V0_[0. 0. 0.]",
                           "V1_[0. 0. 0.]",
                           f_points=_make_edge_f_points(0.26, 0.74),
                           fcoords=np.array([[0.25, 0.0, 0.0],
                                             [0.75, 0.0, 0.0]]),
                           type="E",
                           edge_role_id="edge:default")

    edgegraph_builder = _build_edgegraph(direct_superg)
    edge_nodes = [n for n in edgegraph_builder.eG.nodes if n.startswith("EDGE_")]

    assert edgegraph_builder.eG.number_of_nodes() == 3
    assert edgegraph_builder.eG.number_of_edges() == 3
    assert len(edge_nodes) == 1
    assert edgegraph_builder.eG.nodes[edge_nodes[0]]["edge_role_id"] == "edge:default"
    assert edgegraph_builder.xoo_dict == {0: [1, 2]}
    assert [tuple(entry) for entry in edgegraph_builder.matched_vnode_xind] == [
        ("V0_[0. 0. 0.]", 0, edge_nodes[0]),
        ("V1_[0. 0. 0.]", 0, edge_nodes[0]),
    ]
    assert edgegraph_builder.cleaved_eG.nodes[edge_nodes[0]]["edge_role_id"] == "edge:default"


def test_edgegraph_preserves_roles_through_cleave_with_role_specific_xoo_layouts():
    superg = nx.Graph()
    superg.add_node("VA_[0. 0. 0.]",
                    f_points=_make_vnode_f_points(0.20, x_index_offset=1),
                    fcoords=np.array([0.20, 0.0, 0.0]),
                    type="V",
                    note="V",
                    node_role_id="node:alpha")
    superg.add_node("VB_[0. 0. 0.]",
                    f_points=_make_vnode_f_points(0.80, x_index_offset=0),
                    fcoords=np.array([0.80, 0.0, 0.0]),
                    type="V",
                    note="V",
                    node_role_id="node:beta")
    superg.add_edge("VA_[0. 0. 0.]",
                    "VB_[0. 0. 0.]",
                    f_points=_make_edge_f_points(0.21, 0.79),
                    fcoords=np.array([[0.20, 0.0, 0.0], [0.80, 0.0, 0.0]]),
                    type="E",
                    edge_role_id="edge:bridge")

    edgegraph_builder = _build_edgegraph(superg)
    edge_nodes = [n for n in edgegraph_builder.eG.nodes if n.startswith("EDGE_")]

    assert len(edge_nodes) == 1
    edge_name = edge_nodes[0]
    assert edgegraph_builder.eG.nodes["VA_[0. 0. 0.]"]["node_role_id"] == "node:alpha"
    assert edgegraph_builder.eG.nodes["VB_[0. 0. 0.]"]["node_role_id"] == "node:beta"
    assert edgegraph_builder.eG.nodes[edge_name]["edge_role_id"] == "edge:bridge"
    assert edgegraph_builder.cleaved_eG.nodes["VA_[0. 0. 0.]"]["node_role_id"] == "node:alpha"
    assert edgegraph_builder.cleaved_eG.nodes["VB_[0. 0. 0.]"]["node_role_id"] == "node:beta"
    assert edgegraph_builder.cleaved_eG.nodes[edge_name]["edge_role_id"] == "edge:bridge"
    assert edgegraph_builder.eG.nodes[edge_name]["xoo_f_points"].shape == (6, 5)
    assert set(tuple(entry) for entry in edgegraph_builder.matched_vnode_xind) == {
        ("VA_[0. 0. 0.]", 1, edge_name),
        ("VB_[0. 0. 0.]", 0, edge_name),
    }
