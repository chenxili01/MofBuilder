import numpy as np
import networkx as nx

from mofbuilder.core.write import MofWriter


def test_remove_xoo_from_node_sets_noxoo_points():
    w = MofWriter()
    g = nx.Graph()
    node_points = np.array([
        ["C", "C", 0.0, 0.0, 0.0],
        ["X", "X", 0.1, 0.0, 0.0],
        ["O", "O", 0.2, 0.0, 0.0],
        ["O", "O", 0.3, 0.0, 0.0],
    ], dtype=object)
    g.add_node("V_0", f_points=node_points)
    g.add_node("EDGE_0", f_points=node_points)

    out = w._remove_xoo_from_node(g, {1: [2, 3]})

    assert out.nodes["V_0"]["noxoo_f_points"].shape[0] == 1
    assert "noxoo_f_points" not in out.nodes["EDGE_0"]


def test_rename_node_name_returns_ordered_data():
    w = MofWriter()

    # One node with 3 atoms: METAL(1), HO(2)
    node = np.array([
        ["Zr", "Zr", 1, "OLD", 1, 0.0, 0.0, 0.0, 0, 0.0, ""],
        ["O", "O", 2, "OLD", 1, 1.0, 0.0, 0.0, 0, 0.0, ""],
        ["H", "H", 3, "OLD", 1, 1.1, 0.0, 0.0, 0, 0.0, ""],
    ], dtype=object)

    renamed = w._rename_node_name(
        nodes_data=[node],
        dummy_atom_node_dict={
            "METAL_count": 1,
            "dummy_res_len": 1,
            "HHO_count": 0,
            "HO_count": 1,
            "O_count": 0,
        },
    )

    assert renamed.shape[1] == 11
    assert renamed[0, 3].startswith("METAL_")
    assert renamed[1, 3].startswith("HO_")
