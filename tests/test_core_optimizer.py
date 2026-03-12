import numpy as np
import networkx as nx

from mofbuilder.core import optimizer as opt


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
