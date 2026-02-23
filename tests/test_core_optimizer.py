import numpy as np
import networkx as nx

from mofbuilder.core import optimizer as opt


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
