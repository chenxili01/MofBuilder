from pathlib import Path

import networkx as nx
import numpy as np
import pytest

from mofbuilder.core.node import FrameNode


TESTDATA = Path(__file__).resolve().parent / "testdata"


class _FakeMolecule:

    def __init__(self, matrix):
        self._matrix = np.array(matrix, dtype=int)

    def get_connectivity_matrix(self):
        return self._matrix

    @classmethod
    def read_xyz_string(cls, _xyz):
        # Indexes match rows in node_data used by test.
        # 0: Zr1, 1: O1, 2: H1, 3: X1
        return cls(
            [
                [0, 1, 0, 1],
                [1, 0, 1, 0],
                [0, 1, 0, 0],
                [1, 0, 0, 0],
            ])


@pytest.mark.core
def test_node_fetch_template_supported_and_unsupported():
    node = FrameNode(filepath=str(TESTDATA / "testnode.pdb"))
    node.dummy_node = True
    tpl = node._fetch_template("Zr")
    assert tpl.shape == (8, 3)

    with pytest.raises(AssertionError):
        node._fetch_template("Xx")


@pytest.mark.core
def test_nodepdb2g_skips_metal_to_x_and_h_edges(monkeypatch):
    node = FrameNode(filepath=str(TESTDATA / "testnode.pdb"))
    node.node_metal_type = "Zr"
    node.node_data = np.array(
        [
            ["Zr1", "Zr", 1, "MOL", 1, 0.0, 0.0, 0.0, 1.0, 0.0, "Zr"],
            ["O1", "O", 2, "MOL", 1, 1.0, 0.0, 0.0, 1.0, 0.0, "O"],
            ["H1", "H", 3, "MOL", 1, 2.0, 0.0, 0.0, 1.0, 0.0, "H"],
            ["X1", "C", 4, "MOL", 1, 0.0, 1.0, 0.0, 1.0, 0.0, "X"],
        ],
        dtype=object,
    )
    node.node_xyz_string = "4\n\nZr 0 0 0\nO 1 0 0\nH 2 0 0\nX 0 1 0\n"

    import mofbuilder.core.node as node_mod

    monkeypatch.setattr(node_mod, "Molecule", _FakeMolecule)
    node._nodepdb2G()

    assert node.nodeG.has_edge("O1", "H1")
    assert not node.nodeG.has_edge("Zr1", "X1")
    assert not node.nodeG.has_edge("Zr1", "H1")


@pytest.mark.core
def test_add_dummy_atoms_no_dummy_removes_metal_edges_and_connects_hydrogen():
    node = FrameNode(filepath=str(TESTDATA / "testnode.pdb"))
    node.node_metal_type = "Zr"
    node.dummy_node = False

    g = nx.Graph()
    g.add_node("Zr1", ccoords=np.array([0.0, 0.0, 0.0]), type="Zr")
    g.add_node("O1", ccoords=np.array([1.0, 0.0, 0.0]), type="O")
    g.add_node("H1", ccoords=np.array([1.2, 0.0, 0.0]), type="H")
    g.add_edge("Zr1", "O1")
    node.nodeG = g

    node._add_dummy_atoms_nodepdb()

    assert node.sG is not None
    assert not node.sG.has_edge("Zr1", "O1")
    assert node.sG.has_edge("H1", "O1")


@pytest.mark.core
def test_node_create_runs_on_test_node_file():
    node = FrameNode(filepath=str(TESTDATA / "testnode.pdb"))
    node.node_metal_type = "Zr"
    node.dummy_node = False
    node.save_files = False
    node.create()

    assert node.node_data is not None
    assert node.node_X_data is not None
    assert node.dummy_node_split_dict is not None
