import networkx as nx
import numpy as np
import pytest

from mofbuilder.core.linker import FrameLinker


class _ToyMolecule:

    def __init__(self):
        self._labels = ["C", "O", "O", "C", "O", "O", "H", "H"]
        self._coords = np.array(
            [
                [0.0, 0.0, 0.0],    # 0
                [1.2, 0.0, 0.0],    # 1
                [-1.2, 0.0, 0.0],   # 2
                [3.0, 0.0, 0.0],    # 3
                [4.2, 0.0, 0.0],    # 4
                [1.8, 0.0, 0.0],    # 5
                [5.4, 0.0, 0.0],    # 6
                [-2.4, 0.0, 0.0],   # 7
            ])
        self._matrix = np.array(
            [
                [0, 1, 1, 1, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 0, 1],
                [1, 0, 0, 0, 1, 1, 0, 0],
                [0, 0, 0, 1, 0, 0, 1, 0],
                [0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0],
            ],
            dtype=int,
        )

    def get_connectivity_matrix(self):
        return self._matrix

    def get_coordinates_in_angstrom(self):
        return self._coords.copy()

    def get_labels(self):
        return list(self._labels)

    def get_distance_matrix_in_angstrom(self):
        d = self._coords[:, None, :] - self._coords[None, :, :]
        return np.linalg.norm(d, axis=2)

    def center_of_mass_in_bohr(self):
        return np.mean(self._coords, axis=0) / 0.529177


@pytest.mark.core
def test_linker_boundary_helpers():
    g = nx.Graph()
    g.add_node(0, label="C")
    g.add_node(1, label="O")
    g.add_node(2, label="O")
    g.add_node(3, label="H")
    g.add_node(10, label="C")
    g.add_edge(0, 1)
    g.add_edge(0, 2)
    g.add_edge(2, 3)
    g.add_edge(0, 10)

    boundary = FrameLinker._find_boundary_atom(g, boundary_labels=["O"])
    assert (1, 0) in boundary

    center, dist = FrameLinker.find_closest_center_node(g, [10], 0)
    assert center == 10
    assert dist == 1


@pytest.mark.core
def test_linker_create_lg_excludes_metal_edges():
    linker = FrameLinker()
    toy = _ToyMolecule()
    linker._create_lG(toy)

    assert linker.lG.number_of_nodes() == len(toy.get_labels())
    assert linker.metals == []
    assert linker.lG.number_of_edges() > 0


@pytest.mark.core
def test_linker_process_molecule_and_create_for_ditopic():
    linker = FrameLinker()
    linker.linker_connectivity = 2
    linker.process_linker_molecule(_ToyMolecule(), linker_connectivity=2)

    assert linker.lines is not None
    assert len(linker.lines) > 0
    assert linker.fake_edge in (True, False)


@pytest.mark.core
def test_linker_create_public_api_with_injected_molecule():
    linker = FrameLinker()
    linker.linker_connectivity = 2
    linker.create(molecule=_ToyMolecule())

    assert linker.linker_center_data is not None
    assert linker.linker_center_X_data is not None
    assert linker.linker_center_data.shape[1] == 11



