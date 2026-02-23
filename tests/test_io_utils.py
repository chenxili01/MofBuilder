from pathlib import Path

import numpy as np
import pytest

from mofbuilder.utils.environment import get_data_path
from mofbuilder.utils.fetch import fetch_pdbfile
from mofbuilder.utils.geometry import (
    Carte_points_generator,
    cartesian_to_fractional,
    find_edge_pairings,
    find_optimal_pairings,
    locate_min_idx,
    reorthogonalize_matrix,
    unit_cell_to_cartesian_matrix,
    fractional_to_cartesian,
)
from mofbuilder.utils.periodic_table import (
    element_info,
    get_atomic_mass,
    get_atomic_number,
    get_atomic_radius,
    get_element_symbol,
    is_metal,
    is_metalloid,
    is_nonmetal,
)


class _DummyOstream:

    def __init__(self):
        self.messages = []

    def print_info(self, msg):
        self.messages.append(str(msg))


def test_environment_data_path_points_to_database():
    path = get_data_path()
    assert isinstance(path, Path)
    assert path.name == "database"


def test_fetch_pdbfile_filters_by_keywords():
    nodes_dir = Path(__file__).resolve().parent / "database" / "nodes_database"
    ostream = _DummyOstream()
    matches = fetch_pdbfile(str(nodes_dir), ["12c", "Zr"], ["copy"], ostream)
    assert any(name.endswith(".pdb") for name in matches)


def test_fetch_pdbfile_raises_when_no_match():
    nodes_dir = Path(__file__).resolve().parent / "database" / "nodes_database"
    with pytest.raises(ValueError):
        fetch_pdbfile(str(nodes_dir), ["NOT_A_REAL_FILE"], [], _DummyOstream())


def test_geometry_fractional_cartesian_roundtrip():
    t = unit_cell_to_cartesian_matrix(10.0, 11.0, 12.0, 90.0, 90.0, 120.0)
    inv_t = np.linalg.inv(t)
    frac = np.array([[0.1, 0.2, 0.3], [0.5, 0.5, 0.5]])
    cart = fractional_to_cartesian(frac, t)
    frac_back = cartesian_to_fractional(cart, inv_t)
    np.testing.assert_allclose(frac_back, frac, atol=1e-10)


def test_geometry_helpers():
    assert locate_min_idx(np.array([[3.0, 2.0], [1.0, 4.0]])) == (1, 0)
    r = reorthogonalize_matrix(np.eye(3))
    np.testing.assert_allclose(r, np.eye(3), atol=1e-12)
    pts = Carte_points_generator((1, 1, 1))
    assert pts.shape == (8, 3)


def test_geometry_pairings():
    node_i = np.array([[0, 0.0, 0.0, 0.0], [1, 1.0, 0.0, 0.0]])
    node_j = np.array([[10, 0.05, 0.0, 0.0], [11, 1.1, 0.0, 0.0]])
    pair = find_optimal_pairings(node_i, node_j)
    assert pair == [0, 0]

    edge_pairings = find_edge_pairings(
        sorted_nodes=[0, 1],
        sorted_edges=[(0, 1)],
        atom_positions={0: node_i, 1: node_j},
    )
    assert edge_pairings[(0, 1)] == [0, 0]


def test_periodic_table_lookups():
    assert get_atomic_mass("C") == pytest.approx(12.011)
    assert get_atomic_radius("O") > 0.0
    assert get_atomic_number("Zn") == 30
    assert get_element_symbol(8) == "O"

    info = element_info("Si")
    assert info["symbol"] == "Si"
    assert info["atomic_number"] == 14

    assert is_metal("Fe")
    assert is_nonmetal("C")
    assert is_metalloid("Si")


def test_periodic_table_unknown_element_raises():
    with pytest.raises(KeyError):
        get_atomic_mass("Xx")
    with pytest.raises(KeyError):
        get_atomic_radius("Xx")
