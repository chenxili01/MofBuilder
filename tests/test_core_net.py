from pathlib import Path

import numpy as np
import pytest

from mofbuilder.core.net import FrameNet, arr_dimension, is_list_A_in_B, lname, pname


TESTDATA = Path(__file__).resolve().parent / "testdata"


@pytest.mark.core
def test_create_net_from_test_cif():
    net = FrameNet()
    net.create_net(cif_file=str(TESTDATA / "test.cif"))

    assert net.G.number_of_nodes() > 0
    assert net.G.number_of_edges() > 0
    assert isinstance(net.max_degree, int)
    assert len(net.cell_info) == 6
    assert net.unit_cell.shape == (3, 3)
    assert net.unit_cell_inv.shape == (3, 3)
    assert len(net.sorted_nodes) == net.G.number_of_nodes()
    assert len(net.sorted_edges) == net.G.number_of_edges()
    assert net.linker_connectivity == 2


@pytest.mark.core
def test_unit_cell_roundtrip_and_point_checks():
    net = FrameNet()
    unit = net._extract_unit_cell([10.0, 10.0, 10.0, 90.0, 90.0, 90.0])
    frac = np.array([[0.25, 0.5, 0.75]])
    cart = net._f2c_coords(frac, unit)
    back = net._c2f_coords(cart, unit)
    np.testing.assert_allclose(back, frac, atol=1e-10)

    assert net._check_inside_unit_cell(np.array([0.1, 0.2, 0.3]))
    assert not net._check_inside_unit_cell(np.array([1.0, 0.2, 0.3]))
    assert net._check_moded_fcoords(np.array([0.2, 0.3, 0.4]))
    assert not net._check_moded_fcoords(np.array([1.2, 0.3, 0.4]))


@pytest.mark.core
def test_unique_edge_centers_and_name_helpers():
    net = FrameNet()
    all_e = [
        np.array([0.1, 0.2, 0.3]),
        np.array([1.1, 0.2, 0.3]),  # periodic image of first
        np.array([0.5, 0.5, 0.5]),
    ]
    unique = net._find_unitcell_e(all_e)
    assert len(unique) == 2

    assert pname("V12_[0. 0. 0.]") == "V12"
    np.testing.assert_allclose(lname("V12_[1.0 0.0 -1.0]"), [1.0, 0.0, -1.0])
    assert arr_dimension(np.array([1, 2])) == 1
    assert arr_dimension(np.array([[1, 2]])) == 2
    assert is_list_A_in_B([np.array([1.0])], [np.array([1.0])])
