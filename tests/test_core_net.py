from pathlib import Path

import numpy as np
import pytest

from mofbuilder.core.net import FrameNet, arr_dimension, is_list_A_in_B, lname, pname


TESTDATA = Path(__file__).resolve().parent / "testdata"


def write_test_cif(tmp_path, atom_lines, v_con=1, ec_con=None):
    header = "data_testnet  hall_number: 1, V_con: {}".format(v_con)
    if ec_con is not None:
        header += f", EC_con: {ec_con}"
    cif_lines = [
        header + "\n",
        "_symmetry_space_group_name_H-M    'P1'\n",
        "_symmetry_Int_Tables_number       1\n",
        "loop_\n",
        "_symmetry_equiv_pos_as_xyz\n",
        "  x,y,z\n",
        "_cell_length_a                    10.0\n",
        "_cell_length_b                    10.0\n",
        "_cell_length_c                    10.0\n",
        "_cell_angle_alpha                 90.0\n",
        "_cell_angle_beta                  90.0\n",
        "_cell_angle_gamma                 90.0\n",
        "loop_\n",
        "_atom_site_label\n",
        "_atom_site_type_symbol\n",
        "_atom_site_fract_x\n",
        "_atom_site_fract_y\n",
        "_atom_site_fract_z\n",
    ]
    cif_lines.extend(f"{line}\n" for line in atom_lines)
    cif_lines.append("loop_\n")
    cif_path = tmp_path / "topology.cif"
    cif_path.write_text("".join(cif_lines), encoding="utf-8")
    return cif_path


def write_raw_test_cif(tmp_path, cif_text, filename="topology.cif"):
    cif_path = tmp_path / filename
    cif_path.write_text(cif_text, encoding="utf-8")
    return cif_path


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
    assert {data["node_role_id"] for _, data in net.G.nodes(data=True)} == {
        "node:default"
    }
    assert {data["edge_role_id"] for _, _, data in net.G.edges(data=True)} == {
        "edge:default"
    }


@pytest.mark.core
def test_create_net_preserves_single_role_scalar_outputs(tmp_path):
    cif_path = write_test_cif(
        tmp_path,
        [
            "V1    V   0.2500  0.0000  0.0000",
            "V2    V   0.7500  0.0000  0.0000",
            "E1    E   0.0000  0.0000  0.0000",
        ],
    )
    net = FrameNet()
    net.edge_length_range = [2.0, 3.0]
    net.create_net(cif_file=str(cif_path))

    assert net.linker_connectivity == 2
    assert net.max_degree == 1
    assert net.sorted_nodes == [
        "V40_[0. 0. 0.]",
        "V16_[0. 0. 0.]",
        "V16_[-1.  0.  0.]",
        "V40_[1. 0. 0.]",
    ]
    assert net.sorted_edges == [
        ("V40_[0. 0. 0.]", "V16_[-1.  0.  0.]"),
        ("V16_[0. 0. 0.]", "V40_[1. 0. 0.]"),
    ]
    assert {data["node_role_id"] for _, data in net.G.nodes(data=True)} == {
        "node:default"
    }
    assert {data["edge_role_id"] for _, _, data in net.G.edges(data=True)} == {
        "edge:default"
    }


@pytest.mark.core
def test_create_net_attaches_deterministic_role_annotations(tmp_path):
    cif_path = write_test_cif(
        tmp_path,
        [
            "VA1   V   0.2500  0.0000  0.0000",
            "VB1   V   0.7500  0.0000  0.0000",
            "VA2   V   0.2500  0.2500  0.0000",
            "VB2   V   0.7500  0.2500  0.0000",
            "EA1   E   0.0000  0.0000  0.0000",
            "EB1   E   0.0000  0.2500  0.0000",
        ],
    )
    first = FrameNet()
    first.edge_length_range = [2.0, 3.0]
    first.create_net(cif_file=str(cif_path))

    second = FrameNet()
    second.edge_length_range = [2.0, 3.0]
    second.create_net(cif_file=str(cif_path))

    first_node_roles = sorted(data["node_role_id"]
                              for _, data in first.G.nodes(data=True))
    first_edge_roles = sorted(data["edge_role_id"]
                              for _, _, data in first.G.edges(data=True))
    second_node_roles = sorted(data["node_role_id"]
                               for _, data in second.G.nodes(data=True))
    second_edge_roles = sorted(data["edge_role_id"]
                               for _, _, data in second.G.edges(data=True))

    assert first_node_roles == second_node_roles
    assert first_edge_roles == second_edge_roles
    assert set(first_node_roles) == {"node:VA", "node:VB"}
    assert set(first_edge_roles) == {"edge:EA", "edge:EB"}


@pytest.mark.core
def test_create_net_supports_role_specific_template_types_without_header_metadata(
    tmp_path,
):
    cif_path = write_raw_test_cif(
        tmp_path,
        """data_role_specific_template
_audit_creation_date              2026-03-13
_audit_creation_method            role-specific topology points
_symmetry_space_group_name_H-M    'P1'
_symmetry_Int_Tables_number       1
loop_
_symmetry_equiv_pos_as_xyz
  x,y,z
_cell_length_a                    10.0
_cell_length_b                    10.0
_cell_length_c                    10.0
_cell_angle_alpha                 90.0
_cell_angle_beta                  90.0
_cell_angle_gamma                 90.0
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
VA      VA1  -0.2500   0.0000   0.0000
VA      VA2   0.2500   0.0000   0.0000
VA      VA3   0.0000  -0.2500   0.0000
VA      VA4   0.0000   0.2500   0.0000
CA      CA1   0.0000   0.0000   0.0000
EA      EA1  -0.1250   0.0000   0.0000
EB      EB1   0.1250   0.0000   0.0000
EA      EA2   0.0000  -0.1250   0.0000
EB      EB2   0.0000   0.1250   0.0000
loop_
""",
    )
    net = FrameNet()
    net.create_net(cif_file=str(cif_path))

    assert net.G.number_of_nodes() > 0
    assert net.G.number_of_edges() > 0
    assert len(net.sorted_nodes) == net.G.number_of_nodes()
    assert len(net.sorted_edges) == net.G.number_of_edges()
    assert net.max_degree == 4
    assert net.linker_connectivity == 4
    assert {data["node_role_id"] for _, data in net.G.nodes(data=True)} == {
        "node:VA",
        "node:CA",
    }
    assert {data["edge_role_id"] for _, _, data in net.G.edges(data=True)} == {
        "edge:EA",
        "edge:EB",
    }


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
