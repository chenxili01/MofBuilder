from pathlib import Path

import numpy as np
import pytest

from mofbuilder.core.net import (
    FrameNet,
    arr_dimension,
    is_list_A_in_B,
    lname,
    pname,
)


TESTDATA = Path(__file__).resolve().parent / "testdata"


def _canonical_validation_role_metadata():
    return {
        "schema_name": "mof_reticular_role_metadata",
        "schema_version": 1,
        "family_name": "TEST-MULTI",
        "roles": {
            "VA": {"role_class": "V", "canonical_role_id": "node:VA"},
            "CA": {"role_class": "C", "canonical_role_id": "node:CA"},
            "EA": {"role_class": "E", "canonical_role_id": "edge:EA"},
            "EB": {"role_class": "E", "canonical_role_id": "edge:EB"},
        },
        "connectivity_rules": {
            "VA": {"incident_edge_aliases": ["EA", "EA", "EB", "EB"]},
            "CA": {"incident_edge_aliases": ["EA", "EA", "EA", "EA"]},
        },
        "path_rules": [
            {"edge_alias": "EA", "endpoint_pattern": ["VA", "EA", "CA"]},
            {"edge_alias": "EB", "endpoint_pattern": ["VA", "EB", "VA"]},
        ],
        "edge_kind_rules": {
            "EA": {"edge_kind": "real"},
            "EB": {
                "edge_kind": "null",
                "null_payload_model": "duplicated_zero_length_anchors",
            },
        },
    }


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
    assert all("slot_index" in data for _, _, data in net.G.edges(data=True))
    assert all(isinstance(data["slot_index"], dict) for _, _, data in net.G.edges(data=True))
    assert not any(
        "cyclic_edge_order" in data for _, data in net.G.nodes(data=True)
    )


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
    assert all("slot_index" in data for _, _, data in net.G.edges(data=True))
    assert all(
        set(data["slot_index"]) == set(edge)
        for edge, data in net.G.edges.items()
    )


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
    assert all("slot_index" in data for _, _, data in first.G.edges(data=True))


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
    cv_nodes = [
        node_name for node_name, data in net.G.nodes(data=True) if data["node_role_id"] == "node:CA"
    ]
    assert len(cv_nodes) == 1
    cv_node = cv_nodes[0]
    cyclic_edge_order = net.G.nodes[cv_node]["cyclic_edge_order"]
    assert len(cyclic_edge_order) == net.G.degree(cv_node) == 4
    assert len(set(cyclic_edge_order)) == 4
    for order_index, edge in enumerate(cyclic_edge_order):
        assert cv_node in edge
        assert net.G.edges[edge]["cyclic_edge_order"][cv_node] == order_index
        assert net.G.edges[edge]["slot_index"][cv_node] in range(net.G.degree(cv_node))


@pytest.mark.core
def test_validate_roles_accepts_role_aware_graph_with_metadata(tmp_path):
    cif_path = write_raw_test_cif(
        tmp_path,
        """data_role_specific_template
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
EA      EA2   0.1250   0.0000   0.0000
EA      EA3   0.0000  -0.1250   0.0000
EA      EA4   0.0000   0.1250   0.0000
loop_
""",
    )
    net = FrameNet()
    net.create_net(cif_file=str(cif_path))

    result = net.validate_roles(role_metadata=_canonical_validation_role_metadata())

    assert result.ok is True
    assert result.errors == []


@pytest.mark.core
def test_validate_roles_reports_descriptive_errors_for_missing_slot_metadata():
    net = FrameNet()
    net.G.add_node("V0", note="V", node_role_id="node:VA")
    net.G.add_node("C0", note="CV", node_role_id="node:CA")
    net.G.add_edge("V0", "C0", edge_role_id="edge:EA")

    result = net.validate_roles(role_metadata=_canonical_validation_role_metadata())

    assert result.ok is False
    assert [error["code"] for error in result.errors] == [
        "missing_slot_metadata",
        "connectivity_mismatch",
        "connectivity_mismatch",
        "missing_cyclic_order",
    ]
    assert "slot_index metadata" in result.errors[0]["message"]
    assert "FrameNet.create_net()" in result.errors[0]["hint"]


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
